# Speedups: batched prefill and decode, CUDA graphs, incremental hashing, BF16 and INT8 lanes

The roofline analysis in [ANALYSIS-OVERHEAD.md](ANALYSIS-OVERHEAD.md) proposed five speedups. All five are implemented as an opt-in executor, `src/qwen_fast.{h,cu}`, plus an incremental state digest in the driver. The lean `qwen_push`/`qwen_advance` path is unchanged and remains the default. The new executor reads the lean model's device weights. Each speedup below was measured against that lean path and checked for exactness.

Measured in mrun job `job-1d03b313def2` on Beast (RTX 4080, driver 580.159.03, nvcc 13.0) on 2026-11-13. Source: `claude/speedups` at `f624567`.

- Workload: the overhead job's 16 prompts (17–27 tokens) × 64 greedy tokens. Hashing arms used 8 prompts.
- Every step is bracketed by a device or stream sync. B=1 arm order rotates per prompt.
- No timed region overlapped a foreign GPU process. An earlier run, `job-24ee1aae8339`, was contaminated in its TF32, B128 and some hash arms, so this run replaced it. It gave identical gate results.

Labels: **M** = measured; **D** = derived from measured times. tok/s, GB/s and FLOP/s are D, using bytes or FLOPs per step divided by the measured median. The 48.7 TFLOPS FP32 peak is the vendor figure, asserted and not measured. 669.9 GB/s is the copy bandwidth measured in the overhead job.

## Results

### 1. Batched prefill (all prompt tokens in one pass per layer)

| Arm | Prefill ms per prompt, median (p10–p90) | vs lean |
|---|---|---|
| Lean C0: sequential `qwen_push` | 210.5 (178.9–263.1) M | — |
| Fast engine, one-row passes (F1s) | 174.4 (148.4–217.8) M | 1.21× D |
| **Batched prefill** (F1) | **11.56 (11.47–11.65)** M | **18.2×** D |
| Batched + TF32 | 12.08 M | 17.4× D |
| HF Transformers fp32 (overhead job) | 14.4 M | — |

Predicted ~20×; measured 18.2×. Batched prefill is now 1.25× faster than HF prefill (D).

### 2. Multi-sequence batched decode (FP32, cuBLAS pedantic sgemm; B=1 uses sgemv)

| B | ms/step median (p10–p90), non-graph | graph | Aggregate tok/s (graph) | Effective GB/s, % of 669.9 (graph) | Achieved TFLOP/s, % of 48.7 vendor (graph) |
|---|---|---|---|---|---|
| lean C0 | 10.544 (10.532–10.556) M | — | 94.8 D | 586, 87.4% D | 0.29, 0.6% D |
| 1 | 10.177 (10.128–10.232) M | 9.973 (9.920–10.028) M | 100.3 D | 619, 92.4% D | 0.31 D |
| 8 | 11.543 M | 11.263 M | 710 D | 548, 81.8% D | 2.19, 4.5% D |
| 16 | 11.685 M | 11.399 M | 1,404 D | 542, 80.9% D | 4.33, 8.9% D |
| 32 | 12.064 M | 11.833 M | 2,704 D | 522, 77.9% D | 8.35, 17.1% D |
| 64 | 13.359 (13.196–13.540) M | 13.036 (12.880–13.209) M | 4,909 D | 474, 70.7% D | 15.2, 31.1% D |
| 128 | 19.465 M | 19.266 M | 6,644 D | 321, 47.8% D | 20.5, 42.1% D |

- Aggregate throughput rises 51.8× from C0 to B=64 with graphs, and 70× at B=128 (D).
- The step time is not flat up to the predicted ridge near B≈145. It rises 1.31× from B=1 to B=64 and 1.91× to B=128 (M). At B=64 the pedantic FP32 sgemm reaches only 31% of the vendor peak (D). The practical ridge for this cuBLAS path is therefore around B=32–64, not 145.
- KV capacity is 57,344 B × B × T. The run held 128 sequences × 99 positions, which is 727 MB of K/V. The fast engine's total footprint was 851 MB, and the process peak was 7,173 MB (M, `gpu_peak_bytes`).

TF32 tensor-op sgemm (opt-in, separate flag):

| B | TF32 ms/step | vs FP32 non-graph |
|---|---|---|
| 16 | 12.294 M | 0.95× (slower) D |
| 32 | 12.477 M | 0.97× D |
| 64 | 12.408 M; graph 12.164 M | 1.08× D |
| 128 | 13.770 M | **1.41×**, 9,295 tok/s, 28.7 TFLOP/s D |

TF32 pays off only once the batch is compute-bound, from about B=64.

### 3. CUDA graphs (decode step captured once, replayed per token)

Tokens, positions and greedy ids live on the device, so one graph replays every step.

| B | non-graph → graph | gain |
|---|---|---|
| 1 | 10.177 → 9.973 ms M | 1.020× D (0.20 ms) |
| 8 / 64 / 128 | 11.543→11.263, 13.359→13.036, 19.465→19.266 M | 1.025×, 1.025×, 1.010× D |

- Predicted ~1.13× from the ~1.3 ms launch overhead. Against lean C0, G1 measures 1.057× (D).
- The fast engine already removes most of the lean overhead without graphs: it fuses bias, RoPE and the cache write, fuses residual adds into RMSNorm, needs no host top-2 readback, and drops from ~23 to 11 launches per layer. F1 alone is 1.036× faster than C0 (D). Graphs add the last 2%.
- G1 runs at 92.4% of 669.9 GB/s. That leaves 0.75 ms/token above the pure weight-streaming floor of 9.22 ms (D).

### 4. Incremental state hashing (lean path, 8 prompts × 64 tokens)

| Arm | ms/token median | Overhead vs H0 | Fit of overhead vs position |
|---|---|---|---|
| H0, no hash | 10.543 M | — | — |
| HFULL, full-cache SHA every token (= overhead C3) | 18.854 M | +8.31 ms (+78.8%) D | 1.93 ms + 0.125 ms × pos D |
| HCPU, incremental chain, hashed on the CPU | 12.170 M | +1.63 ms (+15.4%) D | 1.61 ms + 0.0004 × pos D |
| **HASYNC**, same chain hashed on a worker thread | 10.615 M | **+0.071 ms (+0.68%)** D | constant; 1.82 ms drain after the last token M |
| **HGPU**, same chain with SHA-256 leaves on the GPU | 10.673 M | **+0.130 ms (+1.23%)** D | constant |

The target of constant per-token cost is met. The HFULL fit reproduces the overhead job's 1.95 + 0.125 × pos.

### 5. Reduced-precision weight lanes (custom warp-per-row GEMV, FP32 activations, FP32 accumulation)

| Lane | Bytes/step | B=1 ms/token, non-graph → graph | vs C0 | GB/s, % of 669.9 (graph) | B=8 graph | B=64 |
|---|---|---|---|---|---|---|
| FP32 (G1) | 6,174,857,216 | 9.973 M | 1.057× D | 619, 92.4% | 11.263 M | 13.036 M |
| **BF16** | 3,087,718,400 | 5.503 → **5.364** M | **1.966×** D | 576, 85.9% D | 5.730 M | 39.84 M |
| **INT8**, per-row absmax/127 | 1,547,337,216 | 3.224 → **3.075** M | **3.429×** D | 503, 75.1% D | 9.306 M | 70.77 M |

- Predicted ~2× for BF16 and ~4× for INT8; measured 1.97× and 3.43× at B=1.
- The custom GEMV re-streams the weights for every 8 rows, so these lanes lose at B ≥ 16. BF16 at B=64 runs at 1,606 tok/s against 4,791 for FP32. The INT8 unpack is also ALU-bound at B=8.
- These lanes are a batch-1 latency tool as built. A BF16 batched path needs a tensor-core GEMM (activations in BF16) or a weight-stationary tile kernel; that was not built.

## Exactness gates

Reference: lean C0, 16 prompts × 64 greedy tokens. "Final logits" are the logits after the 64th decode step. They are compared only where the arm fed C0's ids, so the context is identical; FORCE arms are teacher-forced on C0's ids. KL is KL(p_C0 ‖ p_arm) in float64.

| Arm | Token ids = C0 | First-step logits max / mean abs Δ | Final logits max / mean abs Δ | KL max | Top-1 same |
|---|---|---|---|---|---|
| F1s: fast engine, one-row prefill, B=1 | 16/16 M | 3.3e-5 / 2.9e-6 M | 5.5e-5 / 3.9e-6 M | 2.1e-11 | 16/16 |
| F1: batched prefill, B=1 | 16/16 M | 3.3e-5 / 3.9e-6 M | 3.0e-5 / 3.3e-6 M | 1.4e-11 | 16/16 |
| B8 / B16 / B32 / B64 / B128 | 16/16, 32/32, 64/64, 128/128, 256/256 M | — | ≤ 5.1e-5 / ≤ 5.0e-6 M | ≤ 3.9e-10 | 16/16 |
| FORCE fp32 B=16 | (forced) | — | 3.7e-5 / 4.0e-6 M | 2.1e-11 | 16/16 |
| **TF32** B=16 / 32 / 64 / 128 | **15/16**, 30/32, 128/128, 128/128 M | 9.0e-3 / 1.3e-3 M | 1.2e-2 / 1.6e-3 M (forced: 1.1e-2) | 2.4e-6 | 16/16 |
| **BF16** B=1 / 8 / 64 | 16/16, 16/16, 64/64 M | 3.2e-5 / 3.2e-6 M | 2.3e-5 / 3.1e-6 M | 7.8e-12 | 16/16 |
| **INT8** B=1 / 8 / 64 | **7/16**, 7/16, 28/64 M | 0.675 / 0.077 M | forced: 0.869 / 0.085 M | 5.2e-3 | first step 13/16 |

- **No FP32 arm flipped a token** (M). Batched prefill, batched decode at every B, and the fast engine's sgemv-plus-custom-attention path all differ from C0 only by summation order, at ≤ 5.5e-5 in a logit. No arm is bit-identical to C0 (0/16), as expected, because the attention kernel and sgemm order differ.
- **CUDA graphs are bit-identical** to the same arm without graphs (M). This holds for final logits 16/16 and ids at B = 1, 8, 16, 32, 64 and 128, for TF32 B=64, and for the BF16 and INT8 lanes.
- **TF32 flips one prompt** (prompt 10, token 54) at B=16 and B=32 (M). That prompt's final logits differ by 18.7 once the contexts diverge.
- **BF16 is exact in weights for this model.** The build counted 0 of 1,543,569,408 matrix elements as not exactly representable in BF16 (M). The FP32 bundle was widened from the BF16 checkpoint, so the BF16 lane differs from FP32 only by summation order. That is why it matches C0 to 2.3e-5 with no flips.
- **INT8 is approximate:** 7/16 prompts keep all 65 tokens, and 3 prompts flip the very first token (M). Teacher-forced KL is ≤ 5.2e-3 nats and mean |Δlogit| is 0.085.

### Incremental digest semantics (`inc-merkle1k-v1`)

| Digest | Definition | Coverage |
|---|---|---|
| Full-cache SHA, existing `S` command / C3 | SHA-256 of all live K/V rows, ordered kind, layer, head, then rows 0..pos−1; plus a separate SHA-256 of the logits | Recomputed every token |
| Incremental chain (new) | Each token t writes one row per layer and head. `kv_root(t)` is the Merkle root over 1 KiB leaves of that row set (57,344 B, same kind/layer/head order). `logits_root(t)` is the Merkle root over 1 KiB leaves of the 607,744 logit bytes. `chain_t = SHA256(chain_{t−1} ‖ kv_root ‖ logits_root)`, starting from 32 zero bytes. Prompt rows enter with a zero logits root. | Every K/V row ever written, once, plus every decode step's logits. Depends on write order and history. |

- The incremental chain is not comparable to the full-cache SHA.
- The two digests cover the same K/V bytes only while rows are append-only and immutable. That holds for the lean path: rollback truncates and later rows overwrite. A caller that rolls back must restore `chain_p` from a per-position stack; the driver does not keep one.
- Gates (M): all three incremental modes give the same chain on 8/8 prompts. Every run passed verification on 8/8: each `kv_root` was re-derived from the final cache and the chain replayed. GPU SHA-256 leaves equal CPU SHA-256 on the same bytes (smoke self-test).

## How to enable each mode

All modes go through `QwenFast` in `src/qwen_fast.h`. Build `make speedups-run`, or link `build/qwen_fast.o` with `-lcublas -lpthread`.

```c
QfWeights w = {...};                                   // device pointers from a loaded QwenModel (see tests/speedups_run.c)
QwenFast *f = qf_open(&w, max_seqs, capacity, max_rows, max_steps);
qf_prefill(f, seq, tokens, n, /*batched=*/1);          // 1. batched prefill
qf_decode(f, B, NULL);                                 // 2. one step for sequences 0..B-1
qf_graph_capture(f, B); qf_graph_step(f);              // 3. CUDA graph replay
qf_set_tf32(f, 1);                                     //    TF32 for multi-row sgemm only
qf_build_low_precision(f, free_cb, ctx); qf_set_lane(f, QF_BF16 /* or QF_INT8 */);   // 5.
qf_read_ids(f, seq, ids, n); qf_read_logits(f, row, host);
```

- Incremental hashing (4) is a driver-level recipe in `tests/speedups_run.c` (`hash_token`, modes HCPU/HASYNC/HGPU), built from `qf_gather_rows` and `qf_sha_leaves`.
- `qf_build_low_precision` with a free callback frees the FP32 matrices, which retires the FP32 and lean paths for that model. Pass NULL to keep both, at +4.6 GB of VRAM.
- Limits:
  - `head` must be 128.
  - Capacity × 4 B must be ≤ 48 KiB (shared-memory scores).
  - The fast engine has no observers, writers or checkpoints. Inspection features stay on the lean path.

## Reproduce

```bash
MRUN_URL=http://127.0.0.1:19025 PYTHONPATH=<mrun-quant-gpu>/src <mrun venv python> research/speedups/launch.py stage NAME
MRUN_URL=http://127.0.0.1:19025 PYTHONPATH=<mrun-quant-gpu>/src <mrun venv python> research/speedups/launch.py submit NAME --vram-mb 9200 --ram-mb 7000
```

- `plan-NAME.json` sets the number of prompts, steps and hash prompts.
- The worker builds in the job, runs `build/speedups-run` (FP32 smoke, arms, hashing, conversion, per-lane smoke, lane arms) and analyses `runs.jsonl` plus the logit banks.
- Measured peak: 7,190 MiB own GPU memory and 4.1 GB RSS (M).

## Files

- [src/qwen_fast.h](../src/qwen_fast.h), [src/qwen_fast.cu](../src/qwen_fast.cu) — the executor and the hashing kernels.
- [tests/speedups_run.c](../tests/speedups_run.c) — every arm and gate. Build with `make speedups-run`.
- [research/speedups/](../research/speedups/) — launcher, worker, plans, source manifests and submission records.
- [docs/evidence/speedups-1d03b313def2/](evidence/speedups-1d03b313def2/README.md) — results, the 0.5 s resource series and the prompts.
