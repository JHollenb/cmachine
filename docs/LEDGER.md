# Per-token ledger: what the forward pass already computes, kept

The ledger is a flat float32 row per consumed token. It records scalars that the forward pass computes anyway, or can get from a few short dot products: RMS scales, the residual cross terms, per-head softmax summaries, a SwiGLU gate census, and an exact split of the top-1 / top-2 logit margin into the embedding plus every attention and MLP write. The layout and semantics are frozen in [src/qwen_ledger.h](../src/qwen_ledger.h). Both executors implement it:

- the lean path (`qwen_ledger_enable` / `qwen_ledger_read`, src/qwen.c, src/qwen_cuda.cu);
- the fast batched engine (`qf_ledger_enable` / `qf_ledger_read` / `qf_ledger_read_rows`, src/qwen_fast.cu), in the FP32 and BF16 lanes. The INT8 lane is refused.

Status: built and measured on one checkpoint (Qwen2.5-1.5B) on an RTX 4080 (Beast), 16 prompts × 64 greedy tokens. The reported run is **full2 (job-5c3bc13ed6fd)**, after the post-audit fix round. Every gate passes against pass criteria written into the plan file before the run. full1 (job-0b940ac2bcc8) is the pre-fix run; its independent audit is in [evidence/ledger-job-0b940ac2bcc8/audit/AUDIT.md](evidence/ledger-job-0b940ac2bcc8/audit/AUDIT.md). Labels: **M** = measured, **D** = derived from measured values, **A** = asserted (not measured).

## Contract (as built)

- **Opt-in.** With the ledger off, both executors launch the same kernels as `claude/speedups`. The lean softmax kernel is the historical one, verbatim. The fast engine's `forward`/`finish` launch the original kernels.
- **Native values are bit-identical with the ledger on.** The ledger kernels repeat the native statements in the same order with the same operands (`--fmad=false`) and add side reductions. In the fast engine these are `rms_rows_lg`, `attention_rows_lg`, `swiglu_rows_lg` and `argmax_update_lg`. The lean path uses its own ledger kernels plus a `softmax_row` ledger variant.
- **Device-side only, no sync inside a pass.** Ledger kernels run on the engine stream and are captured in CUDA graphs. Enable the ledger before capture; toggling it frees the graph. There is one device-to-host copy per `qf_ledger_read_rows` (pinned staging) and one per `qwen_ledger_read`.
- **No extra learned-matrix product.** The margin split uses the two tied rows W_a and W_b in the resident lane representation (BF16 widening is exact) and the 2L+1 retained writes of each readout row.
- **Readouts.** In the fast batched prefill, every prompt token gets a row, but only the last has a readout; readout fields are 0 elsewhere. In the lean path every token has a readout.
- **Precision (header rule 5, updated to what is built).**
  - Side reductions are summed in double: block trees and cross-block sums.
  - Dots, norms and the margin split are double throughout.
  - logZ sums its exp terms in double. The lean path uses double exp; the fast engine uses the native FP32 exp values.
  - Per-element terms of the SwiGLU census are FP32 in both executors: `even`, `odd` with `tanhf`, and their products. The fast engine also computes `prod²` per element in FP32.
  - Entropy terms (`p·logf p`) are FP32 in both executors.
  - The fast engine additionally sums at most 4 census elements per thread in FP32 before the double tree.

## Predeclared pass criteria (written in `research/ledger/plan-full2.json` before full2 was staged)

All continuous fields are judged by a **normalized error** |a−b|/s. The scale s is fixed by what the field means:

| field type | scale s |
|---|---|
| positive magnitudes (ρ, norm², even², odd², prod², c_norm) | \|a\| |
| attn_dot | its Cauchy–Schwarz bound ‖x‖·‖w‖ |
| mlp_dot | ‖r‖·‖w‖ |
| even_odd | √(even²·odd²) |
| margin fields | the row's Σ\|terms\| |
| logZ, entropy | max(\|a\|, 1) |
| probabilities | 1 |

| gate | criterion |
|---|---|
| G1 | ids and every logit bank (first and final step, every arm) bitwise equal, for on vs off and onnr vs off |
| determinism | F1/on vs G1/on (all prefill and decode rows) and on vs onnr bitwise equal |
| G2 (lean vs fast, teacher-forced) | token/position/sources/top1/top2 exact; gate counts within ±2; continuous p99 ≤ 1e-4 and max ≤ 1e-2; argmax mismatches ≤ 1e-4 of head cells, each with \|pmax_lean − pmax_fast\| ≤ 1e-3 (a tie) |
| G3 lean (ledger vs float64 recompute from the same run's ports) | counts exact; continuous p99 ≤ 1e-5 and max ≤ 1e-4; port consistency 0 bits; argmax vs float64 softmax mismatches only where the top-2 float64 probabilities differ by < 1e-5 |
| G3 fast | the G2 tier |
| G3 top1/top2 | ledger top1/top2 equal the float64 top-1/top-2 ids computed from the dumped final normalized states × the readout matrix, except where the float64 gap is < 1e-4 |
| G4 | \|margin − Σ terms\| ≤ 1e-4 absolute on every readout row of every arm; ledger split vs float64 split ≤ 1e-5 of Σ\|terms\| |

The G2 tie bound (1e-3) and the max bound (1e-2, the smoke bug-detector level) reflect the size of the cross-executor probability differences seen in full1. They are therefore not blind to earlier data. All other thresholds come from FP32 storage and term precision.

## Results

### Jobs (M)

All runs used host beast, priority 70, `preflight=True` and `retry_on_kill=False`.

| job | plan | what | job wall (mrun) | peak VRAM (sampler, own process) | peak RSS (sampler, process tree) | reservation VRAM / RAM |
|---|---|---|---|---|---|---|
| job-ade1f7d481ff | feas1 | compile, smoke, 2 prompts × 8 tokens, gates, Part 4 on 16 states | 58.8 s | 6592 MiB | 4069 MB | 9400 / 7000 MB |
| job-8d3e1f797eac | prof1 | per-piece fast-ledger profile, 4 prompts × 32 steps, B=1 and 64 | 56.6 s | 6698 MiB | 3998 MB | 8600 / 6000 MB |
| job-0b940ac2bcc8 | full1 | pre-fix: every arm, 16 × 64, gates, Part 4 on 1024 states | 190.6 s | 6818 MiB | 3956 MB | 9000 / 7000 MB |
| **job-5c3bc13ed6fd** | **full2** | **post-fix rerun, same arms, predeclared criteria** | **216.6 s** (worker 208.6, C process 144.8, build 4.7, Part 4 37.6) | **6818 MiB** | **4110 MB** | 8900 / 6500 MB |

- **RSS source.** All RSS numbers above come from one source: the worker's sampler, which sums RSS over the worker process tree every 0.5 s.
- **mrun reports differently.** For full2, mrun's own figures were peak_rss 3933 MB and cgroup memory peak 7150 MB. The cgroup figure includes page cache and exceeded the 6500 MB RAM reservation without the job being killed. The next reservation should be at least 7200 MB.
- **No contamination.** No foreign GPU process was seen in any timed region of any job; every arm reports 0 contaminated regions.

### Timing, full2 (M; deltas D)

Decode step medians, with p10–p90 in brackets.

- B=1 arms: 16 prompts × 64 steps = 1024 timed steps. G8/G64 steps advance every sequence.
- onnr = ledger on, rows read once after the loop. kernel = onnr − off; readback = on − onnr.
- The reference is the speedups job job-1d03b313def2.

| arm | what | steps | off ms (p10–p90) | on ms (p10–p90) | onnr ms (p10–p90) | on − off | kernel | readback | off / speedups ref |
|---|---|---|---|---|---|---|---|---|---|
| C0 | lean FP32, B=1 | 1024 | 10.537 (10.53–10.55) | 11.304 (11.29–11.32) | — | **+0.767 ms (+7.3%)** | — | — | 0.9993 |
| F1 | fast FP32 non-graph, B=1 | 1024 | 10.195 (10.14–10.25) | 10.543 (10.50–10.60) | — | +0.349 ms (+3.4%) | — | — | 1.0017 |
| G1 | fast FP32 graph, B=1 | 1024 | 9.978 (9.92–10.03) | 10.329 (10.28–10.38) | 10.323 (10.27–10.38) | +0.352 ms (+3.5%) | 0.345 | 0.007 | 1.0005 |
| G8 | fast FP32 graph, B=8 | 128 | 11.255 (11.17–11.34) | 11.642 (11.56–11.73) | 11.628 (11.55–11.72) | +0.388 ms (+3.4%) | 0.374 | 0.014 | 0.9993 |
| G64 | fast FP32 graph, B=64 | 128 | 13.038 (12.87–13.21) | 13.688 (13.52–13.87) | 13.627 (13.47–13.84) | +0.650 ms (+5.0%) | 0.589 | 0.062 | 1.0002 |
| bf16_G1 | fast BF16 graph, B=1 | 1024 | 5.362 (5.28–5.45) | 5.728 (5.65–5.81) | 5.722 (5.64–5.81) | +0.366 ms (+6.8%) | 0.359 | 0.006 | 0.9997 |

| arm | prefill off ms/prompt | prefill on | on − off | off / speedups ref |
|---|---|---|---|---|
| C0 (sequential lean) | 210.39 | 225.59 | +15.19 (+7.2%) | 0.9995 |
| F1 (batched) | 11.60 | 12.01 | +0.41 (+3.5%) | 1.0035 |

- **The off arms reproduce the speedups job.** Ratios are 0.9993–1.0035 (D).
- **Paired per-prompt deltas** (auditor's script, rerun) agree with the pooled medians:

  | arm | paired delta range |
  |---|---|
  | C0 | 0.764–0.770 ms |
  | G1 | 0.347–0.358 ms |
  | bf16_G1 | 0.360–0.369 ms |

- **Readback** is 0.006–0.014 ms per step at B ≤ 8 and 0.062 ms at B=64. The cost is in the kernels.

### Lean-path round (after the audit; M from full1 vs full2, delta D)

**What changed in builder L's kernels** (`src/qwen_cuda.cu`):
- **Gate census:** per-element terms are now FP32 (`tanhf`); the block trees and the cross-block sum stay double.
- **Entropy:** terms are now `p·logf p`, summed in double.
- **Margin kernel:** it was a single block running 57 serial reductions. It is now 2L+2 = 58 blocks, one dot each, with the same per-thread order and tree as before, plus a one-thread kernel that forms and sums the terms in the old fixed order.
- **Unchanged:** dots and logZ stay double. The OFF path is untouched.
- **Stub:** `tests/gpu_stub.c` mirrors the FP32 terms.
- **`make ledger-mechanics`** passes locally with tolerances unchanged (TOL_REL 1e-6). The gate-sum error grew from at most 5.8e-8 to at most 1.8e-7, and the entropy error from at most 5.5e-8 to at most 1.1e-7 (M, CPU stub).

**Result:** lean decode overhead went from +1.057 ms (+10.0%) to **+0.767 ms (+7.3%)**, and prefill from +10.0% to +7.2%.

**Still above the 5% line; per-piece split.** No lean per-piece profile exists. prof1 profiled only the fast engine, and no third optimisation round was run. What the data does and does not support:
- (M) The combined change saved 0.290 ms per token. The margin and census parts cannot be separated from these runs.
- (A, not measured) The lean path still launches about 6 extra small kernels per layer: an attention-ledger variant, three single-block dot kernels (attention dot, MLP dot, prod²), and gate partial + finish. That is about 170 launches per token, plus the 2 margin kernels and one blocking row readback. At a few microseconds each on a non-graph stream, launch-bound small kernels of this kind plausibly account for most of the remaining 0.77 ms. Graph capture or fusing the dots into the native RMS/residual kernels (as the fast engine does) would be the next step.

### Fast-engine optimisation round (before the audit; M from prof1, feas1, full1)

prof1 (job-8d3e1f797eac) measured each piece's cost by skipping one piece at a time (graph decode, ledger on, no readback). Results before the fix:

| piece | B=1 cost ms | B=64 cost ms |
|---|---|---|
| SwiGLU gate census (FP64 `tanh` and products per neuron) | 0.433 | 2.822 |
| RMS dots + retained writes | 0.105 | 0.123 |
| readout top-2 + margin kernel | 0.066 | 0.070 |
| attention softmax fields | 0.062 | 0.197 |
| total ledger (off 9.943 / 12.850 ms) | 0.663 (+6.7%) | 3.201 (+24.9%) |
| everything skipped (row init only), minus off | −0.001 | +0.007 |

**Fix.** The census and entropy per-element terms moved to FP32, with FP64 trees. Kernel overhead per step (onnr − off):

| arm | before (feas1) | after (full2) |
|---|---|---|
| G1 | 0.661 ms | 0.345 ms |
| G8 | 0.693 ms | 0.374 ms |
| G64 | 3.190 ms | 0.589 ms |
| bf16_G1 | 0.673 ms | 0.359 ms |

**What remains (A, from prof1 piece costs; not re-profiled).** About 0.23 ms at B=1 comes from RMS dots, the margin and the attention fields. About 0.11 ms is the census's FP64 tree and partial traffic. The fast graph arms remain above the ~3% target: +3.4–3.5% at B=1 and 8, +5.0% at B=64, +6.8% BF16 (M).

### Gates, full2, judged against the predeclared criteria (M)

All pass (`verdicts.all_pass = true` in results.json).

**G1: pass.**
- 30/30 checks.
- Ids are bitwise equal on vs off and onnr vs off for every arm.
- Logit banks are bitwise equal: first and final step for C0, F1, G1 and bf16_G1, and now also first and final step for G8 and G64. The G8/G64 first-step banks were added in this round.
- Fast-vs-lean ids: F1, G1, G8 and bf16_G1 16/16; G64 128/128.
- BF16 conversion: 0 of 1,543,569,408 weights inexact. INT8 refusal works (smoke).

**Determinism: pass.**
- F1/on vs G1/on: **1349/1349** rows bitwise, now including all 309 non-final prefill rows.
- on vs onnr: 16/16 (FP32 and BF16).

**G2: pass.** 1349 rows; 1040 with a readout in both.
- token, position, sources, top1 and top2 are exact.
- gate_small: 102 of 37,772 cells off by 1. gate_off: 25 cells off by 1. gate_on: 0.
- Normalized error, p99 / max per field:

  | fields | p99 | max |
  |---|---|---|
  | ρ | ≤ 9e-7 | ≤ 1.1e-5 |
  | norm², even², odd², prod² | ≤ 3.2e-6 | ≤ 2.4e-5 |
  | dots, even_odd | ≤ 2.9e-6 | ≤ 1.6e-5 |
  | margin fields | ≤ 1.2e-6 | ≤ 1.6e-6 |
  | logZ | 1.9e-6 | 4e-4 |
  | entropy | 2.8e-6 | 1.7e-3 |
  | pfirst | 1.3e-6 | 4.7e-5 |
  | pmax | 1.8e-6 | 1.1e-3 |

- h_argmax: 1 of 453,264 head cells differs. Prompt 6, position 26, layer 9, head 0: lean picks key 25, fast picks key 0, with pmax 0.3393330 vs 0.3393328. This is a tie.
- The two executors' activations genuinely differ (custom attention, sgemm), so G2 bounds executor divergence plus ledger arithmetic. It cannot isolate the FP32-term change; the audit's point stands that "about 3e-6" is a floor, not a measurement of that change.
- The smoke `rows_disagree` check (1e-2) now covers every row of prompt 0 (all prefill rows and every decode row), not 5 rows. It did not fire.

**G3: pass.** 4 prompts × 16 tokens; 64 lean and 64 fast rows.
- **Lean vs float64 recompute from its own ports:**
  - Counts and argmax are exact.
  - Continuous fields have normalized p99 ≤ 2e-7 and max ≤ 1.1e-6 (logZ).
  - Margin: native vs float64 normalized max 9.9e-7, absolute 9.3e-6.
  - Ledger split vs float64 split: normalized max 3.4e-8, absolute 7.8e-7.
- **Argmax vs a float64 softmax of recomputed scores:** 0 mismatches.
- **Port consistency:**
  - input + attention_write == raw: 0 bits differ.
  - output(l) == input(l+1): 0 bits differ.
  - Native margin == ledger margin.
- **Fast vs truth** (G2 tier): p99 ≤ 2.9e-6. There are 4 gate_small and 1 gate_off boundary counts.
- **top1/top2, now independent.** The worker computes float64 top-1, top-2 and top-3 from the 1024 dumped final normalized states × the readout matrix (Part 4 arithmetic). This is independent of the native logits and of `S.top`.
  - Ledger top1 matches **1024/1024**; ledger top2 matches **1024/1024**.
  - Near-ties (gap < 1e-4): one state on top1 (prompt 10, position 77, gap 9.5e-5); its ids still match. None on top2; the smallest top2–top3 gap is 2.0e-3.
  - pfirst/pmax/argmax in G3 still read the native probability buffer by the header's definition. The new float64-softmax argmax check is the independent part.

**G4: pass.** remainder = margin − (embed + Σ attn + Σ MLP), over every readout row of every arm.

| arm | rows | \|rem\| median | \|rem\| p99 | \|rem\| max | \|rem\|/Σ\|terms\| max |
|---|---|---|---|---|---|
| C0 lean | 1349 | 2.2e-6 | 9.6e-6 | 1.8e-5 | 1.6e-6 |
| G1 / F1 FP32 | 1040 | 2.3e-6 | 8.7e-6 | 1.2e-5 | 1.4e-6 |
| G8 | 1040 | 3.6e-6 | 1.6e-5 | 2.4e-5 | 1.7e-6 |
| G64 | 8320 | 1.4e-5 | 5.1e-5 | 6.9e-5 | 8.3e-6 |
| bf16_G1 | 1040 | 9.5e-7 | 3.8e-6 | 4.8e-6 | 7.2e-7 |

**What the remainder measures.**
- The ledger's split agrees with a float64 split of the same writes to ≤ 7.8e-7 absolute (G3 margin_sum).
- The *native* margin (FP32 logits) differs from the float64 margin of the same final state by up to 9.3e-6 (G3 margin).
- So the closure remainder mostly measures the native FP32 logit rounding, not the ledger's term storage.
- Example (G3 record 0): native margin 0.1624241, float64 margin 0.1624309, ledger Σ terms 0.1624313.
- G64's larger remainder matches the larger logit error of its sgemm summation order.
- The worst \|rem\|/\|margin\| ratios (up to 0.11) occur on rows with margin ≈ 1e-4.

**RMS identity (lean, M).** d(1/ρ_mid² − 1/ρ_in²) vs 2·attn_dot + attn_norm2:
- Median relative 1e-6.
- p99: 1.9e-4 (attention), 7.7e-5 (MLP).
- Max: 0.23 on one MLP cell, from the 1/ρ² cancellation of float32 ρ.

### What the ledger shows (one checkpoint, 16 prompts × 64 greedy tokens, lean C0 decode rows; M)

These describe one model on 16 prompts. Per-layer statistics pool 1024 rows, but rows within a prompt are autocorrelated, so the effective sample is closer to 16 than to 1024. Means and medians are both given. The full per-prompt table is in results.json under `ledger_read.per_prompt`.

**Margin.**
- The top-1/top-2 margin has median 1.54 and mean 2.63; p10 is 0.23 and p90 is 6.9.
- The mean is pulled up by prompt 2 (code), whose mean margin is 7.46. The other prompts' means range from 1.11 to 4.45.
- It splits into:
  - embed: mean +0.008, median +0.002;
  - Σ attention: mean 1.46, median 1.28;
  - Σ MLP: mean 1.17, median 0.77.
- **Last layer's MLP.** It subtracts from the margin on average (mean −1.16, median −0.95). Its mean is negative in all 16 prompts, from −0.36 to −2.14. Through layer 26, the MLP writes add +2.33 on average.
- **Late attention.** Attention layers 19–27 add +1.39 of the +1.46 attention total. Σ attention is positive in all 16 prompts' means.
- **Per-write size.** Mean |contribution| per write is 0.1–0.3 in layers 0–18. In layers 23–27 it is 0.6–1.9 (medians 0.34–1.47).

**First-row mass, effective rows, gates, census, write alignment** (per layer):

| layer | margin attn mean / median | margin MLP mean / median | p(first row) mean | argmax=row0 | eff rows mean / median | gate \|g\|<1 | g<=-5 (frac) | g>=5 | prod2 median | even2 median | odd2 median | corr(even,odd) median | (even2+odd2)/prod2 median | cos(x, attn w) median | cos(r, MLP w) median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.025 / 0.012 | 0.029 / 0.024 | 0.007 | 0.00 | 10.1 / 3.7 | 7989 | 0 (0%) | 0.0 | 103 | 150 | 61.6 | -0.57 | 2.03 | +0.005 | +0.154 |
| 1 | 0.004 / 0.005 | 0.006 / -0.005 | 0.029 | 0.08 | 22.5 / 20.2 | 44 | 6569 (73%) | 0.0 | 52.4 | 3.61e+06 | 3.61e+06 | -1.00 | 1.29e+05 | +0.123 | +0.057 |
| 2 | 0.014 / 0.012 | 0.022 / 0.007 | 0.467 | 0.81 | 8.5 / 6.1 | 207 | 4155 (46%) | 0.2 | 143 | 7.31e+04 | 7.15e+04 | -1.00 | 987 | +0.075 | +0.096 |
| 3 | 0.005 / 0.001 | 0.012 / 0.022 | 0.464 | 0.82 | 8.8 / 6.2 | 952 | 2068 (23%) | 0.2 | 192 | 2.13e+04 | 2.01e+04 | -1.00 | 214 | +0.087 | +0.016 |
| 4 | 0.004 / 0.001 | -0.009 / -0.004 | 0.610 | 0.87 | 5.1 / 4.0 | 2646 | 388 (4%) | 0.1 | 258 | 7.71e+03 | 6.72e+03 | -0.99 | 58.8 | +0.044 | -0.063 |
| 5 | 0.001 / 0.006 | 0.004 / 0.002 | 0.491 | 0.77 | 7.3 / 4.9 | 421 | 2307 (26%) | 0.2 | 189 | 3.48e+04 | 3.29e+04 | -1.00 | 357 | -0.051 | -0.215 |
| 6 | -0.000 / 0.002 | 0.041 / 0.038 | 0.617 | 0.83 | 5.1 / 3.4 | 5432 | 1 (0%) | 0.0 | 227 | 905 | 468 | -0.88 | 5.95 | +0.149 | -0.162 |
| 7 | -0.008 / -0.006 | 0.027 / 0.026 | 0.289 | 0.58 | 12.2 / 9.2 | 6128 | 0 (0%) | 0.0 | 220 | 611 | 255 | -0.82 | 3.91 | +0.100 | -0.178 |
| 8 | 0.002 / 0.002 | 0.006 / -0.000 | 0.535 | 0.79 | 6.4 / 4.6 | 5857 | 0 (0%) | 0.0 | 227 | 734 | 333 | -0.85 | 4.59 | +0.016 | -0.177 |
| 9 | 0.011 / 0.008 | 0.024 / 0.019 | 0.487 | 0.79 | 7.3 / 6.0 | 6404 | 0 (0%) | 0.0 | 224 | 604 | 243 | -0.81 | 3.71 | -0.003 | -0.122 |
| 10 | -0.014 / -0.011 | -0.003 / 0.003 | 0.425 | 0.74 | 9.2 / 7.4 | 6179 | 1 (0%) | 0.0 | 220 | 721 | 323 | -0.85 | 4.74 | -0.097 | -0.104 |
| 11 | -0.004 / -0.008 | 0.014 / 0.014 | 0.326 | 0.68 | 10.5 / 8.8 | 5926 | 6 (0%) | 0.1 | 208 | 904 | 505 | -0.89 | 6.72 | -0.016 | -0.102 |
| 12 | -0.003 / -0.003 | 0.024 / 0.015 | 0.501 | 0.82 | 7.5 / 6.2 | 6545 | 2 (0%) | 0.0 | 190 | 561 | 251 | -0.83 | 4.2 | -0.045 | -0.120 |
| 13 | 0.010 / -0.001 | 0.007 / 0.006 | 0.288 | 0.59 | 10.3 / 8.9 | 7051 | 1 (0%) | 0.0 | 158 | 438 | 184 | -0.81 | 3.85 | +0.089 | -0.077 |
| 14 | -0.000 / 0.003 | 0.016 / 0.013 | 0.606 | 0.82 | 5.9 / 3.7 | 6999 | 0 (0%) | 0.0 | 161 | 468 | 188 | -0.83 | 4.01 | -0.115 | -0.128 |
| 15 | -0.003 / -0.003 | 0.007 / 0.010 | 0.311 | 0.63 | 11.4 / 9.5 | 6982 | 1 (0%) | 0.0 | 164 | 427 | 181 | -0.80 | 3.66 | -0.018 | +0.028 |
| 16 | 0.000 / 0.001 | 0.014 / -0.003 | 0.378 | 0.73 | 11.7 / 10.3 | 6836 | 0 (0%) | 0.0 | 199 | 490 | 208 | -0.78 | 3.46 | -0.198 | +0.049 |
| 17 | 0.020 / 0.016 | 0.030 / 0.036 | 0.463 | 0.80 | 9.2 / 8.0 | 6660 | 1 (0%) | 0.1 | 259 | 691 | 346 | -0.79 | 3.96 | +0.037 | +0.005 |
| 18 | 0.009 / 0.011 | 0.030 / 0.025 | 0.339 | 0.63 | 9.2 / 7.8 | 6686 | 1 (0%) | 0.1 | 314 | 904 | 505 | -0.81 | 4.41 | +0.108 | +0.074 |
| 19 | 0.142 / 0.048 | 0.184 / 0.135 | 0.575 | 0.76 | 5.1 / 3.1 | 5948 | 2 (0%) | 0.1 | 516 | 1.19e+03 | 644 | -0.76 | 3.55 | +0.083 | +0.131 |
| 20 | 0.027 / 0.013 | 0.128 / 0.091 | 0.433 | 0.71 | 9.1 / 7.8 | 5998 | 1 (0%) | 0.1 | 558 | 1.41e+03 | 758 | -0.77 | 3.8 | +0.093 | +0.089 |
| 21 | 0.075 / 0.048 | 0.244 / 0.218 | 0.437 | 0.78 | 9.8 / 8.4 | 4823 | 7 (0%) | 0.3 | 962 | 3.86e+03 | 2.53e+03 | -0.87 | 6.53 | +0.256 | +0.226 |
| 22 | 0.118 / 0.042 | 0.381 / 0.310 | 0.562 | 0.82 | 5.3 / 4.0 | 4233 | 17 (0%) | 0.6 | 1.17e+03 | 5.46e+03 | 3.79e+03 | -0.88 | 7.56 | +0.192 | +0.312 |
| 23 | 0.292 / 0.152 | 0.415 / 0.315 | 0.659 | 0.89 | 4.6 / 3.6 | 3877 | 13 (0%) | 1.1 | 1.24e+03 | 5.78e+03 | 3.98e+03 | -0.89 | 7.62 | +0.404 | +0.306 |
| 24 | 0.135 / 0.058 | 0.351 / 0.223 | 0.593 | 0.83 | 5.6 / 4.1 | 4143 | 18 (0%) | 5.2 | 1.57e+03 | 6.51e+03 | 4.55e+03 | -0.86 | 6.62 | +0.464 | +0.344 |
| 25 | 0.089 / 0.047 | 0.390 / 0.183 | 0.655 | 0.92 | 4.9 / 3.8 | 4920 | 14 (0%) | 4.5 | 1.86e+03 | 6.3e+03 | 4.42e+03 | -0.83 | 5.42 | +0.409 | +0.312 |
| 26 | 0.217 / 0.160 | -0.061 / -0.107 | 0.591 | 0.98 | 7.6 / 6.3 | 5565 | 17 (0%) | 8.0 | 3.05e+03 | 3.75e+04 | 3.58e+04 | -0.96 | 23.2 | +0.637 | +0.338 |
| 27 | 0.290 / 0.256 | -1.163 / -0.950 | 0.007 | 0.00 | 21.1 / 20.1 | 3240 | 121 (1%) | 40.6 | 1.47e+04 | 2.78e+04 | 2.34e+04 | -0.72 | 3.53 | +0.485 | -0.624 |

- **First-row mass.**
  - Layers 0, 1 and 27 put little mass on row 0: mean p(first) 0.007, 0.029 and 0.007.
  - Layers 2–26 put 0.29–0.66 on it, and the argmax is row 0 in 58–98% of heads.
- **Gate census** (8960 neurons; hard-off = g ≤ −5):
  - Hard-off by layer: 1 is 73%, 2 is 46%, 3 is 23%, 4 is 4%, 5 is 26%. Layers 0 and 6–26 have at most 18 (0.2%); layer 27 has 121 (1.4%).
  - Soft (\|g\| < 1): 7989 at layer 0, and 3880–7050 in layers 6–26.
  - Hard-on (g ≥ 5) is rare until layers 24–27: means 5.2, 4.5, 8.0 and 40.6.
- **Census energies.**
  - The product energy prod² = Σ(SiLU(g)·u)² is given with its two parts, even² = Σ(g·u/2)² and odd² = Σ(g·u·tanh(g/2)/2)², and their correlation corr = Σ even·odd / √(even²·odd²). Since even + odd = product, prod² = even² + odd² + 2·Σ even·odd.
  - even and odd are strongly **anti-correlated** in every layer. The median corr is −0.57 at layer 0, −1.00 at layers 1–3 and 5, −0.99 at layer 4, −0.76 to −0.89 in layers 6–25, −0.96 at layer 26 and −0.72 at layer 27.
  - Why: at a hard-off neuron, tanh(g/2) ≈ −1, so odd ≈ −even and the two cancel. The product is then small even though each part is large.
  - So (even²+odd²)/prod² is not 1. Its median is 2.0 at layer 0, about 1.3e5 at layer 1, 987 at layer 2, 214 at layer 3, 59 at layer 4, 357 at layer 5, 3.5–7.6 in layers 6–25, 23 at layer 26 and 3.5 at layer 27.
  - odd²/(even²+odd²) is **not** a share of the product energy. The full1 write-up called it one; that was wrong.
- **Write alignment.**
  - Attention writes are close to orthogonal to the residual in layers 0–20 (median cos −0.20 to +0.15). They align from layer 21 on (+0.19 to +0.64).
  - MLP writes are anti-aligned in layers 4–14 (median cos −0.06 to −0.22) and aligned in layers 21–26 (+0.23 to +0.34).
  - The last MLP write has median cos(r, w) −0.62 and ‖w‖/‖r‖ 0.67.
  - Its alignment with the residual and its negative margin contribution are separate measurements. Their link (alignment along r vs along c) was not tested.

### Part 4: screened exact head, feasibility arithmetic (D from M states; not a measured speedup)

full2's Part 4 output compares equal to full1's (per-k table, spectrum and row norms; M). That is expected, because the lean native path did not change.

**Setup.**
- Basis: rank-k SVD of W·diag(γ). γ matches the TCM γ bitwise.
- States: 1024 lean final-norm decode states.
- A row stays a candidate if its upper bound (head logit + ‖state tail‖·‖row tail‖, padded) reaches the best lower bound.
- MAC ratio = (screen MACs + exact MACs on candidates) / (V·d), with V·d = 233.4 M.
- The float64 exact top1 equals the lean top1 in 1024/1024 states.

| k | spectral energy in top k | σ_k (σ_max 490.7) | state tail/‖n‖ median (p10–p90) | row tail/‖row‖ median | top-1 candidates median (max) | MAC ratio top-1 | uniform-bound variant |
|---|---|---|---|---|---|---|---|
| 32 | 0.179 | 49.2 | 0.94 (0.92–0.97) | 0.91 | 151,936 (151,936) | 1.021 | 1.021 |
| 64 | 0.217 | 45.9 | 0.92 (0.89–0.95) | 0.89 | 151,936 | 1.042 | 1.042 |
| 128 | 0.282 | 42.3 | 0.89 (0.86–0.93) | 0.85 | 151,936 | 1.084 | 1.084 |
| 256 | 0.386 | 37.0 | 0.85 (0.81–0.89) | 0.79 | 151,936 | 1.168 | 1.168 |
| 512 | 0.548 | 33.3 | 0.80 (0.75–0.85) | 0.67 | 151,936 | 1.337 | 1.337 |

**Verdict: negative.**
- Every state keeps the whole vocabulary as candidates at every k, so the screen only adds work (best 1.021 at k=32).
- The spectrum is flat: 55% of the energy is in the top 512 of 1536 directions.
- Per the audit's arithmetic, ‖n‖ = 39.19 for every state. The Cauchy–Schwarz half-width ‖n_tail‖·‖row_tail‖ is therefore about 75 at k=512 and about 120 at k=32, against logit gaps of at most 15.
- The bound is correct, just loose: 0 exact top-1 and 0 exact top-2 escapes.

### Audit must-fix items: what changed

| # | audit item | change |
|---|---|---|
| 1 | "odd share of the product energy" was wrong (`LEDGER.md`, `worker.py` `odd_share_mean`) | The worker now reports per layer, with mean and median: prod², even², odd², the even–odd correlation and (even²+odd²)/prod². The read above describes the cancellation; the word "share" is gone. |
| 2 | "Layers 1–3 are mostly hard-off" | Replaced by per-layer hard-off fractions, including layers 4 and 5. |
| 3 | G3 top1/top2 were copied from the same source as the ledger | The worker computes float64 top-1/2/3 from dumped states × readout for all 1024 decode states and compares ids, with near-ties reported separately. Result: 1024/1024 and 1024/1024, with one near-tie on top1. G3 no longer lists top1/top2 as "exact". |
| 4 | Closure wording | The G4 text now says the remainder mostly measures native FP32 logit rounding. Split vs float64 is ≤ 7.8e-7; native margin vs float64 is ≤ 9.3e-6. |
| 5 | G8/G64 had only final-step banks | Added first-step banks in `tests/ledger_run.c`. G1 now has 30 checks, including `logits_first_G8/G64` (16/16 each). |
| 6 | prof1 "everything skipped" sign | Corrected to +0.007 at B=64. |
| minor | no predeclared G2–G4 thresholds | Criteria are in `plan-full2.json`, written before staging. The worker judges them (`verdicts` in results.json). |
| minor | determinism collapsed prefill rows | The key now includes position: 1349/1349 rows compared. |
| minor | `rows_disagree` covered 5 rows | Now every row of prompt 0 (prefill and decode). |
| minor | RSS sources mixed | One named source (the sampler's process-tree RSS). mrun's figures are given separately for full2. |
| minor | read section: mean only, correlated rows, prompt 2 | Mean and median are given, a per-prompt table is in results.json, the correlated-row caveat is stated, and the cos-to-margin link is marked untested. |

### Auditor's scripts rerun on full2 ([evidence/ledger-job-5c3bc13ed6fd/audit-rerun/](evidence/ledger-job-5c3bc13ed6fd/audit-rerun/); M)

The scripts are copied unmodified. Their outputs are saved as `*.out` next to them.

- **`timing_recompute.py`:** every median, p10/p90 and prefill median recomputes exactly ("OK" on all 17 arm/modes). The overheads, kernel/readback split and baseline ratios match results.json. Paired per-prompt deltas: C0 0.764–0.770, G1 0.347–0.358 ms.
- **`census_identity.py`** (C0 and FORCE_F1 rows, prefill included):
  - Identity residual relative to the cancelling scale: median 2.1e-8, max ≤ 3.3e-7.
  - Counts are ≤ 8960.
  - 0 Cauchy–Schwarz violations.
  - Per-layer corr and (even²+odd²)/prod² match the table above.
- **`g3_recheck.py`** (records 0 and 63):
  - 0 bits differ on all three port identities, including raw + mlp_write == output.
  - Dots ≤ 5.8e-8 relative; logZ ≤ 3.5e-7.
  - Record 0 margin: native 0.1624241, float64 0.1624309, split 0.1624313.
  - The script's `entropy_vs_p64` (1.8e-3 and 4.1e-2) compares the ledger entropy, which is computed from the native FP32 probabilities, against a float64 softmax of recomputed scores. The same records in full1 give the identical 1.84e-3 and 4.10e-2 (`g3_recheck_full1_compare.out`). So this is the native probability error, not the FP32 entropy terms.
- **`custody.py full2`:**
  - All 97 manifest files match the staged payload and the current tree.
  - No tree files are missing from the manifest.
  - The worker sha256 `236f5b7b…` matches results.json meta.

## Deviations and not done

- **INT8 lane refused** (allowed by the spec).
- **FP32 per-element terms in both executors** (census, entropy), with double sums. This is documented in header rule 5. The lean path is no longer an all-FP64 reference; G3 against float64 truth is now the reference check.
- **Fast batched prefill:** only the last row has a readout. A sequential fast prefill keeps only its last one-row pass.
- **Lean overhead still +7.3% (> 5%).** No lean per-piece profile was run (no third round). The suspected remaining cost is launch-bound small kernels (A).
- **Fast graph overhead still above ~3%:** +3.5% at B=1, +5.0% at B=64, +6.8% BF16 (M).
- **G2 thresholds partly informed by full1** (tie and max bounds; see Predeclared criteria).
- **RAM reservation:** full2's cgroup peak (7150 MB, page cache included) exceeded its 6500 MB reservation without a kill.
- **Pre-existing, unchanged:** builder L flagged that the lean off-path softmax kernel (historical, verbatim) reuses `reduction[0]` without a barrier. G1 held in every run, but the race is not fixed, because the off path must stay today's code.
- **Not exercised by these jobs:** the lean BF16-residency margin path (`weight_mode==2`) and the tiled profile.

## Files

- [src/qwen_ledger.h](../src/qwen_ledger.h) — row layout and contract.
- [src/qwen_fast.h](../src/qwen_fast.h), [src/qwen_fast.cu](../src/qwen_fast.cu) — fast-engine ledger.
- [tests/ledger_run.c](../tests/ledger_run.c) — arms and gate data. Build with `make ledger-run`.
- [research/ledger/](../research/ledger/) — launcher, worker (analysis + Part 4), plans, manifests.
- [evidence/ledger-job-5c3bc13ed6fd/](evidence/ledger-job-5c3bc13ed6fd/) (full2, reported; `audit-rerun/`), [evidence/ledger-job-0b940ac2bcc8/](evidence/ledger-job-0b940ac2bcc8/) (full1, pre-fix; `audit/`), [evidence/ledger-job-ade1f7d481ff/](evidence/ledger-job-ade1f7d481ff/) (feas1), [evidence/ledger-job-8d3e1f797eac/](evidence/ledger-job-8d3e1f797eac/) (prof1) — results.json, samples.csv, prompts.txt, plan.json.
