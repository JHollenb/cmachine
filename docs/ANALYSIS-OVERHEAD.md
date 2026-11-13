# Analysis overhead: what inspection costs per token

The native C path streams Qwen2.5-1.5B-Instruct FP32 weights at about 87% of the RTX 4080's measured 669.9 GB/s. Earlier timings measured only the lean path: the forward pass, attention and a top-2 readout. This harness measures the same decoder with each analysis feature switched on, and also times the PyTorch baselines used by earlier comparisons, with and without their instrumentation.

Measured in mrun job `job-d69af5c50a71` on Beast (RTX 4080, driver 580.159.03) on 2026-11-13. Source: `1367ed0` plus this harness. Workload: 16 fixed prompts × 64 greedy tokens, batch 1, FP32, arm order rotated per prompt, and every token bracketed by a device sync. Every C arm produced the same token ids as C0 on 16/16 prompts, and C0 matched HF Transformers on 16/16.

## Results

| Arm | Adds | ms/token median (p10–p90) | vs C0 | Effective GB/s | % of 669.9 |
|---|---|---|---|---|---|
| C0 | lean: forward, attention, top-2 | 10.549 (10.534–10.560) | — | 585 | 87.4% |
| C1 | + full logits download (607,744 B/token) | 10.607 | +0.058 ms (+0.6%) | 582 | 86.9% |
| C2 | + `QWEN_INPUT` residual capture, all 28 layers (172,032 B/token) | 10.791 | +0.242 ms (+2.3%) | 572 | 85.4% |
| C2b | + observer capture of every port (4,858,784 B/token, 448 calls) | 13.914 | +3.37 ms (+31.9%) | 444 | 66.2% |
| C3 | + cache and logits SHA-256 every token | 18.998 (15.779–22.206) | +8.45 ms (+80.1%) | 325 | 48.5% |
| C4 | C1 + C2 + C3 | 19.304 | +8.76 ms (+83.0%) | 320 | 47.8% |
| T0 | HF Transformers fp32, sdpa, KV cache, no hooks | 11.963 | — | 516 | 77.0% |
| T1 | T0 + stacked-memory forward-pre-hooks, `.cpu()` final hidden | 11.994 | +0.03 vs T0 | | |
| T1b | T0 + per-layer last-position hidden, one stacked `.cpu()` | 12.063 | +0.10 vs T0 | | |
| T2 | mdb `DecoderSplice`, `capture_kv=range(28)`, `torch.cat` each step | 12.329 | +0.37 vs T0 (+3.1%) | | |
| T3 | DecoderSplice-style layer loop over a `DynamicCache`, capture off | 11.882 | −0.08 vs T0 | | |

Effective GB/s is 6,174,857,216 weight bytes divided by ms/token. Prefill medians, for prompts of about 20 tokens: C 210.4 ms (sequential pushes), T0 14.4 ms, T2 27.4 ms.

## Reading

- Logits download and residual capture at every layer together cost 0.30 ms/token (2.9%). Routine inspection does not change the bandwidth picture.
- Capturing every port is dominated by per-call launch and sync cost, not bytes. The 4.86 MB at 26.8 GB/s PCIe would be 0.18 ms of the extra 3.37 ms.
- Per-token state hashing is the expensive feature. Its overhead fits about 1.95 ms + 0.125 ms × cached position, because the cache SHA reads 57,344 B per position. It runs on the CPU while the GPU waits: median GPU utilisation is 59% in C3 against 100% in C0. Hashing only new rows, or hashing on the device, would make it constant per token.
- The PyTorch baselines' instrumentation is cheap: +0.03 to +0.37 ms/token. Comparisons against the stacked-memory and mdb native arms are therefore within about 3% of comparisons against lean PyTorch.
- At batch 1 the lean C path is 1.13× faster than lean HF Transformers.
- C5, the writer and guard path with a no-op edit, was not measured. Wiring it needs a `QwenSubspace` and projector.

## Resources over the run

The job sampled every 0.5 s (361 samples) with `nvidia-smi` and `/proc`. Times are seconds since worker start.

| t (s) | Phase | Own VRAM | Process RSS | GPU util | Power |
|---|---|---|---|---|---|
| 0–15 | build and staging | 0 | — | foreign job 7–21% | 54–59 W |
| 15.5 | C model load (6.9 s) | → 6,358 MiB | peak 3.96 GB (mapped bundle + SHA) | — | — |
| ~20–106 | C arms | flat 6,358 MiB | 1.26 GB after unmap | 100% C0, 98% C2, 59% C3 | 145 W C0, 192 W C2, 159 W C3 |
| 106.5 | C exits | → ~0 | — | — | — |
| 111.5 | torch load | → 6,318 MiB (peak 6,327) | peak 9.38 GB | — | — |
| ~115–176 | torch arms | flat | 4.42 GB | 94–97% | ~181 W |

- The SM clock held 2,745 MHz throughout, so there was no thermal throttling.
- CPU sat at about 100% in busy phases, because both runtimes spin-wait on syncs.
- The C process allocates nothing during decode. Its GPU footprint is 6,174,857,216 B of weights, 6,164,432 B of session at capacity 95, and CUDA/cuBLAS context.
- The job's cgroup RAM peak was 8,800 MB against a declared 8,000. That is exactly the kill ceiling max(1.1 × 8000, 8000 + 512), reached during the torch model load. Reserve at least 12.2 GB of RAM for any job that loads the HF model.
- A foreign GPU process overlapped one prompt each of C1, C2 and C4 for about 1.5 s. Dropping those regions leaves every median unchanged.

## Run it

```bash
# from this repository, on the laptop; MRUN_URL may be an SSH tunnel to the scheduler
MRUN_URL=http://127.0.0.1:19025 PYTHONPATH=<mrun>/src <mrun venv python> research/overhead_timing/launch.py stage NAME
MRUN_URL=http://127.0.0.1:19025 PYTHONPATH=<mrun>/src <mrun venv python> research/overhead_timing/launch.py submit NAME
```

- `stage` copies the C sources, `tests/overhead_run.c`, the worker, the torch arms and vendored `mdb`/`mrun.outbox` modules into `research/overhead_timing/payload-NAME/` (ignored by Git). It also writes a source manifest, `sources-NAME.json`.
- `submit` sends one Beast job. Do not edit the payload until the job id is printed.
- On Beast, `run.sh` starts `worker.py`. The worker builds `make cuda` and `make overhead-run`, runs a 4-token smoke and then the timed arms in the C process, does the same in the torch process, and writes outbox rows and `samples.csv`.
- `--ram-mb` defaults to 8000. Pass `--ram-mb 12500`, because the torch load peaked at 9.4 GB.
- `plan-full1.json` sets steps, warmup, smoke length, per-process deadlines and the sampling interval.

## Files

- [tests/overhead_run.c](../tests/overhead_run.c) — C driver for arms C0–C4 and C2b. Build with `make overhead-run`.
- [research/overhead_timing/](../research/overhead_timing/) — `launch.py`, `worker.py` (sampler, C process, torch process, summaries), `torch_arms.py`, `run.sh`, the plan, the source manifest and the submission record.
- [docs/evidence/overhead-d69af5c50a71/](evidence/overhead-d69af5c50a71/README.md) — the summarised results and the full 0.5 s sample series.
- Full run artifacts, including per-token rows `c_runs.jsonl` and `torch_runs.jsonl` and logs: `s3://mrun/runs/job-d69af5c50a71/`. Retrieve them with `mrun artifacts fetch job-d69af5c50a71 --name <file> --dest <dir>`.
