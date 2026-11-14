# Speedups measurement record, job-1d03b313def2

This is the portable record of the mrun job behind [SPEEDUPS.md](../../SPEEDUPS.md). It ran on Beast's RTX 4080 (driver 580.159.03, nvcc 13.0) on 2026-11-13 with priority 70 and a reservation of 7,000 MB RAM and 9,200 MB VRAM. The C process took 180.0 s and the worker 190.3 s. The job succeeded, every smoke gate passed, and no timed region overlapped a foreign GPU process.

- `results.json` (SHA256 `23756496e05f943c3cfca27a04c1bb2acf23a010eee967186597c4ffd40e3296`) holds per-arm medians and percentiles, the derived throughput, bandwidth and FLOP/s, token agreement against C0, the logit gates with per-prompt rows, graph bit-identity, the hashing summary, per-phase resources and run metadata. The metadata covers the source manifest, binary SHA `32386996…`, GPU and nvcc.
- `samples.csv` (SHA256 `bf465915e6f953e2330ec906b1c9ce09196cba41784a76e4b5b7aac20cb7c57c`) holds the 380 samples taken every 0.5 s.
- `prompts.txt` holds the 16 tokenised prompts.

Source: branch `claude/speedups`, commit `f624567`. The payload manifest is `research/speedups/sources-full2.json`.

Per-step rows (`runs.jsonl`) and logs are in MinIO under `s3://mrun/runs/job-1d03b313def2/`; fetch them with `mrun artifacts fetch job-1d03b313def2 --name runs.jsonl --dest <dir>`. The logit banks (`logits-*.f32`, about 300 MB) stayed on Beast under `/mnt/usb-storage/experiment-results/cmachine-speedups/full2-job-1d03b313def2/` and were not uploaded.

Earlier jobs from the same harness:

- `job-637935494ca7` (feasibility, 4 prompts × 16 tokens, commit `1903e69`) passed every gate.
- `job-24ee1aae8339` (full1, commit `f624567`) gave the same gate results. A foreign 2 GB process overlapped all TF32 arms, B128/G128 and 1 of 8 regions in H0, HCPU and HFULL, so its timings were replaced by this rerun.
