# Analysis-overhead timing record, job-d69af5c50a71

This is the portable record of the mrun job behind [ANALYSIS-OVERHEAD.md](../../ANALYSIS-OVERHEAD.md). It ran on Beast's RTX 4080 on 2026-11-13, took 180.9 s and succeeded. It ran with priority 70 and a reservation of 8,000 MB RAM and 8,300 MB VRAM.

- `results.json` (SHA256 `1732463a0a98d18cb71dc230a5cb322b3f55658f4783001aa0608120e32a8549`) holds per-arm medians and percentiles, token agreement, per-phase resource summaries, foreign GPU process flags, and run metadata (model digest, binary SHA, GPU and driver).
- `samples.csv` (SHA256 `ff42eaf619ea476d86c32fb09c3336e333b7ca9f1a872dfd9623a3d89c690957`) holds the 361 samples taken every 0.5 s: phase, GPU memory, utilisation, power, clocks, process RSS and CPU.

Per-token rows, logs and the outbox ledger are not copied here. They are stored in MinIO under `s3://mrun/runs/job-d69af5c50a71/`; the manifest shows 8 of 8 objects uploaded and 547 ledger rows.
