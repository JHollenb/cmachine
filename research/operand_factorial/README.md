# Operand restoration and historical memory

Completed exploratory Qwen2.5-1.5B-Instruct C FP32 CUDA/cuBLAS assay. Restoring the current MLP16 gate/up operands makes its write exact, while earlier candidate-written K/V leaves a measurable downstream discrepancy. The paired native-same-ID memory parent restores exact native logits. See [findings](FINDINGS.md), [prospective plan](PLAN.md), [report](runs/job-1771ced3fed7/report.json) and [verification](runs/job-1771ced3fed7/verification.json).

The assay also decomposes 36 saved operand clocks into signed gate, payload and interaction terms. It preserves source64's positive conditional operand effects, local-to-consumer countertrends and incorrect native arithmetic. No training, persistent model edit, general chat qualification or serving-speed measurement occurs.

The [isolated driver](driver.c) imports the predecessor-frozen C executor and uses ordinary checkpoints for common-parent branches. [Analysis](analysis.py), [independent array audit](audit.py), [figures](plot.py) and [evidence projection](evidence.py) consume saved arrays. Executed sources and assets are under the run's immutable payload; the current verifier adds receipt and cross-source joins without changing those sources.

```sh
rtk proxy common/bin/common exec python -m unittest discover -s saturn/cmachine/operand_factorial -p test_analysis.py
rtk proxy common/bin/common exec python saturn/cmachine/operand_factorial/run.py verify saturn/cmachine/operand_factorial/runs/job-1771ced3fed7
rtk proxy common/bin/common exec python saturn/cmachine/operand_factorial/audit.py job-1771ced3fed7
rtk proxy common/bin/common exec python saturn/cmachine/operand_factorial/plot.py job-1771ced3fed7
rtk proxy common/bin/common exec python saturn/cmachine/operand_factorial/evidence.py job-1771ced3fed7
```

New CUDA reproduction uses `run.py stage`, then `run.py submit PAYLOAD`, then `run.py collect JOB`. Inspect the queue and measured resource exception before submitting; do not duplicate an active suite. All 84 branches pay native operand diagnostics, and continuation uses native execution after one intervention. Native fidelity and task correctness remain separate.
