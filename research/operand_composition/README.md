# Cheap MLP16 → cheap attention17

This experiment connects the frozen receiver-aware rank-64 MLP writer and
rank-128 query/key producer on Qwen2.5-1.5B-Instruct. The four arms share full
FP32 K/V, original residual dimensions and one C FP32/pedantic numerical
implementation. Six two-turn conversations run from empty state under each
arm's own generated and consumed tokens. Then three arms replay the joint
arm's recorded history to measure a complete matched-history interaction.

[Plan](PLAN.md), [math and costs](MATH.md), [driver](driver.c),
[launcher and custody](run.py), [worker](worker.py),
[analysis](analysis.py), [independent numerical audit](audit.py).

Frozen predecessors:

- `experiments/2026-11-06-contextual-union/runs/job-7a1a2715be48`,
  final `qwen25-15b/programs/coupled64.npz`.
- `cmachine/query_response/runs/job-b141eadfe952`, selected source-only
  response query image plus original authenticated B128/K128 banks.

Original D/O/V, other layers, vocabulary head and all learned parameters remain.
This tests compatibility of the two selected cheap components under their own
writes. It does not establish whole-model emulation, arbitrary chat, compressed
memory, improved task truth, or a serving speedup.

Use the shared environment:

```sh
rtk proxy common/bin/common exec python saturn/cmachine/operand_composition/run.py stage
rtk proxy common/bin/common exec python saturn/cmachine/operand_composition/run.py submit PAYLOAD
rtk proxy common/bin/common exec python saturn/cmachine/operand_composition/run.py collect JOB
rtk proxy common/bin/common exec python saturn/cmachine/operand_composition/run.py verify RUN_DIRECTORY
```

Only the submitted payload is executable authority. Preserve the returned exact
job and its source manifest, report, raw artifacts and independent verification.
