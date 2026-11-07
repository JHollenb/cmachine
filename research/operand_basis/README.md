# Source derived gate and up operands

This completed exploratory cut compiles a rank-64 input basis from the original MLP16 gate/up weights and learned RMS scale, then uses only its three derived factors to form those operands throughout its own two-turn conversation. Exact native SiLU and dense down projection remain. It avoids the selected original G/U calls from the first prompt token, but its native-write fidelity does not improve over deletion in aggregate. See [findings](FINDINGS.md), [prospective plan](PLAN.md) and [Astra's independent results review](RESULTS-ADVISOR.md).

## Executed evidence

| Lane | Job | Outcome |
| --- | --- | --- |
| Mechanics smoke, 2 tokens per answer | `job-23dd9b4d0de9` | Completed; 968 primary steps and 8,040 raw files verified. Short token budgets change later conditioning. |
| Main panel, 16 tokens per answer | `job-51df6b792d28` | Completed; 1,128 primary steps, 9,332 raw files and 81 frozen sources verified. |

The main [report](runs/job-51df6b792d28/report.json), [source/raw verification](runs/job-51df6b792d28/verification.json), [independent saved-only audit](runs/job-51df6b792d28/AUDIT.md), [independent statistics](runs/job-51df6b792d28/advisor-independent-summary.json) and exact executed [payload](runs/job-51df6b792d28/payload/) retain the result. The report SHA256 is `e5d23802467bd705ef6e1d9ea7e3dd8925f52589d0dd2942c325404e171a59b7`.

The isolated [driver](driver.c) links a frozen copy of the existing C runtime. The [compiler](compiler.py) uses only exact exported source weights; it does not import a model framework or receive target activations. [Analysis](analysis.py) and [independent analysis](advisor_analyze.py) reconstruct observations from saved arrays. Ordinary serving source and policy are unchanged. The [target-local evidence coverage](index/job-51df6b792d28/coverage.json) and [standard sweep](index/job-51df6b792d28/sweep-results.json) cover 1,311 scalar observations and 1,128 separate same-ID probes; full raw vectors remain the authority.

Subsequent November 6 [operand restoration and historical-memory findings](../operand_factorial/FINDINGS.md) execute the proposed gate/up factorial. They preserve this original result and distinguish conditional operand usefulness, current-write effects and previously written K/V.

## Reproduction

From `/Users/jakeholl/domains`, run all commands through RTK and the shared environment. These saved-only checks do not execute a model:

```sh
rtk proxy common/bin/common exec python saturn/cmachine/operand_basis/run.py verify saturn/cmachine/operand_basis/runs/job-51df6b792d28
rtk proxy common/bin/common exec python -m unittest discover -s saturn/cmachine/operand_basis -p 'test_*.py'
rtk proxy common/bin/common exec python saturn/cmachine/operand_basis/plot.py
```

For a new CUDA reproduction, stage a new immutable payload before submitting it through mrun:

```sh
rtk proxy common/bin/common exec python saturn/cmachine/operand_basis/run.py stage-smoke
rtk proxy common/bin/common exec python saturn/cmachine/operand_basis/run.py submit ABSOLUTE_PAYLOAD_DIRECTORY
rtk proxy common/bin/common exec python saturn/cmachine/operand_basis/run.py collect JOB_ID
```

`stage-panel` selects the full finite panel after a sound smoke. The launcher declares CUDA, FP32 cuBLAS, batch1, context512, CPU1, one resident C model, no retries, strict preflight and measured reservation provenance. It pins Beast for the host-local exported model. Compilation runs in a CPU NumPy subprocess during that same lease; selected source exports stay near compute, while their checksums, manifests, compiled factors and executed binary are collected. There is no training, fresh activation calibration or automatic fallback.

The static factors cost 4,980,736 bytes per rank-64 program. Both source and random banks remain resident in this diagnostic harness. All original model weights and complete K/V remain. Nominal removal of approximately 1.70% of whole-token learned-linear MACs is an arithmetic count, not a latency measurement or whole-model emulation claim.

The next proposed experiment is a common-parent native/approximate gate × up restoration factorial. It has not been executed as part of this cut.
