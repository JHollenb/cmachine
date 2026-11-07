# Nonlinear state quotient: closure, an escaping write, and repair

This is an isolated source-defined assay of the simplifying assumption in the [gate-law synthesis](../gate_law/FINDINGS.md): discarded information stays invisible to every admitted future read and write except through retained quantities. It complements the existing [B3 code collisions](../closure/FINDINGS.md). It does not learn or install a compressed Qwen model.

The [completed findings](FINDINGS.md) give byte-exact16-update closed and repaired trajectories, an escaping-write output distinction at update2, and an independently verified original-Qwen rank lower bound256. Job`job-d57473b075cc` takes11.62scheduler seconds/250MBVRAM, with zero Qwen forwards. [Raw report](runs/job-d57473b075cc/report.json), [verification](runs/job-d57473b075cc/verification.json), [independent integer audit](runs/job-d57473b075cc/independent-audit.json), [Astra critique](REVIEW.md), and [blog synthesis](../../../obsidian/blog/2026-11-06-135500-the-write-that-breaks-a-compact-model-state.md).

The [frozen plan](PLAN.md), [derivation](MATH.md), [C driver](driver.c) and [saved-operand analyzer](analysis.py) define an eight-dimensional nonlinear recurrence. Four reduced-state trajectories seal before four full-state references. The closed control retains two coordinates and omitted energy; the write escape makes omitted orientation consequential; the repair retains one additional coordinate. Native C CUDA RMS/SwiGLU/cuBLAS primitives execute all paths.

A separate fixed256-square original Qwen layer16 down-weight minor supplies an exact modular rank lower bound without a Qwen forward. This tests the applicability of one strong common-linear-space construction, not every possible compact program. Model/input identity, finite shape contracts, own-state update verification and archive custody remain mechanics requirements. Numerical distances remain observations.

Local checks are limited to five exact-integer/math tests, C syntax/sanitizers and one model-free step using the CPU stub. The full16-step assay uses mrun on Beast. Reproduction commands, from the workspace root:

```sh
rtk proxy common/bin/common exec python -m unittest discover -s saturn/cmachine/state_quotient -p test_analysis.py
rtk proxy common/bin/common exec python saturn/cmachine/state_quotient/run.py stage
rtk proxy common/bin/common exec python saturn/cmachine/state_quotient/run.py submit STAGED_PAYLOAD
rtk proxy common/bin/common exec python saturn/cmachine/state_quotient/run.py collect JOB
rtk proxy common/bin/common exec python saturn/cmachine/state_quotient/run.py verify RESULT_DIRECTORY
rtk proxy common/bin/common exec python saturn/cmachine/state_quotient/audit.py RESULT_DIRECTORY
rtk proxy common/bin/common exec python saturn/cmachine/state_quotient/evidence.py JOB
```

Each submission is one finite lease, strict preflight and no retries, with candidates executing before native references. The model bundle is mapped and completely checksummed for identity but never widened into a Qwen GPU weight bank. Standard evidence sweeps have isolated stream-only scope; the whole-invocation Saturn receipt does not certify each internal CUDA operation or semantic behavior.
