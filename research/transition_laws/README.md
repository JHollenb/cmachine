# Substantial compiled MLP transition laws

This finite experiment replaces gate/up construction across all28 Qwen2.5-1.5B MLPs with rank256 shared input factors. It compares a native-receiver compiler against exactly one receiver refresh using the first candidate's own states, with equal factor budget and randomized-SVD dimensions. Full candidate residual/memory, native attention/down/SiLU and vocabulary computation remain.

Completed job `job-a6811af393aa`: physical counters verify 40.895% nominal whole-token linear omission. The refresh improves source coverage and local writes, but both all-layer programs lose useful conversation behavior (0/8 strict structured answers). Read [FINDINGS.md](FINDINGS.md), [ASTRA_REVIEW.md](ASTRA_REVIEW.md) and the [qualified audit](runs/job-a6811af393aa/verification.json), including the preserved frozen-verifier refusal and raw budget-metadata correction. The artifact-only tokenizer audit independently reconstructs all84 answer decodes.

Read [PLAN.md](PLAN.md), [MATH.md](MATH.md) and [REVIEW.md](REVIEW.md) before interpreting results. Prior multi-layer/all-layer pruning is already implemented; the new comparison is substantial factor-map replacement with controlled state-coverage refresh.

From the domains workspace:

```sh
rtk proxy common/bin/common exec python saturn/cmachine/transition_laws/run.py stage
rtk proxy common/bin/common exec python saturn/cmachine/transition_laws/run.py submit STAGED_PAYLOAD
rtk proxy common/bin/common exec python saturn/cmachine/transition_laws/run.py collect JOB_ID
rtk proxy common/bin/common exec python saturn/cmachine/transition_laws/run.py verify JOB_ID
```

`run.py` authenticates the existing C/public runtime and seals new source before mrun strict preflight. `worker.py` owns a single resident C process. It harvests source native and candidate trajectories, computes two fixed-budget receiver subspaces, selects a monitor policy using native-shadow conditional probabilities, seals both factors and selection, then opens fresh evaluation cases. `driver.c` owns original/compiled map dispatch, cache and generated tokens. Native shadow runs follow sealed candidate histories and never supply state to the candidate. Large immutable arrays and factor images are stored together through MinIO; collection authenticates the artifact before extraction.

The live compiler retains original G/U parameters for compilation and independent native diagnostics. Declared whole-token arithmetic savings therefore do not imply that original weights have been removed, and this instrumented job is not a serving latency benchmark. All branch outcomes remain evidence. No default serving mode is changed.
