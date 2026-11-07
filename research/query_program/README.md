# Same-cost conditional query program

One source-only least-squares image replaces the query map on unchanged B128 coordinates. Key formation, native V, bias/RoPE and the 98,464-byte memory reader remain unchanged. Each new query costs the same as the preceding compiled program.

Completed Beast run [job-bd80dd16c1d5](runs/job-bd80dd16c1d5/verification.json): 30 own trajectories, 3,024 token advances, six two-turn cases, 60 byte-exact memory snapshots. The new compact policy improves native-logit fidelity at 23/24 saved boundaries, with identical generated tokens to its baseline. Gains are modest; this is not a whole-model or chat-accuracy result. See [findings](FINDINGS.md), [math](MATH.md), [protocol](PLAN.md), [Astra design advice](ADVICE.md), [pre-run review](PRE_RUN_REVIEW.md) and [independent review](REVIEW.md).

The probability consumer is more mixed: native-to-candidate KL improves at 19/24 boundaries, with two initial countertrends despite improved query/attention/logit L2. [Saved distribution comparisons](runs/job-bd80dd16c1d5/study.json) retain every sign. These favor testing a source-only response-aware objective before enlarging memory or equating reconstruction with predictive quality.

```sh
rtk proxy common/bin/common exec python saturn/cmachine/query_program/run.py stage
rtk proxy common/bin/common exec python saturn/cmachine/query_program/run.py submit ABSOLUTE_PAYLOAD
rtk proxy common/bin/common exec python saturn/cmachine/query_program/run.py collect EXACT_JOB_ID
rtk proxy common/bin/common exec python saturn/cmachine/query_program/run.py verify ABSOLUTE_RESULT_DIRECTORY
```

The frozen payload contains the actual executed source and immutable report. A local strict replay refused 96 duplicated diagnostic scalars because their reference norm used incoming FP32 arithmetic. That refusal is retained in [verification-refusal-diagnostic.json](runs/job-bd80dd16c1d5/verification-refusal-diagnostic.json). The existing exact-environment replay mechanism was reused: [audit_replay.py](audit_replay.py) dispatches a zero-model CPU worker, and [verify_replay.py](verify_replay.py) verifies both scheduler logs, all sources/raw files, the Saturn receipt and exact canonical analysis bytes. CPU job `job-7d8b02674ac8` reproduces the original analysis exactly; no numerical tolerance was relaxed.

[stable_metrics.py](stable_metrics.py) adds independently accumulated FP64 query/attention/output metrics in [fp64-query-metrics.json](runs/job-bd80dd16c1d5/fp64-query-metrics.json), without changing the original evidence. The current working analyzer promotes both operands before computing relative norms, and its regression test covers the dtype error; this is a later source correction, not the executed analyzer. Current run.py adds the authenticated replay-verification route. Those changes are separate from the frozen payload.

The compiler reads only the original three source trajectories, frozen basis/old query image and original model Q weight. Target text is present and hashed in the payload, but fresh prompts are parsed after the image seal and no target enters the solve. Original native source acquisition remains paid, and the new CPU fit is counted separately. Installed serving and the separate public Cmachine repository are unchanged.
