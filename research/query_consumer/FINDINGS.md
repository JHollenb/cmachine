# Native-consumer feedback chooses a different query, with mixed transfer

Selecting the six frozen query images through actual native C source KL chooses dose2, while the preceding attention-response objective chose dose1. Source KL improves a further0.985%, but the gain trades off contexts and clocks. Relative to dose1, target KL improves16/24 readouts and logit L2 improves21/24; all emitted answers stay identical. This strengthens the consumer-feedback mechanism while exposing that the feedback must specify context and future horizon.

[Protocol](PLAN.md), [math](MATH.md), [source selector](driver.c), [independent metrics](metrics.py), [Astra critique](REVIEW.md), [saved-only study](runs/job-97f4f4ac5191/study.json), [horizon audit](runs/job-97f4f4ac5191/horizon-audit.json).

## A frozen family, actual C feedback

Use the same original Qwen2.5-1.5B-Instruct bundle, FP32/pedantic cuBLAS runtime, layer17, B128, compiled K, native V and16-recent/16-prototype reader. The six image bytes come unchanged from [the response experiment](../query_response/FINDINGS.md); no new direction, optimizer or target-derived adjustment is introduced.

Native source193 advances plus six candidate source histories1,158 advances cost1,351 new source advances. Each follows the authenticated original prompt+eight historical own IDs, with independent candidate-owned upper caches. Average KL(native||candidate) across each trajectory, then average three trajectories equally. No vocabulary-size normalization. Exact ties prefer smaller absolute dose then index.

| Frozen dose | Actual C source attention loss | Native source mean KL, nats |
|---:|---:|---:|
|−1|0.04770764|0.004986531|
|0 / LS|0.03915260|0.003711081|
|0.25|0.03777369|0.003441379|
|0.5|0.03678167|0.003210813|
|1 / prior response choice|0.03587933|0.002892418|
|2 / native-KL choice|0.03752226|0.002863934|

Actual C attention loss retains the CPU source surrogate's dose1 minimum, so this ranking difference is not explained by the CPU/CUDA profile change. Native KL selects dose2, the largest tested dose, not a demonstrated global optimum. The selected image SHA is `7b0d047635f17effc91f408ca3b8691269169a0709ffd2fa78fd55623a9f005f`.

C streams every full-vocabulary source row through a second Python FP64 implementation before continuing. All1,158 candidate/native row comparisons agree; maximum absolute discrepancies are3.94e−13 nats KL,1.38e−12 entropy,3.94e−13 native-top logp and3.01e−14 relative L2. Row/stream hashes and all scalar curves remain. The complete temporary streams are removed afterward;63 distinct full-logit vectors at nine source checkpoints per policy support54 independently replayed candidate/native metric pairs. Do not call this full durable logit replay.

Selection/image hashes seal before fresh prompts are parsed or targets tokenized. The C model stays resident while Python prepares the later target manifest. The initial request binds the source family/runtime; the final C result binds both selection and later target manifest under the whole-invocation Saturn receipt.

## The aggregate hides context and time

Dose2 wins the all-clock mean by improving source0 from0.003347841 to0.002919185 nats, while source1 worsens0.003780799→0.004031367 and source2 worsens0.001548615→0.001641251. Per-trajectory all-clock winners are doses[2,1,1].

Post-hoc regional diagnostics choose dose1 for the shared header, dose2 for the prompt body and dose0.5 for the eight-token suffix. Per-trajectory suffix winners are[0.5,0.5,1]. These are diagnostics, not a reselection of the sealed target image. The original selection weights assign37.32% to header clocks,50.24% to prompt-body clocks and12.44% to the later suffix. The first two source phases therefore account for most of the objective.

A saved pending-token audit connects all21 source traces (native+six images across three contexts) to the independently audited full-vocabulary argmax at every source clock. All168 pending choices for the supplied eight-token suffix agree with the original IDs. By deterministic induction, these exact forced traces coincide with eight-step greedy continuations. This supports an on-policy statement for those eight tokens, not longer completion, EOS, another user turn or a separately executed autonomous source lane. Teacher-forcing mismatch does not explain the clock-dependent ranking inside this window; rerunning the same eight tokens would duplicate evidence.

## Target transfer

Six two-turn cases compare LS compact, dose1 compact, source-selected dose2 compact and ordinary native-full execution. Three development cases repeat unchanged; three prospective cases ask ticket-material recall,17×24 then subtraction19, and camping storage prose. All18 candidate arrays seal before six native references. There are no aliases in this run because all three query images differ.

| Selected dose2 compared with | KL gains | Logit L2 gains | Native-top logp gains |
|---|---:|---:|---:|
|LS dose0|17/24|21/24|18/24|
|Prior response dose1|16/24|21/24|18/24|

At16 boundaries still sharing the native consumed history, KL improves13/16 versus LS and12/16 versus dose1. The initial dose2-versus-dose1 KL comparison improves only arithmetic and new arithmetic; recall, prose, ticket recall and camping worsen. Initial L2 improves five of six, with ticket recall the exception. Later operational comparisons include native-history divergence and must retain that qualification.

All six LS/dose1/dose2 emitted-and-consumed target histories are identical. Both recall tasks remain correct. Development arithmetic remains266 then wrong286 rather than279. New arithmetic gives408 then the correct explicit equation `408 - 19 = 389` under every policy, including native. The leading token is408, so a leading-integer-only evaluator would misclassify this calculation; the post-hoc output-only evaluator records formatting and equation truth separately. This correctness already belongs to the baseline and is not created by selection. Explanations are truncated at16 tokens. No general free-chat or answer-accuracy improvement follows.

## Work, custody and qualified recovery

The completed C invocation in `job-97f4f4ac5191` performs1,351 source+2,434 target+32 sham=3,817 physical advances,570 generated/consumed target tokens,24 target paths,72 byte-exact target memory snapshots,1,158 exact source compact reads and1,836 exact target compact reads. All original193 source receiver rows reacquire byte-exactly. Nine historical target paths/135 distinct arrays and all ten bank/family arrays agree exactly with the preceding study. The135 denominator differs from the prior126 because this panel has six compact and three native historical controls.

Serving remains819,200 selected QKV MACs/update,98,464-byte selected logical memory,198 whole-model linear calls versus197 native and14,336 serving transfer bytes/update. Whole-token nominal linear saving remains0.1507%. Source feedback costs267,305 learned-linear calls/2,082,668,150,784 MACs. The full C invocation counts754,943 calls/5,884,838,805,504 MACs. Dense receiver generation, native O/MLP, remaining layers/weights/caches/full vocabulary remain paid; no serving speedup is measured.

Scheduler elapsed100.8s, worker82.49s and driver77.41s; peakRSS2965.9MB/VRAM6406MB. Exact preexisting source data+emitted payload+worst-case target arrays+16MiB metadata allowance bound254,178,756 bytes under256MiB. Actual raw files total219,003,756 bytes. The temporary source streams add bounded host/disk work, not serving state.

There are two retained engineering outcomes. Attempt `job-ad7b8088f482` was cancelled after admission during source work when Astra found the original archive bound omitted copied-payload/metadata costs; it has no completed scientific result. The corrected attempt completed C source selection and every target, then the Python analyzer failed only because its final summary used `advances` instead of driver field `count` twice.

The original scheduler failure and report SHA `0b8f475f60411080434491a0c4010eb496f4da27b3cf1e721e530d337fbeb75c` remain immutable. [Qualified verification](runs/job-97f4f4ac5191/verification.json) authenticates the scheduler-log archive,80 executed source files and651 raw files (733 total archive files), plus the whole-C receipt. [Additive recovery](runs/job-97f4f4ac5191/analysis-recovery.json), SHA `867d7c3e92afa0754722c382f7bd9b3a72432fefe8fc5354211e9cb465a3d7ae`, runs the exact frozen analyzer with only those two summary-key substitutions. No model rerun or tolerance change occurred. Astra independently reproduced the complete recovered analysis and archive checks. This is a completed C assay with recovered analysis, not a successful scheduler job. The post-job evidence sweep verifies3,616 recovered observations; later horizon/study diagnostics have separate custody.

## Working inference and next cut

The native consumer changes which fixed query program it prefers, but an all-clock source average is not an adequate statement of future capability. Context and clock weighting now explain observable disagreements without requiring an unsupported missing-information claim. Six proposals on one direction do not establish optimal F, a sufficient B128 state or absence of a better cheap query.

Next, extend these same three source conversations beyond eight tokens and through one fixed followup, keeping all six image bytes frozen. Predeclare separate prefill, first-answer future and post-followup future measurements and one future-focused source selector. Retain the current all-clock selector as control. On forced native IDs, certify greedy agreement only until each policy's first mismatch; distinguish subsequent forced state from an autonomous branch. Seal the source choice before new targets. Test horizon before adding more contexts, dimensions or memory.
