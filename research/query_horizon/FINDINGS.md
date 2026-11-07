# Longer query futures and the histories they visit — November 6

The completed `job-309bf320370d` selects the existing dose-2 query image using equal-source/equal-turn native KL over longer answer decisions. Future source KL falls from 0.024110526 at dose1 to 0.023686297 at dose2: 1.7595% better, or 3.6048% versus the LS parent. This is the same image as the prior all-clock control, not a newly fitted program. Every candidate eventually disagrees with the native source trajectory at the same first tokens. Lower distribution error therefore does not extend these exact greedy-matching prefixes.

The [frozen executed protocol](EXECUTED_PROTOCOL.md), [executed C source](runs/job-309bf320370d/payload/driver.c), [report](runs/job-309bf320370d/report.json), [independent verification](runs/job-309bf320370d/verification.json), [saved-only study](runs/job-309bf320370d/study.json) and [Astra review](REVIEW.md) define this result. The working PLAN.md, driver.c, worker.py and fresh.json have subsequent development changes; they are not the executed sources of this run. Its source manifest and payload remain immutable.

## What ran

Same Qwen2.5-1.5B-Instruct, C FP32/pedantic cuBLAS, layer17, B128, six previously frozen images with doses0/−1/0.25/0.5/1/2, compiled K/native V, native O/MLP and 16recent+16counted prototypes per KV group. Native extends the same three original prompts to up to32 generated tokens, ingests one shared followup and generates up to32 more. Native answer lengths are32/32,20/17 and32/32; four of six answers hit the cap. All six images then replay those native IDs with their own selected memory and upper caches. There is no new direction, source context, memory capacity or target fitting.

The selector scores the distributions choosing actual answer tokens: if prefill ends at p and k tokens are generated, rows p−1 through p+k−2 count. This includes the first token and the decision choosing EOS, excluding the unused post-answer distribution. Each of two turns gets half of its conversation's weight, and each conversation gets a third. Selection seals before three new targets are parsed/tokenized.

| Fixed query dose | Declared future KL, nats |
|---:|---:|
|0 / LS|0.024572058|
|−1|0.025084588|
|0.25|0.024453219|
|0.5|0.024336877|
|1 / reader-response choice|0.024110526|
|2 / selected and historical all-clock choice|0.023686297|

## Clock weighting matters even when the selected image stays the same

The extended all-clock diagnostic would choose dose1. The first eight answer decisions prefer0.25; the historical eight-consumed-row mask prefers0.5; remaining first-answer decisions prefer1; post-followup decisions prefer2. The old and new eight-token masks differ by one clock, so their difference is not solely a horizon effect. The declared future objective still chooses2 because its post-followup improvement outweighs first-answer regressions.

| Window | Dose1 KL | Dose2 KL | Dose2 effect |
|---|---:|---:|---|
|First8 decisions|0.001500145|0.001646314|Worse|
|Remaining first-answer decisions|0.006855254|0.006883563|Worse|
|Post-followup decisions|0.042826952|0.041906288|Better|
|Extended all clocks|0.025418293|0.025468949|Worse|

Future means improve for sources0 and2, while source1 worsens0.023566873→0.023682091. Two sources are not complete answers within this budget; the result qualifies the recorded windows, not arbitrary completion. All six images lie along one old direction, with the selected point at its grid edge. Neither an optimal F nor insufficient B128 information follows.

## The forced-history boundary is now visible

All six images have identical first greedy disagreements: source0 at the14th token of the second answer, source1 at its3rd token, source2 at the27th token of the first answer. Indexing in raw artifacts is zero based. Before each first disagreement the candidate state is reachable through its own greedy choices; after consuming the mismatched native token, later source states are forced. A later matching argmax cannot restore an autonomous-prefix claim.

Each image has165 scored answer decisions:96 occur on an own-reachable prefix, including the three choices that first disagree;93 choices match while the prefix still matches;69 decisions occur after a forced mismatch. This does not turn all later rows into invalid evidence: they still measure conditional fidelity on the declared native trajectory. They cannot establish how the program behaves on its unobserved autonomous continuation.

## New targets: same compact answers, mixed fidelity

The selected image is byte-identical to the historical dose2 control. Its logical path aliases that control for all three targets, so12 logical policies require9 physical trajectories. Dose2 and dose1 generate identical complete token histories on every target. Their comparison improves native KL at8/12 saved boundaries and relative logit L2 at11/12. Among8 native-history-matching boundaries, KL improves6 and L2 improves7. Initial KL improves only in math: recall and prose worsen. These are correlated clocks from three conversations, not12 independent samples.

Recall answers are correctly `ceramic`, then `steel`. Initial multiplication is correctly418 under all policies. The followup asks to subtract17: native starts `241, Calculation: 418 - 17 = `; both compact programs start `401` and a relevant explanation. All three math followups hit the16-token cap, so this establishes the compact leading answer, not a completed explanation. It is a post-hoc observed advantage over native on one task. Both fixed controls already produce it; the new temporal selection did not cause it. Native-distribution fidelity and task truth are different objectives.

Prose is coherent but changes wording and example between native and compact. No automatic prose truth or quality improvement is established. Once target histories differ, native-versus-candidate logit differences include the history difference; they are policy outcomes, not isolated same-input errors.

## Computation, integrity and scope

One guarded Beast CUDA lease, one resident model, CPU1/B1:2,961 source advances (423 native+2,538 candidate),905 own-history target advances and32 native sham advances, totaling3,898. Worker time105.219s includes build, source feedback, audits and targets; it is not serving latency or a speedup benchmark. Native shams match logits/KV byte-exactly. Source selection and target receipts bind the image, input manifests and whole invocation.

Verification authenticates89 executed sources and385 raw files, recomputes the frozen analysis, joins105 old source operand arrays and1,158 historical full-logit metric pairs, and checks all2,538 candidate source reads plus600 compact target reads. All24 target compact snapshots reconstruct byte-exactly. Contemporary independent C/Python full-vocabulary metrics cover every candidate source clock;84 full vectors remain for durable replay (72 candidate/native comparisons). Full temporary streams were removed as declared, so hashes and scalar rows do not substitute for durable full-vector replay at other clocks.

All six frozen images retain their hashes. Astra independently checks the selection,197 target generated-token decisions and72 durable KL comparisons, with maximum C/independent KL discrepancy3.06e−13. The evidence run/verify returns0 over3,455 observations with target-stream-only coverage; it does not certify task truth or source autonomy. Mechanics/instrument complete; trends heterogeneous; terminal certification not requested.

Serving remains819,200 selected-QKV MACs/update versus3,145,728 native, and98,464 selected logical memory bytes. Whole-token nominal linear savings remain0.1507%. Full native lower receiver production, other27layers, original weights, other caches and vocabulary head remain paid. No serving default, full-model emulation, accurate general chat or measured speedup follows.

The next smallest discriminator is [own-history conditional feedback](NEXT.md), before adding memory or another direction. The [connected MLP-reader experiment](../operand_composition/FINDINGS.md) separately proves an actual cheap-writer/cheap-reader handoff, with full memory and mixed accuracy. Its reader/producer conditions are additional applicability fields, not evidence that this bounded-memory query experiment closes the whole model.
