# Native-value improvement survives the bounded reader

The native-V/compiled-QK policy improves first and followup-prefill full-logit fidelity under both full and compact memory on all three development conversations. In the compact reader, first-readout differences fall from14.47/16.27/6.98% to12.08/12.71/6.29%. Every one of the eight boundaries where all six policies share consumed IDs improves under native V in both readers. This is useful gain transfer, with later countertrends and no task-truth, universal-chat or serving-speed claim.

Completed Beast job: **`job-b94ac2f8a6c9`**. [Report](runs/job-b94ac2f8a6c9/report.json), [strict verification](runs/job-b94ac2f8a6c9/verification.json), [historical joins/work account](runs/job-b94ac2f8a6c9/study.json), [Astra review](REVIEW.md), [frozen plan](runs/job-b94ac2f8a6c9/payload/PLAN.md). Report SHA256: `2ad6b50ea1a88fdd73c2d99978bf2ccce826f13c1eb042362e5cfbe2acca17a3`.

## What actually ran

One original Qwen2.5-1.5B-Instruct model, C FP32 CUDA/cuBLAS profile, selected attention17, capacity512, batch1/CPU1. The authenticated source128 basis/images are reused unchanged from verified `job-27a597ebd5f8`. No new source acquisition, target fit, calibration, image compilation or native target-query refresh occurs.

NN (native QKV), CC (compiled QKV) and CN (compiled QK/native V) each run with native full memory and the unchanged16recent/16countedprototype reader. Eighteen paths advance1,799 consumed tokens; the three new CN/compact paths contribute301 of those. All15 candidates finish and seal227 source/candidate arrays before the three native references. Every policy forms its own queries, writes its own memory, consumes its own generated tokens including EOS and ingests the next user turn. The native MLP and suffix recompute on actual receivers.

The fifteen historical paths match across **213 arrays**, including the nine full-probability captures, and all four frozen bank arrays match. Complete turn records, cache-bound hashes and histories also agree. These are execution controls, not fifteen new generalization examples. Only the three CN/compact trajectories add a new policy combination.

## Same reference, signed gains

All relative differences below use the same NN/full logit vector as reference. They are vector discrepancies, not error probabilities. Full and compact native-V gains subtract squared errors against that same reference; no reference switches to NN/compact.

| Case and clock | CC/full | CN/full | CC/compact | CN/compact | All six consumed histories equal? |
|---|---:|---:|---:|---:|---|
| Recall first prefill |13.843%|11.761%|14.471%|12.083%|yes|
| Recall first answer consumed |10.595%|9.866%|11.511%|10.900%|yes|
| Recall followup prefill |16.324%|11.975%|17.900%|13.373%|yes|
| Recall followup answer consumed |12.777%|11.263%|14.848%|13.835%|yes|
| Math first prefill |15.296%|11.562%|16.271%|12.713%|yes|
| Math first answer consumed |8.700%|7.990%|8.760%|8.250%|yes|
| Math followup prefill |26.006%|21.359%|27.249%|24.319%|yes|
| Math followup answer consumed |64.687%|18.861%|65.014%|19.360%|no|
| Prose first prefill |7.087%|6.432%|6.980%|6.292%|yes|
| Prose first answer consumed |8.082%|9.685%|8.861%|9.691%|no|
| Prose followup prefill |35.309%|29.819%|35.707%|30.267%|no|
| Prose followup answer consumed |7.907%|8.147%|10.487%|10.029%|no|

Initial normalized squared-error gains are0.005330/0.010029/0.000886 under full memory and0.006342/0.010315/0.000913 under compact memory. Compression therefore does not erase the observed initial native-V benefit. It does not uniformly preserve its magnitude: at the same-ID math followup, the gain drops from0.022008 full to0.015106 compact. The prose first-answer boundary is a countertrend in both readers. After histories differ, numbers compare complete operational policies and different inputs; they are not common-parent local effects. Even identical text does not make the upper caches identical.

NN/compact is closer than CN/compact at every listed clock: initial differences 2.035/1.705/0.800%, followup-prefill 7.632/13.823/14.600%. This panel's map/receiver-response approximation remains a larger practical limitation than adding native V can solve. That is a measured ranking of these complete policies, not an additive causal allocation of error or a universal compiler result.

![Native-value gain and memory interaction](runs/job-b94ac2f8a6c9/query-memory-transfer.png)

## The memory-by-value interaction is accounted for

The exact common consumed-ID prefixes of CC/full, CC/compact, CN/full and CN/compact contain94/91/58 positions. Their lower normalized receivers, Q/K, full probability pairs, V pairs and compact key/partition states match exactly throughout. These are correlated clocks in three known conversations. The conditional attention identity is

\[
I=Y_{CC,C}-Y_{CC,F}-Y_{CN,C}+Y_{CN,F}
=(\widetilde P-P)(V_C-V_N).
\]

P is the actual native FP32 probability array; Ptilde lifts the compact nodes' weights across their original member rows. Membership reconstruction uses post-seal diagnostic history only. Neither retired dense rows nor membership lists enter the serving reader. The mathematical derivation and explicit four finite mixing remainders are in [MATH.md](MATH.md).

The initial-prompt interaction norms relative to CN/full attention are2.236/1.438/0.937%; across maximal common prefixes they are7.396/6.874/2.038%. The norm alone is not a signed fidelity benefit. The first32 rows have no multiple-member prototypes: interaction there is only6.90–7.06×10^-8, at the numerical-profile scale. After actual merging starts, it becomes consequential. This distinguishes profile disagreement from aggregation effects in these traces.

The raw formula discrepancy is approximately7–8×10^-8 of CN/full attention. Retaining all four accumulation/mixing/cast remainders closes each common-prefix aggregate to2.26–2.71×10^-16; maximum per-row expansion remainder is6.31×10^-16. These arithmetic accounts establish the measured selected-read identity. They do not predict the nonlinear suffix from I or prove that repairing a covariance term improves later conversation.

## Outputs and limits

All six policies answer blue then green. Deletion in the preceding experiment also preserves those colors, so this panel does not establish semantic necessity of the selected component. All compute14×19 as266 and then give286 for266+13, whose correct answer is279. Arithmetic followup explanations are truncated at16tokens. The native error remains part of the result.

CN/compact produces coherent own prose: “Labeling a notebook helps to organize information and keep track of important notes, tasks, and ideas.” Its later example is “Important: Meeting with John at 3 PM on 10/15/2023”. The corresponding full-memory CN path emits “tasks, or ideas” and a different label example. This exercises new own tokens and memory consumption; three short known conversations do not establish accurate unrestricted chat.

The newer [MLP restoration result](../../../obsidian/blog/2026-11-06-152710-restoring-the-operands-exposed-the-memory-gap.md) shows exact current writes can retain15–41% full-logit discrepancy from candidate-carried upper memory. Our experiment starts the selected policy from token0 and therefore includes its historical effects, but does not isolate or replace every upper cache. Both studies favor evaluating the receiver, actual write, carried memory and later reader together.

## Work, memory and verification

CN uses819,200 selected QKV MAC/update versus3,145,728 native:73.9583% less locally, but only0.150724% of full-token learned-linear work. It pays one fresh basis projection, two compiled maps and one native V map at each selected update. All other27layers, O/MLPs, full vocabulary readout and original model weights remain. The whole invocation counts361,905 learned-linear calls and2,823,271,415,808 nominal MAC, including32 native parity advances. Diagnostic and historical acquisition costs are not eliminated by source reuse.

Bounded memory occupies98,464 logical mutable bytes; the global selected1,048,576-byte arena still exists, remains zero and is unread by the compact reader. Native valid selectedKV costs2,048 bytes/token. At most32nodes/head are read, with4,096 routing distance products on later merges and14,336 serving CPU/GPU bytes transferred per update. Full diagnostic history and packed probabilities are separately paid. No serving allocation, checkpoint lifecycle or latency promotion occurs.

Driver 28.66s, worker 38.03s, mrun-reported elapsed 48.44s and scheduler finished-minus-started 48.30s. The final process result records 4,040.3MB peak RSS and 6,406MB peak VRAM; live telemetry alone had sampled a lower RSS peak. Fixed-order instrumented paths do not establish serving speedup. The declared128MiB archive accommodates116,975,016B geometric worst case; authenticated raw files total100,826,415B, plus report/payload transport. Strict verification authenticates69 sources and281 raw files, the scheduler log and whole-invocation Saturn receipt, and recomputes analysis without model evaluation. All36 compact snapshots and every compact read replay byte-exactly. Six model-free math tests and ASan/UBSan C memory mechanics pass. The isolated evidence sweep runs/verifies2,114 observations cleanly; stream coverage is not terminal certification.

## Design implication

A saved-only path decomposition makes this priority more specific. At the initial logit boundary, `CN/full−NN/full` has relative norm 11.76/11.56/6.43%; the increment `CN/compact−CN/full` has norm 1.50/2.27/0.60%. These vector terms sum exactly along this chosen operational path, with signed cross terms retained in [study.json](runs/job-b94ac2f8a6c9/study.json). They are not independent causal percentages: the logit increments include each policy's historical upper caches.

Memory is still substantial locally. Initial selected-attention formation errors are 35.88/27.32/16.85%; compact-minus-full increments are 15.69/13.94/14.33%, with near-orthogonal cross terms. Prose's local attention error increases 16.85%→22.16% under compression while initial full-logit error decreases 6.432%→6.292%. Consumer geometry and error direction matter more than assuming either local norm or nonzero interaction identifies the right repair.

Keep CN/compact as a declared experimental policy and retain CC/compact and NN/compact controls. Do not add covariance state, a per-token router or another rank merely because an interaction is nonzero. Next discriminate whether a frozen receiver/consumer-aware source program can improve Q/K response without target refresh; evaluate it from token0 against the same paid controls and through its own memory. Dense receiver formation and safe compact checkpoint/restore remain independent implementation gaps before this becomes an efficient stacked or whole-model machine.

## Saved-only followup: query versus key formation

[score_audit.py](score_audit.py) expands recorded post-RoPE initial-prompt operands without another model call. [score-audit.json](runs/job-b94ac2f8a6c9/score-audit.json) binds its helper, raw-file hashes and primary report. The exact real-arithmetic score change is

\[
\Delta s=[\Delta q K_N^T+q_N\Delta K^T+\Delta q\Delta K^T]/\sqrt{128}.
\]

After removing each head/clock's mean over causal positions, the query/key/interaction term norms relative to centered native scores are 22.28/14.69/8.26% for recall, 26.37/16.63/10.59% for math and 15.09/10.09/4.58% for prose. Signed cross terms and raw-score results remain in the artifact. The centered expansion closes to 4.89–6.20e−16. Row centering respects softmax's constant-score gauge; it does not make these norms independent causal shares.

Query formation is the stronger aggregate next hypothesis. In a separate FP64 recomputation using recorded native values, restoring Q while retaining compiled K gives attention errors 25.39/21.43/10.91%, versus 29.19/25.08/13.42% when restoring K alone. Both-compiled Q/K gives 35.88/27.32/16.85%. The FP64 ordinary/both endpoints differ from their saved CUDA outputs by 1.73–1.96e−7 relative; this is a qualified arithmetic diagnostic, not an executed native mixed policy, consumer-logit repair or autonomous continuation.

The per-row countertrend matters: the query term exceeds the key term at only 364/672, 235/528 and 212/516 head/clocks. Larger query errors in fewer rows drive some aggregate dominance. This motivates a query-response-aware cheap compiler objective rather than a universal query bottleneck claim. Defaulting to native Q would also sacrifice most selected-map savings: native Q/compiled K/native V costs 2,981,888 MAC per update, only 5.21% below native QKV, before scalar/transfer work. Any new correction must earn its own repeated-state and later-consumer result.
