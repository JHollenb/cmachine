# All caches can own compressed writes; broad fidelity remains imperfect

All 28 Qwen2.5-1.5B-Instruct cache layers now execute signed-byte append from token zero, with no dense cache slots. Each new actual candidate K/V row is encoded and consumed at its current read. The source conversation, correction, acknowledgement and three later turns all form under that policy. This removes both the remaining 26 dense caches and the previous experiment's native-prefix acquisition boundary.

The byte session performs 5,096 main layer appends across 182 consumed tokens. Native and exact all-layer FP32 callbacks also advance 182 tokens, giving 546 main transitions. All sessions answer `red`, `OK`, `Niko's`, `Mira's badge is red, and Niko's badge is green.`, and `1`, each with EOS. These supplied facts were also insensitive to earlier single-layer deletion controls; this result is not broad chat or semantic-necessity evidence.

State allocation falls from 30,096,896 to 8,830,468 GPU bytes: **21,266,428 bytes /70.66% saved**. Original weights remain, so the saving is only 0.343% of tracked model-plus-state allocation. Full retention still grows linearly. Exact all-layer callbacks match native every captured own K/V/query, sparse port, full readout and logical cache; native source state plus eight historical correction/later boundaries reproduce the predecessor.

The [frozen plan](PLAN.md), [implementation](README.md), [original failed report](runs/job-bbd58171a613/report.json), [saved-only corrected analysis](runs/job-bbd58171a613/reanalysis.json), [qualified verification](runs/job-bbd58171a613/verification.json) and [Astra audit](REVIEW.md) distinguish completed C execution from the post-processing failure and its repair.

## Composition and valid-state rollback

Every selected layer has slot−1, and native K/V pointers are NULL. The allocation transform explicitly skips cudaMalloc(0). Exact callbacks own ordinary FP32 cache arrays; byte callbacks own 28 code/scale arenas, one 524,288-byte decode plane and a 4-byte status flag. Each preceding session is destroyed before the next; model weights are shared within the single C process.

All source/user/template/own-generated/EOS tokens pass through the policy. The only fixed-source information is the model and user token stream. Candidate replies are their own greedy outputs. All own K/V rows are recorded before encoding; queries at 0/17/18/27 are saved, with sparse native consumer ports and full logits at all ten pre-answer/consumed-EOS boundaries. Five snapshots contain all valid byte caches. Independent reconstruction matches 20,384 unique encoded rows and 68,096 snapshot row instances exactly.

Sibling overwrite, nested rollback and original two-token replay reproduce full logits and valid logical cache hashes. At the byte session's correction boundary, an injected layer 27 error leaves 27 cursors at 94 and the last at 93. Token abort restores all 28 to 93 and the exact valid parent before replay. No unreachable suffix is interpreted as valid memory.

Q0 stays exact under identical consumed histories. Every captured higher query changes from native: mean relative differences are 5.4015% at 17, 4.4009% at 18 and 7.0220% at 27; maxima are 18.589%, 16.459% and 36.393%. These include altered earlier layers and their accumulated histories. The intervention begins earlier and covers more memory than the previous two-layer native-prefix experiment, so this is not a controlled layer-count dose curve.

## Larger composition has visible numerical cost

| Turn | Before-answer full-logit difference | After consumed EOS |
|---|---: |---: |
| Source color |3.7322% |8.4825% |
| Correction |6.9864% |9.8392% |
| Recall |5.9600% |9.2860% |
| Summary |6.5626% |7.8351% |
| Count |7.2932% |7.4048% |

All complete histories match native here. These differences measure numerical fidelity, not answer correctness. Mean normalized attention-context error against the byte candidate's own original-row history is 15.8544% at 0, 1.9448% at 17, 2.4708% at 18 and 1.0556% at 27. The local codec and downstream state discrepancies remain separate observations.

Layer 0 exposes why the numerator/denominator metric must be interpreted at its consumer. At position 7/head 3, quantization changes log Z by +35.8289, giving Z'/Z≈3.633×10^15. Native scores are roughly 19,402–19,575. Both native and encoded attention nevertheless put probability essentially 1 on row 7: probability L1 difference is 1.24×10^−27, while context error is 1.714% from decoded values. A common score shift cancels in full softmax. Such unnormalized mass mattered when combining a span with a dense complement, but cannot alone rank complete-bank routing.

There are also actual routing failures. At position 30/head 3, native attention selects row 30 with probability approximately 1; byte attention instead selects row 5. Probability L1 difference is 2 and context error 125.36%. Across 2,184 correlated layer0 head reads, mean probability L1 difference is 0.330219. Answer agreement conceals this internal mismatch.

Astra's independent saved-only diagnostic splits keys into eight blocks of 16 coordinates with separate scales, keeping values row-quantized. On these same saved operands it lowers mean context error 15.8544%→8.0688%, maximum 125.36%→71.81%, and mean probability L1 difference 0.330219→0.182073; roughly 69% of reads improve. It still permits probability L1=2. The [reproducible audit](routing_diagnostic.py) preserves [every read](runs/job-bbd58171a613/routing-diagnostic.json). Strict decrease counts are 1,516 under Astra's scalar summation and 1,517 under the vectorized audit; floating-point near ties affect that count while the reported aggregate effects agree. This is a posthoc local prediction, not an executed new trajectory or held-out result. The prospective layer-0-only storage increment is 56 bytes/token, 28,672 bytes at capacity 512. It remains separate from the weight-storage experiment.

## Actual resources and unchanged knowledge

Native K/V capacity uses 29,360,128 bytes. All-byte arenas use 7,569,408 bytes, with 524,288 decode bytes and 4 status bytes. Session scratch outside K/V is 736,768 bytes, producing 8,830,468 total state bytes. Original model/helper storage remains 6,174,873,620 bytes; total live model-plus-state falls from 6,204,970,516 to 6,183,704,088 bytes. These are successful cudaMalloc request extents, excluding CUDA/cuBLAS internals and host metadata.

At capacity P, current all-byte state allocation is

\[
B_{\mathrm{state}}(P)=(28\cdot528+1024+48)P+712196.
\]

The 48P term is score scratch; 712,196 contains other scratch and the status flag. Thus reducing stored row precision does not remove linear growth. A bounded-retention replacement must bound decode/score scratch and rollback storage too, not just the visible row bank.

The lease takes 33.00 scheduler seconds, 26.77 worker seconds and 16.94 C seconds, with 4,205.4 MB RSS/6,378 MB peak VRAM. The original scheduler status is **failed**, because Python analysis requests an obsolete historical field name after all C sessions finish. Captured trajectories remain evidence; no model rerun is used to repair interpretation.

## Immutable repair and evidence scope

The repair authenticates scheduler/log/report/source/archive membership, every raw byte and the predecessor. It requires the new analysis source to equal the executed source with exactly one field-lookup substitution: accept the historical `*_logical_cache_sha256` field where that predecessor uses it, retaining the older `*_kv_sha256` spelling for older records. Equations, raw observations and tolerances do not change. Native controls and all scientific analyses then pass from saved operands. The original report/source/error remain untouched, and the corrected result is a separate derivative.

Verification binds 79 sources and 774 raw files totaling 78,745,397 bytes, with a public whole-invocation Saturn receipt. Report SHA 256 is `368e835512f7f04cc61ad00cdb1580774d1764aa950ec28840d3e868c4791ecb`; reanalysis SHA 256 is `11b5eaf4d5d7b0c36b93e4d8d3a49958e758e3c53afb5457cfac1a8c110833dc`. The [evidence sweep](index/job-bbd58171a613/evidence-verify.json) completes over 43 source-bound observations and seven applicable predicates, no fires/claims/addresses. Astra independently audits custody, exact controls, encoded rows, query/consumer effects and the source repair.

The result is all-layer approximate memory construction and real cache allocation reduction. Whole learned parameter information, dense forward work, retention policy and broader capability preservation remain separate targets. [Weight storage](../weight_tiles/PLAN.md) and [bounded-retention closure](RETENTION.md) address the next two contracts. Terminal certification is not requested.
