# Paired atoms survive recurring compressed-span reads

A row-quantized key/value representation substantially reduces the [centroid program's response errors](../span_response/FINDINGS.md) while operating at similar stored size. The **17,984-byte signed-byte bank** keeps all 34 paired atoms, rather than averaging groups into eight representatives. Across three linked turns, it replaces the selected span at **89 successive layer17 reads**, including user ingestion and its own generated tokens. The original span's K/V17 slots remain poisoned with NaNs throughout candidate execution; native fallback at that layer is refused.

Mean per-head numerator error is **2.086%**, versus **48.438%** for eight centroids. All six measured byte-bank full-logit differences are below **0.1%**. These are numerical-fidelity observations from one supplied-fact conversation. All answers match even when the selected span is deleted, and full original cache allocation and model weights remain.

The [prospective plan](PLAN.md), [mathematical account](MATH.md), [implementation](README.md), [report](runs/job-d12ce01f244f/report.json), [qualified verification](runs/job-d12ce01f244f/verification.json) and [Astra review](REVIEW.md) retain this boundary. This is standard compressed K/V used as a mechanistic baseline, not a recovered semantic law or constant-size sufficient chat state.

## Representation and actual recurring state

The pinned specimen remains native Qwen2.5-1.5B-Instruct C FP32/cuBLAS, layer17, 12 query heads, two KV heads and width128. Native source and correction/OK/EOS produce the same authenticated cursor93 state as the previous experiment. The memory span C=[59,93) has 34 post-RoPE key/value rows per KV head.

Before opening any later question file, [the C constructor](byte_bank.h) independently encodes each key and value row with FP32 scale max(abs(row))/127 and signed codes in [−127,127]. Division is FP32, nearest-integer ties round away from zero, and zero rows use scale1. Decoding is the actual FP32 code-times-scale multiplication consumed by the candidate. The program contains a 32-byte header, 136 scales (544 bytes) and 17,408 code bytes: **17,984 serialized bytes / 17,952 resident GPU bytes**. Exact34 and Ward-eight float programs freeze at the same source boundary. No query, rank selection, fitting, training or fallback enters construction.

The native chain runs first. After its rollback, the driver requires a completed-token boundary, depth0, cursor93 and no legacy checkpoint, poisons all 69,632 selected K/V17 bytes, increments the epoch and creates a new owned root. A CUDA guard rejects both unarmed and explicitly native-mode attention on that layer's key base. Each candidate child preserves this root. The poisoned prefix, frozen bank and read policy together define the candidate state; the original raw cache alone is no longer its execution contract.

Each policy advances three linked user turns:

1. “Whose badge became green? Reply with just the name.”
2. “In one sentence, summarize the current badge assignments for both people.”
3. “How many people currently have a green badge? Reply with only the integer.”

Budgets are8/32/8. Every appended template/user token and own generated token, including EOS, uses the selected layer17 read. The byte kernel decodes directly into the counted packed float workspace; no full decoded bank remains resident. Original source positions[0,59), all new positions[93,length), other27layers and all dense model projections remain native and paid. Newly written memory is not compressed in this experiment.

The five policies each execute89selected reads, giving445read applications and538total full token transitions including the shared59source and34correction transitions. Raw poisoned-cache hashes remain distinct from native. A separately named complement-cache digest zero-masks only the deliberately poisoned C17 slots; exact34 reproduces this complement and every captured native query/port/readout byte. That masked comparison is never labeled raw full-cache equality.

## Response quality and native consequences

Each response row below summarizes1,068correlated head reads:12heads at89token positions within one conversation. The same-position queries also match across policies in this run. These are not1,068independent examples or a general chat benchmark.

| Program | Serialized bytes | Mean predicted/exact Z | Mean relative N error | Mean relative conditional span-mean error |
|---|---:|---:|---:|---:|
| Exact34 | 69,936 | 1 | 0 | 0 |
| Eight centroids | 16,480 | 0.617861 | 48.438% | 35.832% |
| Paired signed-byte rows | **17,984** | **0.998414** | **2.086%** | **1.333%** |
| Delete C | — | 0 | 100% | 100% |

The byte program uses9.13%more serialized bytes than eight centroids and is3.872×smaller than the native69,632-byte span representation. The mean numerator error is about23.2×smaller than the centroid error on these queries. Byte Z ratios range0.887345–1.055991; its worst numerator error is12.820% and worst conditional-mean error5.418%. Retain that spread: near-unity mean partition ratio is not a uniform error guarantee.

Full-vocabulary logit differences from native are relative L2:

| Linked turn / boundary | Byte rows | Eight centroids | Delete C |
|---|---:|---:|---:|
| Recall / before answer | **0.09322%** | 4.59927% | 4.33573% |
| Recall / consumed EOS | **0.08553%** | 1.17979% | 2.36622% |
| Summary / before answer | **0.03333%** | 0.60395% | 1.21131% |
| Summary / consumed EOS | **0.06750%** | 0.94315% | 1.37525% |
| Count / before answer | **0.07344%** | 2.15325% | 2.61534% |
| Count / consumed EOS | **0.06119%** | 1.05335% | 1.27357% |

All own histories and clocks match native here, qualifying these as same-history state-fidelity comparisons rather than divergent-text comparisons. They include accumulated upper-layer effects from earlier candidate reads. In particular, eight centroids are slightly worse than deletion at the first recall logits despite a better local attention context. Local improvement does not imply monotonic downstream improvement.

Every policy answers `Niko's`, `Mira's badge is red, and Niko's badge is green.`, and `1`, each with EOS. Deletion removes C only from layer17, while information in other layers, source history and prior own answers remains. Matching behavior therefore supplies no semantic necessity claim. The final count is a bounded supplied-fact counting example, not general arithmetic accuracy.

## Why the simpler representation helps

For a cluster, let a=q times the executed scale, k=kbar+u and v=vbar+w. The exact response contains both E[exp(a·u)] and E[exp(a·u)w]. Centroids replace these with1and0. The prior audit showed that lost score dispersion and finite score/value dependence are both substantial, with query-dependent cancellation.

The byte program keeps the paired exponential sum itself:

\[
\widehat Z(a)=\sum_i e^{a^\top\hat k_i},\qquad
\widehat N(a)=\sum_i e^{a^\top\hat k_i}\hat v_i.
\]

It preserves the mixture's form and pairing, not exact native dispersion or dependence. The remaining approximation comes from represented coordinates rather than collapsing several atoms into one. This is a stronger minimal baseline before introducing costly moment tensors. A Gaussian key distribution plus affine E[v|k] would close both terms analytically; that remains an assumption about an approximation family, not an observed law of these finite native rows.

Source-only maximum row reconstruction errors are0.250779/0.249804for keys and0.153385/0.100135for values in the two KV heads. They support prospective query-conditioned bounds without future-query fitting. With eta_K=max_i||khat_i−k_i||, eta_V=max_i||vhat_i−v_i|| and epsilon=||a||eta_K,

\[
e^{-\epsilon}\le\widehat Z/Z\le e^\epsilon,
\qquad
\|\hat\mu-\mu\|\le\eta_V+2V_{\max}\tanh(\epsilon/2).
\]

These are ideal-read bounds using actual FP32-decoded source errors. Native dot products, softmax and accumulation remain separate. Observed epsilon ranges0.3020–0.8969; maximum absolute log Z error is0.11952. The largest mean bound is11.528value units, compared with maximum measured mean error0.37421, so the bound is conservative. It diagnoses support for future queries rather than acting as a quality gate. Arbitrarily unbounded queries do not obtain a fixed uniform bound.

## An architectural limit exposed by recurrence

All445captured Q17 arrays match native exactly. A layer17 intervention cannot change earlier layers' state. With identical emitted token histories, the lower-layer computation generating Q17 and the newly written KV17 stays native. The changed layer17 read does alter current higher-layer computation and its retained history, which then affects later logits. The original first-logit experiment and this recurring experiment therefore probe different accumulated upper-state consequences even though Q17 itself remains fixed on matching token histories.

This is useful causal structure, not proof that the approximation tolerates changed own-query feedback. A changed emitted token would change the later lower-layer input; composing approximations across layers can also change a higher selected layer's live queries. Neither event occurs in the present comparison. A future stacked-layer assay should retain the lower intervention's effect on the higher query instead of borrowing a native query.

## Execution, custody and the numerical-profile incident

Successful CUDA job [`job-d12ce01f244f`](runs/job-d12ce01f244f/job.json) takes23.78scheduler seconds,19.91worker seconds and14.37instrumented C seconds. Construction/export takes about0.00153seconds. Peak RSS is3,876.2MB and VRAM6,378MB. The prototype retains all original cache allocation/model weights,1,048,576GPU bytes of packed scratch and104,304GPU bytes across all three control/candidate banks. Byte attention still processes34span rows and pays their104,448QK/value MACs per read, plus17,408decode multiplications; it reduces stored precision, not row count. There is no measured serving speedup or net allocation reduction.

Verification binds76sources and1,107raw files totaling27,901,765bytes, predecessor identity, scheduler-sealed report/log, bank construction and the public Saturn invocation receipt. Thirteen model-free checks and C syntax checking pass. The first CUDA attempt stopped before candidate execution because the harness incorrectly required phase1 after rollback; the actual runtime restores phase0 and permits both completed-token phases. [Attempt history](ATTEMPTS.md) preserves it as incomplete mechanics evidence.

The first local strict recomputation also refused1,448derived centroid-only fields, with maximum absolute difference7.76e-8. The vectorized diagnostic computed log(count) in FP32; its Linux NumPy and local NumPy profiles differ at that precision. Primary C predictions, native/exact/byte response fields and raw numerical consumers are unaffected. [The original refusal](runs/job-d12ce01f244f/verification-incident.json) remains intact.

Rather than changing a tolerance or running models again, CPU-only [`job-978a534ae1d1`](runs/job-d12ce01f244f/audit-replay-job.json) checks every frozen source/raw hash and reexecutes the exact original analyzer in its original environment. Its canonical analysis bytes match the producer digest exactly, with zero tolerance. This3.26-second replay peaks at85.6MB RSS and zero VRAM. The [verification](runs/job-d12ce01f244f/verification.json) explicitly records same-environment reproduction, not cross-platform exact arithmetic. [The verifier](verify_replay.py) joins both scheduler seals and the unchanged primary artifacts. Astra's independent FP64 summation confirms byte response fields within4.54e-15 and confirms that the centroid's small log-count differences cannot affect the substantive comparison.

The [standard evidence sweep](index/job-d12ce01f244f/evidence-run.json) and [verification](index/job-d12ce01f244f/evidence-verify.json) finish successfully over62source-bound observation rows and six applicable predicates, with no fires, claims or addresses. This is target-stream coverage. Report SHA256 is `f954bc40e87e7315919ee5cff6602ad276df9db08ade5f4175241400969b4ac6`. No terminal certification is requested.

## The next closure is an append operation

The next minimal increment is to encode each actual own-state, post-bias/post-RoPE newly written K/V17 row, consume its decoded representation in the current and subsequent reads, and only then make the original row unavailable to attention. Encoding and cursor advancement must commit together; rollback truncates the encoded cursor alongside token state. Keeping the current read FP32 would be a different hybrid contract and must be named explicitly. Quantization of the candidate's actual row must be measured separately from its accumulated difference from a native trajectory.

At this layer an appended byte row costs528bytes/token versus2,048FP32bytes/token, excluding allocator metadata and paid decode workspace. This gives a clear growing compressed-memory mechanism, with linear context growth. It does not supply constant-size semantic state. Physical cache ownership, stacked layers, longer contexts, changed own queries and broader chat/math/recall consumers remain separate necessary experiments before a claim of compact sufficient chat.
