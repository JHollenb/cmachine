# Formed context and the question episode make distinct coupled contributions

2026-11-06. The two missing hybrid policies now separate the correction episode from the later question/answer episode. A dense question episode on PROJECT-formed context still differs from native, despite identical local MLP16 computation. Applying PROJECT only during the later question episode produces a larger full-logit difference by the measured norm. Their joint effect is conditional, with interaction opposing both individual effects in the signed energy account. Every policy still answers the short color questions correctly.

The [prospective plan](PLAN.md), [original report](runs/job-c593b2c6e4a6/report.json), [verification](runs/job-c593b2c6e4a6/verification.json), and [Astra review](REVIEW.md) retain the bounded scope: two new hybrid policies joined to two authenticated historical endpoints, one supplied-fact correction, two independent readers. This is not four new independent trials or compact sufficient chat state.

## The two episode policies

The original Qwen2.5-1.5B-Instruct executes in the same historical C FP32/cuBLAS profile at layer16, with the same276536-byte frozen B3. Source/template/tokenizer/runtime/weight identities remain fixed. The correction changes Niko's badge to green while keeping the other person's badge unchanged, and requests OK. Later independent questions ask for Niko and Mira, with8-token budgets.

| World | Correction and own acknowledgment | Each later question and own answer | Evidence |
|---|---|---|---|
| NN | Native | Native | Historical whole-turn endpoint |
| PN | PROJECT | Native | New hybrid |
| NP | Native | PROJECT | New hybrid |
| PP | PROJECT | PROJECT | Historical whole-turn endpoint |

Both new policies generate OK+EOS, then independently green+EOS and red+EOS. All four worlds have identical actual consumed histories at both measured boundaries. Each question episode includes its full template, question ingestion, newly written KV and generation; it is not one isolated attention reader. Every generated token is consumed, and Mira's branch never receives the Niko answer.

The causal clocks agree in all worlds: source cursor59, post-correction93, first-answer readout117 and after consumed EOS119. Original full residuals, all-layer cache and dense weights remain.

## Historical joins authenticate the factorial

Historical NN/PP come from [job-a071c7a43a38](../whole_turn/runs/job-a071c7a43a38/report.json), report SHA `cda1042c01e53762643245f13573431e2f17983e46934a69cb6bf7a8e88ed444`. The new request binds its projection and raw endpoint hashes. The C driver never reads those endpoint vectors; analysis loads them only after new generation.

Each new correction episode is an unchanged replay of one prior episode. Before each reader runs, position, full actual token-history digest and complete valid-KV hash match the historical correction. Offline verification also matches484captured correction files byte-for-byte and both prefill/final readout metadata. A failed replay join would invalidate the historical comparison rather than count as a scientific effect.

The source and both correction parents are therefore qualified across runs without repeating NN/PP reader executions. This saves experiment work while keeping parent identity explicit. It is not a serving-performance result.

## Formation and question effects remain after separation

Full-logit relative difference is \(\|Y-Y_{NN}\|_2/\|Y_{NN}\|_2\). The first boundary selects the initial answer token; the final boundary is after consumption of EOS and describes a subsequent continuation state.

| Reader / boundary | PROJECT formation, dense question PN | Dense formation, PROJECT question NP | PROJECT both PP |
|---|---:|---:|---:|
| Niko / pre-answer |9.87%|12.93%|17.07%|
| Mira / pre-answer |9.86%|16.88%|20.49%|
| Niko / post-EOS |11.03%|22.14%|22.92%|
| Mira / post-EOS |10.50%|23.12%|23.12%|

Formation alone leaves a measurable discrepancy under subsequent dense computation. Switching equations does not restore native same-text state. Question approximation alone has the larger isolated norm at all four boundaries in this specimen, while formation remains consequential. The earlier zero policy also preserves all three short answers, so no semantic-necessity claim follows from these outputs.

The nearly equal final Mira errors conceal a substantial conditional effect. NP's exact value is23.1168%, versus PP23.1228%; yet \(\|Y_{PP}-Y_{NP}\|/\|Y_{NN}\|=8.55\%\). Similar distances from a reference do not imply similar vectors or dispensable formed context.

## The local computation stays fixed while later readouts move

For PN versus NN, the question policy is dense in both. For NP versus PP, it is PROJECT in both. Across all104matching-history question transitions, captured layer16 entry, raw receiver, normalized MLP input, MLP return, layer exit and newly formed K/V16–17 are byte-identical to the corresponding historical policy.

Thus formation's effect is not a change in the current MLP16 input or write in these comparisons. The higher-layer history formed during correction differs, and later computation consumes that history. Identical current K/V17 is compatible with this: K/V17 forms before layer17 attention reads old memory. We have not captured the exact first changed attention output or current K/V18 here.

This supplies a useful dependency distinction for stacked state: a shared lower computation can coexist with branch-specific higher memory on the same text. It does not make either state fragment compact or provide a within-chat memoization speedup.

## Expand the interaction before assigning importance

For each common boundary, define

\[
A=Y_{PN}-Y_{NN},\qquad B=Y_{NP}-Y_{NN},
\]

\[
I=Y_{PP}-Y_{PN}-Y_{NP}+Y_{NN},\qquad
D=Y_{PP}-Y_{NN}=A+B+I.
\]

A and B are complete correction-episode and question-episode effects. Their conditional versions are \(A+I=Y_{PP}-Y_{NP}\) and \(B+I=Y_{PP}-Y_{PN}\).

| Reader / boundary | Interaction norm / joint norm | Interaction signed projection onto joint |
|---|---:|---:|
| Niko / pre-answer |28.51%|−1.65%|
| Niko / post-EOS |35.36%|−14.47%|
| Mira / pre-answer |23.05%|+3.15%|
| Mira / post-EOS |36.98%|−13.08%|

The interaction norm is substantial, but it is not an additional same-direction error budget. Across all four boundaries, A and B have a positive inner product, while I has a negative inner product with each. Interaction partly cancels the individual episode effects. Its projection onto the joint still changes sign across cases because its own power also contributes.

The full identity is

\[
\|D\|^2=\|A\|^2+\|B\|^2+\|I\|^2
+2\langle A,B\rangle+2\langle A,I\rangle+2\langle B,I\rangle.
\]

At the Niko pre-answer boundary, dividing by joint power gives

\[
1=0.33460+0.57389+0.08128+0.20582-0.04693-0.14867.
\]

Signed cross terms are needed to interpret the norms. These projections are geometric measurements, not semantic causal percentages. The factorial is a measured post-target decomposition of frozen prospective policies; it is not a pretarget forecast of the two hybrid results.

## Numerical audit and custody

Pointwise closure is exactly zero in the retained FP64 vector account. Subtracting large independently reduced powers leaves producer remainders up to1.82×10⁻⁹; Mac recomputation differs in the last bits. Astra identified the inherited fixed1e-14 verifier risk before the first offline verification. The [declared numerical policy](NUMERICAL_AUDIT_POLICY.md) bounds only this derived remainder using FP64 gamma_(2n+16), absolute-product/Cauchy–Schwarz scale and the measured pointwise defect. Producer/local values and bounds are retained in [numerical-audit.json](runs/job-c593b2c6e4a6/numerical-audit.json), with both verifier sources and policy snapshotted. The original report is unchanged, and every constituent power/cross term, raw byte and other arithmetic comparison retains the strict contract. No failed model run or model rerun is involved.

Beast job `job-c593b2c6e4a6` takes23.40scheduler seconds, worker20.13seconds and Cdriver10.67seconds; peak RSS4028.3MB/VRAM6378MB. One resident CUDA FP32 model, batch1,512capacity, one CPU thread, no retries. There are59source advances plus172observed hybrid advances,231full token computations total. At MLP16 those172transitions execute172G/U and172down calls, zero compilation calls;86PROJECT transitions skip172G/U calls. Original weights and cache remain, and capture costs preclude a serving-speed claim.

The exact bound is214observed transitions and27,481,952raw/payload bytes, below the48MiB archive cap. Verification authenticates184source/payload files and1237raw files totaling14,834,566bytes, the scheduler log/report joins, source helper extraction, predecessor projection and whole-invocation Saturn receipt. Twelve model-free tests and Csyntax checking pass. The callback-free historical verifier regression reproduces its original verification hash exactly. Astra independently audits the source/raw hashes, replay joins, local controls and factorials.

The standard [evidence run](index/job-c593b2c6e4a6/evidence-run.json) and [verification](index/job-c593b2c6e4a6/evidence-verify.json) cover114source-bound rows with2applicable predicates, no fires/claims/addresses. This is stream coverage alongside the custom numerical/custody audit, not semantic certification. The installed development chat and shared C runtime are unchanged.

## Consequence for the next causal cut

The episode split localizes two contributors and their conditional combination. A compact state must preserve consequential formed memory and the operations that read it; restoring a dense operation alone does not recreate native context. The measured question-episode norm is larger here, while the persistent formation effect and signed interactions remain material.

The next useful microscope cut is the first higher-layer read of the changed correction history. At layer17, current Q and newly formed question KV depend on question policy beta, while retained correction-band KV depends on formation policy alpha. Earlier source memory remains fixed. The existing local controls and architecture motivate the real-arithmetic score prediction, per correctly mapped grouped-query head:

\[
\Delta_\alpha\Delta_\beta S
=\frac{(q_P-q_N)(K_C^P-K_C^N)^\top}{\sqrt{128}}
\]

on correction-band columns C, with zero interaction predicted off that band by paired operand equality. Use actual biased post-RoPE Qwen2.5 operands and keep native FP32 dot-product remainders separate. Softmax's shared denominator and value mixing can spread output interaction across the complete attention row; score support is not output support. A small read-only trace of Q, permitted old/new K/V, attention output and receiving write can test this without historical KV mutation. This is a proposed mechanism-localization cut, not another run executed here. Arbitrary chat sufficiency, semantic necessity and total-state/resource compression remain open.
