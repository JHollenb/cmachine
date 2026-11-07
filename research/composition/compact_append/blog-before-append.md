---
title: "2026-11-06 12:55:44 — A frozen MLP program updates a conversation"
timestamp: 2026-11-06T12:55:44-08:00
updated: 2026-11-06T14:56:01-08:00
type: blog
category: research
status: bounded-experiment-with-next-cut
tags: [research, architectural-finding, model-analysis, symbol-table, prediction-arc, time-circuit-architecture, saturn, state-machine, language-models, context, memory, mlp]
source_findings: ../../saturn/cmachine/composition/whole_turn/FINDINGS.md
source_review: ../../saturn/cmachine/composition/whole_turn/REVIEW.md
factorial_findings: ../../saturn/cmachine/composition/formation_read/FINDINGS.md
attention_findings: ../../saturn/cmachine/composition/attention_read/FINDINGS.md
span_findings: ../../saturn/cmachine/composition/span_response/FINDINGS.md
span_review: ../../saturn/cmachine/composition/span_response/REVIEW.md
recurrent_findings: ../../saturn/cmachine/composition/span_recurrent/FINDINGS.md
recurrent_review: ../../saturn/cmachine/composition/span_recurrent/REVIEW.md
---

Our C runtime can now apply a frozen approximation while ingesting a changed user message, retain the resulting context, and answer later questions from its own state. In the tested conversation, Niko's badge changes from blue to green while Mira's remains red. The approximation acknowledges the correction and answers both follow-ups correctly.

Deleting the same MLP write also preserves those answers. That control makes the result more informative: we have extended autonomous execution through context formation, but successful recall alone does not establish that the frozen program recovered the necessary computation.

The numerical measurements reveal a more specific effect. The program improves full-vocabulary logit fidelity over deletion before each answer, then loses that advantage after consuming the end-of-turn token. Factoring those differences shows partial cancellation at every measured boundary. The cancellation is strong enough to help before the answers and too weak afterward.

The [complete findings](../../saturn/cmachine/composition/whole_turn/FINDINGS.md) and [saved-data audit](../../saturn/cmachine/composition/whole_turn/runs/job-a071c7a43a38/additive-audit-v2.json) bind these observations to one supplied-fact conversation and the Qwen2.5-1.5B-Instruct C FP32 implementation. The mathematical target remains the one developed in [[2026-11-06-115551-from-a-c-chat-runtime-to-compact-sufficient-state|compact sufficient chat state]]: retain enough information to advance the conversation and support its future consumers.

## Context formation is part of the computation

The [[2026-11-06-004500-what-the-c-state-machine-learned-about-carry-and-context|earlier C work]] and [generated-token composition experiment](../../saturn/cmachine/composition/FINDINGS.md) replaced a selected MLP operation at the final question token and during the answer. Earlier question tokens entered through dense native computation. A local replacement could therefore generate a plausible continuation without demonstrating that it could form the context it would later read.

This experiment moves the boundary. After an authenticated native source conversation and its own answer, every appended template token, user token and generated token passes through the selected MLP16 policy. There is no new calibration or native-state refresh.

The new user message says:

> Correction: Niko's badge is now green. Keep the other person's badge unchanged. Reply only with OK.

After the branch's own acknowledgment, two independent children ask for Niko's current color and Mira's unchanged color. Both start from the branch's actual post-correction state. Mira's reader never receives the Niko answer as an additional cue.

| Policy during all new ingestion and generation | Acknowledgment | Niko afterward | Mira afterward |
|---|---|---|---|
| Native dense C | OK + EOS | green + EOS | red + EOS |
| Frozen B3 approximation | OK + EOS | green + EOS | red + EOS |
| Zero MLP16 write | OK + EOS | green + EOS | red + EOS |
| Arithmetic correction | OK + EOS | green + EOS | red + EOS |

Each reply contains one text token and EOS. Every emitted token is consumed, so the retained state includes the complete declared continuation. Generated token IDs remain authoritative; decoding them into prose does not rebuild the model's history.

All policies produce identical token histories here. That gives us a clean comparison of different computations and caches on the same text. It also limits the behavioral conclusion: one correction and two short readers are a small development example, and deletion passes it too.

## The approximation factors the affine operands

At the selected MLP, let the actual post-attention receiver be \(r\), with native RMS scale \(\rho(r)\). The original computation is

\[
x=\rho(r)(\gamma\odot r),\qquad
M(r)=D\left[\operatorname{SiLU}(Gx)\odot Ux\right].
\]

The frozen basis gives \(r=Bc+e\). For the gate projection,

\[
Gx=\rho(r)\sum_j c_jG(\gamma\odot b_j)
+\rho(r)G(\gamma\odot e),
\]

and similarly for the up projection. B3 stores the original-weight images of three basis columns and omits the live complement term. It retains the actual receiver for RMS scaling and the residual skip, and still evaluates the nonlinear gate, product and down projection.

The correction arm restores the complement's original G/U projections before the nonlinear consumer. This pays dense work and supplies an arithmetic control. Its same-input write differences stay between \(5.56\times10^{-7}\) and \(7.46\times10^{-7}\), consistent with the changed FP32 ordering. It is not a savings result.

B3's local write errors are much larger: 62.30–118.22%, averaging 101.85% across 86 transitions. At 69 transitions its write is farther from dense than zero is. The model nevertheless produces the same short answers. This is why local numerical fidelity, downstream tolerance and task behavior require separate measurements.

The 276,536-byte B3 package is additional compiled program data. The full original weights and KV cache remain resident; three coordinates do not replace the conversation's state.

## A causal boundary makes the reference trustworthy

Split the persistent computation at MLP16. Let \(P\) contain lower-layer KV and its clock, position and mask metadata; let \(Q\) contain the higher-layer history. For a supplied next token \(u\),

\[
(P',r)=F_{\rm lower}(P,u),\qquad
Q'=F_{\rm upper}(Q,r+M^\pi(r)).
\]

The selected policy changes the MLP after its current K/V has already formed. Under an identical consumed history, that intervention cannot change its earlier dependencies. Generated-token selection can eventually change a later input, so the shared-history condition matters.

After all primary outputs seal, paid dense diagnostic replays consume each candidate's exact token sequence. They never supply state to the candidate. Across all 258 comparisons, captured layer16 entry, raw receiver, normalized input and KV16 are byte-identical. The dense return is therefore a qualified same-input reference for the local MLP comparison.

The distinction is useful for the symbol table. A reusable computation record should bind the live operands, read support, clock and numerical implementation. Matching current text or a small coordinate vector is insufficient when its hidden history differs. The [[2026-11-06-092300-a-symbol-table-that-checks-its-inputs|applicability work]] and [[2026-11-06-093800-predicting-how-an-error-moves-through-the-next-block|next-block error study]] develop the same issue at other boundaries.

## The benefit changes after the answer

We measure full-logit relative error as \(\|Y_\pi-Y_N\|_2/\|Y_N\|_2\). The pre-answer readout selects the first answer token. The later readout follows consumption of EOS and describes a subsequent continuation state.

| Turn | B3 before answer | Zero before answer | B3 after EOS | Zero after EOS |
|---|---:|---:|---:|---:|
| Acknowledgment | 36.92% | 39.65% | 21.05% | 20.19% |
| Niko read | 17.07% | 21.02% | 22.92% | 20.53% |
| Mira read | 20.49% | 25.48% | 23.12% | 20.85% |

B3 is closer at all three pre-answer boundaries and farther away at all three post-EOS boundaries. Those later differences are not incorrect delivered answers. They show that a local advantage does not automatically remain an advantage as the state advances.

Even the choice of observable matters. For Mira, B3's winning-token margin is much closer to native than zero's. For Niko, zero's margin is closer, despite B3's better full-vocabulary distance. These measurements describe different properties of the same consumer.

## Signed geometry accounts for the numerical reversal

Let \(e_Z=Y_Z-Y_N\) be zero's error and \(\Delta=Y_P-Y_Z\) the complete B3-versus-zero policy effect. Then

\[
e_P=e_Z+\Delta,
\]

\[
\|e_P\|^2=\|e_Z\|^2+\|\Delta\|^2
+2\langle e_Z,\Delta\rangle.
\]

For the Niko pre-answer readout, dividing every term by zero's error power gives

\[
0.6592=1+0.6443-0.9851.
\]

The added effect has substantial power, but its opposition to zero's error more than compensates. After consuming EOS,

\[
1.2459=1+0.5117-0.2658.
\]

The cross term remains negative. B3 still cancels part of zero's error, but its added power is larger than that benefit. All six measured boundaries have a negative cross term; the reversal is not simply cancellation becoming reinforcement.

This account concerns the complete policy, including the memory it formed and the computation that reads it. It does not identify which part supplied the cancellation. The [actual-writer study](../../saturn/cmachine/FINDINGS.md#actual-writer-projections-remove-a-native-cancellation), [[2026-11-06-111533-improving-an-mlp-on-its-own-predicted-states|own-state MLP refinement]] and [[2026-11-06-105404-mathematical-accounts-of-model-computation|broader mathematical accounts]] show why that separation matters: a better local term can lose useful cancellation elsewhere.

## Separate forming memory from reading memory

The [completed follow-up](../../saturn/cmachine/composition/formation_read/FINDINGS.md) crosses two episode policies. Let \(\alpha\) select dense or B3 throughout the correction and acknowledgment, and \(\beta\) select dense or B3 throughout a later question and answer:

\[
Y_{\alpha\beta}=O\left(T^{\rm read}_\beta
\left(T^{\rm update}_\alpha(S_0,U)\right)\right).
\]

Dense/dense and B3/B3 come from the authenticated earlier experiment. Two new mixed policies complete the square:

| Correction episode | Reader episode | Question |
|---|---|---|
| Dense | Dense | Native reference |
| B3 | Dense | What did approximate formation change? |
| Dense | B3 | What does approximate reading change on native-formed memory? |
| B3 | B3 | How do the two effects compose? |

For a common observable, define formation effect \(A=Y_{10}-Y_{00}\), reader effect \(B=Y_{01}-Y_{00}\), and interaction

\[
I=Y_{11}-Y_{10}-Y_{01}+Y_{00}.
\]

Then \(Y_{11}-Y_{00}=A+B+I\). The mixed policies locate the complete numerical effect more precisely than another collection of similar correct color answers.

A dense question episode on B3-formed memory is a deliberate hybrid. It does not restore native memory. The earlier dense diagnostics rebuild the correction densely, so they did not already supply this mixed arm. Each new policy preserves its own generated history. All four histories happen to match here, including after EOS.

## Both episodes contribute and their effects interact

The two new policies still acknowledge OK and answer green/red with EOS. Before each question, their rebuilt correction state matches the corresponding historical state in full actual history, position and valid-KV hash. Offline checks match 484 correction files exactly, allowing the two new policies to join the two historical endpoints.

At the first answer readout, full-vocabulary relative differences from native are:

| Question | B3 correction then dense question | Dense correction then B3 question | B3 throughout |
|---|---:|---:|---:|
| Niko | 9.87% | 12.93% | 17.07% |
| Mira | 9.86% | 16.88% | 20.49% |

Dense computation after an approximate correction retains a roughly 10% discrepancy. Approximation during the question episode has the larger isolated norm here. Both effects matter, and their combination is not an independent sum: interaction norm is 23–37% of the joint norm across the two questions and their pre-answer/post-EOS boundaries.

The signed account prevents that interaction from becoming a misleading extra error budget. Correction and question effects align positively at every measured boundary, while interaction has a negative cross term with each. For the Niko pre-answer readout, the complete energy identity divided by joint power is

\[
1=0.33460+0.57389+0.08128+0.20582-0.04693-0.14867.
\]

The first three terms are correction, question and interaction powers; the remaining terms are their pairwise cross terms. Interaction partly cancels the individual effects. These are geometric measurements, not percentages of semantic responsibility.

Even a nearly unchanged error norm can hide a large state effect. After Mira's EOS, dense formation with a B3 question differs from native by 23.1168%, while B3 throughout differs by 23.1228%. Yet those two logit vectors differ from each other by 8.55% of the native norm. Formed context has not become irrelevant; its conditional effect changes direction and cancellation.

Across 104 matching-history controls, fixing the question policy also fixes the captured MLP16 entry, normalized input, write, exit and new KV16/17 byte for byte, despite the changed correction history. Later logits still differ. The formation effect therefore survives in history consumed above this local boundary, rather than appearing as a changed current MLP16 write.

The new run takes 23.40 scheduler seconds, with peak RSS 4,028.3 MB and VRAM 6,378 MB. Verification covers 184 source/payload files and 1,237 raw files; Astra independently checks the joins and factorials. [New findings, code and audit](../../saturn/cmachine/composition/formation_read/README.md).

## Layer17 shows where memory and question computation meet

The episode split establishes a practical constraint: a compact state has to preserve both consequential formed memory and the computation that later consumes it. Resuming dense equations retains about a 10% pre-answer logit discrepancy on B3-formed memory. Similar final error norms can also conceal a large conditional displacement, so we need the actual vectors and their signed combinations.

The [completed attention-read cut](../../saturn/cmachine/composition/attention_read/FINDINGS.md) follows the two episode states into layer17. On common text, the current query and new question KV depend on question policy \(\beta\), while retained correction-span KV depends on correction policy \(\alpha\). Earlier source memory is fixed. All of those paired dependency observations hold in the captured operands.

For correction columns \(C\), this predicts a real-arithmetic score interaction

\[
\Delta_\alpha\Delta_\beta S
=\frac{(q_P-q_N)(K_C^P-K_C^N)^\top}{\sqrt{128}},
\]

with the correct grouped-query head mapping. Outside that span, the paired operand dependencies predict zero interaction. Native scores confirm that support: their interaction is exactly zero on both the earlier source and current question bands at all four measured boundaries. For unscaled QK, the bilinear prediction differs from native within-span scores by about \(3.84\text{–}4.49\times10^{-6}\) relative L2; the scale and multiplication remainders are accounted separately. Retaining the measured FP32 dot-product remainders closes the unscaled vector account to about \(10^{-13}\).

Softmax then spreads the interaction across the attention row. Between 48.45% and 73.94% of absolute probability-interaction mass lies outside the correction span, summed over the twelve query heads. These are fractions of a measured vector's absolute mass, not semantic importance scores. Changed score support is local; changed attention allocation is distributed.

The [prospective plan](../../saturn/cmachine/composition/attention_read/PLAN.md) selected the four worlds at pre-answer and consumed-EOS clocks for both questions, giving sixteen traces. The instrument captures native unscaled QK before softmax, native probabilities afterward and valid historical K/V at layer17, using actual biased, post-RoPE operands. The FP32 scale, dot-product remainder and value-accumulation remainder stay separate from the real-arithmetic factorization. All twenty full-logit files and complete-KV hashes match the earlier uninstrumented worlds exactly. These selected clocks locate the interaction in the attention computation; they do not identify its first temporal onset.

Values also matter even though their four-world interaction is exactly zero. The native context interaction combines changed allocation over baseline values, changed allocation over changed correction values, and changed allocation over changed question values. Those terms have opposing signed cross terms. The [full value-factorial account](../../saturn/cmachine/composition/attention_read/FINDINGS.md#values-and-allocation-form-the-context-together) closes to about \(2\times10^{-15}\) after retaining native accumulation remainder.

## A fixed query admits a smaller update account

There is a useful factorization inside this attention read. For one actual post-RoPE query, its permitted attention support and correctly mapped KV head, write

\[
Z(q)=\sum_i e^{s_i(q)},\qquad
N(q)=\sum_i e^{s_i(q)}v_i,\qquad
c(q)=\frac{N(q)}{Z(q)}.
\]

Suppose the correction changes only a span C while this query stays fixed. Its keys can change the scores and its values can change the weighted numerator. Then

\[
c'(q)-c(q)
=\frac{\Delta N_C(q)-c(q)\Delta Z_C(q)}{Z'(q)}.
\]

The first term adds the changed span's weighted content. The second renormalizes the complete previous read. Even an unchanged source token receives a different probability when the correction changes the denominator. Outside C, every probability in the same head changes by the common factor \(Z(q)/Z'(q)\).

For a fixed supplied query, a scalar denominator and vector numerator are enough to advance the read as new entries arrive:

\[
Z_{t+1}(q)=Z_t(q)+e^{s_{t+1}(q)},\qquad
N_{t+1}(q)=N_t(q)+e^{s_{t+1}(q)}v_{t+1}.
\]

The eight measured fixed-query correction contrasts reproduce the update identity to roughly \(2\times10^{-15}\) L2 in the FP64 diagnostic calculation. Its prediction differs from actual native attention changes by \(2.81\text{–}6.45\times10^{-7}\) relative L2, retaining native softmax and accumulation differences. In Niko's pre-answer native-question read, head9 multiplies every unchanged off-span probability by about 0.40259. With the B3 question query, the corresponding ratio is about 0.60083. Which query reads the memory determines the correction's denominator effect.

This is a compact sufficient account of that particular attention read in real arithmetic; a stable incremental implementation also retains its log-sum-exp gauge and binds the mask, scale and numerical profile. It does not yet give us compact sufficient chat state. A later query changes the exponential weights on all retained entries, so one pair \((Z(q),N(q))\) generally cannot answer it. Restricting future queries to a small fixed family would make a bank of these accumulators sufficient for those reads. The unresolved computation is how to support the queries a conversation actually generates with a representation that remains small and can update itself. These measurements use captured native operands; they do not establish that restricted query family.

## The MLP opposes an interaction that can still grow

At layer17 entry, the four-world interaction is zero. The query/history product introduces it through attention, and the output projection carries it into the receiving MLP. We can now follow the actual native MLP17 response rather than infer it from final logits.

| Reader and clock | Incoming receiver interaction L2 | MLP-return interaction L2 | Layer-output interaction L2 |
|---|---:|---:|---:|
| Niko before answer | 2.686 | 2.654 | 2.803 |
| Niko after EOS | 5.287 | 3.943 | 4.978 |
| Mira before answer | 2.043 | 2.152 | 2.414 |
| Mira after EOS | 5.624 | 4.326 | 5.376 |

The MLP interaction has a negative inner product with the incoming receiver interaction in every case. Yet the combined interaction grows before both answers and shrinks after both EOS tokens. Its opposition is only one term in the energy identity; the added MLP interaction has its own power. This is the same distinction that mattered in the earlier B3-versus-zero account. A negative cross term does not by itself identify a repair mechanism or guarantee a smaller downstream difference. The [signed MLP audit](../../saturn/cmachine/composition/attention_read/runs/job-812a0486995d/mlp-interaction-audit.json) preserves the finite addition terms as well.

The completed trace takes 29.49 scheduler seconds on the Beast, with peak RSS 3,914.7 MB and VRAM 6,378 MB. Strict verification covers 73 source files and 301 raw files; Astra independently reproduces the main accounts. An initial missing-module packaging failure stopped before model execution and remains in the [attempt history](../../saturn/cmachine/composition/attention_read/ATTEMPTS.md). Full weights and KV remain resident, so these instrumentation costs establish no serving-speed or state-compression gain.

## A memory program now precedes its future queries

We have now run that prospective cut. The C constructor sees only the native correction span's 34 layer17 key/value rows per KV head. Before opening either new question's token file, it clusters those rows in key space and freezes a program containing counts, mean post-RoPE keys and associated mean values. No later query is supplied to construction. The actual response predictor takes only the frozen program and the live query; it does not consult the original correction-span K/V.

For cluster j with count n_j, mean key kbar_j and mean value vbar_j, its response is

\[
\widehat Z_C(q)=\sum_j n_j e^{q^\top\bar k_j/\sqrt{128}},\qquad
\widehat N_C(q)=\sum_j n_j e^{q^\top\bar k_j/\sqrt{128}}\bar v_j.
\]

Counts matter: one centroid stands for several original entries. At the selected read, the C runtime combines these virtual entries with the remaining dense history under one global softmax. It pays for that complement. It substitutes only the final question token's layer17 attention read; subsequent native generation can read the original memory again. This is a test of prospective span computation with a real consumer, before attempting an ongoing compressed loop.

The eight-centroid program occupies **16,480 bytes including its header**, versus **69,632 bytes** for the selected native span. One centroid and exact34 provide smaller and exact controls; deleting the span from that one read provides an omission comparison. Two independent later questions ask whose badge became green and request a sentence summarizing both assignments.

| Later question | Eight-centroid attention-context difference | Eight-centroid first-logit difference | Deletion first-logit difference |
|---|---:|---:|---:|
| Recall | 7.648% | **1.592%** | 3.640% |
| Prose | 4.910% | **0.783%** | 1.733% |

Differences are relative L2 against native computation. Exact34 reproduces native bytes across all captured ports, full first/final logits and complete-cache hashes. Every arm, including deletion, generates the same own answers: `Niko's` and `Mira has a red badge, and Niko has a green badge.` Each branch consumes its own output through EOS. The measured advance is numerical fidelity using a smaller local representation. These answers do not discriminate semantic necessity, and the prototype still holds all original model weights and K/V.

There is a useful mathematical prediction behind the improvement. With exact cluster means, convexity gives

\[
Z_1(q)\le Z_8(q)\le Z_C(q)
\]

for every query, since eight clusters refine the single cluster. Every measured ideal-mean head follows that ordering. No analogous theorem orders the vector numerator or final answer. Eight happens to improve the numerator, conditional span mean and full context over one at all 24 head reads here; these are 12 correlated heads in each of two prompts. One centroid's recall logits also end slightly farther from native than deletion after EOS, despite improving the initial read.

## The missing state has a mathematical shape

The eight-centroid denominator is still only 62–64% of the exact span denominator on average, and its numerator has 47–49% relative error. Small full-logit differences therefore should not be mistaken for accurate recovery of the span response. The unchanged complement and signed downstream computation matter.

We can factor exactly what the centroid leaves out. Within a cluster, set a=q/sqrt(128) and t=a^T(k−kbar). An average over its members gives

\[
E[e^{a^\top k}]=e^{a^\top\bar k}E[e^t],
\]

\[
E[e^{a^\top k}v]
=e^{a^\top\bar k}\left(\bar v E[e^t]+E[e^t(v-\bar v)]\right).
\]

The centroid replaces E[e^t] with one and discards the second vector term. Across clusters, define

\[
D=\sum_j n_j e^{a^\top\bar k_j}\bar v_j(E_j[e^t]-1),\qquad
C_v=\sum_j n_j e^{a^\top\bar k_j}E_j[e^t(v-\bar v_j)].
\]

Then exact-minus-ideal numerator is D+C_v. D is lost score dispersion; C_v is finite exponential-score/value dependence, which includes more than linear covariance. Stored FP32 means and the primary CPU predictor add a separately measured remainder. The [saved-only audit](../../saturn/cmachine/composition/span_response/runs/job-524febce43e5/omission-audit.json) uses original span data only after primary predictions are sealed; it supplies an explanation, with no refit or new model call.

Astra independently reproduced both this account and the original clustering and predictions. With each head's numerator terms divided by its exact span Z before concatenation, the eight-centroid recall dispersion/dependence norms are 9.337 and 8.546, with a negative cross term. Prose gives 8.621 and 6.864, with a positive cross term. Both omitted pieces are substantial compared with the roughly one-millionth storage/CPU remainder. They partly cancel for recall and reinforce for prose. These are value-unit diagnostics, not additive shares of semantic responsibility or full-attention importance.

This changes our next question. Fixing the missing denominator alone cannot generally recover the missing numerator. A compact program needs to represent how scores spread within its groups and how the associated values vary with those scores. Moment expansions give one possible route: because E[t]=0,

\[
E[e^t]=1+\tfrac12 E[t^2]+\tfrac16 E[t^3]+\cdots,
\qquad
E[e^t(v-\bar v)]=E[t(v-\bar v)]+\tfrac12 E[t^2(v-\bar v)]+\cdots.
\]

This is a speculative representation design, not a result that second moments suffice. Full key/key and value/key moment matrices at width128 can cost more than this short original span. The next candidate must price its memory and computation, freeze from memory alone, and face later queries through the actual native consumer. The [complete findings](../../saturn/cmachine/composition/span_response/FINDINGS.md) and [Astra review](../../saturn/cmachine/composition/span_response/REVIEW.md) retain that distinction. The single Beast lease took 23.64 seconds and peaked at 6,378 MB VRAM; its strict verification covers 72 sources and 402 raw files. These are instrumented assay costs, not a serving benchmark.

## Keeping the paired atoms is a better small baseline

The next cut supplies a simpler answer to the dispersion and value-dependence losses. We keep all 34 key/value pairs and reduce each row's coordinate precision. A signed-byte row uses one FP32 scale and 128 signed codes. Keys and values are encoded separately, with their original pairing retained. This is standard compressed K/V; its role here is to test whether centroid aggregation discarded more useful information than a similar byte budget requires.

The new bank takes **17,984 bytes**, only 9.13% more than eight centroids and 3.872 times smaller than the original selected FP32 span. It retains the paired exponential mixture

\[
\widehat Z(q)=\sum_i e^{a^\top\hat k_i},\qquad
\widehat N(q)=\sum_i e^{a^\top\hat k_i}\hat v_i,
\qquad a=q\times\text{executed attention scale}.
\]

Both dispersion and score/value dependence now follow the reconstructed atoms. They are approximate because coordinates are quantized, but no longer disappear because several atoms were averaged together. [The C implementation](../../saturn/cmachine/composition/span_recurrent/byte_bank.h) constructs this representation from memory alone before opening the future question files.

We also move beyond one use. Native, exact34, eight-centroid, byte and deletion policies each advance the same linked three-turn conversation: identify whose badge became green, summarize both assignments, then count how many people have a green badge. Every later layer17 read uses the selected policy, during user/template ingestion and own generated tokens through EOS. After the native control, the original correction-span K/V17 is overwritten with quiet NaNs. Candidate execution owns a new root, and the CUDA wrapper refuses native fallback at that layer. The byte bank decodes into counted workspace without consulting those erased slots.

| Response over each policy's 89 reads | Eight centroids | Paired byte rows |
|---|---:|---:|
| Mean predicted/exact span denominator | 0.617861 | **0.998414** |
| Mean relative span numerator error | 48.438% | **2.086%** |
| Mean relative conditional span-mean error | 35.832% | **1.333%** |

Each mean covers 12 correlated heads at 89 positions in one conversation. Byte's worst numerator error remains 12.82%, so the mean is not a uniform guarantee. Its full-vocabulary logit differences stay between **0.0333% and 0.0932%** across all three pre-answer and consumed-EOS boundaries. The centroid range is 0.604–4.599%; at the first recall boundary centroids are slightly worse than deletion's 4.336%. Exact34 reproduces every captured native query, sparse port and full readout, with raw poisoned-cache hashes explicitly distinguished from equal complement-cache hashes.

Every policy answers `Niko's`, `Mira's badge is red, and Niko's badge is green.`, and `1`. Deletion still preserves these answers; information in other layers and later context remains. The stronger evidence is repeated numerical fidelity with the selected original span unavailable to that reader. This does not demonstrate that the span was necessary for those answers.

## The recurrence exposes another useful boundary

Astra independently verifies all 445 token transitions and query arrays. Every Q17 is exactly native. The causal reason is instructive: an intervention at layer17 cannot change the earlier layers that produce its query. Because emitted token histories match, those lower-layer inputs also match. Upper-layer states do evolve under repeated altered reads and produce different later logits, but changed queries feeding back into this selected layer are not exercised here. A stacked-layer experiment would let a lower intervention change a higher selected layer's query; a changed emitted token would also change future lower-layer inputs.

The [mathematical note](../../saturn/cmachine/composition/span_recurrent/MATH.md) makes a useful assumption explicit. If keys within a component were Gaussian and the conditional value expectation were affine in its key, both response terms would close:

\[
Z_j=n_j e^{a^\top\mu_{k,j}+\frac12a^\top\Sigma_{kk,j}a},\qquad
N_j=Z_j(\mu_{v,j}+\Sigma_{vk,j}a).
\]

This could make moment state sufficient under that distributional assumption. We have not established it for finite native attention memory, and full covariance matrices can cost more than these original rows. Keeping paired atoms at reduced precision earns the current evidence with fewer assumptions. It also gives prospective query-conditioned error bounds from source reconstruction error, while leaving arbitrary-query and long-continuation guarantees open.

The [completed recurring findings](../../saturn/cmachine/composition/span_recurrent/FINDINGS.md) bind the 23.78-second Beast run to 76 sources and 1,107 raw files, with 6,378 MB peak VRAM. Astra reproduces byte response measurements with independent FP64 summation. An over-restrictive boundary assertion stopped an earlier attempt before candidate execution. Local strict analysis replay also exposed FP32 log-count differences across NumPy environments; the refusal remains recorded, and a separate CPU-only replay reproduces the frozen analysis bytes exactly in its original environment. No tolerance was relaxed and that audit makes no model calls. [Attempt and verification details](../../saturn/cmachine/composition/span_recurrent/ATTEMPTS.md).

## What this advances toward compact chat

The initial whole-turn experiment takes 31.22 scheduler seconds on the Beast, with peak RSS 3,993.1 MB and VRAM 6,378 MB. Its verification covers 67 frozen sources and 4,215 raw files; Astra independently audits the histories, parents, hashes and numerical summaries. Capture and diagnostic replay are included in these costs, so they are not a serving benchmark. [Protocol, code and evidence](../../saturn/cmachine/composition/whole_turn/README.md).

We now have an approximation that executes through complete new user ingestion and later reads, a measured separation of correction and question effects, a native attention account of how those states meet, a source-only response program, and recurring compressed reads after its original layer17 span is erased. The deletion controls preserve the gap between execution and useful recovered computation. The factorization identifies information missing from centroid aggregation, while the paired-byte result shows that a similar storage budget can preserve much more of the response. The original MLP work continues to identify how omitted operands meet nonlinear consumers.

The next closure step has a concrete shape: quantize each actual newly written post-RoPE K/V pair, commit its encoded cursor, consume it in the current and later reads, and make its original row unavailable. Rollback must truncate encoded ownership together with token state. At this layer that would cost 528 bytes per appended token versus 2,048 FP32 bytes, plus allocation metadata and decode workspace. It would test growing compressed memory; it still grows linearly with context.

Compact sufficient chat remains a larger closure claim. The retained state must support future queries, admit new memory, compose across spans and layers, and preserve useful later computation. Our C machine already persists and advances a conversation using native model state. We now have recurring replacement of one fixed memory span, while new rows, other layers, full model weights and the original cache allocation remain. Compressed append, changed-query feedback, broader consumers and actual physical resource savings are the next separate questions.
