---
title: "2026-11-06 12:55:44 — A frozen MLP program updates a conversation"
timestamp: 2026-11-06T12:55:44-08:00
type: blog
category: research
status: bounded-experiment-with-next-cut
tags: [research, model-analysis, symbol-table, prediction-arc, time-circuit-architecture, saturn, state-machine, language-models, context, memory, mlp]
source_findings: ../../saturn/cmachine/composition/whole_turn/FINDINGS.md
source_review: ../../saturn/cmachine/composition/whole_turn/REVIEW.md
factorial_findings: ../../saturn/cmachine/composition/formation_read/FINDINGS.md
next_cut: ../../saturn/cmachine/composition/attention_read/PLAN.md
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

## The next mathematical target is the first higher attention read

The episode split establishes a practical constraint: a compact state has to preserve both consequential formed memory and the computation that later consumes it. Resuming dense equations retains about a 10% pre-answer logit discrepancy on B3-formed memory. Similar final error norms can also conceal a large conditional displacement, so we need the actual vectors and their signed combinations.

The episode split suggests a smaller causal question at layer17. On common text, the current query and new question KV depend on the question policy \(\beta\), while retained correction-span KV depends on correction policy \(\alpha\). Earlier source memory is fixed.

For correction columns \(C\), this predicts a real-arithmetic score interaction

\[
\Delta_\alpha\Delta_\beta S
=\frac{(q_P-q_N)(K_C^P-K_C^N)^\top}{\sqrt{128}},
\]

with the correct grouped-query head mapping. Outside that span, the paired operand dependencies predict zero interaction. This is a proposal to test using actual biased, post-RoPE Qwen2.5 operands and separately retained FP32 dot-product remainders. The current experiment has not captured that score interaction.

Softmax normalization can spread an effect from those columns across the complete attention row, and value mixing can create further output interaction. A small trace of Q, permitted historical K/V, attention output and the next write would begin locating the computation behind the episode-level cancellation. It needs no mutation of historical memory.

The [prospective attention-read plan](../../saturn/cmachine/composition/attention_read/PLAN.md) measures the four worlds at the pre-answer and consumed-EOS clocks for both questions. It explicitly captures native unscaled QK before softmax, native probabilities afterward and the valid historical K/V. The FP32 scale, dot-product remainder and value-accumulation remainder stay separate from the real-arithmetic factorization. Historical full-logit and complete-KV equality qualify the read-only instrument. These selected clocks test the proposed attention mechanism at two endpoints; they do not identify its first temporal onset.

## A fixed query admits a smaller update account

There is a useful factorization inside this attention read. For one query and its correctly mapped KV head, write

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

This is a compact sufficient account of that particular attention read. It does not yet give us compact sufficient chat state. A later query changes the exponential weights on all retained entries, so one pair \((Z(q),N(q))\) generally cannot answer it. Restricting future queries to a small fixed family would make a bank of these accumulators sufficient for those reads. The unresolved computation is how to support the queries a conversation actually generates with a representation that remains small and can update itself. The current cut tests the fixed-query correction formula on captured operands; it does not assume that restricted query family exists.

## What this advances toward compact chat

The initial whole-turn experiment takes 31.22 scheduler seconds on the Beast, with peak RSS 3,993.1 MB and VRAM 6,378 MB. Its verification covers 67 frozen sources and 4,215 raw files; Astra independently audits the histories, parents, hashes and numerical summaries. Capture and diagnostic replay are included in these costs, so they are not a serving benchmark. [Protocol, code and evidence](../../saturn/cmachine/composition/whole_turn/README.md).

We now have an approximation that can execute through complete new user ingestion and later reads, together with a measured separation of the correction and question effects. The deletion control preserves the gap between that capability and useful recovered computation. The next advance is to locate the higher-layer context read behind those effects, then seek a smaller representation that preserves its dependencies as it updates.
