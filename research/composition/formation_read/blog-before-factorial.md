---
title: "2026-11-06 12:55:44 — A frozen MLP program updates a conversation"
timestamp: 2026-11-06T12:55:44-08:00
type: blog
category: research
status: bounded-experiment-with-next-cut
tags: [research, model-analysis, symbol-table, prediction-arc, time-circuit-architecture, saturn, state-machine, language-models, context, memory, mlp]
source_findings: ../../saturn/cmachine/composition/whole_turn/FINDINGS.md
source_review: ../../saturn/cmachine/composition/whole_turn/REVIEW.md
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

The next cut crosses two episode policies. Let \(\alpha\) select dense or B3 throughout the correction and acknowledgment, and \(\beta\) select dense or B3 throughout a later question and answer:

\[
Y_{\alpha\beta}=O\left(T^{\rm read}_\beta
\left(T^{\rm update}_\alpha(S_0,U)\right)\right).
\]

Dense/dense and B3/B3 are already measured. The two mixed policies complete the square:

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

Then \(Y_{11}-Y_{00}=A+B+I\). Measuring the mixed policies will locate the complete numerical effect more precisely than another collection of similar correct color answers.

A dense reader on B3-formed memory is a deliberate hybrid. It does not restore native memory. The earlier dense diagnostics rebuild the correction densely, so they do not already supply this mixed arm. Each new policy must preserve its own generated history; any token divergence remains part of the result.

## What this advances toward compact chat

The experiment takes 31.22 scheduler seconds on the Beast, with peak RSS 3,993.1 MB and VRAM 6,378 MB. Verification covers 67 frozen sources and 4,215 raw files; Astra independently audits the histories, parents, hashes and numerical summaries. Capture and diagnostic replay are included in these costs, so they are not a serving benchmark. [Protocol, code and evidence](../../saturn/cmachine/composition/whole_turn/README.md).

We now have an approximation that can execute through complete new user ingestion and later reads in this bounded conversation. The deletion control preserves the gap between that capability and useful recovered computation. The next advance is to identify which formed state and which reader operation produce the observed consumer effects, then seek a smaller representation that preserves those dependencies as it updates.
