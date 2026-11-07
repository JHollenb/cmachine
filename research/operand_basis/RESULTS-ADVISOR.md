# Astra: what the source-only operand basis established

2026-11-06. Independent additive interpretation of `runs/job-51df6b792d28/report.json`, SHA256 `e5d23802467bd705ef6e1d9ea7e3dd8925f52589d0dd2942c325404e171a59b7`. No native execution was launched by this review. `advisor_analyze.py` recomputes the following comparisons from collected FP32 arrays, checks their recorded SHA256 values, and writes `runs/job-51df6b792d28/advisor-independent-summary.json`. This is a bounded interpretation, not a new terminal certification.

The experiment closes an implementation gap: a weight-only compiled operand program actually replaces both original selected-layer gate/up projections from the first prompt token through two-turn, own-token continuation. It does **not** close the fidelity gap. The rank-64 source program carries more joint operator energy than its random control, yet its pooled native-write error is slightly worse than deleting the write. It changes the generated behavior beyond deletion, including an incorrect arithmetic continuation. That is useful evidence against identifying source spectral energy with the computation the nonlinear consumer needs.

## Mechanics and the computation that was paid

The raw arrays independently establish exact byte equality of the raw MLP16 receiver and normalized input between each candidate and its native same-ID replay at all 1,128 primary token steps. This validates the triangular same-history premise and the copied selected-layer attention prefix on this panel. There were 273 source64 invocations and 285 each for random64, B3 and zero. The executed actual-call ledger reports no original selected G/U call in any primary path. Source64 and random64 perform three derived projections and the unchanged down projection. All other model layers, native normalization, SiLU, full K/V, vocabulary projection and sampling remain paid.

There are 2,541 physical full token steps: 1,128 primary, 285 native-own and 1,128 native same-ID. Another 1,128 standalone dense-write diagnostics run after the primary seal. These are additional to compilation. The selected two-map arithmetic reduction is real in the declared operation count; no arm-isolated latency result or whole-model forward-pass-free result follows from this job's wall time.

The source basis captures 12.4482% of the unweighted folded G/U operator's Frobenius energy, versus 4.20127% for random64. These are source-operator measurements, not fractions of task computation recovered. Every outcome remains a three-case bounded observation, and hundreds of correlated token steps are not hundreds of independent conversations.

## Local writes, with the correct native receiver

For each candidate's own receiver, define native write w and candidate write ŵ. The pooled error below is

\[
E=\sqrt{\frac{\sum_t\|\widehat w_t-w_t\|_2^2}{\sum_t\|w_t\|_2^2}}.
\]

Deleting the write gives E=1 exactly. These rows have different own histories once their tokens diverge.

| Program | Own token steps | Pooled E | Individual writes closer than zero |
|---|---:|---:|---:|
| Source64 | 273 | 1.008661 | 88/273 |
| Random64 | 285 | 0.996941 | 163/285 |
| B3 | 285 | 1.027298 | 23/285 |
| Zero | 285 | 1.000000 | 0/285, equality by construction |

The strict shared-history comparison reaches the same conclusion without mixing receiver distributions. Source64 has lower error than random64 at 78/221 shared-history steps; pooled errors are 1.001577 and 0.995234. It beats B3 at 154/218, with pooled errors 1.000882 and 1.022933. It beats zero at 87/221, but pooled error is 1.001577. Thus source orientation improves on the previous B3 interface, but the matched random and deletion controls do not support a general native-write reconstruction advantage.

At the twelve saved selected operand sites per nonzero arm, mean relative gate/up errors are source64 0.6715/0.9402, random64 1.0002/1.0197, and B3 0.6656/0.8568. These are selected-clock descriptive means on each arm's actual history. Better individual operand reconstruction did not automatically produce a better joint nonlinear write. B3 is an especially clear warning: smaller selected operand errors coexist with worse pooled writes. Down-projection directions, gate/up conjunctions, error correlations and history all matter.

## Four readout clocks

This table independently recomputes full-vocabulary relative L2 error against native replay of **that candidate's actual consumed IDs**. Columns are before answer 0, after consuming answer 0, after supplied followup, and after consuming answer 1. Post-answer readouts are next-token predictions, not scores of the answer just given.

| Case | Program | First | After answer 0 | After followup | Final |
|---|---|---:|---:|---:|---:|
| Recall | Source64 | .2970 | .3213 | .2472 | .3815 |
| Recall | Random64 | .2540 | .3440 | .2356 | .3453 |
| Recall | B3 | .1595 | .3321 | .2235 | .3324 |
| Recall | Zero | .2328 | .3340 | .2348 | .3363 |
| Arithmetic | Source64 | .4146 | .3108 | .5302 | .3014 |
| Arithmetic | Random64 | .3111 | .3163 | .4509 | .2301 |
| Arithmetic | B3 | .2401 | .3187 | .4920 | .4698 |
| Arithmetic | Zero | .3117 | .3151 | .4308 | .2616 |
| Prose | Source64 | .2538 | .4502 | .2913 | .1898 |
| Prose | Random64 | .2341 | .2595 | .2812 | .2483 |
| Prose | B3 | .2780 | .2038 | .2601 | .1723 |
| Prose | Zero | .2273 | .2726 | .2858 | .2331 |

The rank ordering changes with the clock. Source64 slightly improves the post-first-answer error versus deletion in recall/arithmetic and the final prose error, while other boundaries worsen. In prose, those later comparisons involve different token histories, although each candidate's error uses its own proper same-ID native reference. They cannot be promoted to a controlled same-input superiority claim. All recall boundary argmax tokens agree with their native same-ID references despite substantial full-vector differences. Source64 disagrees at the first arithmetic token, first prose token and post-first-answer prose readout. Argmax agreement at the beginning of arithmetic answer 1 is not evidence that the later digits agree.

## Own-token answers and task truth

All five programs return recall `blue` then `green`, including deletion. This task does not establish that the new operand program is necessary.

On arithmetic, source64 gives `391` then `357`, ends, and omits the requested explanatory sentence. Random64, B3 and zero give `391` then a continuation beginning `391 - 24 = 367`; their second answers hit the 16-token budget. Native gives `401` then `157`, also incorrect. The true first answer is 391; the true second answer after 391 is 367. Even relative to native's own prior 401, its followup 157 is incorrect (401−24=377). Consequently greater native fidelity and greater task correctness are demonstrably distinct objectives here. Source64's shorter transcript is an early stop, not a computation saving certified at matched answer completion.

All prose answers are truncated at the declared 16-token budget. Random64 and zero have identical delivered prose continuations in both turns; source64 and B3 differ. Source64's second answer reaches `The word "bank` whereas B3 reaches `The word "happy`; these short fragments are informative differences, not completed semantic-quality evaluations. No universal meaningful-chat claim is supported by the fragments.

## The remaining mathematical gap and smallest next cut

This result does not refute low-cost operands. It challenges this particular source metric: a common rank-64 right subspace of [GΓ;UΓ] approximates the linear maps in an unweighted global sense. The actual write is

\[
w=D[\phi(g)\odot u],\qquad \phi=\operatorname{SiLU}.
\]

Its relevant errors are joint and consumer-weighted. With δφ=φ(ĝ)−φ(g) and δu=û−u, the finite identity is

\[
\widehat w-w=D[\delta\phi\odot u+\phi(g)\odot\delta u+\delta\phi\odot\delta u].
\]

This explains why independently smaller gate/up norms need not rank the resulting writes, and why increasing rank without locating the missing term is premature.

The smallest informative next native contrast is a bounded **same-parent gate/up restoration factorial**, using frozen source64 and the original receiver/skip/cache: (ĝ,û), (g,û), (ĝ,u), (g,u), plus zero if useful. Reuse fixed selected clocks across the three existing cases, rather than selecting only the arithmetic failure; include the first differing arithmetic digit as an explicitly retrospective localization. Produce the four writes through the unchanged down map and consume them through the same native suffix. This identifies whether restoring gate, payload or their interaction repairs the immediate reader, while retaining finite signed interactions. The intermediate native operands are deliberately paid diagnostic controls, never eligible inputs to a claimed source-only candidate.

For a common parent, the exact factorial write interaction is D[(φ(g)−φ(ĝ))⊙(u−û)]. The analogous output factorial is still well-defined, but nonlinear suffix transport means its magnitude is not a percentage share of the failure. If all four local alternatives have similar output or fail to explain the continuation, the omitted cause lies in accumulated context/history, requiring a separate common-parent cache contrast. Do not infer that a one-step restoration repairs autonomous recurrence.

The C state-machine architecture remains useful: it owns actual token history, clock, cache and continuation and exposes this precise causal interface. The next improvement should be to the operand program's consumer-relevant information, not to its authority to advance state. Full-state autonomous execution is working; useful cheap replacement of this nonlinear write is still an open empirical problem.
