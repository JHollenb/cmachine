# Advice for the QK × V causal cut

2026-11-06. This is read-only design review plus saved-array analysis of
`../query_memory/runs/job-c7334bbc5bb8`; no new model execution was performed.

The proposed seven-arm assay is the smallest useful next native-consumer cut.
Keep source128, the original layer17 site, the three development cases, and the
existing budgets fixed. The purpose is to localize the remaining compiler error
between addressing (Q/K together) and value payload, before choosing a larger
representation or covariance correction.

## The controlled comparison

Use four full-attention arms: NN (native QK/native V), CN (compiled QK/native V),
NC (native QK/compiled V), and CC (compiled QK/compiled V). Retain native QKV with
compact memory, compiled QKV with compact memory, and zero attention. There are
five historical endpoint joins including native, or four repeated non-native
policies plus native. Reusing the authenticated basis and three compiled images
does not require new source fitting or new image compilation.

For a common token history and common selected receiver sequence, let A_N and
A_C denote the native and compiled QK softmax attention weights, including causal
masking, actual RoPE positions, and grouped-query mapping. Then

    a_NN = A_N V_N
    a_CN = A_C V_N
    a_NC = A_N V_C
    a_CC = A_C V_C

and the real-arithmetic interaction is

    I = a_CC − a_CN − a_NC + a_NN = (A_C − A_N)(V_C − V_N).

The joint defect is the sum of the addressing main effect, the value main
effect, and I. Report these signed vectors or their inner products as well as
their norms. Adding relative error magnitudes is not an attribution. The O
projection is linear, but the receiving MLP and subsequent network are not;
this exact factorization does not apply unchanged to logits.

Verify identical normalized receiver bytes over the maximal common consumed-ID
prefix across the four full-attention arms. This may extend beyond initial
prefill through recall and portions of the follow-up. After histories diverge,
preserve all own-policy observations but stop calling the four-way contrast a
common-operand factorial. Equal token counts or equal latest tokens alone do not
establish a common history.

## What the existing arrays already predict

A post-hoc FP64 full causal attention calculation on the final source128 run's
saved Q/K/V gives the following pooled initial-prompt attention differences,
relative to the native attention norm. Native and CC replay agree with their
saved outputs to floating-point roundoff.

| Case | CN error | NC error | CC error | Interaction norm |
|---|---:|---:|---:|---:|
| Recall | 35.88% | 35.63% | 46.77% | 20.65% |
| Arithmetic | 27.32% | 30.11% | 36.99% | 17.11% |
| Prose | 16.85% | 19.29% | 24.48% | 7.72% |

Addressing and value defects are comparably consequential here. Their main-effect
cosines are positive, approximately 0.54, 0.53, and 0.34; the interaction changes
the joint result materially. The larger raw V reconstruction error therefore
does not justify treating V as the sole bottleneck. Both native-map repairs
improve these local attention vectors, but the existing arrays do not establish
their full-suffix or autonomous-continuation effects. That is the information the
new mixed arms supply. These post-hoc development observations must not be
presented as a new held-out prediction test.

An additional model-run common-parent panel is unnecessary initially. The common
prefix gives the controlled local contrast, and the own trajectories give the
repeated-policy contrast. A later same-parent probe would answer a narrower
question about accumulated state only if those two views leave an important
ambiguity. Do not turn this small cut into another full factorial over parent
cache, current Q, current K, current V, memory algorithm, and suffix.

## Implementation risks and accounting

In NC, native Q and K execute before the first factored map V. Compute Bn lazily
at the first factored map, exactly once per selected token, and reset the ready
flag each token. A wrapper that assumes the Q call always prepares coefficients
will silently reuse stale values in NC. Bind every map choice and saved program
bank to model identity, layer, rank, dimensions, and actual receiver lifetime.
Preserve native biases, RoPE, operation order within each map, and causal append
before read. Count native and compiled Q/K/V calls separately.

The rank128 selected-QKV MAC account is:

| Arm | MAC/update | What remains native |
|---|---:|---|
| NN | 3,145,728 | Q, K, V |
| CN | 819,200 | V |
| NC | 2,981,888 | Q, K |
| CC | 458,752 | No selected QKV map |

Each factored arm uses one coordinate projection plus three output maps, giving
four selected matvec calls rather than three. The mixed arms are paid native-map
ceiling diagnostics; they are not newly autonomous cheap programs. Original
weights, all other native layers, full selected attention, diagnostic histories,
and native control work remain in the ledger. Historical acquisition/compilation
cost is reused, not erased. Keep the existing compact-reader allocation and
CPU/GPU transfer limitations explicit for its two controls.

Candidate sealing before native references remains useful custody. Native maps
inside a declared mixed candidate are part of its intervention, not an undeclared
oracle refresh. They must be labeled as such in the receipt and cost counters.
All source selection is already fixed; no target-driven basis or rank changes
belong in this cut.

## Alternatives and interpretation

Covariance correction targets the memory approximation, while native-QKV compact
memory already has much smaller initial logit differences than compiled QKV.
Starting with the QK/V cut therefore addresses the larger measured formation
defect. Receiver closure is essential for eventual whole-model emulation, but it
changes a much larger boundary and obscures this remaining component error.

The new run can guide whether a separate value representation, a query/key
objective, or a joint consumer-weighted objective deserves investigation. It
cannot establish Q versus K attribution, whole-model closure, semantic necessity
on these deletion-insensitive cases, or a latency improvement. Retain signed
effects, cancellation, follow-up regressions, and wrong native answers as
observations. No individual fidelity threshold should discard an otherwise valid
mixed-arm trajectory.

## Saved diagnostic provenance

The saved-only factorial above uses initial-prompt Q/K/V arrays from report
`f033d61f0e02a047af0991bfa89845600eaafd3afd96d34fe41218caaffd933e`,
whose raw array hashes were independently verified in the predecessor review.
It uses full causal softmax with scale 1/sqrt(128), twelve query heads mapped
six-to-one onto two KV groups, and FP64 arithmetic; it does not call the model.
Its mixed corners are post-hoc predictions, not executed mixed-policy histories.

The reused FP32 program bank is exactly:

| File | Bytes | SHA256 |
|---|---:|---|
| program128-basis.f32 | 786,432 | bb1fa5611aa1b8ee4af91c92597bf9f4d625396ef1c6285f0baf983fb88ede33 |
| program128-q.f32 | 786,432 | 8430d5d05e410d0e5b0a2641153ddac45321877ba4a300cbea4d63145e668493 |
| program128-k.f32 | 131,072 | 60338617616cb20012c9a7e5ec54c7d8d8b64d87cf5ac3f7631e24f3d6ad6884 |
| program128-v.f32 | 131,072 | fd7597bc75d09095f7b455a2b8e190b820c0b162e2fc83af57c79b3860710e68 |

Model body SHA256:
`8b0589b6f367c42e457d53f22af32882f4f0d807803481cdd65c675e33f12fbc`.
Model identity:
`dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`.
