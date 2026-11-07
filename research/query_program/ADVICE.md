# Better query formation at unchanged serving cost

2026-11-06. Design advice only. No shared runtime edits or model runs performed.

Start with a source-only least-squares replacement of the existing rank128 query
image, retaining the original basis, key image, native V, and memory. This most
cleanly isolates the hypothesis: can the existing coordinates predict a better
query response? It adds no dynamic map, coordinate projection, state, or memory
capacity. A negative result is informative because it separates insufficient
coordinate information from a poorly calibrated query decoder.

## Candidate and finite contract

Let X be the authenticated 193×1536 matrix of source receiver rows, B the frozen
128×1536 basis, and C0 the current 1536×128 query image. Read the original layer17
query weight Wq from the authenticated checkpoint and widen its stored values in
the established way. In CPU FP64 compute

    Z = X Bᵀ
    Y = X Wqᵀ
    R = Y − Z C0ᵀ
    D = argmin_D ||Z D − R||F²
    Cnew = FP32(C0 + Dᵀ).

Use a stable QR/SVD least-squares solve, not explicit normal-equation inversion.
The target is the unbiased, pre-RoPE query map. Original bias and RoPE remain
native, applied after the replacement matvec. The deployed path still computes
z=Bn and q=Cnew z+bq at the actual position. It must not add a second correction
matvec dynamically: fold the fit into one image before serving.

This is a conditional query predictor, not the exact original projection WqBᵀ.
It cannot reconstruct arbitrary new input directions invisible to B. Its source
objective is query L2 fit; it is not an attention- or suffix-optimal objective.
A constant invertible weighting of all query output coordinates would not change
the unconstrained least-squares minimizer. Consumer-specific weighting would
need sample/head-dependent structure and would be a different candidate.

The source-only coordinate matrix is numerically healthy: singular values range
from 272.960 to 8.322, giving condition 32.80. There is no present numerical reason
for an arbitrary ridge parameter or cutoff sweep. Use one declared solve and
retain its full singular spectrum. Rank/finite-contract failures are mechanics
issues; source error size and target improvement are observations, not gates.

## Feasibility evidence and source limitation

An independent source-only calculation finds that B leaves 8.004% relative input
residual on these source rows. Regressing that omitted input residual on Z
predicts 61.25% of its norm, reducing the overall source-input residual to 6.327%.
The fitted input-completion map has spectral norm 0.504. These are in-sample
properties, not predictions of target quality or causal attribution percentages.
They support a plausible modest correction without claiming that the missing
directions are generally recoverable.

The source consists of three 66/63/64-step trajectories, including eight generated
tokens per trajectory and repeated headers. It is not 193 independent contexts.
The basis itself used all three source trajectories, so a leave-one-prompt query
fit with that fixed basis is not clean held-out representation validation; some
training splits are also rank-deficient at128. Do not turn an optimistic source
fit into a generalization claim.

The source-input SHA256 is
`4c6796f4a62ee875088560e0d0441047d77adffa190c9e6dd0fe2d7801fe4314`.
The source128 basis SHA256 is
`bb1fa5611aa1b8ee4af91c92597bf9f4d625396ef1c6285f0baf983fb88ede33`.
Use the authenticated report/bank chain already established through
`job-b94ac2f8a6c9`; the original193 rows originate in `job-2230480e1c18`.

## Why this candidate first

A rank16/32 residual query branch measures extra input directions and may improve
coverage, but changes both information and cost and needs another projection and
query-image application unless compiled into a larger bank. It is the natural
fallback if the existing coordinates prove insufficient, rather than the first
way to test decoder calibration.

A separate query-specific rank128 basis could exploit source query geometry but
requires another 196,608-MAC coordinate projection because K retains its current
basis. It also changes representation and source objective together. This is
larger and less controlled than replacing C0 alone. Neither alternative fixes
limited source coverage automatically.

Keep target cases, outputs, and target-side score diagnostics out of the solve.
The choice to attempt query calibration is informed by the preceding development
results and should be stated as such. Freeze the compiler recipe and all fit
choices before prospective cases; seal emitted candidate bytes before their
execution. If fresh case text is chosen before final image emission, distinguish
that from a literal after-image-seal prompt reveal. The essential fitting
boundary is that no prospective activations, outputs, or comparisons enter the
compiler or alter its choices.

## Lightweight native-consumer assay

Retain old CN/full and CN/compact and run new-query CN/full and CN/compact. Include
NN/full and NN/compact references. This six-arm core distinguishes query-map
formation from its transfer through memory without adding memory parameters.
The three old development cases give exact historical joins for four endpoints
per case. A small separately labeled prospective group, fixed under the declared
quarantine, tests whether the same candidate transfers beyond those cases.

If the paid native-Q ceiling is desired, add native Q + compiled K + native V
with compact memory only. That is a seventh arm, not a cheap serving policy:
its selected QKV cost is 2,981,888 MAC/update versus 819,200 for old/new CN.
NN references alone do not isolate that ceiling because they also restore K.
There is no need to rerun NC, deletion, multiple ranks, damping sweeps, or multiple
candidate fits in this first assay.

Use actual own receivers, causal K/V appends, emitted IDs, and later user/header
tokens from token0. Save full logits at existing boundaries and enough Q/K/V,
probability, and memory data to reconstruct the numerical program. On shared
histories, confirm K and V formation is unchanged between old and new query
programs; compact key partitions must also match, although read weights change.
Beyond token divergence, retain complete own-policy outcomes without a
same-cache interpretation. Restore the original query image as a zero-change
control and preserve ordinary native parity.

Primary candidate arrays seal before native references and post-hoc comparisons.
Keep source-fit residuals, operator norms, query/score/attention errors, logits,
text quality, and truth separate. Finite divergence or a worse prospective case
remains evidence. No quality threshold should erase a valid path.

## Cost and inspectability

Old and new CN have identical dynamic learned-linear cost: one shared rank128
coordinate projection, two compiled maps, and native V total 819,200 selected
MAC/update and 198 whole-token matvec calls. The new query image replaces a
786,432-byte image. An experiment retaining both images pays that extra storage;
it should not claim an allocation reduction. Memory stays98,464 bytes.

Record source rows, B/C0 hashes, original Wq tensor name/dtype/shape/offset/hash,
checkpoint identity, solve profile, singular values, emitted FP32 image hash,
source residual diagnostics, static matrix operations, QR/SVD/scalar work, and
compilation wall time. Read source weights only for static compilation and the
explicit native controls; do not query a target native map to refresh candidates.
Historical source acquisition remains paid even when its rows are reused.

The initial outcome sought is narrower than replacing the model: better query
formation at exactly the current CN cost, with measured own-state transfer through
the unchanged bounded memory. A clean source fit alone does not establish it.
