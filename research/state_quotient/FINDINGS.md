# A nonlinear compact state closes until its writes expose omitted orientation

November6,2026. The [completed Beast job](runs/job-d57473b075cc/report.json) gives a constructive result and a controlled counterexample. A two-coordinate state plus invariant energy advances a supplied nonlinear residual program exactly through16 updates. One added off-space write makes the discarded sign consequential. Retaining that coordinate restores byte-exact own-state execution through the same horizon. Separately, an exact original-weight minor proves Qwen's layer16 down map has rank at least256, excluding a common read/write space below that rank for this strong sufficient construction.

The first result is a known-wiring organism through unchanged C CUDA primitives, not a compressed pretrained model. The second is a weight-rank lower bound, not a theorem against every compact or approximate model emulator. No Qwen forward, token generation, learned state discovery or serving installation occurs in this job.

![Own-state closed, escaping and repaired trajectories](runs/job-d57473b075cc/state-quotient.png)

The circles in the middle panel are the repaired own-state trajectories, exactly over their full-state references. Distances are measurements on this supplied organism, not language-model quality scores.

## The condition that works

The [frozen plan](PLAN.md) supplies d=8, intermediate width4, epsilon1e−6 and the recurrence

$$
x^+=x+D[\operatorname{SiLU}(G\rho(x)x)\odot U\rho(x)x],\qquad
\rho(x)=(\|x\|^2/8+10^{-6})^{-1/2}.
$$

The initial states are (.75,−.5,+1,0,…,0) and (.75,−.5,−1,0,…,0). G/U and the declared reader see only coordinates0/1. In the closed condition, D also writes only those coordinates. The code z=(x0,x1),κ=1 therefore has all required read operands, and its writes leave omitted orientation unchanged. The compiled two-coordinate G/U/D program carries its own state; no future full-state values refresh it.

Both full-state signs produce exactly the same declared trajectory. Every retained state, RMS scale, gate/up/product and retained write agrees byte-for-byte with the reduced program over16 updates. This demonstrates one finite nonlinear closure and its cost reduction on a supplied domain. It does not establish universal FP32 energy-only equivalence: the compact executor forms a canonical eight-slot RMS representative, and the supplied omitted squares ±1/0 match the native reduction leaves. Full-width normalization, constants and scratch remain paid. [Full mathematical contract](MATH.md).

## The one write that breaks it

The escaping condition changes only D[2,0] from0 to.25. Initial normalized reads and first gate/up/write amplitudes remain shared. But the omitted coordinate is now written:

$$
e^+=e+Vw,\qquad\kappa^+=\kappa+2e^TVw+w^TV^TVw.
$$

The old code knows κ and not the signed write coupling eᵀVw. At the first update, the two cross terms are −0.1422131658 and+0.1422131658, while the common squared-write term is0.0050561461. Observed omitted energies become0.8628430357 and1.1472692480, with separately retained FP32 addition remainders about+5.54e−8/−6.38e−8.

Encoded omitted energy therefore separates after one update. The first subsequent normalization read separates at zero-based read1; the declared coordinate output separates after update2. The old invariant-energy candidate follows its own trajectory and ends22.23%/21.28% away from the two respective full-state outputs, measured as relative L2 on coordinates0/1. The two native final declared outputs are0.437324L2 apart. A current read can agree while the next write makes a discarded distinction readable.

## One coordinate repairs the interface

The repaired program retains coordinate2, sets residual omitted energy to0, and compiles three-coordinate maps. Its plus and minus paths use their own retained initial sign and own updates. All retained states, normalization scales, gate/up/products and writes match their corresponding escaping full-state reference exactly through16 updates.

| Program | Retained episode numerical state | Static affine coefficients | Nominal affine MAC/update | Result in this assay |
|---|---:|---:|---:|---|
| Full8-state reference |32B|384B|96|Reference |
| Closed2-coordinate plus κ |12B|96B|24|Byte-exact16-update trajectory for both signs |
| Old code after escaping write |12B|96B|24|First declared-output defect at update2 |
| Repaired3-coordinate plus κ_perp |16B|144B|36|Byte-exact16-update trajectory for each sign |

The table counts retained coordinates plus one explicit episode-invariant energy for compact programs, not process memory. A common eight-slot RMS representative, gamma, scratch, transfers and recording are additional. Intermediate nonlinear width remains4. This is a reduction of the supplied affine program, with no matched latency/speedup claim and no corresponding Qwen serving saving.

These are nominal **dense** affine counts for cuBLAS calls on a deliberately sparse known-wiring organism. A hand-sparse full-state implementation could also exploit its supplied zeros. Each update still launches three matvecs. The experiment establishes the state-interface mechanism and the executed smaller map geometry, not an optimized baseline comparison or a discovered efficient pretrained algorithm.

## Original Qwen weights constrain the strong assumption

The driver authenticates the complete3,087,469,344-byte original TCM bundle, CPU-maps original layer16 `down_proj.weight` of shape1536×8960, and extracts its fixed leading256×256 minor. BF16 widens exactly to FP32 on CPU. No Qwen GPU weight bank loads and no Qwen forward executes. The saved minor is262,144B, SHA`494c7c4fe0081c1b3c9e4ccb9d677b382d9844b0b12618c19303022d7f09104a`.

Exact mapping of its dyadic rationals to p=2³¹−1 gives determinant **1932435391**, nonzero. Therefore the minor is nonsingular over the rationals/reals, and rank(D)≥256. An independent stdlib implementation maps values with `float.as_integer_ratio`, performs arbitrary-precision unnormalized elimination and obtains the same rank/determinant. The producer uses an independent bit-field map and NumPyint64 normalized elimination.

The constructive quotient requires every column of D to belong to the retained space, hence rank(R)≥rank(D). This excludes rank3,rank64 and all ranks below256 **for that condition**. It does not prove full1536 rank, that reachable SwiGLU writes span all columns, that approximate low-rank programs are useless, or that no nonlinear/consumer-relative/restricted-domain state can be smaller. Combined with the earlier [exact B3 collisions](../closure/FINDINGS.md), it makes a universal tiny common linear interface an unsupported design default.

## What the newest posts change

The [completed attention trace](../composition/attention_read/FINDINGS.md) now gives a native conditional update account. At a supplied fixed query,

$$
\Delta c=\frac{\Delta N_C-c_{\rm before}\Delta Z_C}{Z_{\rm after}}.
$$

Its native context-change discrepancy is2.81e−7–6.45e−7relative. Changed correction scores have localized support, yet48.45–73.94% of probability-interactionL1mass lies outside that span after normalization. Keep denominator competition and changed-value/allocation coupling. A compact response function over future queries must support the queries produced by its own state, not only one supplied query. This is a completed result; earlier review prose marking that trace proposed is now historical.

The [new selected-only image reader](../../../time-circuit-explorer/docs/selected-reader-image.md) removes88.93% of comparable reader numerical state, retaining objects and their spatial swap on one known pair while signed PNGresponse error rises10.72→19.37%. Full K/V scanning and most original learned work remain. It supports useful approximation even when discarded information is observably not invisible. The exact closure condition is sufficient for exact emulation, not a necessary condition for a useful approximate program. The uniform8,208Bstate instead advances its own law exactly while losing the behavior. Preserve both results and the source's additive reporting correction.

The [new computation-law campaign](../../../artifacts/mdb-astra-generalization-20261106/RESULTS.md) qualifies source-derived even-SwiGLU and immediate readout laws, plus full finite image forecasts. It retains original factors and observes rather than predicts later Qwen cached steps. These laws suggest operation-specific compilers; their actual operand formation, future query coverage and finite ABI still count. None transfers measured0.5B CPU errors automatically to the C1.5B CUDA profile.

## Execution, verification and limits

Job`job-d57473b075cc` takes11.62scheduler/10.35worker/7.81driver seconds, with sampled3055.0MBRSS/250MBVRAM. The full model mapping/checksum contributes to CPU/RSS cost. Eight paths execute128 component updates,384 tiny native matvecs,128 RMS and128 SwiGLUs; four compact paths and all source matrix files seal before four full references. These128 steps and71 observation rows are dependent controlled coverage, not independent language trials.

Strict [verification](runs/job-d57473b075cc/verification.json) authenticates60source files,67raw files/285,672B, candidate34-file seal, complete scheduler/log/report joins, executed analysis, loaded-model digest/identity and whole-invocation Saturn receipt. ReportSHA`10dbec1fc475d10366a9ce33c8d0b0fcdbff4e3ecf010458b59fca3d8dbca384`. The [independent audit](runs/job-d57473b075cc/independent-audit.json) additionally checks672retained-state scalar additions byte-for-byte. Five exact-integer tests and a one-step model-free ASan/UBSan mechanics check pass. Standard isolated [evidence run](index/job-d57473b075cc/evidence-run.json)/[verify](index/job-d57473b075cc/evidence-verify.json) return0 over71rows/5applicable predicates with no fires/claims/addresses; this is stream coverage, not semantic certification. [Astra review](REVIEW.md) records preflight and post-run critique.

Intent-first admission inherited13312MBRAM/10240MBVRAM from unrelated generalized full-model history and queued behind a sound image job. We did not cancel that job, bypass admission or resubmit. Actual component measurements are now available; `mdb-component-aware-mrun-planning` in the request ledger is upvoted with this case. Reservation is not actual consumption or a speedup.

The shared runtime and installed serving are unchanged. The resulting design rule is concrete: discover a reusable, economical interface closed under its declared reads **and writes**; where exact closure is too expensive, retain an explicit approximation domain and test its own future query/read/write trajectory through the actual consumer. Accurate unrestricted chat and forward-pass-free emulation remain open.
