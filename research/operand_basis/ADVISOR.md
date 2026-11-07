# Prospective mathematical review: source-defined gate/up operands

2026-11-06. Astra advisor. No native execution by this review. Recommendations below precede the operand-basis outcomes; the final bound PLAN and executed manifest determine the actual experiment.

The proposed cut is informative because it changes the missing information mechanism while holding the nonlinear consumer fixed. B3 previously saved two projections but discarded consequential operands. The source-anchored gate law preserved useful writes while paying both projections. A weight-derived common basis can test an intermediate point: fewer operand-formation MACs, more information than the three-coordinate B3 interface, exact native SiLU on the candidate's own operands, and the unchanged down projection.

## Fix coordinates before compiling

My recommendation is the raw residual receiver r, with native RMS scalar ρ(r) and learned normalization scale Γ retained explicitly. Define A=GΓ and B=UΓ from the exact exported FP32 weights. For orthonormal basis columns Q and z=Qᵀr,

\[
\widehat g=\rho(r)(AQ)z,\qquad
\widehat u=\rho(r)(BQ)z,\qquad
\widehat M(r)=D[\operatorname{SiLU}(\widehat g)\odot\widehat u].
\]

This convention includes normalization scale in source selection, rather than selecting directions for G/U and then applying them in a different coordinate system. A normalized-input convention is also legitimate, but is a different experiment; do not mix its basis-selection metric with the raw-receiver invocation. Native ρ remains a full-vector reduction and is paid. The full receiver and residual skip remain in the host.

The equations are a real-arithmetic description. Native C forms `(r*ρ)*γ` before the dense maps. Folding γ into A/B and moving ρ outside their products changes finite association even at full rank. Declare the coefficient-construction precision, FP32 casts, and invocation order. Do not label the approximation finite-exact or attribute every discrepancy solely to rank truncation. “Exact SiLU” means the unchanged C kernel applied to the candidate's own approximate g/u, not a native target operand refresh.

## Weight-only compiler and matched control

The natural source objective is a common right subspace for J=[A;B]. A fixed-seed randomized right-singular-subspace algorithm is appropriate for this bounded first run: fixed rank 64, fixed oversampling and power iterations, fixed QR/SVD association and sign convention, with no future receiver calibration. Matrix-free products `Jᵀ(JQ)` avoid constructing the full d×d Gram matrix. A few such passes and a thin SVD are preferable to an unnecessary full SVD. Report the actual algorithm and compilation cost; do not claim exact top singular vectors or a certified optimum from an approximate eigensolver.

Select an unweighted joint operator unless a different source-only weighting is explicitly frozen. Minimizing its Frobenius residual measures operand representation, not downstream write or task optimality. Spectrum, captured energy, basis orthogonality and finite program reconstruction are diagnostics. They do not provide scientific stopping thresholds.

The rank-64 random control must have the same coordinate system, native RMS scalar, rank, dtype, projection/reconstruction grouping, kernels, exact SiLU/down, and invocation schedule. If one basis is constrained to contain B3, constrain both and randomize only the equal-size complement. If neither contains B3, say so. Comparing the structured basis with rank-3 B3 alone would conflate source structure with purchasing additional coordinates.

The historical B3 bank has source activation calibration. It is a valuable predecessor control, but has different supplied dependencies from a weight-only program. Record them explicitly. Do not use historical target activations to tune the new rank or directions.

## Primary continuation contract

A weight-only basis does not need a native source-prefix bootstrap. Prefer applying the selected policy from the first supplied prompt token and through every later user/template/generated token. This closes the selected-layer ingestion boundary and makes any remaining original computation explicit. It still replaces only one operation within a dense full-state host.

Freeze programs and prompts before candidate outputs. Candidate invocations may read their own current r, ρ and immutable compiled factors; the two original selected-layer full-width G/U matvecs are prohibited. Neither exact target G/U nor a dense complement repair may supply a primary candidate. Dense down, all other layers, complete K/V and vocabulary readout remain paid.

The candidates should complete and seal their own traces before native reference/diagnostic values are made available to them. Independently record (a) native own history, (b) native replay on the candidate's actual consumed history, and, where requested, (c) native selected-site computation on the candidate's actual receiver. These answer different questions. Do not compare post-divergence local writes as though their receivers were shared. In the triangular MLP16 arrangement, equal complete token histories imply equal earlier inputs, but that must be checked before relying on it.

Every emitted token must be consumed once, with EOS versus budget truncation explicit. A two-turn episode should ingest the second user's actual text under the same policy; it must not rebuild that context through a dense teacher. Include the proposed deletion-sensitive prose/completion case because the earlier color/arithmetic answers often survived zero. No finite wrong answer or loss of parity is an automatic scientific stopping condition.

## What the contrasts answer

- Structured64 versus random64 tests the contribution of source-derived orientation at matched representation cost.
- Structured64 versus B3 tests a richer operand program, with rank and source dependencies explicitly different.
- Structured64 versus zero asks whether its contribution improves the native consumer beyond preserving the residual alone.
- Structured64 versus native measures bounded fidelity; native task correctness remains a separate observable.

Retain complete first-answer logits, actual tokens, cache boundary identity, local writes, gate/up error diagnostics, and both pre-answer and post-consumed-EOS readouts. The earlier studies show that those boundaries can reverse rankings. Report per-case effects and signed error interactions; do not turn a lower mean error into a universal claim. Decompose current approximation versus accumulated state versus changed history only with the corresponding actual controls.

## Expected computational account

With d=1536, h=8960 and rank k=64, original G/U cost 2hd=27,525,120 nominal MACs per invocation. The common input projection plus two low-width maps cost k(d+2h)=1,245,184. Their three matrices occupy 4,980,736 FP32 bytes per program before metadata/workspace. Native D still costs 13,762,560 MACs.

The replacement uses three derived matrix-vector operations instead of two original ones. It can therefore reduce arithmetic without reducing call count or measured latency. Count native and derived calls separately. The nominal reduction is about 1.70% of the previously reported whole-token learned-linear MACs; all projection, norm, nonlinear, memory, and compilation costs remain visible. This is not forward-pass-free whole-model execution, nor a measured speedup.

## Run integrity and resource scope

I reviewed the current mrun API's intent-first/custom-worker contract, strict preflight, and prepare-once/dispatch-many guidance. One CUDA-admitted lease and one original resident C model are appropriate for these divergent own-token branches. The prepared selected-readout campaign API assumes pure functions of one immutable token sequence and does not justify merging different candidate histories.

Export the selected exact weight bytes once. A CPU compiler subprocess can run while C retains the model, provided the whole lease accounts for both processes, exported arrays, BLAS threads, and compilation memory. A two-phase reload is also valid if both loads are paid and recorded. Do not run duplicate independent full suites to manufacture parallelism. Bind compiled-program hashes, exported weight/model/profile identities, candidate call prohibitions, source selection, and actual code versions before outcomes.

Only integrity, finite safe continuation, and resource contracts can stop the relevant branch. A poor rank-64 consumer result is still informative. It does not prove that all low-cost operands fail, and a good result does not establish a reduced sufficient chat state. The decisive gain would be useful own-history computation with two original projections actually absent from the candidate invocation.

## First implementation review, before launch

The PLAN now chooses the recommended raw-receiver convention and weight-only compilation, applies candidates from the first prompt token, and separates B3's historical activation calibration. The compiler's two-sided randomized range iteration and thin SVD are mathematically coherent: all selection inputs are the exported joint gate/up maps and gamma. This reviews a work-in-progress implementation, not completed mechanics.

These actionable issues were sent to the coordinator in addition to the C syntax/JSON/CLI defects already being repaired:

1. **One emitted basis on both sides.** Initially images AQ/BQ were compiled from Q64, while runtime used Q32. Prefer rounding Q first and compiling images from those exact emitted values widened to FP64; otherwise record the additional mismatch. Measure emitted FP32 orthogonality separately.
2. **Same-ID continuation is a distinct control.** The first driver emitted an empty `same_ids` array. Dense writes on saved receivers do not replace the PLAN's native replay on actual consumed histories. Implement that replay or narrow the PLAN explicitly before execution.
3. **Triangular parity validates the copied selected-layer path.** For MLP16-only policies, candidate and native same-ID replay must have identical raw MLP16 receiver and normalized input at each consumed token. This checks the newly copied attention prefix; an ordinary native arm bypasses that function and cannot validate it.
4. **Seal program identity with the primary forecasts.** The initial seal covered top-level result arrays and driver prefix, while compiled hashes appeared only at the end. Bind weight-export identity, compiler options/source, bank manifest and loaded factor hashes before primary/native comparisons, and join loaded factors to the manifest.
5. **Capture counts must mean physical work.** Initial `primary_steps`/`reference_steps` counted arms rather than tokens. Rename or move increments. Enforce selected G/U prohibitions at the actual matvec boundary rather than hand-assigned counters. Distinguish original, derived, compilation and post-seal work.
6. **Bound collection using tokenizer geometry.** The ~110 MB original G/U export plus per-token gate/up arrays, diagnostic writes, native replays and logits can exceed 256 MiB. Original exports may remain near compute with authenticated recipes/hashes, while compiled factors and decisive evidence must be collected. A compiled directory outside `RAW` must not disappear from custody.

The selected-layer formula is consistent with the row-major ABI: QT maps d→64, the images map 64→h, actual rho scales each, and original SiLU/product/down consumes them. The normalized input is a diagnostic/native-norm computation; it does not secretly invoke original G/U in this candidate formula. Three derived calls replace two original calls, so measured timings are still needed for a latency claim.

## Final bounded source review, still before native outcomes

The revised compiler builds both images from the exact emitted Q32 basis widened to FP64, then records their FP32 casts. The revised driver invokes source/random factors on its own raw receiver, applies its own native RMS scalar, and uses unchanged SiLU/product/down. The native arm now follows the original runtime. The GNU linker wrapper instruments actual `gpu_matvec` calls and identifies selected original G/U/down by their resident weight pointers; the Makefile wraps the symbol at the final link. This is a materially stronger prohibition check than hand-assigned counters.

Native same-ID replay now consumes the candidate's actual recorded IDs, including its own generated tokens and later supplied template. It records the four requested readout clocks. The candidate/native same-ID raw receiver and normalized-input byte comparisons remain the decisive validation of the copied selected-layer attention prefix. The compiled input manifest, bank manifest and six factor hashes are bound before primary outcomes; actual bank immutability verification, selected operand diagnostics and final clock/accounting adjustments were identified by the coordinator as remaining implementation work at this review snapshot.

No additional mathematical or operand-leakage blocker was found in this bounded reread of `driver.c`, `Makefile`, `worker.py`, `run.py` and the emitted-basis compiler changes. This is readiness for the declared mechanics smoke after those known fixes, not a claim that native mechanics have already passed. In particular, smoke must establish copied-prefix parity and the original-call prohibition on actual CUDA execution.

One accounting clarification: the current `work.reference_steps` increments native-own steps, whereas same-ID replay is reported separately by replay count and its consumed-ID arrays. Total physical token work must add the lengths of those same-ID arrays; it must not equate replay count with token count or treat `reference_steps` alone as all reference computation. Post-seal standalone dense-write diagnostics and the source compiler are additional paid work. This is a reporting correction, not a scientific quality gate.
