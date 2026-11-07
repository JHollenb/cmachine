# Audit policy for the derived FP64 power-closure remainder

This policy concerns only `power_closure_residual` in the saved factorial analysis. It does not relax source hashes, arrays, token histories, state replay, native FP32 computation, or the constituent powers and signed inner products. The producer's value remains immutable in its original report. Local recomputation and both values/bounds are retained separately.

The prior carry-formation audit already exposed this instrument issue: subtracting independently reduced large FP64 powers leaves a near-zero remainder whose last bits depend on BLAS reduction order. A fixed absolute1e-14 comparison between those remainders is not calibrated to their scale. Astra identified the same risk in this analyzer before its first offline verification. It does not require another model run.

For vectors A,B,I,D with n=151936 entries and the pointwise identity defect r=D−(A+B+I), use FP64 unit roundoff u=2⁻⁵³, m=2n+16 and gamma=m*u/(1−m*u). A conservative absolute-product envelope for the summed dot reductions is

scale=||D||²+(sqrt(||A||²)+sqrt(||B||²)+sqrt(||I||²))².

Cauchy–Schwarz bounds the absolute pairwise dot-product sums by the cross terms in this square. Add the effect of the retained pointwise defect using ||r||≤sqrt(n)*maxabs(r):

bound=gamma*scale+2*||D||*sqrt(n)*maxabs(r)+n*maxabs(r)².

Require each finite producer/local remainder independently to lie within this bound around zero. Every other comparison keeps the inherited strict policy, including each constituent power/cross term and the pointwise defect. This is an integrity check of a declared arithmetic identity, not a scientific quality gate. Its purpose is to reject an invalid identity without converting harmless reduction-order differences into apparent model evidence. The per-field audit records preserve magnitudes; no original remainder is rewritten into a pass value.
