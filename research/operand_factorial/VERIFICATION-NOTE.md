# Saved reduction roundoff policy

The two-cut smoke `job-13b692356e63` completed its native execution and exact historical/common-parent integrity checks successfully. The first Mac saved-data verification rejected differences solely in `power_closure_residual` fields: subtraction remainders of FP64 energy accounts whose terms are approximately 10^5. Linux and Mac reduction order changed those tiny derived remainders by up to about 2.4e−9; raw FP32 arrays, source identities and substantive metrics were unchanged.

The verifier now uses a scale-derived FP64 reduction bound only for that named diagnostic under `signed_energy` and `signed_power_account`. With machine epsilon e, n=2*vocabulary_width+32 and gamma_n=n*e/(1−n*e), the allowed difference is 2*gamma_n times the sum of the absolute power/cross terms. This conservative dot/reduction bound accounts for two evaluations and subtraction. It is tied to arithmetic geometry, not calibrated to make a scientific effect pass. All other comparisons retain the original policy.

`verification-roundoff.json` preserves each producer value, local recomputation, difference, scale and bound. The original report, its raw arrays and executed analyzer remain immutable. The initial verifier refusal is retained in this additive note. No scientific quality threshold or additional model execution is introduced by this correction.
