# Two history measures, one frozen program

For a token prefix h, let S_N(h) and S_F(h) be separately constructed native and compact states. Their caches differ, while their consumed token bytes agree. Define ell_F(h)=KL(p_N(.|S_N(h))||p_F(.|S_F(h))). The experiment compares L_N=E_(mu_N)[ell_F] on archived native-generated histories with L_F=E_(mu_F)[ell_F] on newly compact-generated histories, using equal source/turn and within-turn decision weights.

The exact finite-measure account is L_F−L_N=integral ell_F(h)(dmu_F(h)−dmu_N(h)). A mean difference describes conditional fidelity under two visited-prefix distributions. It does not isolate cache error, local query error or semantic usefulness. Each state's hidden computation is its evaluator's own computation from token zero.

Compact autonomous authority is u_(t+1)=argmax H(S_F(h_t)); the native shadow then consumes that exact u_(t+1) in its own state. The shadow's pending token is an observation, never authority over the compact branch. For a prefill ending at p followed by k generated tokens, decision rows p−1 through p+k−2 count, including the EOS choice. Unused post-budget/post-EOS distributions do not count. The fixed followup skips an already consumed leading template EOS exactly once.

Native/compact source histories coincide only until their first different generated token. Within that exact common prefix, evaluator-specific states and retained arrays should be byte-identical across history conditions. Beyond it, time-aligned summaries are not same-state comparisons, and differing EOS times change both the support and length-normalized weights. Preserve lengths and common-window diagnostics rather than padding or erasing rows.

The native-history replay is an instrument control against archived hashes. The autonomous-history branch provides new evidence. No query image changes; no held-out transfer or accuracy improvement can be attributed to fitting or selection in this run. Serving stays819,200selected QKV MAC/update and98,464selected memory bytes; measuring the shadow is newly paid experimental work.
