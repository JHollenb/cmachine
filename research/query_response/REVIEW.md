# Astra review and primary critique

Astra advised a single coordinate-preconditioned response-gradient direction with a small source-only dose panel. It checked the squared-loss factor2, curvature step, half-split RoPE adjoint, separate source clocks, compact node mass, and the24-path dispatch. An independent toy directional derivative agreed to7.2e−11 relative. All74 staged source hashes matched.

After execution Astra independently checked74 source/392 raw hashes, report identity, six exact dose/direction image constructions, the frozen analyzer,48 snapshots and zero compact read replay error. Its stable stdlib-fsum logit audit confirms22/24 L2,18/24 KL,13/16 same-native-history KL and17/24 native-top-logp gains. All six compact histories are identical to the LS baseline. No answer improvement is established.

The clearest mechanism observation is source response loss−8.36% while source Q error rises5.77699→5.81044%. Initial Q error also rises in five of six targets while all six initial attention/layer-output/logit comparisons improve. Query fidelity and useful reader control are distinct.

My critique: the objective is still a source attention surrogate. It balances three correlated source trajectories and cannot observe sensitivity of O/MLP, upper caches or the full-vocabulary distribution. Three initial KL countertrends demonstrate this limit directly. All selected target memory histories remain unchanged, so it does not yet show an improved autonomous token trajectory or a solved self-produced state transition. Native-reference KL also rewards native arithmetic mistakes; it is not correctness.

The source node construction is CPU64 with FP32 state rounding, not captured CUDA operands. Native C transfer is therefore necessary, and is measured here. Public image Act/restore receipts cover image-only state; the separate receipt covers the complete C invocation. We do not claim internal C cache rollback from those image receipts.

The original verifier hashed proposals and recomputed source diagnostics but did not explicitly tie each saved image to the stored direction and reported dose. Astra requested this connection. A separate post-run verifier now checks all six byte-exact constructions; current compiler adopts it. The frozen executed payload/report is unchanged, and this strengthening is recorded additively.

Next: choose among the already-frozen six images using actual native C KL on the original source IDs, preserving candidate-owned upper caches. This spends newly counted source forwards, not target feedback. Keep the source reader loss and all rejected images as evidence. Re-evaluate fresh targets only after sealing the source-selected choice. Do not enlarge the memory or add another proxy before testing this direct consumer.
