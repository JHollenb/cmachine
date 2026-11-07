# Proposed next cut: select the same query direction through the native consumer

Status: executed in [query_consumer](../query_consumer/FINDINGS.md). The native source-KL selector chooses dose2, with context/clock tradeoffs and mixed target transfer. The original proposal below is retained as the pre-run design; the [next proposed cut](../query_consumer/NEXT.md) extends future horizon on the same sources.

Question: does source-only selection through actual native C distributions improve transfer beyond source attention MSE? Freeze the existing six image files and source direction; add no parameters, query dimensions or memory. Keep the compiled-K/native-V policy and the compact reader unchanged.

Use the original three authenticated source token streams, including their historical eight-token source suffixes. For every image, ingest those same fixed IDs from token zero while carrying candidate-owned selected memory and all upper caches. Native reference follows the same IDs. This is forced source continuation, not autonomous source rollout, and its native distributions are explicitly allowed source feedback.

Measure KL(native||candidate), native-top log probability, attention response, logit L2, concentration and per-trajectory curves at matched clocks. Declare one trajectory-balanced source KL selection rule before execution. Stream or bound full-vocabulary readouts; do not allocate a six-image ×193×151936 tensor. Reuse one native source reference and one resident C model inside one finite mrun lease. Native plus six candidate images imply at most7×193=1,351 new source advances, before any explicitly counted sham. Zero-dose duplicate aliases need no duplicate physical work. Count every new source forward; this is no longer cached-array-only compilation.

Retain source MSE and KL disagreements and every rejected image. Seal the source-selected image before parsing genuinely new target prompts. Evaluate compact transfer and own continuations at the same serving MAC/state budget; use full/compact development contrasts only if needed to distinguish objective transfer from memory-specific compensation. Preserve native mistakes separately from fidelity.

This cut tests consumer misalignment. It does not establish a sufficient B128 state, cheap lower-layer receiver formation, whole-model closure, accurate unrestricted chat or serving speedup. A later autonomous source feedback lane must be named separately from the fixed-ID comparison.
