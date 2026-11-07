# Operand restoration and carried-state separation

Exploratory follow-up to the November 6 size/reasoning/weight review. No training, checkpoint edit, candidate promotion or speed claim.

## Questions and alternatives

The source64 operand basis captures more gate/up weight energy but loses native write fidelity. Is the missed information chiefly in gate, payload, their conjunction, or later accumulated state? We first decompose all 36 saved operand clocks in job-51df6b792d28. We then restore gate/up separately at a common parent and pass each write through the unchanged native suffix.

For s=SiLU(g), shat=SiLU(ghat), delta_s=s-shat and delta_u=u-uhat, the signed write error is D[delta_s*uhat + shat*delta_u + delta_s*delta_u], plus native finite remainders. Preserve cross-term Gram matrices; squared term norms are not additive causal shares.

## Specimen and execution

Qwen2.5-1.5B-Instruct, original C FP32 CUDA/cuBLAS execution-v2 bundle and predecessor-frozen runtime. MLP16, batch1, CPU1, context512. Source64 and random64 factors and actual consumed token histories are imported by hash. Saved B3 clocks participate only in the algebraic decomposition.

Use recall initial prompt, arithmetic initial prompt and arithmetic followup completion as six policy-specific clocks: three per source64/random64. Followup histories differ by policy and are compared only within their admitted history. These are inspected development cases. They provide mechanism observations, not held-out capability estimates.

For each clock reconstruct both candidate-carried and native-same-ID parents before its last input token. From one checkpoint per parent run 00 (candidate gate/up), 10 (native gate only), 01 (native up only), 11 (native both), deletion, ordinary native, and repeated 00. Capture the original receiver, native and candidate operands, consumed write, full immediate logits and four native greedy continuation tokens, consuming every delivered token including EOS. Every branch shares the same valid prefix K/V within that parent. Verify prefix digests before each branch and after rewind. All branches pay original native projections as diagnostics; this is not source-only serving.

The native-parent/11 arm must reproduce the saved native-same-ID readout at that clock. The candidate-parent/00 arm must reproduce the saved candidate readout. Native and 11 must agree exactly within each parent; repeat00 must agree exactly with 00. A byte disagreement diagnoses an instrument problem. Candidate-parent/11 versus native-parent/11 separates carried-state differences after repairing the current write. Future tokens use native execution after the one intervention and do not establish recurring candidate repair.

## Mechanics, observations and custody

Only source/model identity, shape/finite contracts, exact replay mechanics, resource limits and archive integrity can stop an applicable computation. Scientific effects remain signed observations, including restoration that worsens fidelity, correct answers that differ from native, and persistent errors after repair. Terminal certification is not requested.

One finite mrun CUDA lease; no retries, no training, no concurrent duplicate suite. Use the measured predecessor envelope only as an explicit same-backend exception: roughly 4.1GB RSS/6.4GB VRAM plus a 55MB down export and small capture buffers. Request 6144MB RAM/VRAM, strict preflight, timeout840s, expected below five minutes. The prior job's source-weight compiler is unnecessary. All runtime/helper code, bank bytes, saved clocks and token IDs are frozen before submission. Retain scheduler/log binding, source/raw hashes, full vectors, original down weights, local decomposition and whole-invocation Saturn receipt. Index scalar observations through Saturn's offline evidence plane after collection.
