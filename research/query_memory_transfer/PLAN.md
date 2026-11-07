# Native-value gain under recurrent bounded memory

Development mechanism assay, frozen before execution. No fitting, model selection, new source acquisition, new rank, or refresh from target native queries. No terminal capability or latency certification.

Reuse authenticated source128 program bytes and the exact original C FP32 CUDA runtime from verified `query_value_cut/runs/job-27a597ebd5f8` (report SHA256 `da86f60e282cfdf93cd72dffabd98c91cabb7bfb9c3b46450421cc08ca207052`). Layer17, capacity512, one resident Qwen2.5-1.5B-Instruct, physical batch1, CPU1. Retain the same three two-turn development conversations and budgets8/16/32.

Three map policies (NN nativeQKV, CC compiledQKV, CN compiledQK/nativeV) cross two readers (full native FP32 history, bounded FP64 16recent+16countedprototypes). Execute policies1,2,3,5,7 for each task, seal all fifteen candidate paths, then execute three NN/full references. No current NC or deletion repeats. Fifteen repeated paths plus four source arrays must join the previous immutable run byte-exact; three CN/compact paths are new evidence.

Every path starts from token0, produces its own queries, writes its own memory, consumes every own generated token including EOS and receives the second user turn. MLPs recompute on actual receivers. Every path retains its own upper-layer caches. Same consumed IDs do not imply a common whole-model cache.

Primary observable: native-V signed squared full-logit-error improvement CC→CN under each reader, using the SAME NN/full reference and normalization. Keep all four readout boundaries, text, top IDs, truncation, own-KV attention error and token-history matching. Initial/followup and post-divergence results remain separate; no quality threshold may suppress rows. NN/compact diagnoses memory without map approximation. Historical deletion sensitivity remains a limitation of these development tasks.

Post-seal mathematical audit on the maximal exact CCfull/CCcompact/CNfull/CNcompact consumed-ID prefix:

`I = Y_CCcompact - Y_CCfull - Y_CNcompact + Y_CNfull = (Ptilde-P)(V_C-V_N)` in real arithmetic.

Require measured Q/K/receiver pairs, V pairs, native full probabilities and compact key/partition/count state identities before interpreting the factorization. A mismatch labels the instrument inapplicable; it does not erase the policy outcomes. Reconstruct prototype memberships and lifted compact weights only after sealing. They are diagnostic O(n) state and cannot enter the C reader. Native full probabilities are actual saved FP32 outputs. All four saved-output-minus-weighted-row-mixture remainders remain explicit, including accumulation, reduction and casts. Positions1–32 are numerical-profile controls; prototype merges start at33. A nonzero interaction norm is not a signed benefit or a success gate.

Mechanics checks: finite/shape/source/model identity, pending-token authority, candidate seal, exact native sham, exact compact snapshot/mass replay, all native/compiled calls and MACs, selected dense-cache no-write sentinel, bound128MiB. Keep exploration and terminal status separate. The archive bound116,975,016B includes8MiB payload/metadata margin; lifted quadratic weights are reconstructed, never archived.

One strict-preflight Beast lease with measured custom-C reservation6144MB RAM/VRAM (ceiling6758MB), no retries, stop after panel. Previous peak RSS4127.1MB/VRAM6380MB supports this exception to the generic HF planner. One C model, sequential paths, no accelerator concurrency. Expected under2min. Exact-job collection and frozen-source analysis verification precede interpretation.

CN pays819,200 selected linear MACs versus3,145,728 native (73.9583% saving); whole-token linear saving is only0.150724%. Compact mutable logical state98,464B; selected1MiB arena and remaining27layers/full weight bank still allocated. Serving transfers14,336B/advance plus separately paid diagnostics. No measured-speedup claim.

Recent MLP restoration results (blog15:16 and15:27 November6) motivate tracking historical state: exact current native writes can retain15–41% readout error with candidate memory. This run tests recurring selected-attention policy from token0; it does not perform native-history repair or establish a full-model compact state.
