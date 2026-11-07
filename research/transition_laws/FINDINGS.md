# Substantial arithmetic replacement works mechanically and loses these conversations

The completed Beast CUDA job `job-a6811af393aa` removes **40.895% of nominal whole-token learned-linear MACs** by replacing gate/up construction in all28 MLPs with layer-specific rank256 factor programs. Both all-layer programs produce repetitive or unusable replies and answer **0/8** structured questions correctly under the requested strict format. Native answers7/8 and the same native-compiled factors at MLP16 alone answer6/8.

The one own-state receiver refresh has real positive local effects. Pooled reconstruction residual energy on p0-owned source states falls **22.7289%→2.6966%**, improving all28 layer aggregates. Held-out same-input MLP-write discrepancy falls **58.6521%→49.1031%** pooled relative L2, with25/28 layer aggregates improving. Its fresh conditional native KL nevertheless changes **5.23017→5.29082**, slightly worse. This is evidence that repairing receiver coverage and improving current writes did not preserve the full future in this fixed family; it is not evidence that economical transition laws are absent.

The [prospective protocol](PLAN.md), [mathematics](MATH.md), [executed report](runs/job-a6811af393aa/report.json), [independent Astra geometry](runs/job-a6811af393aa/analysis/own_geometry.json), [Astra review](ASTRA_REVIEW.md) and [blog](../../../obsidian/blog/2026-11-06-184700-compiling-cheaper-transitions-on-the-states-they-create.md) supply the full record. Report SHA256: `4fa4f4adf120de07d9ec7162c532139f26cf156b3e9bfa6091003e9295931b11`.

## Fixed compiler comparison

Original Qwen2.5-1.5B-Instruct, C FP32/pedantic cuBLAS, full1536 residual, full FP32 K/V at all28 layers, native RMSNorm/SiLU/down maps, attention and vocabulary computation. The candidate code is operation-local; no tiny latent substitutes for the conversation state.

Six designated source conversations supply native trajectories and a second set generated entirely by p0. Each contributes64 distinct evenly spaced consumed clocks. p0 fits two equally weighted native copies; p1 fits one native and one p0-owned copy. Both randomized-SVD problems are768×1536, q272/two power iterations/rank256/fixed per-layer seeds. C compiles `G E^T` and `U E^T`; no optimizer or answer label enters fitting. Exactly one refresh runs.

Three separate monitor conversations choose p1 using equal-case/equal-turn conditional native-shadow KL: **6.08113→4.33206**, a28.76% improvement. Both factor banks and selection seal before the six fresh evaluation conversations are parsed or tokenized. Own candidate trajectories seal before native shadows consume their IDs. Shadows own their native caches and never supply candidate tokens or activations.

## Actual work and fresh outcomes

| Policy | Nominal linear MAC/token | Linear calls/token | Fresh conditional KL | Fresh logit relative L2 | Strict structured answers |
|---|---:|---:|---:|---:|---:|
|Native|1,543,569,408|197|0|0|7/8|
|p0 at MLP16 only|1,521,025,024|198|0.14105|0.13792|6/8|
|p0 at all28|912,326,656|225|5.23017|1.13678|0/8|
|p1 at all28|912,326,656|225|5.29082|0.99628|0/8|

KL/L2 average equally over six conversations and their two answer turns, each using its own generated history. They are policy-level conditional measures, not matched-state causal comparisons between p0 and p1. Fresh p1 KL improves in6/12 turn averages and worsens in6/12; L2 improves9/12. Those12 clocks are correlated outcomes of six conversations.

Native and single-layer programs return456/474 and351/325 on the two arithmetic sequences. The single-layer program misses the explicit amber→black correction. Native and single-layer both miss the red/blue ticket exchange. The all-layer native-fit program returns fragments such as “The integer.”; the refreshed program returns1 for all four arithmetic questions. Its first ticket answer contains “red” within a capped repetitive phrase but violates the requested one-word output. Preserve this partial mention separately from strict correctness.

Native and single-layer prose is relevant, with all four prose answers hitting the32-token budget. All-layer prose degenerates into repetition rather than supplying the requested explanations. No automatic prose-quality terminal metric is installed. [Strict scoring and per-answer record](runs/job-a6811af393aa/analysis/summary.json) retain all text and caps.

Conditional top-token agreement is not chat success: p0 agrees with a native shadow on107/184 decisions while producing unusable answers. Native is conditioned on p0's already repetitive prefix at those later decisions. These are not107 native-trajectory tokens preserved.

## What the refresh actually repaired

The same fixed source sample banks give a direct cross:

| Retained basis | Native source residual energy | p0-owned source residual energy |
|---|---:|---:|
|p0|0.6297%|22.7289%|
|p1|2.7442%|2.6966%|

Each entry pools squared residual over squared input norm across the selected source samples and28 layers. The refresh trades native coverage for p0-owned coverage in28/28 layers. Mean-layer statistics are retained independently in [the geometry audit](runs/job-a6811af393aa/analysis/own_geometry.json). p1-owned source receivers were not harvested into another fit; a fixed point was not measured.

At the24 saved fresh boundaries per policy, native `M(x)` is recomputed on that policy's actual normalized input after its primary trajectory closes. Comparison with the executed write isolates a local operator defect at that input, with no suffix intervention. p0/p1 pooled write L2 is58.6521%/49.1031%; these are each on their own receivers, not the same receiver across policies. Native-control writes reproduce exactly at every saved native source/evaluation boundary.

The supported inference is limited but useful: this covariance refresh repairs its geometric target and improves local write fidelity, while its own future remains poor. The remaining local write discrepancy is large, so the experiment does not isolate composition as the sole failure. It does not optimize the actual nonlinear response or differentiate the full future. Rank, approximation family, source coverage, inherited history, clock weighting and cross-layer interactions remain live alternatives.

## Paid compilation and evidence

There are42 own two-turn trajectories,4,609 primary transitions,4,609 native-shadow transitions and64 native-sham transitions. Two factor compilations perform28,672 original G/U matvecs and394,600,120,320 MACs. Local native-write diagnostics add14,112 matvecs and194,217,246,720 MACs. The complete C ledger records1,945,613 linear calls and13,241,744,490,496 MACs, including all primary, shadow, sham, compilation and local-diagnostic work.

One factor bank is557,842,432 bytes; both remain resident alongside the6,174,857,216-byte original model. Full state remains allocated. No native-weight exclusion, serving installation or measured serving speedup is credited. Worker time143.38s excludes final custody packaging/upload; the full scheduler lease is237.71s, peak RSS3876.3MB and VRAM7958MB. Those instrumented timings are not per-request latency.

Custody authenticates the2,861,363,200-byte archive,66 executed source files and1,170 declared raw files totaling2,855,252,542bytes. The first worker's object reference omitted the MinIO endpoint; collection explicitly resolved the documented legacy9010 service and checked the complete archive SHA256 before extraction. Future workers now include the endpoint. The initial `job-1df6797e0f96` was killed during compilation by a native-only VRAM envelope; no completed candidate comparison is claimed from that attempt.

The original frozen verifier refusal is retained in [audit-history](runs/job-a6811af393aa/audit-history/frozen-verification.json). It mishandles the nested public Saturn receipt and answers-only turn metadata. A raw C metadata defect also records per-turn budget0 even though execution uses the case budget8/16/32; the derived audit joins those declared inputs, actual command and token counts without changing raw rows. The future C source assigns the metadata field. Artifact-only rebuilding of the public invocation reproduces its before/after cuts and entire receipt exactly, with zero model execution. [Receipt rebuild](runs/job-a6811af393aa/audit-history/receipt-rebuild.json).

The [qualified saved-only verification](runs/job-a6811af393aa/verification.json) validates all42 paths and1,170 raw-file hashes, with947 independently recomputed decision rows. [Versioned audit sources and qualifications](runs/job-a6811af393aa/audit-history/qualification.json) preserve the executed source snapshot and the original refusal. A separately hash-checked fast-tokenizer audit reconstructs all84 answer texts exactly from authoritative IDs and independently reproduces the strict7/8,6/8,0/8,0/8 counts. [Semantic audit](runs/job-a6811af393aa/analysis/semantic-audit.json). Saturn evidence run/verify both return0 over947 observations and16 applicable predicates, with target-stream-only coverage and no claims granted.

Mechanics and scientific quality remain separate. This is a bounded completed experiment with adverse whole-stack behavior and positive local trends; no general emulation or terminal promotion claim is made.
