# Response-aware query compilation at the same serving cost

The same B128 code supports a query image that better controls the approximate memory even while becoming a slightly worse predictor of native Q. Source response squared loss falls **8.36%**. Initial native attention, selected layer output and full-logit L2 improve in all six conversations. Compact L2 improves at **22/24** saved boundaries, native-distribution KL at **18/24**, with unchanged emitted tokens. Three initial KL countertrends remain. This is a bounded computation improvement, not answer improvement or sufficient-state closure.

Executed `job-b141eadfe952` on Beast, original Qwen2.5-1.5B-Instruct, C FP32/pedantic cuBLAS, selected layer17. [Protocol](PLAN.md), [derivation](MATH.md), [report](runs/job-b141eadfe952/report.json), [verification](runs/job-b141eadfe952/verification.json), [saved-only study](runs/job-b141eadfe952/study.json), [Astra review](REVIEW.md).

## Source construction

The exact LS query parent is `99e6fa43c6fbed8d01a78f2255b7a3ba12d57b49393f0a32392e3c292656e1c6`. B128, compiled K, native V, bias/RoPE, memory and all other weights stay fixed. Reconstruct source Q/K/V from the authenticated original bundle using FP64; reset three source clocks at66/63/64. The teacher is CPU64 native-full attention. Candidate source nodes use rounded FP32 compiled keys/native values and the unchanged FP64 centroid/count law. These are source surrogates, not captured byte-exact CUDA teachers.

One analytic response gradient, pulled back through half-split RoPE and coordinate-preconditioned with source least squares, gives one direction. Its quadratic response prediction gives line step1921.773684376. Evaluate six declared doses after FP32 image rounding; select source-only minimum. No target outputs, gradients or checkpoint selection enter this choice. The fresh prompt bytes are visible to the author and payload hashing, but parsed only after the image seal. No new source model advances occur.

| Dose | Source response squared loss | Query-vector error |
|---:|---:|---:|
|−1|0.04770763|5.81044%|
|0 / LS|0.03915260|5.77699%|
|0.25|0.03777369|5.77908%|
|0.5|0.03678167|5.78537%|
|1 / selected|0.03587933|5.81044%|
|2|0.03752226|5.90964%|

The selected query displacement is0.6236% of source LS query norm. The three source response relative errors change19.647→18.692%,19.614→18.872%,20.096→19.257%. Mean node attention entropy rises1.297923→1.304931 nats. Source finite-difference derivative discrepancy is1.47e−7 relative. All193 rows belong to three correlated trajectories already used to construct B; this is in-sample construction evidence.

Selected image SHA: `842d8d3d3654062e9ea0a6935815f0a015ba88c2df83ab60eadf92549343cd04`. All six proposals are retained. Saturn image-only Act/restore receipts verify exact closure to the same parent; there is no optimizer or RNG state. These receipts certify image installation/restoration, not an internal C cache rewind. The separate Saturn receipt binds the complete C invocation.

## Native transfer and countertrends

Three development conversations use native-full, LS-full, response-full, LS-compact and response-compact. Three genuinely new prospective conversations use native-full, LS-compact and response-compact. All18 candidate paths seal before six native references. Each path executes its own queries, appends its own memory, recomputes O/MLP, carries its own upper caches and consumes its own emitted tokens.

| Initial compact native-logit L2 | LS query | Response query |
|---|---:|---:|
|Development recall|11.781%|11.729%|
|Development arithmetic|12.357%|12.201%|
|Development prose|6.175%|5.968%|
|Prospective silk recall|9.022%|8.989%|
|Prospective23×18|9.497%|9.190%|
|Prospective seed-packet prose|4.032%|3.915%|

Target initial Q error becomes slightly worse in five of six cases while attention, layer output and logits improve in all six. This directly distinguishes response-aware compensation from query reconstruction.

Across24 compact boundaries:22 improve logit L2,18 improve KL(native||candidate),17 improve native-top log probability. On16 boundaries where both policies still share the native consumed history:15 improve L2 and13 improve KL. The12 development full-memory comparisons improve L2 at10 boundaries. Later comparisons after native-history divergence are operational policy distances, not fixed-parent local causal errors.

The two compact L2 reversals are development recall followup-prefill13.100→13.137% and prospective arithmetic final81.157→81.256%. Initial KL worsens for development recall0.000204412→0.000231510 nats, prospective silk recall0.00446452→0.00478025, and prospective seed prose0.00289187→0.00441271. Development prose followup-prefill also worsens KL despite better L2. Tiny EOS-boundary effects and the larger post-divergence arithmetic KL reversal remain in the raw rows. Do not rank18/24 versus the preceding study's19/24: the prospective cases changed.

All six paired compact token histories remain identical; the three paired development full-memory histories also agree. All paired compact selected receivers, coordinates, K/V and24 memory snapshots remain byte-identical, while queries/attention/upper writes change. Thus we have not yet exposed target-history changes driven by this new query image.

Both recall tasks answer correctly. Development arithmetic remains266 then wrong286 instead of279. New arithmetic answers414 then wrong459 instead of441; native answers414 then wrong451. Prose remains coherent but differs from native. Explanations are truncated at the declared16-token budget. Native-distribution fidelity is distinct from task truth.

## Conditional algebra and costs

The development full/compact factorial retains the conditional identity `I=[(Ptilde_response−Ptilde_LS)−(P_response−P_LS)]Vnative` on matched selected inputs and common histories. Initial interaction norms are2.394/2.393/2.358% of LS-full attention; aggregate four-mix remainder closure is2.05–2.30e−16. Before information merging, effects are only~2.3e−7 numerical-profile differences. This algebra is a source-bound accounting identity, not a universal factorization after state histories change.

Serving is unchanged:819,200 selected QKV MACs/update versus native3,145,728;198 whole-model linear calls versus197 native;98,464-byte logical selected memory;14,336 serving CPU/GPU transfer bytes/update. Whole-token linear reduction remains0.1507%. The static per-policy query image is786,432 bytes, replacing an equal-sized image; diagnostic comparison residency carries both. CPU source affine construction is651,395,072 nominal MACs, plus separately paid gradient, response, QR/SVD, hashing and diagnostics. Compilation took3.63s. Original source acquisition and B compilation remain paid historically.

The run completed24 paths/2,412 own advances,550 emitted-and-consumed tokens,96 full readouts and48 exact memory snapshots. Native sham costs32 additional physical advances. The C ledger records483,280 learned-linear calls/3,768,267,964,416 MACs including sham. One resident C model, CPU1/B1, no retries; elapsed61.85s, worker48.18s, driver34.95s; peakRSS4224.3MB/VRAM6380MB under6144MB declared reservations and their measured1.1 ceiling. Instrumented path timings do not establish serving speedup.

## Custody and next cut

Report SHA `2ff3b934b26b496c054be44feef4767bedda779866bb33372a6498f4b1addc53`. Independent saved-only verification authenticates74 source/392 raw files and recomputes the frozen analyzer with zero models. All compact reads replay exactly. Nine historical development paths/126 arrays match the preceding study, with LS baseline names mapped explicitly; B/K/V match their old banks and Q matches the prior selected LS image. Raw files total145,626,563 bytes under the173,856,840-byte precomputed bound and256MiB limit. The evidence sweep verifies2,781 primary observations with no fires; additive KL/study diagnostics have separate custody and are not retroactively called primary stream coverage.

Astra identified one audit strengthening after the run sealed: tie proposal bytes to the saved direction/dose. [Additive update verification](runs/job-b141eadfe952/update-verification.json) confirms all six exactly equal `FP32(LS+recorded_step*saved_direction)`. Current compiler includes that check for future runs; frozen payload/raw/report remain unchanged. No tolerance is loosened and no C rerun was needed.

The smallest next experiment is source-only selection among the existing six frozen images using actual native C source-distribution KL, with fixed original source token IDs and candidate-owned upper caches. Count those new source forwards explicitly. This tests remaining consumer misalignment without a new direction, more parameters or larger memory. It would use forced source histories, not autonomous source rollout. Freeze its choice before new targets. Dense receiver generation, other27 layers, O/MLP, full weights/caches and vocabulary head remain paid; accurate free chat and forward-pass-free whole-model emulation remain open.
