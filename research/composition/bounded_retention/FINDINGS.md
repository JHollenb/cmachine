# Bounded storage works; uniform sampled memory does not preserve this chat

November 6, 2026. The [completed experiment](runs/job-1d8106dee187/report.json) implements all-layer bounded attention support, immediate encoded append and bounded recovery. It runs five own-state policies from token zero, sharing original BF16 weights through the previously verified FP32 tile consumer. The reservoir policy uses 16 anchors, 64 recent rows and 32 sampled evicted rows. Its complete tracked GPU state, including two marker snapshots and a separate abort snapshot, is **9,279,620 bytes**. It advances beyond physical support, restores overwritten data exactly and aborts an occupied reservoir replacement correctly.

The memory summary fails the delayed consumer: all bounded arms answer **zero of four** late questions correctly. Native and full-byte retention answer all three factual questions correctly and give **40** for a sum whose declared truth is **46**. Bounded execution, useful recall and native fidelity are different observations. No terminal certification is requested.

[Prospective plan](PLAN.md), [implementation](README.md), [strict saved-only verification](runs/job-1d8106dee187/verification.json), [Astra review](REVIEW.md) and [evidence sweep](index/job-1d8106dee187/evidence-verify.json) retain the execution and its adverse results.

## A consumer that actually distinguishes retention

The source supplies Eno=17, Tavi=23 and Luma=amber. A later message explicitly changes Tavi to29. The neutral passage contains 251 template/user tokens and requests only OK. Every arm generates and consumes its own acknowledgement. An early recall child is rolled back before the correction. Four late children independently fork from each arm's delayed root, preventing one answer from rehearsing a fact for another.

Native/full-byte/reservoir roots are at364; anchors and recent-only roots are370 and377 because their own outputs diverge. The largest main-path cursor is409, within the declared logical capacity768. This demonstrates continued append beyond bounded physical support, not unlimited positions or extension beyond the model's maximum position contract.

All literal fact clauses start after anchor16. At the reservoir root, the original clauses occupy42–50,51–59 and60–66; the correction occupies86–100. Recent support begins at300. Every clause is outside anchors/recent memory. The 32 sampled addresses represent284 evictions. They retain two Luma-clause rows and one correction-clause row, but none from either original number clause. Literal token membership is not a semantic-necessity claim: later contextualized rows could still carry those facts, and isolated clause fragments need not preserve a usable address/value binding.

| Policy | Early Eno | Late Eno=17 | Current Tavi=29 | Luma=amber | Sum=46 |
|---|---|---|---|---|---|
| Native FP32 cache |17|17|29|amber|40|
| Full byte cache |17|17|29|amber|40|
| Recent64 |Truncated question echo|Empty/EOS|Empty/EOS|Empty/EOS|Empty/EOS|
| Anchors16+recent64 |17|12345678, truncated|12345678, truncated|Green|15|
| Anchors16+recent64+reservoir32 |17|3|12345678, truncated|Green|15|

These are one dependent development conversation, not five independent benchmarks or an exclusion test against all bounded memory. The native sum40 equals17+the old23. Stale-value composition is a concrete alternative to arithmetic failure; the run does not determine which computation produced40.

## What is bounded and what it costs

The callback retains actual post-RoPE K/V pairs, quantized by the existing row codec. Before overwriting a recent slot, one uniform reservoir decision is shared across all28 layers. The current row is encoded before the same token's attention read. Rows are gathered in absolute-position order. Reservoir entries receive represented mass n/m, m=min(n,32), through an FP32 log-weight bias before historical softmax scaling.

| Policy | Physical row capacity/layer | State GPU bytes | Of which recovery | Model plus state GPU bytes |
|---|---:|---:|---:|---:|
| Native |768|44,789,248|0 full-copy snapshots|3,169,224,212|
| Full byte |768|48,781,316|35,885,568|3,173,216,280|
| Recent |64|6,389,252|4,661,760|3,130,824,216|
| Anchors+recent |80|7,352,708|5,371,392|3,131,787,672|
| Reservoir |112|9,279,620|6,790,656|3,133,714,584|

These are successful CUDA allocation requests, excluding library/device runtime overhead and host metadata. All byte policies pay the same three-copy recovery policy. Native retains the original prefix-based rollback semantics, so its zero full-copy-snapshot cost does not supply identical overwritten-ring recovery. The full-byte control's larger state is primarily its deliberately expensive full snapshots; report serving and recovery separately.

For support C, serving/session/scratch state is712,196+15,864C bytes. Three recovery buffers add3×(14,784C+607,744). Total tracked GPU state is therefore

\[
B_{\rm state}(C)=2{,}535{,}428+60{,}216C.
\]

It depends on configured support, not elapsed position. C112 saves80.98% against the same-recovery full-byte control. The 112-row arena itself costs1,655,808 bytes. Cache host structures/arrays add3,152 bytes under this build; Qwen session/model metadata and audit artifacts are separate. The model/helper/tile budget remains3,124,434,964 bytes, retaining all learned parameter information and forward work. A model-free actual-session fixture confirms equal allocations at logical capacities1024 and2048 with the same support; the real-model run uses768.

## Recovery is part of the state

Two live marker slots and a dedicated pending-token slot have fixed allocation. They copy encoded arenas, row addresses, population count, RNG and full logits. Model/session/epoch/top-marker identity is checked before mutation. Unsupported raw restore, expired marker, out-of-order marker and a third simultaneous marker refuse without mutating the session. Nested sibling overwrite and replay reproduce encoded state and full logits exactly.

Every byte arm exercises a last-layer failure after physical row mutation. The reservoir additionally searches a bounded reversible branch until an occupied sample slot is replaced, fails at layer27, and restores arena, addresses, RNG, population, pending/top and logits exactly before replay. This real branch exercised that case. The deterministic CPU fixture independently forces the same condition. Reset clears support and advances epoch; stale handles remain unusable.

## Factoring the failed read

The [saved-only v2 diagnostic](runs/job-1d8106dee187/diagnostic-v2.json), generated by [diagnose.py](diagnose.py), adds no model calls. At four matching-history late prefills and layers0/17/27, let F be full-byte memory and C the candidate's memory. Define five normalized attention contexts:

1. u0: full query and full memory.
2. u1: same query and full-memory carriers restricted to candidate addresses, with unit weights.
3. u2: same support/carriers with candidate n/m weights.
4. u3: same query/weights with candidate's own K/V carriers.
5. u4: candidate's own query, weights and carriers.

Then u4−u0=(u1−u0)+(u2−u1)+(u3−u2)+(u4−u3). The diagnostic retains every signed vector, four-by-four Gram matrix, normalizer/probability change and native-arithmetic remainder. This is an ordered numerical identity, not causal percentages of the answer. Individual norms can exceed the final difference because terms reinforce or cancel.

| Layer | Omission | Multiplicity | Changed carriers | Changed query | Combined ideal change |
|---|---:|---:|---:|---:|---:|
|0|7.019%|12.012%|0|0|11.541%|
|17|15.049%|20.244%|66.666%|27.497%|73.552%|
|27|48.872%|50.687%|23.666%|51.352%|52.997%|

Each entry is the mean of four whole-context norm ratios against its own full-byte context, not a proportion or independent sample estimate. Layer0 has exact queries and retained encoded carriers under matching histories, isolating support and multiplicity. Higher layers also encode altered previous computation and ask altered queries. Mean full-byte probability on omitted rows is19.30%/10.63%/50.68% at0/17/27, with individual heads reaching72.29%/44.97%/75.76%. These probabilities concern full-byte attention, not the native FP32 distribution.

The telescoping identity closes within3.13e−15 absolute L2. Largest relative executed-versus-ideal arithmetic remainder is6.21e−6. Astra independently reproduces every signed term exactly. Unit-support probability L1 equals twice omitted full-byte mass within4.44e−16, and its log-partition change equals log(1−omitted mass) within9.0e−13. Weighting improves support-only error in0/4,2/4 and1/4 cases at0/17/27; preserve that heterogeneity. The earlier [v1 derivative](runs/job-1d8106dee187/diagnostic.json) combines omission and weighting; its original bytes/source remain preserved. V2 adds the unit-weight intermediate rather than overwriting v1.

## Interpretation and next discriminating cut

Uniform sampling preserves an exponential mixture and paired values rather than collapsing everything into a first moment. For an independently fixed query/population, weighted numerator and denominator estimates are unbiased; their ratio and adaptive own-state futures are not covered. Sampling can miss rare future-relevant rows. With284 evictions and sample32, the chance of missing all nine rows in one fixed clause is about33.5%; actual semantic information is not limited to those nine rows.

The observed failure includes support loss, amplified surviving samples and accumulated carrier/query changes. Reweighting does not uniformly improve individual normalized reads. A past-attention salience policy could still miss facts first requested later. The next controlled repair should expose which additional source-derived distinctions preserve the intended future response, with a same-size support control and own-state continuation. Separately vary the correction's old/new values and include an explicit arithmetic-operands control to distinguish stale retrieval from addition. No new sampler or oracle fact parser is promoted on these observations.

This fits the [nonlinear state-closure result](../../state_quotient/FINDINGS.md): a useful interface must survive its writes and later reads, not merely fit an instantaneous state. Here the actual decoder demonstrates that loss in early support changes later retained carriers and query formation. Fixed storage is implemented; general compact sufficient chat is still open.

## Execution and custody

Job`job-1d8106dee187` succeeds in64.78 scheduler seconds,57.53 worker seconds and44.65 C seconds; peak RSS4,227.7MB/VRAM3,526MB. One weight load, five sequential sessions and40 turns consume2,626 main tokens. Including lifecycle work there are2,656 completed tokens plus five partial aborts:524,192 learned matvecs,2,332,842 tile calls and2,661 embedding lookups. These instrumented timings include loading, snapshots, tracing and diagnostics; no serving speedup is established.

Strict saved-only verification authenticates73 sources,1,060 raw files/80,232,997 bytes, scheduler log/report seal, original weight parent, deterministic recovery transform, actual token files and whole-invocation Saturn receipt. Report SHA256`eb72cf9b4111d5e3f9310ee3886add6d86eff7d0bfe643ea5848b64c70726df6`. It reconstructs25,296 selected own encoded rows across all main transitions. Astra independently audits custody, all token clocks, RNG/support,7,440 selected root row instances and exact counters. Three local tests include ASAN/UBSAN, exact ties-away codec edges and reservoir support invariants; C/Python syntax passes. The evidence run/verify completes over74 observations/13 applicable predicates, no fires/claims/addresses; this is stream coverage, not semantic certification. Installed serving remains unchanged.
