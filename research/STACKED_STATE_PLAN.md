---
title: Stacked state and reliable chat in the C model machine
date: 2026-11-05
type: research-plan
status: prototype-runtime-increments-implemented
author: Jake Holl with Codex and Astra
tags: [research, saturn, state-machines, time-circuits, operational-variables, attention, mlp, qwen, memory]
---

November6subsequent update: the [drawer suffix crossing](../../artifacts/qwen-relational-reader-cut-20261106/RESULTS.md) has executed. Native Q rescues both supplied initial reads with unchanged bounded memory; exact current read still leaves a higher-layer history effect at256tokens. A [source-only same-memory response target](drawer_suffix/FINDINGS.md) is prepared from162own-history decisions, with no fit or new model call. It replaces the earlier proposed full-native local target for the next bounded query direction. Historical proposed cuts below remain chronological; follow the [revised assay](query_own_history/NEXT.md) for current execution status. Native-Q zero-read/value-permuted controls remain the distinct next payload-attribution diagnostic.

The C machine gives us an executable reference for advancing model state. Nested resumable computations, selected read/write observations, a controlled nonlinear MLP program and actual writer interventions are implemented. Natural-context reuse exposes an incomplete receiver interface; one calibrated operand partially transfers. The latest actual-writer panel shows that native attention/MLP writes cancel part of inherited state outside the supplied projector. Projecting both removes that cancellation. The design now explicitly preserves the joint receiving state, conditional writer responses and retained-parent identity. A general semantic stack and a cheap predictor for unrestricted chat remain research targets. The [original Astra review](../../artifacts/cmachine-stacked-state-review-20261105/REVIEW.md), [controlled-plane review](../../artifacts/cmachine-receiver-plane-20261105/REVIEW.md), [reuse/enrichment review](../../artifacts/cmachine-receiver-reuse-20261105/REVIEW.md), [component review](../../artifacts/cmachine-carry-read-20261105/REVIEW.md) and [actual-writer review](../../artifacts/cmachine-actual-writers-20261106/REVIEW.md) retain the iterations and critiques.

The most useful simplifying assumption is that **a recurring computation has a reusable interface to its context**. Its variables can move between physical carriers, and its memory can grow. What must remain understandable is how it obtains operands, changes state and makes the result available to the next consumer. The latest results make that interface more specific: it must include persistent memory, query-dependent addressing and the receiving state.

## November 6 query-memory component update

The [matched-history experiment](query_own_history/FINDINGS.md) now supplies the proposed own-history native-shadow feedback. With all programs/prompts/budgets/memory fixed, conditional KL rises26.46% on the compact branch's visited prefixes, entirely from cards; its shorter answer retains both correct bindings. Other contexts improve. Shared prefixes133/106/82tokens and all retained operands/tops match byte-exactly. Candidate seals precede independent shadow construction; no native state repairs the candidate. History-distribution qualification is now measured under the original schedule, not merely proposed.

The [saved reader account](query_own_history/runs/job-47cc27324445/study.json) separates Q, key projection, their interaction and aggregation at actual Q, including numerical remainders. A near-zero vector-account closure is algebraic, and local norms do not order KL or semantics. Prioritize the newer drawer native-suffix task cut if not already owned; one own-state query-response direction is a distinct cheap-program lane. Keep rank and memory fixed, signed controls and source/target seals explicit. [Astra review](query_own_history/REVIEW.md), [next plan](query_own_history/NEXT.md).

The [longer-horizon result](query_horizon/FINDINGS.md) now executes the proposed temporal cut. Future source KL selects the same dose2 image with a1.76% advantage over dose1; the extended all-clock diagnostic would choose1. All six images have the same first source disagreements, so later native-forced rows cannot establish autonomous continuation. Fresh compact answers remain identical; correct401 versus native241 makes task truth separate from native fidelity. Keep producer policy and history condition in the existing consumer/clock/horizon contract. Fixed-source feedback estimates loss on native-visited prefixes, not on the program's own future prefix distribution.

Next acquire own-history conditional feedback using an independent native shadow consuming the candidate's exact IDs, with no state/token injection. Keep B128, query image, compiled K/native V and memory fixed before another direction or memory expansion. The [connected MLP-reader result](operand_composition/FINDINGS.md) separately adds actual cheap-producer qualification; its full-memory scope differs from this compact reader. Existing stack/replay machinery does not by itself supply an accurate operator or remove dense lower receiver formation. [Executed protocol](query_horizon/EXECUTED_PROTOCOL.md), [Astra review](query_horizon/REVIEW.md), [next discriminator](query_horizon/NEXT.md).

The [native-consumer selector](query_consumer/FINDINGS.md) adds an explicit requirement to query-image provenance: source contexts, feedback consumer, scored clocks and future horizon. Actual C source KL chooses dose2 while attention response chooses1; shared headers/prompt bodies/eight-token suffix prefer1/2/0.5. All source suffix pending choices agree with greedily produced IDs for eight tokens, so the ranking difference already occurs on-policy there. Aggregate source gains trade off contexts and target effects remain mixed with unchanged answers.

Keep the six images frozen. Next isolate longer future and followup scoring on the same sources, with predeclared future-focused feedback and an all-clock control, before changing code dimensions, context coverage or memory. Existing targets cannot select a post-hoc suffix dose. The corrected summary/recovery changes two accounting field accesses only; scheduler failure and raw evidence remain. This result does not establish optimal F, lost information in B128, production stack ownership or cheap dense receiver formation. [Continuation math](query_consumer/MATH.md), [next cut](query_consumer/NEXT.md).

The [response-aware query experiment](query_response/FINDINGS.md) now separates native-query reconstruction from useful reader control. With the same B128 and memory, one source-only update improves source response loss8.36% while query error rises; all six initial attention/layer-output/logit comparisons improve.22/24 compact L2 and18/24 KL comparisons improve, but three initial KL effects worsen and every answer stays unchanged. Treat F as a consumer-specific control image, distinct from encoder B. Its useful geometry depends on keys, values, memory counts and downstream consumers; Bx is not thereby sufficient for arbitrary futures.

Keep the six frozen dose images for the next source-only native-C KL selection, count its new source forwards, and preserve candidate-owned upper caches under the original fixed source IDs. This tests consumer mismatch without larger memory or a new direction. Source forced history is not autonomous rollout. The existing image install/restore closure and whole-invocation receipt do not imply an internal C-cache rewind or production stack ownership. [Math](query_response/MATH.md), [scope/audit](query_response/REVIEW.md).

The [same-cost query refit](query_program/FINDINGS.md) now earns a source-bound conditional query-image choice in the experimental wrapper. It changes the decoder on the existing 128-coordinate receiver code, using only the original three calibration trajectories. It improves 23/24 compact full-logit boundaries across six conversations at unchanged dynamic MACs/state, with no generated-token change. Native-query controls remain distinct paid policies; their larger initial gains and later reversal rule out assuming native Q is always the best recurring transition.

Separate the receiver code Bx from the query image that interprets it. An orthogonal reconstruction is one decoder, not a sufficient-state theorem or necessarily the best predictor. The fitted decoder's ideal oblique projection preserves the code and infers source-correlated omissions; it cannot distinguish arbitrary equal-code inputs with different required queries. Native-distribution KL improves at 19/24 boundaries, with two initial countertrends despite better query/attention/logit L2. This favors a response-aware same-budget objective using the actual approximate keys/native value response before buying additional query information. Keep K/V, memory capacity, source selection and target feedback independently declared. No production stack ownership or whole-model closure follows. [Derivation](query_program/MATH.md), [qualified replay/correction](query_program/README.md).

The subsequent [query/key-versus-value cut](query_value_cut/FINDINGS.md) now qualifies independent map choices in the experimental wrapper. Restoring native Q/K or native V improves all first readouts against source128-both, with actual native probability/value pairs supporting the finite interaction account on exact common token prefixes. A local attention ranking reverses at the prose full readout. Treat QK, V, memory response and later consumption as separate interfaces whose effects must be measured jointly; neither raw V error nor one local norm is an adequate global compiler objective.

The [completed bounded-memory transfer cut](query_memory_transfer/FINDINGS.md) now qualifies native-V/compiled-QK as an explicit experimental policy alongside native-all and compiled-all controls. It improves initial compact-reader full-logit differences 14.47/16.27/6.98% → 12.08/12.71/6.29%, and all eight shared-history readouts improve under native V in both readers. The gain's magnitude weakens at four and strengthens at four; later divergent-policy countertrends remain. Native-all compact memory is still closer than the mixed compact policy at every saved readout. Keep the original source bank immutable, avoid an automatic per-token router or another rank increase, and count the paid native V map. Approximate query/key formation can still alter which native values are selected or merged.

The selected interaction `(Ptilde−P)(V_C−V_N)` now has an actual-probability and lifted-weight account on paired own prefixes. Audit memberships are not serving state. First 32 rows distinguish numerical profiles; merges add information loss thereafter. This supports a receiver/consumer-aware operand-program experiment before purchasing more moment storage. It does not change the compact-state checkpoint lifecycle below or remove dense receiver generation. [Math and costs](query_memory_transfer/MATH.md), [Astra review](query_memory_transfer/REVIEW.md).

The subsequent [saved-only Q/K decomposition](query_memory_transfer/runs/job-b94ac2f8a6c9/score-audit.json) points more specifically toward query formation in aggregate. Row-centered query terms exceed key terms in norm on each initial prompt, and CPU64 native-Q/compiled-K diagnostics improve attention more than the converse. Per-head/clock rankings remain heterogeneous; these diagnostics are not executed native repairs. Test a cheap query-response correction through its own recurring state before changing memory capacity. Native-Q/compiled-K/native-V would cost 2,981,888 selected MAC, only 5.21% below native QKV, so native Q is a paid diagnostic control rather than an assumed efficient serving choice.

Shared consumed IDs qualify the selected lower receiver here, but upper cache history remains policy-specific. The actual interaction identity `(A_C−A_N)(V_C−V_N)` belongs to attention on paired operands/probabilities; later logits are operational policy interactions. The completed panel's later countertrends and unchanged wrong arithmetic prohibit a general accuracy or uniform-recovery claim. Dense receiver generation and the compact-state checkpoint lifecycle remain separate missing computations.

The [completed query-memory experiments](query_memory/FINDINGS.md) add an executable future-query response boundary. One original Qwen attention component forms its own Q/K/V on the branch's actual normalized receiver, appends a bounded recent/prototype memory and returns attention through the recomputed native MLP and later suffix. Source32→128expansion improves initial attention and logits; a checkpoint-only compiler and later own-state effects remain heterogeneous. Read/query formation errors, memory approximation, semantic usefulness and whole-model cost are separate axes. Do not infer full closure from exact replay of the selected approximate law.

The compact memory's update mutates old prototype sums/counts. Cursor-only rollback over immutable dense prefixes cannot restore it. Before serving integration, give the component explicit allocation, append/read, checkpoint and restore ownership, bound to the existing model/profile/parent lineage. The simplest correct first implementation copies its98,464-byte numerical state only at a token-boundary checkpoint and restores it with the native marker. Sixty-four full snapshots would cost6,301,696host bytes; that optional cost must be counted. An undo journal is a later measured optimization, not a prerequisite for discovery. Until that lifecycle exists, the isolated experiment makes no compact-state stack claim.

The current wrapper still allocates the replaced full selectedKV arena. A later component allocator should omit that arena only when no native reader can reach it, retain the ordinary dense policy and validate exact lifecycle/replay mechanics. Actual lower-layer receiver formation, other-layer state, O/MLP and readout remain paid. A compact whole-model successor must generate the receiver without a native refresh and preserve query-selected memory content. The missing covariance/dispersion terms in [MATH.md](query_memory/MATH.md) nominate bounded corrections; none is yet a validated repair.

## November 6 design update after actual writer intervention

The [completed four-policy panel](FINDINGS.md#actual-writer-projections-remove-a-native-cancellation) changes actual attention/MLP15 writes and recomputes the receiving MLP and every later consumer. Native outside-P writes oppose inherited state with cosine −0.7232, reducing its norm from 18.8595 to 13.7545. Both projected writes remove that cancellation. The policy interaction at layer16 input factors as −(I−P)(mA−m0), with explicit finite remainders. P is supplied; this is not a discovered semantic operation or a law for later logits.

### Keep projected coordinates as a view of complete state

For C=I−P, the ideal joint update is

$$
Py=Px+Pa+Pm(x+a),\qquad Cy=Cx+Ca+Cm(x+a).
$$

The MLP reads the combined receiver x+a. Its normalization and gate depend on both projected and omitted components. Retaining only coordinates in P, or independently zeroing each writer's complement, changes this computation. Native execution already keeps the full residual, RMS scale and K/V; no extra complement tensor is needed in ordinary serving. Diagnostic coordinates and omissions are computed on demand. Any future compact state must demonstrate adequate continuation from its own emitted state rather than assume this decomposition is closed.

### Bind an intervention to the retained parent

Clock position does not identify context. After rollback, another history can reach the same session/epoch/position. The [parent-lineage update](PARENT_LINEAGE.md) adds a boundary identity saved/restored with checkpoints and stack frames, and binds `QwenWriter` to that identity. Actual parent rollback permits replay; rebuilding another parent at the same position does not. This adds 512 host bytes across 64 markers and no GPU storage or model operations. Cross-context `QwenSubspace` reuse intentionally retains its existing model/layer contract; `QwenPlane` retains exact live-input/read guards.

### Measure each candidate from its own baseline

The [new own-state MLP result](../../obsidian/blog/2026-11-06-111533-improving-an-mlp-on-its-own-predicted-states.md) improves response while worsening baseline fidelity. For an approximate program with baseline P0 and edited result P1, retain native N0/N1 and report

$$
e_0=P_0-N_0,\quad e_{\Delta}=(P_1-P_0)-(N_1-N_0),\quad e_1=e_0+e_{\Delta}.
$$

Endpoint, baseline and response errors have separate meanings; their powers include a signed cross term. Do not substitute N0 for the approximate program's own baseline. The current writer factorial consists of intentional native interventions with exact native00, so its observed policy differences are not errors of a new approximate baseline program.

### Optimize only established dependencies

The [new complete masked-encoder image result](../../time-circuit-explorer/docs/complete-masked-encoder-image.md) removes unread padding K/V while retaining padding queries and residual/MLP rows. Its two fresh prompt trajectories and images match retained references, but the measured candidate encoder is slower. Our causal decoder has no equivalent declared unread-key class. Do not import that elimination rule without establishing its support here.

The [new FLUX carry/flow cut](../../obsidian/blog/2026-11-06-003759-from-exact-readers-to-causal-computation-families.md) also rejects universal carry dominance: flow alignment exceeds carry in all eight tested recipient/consumer comparisons, while both contribute to the outcome. Retain conditional effects at each consumer. Preserve native numerical laws, Qwen2.5 bias terms and RMSNorm behavior; analytical parity or LayerNorm invariance from another architecture is not an interchangeable kernel.

These changes require one small runtime ownership guard and clearer experiment contracts. They do not require a generic router, another dense state copy, a new semantic stack or automatic projection of native writes. The historical ordered plan below remains the development path, with completed stages recorded in the findings.

### Compose according to dependency, then distinguish history from execution error

The newer [C composition findings](composition/FINDINGS.md) and [autonomous-chat account](../../obsidian/blog/2026-11-06-115221-how-we-get-from-state-prediction-to-autonomous-chat.md) establish that recurring B3 execution already advances its own tokens and full memory. A new generic chat loop is unnecessary. Correct short answers also survive whole-write deletion, and 14/15 later shared-input approximate writes are farther from dense than zero, so the next consumer must discriminate useful replacement content.

For this post-KV16 MLP cut, partition state into lower KV through16 and upper KV/output. On identical consumed histories, the lower transition and receiver can remain shared while upper state follows its own writer policy. The study directly checks captured layer16 operands and KV16, not every lower array. Token divergence feeds upper differences back into the lower computation. Reuse must bind the complete relevant dependency history, operands and numerical profile; equal current tokens or token counts are insufficient. No new memoizer is justified by these cross-branch repeats alone.

Seal each candidate's own trajectory before diagnostics. Keep native own-history continuation, native rebuild on candidate consumed token IDs, native one-step execution on candidate cache/position/pending token, and teacher-forced candidate replay separately labeled. “Same text” means exact consumed IDs including user/header/special tokens and consumption status; decoded/re-tokenized text or a delivered-but-unconsumed token is not an equivalent cache history. Compare at declared aligned clocks and shapes. Exact fallback on branch-owned operands remains autonomous and paid; it does not repair earlier cache error. Native recovery needs a complete checkpoint and explicit replay on that consumed history.

The [source-only gate-law proposal](../experiments/2026-11-05-operational-variable-closure/PLAN-AUTONOMOUS-CHAT.md) is now implemented as a distinct `QwenGateLaw` API and [qualified in the C FP32 profile](gate_law/FINDINGS.md). Dense projections and full state remain. Anchor bytes, coefficient precision and evaluation order are frozen; future target refresh and pre-seal exact selected-site diagnostics are absent. Reusable law identity binds model/layer/source/coefficient bytes, and each application binds its actual parent. This is a separate program from B3, with no automatic serving policy.

The two-turn result sharpens the next cut: all 93 evaluated law writes are closer to their same-input native write than zero, but recurring follow-up logit differences reach 4.108%/10.391%. A native step on actual candidate cache separates current-law from accumulated-state effects. Once has zero current-step defect and retains historical error. Exact fallback therefore cannot silently certify recovery. All policies preserve the tested colors; native arithmetic itself is wrong, and deletion improves one own-answer subtraction consistency measure while losing model fidelity. Keep task truth and numerical fidelity separate.

A cheap forward-work reduction next needs a composed gate/up/write program that forms its own consequential operands. The new scalar law alone leaves all 197 dense matvecs per token. Use the existing B3/complement account to diagnose which omitted operand must be represented; retain full native context until the candidate demonstrates its own transition closure. Attention response over future queries, all-layer cache updates and vocabulary readout remain separate costs and interfaces. [Full computation comparison and next steps](gate_law/FINDINGS.md#what-forward-pass-free-emulation-requires-next).

### Test state closure under writes, and qualify the future reader

The [new nonlinear quotient experiment](state_quotient/FINDINGS.md) now executes a constructive control: two coordinates plus invariant omitted energy exactly reproduce16native updates on a supplied8-state organism. One escaping write changes omitted energy through2eᵀVw, exposing the discarded sign to the next normalization and subsequent output. Retaining one more coordinate restores byte parity. The finite contract preserves the supplied RMS reduction leaves; this is not arbitraryFP32orientation equivalence or globally minimal state. An exact original Qwen down-weight minor proves rank≥256, making a tiny common linear read/write space an unsupported default while leaving other representations open.

The [completed attention-read trace](composition/attention_read/FINDINGS.md) separately qualifies fixed-query numerator/denominator updates to2.81e−7–6.45e−7relative native context-change discrepancy. Softmax redistributes interaction outside changed score support, so retain normalization mass and value/allocation coupling. A reusable query-conditioned response family must cover the queries the branch itself produces; a supplied current-query accumulator is insufficient evidence of that closure.

Keep two design lanes explicit. Exact compilation narrows the admitted domain until reads and writes close; useful approximation retains a declared domain and measures its own baseline, response and actual consumer. The [selected-only image reader](../../time-circuit-explorer/docs/selected-reader-image.md) illustrates the latter:88.93%comparable numerical reader-state reduction with useful objects/spatial swap retained, but changed outputs/full K/Vscan/learned suffix. Do not require exact global invisibility as a prerequisite for exploratory approximate programs, and do not promote exact execution of an inadequate mean state as faithful model emulation. No production runtime change is justified by the controlled quotient alone.

Implementation update, November 5: the strict C/client stack and selected operation observations now have [bounded Beast results](FINDINGS.md#nested-continuations-and-selected-operation-ports). Three first-readout arrays and own-token sequences match the preserved earlier C binary exactly; real-model rollback reproduces valid KV and full logits. Durable source-frame/StateCut integration and the semantic-law stages remain open. The implementation keeps complete token checkpoints and replay rather than arbitrary interior restoration.

The next implemented increment is an [explicit nonlinear receiver-plane program](PLANE_EXPERIMENT.md), with [completed real-model observations](FINDINGS.md#compiled-nonlinear-receiver-plane-and-its-continuing-state). It compiles the gate/up input projections on three raw operands while retaining the actual RMS reduction, SiLU, down projection, residual skip and later dense computation. Six compiled doses reproduce dense MLP writes within 6.83e-7 relative error and preserve own generated IDs. This tests an executable subgraph bypass without assuming a global quadratic output law. New [OVC8 findings](../experiments/2026-11-05-operational-variable-closure/FINDINGS-OVC8.md) reinforce the distinction: making the final four layers exact improves the earlier polynomial forecast modestly, while a larger receiving-state discrepancy remains before H24. They do not identify which primitive is responsible or establish this C program's performance.

### The next reuse cut, proposed with Astra

Status update: this historical proposal is now executed by `job-96e7f0b9e3ae`, followed by the prospectively frozen one-operand enrichment in `job-c034107916a2`. Their [findings](FINDINGS.md#reusing-a-frozen-basis-on-natural-later-receivers) preserve substantial write errors despite correct short answers. The derivation and original proposed boundary below remain useful context; the next unexecuted cut is component correction.

After qualifying the controlled plane, test the next naturally generated token at the same layer using only the preceding token's frozen operands. Keep the exact guarded receiver API unchanged; approximation across clocks needs a separately labeled operation.

Let \(B=[x_0,a_0]\), from the preceding layer input and attention write. On the next token's actual raw receiver \(r\), use a stable small least-squares solve:

\[
c=B^\dagger r,\qquad e=r-Bc.
\]

Retain the actual residual skip and its actual native RMS scale \(\rho\). The existing frozen numerator vectors give

\[
\widehat g=\rho G\operatorname{diag}(\gamma)Bc,\qquad
\widehat u=\rho U\operatorname{diag}(\gamma)Bc,\qquad
\widehat m=D[\operatorname{SiLU}(\widehat g)\odot\widehat u].
\]

No new target-dependent gate/up compilation is permitted. Report the basis rank and conditioning, omitted operand \(e\), actual write error, higher-layer KV and later consumer effects. Residual norm is a diagnostic, not a scientific stop gate. A large residual can be irrelevant to a particular consumer, and a small one can remove decisive evidence.

Use three arms from the same native parent: dense computation; the frozen-subspace candidate; and a diagnostic correction adding \(\delta g=\rho G\operatorname{diag}(\gamma)e\), \(\delta u=\rho U\operatorname{diag}(\gamma)e\) before the exact nonlinear operations. The correction spends dense work and cannot count as a saving. It separates omitted-input effects from arithmetic reassociation. Preserve the candidate's own continuation even if agreement is poor. This first clock-to-clock reuse test would establish more than controlled interpolation, while still making no semantic-frame or general-chat claim.

Astra's interpretation review refined the first target to the final prompt token of a new collateral-question paraphrase after the model's own first answer. Reusing a program to predict another EOS would be a weaker consumer test. A naturally formed receiver for “Tell me Niko's badge color, just the color” changes the context and requested binding while keeping the exact native template and later own-state answer. The basis remains frozen from the original Mira question. This receiver and its Mira counterpart are now measured separately from the original nineteen-branch panel.

The current common-scale control has a complementary purpose. For positive \(\lambda\) and negligible epsilon, \(m(\lambda r)\approx m(r)\), but the complete layer update changes by approximately \((\lambda-1)r\). An invariant MLP branch can therefore coexist with changed higher-layer KV and later answers. The residual skip must stay in the computation account.

This also explains how the change becomes an address or value at the next layer. Write \(y=r+m(r)\). Ignoring epsilon only for this derivation, the next normalization has Jacobian

\[
J_N(y)=\rho_y\operatorname{diag}(\gamma_y)
\left(I-\frac{yy^\top}{\|y\|^2}\right).
\]

When the MLP write stays fixed under positive receiver scaling, \(dy/d\lambda\approx r\). The change in the next normalized input is consequently

\[
\frac{dN(\lambda r+m(r))}{d\lambda}\bigg|_{\lambda=1}
\approx \rho_y\operatorname{diag}(\gamma_y)P_y^\perp r
=-\rho_y\operatorname{diag}(\gamma_y)P_y^\perp m(r).
\]

Thus changing the carry magnitude changes its balance with a nonparallel MLP write, even when the write itself is invariant. The next key/value projections can consume that directional change. If carry and write are parallel, this particular first-order normalized effect vanishes in the zero-epsilon idealization; nonzero effects are not guaranteed for every state. The actual experiment retains epsilon and native arithmetic throughout. This is a derived mechanism to compare with the measured state trajectory, not an empirical claim before the run.

There is also a precise clock prediction, recorded while `job-b18afb3e1bbc` is queued. Editing token t's MLP16 can change that token's K/V beginning at layer17. The common next token t+1 traverses unchanged lower-layer memory, so its incoming layer17 residual and freshly formed K/V17 should remain identical. Layer17 attention then reads the changed older memory; its consequences can enter the next token's K/V18 and above. The panel's captured layers17–27 can test this one-layer shift of the continuation frontier. They do not directly observe lower-layer current-token rows. The [offline layerwise analyzer](tools/plane_frontier.py) preserves each key/value comparison as an observation rather than turning this prediction into a scientific stop gate.

### Completed: correct carry and read separately

The source B2 plane incurs 72–83% write errors on later natural receivers. One explicitly compiled Mira calibration operand improves write error 74.6%→61.8% and 81.0%→60.7% on two fresh Niko wordings. Yet the second maximum full-logit error worsens 1.928→2.586. Correct supplied-color answers conceal these differences. Residual correction recovers approximately 6e-7 write fidelity, so omitted computation, rather than arithmetic alone, remains the main local gap.

The [posthoc component analysis](results/job-c034107916a2/receiver-components-v2.json) projects actual captured entry and attention writes against B3. The remaining carry has signed energy shares 69.49%/66.45%; the carry/read cross terms are negative. These are geometric attributions, not semantic roles or causal importance. Blindly appending directions would fit these contexts without resolving that distinction.

The [five-arm protocol](CARRY_READ_EXPERIMENT.md) is completed by `job-00038c17cfcd`. On the new question, carry correction removes 58.8% of full-logit error norm and 36.7% of next-KV error norm. Read correction barely helps the local write and worsens those downstream errors. The executed joint write interaction is 19.13% of native-write norm; the separate split/full discrepancy is 4.55e-7. Correct decoded colors remain insensitive. This supports a consequential omitted carried operand on this query, with substantial carry/read coupling, not a general carry-only mechanism.

### Completed: distinguish inherited carry from newly added writes

Completion update: job`job-0e69e4a9b62e` now executes this formation cut. The two operands strongly oppose each other (cosine−0.7675); isolated corrections worsen native error while both reduce MLP-write error54.2%→43.7%. Their interaction/joint ratio is148.6%locally and471.0%at logits. This supports retaining coupled formation rather than selecting a single producer. The [next native-writer protocol](WRITER_INTERVENTION_EXPERIMENT.md) edits actual layer15 writes and lets its MLP and every later receiver evolve. That next API/panel remains proposed; the earlier-layer ports and explicit correction API are now implemented and natively exercised.

Consumer-analysis update, November6: the [implemented analyzer](tools/consumer_contrasts.py) now follows the operational four-arm contrast through49retained consumers. Interaction/full-correction ratios are29.99%at the MLP write,53.76%at full logits and64.24%at common-next upper K/V. Read correction alone worsens downstream fidelity, but its conditional addition after carry closes the remaining numerical error. These are separate port denominators, not amplification or semantic necessity. The [holistic review](../../artifacts/cmachine-holistic-review-20261106/REVIEW.md) applies the latest FLUX consumer/routing and M5g generalization lessons; the [formation protocol](CARRY_FORMATION_EXPERIMENT.md) now requires conditional effects and interactions at each consumer before selecting an upstream cut.

The [formation protocol](CARRY_FORMATION_EXPERIMENT.md), refined with Astra, first captures actual layer-15 entry, attention write, post-attention receiver, MLP write and output. Split layer-16's omitted carry into the projected inherited layer-15 input and a complementary new-write operand, retaining exact addition/projection discrepancies. Use five new common-parent arms: dense, B3, inherited-only correction, new-write-only correction and full-carried correction. The final arm still omits the layer-16 read complement; it is a decomposition reference rather than exact-native execution.

The original decision rule was to split attention15 from MLP15 if new-write correction explained the effect, move earlier if inherited carry dominated, and retain interactions when coupled. The completed panel supports the coupled branch. The ports and this panel are implemented; the next unexecuted cut is the native-writer policy factorial, which recomputes all actual receiving operations rather than further splitting a fixed correction input.

## What the recent work establishes

| Work | What it gives us | Consequence for the machine |
|---|---|---|
| [Complete state and context mathematics](../../obsidian/blog/2026-11-05-153154-state-context-and-the-mathematics-of-time-circuits.md) and the [speculative algebra](../../obsidian/blog/2026-11-05-161604-a-speculative-algebra-of-model-computation.md) | A complete transition reference and the hypothesis of reusable operational variables | Preserve the full state while testing smaller descriptions against it |
| [C implementation and results](FINDINGS.md) | Sixteen matching ISA committed states, five actual anchored MLP substitutions, and three independent Qwen prefill/generation windows matching native FP32 tokens | Use this executor as the reference instrument; ordinary Qwen runs the complete equations and the explicit receiver-plane experiment separately bypasses two selected projections |
| [Latest learned x86 machine](../../obsidian/blog/2026-11-04-160430-the-learned-x86-machine-where-it-stands-and-what-is-next.md), latest M5g sections | The separate M5g record machine reaches96/96development programs but ties M5f at7/12fresh freestanding completion; unseen glibc coverage improves. Literal byte-key records miss new familiar-field combinations | Recover reusable operand formation and applicability across combinations; borrow typed atomic effects while distinguishing this record learner from the neural R3 specimen ported to C |
| [Operational-variable follow-ups](../../obsidian/blog/2026-11-05-161109-the-one-assumption-that-would-make-model-computation-easier-to-understand.md) | Identical selected source rows can produce different answers; the missing distinction survives in earlier memory writes | A readable residual chart is insufficient as a frame's entire state |
| [OVC9 factored reader](../experiments/2026-11-05-operational-variable-closure/FINDINGS-OVC9.md) and [OVC10 local normalization](../experiments/2026-11-05-operational-variable-closure/FINDINGS-OVC10.md) | Same-memory full-axial margin error stays near0.861logits with composed local polynomial primitives, versus about4.982 for the same-tail global quadratic | Preserve finite own-state composition and current products; a global degree-two requirement can discard useful structure. Dense weights/context anchors remain supplied, and native-framework arithmetic does not qualify C matvec parity |
| [Memory-band mediation](../experiments/2026-11-05-operational-variable-closure/FINDINGS-OVC5.md) and [K/V mediation](../experiments/2026-11-05-operational-variable-closure/FINDINGS-OVC6.md) | Edited KV16–23 determines all tested target-color flips; values alone flip 13/16, keys alone 0/16, both 16/16 | Trace the writer, persistent payload, address and later query reader together; this band is a starting address for this specimen, not a universal register file |
| [Diffusion from initial noise and its fresh four-context factorial](../../obsidian/blog/2026-11-05-140315-predicting-a-diffusion-image-from-its-initial-noise.md) | Complete own-state 112-transition forecasts produce actual VAE images before native denoising, using original weights and supplied prompt encodings; fresh-context latent errors range 1.923–11.018% | Long numerical closure is a real achievement; retain context-dependent error and distinguish dense equation execution from a cheaper recovered semantic program |
| [Following attention and the MLP](../../obsidian/blog/2026-11-05-153508-following-state-through-attention-and-the-mlp.md) and [stream interactions](../../obsidian/blog/2026-11-05-162943-where-image-and-text-computations-mix.md) | Cross-stream reads, normalization competition and receiving MLPs explain state-dependent interactions | A frame's returned value needs a declared consumer and context; additive response composition can miss interactions |
| [Following state changes](../../obsidian/blog/2026-11-05-172526-discovering-model-programs-by-following-state-changes.md) | Exact reuse of invariant operations works on two known FLUX states; close endpoint predictions can still miss fine responses badly | First establish a qualified numerical baseline, then replace one affected boundary and measure the response through its consumer |
| [Open-chat prediction studies](../../obsidian/blog/2026-11-04-093740-chat-is-not-a-template-the-switch-on-qwens-own-replies.md) and [own-state prediction chains](../../obsidian/blog/2026-11-05-050409-what-one-qwen-column-reads-from-memory-and-what-a-sealed-16-token-chain-does-when-that-read-is-cut-short.md) | Token tables lose contextual computation; sparse reads still touch broad memory over time; local MLP expansions do not yet save work | Keep full memory and dense fallbacks while discovering which live contextual reads actually form the next write |

These results fit one account: a source distinction is formed into operands, transformed into persistent state, then read under a later context. The variable's meaning depends on that path. The [time-circuit frontier work](../../obsidian/blog/2026-10-08-111437-time-circuits-as-state-machines-and-frontier-reduction.md) already showed why repairing an immediate answer can leave its future memory wrong. The new binding experiments identify the same problem with stronger source-row equality controls.

The ISA work supplies an unusually concrete precedent. M5f composes learned records with byte fields, addressed operands, effects and applicability conditions; its execution can be checked against a declared ISA specification. Its updated development/known-set learned-only result is 90/102 programs, while the fresh4 one-look panel completes only 2/12 standalone and 6/12 with the proved fallback, with no wrong committed step. Better local coverage does not guarantee a complete new program. This is a separate system from the neural R3 Mamba model used in our C port. Qwen does not come with a comparable complete specification of correct conversation, so ISA's proof machinery cannot simply become a semantic gate for chat discovery.

The diffusion results also prevent a misleading interpretation of prediction error. In the newest stream factorial, whole-state errors around 0.28% for image and 0.09–0.11% for text coexisted with roughly 81–133% errors on small effects. A missing fine response can come from numerical execution as well as omitted computational variables. We must distinguish those before enlarging a semantic theory.

The full-image result is stronger in horizon: four fresh prompt conditions each advance through their own complete 112-transition denoiser forecast. Their visible color and placement patterns survive, alongside an 11.018% latent-error case and a maximum local RGB-channel error of 196. These forecasts still execute learned-weight arithmetic and receive native prompt encodings. They establish an addressable long computation, rather than a semantic compiler or a speedup.

That distinction explains where chat stands. Full equations and complete state already generate chat; the C port makes that computation independently executable. A cheap law that skips most of those equations faces a harder contextual transition. In the audited Qwen reuse study, the preceding column's middle-layer MLP input is about 0.8 of the current input norm away. Consecutive tokens do not supply the nearby receiver that a local response approximation needs. Each chosen token also changes the future memory and can cross a discrete decision boundary. The useful transfer from diffusion is its discipline of complete own-state continuation, response-level tests and explicit numerical contracts.

## The mathematical target is a sufficient interface

Bind the checkpoint, arithmetic schedule, tokenizer, clock and state schema in an execution contract \(\alpha\). A native transition is

\[
S_{q+1}=T_{q,\alpha}(S_q,u_q),\qquad y_q=O_{q,\alpha}(S_q).
\]

At a Qwen token boundary, the neural state contains the complete valid all-layer K/V prefix, position and pending token. Our greedy runtime also needs its pending-token delivery policy. The application adds its transcript, processed token IDs and source scope. A layer-interior cut additionally needs the residual, phase and which layers have written the current token's K/V. A layer number and a final residual alone do not close that state.

A proposed interface \(z_q=C_q(S_q;\Gamma)\) is qualified by a context contract \(\Gamma\): allowed operations, future query family, observations and horizon. We seek

\[
C_{q+1}(T_q(S,u);\Gamma')\approx f_q(C_q(S;\Gamma),u),
\qquad O_q(S)\approx g_q(C_q(S;\Gamma)).
\]

If the update reads an additional context function \(\rho(S,u)\), that function is part of the interface and its cost. Calling an arbitrary native suffix to compute \(\rho\) would conceal the computation we are trying to explain.

A useful intermediate account keeps complete backing memory while exposing the operation:

\[
r_q=R_q(M_q,Q_q(z_q,u_q)),\qquad
(z_{q+1},w_q)=f_q(z_q,r_q,u_q),\qquad
M_{q+1}=U_q(M_q,w_q).
\]

Here \(Q\) forms the query, \(R\) reads context, \(f\) combines operands and \(U\) commits the write. Merely renaming the whole transformer with these letters would explain nothing. Each function needs declared physical operands and an inspectable law, and any efficiency claim must count its implementation.

One way to state exact sufficiency is future equivalence. Two states may share an interface only if every allowed future input sequence produces the same declared observations. With a continuation-closed input family, appending one allowed input must preserve that equivalence. This is a requirement on the proposed representation, not a claim that we have found a small representation for general chat.

The operational-variable experiments give a concrete counterexample to one candidate: even bit-identical complete selected source rows leave different later answer margins. Therefore no single-valued function of just those rows and the same query can reproduce that panel. The next refinement belongs in the writer–memory–reader interface, rather than in a more complicated decoder of the same insufficient rows.

We should loosen demands for a tiny fixed vector, one permanent subspace or globally linear dynamics. We should tighten demands for explicit live operands, persistent writes, source provenance, context validity and continuation from the proposed state itself.

## Three meanings of stacked state

**Layer composition already exists.** The executor composes the model's attention, normalization, MLP and residual operations through every layer and token. For a layer, an abbreviated real-arithmetic account is

\[
a_\ell=W^O_\ell\operatorname{Attn}(q_\ell,K_\ell,V_\ell),\quad
r_\ell=x_\ell+a_\ell,\quad
x_{\ell+1}=r_\ell+m_\ell(\operatorname{RMSNorm}(r_\ell)).
\]

Each layer writes the current token's K/V before its attention read. Its MLP changes the residual from which higher layers form their K/V; it does not rewrite that layer's already committed K/V. This timing is essential when identifying a writer.

**Nested continuations are implemented as an engineering stack.** A frame saves where a computation can resume. A child advances from that parent, and the controller can retain the child or discard its suffix and return to the parent. This is an execution stack with exact ownership rules.

**Reusable semantic frames are the scientific target.** A frame would represent a binding, subproblem or intermediate operation through an interface that its later consumers can use across contexts. A host-supplied stack does not establish that the model has discovered calls, returns, recursion or local variables. Those behaviors need their own tests.

### The smallest useful continuation stack

Keep one resident model, one flat K/V arena and a vector of ancestor markers. At a valid token boundary, a marker records:

```text
model and numerical-contract identity, session/reset epoch
valid KV cursor, pending token, delivery policy, valid readout metadata
parent marker, transcript/token-log boundary, source-frame reference
```

Use three operations:

- `push`: retain the current boundary and enter a child.
- `rollback`: restore the top retained boundary and discard that child's executable suffix and metadata.
- `commit`: remove the top retained boundary while keeping the current child's state.

The existing Qwen checkpoint's latest-parent restriction protects against using a cursor whose required suffix has been overwritten. A nested stack can retain multiple ancestors safely because every live marker names a prefix that remains immutable. Rewinding discards all dependent descendants. Token position alone is not a sufficient handle; session identity, reset epoch and ancestor ownership must agree.

No K/V copying or paging is required for this strict last-in, first-out case. Every child writes at positions after its parent. Discarding a child invalidates and reuses those positions. The preallocated arena stays allocated; logical reclamation does not reduce its physical allocation. The ISA backend can use small complete state copies instead of imposing Qwen's cursor representation on its recurrent state.

This contract forbids in-place edits to a retained ancestor's K/V. Reject unsupported prefix edits explicitly. A counterfactual prefix-memory intervention needs private branch storage or restoration from a complete immutable StateCut under a separate ownership contract. The first C write intervention should modify the current residual and newly formed suffix, leaving retained ancestors untouched. The original OVC memory-interchange assay can continue through its existing complete native StateCuts.

The client must restore its transcript, processed-token boundary, pending-delivery policy and source scope with the model cursor. Restoring only K/V would allow a plausible-looking but inconsistent conversation. Appends must preserve the existing token-prefix check and capacity preflight. Historical output and evidence remain recorded even when their branch is abandoned.

Committing child tokens leaves them readable by attention even if their source names leave scope. Scope metadata does not provide neural isolation. Rollback also cannot retract already displayed text or undo a tool's external effects; the execution log and external-result records remain append-only.

For this Qwen FP32 profile,

\[
B_{KV/token}=2Lh_{KV}d_hb
=2\cdot28\cdot2\cdot128\cdot4
=57{,}344\text{ bytes}=56\text{ KiB}.
\]

A 4,096-token arena therefore contains 224 MiB of K/V. Nested markers add small host metadata, rather than another 224 MiB per frame. The resident FP32 parameter bank remains 6,174,857,216 bytes. The current single-parent checkpoint already avoids K/V copies; the stack adds nesting without introducing cache duplication. It does not remove the model's matrix work or extend its context limit.

Retaining arbitrary siblings, reopening discarded branches without replay, or editing old prefixes requires additional storage or reconstruction. Defer copy-on-write pages until a measured workload needs that functionality. For now, replay a retained token branch from its nearest compatible ancestor and count the work.

### Reuse Saturn's existing source stack

[SourceFrameStack](../docs/SOURCE-FRAME-STACK.md) already provides lexical scope, shadowing, persistent ancestry and links to complete StateCuts. Its [bounded native Qwen assay](../experiments/2026-09-30-source-frame-stack/FINDINGS.md) includes own-history child continuation, root rewind, siblings and fresh-process replay. Source selection was supplied by the host.

That stack changes source visibility; a native adapter separately restores executable state. Connect the C continuation handle to its existing source-frame reference. Reuse the [public backend and DebuggerAdapter seam](../../saturn-pub/packages/mdb/docs/BACKENDS.md). A second general frame store, graph database or scheduler is unnecessary.

An external fact binding affects a neural answer only when it enters the computation: through explicit input tokens or a qualified native state writer. Adding metadata to a source frame alone does not teach Qwen the fact.

## How frame memory can compose mathematically

For one query and head, split an actual history into disjoint memory spans \(M_i\). With masked, positioned scores \(s_j=q^Tk_j/\sqrt{d_h}\), each span offers a stable softmax read interface:

\[
m_i=\max_{j\in M_i}s_j,\quad
z_i=\sum_{j\in M_i}e^{s_j-m_i},\quad
n_i=\sum_{j\in M_i}e^{s_j-m_i}v_j.
\]

For nonempty visible spans, combine them by

\[
m=\max_i m_i,\quad
z=\sum_i e^{m_i-m}z_i,\quad
n=\sum_i e^{m_i-m}n_i,\qquad A=n/z.
\]

An empty or fully masked span contributes zero mass and numerator. The complete read must have at least one visible source; an entirely masked read needs an explicit refusal or other declared behavior. This is an exact real-arithmetic composition law. A native implementation needs a declared reduction order and measured arithmetic contract.

This makes a useful stacked-memory interface possible: a parent and child contribute payload and normalization mass to the same current query. Retaining only each frame's normalized answer loses how strongly it competes with the others. If an omitted nonempty span has fraction \(\beta\) of the full softmax mass, with a nonempty retained set, then

\[
A_{full}=(1-\beta)A_{kept}+\beta A_{omitted},\qquad
A_{full}-A_{kept}=\beta(A_{omitted}-A_{kept}).
\]

Small attention mass is helpful only relative to the payload difference and the later consumer's sensitivity.

The tuples depend on the current query. Obtaining them still reads the frame's relevant keys and values unless we establish another representation. This is a compositional account of a read, with no automatic storage or compute saving. It combines disjoint spans from one valid causal history. It does not combine alternative sibling futures or make independently prefetched contexts interchangeable.

Implement the tuples as an optional diagnostic over selected frame spans first. Leave the current attention kernel unchanged until a separate numerical and cost comparison justifies replacing it.

In a decoder, higher-layer keys and values depend on the earlier context that formed them. Changing that context generally requires recomputing affected descendants. Rotating positional keys alone cannot repair altered contextual values and queries. Sequential state transitions compose as \(T_v\circ T_u\); adding activation differences is not an equivalent general composition rule.

## Tie the stack to the MLP work

For a SwiGLU MLP with normalized receiver \(x\),

\[
m(x)=\sum_i d_i\,\sigma(g_i^Tx)(g_i^Tx)(u_i^Tx).
\]

The fixed weights specify projections and output directions. The receiving state supplies two operands and a smooth coefficient per term. An attention read changes that receiver, its normalization and potentially all three factors. A useful semantic frame must therefore explain how its stored information forms the receiving operands, not only which output direction correlates with its label.

For \(a=\operatorname{SiLU}(Gx)\), \(b=Ux\), the finite branch response is

\[
\Delta m=D[a\odot\Delta b+b\odot\Delta a+\Delta a\odot\Delta b].
\]

The interaction survives even before accounting for changed normalization and numerical roundings. The current ISA quadratic law demonstrates a real dense-row bypass on one nominated anchored plane. The Qwen studies show that middle-layer MLP responses are smooth but can have substantial rank; the successful midpoint-Jacobian predictor costs roughly five to six native blocks. Neither result supports deploying a general cheap Qwen MLP replacement yet.

Treat a recovered law as an optional operation entry: a live operand resolver, receiving conditions, transition rule, read/write footprint, downstream consumer, numerical contract and counterexamples. The dense equation remains the reference implementation. A selector that first evaluates the whole replaced block has not saved that block.

## Accurate free-form chat has three separate requirements

**Native fidelity:** reproduce the chosen model and arithmetic policy from its own context. The current C engine has encouraging bounded evidence, not broad qualification across conversation. Expand the independent multi-turn panel and localize the first numerical disagreement at real operation boundaries.

**Stateful prediction:** replace some computation with a reusable account that writes the next memory as well as producing the next token. Test subsequent tokens and user turns from that predicted memory. Matching teacher-forced answers or copying a current residual does not establish this.

**Answer correctness:** give correct arithmetic, recall the intended fact, follow the request and handle missing or conflicting evidence sensibly. Our native FP32 model and C both answered 928 to `37 times 24`; the correct answer is 888. An exact continuation stack preserves that mistake unless some additional computation changes it.

Keep three visible arms: plain native execution; the same model with an explicit task/memory controller; and, where appropriate, the controller with declared tools such as an arithmetic evaluator or retrieval from supplied sources. Native outputs remain intact. Tool results are ordinary declared inputs with provenance, never hidden repairs to the baseline. Open explanation quality needs separate human assessment; exact wording match is not a correctness score.

A child task can either commit its whole generated history or return an explicit result. For a compact return, restore the caller and append a visible result record, then let the parent model encode it. This defines a different, explicitly controlled context. Merging the child's raw K/V into the caller is not a valid general return operation. A model-generated call/return policy should be compared with host-supplied control before attributing stack use to the model.

Approximate execution also needs a decision-aware error account. If logit error is bounded by \(\epsilon\) in the infinity norm, a top-two margin above \(2\epsilon\) suffices to preserve the greedy decision. An observed average error is not such a bound, and a candidate margin alone does not certify safety. Small errors near a tie can change the emitted token and every later context.

Returning to dense computation after consuming approximate history does not restore the original trajectory. A baseline-preserving fallback must recompute from a compatible exact ancestor, or retain an exact shadow and count its cost. During discovery, keep the approximate branch as evidence even when it diverges.

## Ordered implementation and research plan

### 1. Add strict nested continuations

Extend the current one-parent mechanism with ancestor markers and `push`, `rollback`, `commit`. Synchronize the token log, transcript and source-frame cursor. Keep greedy decoding, one session, the existing flat arena and explicit limits. Add no branch pager or automatic memory summary.

Mechanics checks should cover three nesting levels, same-position pushes, nested rollback followed by a new child, committing an inner frame then rolling back its outer frame, delivered versus undelivered pending tokens, reset invalidation, foreign handles, capacity rejection and transcript agreement. A restored top-two prediction remains valid for greedy delivery; overwritten full logits or scratch values must be marked invalid and recomputed by replay before inspection. On Beast, compare complete same-C readout bytes and generated IDs against replay from the same parent. These are execution correctness requirements, separate from model answer quality.

Deliverable: a cheap stack whose restored state actually resumes the declared conversation. The ISA backend uses its complete state-copy path; it does not gain x86 stack-memory semantics from this controller.

### 2. Expose a few genuine operation boundaries

Refactor `qwen_push` into begin-token, advance-layer and finish-token operations while preserving GPU operation order. Start with layer entry, positioned Q/K/V, attention return, MLP branch and committed K/V as optional typed observations. Use complete token-boundary checkpoints and deterministic replay to reach the selected operation. Add only the particular intervention needed by the first assay.

Arbitrary interior checkpoints can wait. If later introduced, a partial-layer cut must bind the residual, phase and per-layer current-token cache validity. Unsupported cuts must remain explicit.

Keep tracing off by default. Retain selected device-side values and copy them at declared observation boundaries; avoid host callbacks or full tensor dumps after every kernel. Connect these safe points to the existing public adapter. The current whole-process receipts remain valid at their stated scope and must not be relabeled as internal operation receipts.

Deliverable: a replayable instrument that can identify the first discrepancy and the later consumer without replacing the runtime with a generic graph engine.

### 3. Close one writer–memory–reader computation

Start at the existing OVC6 boundary, where every K/V intervention shares the complete incoming query H16. Use original positioned keys and values as declared operands to qualify the first affected attention read, then its receiving normalization and MLP. Freeze those conditional predictions before new native consumption. This isolates arithmetic and channel interactions without claiming to construct a fact from ordinary input.

OVC6 captures a complete multi-token query-suffix matrix; C executes one token at a time. Retain the original native query geometry for that conditional assay, or separately qualify a C one-token counterpart. FP32 dtype alone does not transfer exact equality across GEMM, matvec and attention reduction schedules. The known memory band nominates a scientific boundary; it does not grant numerical or checkpoint portability.

Use the known values-only misses alongside successful cases as the first development contrast. Hold each source history fixed and vary the later query's target and wording. Preserve the K-only, V-only and joint controls. Locate whether the decisive difference first appears in routing, payload read, receiver response or later feedback. These cases were selected after their outcomes and are development evidence; a frozen explanation then needs a new source/query/receiver combination.

Then move the forecast boundary backward. Form the carrier from the machine's own ordinary-input execution, predict the next write, and let a later question consume that predicted state. Use new wordings, values, source locations, overwrites and distracting bindings to expose missing operands. A donor remains a microscope control, not a serving dependency.

Test a minimal interface, then interventions that leave it equal while changing excluded state. When two equal interfaces produce different futures, add the first explanatory dependency and retain the counterexample. An exhaustive full-state identity always remains available; the research question is whether the refined interface makes the computation simpler and transfers.

The new scientific result must go beyond reproducing known architectural equations. Freeze a declared read interface that is simpler than supplying the complete native computation. Use it to predict which binding the live query selects, what payload change is written and how the target versus collateral query will consume it. Require the proposed operands and rule to transfer to a new combination while the candidate forms its own carriers. Reconstructing dense attention is the control; finding a reusable trained operand/consumer relationship is the explanatory advance. A faster implementation requires the additional observation that some native work was actually avoided.

Deliverable: one bounded operation whose resolver, write and later read are explained prospectively. Measure answer effect, collateral effect, own-state continuation and actual work. A useful explanatory law may precede an efficient replacement.

### 4. Evaluate nested reasoning and open chat separately

Begin with a small panel of short multi-turn sessions covering arithmetic, delayed recall, revisions/conflicting facts and unconstrained explanations. Give each arm the same initial user turns and declared budget. Retain each arm's own replies; use separate same-history comparisons to localize numerical error. Preserve EOS, truncation and unresolved questions explicitly.

For stack behavior, begin with a controlled nested subproblem, then ask whether model-generated decisions to call and return preserve the intended variables. Vary nesting depth and paraphrase the surrounding language. A supplied parser or evaluator must be labeled as supplied. Add arbitrary chat only after this makes the bookkeeping and failure modes interpretable.

Report native-token fidelity, task correctness, supported versus unsupported factual claims, memory-update consistency, first divergence and cost separately. The tests choose the next causal cut; a missed score does not erase a useful signed response or counterexample.

Deliverable: an honest account of which improvements come from execution fidelity, extra memory, explicit control or tools, and which native computations we can replace.

### 5. Optimize only the work that the evidence identifies

Measure resolution, cache validation, copies, replacement computation, replay, loading and peak RAM/VRAM together. Compare invariant-prefix reuse, a genuinely skipped bounded law and full dense execution under the same input and numerical policy. The latest FLUX exact-reuse test skipped operations but did not yet reduce wall time; this is a direct warning against treating fewer counted operations as a speedup.

Keep real-model work in finite mrun CUDA leases on Beast, one resident model and one informative panel at a time. No training is needed for these first stages. Paging, quantization, automatic summarization and broader architecture support can wait for an observed bottleneck or a qualified interface.

## Decision

Build the strict continuation stack and operation-boundary instrument first. Use the existing binding-memory experiments to recover one context-qualified read–update–write interface. Evaluate native fidelity, own-state prediction and answer correctness independently. This makes stacked state useful immediately while keeping the larger goal precise: a model computation is understood when we can resolve its operands, advance its relevant state and predict how a later consumer will use the result.
