# Mathematical models and concept reasoning in cmachine

Date: November 6, 2026. Status: proposed research plan, informed by an Astra advisory review of the recent work and a focused critique of this draft. This plan adds no serving policy or experimental result.

Build an inspectable mathematical machine from the existing learned operator cells, FieldLM concept records and neural rewrite scheduler. First make their computations explicit in C. Then test whether the combined system can produce better mathematical programs on unfamiliar problems while consuming its own results. Use the learned matrix algorithm's existing C implementation as the first practical downstream consumer. Extend to Qwen-controlled operation and operand selection after the mathematical interface is measured.

## What the recent work changes

The symbol-table work connects a computation to its real inputs, domain, model/profile, source addresses, clock, writes and later consumers. A known equation becomes useful when its operands can be resolved and its result remains usable in the future it creates. The [applicability experiment](../../../obsidian/blog/2026-11-06-092300-a-symbol-table-that-checks-its-inputs.md) selects among qualified methods from source conditions; it does not automatically discover semantic names or attest arbitrary numerical backends.

The latest Qwen experiments separate three problems: computing an operation, selecting it in context and preserving the relationships needed by its next consumer. The [operation-reliability result](../../../saturn/experiments/2026-11-06-operation-reliability/FINDINGS.md) obtains 24/24 correct fresh additive chains versus 11/24 wholly correct ordinary traces, carrying all 180 model-computed intermediate values. It supplies the parser, program counter and operation addresses, restricts the arithmetic family to values 0–9 and increments ±1/±2, and pays 1.7699 times the analytical linear work. It is an especially useful control for a later learned dispatcher.

The [contextual-query result](../context_query/FINDINGS.md) improves fresh own-history mean KL by 13.59% under its declared conditions, but the later binding failure and [Q/K/history cut](../query_key_cut/README.md) show why a corrected current operand cannot repair an incompatible inherited cache. Our math experiments must likewise carry actual produced state and library revisions. A locally correct operation is insufficient evidence of a correct longer computation.

`cmachine` already owns Qwen's numerical execution, K/V, token commitment and continuation. Its [current architecture](../../docs/ARCHITECTURE.md) provides narrow operation APIs; the core still performs the dense model. This work can use full state and full memory while improving mathematical capability. Solving compact general-chat memory is a separate research question.

## Existing ingredients and their boundaries

| Ingredient | Existing computation and evidence | Use in this plan |
| --- | --- | --- |
| Exact learned seven-product cell | Randomly initialized bilinear networks learned U, V and W; the exported integer factors have complete ordered coefficient certificates. Two frozen circuits are [available locally](../../../model-experiments/experiments/2026-10-17-learned-matrix-c-scale/frozen-rules.json). | First exact learned operation. Execute the actual factors and expose each linear form, product and output combination. |
| Approximate 84-parameter cell | The [October 30 exports](../../../model-experiments/experiments/2026-10-30-architecture-matched-learning/exports/README.md) evaluate W[(Ua) elementwise-multiplied by (Vb)]. Every saved floating circuit has nonzero exact rational residuals. | Separate numerical specimen for error propagation and future feedback; exact-ring rewrite certificates do not certify its floating execution. |
| FieldLM concepts and metaconcepts | [Structural acquisition](../../../obsidian/moonshots/2026-10-30-fieldlm-growing-mathematical-concepts.md) stores acquired executable rewrites and parent dependencies. [Metaconcepts](../../../obsidian/moonshots/2026-10-30-fieldlm-mathematical-metaconcepts.md) add domain/operand-role matching and neutral reassociation. Search, primitive semantics and verification are supplied. | Typed law library, domain guards, explicit bindings, module reuse and measurable concept growth. |
| Concept graph and learned scheduler | [V0](../../../model-experiments/experiments/2026-11-02-concept-graph-v0/RESULTS.md) supplies a domain-aware equality graph and whole-output sharing cost. [V1](../../../model-experiments/experiments/2026-11-02-concept-graph-v1/RESULTS.md) trains a 21-feature, 32-hidden-unit MLP to rank candidate rewrites. | Reuse the existing graph engine and frozen scorer. A neural proposer already exists in this bounded form; broader concept selection remains to be tested. |
| Later library growth and exact scheduling | [V2a](../../../model-experiments/experiments/2026-11-02-concept-graph-v2a/RESULTS.md) selects supplied expansion/division schemas; full search improves 12/16 planted goals, while the grown learned scheduler regresses on another family. [V2b](../../../model-experiments/experiments/2026-11-02-concept-graph-v2b/RESULTS.md) certifies the published fixed-factorization schedule and improves two archived greedy schedules from 57 to 56 additions. | Preserve base/grown library-scorer pairs, test interference before new acquisition, and reuse exact signed-binary scheduling as a strong control. |
| Native C matrix consumer | The [learned C circuit](../../../model-experiments/experiments/2026-10-17-learned-matrix-c-scale/RESULTS.md) runs exact uint64 arithmetic modulo 2^64 through 8192-square matrices. It beats its named classical control but is 10.6% slower than its matched Strassen control at 8192. | An executable mathematical artifact whose linear program can be improved and measured. No new exponent or fastest-matmul target is needed. |

V1 is useful but heterogeneous: across 22 goals it beats random scheduling on 18, ties one and loses three; SymPy beats it on 13. Several larger product families reach their reference optimum, while repeated-polynomial-factor problems expose a missing concept. Its one training seed does not establish training stability. The existing scalar-trained scorer's transfer to ordered matrix graphs is an experiment.

The subsequent V2a acquisition addresses that particular concept gap using supplied `expand` and `divide` schemas. It improves the planted panel but sends the learned scheduler's binomial-k7 cost from 13 to 770. Remaining divisor discovery and cross-family interference are the current questions. Bind each library revision to its scorer and feature contract; a larger library is not automatically a better operating configuration. These November 2 follow-ups were located while preparing the proposal blog and refine the earlier V1-only inventory.

## The execution interface

Keep immutable programs separate from invocation state. Bind the concept library, scorer and feature-schema revisions together. A mathematical operation record should contain:

- Identity and provenance: operation ID, artifact hash, learned/supplied origin, coefficient bytes, numerical profile and certificate where applicable.
- Interface: ordered argument roles, domain, shapes, output roles, conditions and parent-module dependencies.
- Invocation: actual argument bindings, input-state hash, graph/program counter, concept-library revision, returned values and next consumer.
- Cost: compilation, candidate construction, feature formation, scoring, lookup, arithmetic, temporary storage, transfers and replay.

Use the same qualified-entry discipline as the symbol table. Do not identify two invocations merely because their tensor shapes or semantic labels match. Match the complete declared interface, and preserve previous misses and counterexamples.

For a mathematical session, retain the graph and roots, value slots, program counter, concept revision and continuation ancestry. Include search frontier/RNG when replaying search. A future shared Qwen session additionally retains token/KV ownership. Rollback must restore every mutable store inside its declared boundary; physical work counters remain outside rollback.

The first implementation can be a research C library/driver under this directory, linked to existing model/hash utilities. Python initially performs export, graph reasoning, native FieldLM query execution, research orchestration and analysis; C executes the frozen scorer and numerical consumers. This is explicitly a hybrid research system until its required graph/matching subset has a qualified C implementation. Exporting a solved answer or preselecting held-out bindings is not that implementation.

## Stage 0 Recover and bind the small artifacts

Start from the two local exact frozen rules, the three October 30 floating exports and the generated C consumer. Recover V1's frozen `scorer.json`, feature ordering/normalization and source-bound receipts from its [offload manifest](../../../model-experiments/experiments/2026-11-02-concept-graph-v1/results/OFFLOADED.json). Recover concept-library exports and any remaining source receipts needed to identify what actually ran. Fetch the small referenced objects instead of rerunning their training or restoring every historical archive.

Inventory each artifact's source/checkpoint, domain, backend, supplied primitives, learned content, serving dependencies and demonstrated horizon. Pin the C source revision and preserve current unrelated working-tree changes. The original verified native release and later operand APIs have different evidence scopes.

Deliverable: a compact artifact manifest with resolvable origins, three explicit numerical domains (bounded exact integers, uint64 ring and declared floating profile), and the unchanged reference executors.

## Stage 1 Execute and inspect the math models in C

1. Add a small bilinear-cell executor that exposes Ua, Vb, seven products and W's output. Load the actual frozen factors. Retain separate exact-word and floating profiles. The approximate neural cell must consume its own outputs during repeated multiplication.
2. Add the frozen scorer's input normalization, 21→32→1 ReLU network and stable top-k decision. Compare feature bytes, scores, ranking ties and selected candidates with the frozen Python implementation. Do not assume equal FP32 dtype means identical reductions or choices.
3. Bind native FieldLM role queries to the graph engine and emitted operation calls. Trace repeated-variable equality, left/right operand order, domain applicability, parent-module retrieval and concept revision through one final C numerical consumer.

The initial mechanics panel is deliberately small: ordered basis-pair checks for the exact bilinear identity; fresh small matrix inputs; one same-parent replay; one wrong factor/argument binding; one missing parent; and a short sequence using actual computed matrices. Certify bounded integer overflow before execution, or select explicitly defined unsigned modular arithmetic. Preserve a refused or partial-state invocation according to its contract.

Deliverable: model-free C mechanics plus a small mrun source-bound comparison. This establishes integration and computational understanding; capability improvement is tested next.

## Stage 2 Use concepts to improve the learned matrix program

Begin with a few small exact expression graphs, including

    (((AC + BC) + AD) + BD) → (A + B)(C + D)

Use ordered matrix variables and both valid factor sides. Include a graph where a neutral reassociation exposes a later reduction and a multi-output graph where local factoring would duplicate retained work. These are interface checks using already studied identities, not fresh evidence of concept invention.

Then make the fixed learned matrix circuit the first substantive consumer. Represent its input linear forms and all four output combinations as shared expression graphs. Preserve operand order across products. Initially hold the seven product slots and U/V/W identity fixed; search only equivalent sums, sign arrangements, shared intermediates and executable schedules. Reuse V0/V1's roots-aware DAG cost and extraction instead of building another graph engine.

The measured generated-source account for learned811 is 12 input binary additions/subtractions plus 14 output accumulations, versus 10 plus 12 for its Strassen control. Those 26/22 totals exclude zeroing, packing, loops and unary negation and are not minimal circuit bounds. Reduced arithmetic is a hypothesis about the timing gap, not an established causal explanation.

Compare a short sequence of informative arms:

| Comparison | Question |
| --- | --- |
| Unchanged learned C program versus generic CSE, conventional linear simplification and V2b exact scheduling within its declared gate model | What is already achievable by existing compilation and scheduling methods? |
| Fixed concept library with bounded full search, heuristic, random and learned scheduling | Does the learned choice policy help under the same available rules and primary expansion budget, including when ordinary breadth can find the route? |
| Current versus grown concept library with a frozen selection policy | Does a reusable concept extend the computations the system can reach? |
| Original, conventionally simplified and concept-produced programs in the same C generator | Does the mathematical change improve the executed consumer? |

Start with a frozen existing library-scorer pair. Before new acquisition, compare the existing base and V2a-grown libraries with full, heuristic and compatible learned selection, retaining the families that previously benefited and regressed. Cross scorer/library revisions only when the feature contract supports it, as an explicitly labeled diagnostic. Expand to concept acquisition when the trace identifies a missing reusable transformation. Grow executable bodies using discovery examples, preserve assumptions/dependencies, certify exact identities, then freeze the library before opening new goals. Mark manually supplied rules or exact-division/GCD routines as supplied. Do not credit their answers to learned discovery.

Reuse V2b's exact signed-binary scheduler with its explicit conventions: copies/signs free, doubling one gate, declared auxiliary budget and output-transposition correction where used. Its published 55-addition result and the C generator's 26/22 accounting are different cost models. Report them separately, then measure actual emitted C. A truncated auxiliary search yields a bounded observation, not a minimality proof. This stage does not repeat the rank-23 flip-graph campaign.

Freeze a structurally disjoint set of new signed linear circuits and multi-output contexts. Renaming variables alone is a robustness probe, not a new structural family. Seal goal generation and evaluation before search outcomes; quarantine the historical known circuits as development. Preserve common candidate generation, feature access and extraction for scheduler comparisons. Report both fixed-expansion/node budgets and fixed wall-time results: cheaper scoring and more search are separate contributions. Charge training and acquisition separately.

Verify the complete mathematical identity of each emitted program, then run it on fresh exact-word matrices and actual output-dependent continuations. Apply generic simplification, concept search, extraction, code-generation changes and tuning equally to both learned circuits and the known Strassen control; preserve unchanged historical implementations as references. Compare original and optimized versions with tuned Strassen, Winograd where available/qualified, classical C and the conventional simplifier under identical kernels, packing and allocation policy. Small calibration chooses a useful timing size; an 8192 rerun is unnecessary until the smaller test reveals a change worth measuring. Count real scalar work, memory traffic and wall time independently.

Deliverable: a prospective account of whether existing concepts and neural scheduling produce cheaper correct mathematical programs, what transfers to new structures, and what each component contributes. A tie or a negative timing result still identifies whether the limitation lies in vocabulary, matching, choice, extraction or execution.

## Stage 3 Close the required concept interface in C

After Stage 2 identifies the useful operation subset, lower frozen FieldLM records and generic projected read/join programs into a versioned C representation. The C executor must resolve new live bindings, enforce repeated-variable identity, fetch parent bodies and update graphs; export must not supply the selected held-out plan.

Qualify every needed query and rewrite trace against native `RoleChartField`/`FieldFabric`, including ordering, domain refusals, shared outputs and library install/uninstall. An equal-information indexed dictionary is the control. The C representation is a compiled subset with measured parity, not a claim that all FieldLM functionality has been ported. If Python still selects candidates, report it as an external controller and count its cost.

Deliverable: the smallest complete math session whose operation selection, argument resolution, numerical execution and own-state continuation are owned by C for the declared tasks. No general graph-runtime rewrite is a prerequisite for Stage 2.

## Stage 4 Let a model choose computations and their arguments

Port the existing source-driven additive controller to full-cache C Qwen as a baseline, preserving the exact prompt/tokenization contract and its own scalar results. Framework FP32 and C/cuBLAS need separate numerical qualification; the historical 24/24 result cannot be assumed to transfer. Use ordinary text result records and re-encode them after restoring a caller; arbitrary sibling K/V merging is not call/return.

Then ask Qwen to emit a small typed plan selecting an operation/concept ID, actual operand references and result destinations. Initially use structured inputs; later vary names, instruction order, irrelevant facts, overwrites and language. Syntax/type validation may refuse malformed plans but must not silently compute the missing solution. The operation menu and primitive semantics are declared supplied ingredients.

Compare the same Qwen under ordinary prompting, matched-budget prompting/branching, supplied source controller, model-selected plans with conventional exact operations, and model-selected plans with learned cells/concept reuse. Keep operation selection, argument binding, local numerical accuracy, complete chain correctness, final answer and cost visible separately. Test fresh compositions that require more than one call and later use of the produced result. Correct a binding or arithmetic operation only in separately labeled microscope controls.

This stage can demonstrate increased capability of the composite system. An improvement due to explicit control or a supplied exact kernel does not establish that Qwen's weights acquired that arithmetic. To make a base-model claim, separately train a declared module or test a no-tool evaluation with fresh held-out problems.

## Stage 5 Improve the smallest measured bottleneck with Saturn

Choose the next update from the traces: a missing concept, an ineffective scheduler, an incorrect operand/source router or an approximate cell with compounding numerical error. Freeze other components. Reuse the existing `DeterministicCheckpointController` and evaluate actual emitted programs, downstream numerical consumers and own-state rollouts after short update intervals.

Capture model and optimizer state, RNG, sampler/cursor, graph/library revision and all mutable feedback state. Keep discovery, continuation-selection feedback and sealed evaluation distinct. Rejecting a continuation preserves its candidate measurements and raw artifacts. Train on the real limitation: a scheduler cannot select a concept the library cannot express, and a perfect arithmetic cell cannot repair a wrong operand address.

## Meaning of demonstrably smarter

The first bounded capability target is an increased ability to construct correct, cheaper mathematical programs on new structures using reusable concepts and learned selection. The later target is more correct multi-step answers when the system selects its own operations and carries its own results.

Report exact correctness and complete-chain outcomes, improvements/ties/regressions by family, cost-to-solution curves, learned-versus-supplied contributions and failures at each interface. Native-logit fidelity, copying a known formula, lower operation count and lower runtime each answer different questions. Require only integrity, instrument validity and safe execution during discovery; apply a named terminal bundle when certifying an exact identity, a bounded capability improvement or a timing claim.

## Execution and first deliverable

This turn prepares the plan only. Recover/bind artifacts and complete tiny C mechanics before the first scientific lease. That lease has this proposed fixed geometry:

- Specimens: learned811 and the known Strassen control; exact uint64 arithmetic and the single frozen V1 scorer. No floating-cell comparison or training.
- Consumer: their fixed input/output linear programs, with the same frozen concept library, generic simplification and extraction. First run the four small domain/binding/sharing fixtures above as instrument checks.
- Fresh contexts: three source-sealed structural variants per circuit, such as an extra output retaining an intermediate, a signed caller combining outputs and a two-call composition consuming the previous result. These test contextual reuse of the known circuits, not discovery of independent multiplication laws. Keep renaming controls separate.
- Search: bounded full, heuristic, random and learned scheduling; at most 2,000 applied candidate rewrites and 10 seconds per arm, with incremental candidate/materialization limits enforced inside enumeration. Freeze the exact caps and seed before submission. Maximum 24 search paths; time-limited and expansion-limited rows retain their actual denominator.
- Execution: one 256-square timing size after a mechanics/calibration check, using the same fresh full-word inputs and three rotated repetitions for unchanged and emitted programs. Verify complete output hashes and report timing variability. This is an exploratory small-size measurement.
- Envelope: one finite CPU mrun lease, strict preflight, one numerical thread, no retries or nested jobs, and a ten-minute wall limit. If mechanics/calibration projects an overrun, reduce the declared pilot before submission. Leave concept acquisition, additional cells/seeds, Winograd, larger matrices and Qwen outside this lease.

Measure the pilot before choosing expansion. Reuse source snapshots and artifact-only audits; no full model retraining is needed. Qwen work later uses one resident CUDA model with explicit CUDA admission and full-cache reference state.

Any expected run over 15 minutes receives the workspace-required independent Luna review of code and mrun settings before launch. Do not duplicate an active accelerator suite. Use standard Saturn evidence/custody plus custom identity, binding and arithmetic audits; the current `tools/saturn_bridge.py` receipts cover whole-process invocations, so an interior math-step claim requires an explicitly implemented Adapter/state schema and its own closure checks.

After the pilot, pursue concept growth when a specific missing transformation is demonstrated; improve selection when another search arm finds the useful route; otherwise retain vocabulary-versus-search uncertainty. A bounded search miss does not establish unreachability. Only exhaustive completion in a declared finite grammar could support that narrower claim. Fix the compiler if symbolic savings disappear in C, or proceed to model-controlled dispatch once the mathematical component works. None of those decisions requires claiming that a missed exploratory score disproves the underlying idea.

## Astra advisory decisions incorporated

Astra recommended the existing learned circuit as the first substantive mathematical consumer, with a few domain checks before it and Qwen dispatch afterward. Its draft critique added the bounded full-search control, symmetric optimization of the strongest control, explicit Python graph/query ownership, a finite first-pilot geometry and the distinction between a bounded search miss and demonstrated missing vocabulary. The remaining uncertainties are empirical: scorer transfer, useful concept coverage, extraction quality, actual C cost and later autonomous operation selection.

After reviewing V2a/V2b, Astra additionally recommended preserving library-scorer pairs, testing existing growth for cross-family interference and using the exact signed-binary scheduler as a stronger conventional control. The first small integration pilot remains fixed; the existing-library comparison precedes further acquisition.
