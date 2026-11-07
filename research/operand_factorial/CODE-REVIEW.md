# Astra: bounded implementation review

2026-11-06. Additive, prospective source review. No native execution. Review status below refers to the implementation snapshot, not a historical run or scientific outcome. The driver had not yet appeared at the first review; this is not launch clearance.

## Launcher and worker, first snapshot

`run.py` authenticates the predecessor report, exact source factors and selected historical arrays before staging. Its thirteen cut positions and retrospective arithmetic input18 at79 are consistent with the recorded history. Continuation IDs are sliced literally from the old consumed sequence; final cuts correctly have no future events. No target fitting or recompilation is introduced. The native same-ID observation is bound separately from the candidate observation.

The launcher declares one CUDA lease, one model, CPU1, B1/context512, strict preflight, no retries and a finite cutoff. Its 6,144MB reservation uses the measured predecessor rather than a guessed model footprint. Final validity of the stated extra-session/checkpoint bound depends on the driver allocations. One extra Qwen K/V session at 28 layers ×512 tokens ×256 KV width ×2(K,V) ×4 bytes is about29.36MB, well below the stated approximately364MB margin from predecessor measured peak to guard. Multiple simultaneous retained device copies must be counted if introduced; sequential restoration does not require five parallel resident sessions.

The worker imports only frozen factors/histories and invokes the C process. Its archive preflight explicitly includes full vocabulary outputs and selected operands. It needs the final driver capture geometry to validate the bound. The independent native controls are paid diagnostics. Their existence in this new process does not retroactively make the old source-only execution donor-assisted.

## Analyzer corrections sent while schema was still being implemented

The coordinator had already identified the averaged-main-effects accounting bug: gate_main+up_main equals the joint effect, so adding interaction again double counts it. I did not treat that known issue as a new finding.

Additional issues in the initial analyzer snapshot:

1. `_exact_mlp_interaction` forms the 8,960-channel pre-down product `(SiLU(g1)-SiLU(g0))*(u1-u0)` and compares it with a 1,536-dimensional write interaction. These are different spaces. Retain a product-space identity and compare the write factorial with the driver's native-down projection of the identity vector, or supply a correctly identified down map. Do not call a shape-missing comparison a verified write law.
2. `_arrays_for_clock` assigns hidden-size1536 to gate/up vectors. Their required size is intermediate8960. Product channels likewise have the intermediate dimension.
3. Array-valued fallbacks using Python `or` (`gate_arrays.get('Y10') or ...`, up equivalent, and write/output fallback) raise NumPy ambiguous-truth errors. Test `is None` explicitly.
4. The initial main analysis computes only write factorials and a write decomposition. The declared scientific consumer requires full-logit factorials and the full-logit C−L/L−H account too. Saving logits and greedy choices is not a substitute for analyzing that account.
5. The decomposition initially reports only unnormalized terms. Use H as the common denominator when reporting normalized C−H, C−L and L−H, and ||H||² for normalized squared/cross terms. Different denominators would break the stated normalized identity.

These are instrumentation/mechanics corrections, not added scientific quality gates. The final analyzer must also keep the factor-assembly identity restricted to the intervention clock: subsequent states have different receivers and use all-native transitions, so the original mixed-operand product identity is not a new factorial assembly law at those future layers. Their output factorials remain legitimate transported effects of the original intervention.

## Pending driver checks

Before marking this review complete, inspect common-parent restoration and valid-cache ranges, source64 immediate replay, native H replay, native Y11 write parity, operand buffers before in-place SiLU, all-native future transitions, physical work accounting, resident-session/checkpoint geometry, and the projected finite-identity control. A valid current-word factorial does not certify autonomous continuation; this protocol deliberately forces at most two historical IDs.

## First driver snapshot: actionable mechanics findings

The driver reconstructs each parent by replaying the same frozen prefix under source64, then applies one operand choice and follows up to two historical IDs under MODE_NATIVE. Native H rebuilds the identical prefix under native execution. This policy matches the plan. Sequential fresh reconstruction is valid if its recorded parent/receiver equality is verified; it uses one session, so does not add multiple resident K/V checkpoints. Gate/up captures occur before the in-place SiLU/product and are therefore at the intended operand boundary.

The following concrete draft defects were sent to the coordinator before any native launch:

- The native/Y11 call assertion expects two gate calls although the wrapper has separate gate and up counters. Expected counts are gate1/up1/down1.
- `interaction()` calls `save_device(array_dir,out_product,W,h)` before initializing the filename and reads h=8960 values from the d=1536 allocation W. Remove that stray call; the subsequent named product-from-P and write-from-W saves are the intended ones.
- `current_row` is called with `s->position` after token completion increments the position. Capture the saved processed position instead, for both current and future tokens. Otherwise the array is the next, unwritten cache row.
- `REQUIRE(fflush(out))` rejects a successful flush. Require return value zero.
- Local `Capture` conflicts with the type exported by the frozen `assay_io.h`; use a distinct type name.
- `StateResult.prefix_sha` is never filled. Remove the field or populate a precisely declared valid-prefix digest; an empty string must not masquerade as a state binding.
- The final fields named primary native gate/up/down counts use the wrapper counters left over from the last native-reference token. Accumulate physical calls by phase, including prefix reconstruction, or remove/rename the misleading fields.

An integration issue also remains in this snapshot: `run.py` stages historical arrays and expected boundaries, but worker/analyzer do not yet visibly compare them. New-output hashes alone cannot establish old-output parity. The declared mechanics require old-source Y00 receiver/norm/write and boundary joins, old-native H boundary joins, Y11 paid-native-write joins, and equality across the five candidate parent digests and receivers. These are the existing plan's integrity checks; no fidelity threshold is being added.

These findings describe draft source under active correction. They must be resolved and reflected in the finalized snapshot before this note can be read as approval of native mechanics.

## Dedicated integrity module review

The subsequent `integrity.py` implements the planned historical and common-parent joins separately from the scientific analyzer. Its worker invocation and saved-only rerun through the archived module are appropriately separated from scientific quality metrics. Gate/up versus write/cache dimensions, expected physical selected-map counts and literal future-token checks match the draft driver schema. It authenticates the twelve fixed old Y00/H full readouts and valid-cache digests, all selected old receiver/norm/source-write arrays, and all Y11 old native writes. The retrospective digit has its own saved-array joins and expected wrong-token replay rather than pretending a previously uncaptured full-vector endpoint existed.

The module also checks raw receiver/norm equality across all six worlds at each available horizon. This is mathematically justified here: the changed MLP16 write cannot alter the preceding layers or the next MLP16 input under identical consumed IDs, even when upper historical K/V differs. It would not be a valid invariant for arbitrary earlier-layer interventions.

Two integration details were sent back: (1) the now-required future `prefix_kv_sha256` must be populated before processing each token and match the previous completed state's digest; the earlier driver left it empty; (2) join the driver's three reported factor hashes to the authenticated frozen factor-file hashes, rather than validating staged files without checking the driver's identity report. The remaining driver checkpoint/counter changes were still in progress at this review. No additional scientific controls are requested.

## Updated active main review

The updated active main and helpers resolve the earlier invalid buffer save, native counter assertion, row-position offset, inverted flush and Capture type conflict. Pre-step prefix digests are populated; actual loaded factor hashes are joined to file hashes, sizes/profile are recorded, and per-token selected operations accumulate across execution. The common-parent policy and all-native historical-ID future remain correct in the reviewed sequential-replay version. No additional mathematical launch blocker was found in that snapshot. This readiness is for the declared mechanics smoke, not a claim of native parity before running it.

Two remaining reporting details were identified. The array seal is made before native H execution, while the new driver JSON is emitted after H and embeds H in each cut. Thus the JSON prefix named `primary_seal` is an after-reference metadata digest, not a pre-H metadata commitment; distinguish the two scopes explicitly. Also, `aggregate_*` counters accumulated by `step_token` omit the separate interaction diagnostic's one native-down matvec per cut. Count that diagnostic separately or state the exclusion whenever quoting total work.

Checkpoint reuse was proposed after this snapshot. It is a runtime optimization only if it restores the exact complete parent and does not alter future policy, tensor association, valid-cache ranges or delivered-token authority. That change requires review of its actual implementation before carrying the sequential-replay readiness statement forward.

## Final checkpoint review before smoke

The final driver rebuilds one source64 prefix per cut, calls `qwen_checkpoint`, and restores that same parent for each of the five branches. Native H receives no candidate checkpoint and rebuilds its own identical-ID prefix natively. I inspected the checkpoint/restore implementation in the authenticated predecessor runtime, rather than assuming a generic tensor snapshot.

This checkpoint is a metadata rewind, not a copy of all tensors. It is sufficient for the present append-only token-boundary use: the branch never changes older valid cache rows, restored position excludes later branch rows, subsequent current rows are overwritten before being read, and `qwen_begin_token` replaces x with the supplied input embedding. All selected cuts have positive positions; session, epoch, checkpoint serial and capacity remain compatible across the five restores. This does not justify using the same checkpoint for arbitrary in-place edits of older cache rows or for interior-layer state.

The final accounting separates prefix reconstruction, the five restored branches, native H and diagnostic down calls. The array-before-H and metadata-after-H seal timings are explicit. Previously identified source-level blockers are resolved in this snapshot. No remaining mathematical or protocol launch blocker was found: the implementation is ready for the declared mechanics smoke, whose actual historical parity and saved-only integrity results remain required evidence.
