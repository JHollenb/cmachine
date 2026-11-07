# Prospective Astra advice: query-qualified memory response

November 6, 2026. Written before any memory_response model outcomes. This note applies to the frozen four-corner development panel, not to general chat certification. No native execution was performed to prepare it.

## Question and construction

The hypothesis is that a receiver-selected small support can preserve consequential distinctions that an equally short recent window loses. An alternative is that many paired atoms must remain available, with reduced coordinate precision offering a better tradeoff than support reduction. A mechanically closed loop does not choose between those hypotheses.

Let x encode the initial red/blue assignment to Mira and Niko and u encode retention versus exchange. Mira's final blue indicator is x XOR u; Niko's is its complement. All four corners must run. Each policy starts empty, consumes supplied record/update/distractor text and its own generated acknowledgements, then forks two questions independently. A Mira answer must never enter the Niko question parent. The exchange is an additional computation, so retain per-corner results rather than labeling every wrong answer as forgetting.

The five policies in PLAN.md are native FP32 full memory, paired-byte full memory, paired-byte top-four reading, paired-byte recent-four storage/reading, and native-parent historical-memory deletion. The first four form their own histories from empty state. Deletion forms its parent natively, then excludes pre-fork keys while preserving absolute positions and every newly appended query row. It is a paid dependence diagnostic, not a cheaper prefix-construction policy.

Top-four retains and scans the complete encoded bank. Recent-four has a four-row ring and must retain rollback snapshots. Their common maximum value-mixing support is four rows per head; their storage, score work and total costs are different. No centroid mutation, support sweep, rank search or target-dependent fitting is needed in this first panel.

## Exact response identity and finite qualification

For one head, fixed query q, paired keys/values (k_i,v_i), and nonempty selected support S, write weights w_i=exp(q^T k_i/sqrt(d)), Z_S=sum_S w_i and N_S=sum_S w_i v_i. Define the corresponding complement quantities Z_T,N_T. When T is nonempty, A_S=N_S/Z_S, A_T=N_T/Z_T and epsilon=Z_T/(Z_S+Z_T). Partitioning the original numerator gives

    A = (1-epsilon) A_S + epsilon A_T,
    A - A_S = epsilon (A_T - A_S).

If T is empty, epsilon is zero and A=A_S; no complementary conditional mean is required. In any norm,

    ||A-A_S|| = epsilon ||A_T-A_S||
             <= 2 epsilon max_i ||v_i||.

Thus omitted mass and payload contrast jointly determine the error. A small mass alone supplies no dimensionful error guarantee without a value bound. Likewise a large omitted mass can have a small error if the two conditional means agree. The equality concerns the same operands and exact real normalization. Compare the actual finite CUDA result to the independently accumulated expression and retain its finite remainder; do not silently reassociate the serving program.

The selection law uses the actual current query, deterministic support ties and the declared GQA mapping. It must include the newly encoded current row among eligible rows. Native signed-byte reconstruction changes both the logits and payloads, so conditional codec error and support error need separate fields.

## State closure and what this assay does not prove

For native phase transition F_phi, retained encoding C_phi and observable O_phi, an exact sufficient interface would satisfy

    C_next(F_phi(S,u)) = G_phi(C_phi(S),u),
    O_phi(S) = H_phi(C_phi(S)).

Equivalently, every pair of states in the same encoding fiber must have the same admitted observation and the same successor encoding for each admitted input. Necessity follows because a deterministic G/H receives identical arguments. Sufficiency follows by defining G/H on each fiber; induction then preserves every admitted finite future. This is a well-definedness result, not a proof of cheap implementation. The identity encoding trivially qualifies while retaining all original work.

For a state pair with identical code and differing outputs Y_0,Y_1, any deterministic same-code predictor has worst-case norm error at least one half ||Y_1-Y_0||, by the triangle inequality. Conversely, finite success on this panel cannot establish fiber invariance on arbitrary states or queries.

The byte policy provides an executable approximate transition, not an exact quotient of the native transformer. It appends E(k,v) formed from its own current residual and then reads the resulting bank. Top-four changes the response law; recent-four also changes retained reachability. All-layer deployment changes subsequent own queries. Native-versus-byte and byte-full-versus-top-four are therefore composed policy comparisons, not automatically controlled local effects at identical operands.

At the named layers17/18, a post-seal audit can evaluate original-own full attention, decoded-byte full attention and decoded-byte top-four attention on each candidate's same actual query and own-row history. Audit-only full histories must be inaccessible to the serving reader. These diagnostics separate conditional codec/selection effects from native-trajectory drift without repairing a candidate.

For comparison with a native trajectory, the exact error telescope is

    S_hat_next - S_next
      = [F_hat(S_hat)-F(S_hat)] + [F(S_hat)-F(S)].

The first term is replacement defect at its own state; the second is state-transport difference. Evaluating native F on candidate state is paid post-seal diagnosis. No global contractivity, additive causal share or bounded long-horizon error follows from this identity alone.

## Mechanics that protect the interpretation

- Exact FP32 callbacks must reproduce the ordinary native instrument at the declared ports and numerical geometry.
- Encode, validate and publish the current row before its current attention read. Preserve absolute token positions through history deletion and all rollback paths.
- Restore overwritten ring contents as well as logical cursors. A cursor-only rewind cannot undo eviction. Test a partial multi-layer update and a sibling fork, not only completed-token rollback.
- Every generated token, including EOS, is consumed exactly once. A supplied next-turn suffix may contain the prior assistant terminator: skip that supplied terminator only if the generated EOS was already consumed; otherwise consume it once as supplied template control.
- Restore the same candidate parent for both questions. Token history, cache ownership, ring contents, pending-token/control phase and position must agree at restore.
- Preserve actual model/tokenizer/source identity, finite tensor shapes, CUDA resource limits, raw extents and custody. Count aborted physical attempts separately from restored logical steps.

These are instrument/safety contracts. Native answer correctness, deletion sensitivity, margins, local norms and agreement rates are observations rather than admission gates. Retain every corner, incomplete answer, native mistake, changed history and optional diagnostic failure. No outcome-dependent prompt replacement or text/support tuning follows the smoke.

## Reporting and costs

Report exact emitted and consumed IDs, completion/budget status, full-vocabulary first-readout differences, red/blue margins, signed binding/update contrasts, task truth and own-history equality. Comparisons after token divergence are policy outcomes unless explicitly qualified on identical histories. A numerical match to native can reproduce a native task error; task truth and emulation fidelity are separate observables.

All original dense projections and nonlinear operations remain paid. Count encoded state, ring/checkpoint/token-abort backups, decode workspace, host audit histories, support selection, score/value mixing and compilation separately. Top-four needs the full score scan; it is not fixed-size memory. Native FP32 history deletion is a diagnostic, not a serving-cost competitor. No speedup claim follows from a component MAC count or instrumented fixed-order timings.

The prior compact_append lease measured4060.1MB RSS and6378MB VRAM with one resident C model. The new all-layer arena, snapshots and audit buffers require explicit extent accounting; the inherited measured envelope is a starting point, not a proof of fit. Use one strict-preflight finite mrun CUDA lease per lane, one resident model, sequential policies and no retry or duplicate suite. The smoke is for execution integrity, not behavioral selection.

## Prospective source review

The reviewed inputs.json truth table implements x XOR u correctly. The hypothesis post, PLAN.md and launcher distinguish policy from conditional-response comparisons and support size from storage. The worker's suffix files begin with the assistant EOS151645; final driver review must verify the consume-once rule above. At this writing the driver and CUDA cache code are still under implementation, so this note does not certify their mechanics. A minor narrative correction was sent separately: the earlier20.9% pooled restoration error belongs to twelve fixed cuts, not a pooled set including the retrospective thirteenth cut.

The directional predictions are hypotheses: paired-byte full memory may preserve response better than aggressive eviction, and query-selected support may preserve a binding distinction better than a recent window. Failure of top-four would concern this law and domain; success would not establish arbitrary-query sufficiency or a forward-pass-free model. The next compiled-query contrast should remain separate so this experiment can identify the memory/read interface first.
