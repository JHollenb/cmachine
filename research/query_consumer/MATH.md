# Select through the complete source computation

The immutable image family is Fi=FLS+αiD, rounded to FP32 in the preceding experiment. Its doses are {0,−1,1/4,1/2,1,2} along one previously computed source-response direction. We do not recompute D or select a new direction from current targets.

For each original source trajectory c, let ut be the authenticated prompt plus its historical eight generated token IDs. Each policy carries its own complete state under these fixed inputs:

\[
S_{t+1}^{(i)}=T_{F_i}(S_t^{(i)},u_t),\qquad \ell_t^{(i)}=H(S_{t+1}^{(i)}).
\]

State includes every layer's K/V, position, selected compact memory and other mutable runtime data. The selected layer forms z=Bx from its actual lower-layer receiver, q=Rt(Fiz+bq), compiled K and native V. O/MLP and upper layers recompute normally. Different query images therefore affect both the current logits and later upper cache reads, even when source IDs stay fixed.

The native reference follows the same IDs with ordinary native Q/K/V and full memory. Write pt=softmax(ℓN,t) and qt(i)=softmax(ℓi,t). Select

\[
i_* = \arg\min_i\frac13\sum_{c=1}^3\frac1{n_c}\sum_{t\in c}\mathrm{KL}(p_t\|q_t^{(i)}).
\]

KL is measured in nats, summed over all 151,936 vocabulary entries. It is not divided by vocabulary size. Exact ties choose smaller absolute original dose, then smaller index. Preserve the entire dose curve and per-trajectory/header/body/historical-suffix results. Equal weighting of three trajectories does not turn their 193 correlated clocks into independent examples.

The reader objective used previously measured the error before O/MLP and the remaining model. This objective measures the full source distribution and candidate-specific cache consequences. In a local fixed-input linearization,

\[
\delta S_{t+1}=A_t\delta S_t+B_t\delta F,
\qquad \delta\ell_t=C_t\delta S_{t+1},
\]

and

\[
\mathrm{KL}(p_t\|q_t)\simeq\frac12\delta\ell_t^T[\operatorname{diag}(p_t)-p_tp_t^T]\delta\ell_t.
\]

The state recurrence transports the effect through prior writes. The probability geometry weights the consequences through the actual vocabulary consumer. The finite frozen-image panel measures this without computing those full-model Jacobians or adding an optimizer.

This is still source-distribution fitting. All source histories are forced, so a proposal is not allowed to replace ut with its own argmax during selection. Autonomous token switches produce a different recurrence and need a separate assay. Target execution does use each policy's own emitted-and-consumed IDs, and target outcomes cannot change i*.

C evaluates stable FP64 log-softmax differences from actual FP32 CUDA logits. A separate Python calculation consumes exact full rows while the process waits; both implementations check row hashes and full-vocabulary metrics. Those complete streams are transient, so the durable independent logit replay is limited to nine selected clocks per policy. Source selected-layer operands, all scalar curves, live comparison records and row/stream hashes remain in custody.

The C invocation request binds the frozen source/image family and runtime before the source phase. The source selection/image seal then precedes target parsing. The returned driver binds the selection hash and the later target-manifest hash; the whole-invocation Saturn receipt covers that returned result. This does not claim that the target manifest existed at the initial source boundary or that a whole C cache was restored by an image-only Act.

Changing this offline selector costs 1,351 source advances plus the separate native sham. Serving the selected image costs exactly as much as the other frozen images. It cannot manufacture missing information in Bx, replace the dense producer of x, or certify that a native answer is true.
