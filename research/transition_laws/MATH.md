# Transition laws for an all-layer Qwen MLP factor map

This note fixes the mathematical contract for the `transition_laws` experiment. It is a
factorized MLP experiment with an exact native consumer around it. It is not yet a claim that
chat has a small closed state, and it is not a claim that the all-layer cache work already found
this law. The older all-layer program established callback, cache, rollback, and own-token
execution mechanics; this experiment asks whether a lower-cost *operator* can advance the same
state while the native downstream consumer remains available.

Related evidence:

- [`compact_all/FINDINGS.md`](../composition/compact_all/FINDINGS.md) records exact all-layer
  callback and byte-cache execution, but also shows that dense weights and linear work remained
  paid.
- [`contextual-union/FINDINGS.md`](../../experiments/2026-11-06-contextual-union/FINDINGS.md)
  records rank-64/128 receiver-aware factors at a selected MLP layer, including own-state drift
  and the difference between selected-layer MAC savings and whole-model cost.
- [`operand_composition/analysis.py`](../operand_composition/analysis.py) defines the recent
  four-arm same-parent MLP/query factorial and its same-token native replay controls.
- [`STACKED_STATE_PLAN.md`](../STACKED_STATE_PLAN.md) states why an interior layer state needs
  the residual phase and valid K/V ownership in addition to a final hidden vector.

## 1. The native transition

Index layers by \(\ell\in\{0,\ldots,27\}\), and let \(x_{t,\ell}\in\mathbb R^d\) be the
residual stream for token position \(t\) at the MLP input of layer \(\ell\). The attention
subgraph has already produced the post-attention residual at this point. With the usual Qwen
RMSNorm convention,

\[
 r_{t,\ell}=N_\ell(x_{t,\ell})
 =\gamma_\ell\odot
 \frac{x_{t,\ell}}{\sqrt{d^{-1}\sum_{j=1}^d x_{t,\ell,j}^2+\epsilon_\ell}}.
\]

The native gate, up, SiLU product, down projection, and residual update are

\[
 \begin{aligned}
 g_{t,\ell}&=G_\ell r_{t,\ell}, &G_\ell&\in\mathbb R^{h\times d},\\
 u_{t,\ell}&=U_\ell r_{t,\ell}, &U_\ell&\in\mathbb R^{h\times d},\\
 m_{t,\ell}&=\operatorname{SiLU}(g_{t,\ell})\odot u_{t,\ell},\\
 w_{t,\ell}&=D_\ell m_{t,\ell}, &D_\ell&\in\mathbb R^{d\times h},\\
 x_{t,\ell+1}&=x_{t,\ell}+w_{t,\ell}.
 \end{aligned}
\]

Here \(d=1536\) and \(h=8960\) for the pinned Qwen2.5-1.5B profile. Biases, if present in the
runtime projection, belong in the definitions of \(g\) and \(u\); the factor files must state
whether they are folded into a bias or retained as a separate native term. Across all 28 layers,
the experiment keeps \(N_\ell\), SiLU, \(D_\ell\), the residual addition, attention, K/V writes,
and every operation outside each gate/up pair native. Only the two gate/up linear maps at each
layer are replaced; later MLP layers are not assumed native when their own gate/up pair is reached.

At a token boundary, the complete serving state is more than \(x\):

\[
 S_t=(K_{0:27,<t},V_{0:27,<t},\;t,\;u_t,\;\text{pending/readout policy},\;\text{token history}).
\]

The current token's K/V rows are appended during its layer traversal. Thus a layer-interior
state must also identify the phase and which current rows have been committed. An approximate MLP
law is a transition law only when its resulting residual is passed into the rest of the native
layer, its K/V writes are formed from that resulting state, and the next token is selected from
the candidate's own logits. A native shadow that consumes supplied token IDs is a conditional
consumer test, not an autonomous continuation.

## 2. The rank-\(r\) factor law

For each layer, choose \(E_\ell\in\mathbb R^{r\times d}\) and low-dimensional output maps
\(L^g_\ell,L^u_\ell\in\mathbb R^{h\times r}\). The candidate transition is

\[
 \begin{aligned}
 z_{t,\ell}&=E_\ell r_{t,\ell},\\
 \widehat g_{t,\ell}&=L^g_\ell z_{t,\ell},\\
 \widehat u_{t,\ell}&=L^u_\ell z_{t,\ell},\\
 \widehat m_{t,\ell}&=\operatorname{SiLU}(\widehat g_{t,\ell})
                        \odot\widehat u_{t,\ell},\\
 \widehat w_{t,\ell}&=D_\ell\widehat m_{t,\ell},\\
 \widehat x_{t,\ell+1}&=x_{t,\ell}+\widehat w_{t,\ell}.
 \end{aligned}
\]

The rank is \(r=256\). “Shared input” means that the gate and up maps at a given layer use the
same \(z=Er\); it does not silently mean one universal \(E\) across all 28 layers. Unless the
runtime protocol explicitly says otherwise, the typed object is
\(\{E_\ell,L^g_\ell,L^u_\ell\}_{\ell=0}^{27}\), with one map triple per layer. A single
cross-layer \(E\) would be a separate, stronger hypothesis and must be named as such.

The dense gate/up/down linear work is

\[
 C_{\rm dense}=3dh.
\]

The factor work, retaining native \(D\), is

\[
 C_{\rm factor}=dr+2hr+dh,
 \qquad
 \Delta C= C_{\rm dense}-C_{\rm factor}
                 =2dh-r(d+2h).
\]

For the pinned dimensions,

\[
 \begin{array}{rcl}
 dr&=&393{,}216,\\
 2hr&=&4{,}587{,}520,\\
 dh&=&13{,}762{,}560,\\
 C_{\rm factor}&=&18{,}743{,}296,\\
 C_{\rm dense}&=&41{,}287{,}680,\\
 \Delta C&=&22{,}544{,}384.
 \end{array}
\]

Thus the factor graph is 45.39% of the selected MLP linear work and saves 54.61% of that
selected work, before accounting for memory traffic, launch overhead, attention, normalization,
and the retained down projection. Per layer the factor coefficients contain
\(rd+2rh=4{,}980{,}736\) FP32 values, or 19,922,944 bytes. Across 28 layers that is
139,460,608 FP32 values, or 557,842,432 bytes before metadata. These are storage arithmetic, not
a measured latency claim.

The stated native per-token ledger is 1,543,569,408 MACs. If that ledger includes exactly the
28 \(3dh\) MLP terms above, replacing all those terms changes the arithmetic ledger by
\(28\Delta C=631{,}242{,}752\) MACs. The remaining model operations still execute. The report must
publish its counted categories instead of presenting this subtraction as whole-model speedup;
the full attention history, K/V writes, output head, and runtime overhead can dominate.

## 3. What E0 and E1 mean

Let \(R_{\rm nat}\) be the matrix of normalized MLP inputs used for fitting, with rows ordered by
case, turn, token clock, and layer according to the sealed manifest. This ordering is part of the
experiment, not an incidental NumPy order. The equations below assume uncentered PCA, because the
runtime law has no mean subtraction. If the producer centers, it must retain and serve
\(\mu_\ell\) and use \(E_\ell(r-\mu_\ell)\); silently centering during fitting and omitting the
offset at serving would be a different operator.

For layer \(\ell\), define

\[
 C_{\rm nat,\ell}=R_{\rm nat,\ell}^{\mathsf T}R_{\rm nat,\ell}.
\]

Let \(Q_{0,\ell}\in\mathbb R^{d\times r}\) be the rank-\(r\) basis returned by the fixed
uncentered randomized SVD of \(R_{\rm nat,\ell}\), with oversampling \(q=272\), two power
iterations, a fixed per-layer seed, and max-pivot-positive row signs. It approximates the top
right singular subspace; it is not the exact optimum of a full SVD. Then
\(E_{0,\ell}=Q_{0,\ell}^{\mathsf T}\), up to the declared sign/order convention. Its source projection is

\[
 P_{0,\ell}=Q_{0,\ell}Q_{0,\ell}^{\mathsf T},
 \qquad
 \|r-P_{0,\ell}r\|_2
\]

for a new receiver \(r\). This is an input reconstruction statistic, not a bound on the
nonlinear MLP write error: SiLU and the gate/up product can amplify a small projection error.

The protocol uses 384 native source rows from six two-turn conversations, with 64 evenly spaced
rows per conversation. E0 fits a 768-row matrix made from two equally weighted copies of the
native source rows, each scaled by \(1/\sqrt2\); this makes the duplicated source distribution
explicit without changing its PCA subspace. E0 then generates its own source trajectories. Let
\(R_{\rm own,0}\) be the 384 rows collected from those E0 trajectories. E1 fits the same-rank,
same-size PCA to

\[
 R_1=\begin{bmatrix}R_{\rm nat}/\sqrt2\\R_{\rm own,0}/\sqrt2\end{bmatrix},
\qquad E_{1,\ell}=Q_{1,\ell}^{\mathsf T},
\]

where \(Q_{1,\ell}\) uses the same randomized-SVD dimensions, per-layer seed rule, and sign
canonicalization. Both E0 and E1 are randomized-SVD approximations rather than exact full-SVD
optima.

This is a declared source-stage refresh, not an online optimizer. It is valid only if the E0 own
rows and all fitted maps are sealed before any held-out case is tokenized or inspected. Equal row
count alone does not guarantee equal *case* weighting; the manifest should retain the case/turn
membership so a later analysis can report whether long turns dominate the fit.

For a candidate map \(E_i,L^g_i,L^u_i\), the local errors worth retaining are

\[
 \delta_z=\widehat z-z_{\rm native\,projection},\quad
 \delta_g=\widehat g-g,\quad
 \delta_u=\widehat u-u,\quad
 \delta_w=\widehat w-w,
\]

followed by residual, K/V, attention-context, logits, and token differences. The native \(g,u,w\)
on the candidate's actual input are useful diagnostics: they separate “the factor law is wrong
at this receiver” from “the candidate trajectory reached a different receiver.” They do not grant
the candidate a native state; they are paid shadow computations.

## 4. Native-shadow KL and own-state meaning

At each of three monitored candidate cases, the candidate policy generates its own token IDs. A
native-shadow session is then fed those exact IDs and the same user-history boundaries. For a
decision clock \(q\), let \(p^{\rm cand}_q\) be the candidate distribution and
\(p^{\rm nat|ids}_q\) the native distribution after consuming the candidate's IDs. The primary
conditional discrepancy is

\[
 D_{\rm KL}(p^{\rm nat|ids}_q\,\|\,p^{\rm cand}_q)
 =\sum_v p^{\rm nat|ids}_q(v)
   \log\frac{p^{\rm nat|ids}_q(v)}{p^{\rm cand}_q(v)}.
\]

The reverse direction may be retained as a diagnostic, but the direction and smoothing convention
must be fixed before reading outcomes. Average equally over case and turn rather than letting the
longest answer dominate. Keep full distributions and decision IDs, not only the mean KL.

This same-ID shadow answers: “given the candidate's discrete history, how different is the native
consumer at this clock?” It does not answer: “would the native model have produced the candidate
history?” For that, preserve a separate native-owned trajectory. If the candidate and native take
different tokens, a comparison at equal wall-clock position mixes two effects: operator error and
history divergence. The correct decomposition is:

\[
 \text{candidate} \longrightarrow
 \begin{cases}
 \text{native shadow on candidate IDs} & \text{conditional consumer error},\\
 \text{candidate next IDs} & \text{closed-loop policy behavior},
 \end{cases}
\]

with the native-owned path retained as a third reference. Do not call a lower shadow KL an
autonomous improvement unless the candidate also advances its own state and its token history
under the declared serving policy.

## 5. What the experiment can and cannot establish

The six source conversations define the fit distribution; the three monitored cases are a
development diagnostic; the six fresh cases are held out for E0/E1 comparison. The native
shadow's use of candidate IDs is allowed because it is a consumer control, but its outputs must
not be used to refit E1 or select the held-out policy. The selection rule should be explicit: if
the protocol chooses lower monitor KL, report that as a controller choice and retain the full
held-out E0/E1 results, including regressions.

The experiment can establish a qualified trend that a rank-256 gate/up law advances a complete
Qwen state with a measured local and downstream error profile, and whether adding E0 own-state
rows improves that profile on fresh cases. It can establish an arithmetic cost ledger for the
factorized MLP. It cannot by itself establish:

1. a universal low-dimensional chat state;
2. a source-free or forward-pass-free emulator (the original dense weights, native D, attention,
   K/V, and readout remain available);
3. an exact all-layer cache compression law (K/V remain native and full); or
4. semantic chat competence from supplied two-turn templates or token agreement alone.

Rank-256 source reconstruction is not transition closure. A future receiver can leave the source
span, and the nonlinear gate/up product can make the omitted direction matter. Closure requires
measuring candidate residuals, K/V writes, and later readouts on own states. A low shadow KL on
three cases is a useful trend, not a theorem.

## 6. Minimum independent verification

Before scientific interpretation, the analyzer should independently verify:

- model, tokenizer, runtime, factor, and input hashes; layer count 28; shapes
  \(E:[256,1536]\), \(L^g,L^u:[8960,256]\), native \(D:[1536,8960]\), and FP32 layout;
- E0/E1 rank, SVD/PCA convention, row scaling, case/turn membership, and seal time relative to
  monitor and held-out tokenization;
- source/own/held-out disjointness and the exact 384/384/768 row counts;
- candidate token authority, native-shadow same-ID replay, native-owned reference, prefix/KV
  lineage, position, and pending/readout policy;
- finite values, full-vocabulary logits, decision-logit timing, and per-layer factor/native
  physical call counts;
- the factor MAC identity above and a category-separated cost ledger.

Numerical FP64 recomputation is an independent diagnostic. It should preserve signed effects and
rounding differences; it is not an arbitrary scientific pass/fail gate. Mechanics failures
(wrong model, shape, token history, seal ordering, or replay identity) are separate from a poor
factor trend. A poor trend is evidence about this rank/law under this context distribution, not
evidence that transition laws are absent.

The most informative result is therefore a paired table: E0 versus E1, source versus own versus
held-out, and native-owned versus native-shadow versus candidate-owned, with local \(z/g/u/w\),
K/V, logits/KL, token IDs, and cost shown together. That table keeps the exact consumer, the
advancing state, and the approximation law attached to one another.
