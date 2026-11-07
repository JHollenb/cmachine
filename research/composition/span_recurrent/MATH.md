# The memory object is a paired exponential measure

For one attention head and one permitted memory span, put a=q times the executed attention scale. The memory's response is

\[
Z(a)=\sum_i e^{a^\top k_i},\qquad N(a)=\sum_i e^{a^\top k_i}v_i,\qquad \mu(a)=N(a)/Z(a).
\]

The keys and their paired values define a finite exponential measure. Z is its partition function; N is a value-marked transform of the same atoms. The complete attention read combines these unnormalized objects with the complement before dividing. This is why retaining only the conditional span mean is insufficient.

## Aggregating atoms changes the transform

For a cluster with uniform empirical expectation E, write k=kbar+u, v=vbar+w and E[u]=E[w]=0. Then exactly,

\[
Z_j=n_j e^{a^\top\bar k_j}E[e^{a^\top u}],
\]

\[
N_j=n_j e^{a^\top\bar k_j}
\left(\bar v_jE[e^{a^\top u}]+E[e^{a^\top u}w]\right).
\]

A first-moment centroid sets the first expectation to one and the second to zero. The [prior saved-data audit](../span_response/runs/job-524febce43e5/omission-audit.json) measures both omitted terms and their signed interaction. They are not solely a denominator defect.

Expanding the centered exponential gives

\[
E[e^{a^\top u}]=1+\tfrac12 a^\top\Sigma_{kk}a+\tfrac16 E[(a^\top u)^3]+\cdots,
\]

\[
E[e^{a^\top u}w]=\Sigma_{vk}a+\tfrac12 E[(a^\top u)^2w]+\cdots.
\]

The first terms are score variance and key/value cross covariance. The full second expression is finite exponential-score/value dependence and generally needs higher moments as the query moves. A denominator correction retains none of its vector direction by itself.

One simplifying assumption does close these expressions: suppose the within-component key distribution is Gaussian and E[v|k] is affine in k. Then

\[
Z_j=n_j\exp\left(a^\top\mu_{k,j}+\tfrac12a^\top\Sigma_{kk,j}a\right),
\qquad
N_j=Z_j\left(\mu_{v,j}+\Sigma_{vk,j}a\right).
\]

This is exact for that assumed distribution, not for a generic finite native span. A nondegenerate finite empirical distribution is not a Gaussian. Moreover, full 128-by-128 key/key and value/key matrices can exceed the cost of these 34 original rows. Low-rank structure or a restricted future-query family would have to earn their approximation empirically. Adding tensors before comparing a simple paired-atom baseline would obscure the storage question.

## Keeping atoms at lower precision

The current candidate keeps every row and its value association. For an FP32 row x,

\[
s=\operatorname{FP32}(\max_d |x_d|/127),\qquad
b_d=\operatorname{clip}_{[-127,127]}\operatorname{round}_{\rm away}
\left(\operatorname{FP32}(x_d/s)\right),
\qquad \hat x_d=\operatorname{FP32}(b_ds).
\]

Zero rows use s=1 and all-zero codes. The program retains the resulting paired mixture, with reconstructed atoms khat_i,vhat_i. It does not make native dispersion or value dependence exact; the remaining discrepancy is component quantization rather than collapsing several atoms into one. This is standard compressed K/V, rather than a semantic state quotient.

Measured source-only reconstruction errors yield a prospective query-conditioned bound. Let

\[
\eta_K=\max_i\|\hat k_i-k_i\|_2,\qquad
\eta_V=\max_i\|\hat v_i-v_i\|_2,\qquad
\epsilon(a)=\|a\|_2\eta_K.
\]

Then every score changes by at most epsilon and

\[
e^{-\epsilon}\le \hat Z/Z\le e^{\epsilon}.
\]

For original and reconstructed within-span softmax probabilities p and phat, exponential reweighting with score perturbations in [−epsilon,epsilon] gives

\[
\|\hat p-p\|_1\le 2\tanh(\epsilon/2).
\]

With Vmax=max_i||v_i||, expand the mean difference using original values:

\[
\hat\mu-\mu=\sum_i(\hat p_i-p_i)v_i+
\sum_i\hat p_i(\hat v_i-v_i),
\]

\[
\|\hat\mu-\mu\|_2\le
2V_{\max}\tanh(\epsilon/2)+\eta_V.
\]

These bounds use actual FP32-decoded source errors and the declared scale, avoiding an assumption that ideal half-bin bounds survive stored-scale arithmetic exactly. Native dot-product, softmax and accumulation discrepancies are separate. The bounds can be loose and have no boolean authority over exploratory evidence. Unbounded arbitrary query norms do not admit a fixed epsilon from this argument.

## What composes, and what still grows

Unnormalized response objects compose by addition:

\[
(Z_{A\cup B}(a),N_{A\cup B}(a))=(Z_A(a)+Z_B(a),N_A(a)+N_B(a))
\]

for disjoint permitted supports at the same query and numerical contract. A stable implementation also retains its exponential gauge. A byte-row encoding admits an append operation: quantize a newly written paired K/V row and add its encoded atom. That is a simple incremental memory mechanism, but its number of rows grows with context. The present experiment repeatedly reads a fixed compiled span; newly written complement rows remain native FP32.

Exact arbitrary-query preservation also limits what constant-size aggregation could mean. Distinct finite exponentials e^(a^T k_i) are linearly independent on an open query domain: choose a line whose projections of all distinct keys are distinct, then the resulting one-dimensional exponential functions are independent. Thus an exact finite exponential-sum representation of Z generally retains each distinct key and its multiplicity. Equal keys merge exactly by retaining multiplicity for Z and the sum of associated values for N, equivalently a count and mean value. This is a statement about that representation class and unrestricted exact queries, not a general encoding-size impossibility theorem, nor a refutation of approximate, low-rank or restricted-query chat state.

Finally, a good local response does not by itself control autonomous chat. If e_t is state discrepancy, a local analysis of a declared continuous-state transition under common token inputs can have the form

\[
\|e_{t+1}\|\le L_t\|e_t\|+\delta_t,
\]

where delta is the current approximation defect and L_t captures the actual continuation's sensitivity. The product of later sensitivities transports each defect forward. Greedy token selection introduces a discrete boundary: small logit errors can change the token when the native margin is small. After token divergence, the account needs an input-mismatch term or an explicit switched-system treatment. An own-token recurring test therefore supplies evidence that a one-use same-query assay cannot. It still needs actual histories, later states, wider contexts and resource measurements before becoming a claim of compact sufficient chat.

These two wording clarifications follow Astra's post-submission review. The executed payload preserves the original mathematical note; no code, raw result or producer report is changed.
