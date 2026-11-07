# A query-producing memory program

The executable boundary is one attention component of original Qwen2.5-1.5B-Instruct at zero-based layer 17. Its normalized input comes from the candidate's own full native lower-layer computation. The small program forms Q/K/V, advances its own memory, reads that memory and supplies the attention return to the original O/MLP and later consumers. This is a repeated component computation, with an approximate query response and a paid dense scaffold.

## State and order

At token t let x_t be the actual layer receiver and n_t its original weighted RMS-normalized input. For row-orthonormal B∈R^{r×1536}, compile

$$
C_q=W_qB^T,\quad C_k=W_kB^T,\quad C_v=W_vB^T.
$$

The dynamic program computes

$$
c_t=Bn_t,\quad
\widehat q_t=R_t(C_qc_t+b_q),\quad
\widehat k_t=R_t(C_kc_t+b_k),\quad
\widehat v_t=C_vc_t+b_v.
$$

Here R_t is the original head-wise RoPE at the actual native position, and b denotes the original bias if present. Keys are stored after that rotation. Prototype means are never rotated a second time. With e_t=(I−B^TB)n_t, the real pre-bias query defect is −W_qe_t, and likewise for K/V. Receiver energy alone does not determine these defects; weight amplification and the downstream consumer matter.

For memory M_t,

$$
M_{t+1}=U(M_t,\widehat k_t,\widehat v_t,t),\qquad
\widehat a_t=\mathcal R(M_{t+1},\widehat q_t).
$$

The program appends the current row before reading. The resulting attention return enters the original O projection and residual/MLP computation; its state changes can therefore change logits, tokens and subsequent branch-owned context. Each branch consumes its own emitted IDs, including EOS and later exact user/header tokens. It receives no native target-query or memory refresh. Full diagnostic histories are outside the reader's interface.

The retained state has a 16-row exact recent ring per KV group and 16 older prototypes. A retired row fills an unused prototype, then merges with the nearest current key mean by squared Euclidean distance. Each prototype stores (count, key sum, value sum), using FP64 sums. This supplies a total, deterministic update on the admitted finite domain; it does not claim that the selected summaries are sufficient for native attention.

## Memory as a response function over future queries

For one grouped-query head, native attention is

$$
Z(q)=\sum_i e^{q^Tk_i/\sqrt d},\quad
N(q)=\sum_i e^{q^Tk_i/\sqrt d}v_i,\quad
a(q)=N(q)/Z(q),\qquad d=128.
$$

A fixed-query numerator/denominator pair cannot generally answer a new query. Our memory instead retains a small positive discrete measure. Prototype j has mass m_j, position k̄_j and value v̄_j; recent rows are mass-one points. It defines

$$
\widehat Z(q)=\sum_j m_j e^{q^T\bar k_j/\sqrt d},\quad
\widehat N(q)=\sum_j m_j e^{q^T\bar k_j/\sqrt d}\bar v_j,\quad
\mathcal R(M,q)=\widehat N(q)/\widehat Z(q).
$$

The implementation evaluates scores qᵀk̄_j/√d+log(m_j), subtracts their maximum and normalizes. All masses are positive, so the shifted denominator is at least one. This answers arbitrary new finite queries through the retained response function, without a dictionary of future query vectors or access to original retired K/V rows. At most 32 entries are examined per head.

## Exactly what compression drops

Within a cluster write k_i=k̄+δk_i, v_i=v̄+δv_i and s_i=qᵀδk_i/√d. Using the uniform mean within that cluster,

$$
A(q)=\mathbb E[e^{s_i}],\quad
Z_C(q)=m e^{q^T\bar k/\sqrt d}A(q),
$$

$$
N_C(q)=m e^{q^T\bar k/\sqrt d}\left[A(q)\bar v+\operatorname{Cov}(e^{s_i},v_i)\right].
$$

The program substitutes A(q)=1 and covariance=0. In real arithmetic, Jensen gives A(q)≥1 because mean s_i=0. The omitted normalization mass is nonnegative, but normalized output error can have either sign because both allocation and values change. Equal key/value means do not identify the full response function. The missing covariance is a direct reason a prototype can retain mean content and still lose query-selected content.

Expanding around q=0, equivalently in the centered scores s_i, gives

$$
A(q)=1+\tfrac12\mathbb E[s_i^2]+\tfrac16\mathbb E[s_i^3]+\cdots,
$$

$$
\operatorname{Cov}(e^{s_i},v_i)
=\mathbb E[\delta v_i\,\delta k_i^T]q/\sqrt d+O(\|q\|^2).
$$

This suggests a measured next refinement: retain a few consumer-relevant key/value covariance directions or split clusters under query sensitivity. A full 128×128 covariance per prototype would overwhelm the current small state; the expansion is a hypothesis for choosing a cheap correction, not permission to allocate every moment. Dense target-derived corrections would change the claim and must be paid explicitly.

## Two different closure statements

The approximate program's update/read law is explicit: its next memory and attention are computable from its own state and new operands. Independent byte replay tests that implementation claim. Native quotient closure would additionally require, for an encoding C,

$$
C(T(S,u))=\widehat T(C(S),u),\qquad
O(S,u)=\widehat O(C(S),u)
$$

over the admitted future inputs and writes. Our numerical discrepancies show that this stronger native-equivalence claim has not been earned. Exact execution of a chosen approximate algorithm is distinct from exact emulation of the model. The full residual receiver, normalization, other 27 layers' memory and output computation still belong to the native scaffold; this experiment does not discover their compact transition.

## Choosing a source objective

The first compiler uses 193 normalized source receivers and pivoted uncentered Gram–Schmidt. A separate checkpoint-only compiler takes a randomized right subspace of J=[Wq/‖Wq‖F;Wk/‖Wk‖F;Wv/‖Wv‖F]. Its isotropic Frobenius objective does not weight actual input frequency or the attention/O/MLP consumer. The measured compiler comparison demonstrates that these objectives differ consequentially.

A data-weighted linear target would minimize

$$
\sum_{W\in\{W_q,W_k,W_v\}}\alpha_W\operatorname{tr}\left[W(I-P)\Sigma_n(I-P)^TW^T\right],\qquad P=B^TB.
$$

An actual attention objective must also account for query/key coupling and values; a downstream objective may include the O projection, residual and MLP. Such fitting must use source/development evidence and then face new own-state histories. It must not substitute a native receiver after the compact program has already changed that receiver.

## Costs and boundaries

Native selected QKV costs 3,145,728 nominal MAC/update. The compiled path costs(1536+2048)r=3584 r:114,688 at 32 and 458,752 at 128. Original bias/RoPE and the rest of the model remain. Against 1,543,569,408 native learned-linear MAC/token, these reductions save 0.1964% and 0.1741% respectively. An extra coordinate-projection call means 198 rather than 197 matvec calls despite fewer products.

The complete bounded numerical memory is 98,464 bytes, versus 2048 tvalid native selected-layer bytes and 1,048,576 bytes at 512-token capacity. At the tested 94–124 token lengths it saves roughly 49–61% of logical valid selected memory. Its size first undercuts that valid-prefix count at 49 tokens. The original selected global arena remains allocated but zero, unwritten and unread; the full original model bank also remains resident. Diagnostic recording is separately paid O(t)storage.

Reading score/value products change from 3072t per update to at most 3072×32. Over n updates the native account is 3072 n(n+1)/2; the bounded account is 3072Σ_{t=1}^n min(t,32). Prototype routing adds 4096 distance products per update after 32, alongside CPU exponentials, logarithms, divisions and sums. The current wrapper transfers 14,336 serving bytes between CPU/GPU per update, plus diagnostic transfers. Compilation, source acquisition, native scaffold, timing and actual allocation must remain in the ledger. Product savings alone establish no latency gain.

The admitted session has capacity 512; count and position integrity are tested within that bound. This is repeated state advancement, not an unbounded-lifetime serving claim. A future standalone participant must own allocation and checkpoint/rollback of its compact state rather than inherit the current wrapper's global cache arena or cursor-only checkpoint assumptions.
