# Response closure, selection and the remaining operand obligation

These are real-arithmetic statements. Exact runtime replay, finite reductions and observed response errors are separate measurements. Astra checked the constructive law and bound during the experiment review. The feature representation below is a sufficient condition, not a representation learned or demonstrated by this panel.

## Phase-specific closure

Let the original transition be F_phi(S,u), the represented state C_phi(S), the proposed update G_phi and the required observation O_phi. Exact closure requires

\[
C_{\phi^+}(F_\phi(S,u))=G_\phi(C_\phi(S),u),\qquad O_\phi(S)=H_\phi(C_\phi(S)).
\]

Equal represented states must have equal required observations and equal successor codes under every admitted input. Conversely, these two fiber conditions define H and G consistently. Applying them repeatedly proves preservation of every finite admitted future by induction. Choosing identity C and the entire original F satisfies the equations without saving work. Representation size and execution cost are independent obligations.

For approximate transitions, if a comparison metric admits native local Lipschitz constants L_t and defects delta_t on the states actually reached, then e_(t+1) <= L_t e_t + delta_t. Repeated substitution gives a sum of defects multiplied by subsequent L factors. A small one-read defect alone does not establish a small multi-layer or multi-token error. Greedy token decisions add a discontinuity: a logit infinity-norm error below half the native top-two margin is sufficient to preserve that particular decision, but is not a universal continuation bound.

## Selected support identity

At one fixed query and paired bank, let p be normalized full attention, S selected support and epsilon the p mass outside S. Define A_S and A_T by independently renormalizing the selected and omitted parts. For nonempty masses,

\[
A=(1-\epsilon)A_S+\epsilon A_T,\qquad A-A_S=\epsilon(A_T-A_S).
\]

This follows by partitioning the numerator. The selected-weight renormalization is part of the effect. The omitted contribution epsilon A_T alone is not the complete error. Zero-mass parts contribute zero and can be defined by a continuous convention. A changed key codec changes p; a changed value codec changes the paired values. Those effects are distinct from selecting support in the same decoded bank.

## Constructive sufficient-statistic hypothesis

Suppose the positive attention kernel has an economical finite feature representation on a declared query/key domain:

\[
K(q,k)=e^{q^Tk/\sqrt d}=\phi(q)^T\psi(k),\qquad \phi,\psi\in\mathbb R^r.
\]

Store

\[
z=\sum_i\psi(k_i),\qquad B=\sum_i\psi(k_i)v_i^T.
\]

The exact response and append become

\[
A(q)=\frac{\phi(q)^TB}{\phi(q)^Tz},\qquad
z^+=z+\psi(k_*),\quad B^+=B+\psi(k_*)v_*^T.
\]

The state contains r(d_v+1) scalars per K/V group, independent of the number of rows. Equal z and B give equal responses to every admitted query; appending the same pair gives equal successors. This proves closure of this memory interface. Feature evaluation, the rank-one write and read must still be economical. The model must still produce its own q, k and v. The theorem does not prove closure of the entire model.

## Approximate positive kernel bound

Let K_i > 0 and Khat_i = K_i(1+delta_i), with |delta_i| <= epsilon < 1 for every admitted pair. With p_i=K_i/sum_j K_j and A=sum_i p_i v_i,

\[
\widehat A-A=\frac{\mathbb E_p[\delta_i(v_i-A)]}{1+\mathbb E_p\delta_i}.
\]

The identity follows by subtracting A and using E_p(v_i-A)=0. Triangle inequality and the lower denominator bound give, in any vector norm,

\[
\|\widehat A-A\|\leq\frac{\epsilon}{1-\epsilon}\mathbb E_p\|v_i-A\|.
\]

The relative assumption guarantees every Khat_i and the nonempty denominator are positive. Individual feature coordinates need not be positive. Large signed feature cancellations can still be numerically unstable. If values are approximated too, add E_phat ||vhat_i-v_i|| to the bound. Neither low captured matrix-energy error nor a low mean score error proves this uniform relative condition.

## Why finite rank is a substantive assumption

On a finite query/key grid, an exact factorization exists with rank equal to the kernel matrix rank; that rank can be as large as either grid dimension. Existence is not compression.

For arbitrarily many distinct scalar keys a_i on an open scalar query interval, the functions exp(a_i t) are linearly independent. If sum_i c_i exp(a_i t)=0 throughout that interval, differentiating orders 0 through m-1 at one interior t_0 gives a Vandermonde system in c_i exp(a_i t_0). Its determinant is product_(i<j)(a_j-a_i), which is nonzero for distinct keys. Therefore all c_i vanish. Arbitrarily large m excludes a fixed finite separable feature dimension for this unrestricted exact exponential kernel. The same argument applies along a generic line in a vector query domain with distinct key projections.

This is a statement about finite separable kernel features. It is not an impossibility theorem for every nonlinear or approximate memory representation. Restricting reachable queries, accepting approximation or allowing representation growth changes the question.

## What this experiment tests

Byte-full preserves paired atoms with reduced precision. Top-four preserves the full bank and selects by each live query. Recent-four tests a particular bounded eviction rule. The deletion control tests use of pre-question context. Conditional replay isolates codec and selection on each policy's own observed query. None constructs phi/psi, proves finite rank, establishes a uniform kernel error bound, or makes native operand formation free. These distinctions govern the interpretation of [the results post](../../../obsidian/blog/2026-11-06-154253-a-hypothesis-for-cheaper-model-state-transitions.md).
