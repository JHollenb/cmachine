# Compile the response on the state that will consume it

For actual compact memory M, let its counted prototype/recent slots be (k_r,v_r,c_r). Per head,

\[
p_r(q)=\frac{c_r\exp(q^Tk_r/\sqrt d)}{\sum_s c_s\exp(q^Tk_s/\sqrt d)},\qquad
A_M(q)=\sum_r p_r(q)v_r.
\]

Native same-text Q and the cheap Q are already RoPE-transformed in the archived arrays. Their lower normalized receivers agree, but cheap Q uses B128 coefficients z rather than the full dense projection. For a future fit write q_F=R_t(Fz+b), holding keys, values, memory and bias fixed.

## History and current computation

Let Z(H,q,R) be full initial logits from a restored historical parent H, current query q and current reader R through the native O/MLP/later suffix. N and C label native and compact conditions; B labels bounded compiled-key memory. Then

\[
Z(C,C,B)-Z(N,N,N)
=[Z(C,C,B)-Z(C,N,N)]+[Z(C,N,N)-Z(N,N,N)].
\]

These are exact signed vector differences at the measured endpoints. For the linear first-token contrast m=Z[Willow_1]−Z[Cobalt_1], the256-token account is

\[
-5.514823914=-3.307491302-2.207332611.
\]

The first term changes the current reader at fixed compact history. The second changes historical consumer state after an exact current selected-layer result. They are conditional terms, not causal percentages. Exact current norm, attention and post-MLP residual across parents place the remaining initial difference in higher-layer historical K/V.

The rescued native-Q/bounded read contributes+2.078393936relative to the full-native current read on compact history. Combined with the−2.207332611historical shift, its margin is only−0.128938675below the native baseline. Here an approximate local read compensates for the altered consumer. This does not establish that compensation is globally optimal or remains useful after advancing state.

## Factor the proposed fitting objective

On each original source-owned decision let

\[
a_F=A_{M_C}(q_F),\quad a_*=A_{M_C}(q_N),\quad a_N=A_{M_N}(q_N).
\]

The proposed same-memory target minimizes L_same=E_w||a_F−a_*||². The previous full-native target minimizes L_full=E_w||a_F−a_N||². With e=a_F−a_* and d=a_*−a_N,

\[
L_{full}=L_{same}+2\mathbb E_w[e^Td]+\mathbb E_w\|d\|^2.
\]

The retained source arrays give

\[
72.531326126=17.337200973+8.859180975+46.334944177.
\]

The weighted finite closure remainder is8.16e−16. These unnormalized response-space squared norms are source diagnostics, not KL, answer scores, causal shares or a fitted improvement. A smaller objective with a different target does not imply a better program. The target has not yet passed through the source native consumer.

Nor is the teacher displacement an irreducible approximation error: another query might compensate for the retained memory differently. Equal source/turn probability mass does not equalize response energy or gradient magnitude. [Per-source/turn losses and energies](source-targets-v1/profile.json) remain visible, as do signed per-row cross terms; neither an aggregate sign nor an energy norm can certify downstream task behavior.

Per head the reader Jacobian is

\[
J_q=\frac{\partial A_M}{\partial q}
=\frac1{\sqrt d}\sum_r p_r(v_r-A_M)k_r^T.
\]

Holding the source memory and coefficients fixed, a surrogate gradient is

\[
\nabla_F L_{same}=2\sum_i w_i R_i^T J_i^T e_i z_i^T.
\]

The full-native target adds2sum_i w_i R_i^T J_i^T d_i z_i^T. This extra term pulls toward a different memory's response even when the compact memory already answers the query compatibly. The drawer result motivates testing that pull, not assuming it is always harmful. The actual FP32 reader rounds its output; this derivative is a smooth FP64 surrogate, with numerical differences retained and actual suffix feedback required.

The target data use three sources, two turns and n_cjdecisions per turn: w_cjt=1/(3·2·n_cj). All420source clocks replay the actual bounded read;162scored decisions receive teachers. No drawer row, newly generated state, optimizer update or model invocation enters this preparation. Future candidate rollouts need fresh memories and queries; the fixed-memory derivative is a local proposal, not differentiation through the entire recursive state machine.

## The simplifying assumption remains a hypothesis

An economical code E and transition G would need E(T(S,u))≈G(E(S),u) and O(S)≈D(E(S)) on the program's admitted own-reachable states. This requires preserving distinctions used by future queries and updating them after model outputs and new user turns. A compatible current response is one boundary of this diagram. Neither an exact current read nor the existence of a rescuing paid query establishes that E contains enough information to produce that query or advance the rest of the model cheaply.
