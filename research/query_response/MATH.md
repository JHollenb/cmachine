# Query as control of a memory response

At a fixed receiver, z=Bx. The mutable program is F, with pre-rotation query Fz+bq. The actual reader consumes q=Rt(Fz+bq), where Rt is half-split RoPE. All six query heads sharing a KV group see the same nodes, but choose their own probabilities.

For a compact source state, let node keys/values be ki,vi and counts mi. Recent rows have mi=1; prototypes carry count, mean key and mean value. With d=128,

\[
s_i=q^Tk_i/\sqrt d+\log m_i,\quad p_i=\operatorname{softmax}(s)_i,\quad a(q)=\sum_i p_i v_i.
\]

Key-only prototype assignment makes these source nodes independent of the query image. This permits a small exact derivative of the CPU64 source reader without differentiating through clustering. It does not imply target states stay fixed after tokens diverge.

The source target aN is native-full CPU64 attention reconstructed from the original weights and original source receivers. Reset the clock and memory at each of the three source trajectories. Define Ec as that trajectory's total squared teacher-response norm and let wt=1/(3 Ec). The loss is

\[
L(F)=\sum_t w_t\|a_t(R_t(Fz_t+b_q))-a_t^N\|^2.
\]

This weighting balances three correlated trajectories by their teacher energy, rather than over-weighting individual tiny-response rows. It uses no fresh target outputs or labels.

The response Jacobian is

\[
J_q=\frac1{\sqrt d}\sum_i p_i(v_i-a)k_i^T.
\]

For e=a−aN,

\[
g_q=2wJ_q^Te=\frac{2w}{\sqrt d}\sum_i p_i[e^T(v_i-a)]k_i,
\quad g_{\rm pre}=R_t^Tg_q,
\quad G_F=\sum_t g_{{\rm pre},t}z_t^T.
\]

This exposes what Euclidean query fitting misses: query directions matter through key contrasts and value contrasts. Equal shifts of every score disappear under softmax; identical values give zero response derivative. A query can move farther from native Q while compensating for approximate keys and memory. Native Q is not automatically optimal for the approximate reader.

Let Z have one source feature row per clock and Gpre the corresponding pre-rotation gradients. Use the coordinate-preconditioned direction

\[
D^T=-\arg\min_C\|ZC-G_{\rm pre}\|_F^2,
\qquad D=-G_F(Z^TZ)^{-1}
\]

in ideal full-rank arithmetic; implementation uses least squares, never an explicit inverse. Consequently ⟨GF,D⟩=−||PZ Gpre||²≤0 before finite rounding. A centered finite-difference check measures the derivative instrument.

For this one direction, let δat=Jqt Rt Dzt. The linearized loss is L(F)+α⟨GF,D⟩+α²C, with C=Σt wt||δat||². Its predicted line step is

\[
\alpha_{\rm lin}=-\langle G_F,D\rangle/(2C).
\]

Evaluate the fixed doses {0,−1,1/4,1/2,1,2} times this step after rounding each image to FP32. Choose the smallest source loss, with smaller absolute dose then index resolving exact ties. Preserve every image and source curve, including wrong sign and no-op. Each candidate is a Saturn image Act, restored to the exact parent after source evaluation; no optimizer, RNG or epoch cursor is hidden outside this closure.

The fitted image is then frozen. Actual C execution computes its own q/K/V, appends its own compact state, recomputes O/MLP and continues through its own upper-layer caches and emitted tokens. The objective never sees these target outcomes. Full-vocabulary KL, logit geometry, attention response and task truth remain separate observations: no implication from one loss to all consumers is assumed.

The source teacher uses FP64 affine/RoPE/attention reconstruction; candidate nodes round compiled keys/native values to FP32 before the unchanged centroid law. CUDA arithmetic differs. Native C continuation and independent saved-state replay test that profile transfer. The experiment changes one selected query map, not the lower receiver producer, other layers, full vocabulary head or original weight residency.
