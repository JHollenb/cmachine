# A query must close on the state that actually exists

Let the candidate's layer/token state be `S_(l,t)`, with live normalized receiver `x_(l,t)` and memory `M_(l,t)`. The runtime now directly executes

\[
z_{\ell,t}=B_\ell x_{\ell,t},\quad
q_{\ell,t}=R_t(H^q_\ell z_{\ell,t}+b^q_\ell),\quad
k_{\ell,t}=R_t(H^k_\ell z_{\ell,t}+b^k_\ell),\quad
v_{\ell,t}=W^v_\ell x_{\ell,t}+b^v_\ell.
\]

No target trace supplies `z` during serving. The candidate appends its own K/V, reads its own memory and consumes its own selected tokens. The lower receiver, V map and unselected model computation remain dense.

For a fixed counted memory with rows `(k_i,v_i,m_i)`, define

\[
p_i(q)=\frac{m_i e^{q^Tk_i/\sqrt d}}{\sum_jm_j e^{q^Tk_j/\sqrt d}},\quad a_M(q)=\sum_i p_i(q)v_i.
\]

Expanding the derivative and factoring the normalization term gives

\[
J_q=\frac{1}{\sqrt d}\left[\sum_i p_i v_i k_i^T-a_M\left(\sum_i p_i k_i\right)^T\right]
=\frac{1}{\sqrt d}\sum_i p_i(v_i-a_M)k_i^T.
\]

Thus `J_q^T e` is a streaming key/value covariance contraction. It does not require a dense covariance matrix, a full denoiser or a transformer backward pass. This is a local derivative of one fixed reader. It does not differentiate prototype assignments, changing histories or future model writes.

The teacher here is `a_(M_C)(q_N)` on source compact histories. It changes the question asked of the actual compact memory rather than simultaneously changing the memory. With decision weights `w` giving each of three sources and two turns equal mass,

\[
L(H)=\sum_s w_s\|a_{M_s}(q_s+R_s((H-H_0)z_s))-a_{M_s}(q^N_s)\|^2.
\]

We compute the query gradient, undo RoPE, fit one ridge-smoothed direction in the frozen coordinate interface and take a bounded Gauss–Newton line proposal:

\[
D=-[(Z^TZ+\lambda I)^{-1}Z^TG]^T,\quad
\alpha=\min\left(-\frac{\langle G,ZD^T\rangle}{2\sum_s w_s\|J_sR_s(Dz_s)\|^2},\ \frac{0.15\|Q\|}{\|ZD^T\|}\right).
\]

Four signed doses generate immutable FP32 images. A centered finite difference checks the smooth derivative instrument. Source own-token native-suffix KL chooses the image; local loss alone has no continuation authority. Every rejected image is restored exactly through a Saturn image-only transaction and retained as evidence.

For layer18, source receiver SVD supplies `B_18`, and original recipient maps supply `H_18 = W_18 B_18^T`. This tests the method across a changed producer, not portability of layer17 weights. The missing term is explicit:

\[
W_{18}x-H_{18}B_{18}x=W_{18}(I-B_{18}^TB_{18})x
\]

in exact arithmetic for an orthonormal basis. Small omitted receiver energy alone need not imply a small query error or harmless future. Fresh full-memory and prototype-memory layer18 arms distinguish the operand approximation from additional history compression, subject to their different own-token conditional domains.

Each compiled Q/K/native-V layer pays `128*(1536+1536+256)+1536*256 = 819200` learned-linear MACs instead of 3145728 for native Q/K/V. That is a 74 percent selected-boundary reduction but only 0.151 percent of the original token's 1543569408 nominal learned-linear MACs per layer. It excludes normalization, RoPE, attention, CPU transfers, compilation and feedback. A whole-model speed claim does not follow.
