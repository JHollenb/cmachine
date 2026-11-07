# Factor the address/content interaction at the actual reader

For native and compiled query/key histories on an identical consumed-token prefix, define attention probability matrices A_N and A_C. Values are V_N and V_C. Each grouped-query head selects its original KV group. The four real-arithmetic readouts are

$$
a_{NN}=A_NV_N,\quad a_{CN}=A_CV_N,\quad
a_{NC}=A_NV_C,\quad a_{CC}=A_CV_C.
$$

The separate approximation effects relative to NN are

$$
e_Q=(A_C-A_N)V_N,\qquad
e_V=A_N(V_C-V_N).
$$

The combined error includes their interaction:

$$
a_{CC}-a_{NN}=e_Q+e_V+I,\qquad
I=(A_C-A_N)(V_C-V_N).
$$

This interaction is not automatically beneficial or harmful. The squared norm contains three signed cross terms,

$$
\|e_Q+e_V+I\|^2=\|e_Q\|^2+\|e_V\|^2+\|I\|^2
+2e_Q^Te_V+2e_Q^TI+2e_V^TI.
$$

Large raw V error alone therefore does not establish that V restoration will dominate a local reader or later consumer. Addressing error can amplify or hide value error. Norms cannot be added as causal percentages.

## The common-history boundary

All policies start from token 0 and alter only layer 17. On a common consumed-ID prefix, earlier layers produce the same normalized receiver. The experiment verifies those receiver bytes, Q/K equality within address pairs and V equality within value pairs. Histories include exact headers, generated tokens, EOS and later user inputs. Their maximal shared prefix is computed from actual consumed IDs, not decoded text or equal lengths.

After the first consumed-token difference, the four policies also have different lower states. Their algebraic four-corner output contrast remains defined, but the factorization with shared A_N/A_C and V_N/V_C is no longer the claimed explanation. Upper-layer caches already follow their own policies before token divergence, so even same-ID full-logit contrasts are recurring-policy outcomes. This is not a single current-read restoration on one frozen full-model cache.

## Native numerical mixing remainder

The C wrapper records the actual native FP32 softmax probabilities P_i before they are overwritten, alongside actual FP32 values and attention returns y_i. A saved-only FP64 calculation forms m_i=P_iV_i and R_i=y_i−m_i. It separates the native value-mixture profile from the conditional real-arithmetic law.

When probability and value pairs match, the measured interaction obeys

$$
y_{CC}-y_{CN}-y_{NC}+y_{NN}
=(P_C-P_N)(V_C-V_N)
+(R_{CC}-R_{CN}-R_{NC}+R_{NN})+r_{64}.
$$

The saved-only floating-point expansion remainder r 64 is reported. It is not silently assigned to one cast or primitive. The source query/key error, native probability changes and native mixing error have different denominators and meanings.

O projection, residual addition, the recomputed receiving MLP and later attention/MLPs are original native computation. Their factorial is measured directly. The attention-space identity is not substituted for their nonlinear response. Coherent answers and task correctness remain separate observations, especially because deletion preserved the earlier short answers and native arithmetic itself was wrong.

## Marginal work explains a possible design tradeoff

The single frozen 128 program stores 1,835,008 numeric coefficient bytes plus 512 coordinate bytes. Its source acquisition and compilation were paid previously. The present job reuses those exact arrays and performs no compilation forward.

Native selected QKV costs 3,145,728 MAC/update. Compiling both QK and V costs 458,752. Keeping native V while compiling QK costs 819,200, adding 360,448 MAC over CC. Keeping native QK while compiling V costs 2,981,888, saving only 163,840 over native. This makes the two restoration directions economically asymmetric even when their consumer effects are comparable. The full native token still contains 1,543,569,408 nominal learned-linear MAC, and no measured latency gain is inferred from the product account.

No new compact whole-model transition is claimed. The missing future receiver, all other layers, original weight residency and bounded-memory checkpoint ownership remain outside this causal cut.
