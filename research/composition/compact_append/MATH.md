# Append closure is a transition contract

Let the token-boundary state contain a cursor p, session metadata m, dense memories for unselected layers, and encoded memories C for selected layers. A selected cache is a valid prefix of positioned paired rows:

\[
C_{\ell,p}=\{(s^K_i,c^K_i,s^V_i,c^V_i):0\le i<p\}.
\]

For a width 128 row x, the executed representation is

\[
s(x)=\max_j|x_j|/127,\quad
c_j(x)=\operatorname{clip}_{[-127,127]}\operatorname{roundAway}(x_j/s),\quad
\hat x_j=\operatorname{FP32}(s c_j).
\]

The zero row uses scale 1. Scale formation and division are FP32; nonfinite components and invalid scales refuse publication. This is an approximate coordinate representation, not exact semantic state extraction.

At a selected layer, the current candidate residual produces its own post-RoPE q, k and its own value v. Define append A and read R:

\[
C_{\ell,p+1}=A(C_{\ell,p},E(k_{\ell,p}),E(v_{\ell,p})),
\qquad
a_{\ell,p}=R(q_{\ell,p},D(C_{\ell,p+1})).
\]

The p+1 matters. The current row is encoded before the same token's attention read. A scheme that uses FP32 for the current row and encodes only for later tokens executes a different hybrid transition. The row comes from this candidate's own computation, rather than a native donor with a potentially different prior state.

At each query head, the actual decoded bank supplies the paired exponential response:

\[
Z(q,C)=\sum_{i=0}^{p}\exp(\alpha q^\top\hat k_i),\quad
N(q,C)=\sum_{i=0}^{p}\exp(\alpha q^\top\hat k_i)\hat v_i,\quad
R(q,C)=N/Z.
\]

The CUDA implementation retains native-shape score/value GEMMs and native softmax. Saved-only FP64 sums are independent diagnostics of these represented operands; they do not replace the executed FP32 kernels.

## Two selected layers test composition

Write F17 for attention output projection, residual and MLP through layer17, and G18 for normalized query formation at 18. Then

\[
q_{18}=G_{18}(F_{17}(h_{17},C_{17})).
\]

Changing C17 can change q 18 before the token is selected. Comparing byte17+18 against byte17 isolates the additional layer18 representation under a common changed upstream query, while matching consumed token histories. This is cross-layer composition. It is not continuous feedback from layer18 output into its own next query: with fixed lower computation and identical token histories, that query's causal inputs remain fixed. Different emitted tokens provide another return path, with an additional input-mismatch term in any error recurrence.

## Rollback changes reachability

A marker at r owns session identity, epoch, cursor and token metadata. Restoring it applies

\[
(p,m,\{C_{\ell,p}\})\longmapsto
(r,m_r,\{C_{\ell,p}[0:r]\}).
\]

Suffix bytes may remain physically allocated, but reads can only reach the restored valid prefix. A sibling overwrites its own suffix. An injected failure before publication leaves its new row unreachable; if an earlier selected layer has already published, abort must truncate every selected cursor and restore token metadata together. Scratch is recomputed on the next token. The experiment checks valid logical cache equality and full-logit replay, not zeroing every invalid allocated byte.

## Compact storage versus sufficient state

With two KV heads, both K and V, and width 128, a native FP32 row costs 2048 bytes per layer/token. Codes and scales cost 528 bytes. For L selected layers at capacity P, byte cache storage is 528 LP; the shared decode workspace is 1024 Pbytes and the status flag 4 bytes. Native selected storage is 2048 LP. Net requested GPU allocation savings are

\[
\Delta B=(1520L-1024)P-4.
\]

This formula excludes unchanged session scratch, full model weights, host metadata and driver/library allocations. It assumes one shared head-plane workspace and sequential layer execution. For P=512 and L=1/2, savings are 253948/1032188 bytes. Attention still scores all valid rows; capacity and memory growth remain linear.

For an exact state abstraction E and native transition T, strong closure would require an abstract transition satisfying E(T(S,x))=T_hat(E(S),x) and an output decoder preserving the declared future consumers. Quantized K/V alone does not establish that identity. This experiment constructs a concrete approximate transition and measures its error, rollback and allocation contracts. General compact sufficient chat additionally requires a declared future-query domain, adequate retention or eviction, bounded continuation error and evidence across broader tasks.
