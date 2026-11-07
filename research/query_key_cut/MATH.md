# Conditional Q/K and history effects

The current read uses the same actual receiver and value history, with two queries and two full key histories. Native here means original layer18 projections on the actual layer17compact-produced receivers; it does not mean a separate full-native model history.

For historical parent H, query choice q and key history K,

\[
a(q,K,V)=\operatorname{softmax}(qK^T/\sqrt d)V,\qquad Z_H(q,K)=\text{unchanged suffix logits after this read}.
\]

Define C/N for compiled/native operands. The four corners are CC, NC, CN and NN, where the first index is query. Conditional query and key effects are

\[
\Delta_Q^{K_C}=Z_H(N,C)-Z_H(C,C),\quad
\Delta_K^{Q_C}=Z_H(C,N)-Z_H(C,C),
\]

\[
I_{QK}=Z_H(N,N)-Z_H(N,C)-Z_H(C,N)+Z_H(C,C).
\]

Thus the full current replacement obeys the exact vector identity

\[
Z_H(N,N)-Z_H(C,C)=\Delta_Q^{K_C}+\Delta_K^{Q_C}+I_{QK}.
\]

Their norms do not add. The scalar first-token margin is the linear functional `Z[silver_first]-Z[amber_first]`; it inherits the same signed factorization. Complete generated locations, EOS and later answers remain separate observables.

Two parents consume exactly the same prefix. Only the recurring layer18read policy differs. Lower layers and current layer18receiver therefore match, and both native/compiled key streams and V match across parents. Current NN attention and completed layer18output should also match. If higher-layer attention and final logits still differ, the current selected operation has closed while inherited higher-layer cache remains consequential:

\[
\Delta_H=Z_{H_C}(N,N)-Z_{H_N}(N,N).
\]

Each branch recomputes the last consumed token before the first wrong location token. Its changed attention writes a residual, changes the receiving MLP, then writes current higher-layer K/V before vocabulary selection. The branch's own selected token becomes the next input; all later reads return to its original historical policy. This tests a single current intervention, not a recurring repair or a new cheap compiler.

The raw circuit records retain input x, attention a, output write W_o a, post-attention r, normalized MLP input, MLP write m, and xplus for each layer18–27. Native FP32 additions obey r=round(x+W_o a) and xplus=round(r+m). Local vector opposition or a small reconstruction norm does not independently certify semantic repair; the actual emitted location is the consumer outcome.

Wrong-time Q tests clock-specific addressing. Reversing V with Q/K fixed tests whether the key/value pairing matters. Zeroing the consumed read tests whether a rescue is compatible with merely deleting a harmful write. These controls distinguish useful current consumption from query information already present in the lower residual; none by itself establishes a unique or minimal answer carrier.
