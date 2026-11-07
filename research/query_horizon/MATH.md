# A prediction clock is different from a consumed-token clock

After consuming input token ut, the C state St contains caches, compact memory, position and a pending next token. Its vocabulary logits ℓt predict ut+1. Consuming that pending token is a separate state transition. A source trace therefore records both the consumed-token position and the role of the following prediction.

For answer j in source c, let Pc,j be the sequence of row indices immediately before each native-generated token is consumed. If prefill ends at position p and the answer has k tokens, then

\[
P_{c,j}=\{p-1,p,\ldots,p+k-2\}.
\]

The row p−1 chooses the first answer token. An EOS token contributes its choosing row. The row after consuming EOS, or after the kth budget-limited token, is not used to choose a supplied answer token and is excluded from this answer's score. Template ingestion has separate clocks; a previously consumed EOS is skipped rather than added twice.

The historical eight-consumed-token diagnostic used rows {p,…,p+7}. The new first-eight-decision diagnostic uses {p−1,…,p+6}. They share seven clocks, but are different observables. Both remain visible alongside decisions9–32 and post-followup predictions. Joining the original source prefixes checks their actual state and logit identities rather than assuming equivalence.

For immutable query image Fi, native creates the extended source token history once. The candidate then follows those fixed tokens while advancing its own state:

\[
S_{t+1}^{(i)}=T_{F_i}(S_t^{(i)},u_{t+1}^N),\qquad p_t^i=\operatorname{softmax}(H(S_t^{(i)})).
\]

The native teacher uses ordinary native Q/K/V and full memory. The candidate uses the fixed B128 query image, compiled K/native V, compact selected memory and its own upper caches; O/MLP recompute normally. No candidate feedback creates or modifies the native source IDs.

The predeclared future-focused source loss is

\[
L_{\rm future}(F_i)=\frac13\sum_{c=1}^3\frac12\sum_{j=0}^1\frac1{|P_{c,j}|}\sum_{t\in P_{c,j}}\mathrm{KL}(p_t^N\|p_t^i).
\]

Each conversation and each of its two answer windows has equal weight, even when EOS makes answer lengths differ. Prefill, user followup ingestion and unused post-answer predictions cannot dominate this selector. Their metrics and extended all-clock loss remain diagnostic. Selection is only among the six already-frozen images, with exact ties resolved by smaller absolute original dose then index.

Source replay is forced by design. At every scored decision compare argmax(pti) with the supplied next native token. A candidate's exact greedy-prefix claim lasts only until the first disagreement; matching later tokens cannot restore it. Keep turn-specific disagreements distinct from cumulative trajectory validity. This separates a useful same-history response from evidence about autonomous future generation after divergence.

Target conversations follow each policy's own emitted-and-consumed IDs. The source-selected image seals before target parsing or tokenization. Historical dose2 and response-MSE dose1 remain fixed controls, even if the extended source all-clock diagnostic chooses a different image. Exact image/policy aliases execute once.

This changes the feedback horizon and scoring surface. It adds no direction, source context, query dimension or memory. It cannot establish a globally optimal F, sufficient B128 state, a cheap producer for the dense receiver, whole-model emulation or correctness of native answers. Those remain distinct questions.
