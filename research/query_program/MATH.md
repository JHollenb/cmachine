# Better query prediction on unchanged coordinates

The native selected query before bias/RoPE is q=Wq x. The original program retains z=Bx with 128 coordinates and computes `q0=F0 z`, with F0 the stored FP32 image of Wq B^T. B is approximately row-orthonormal. This approximates an orthogonal input reconstruction and then applies the native query map.

Our new program uses the same z and replaces only its query image. On the original source rows X, define Z=XB^T and Y=XWq^T. One FP64 least-squares solve gives

\[
F_*^T=\mathop{\arg\min}_C\|ZC-Y\|_F^2
=(Z^TZ)^{-1}Z^TY
\]

when Z has full column rank. The implementation uses NumPy's SVD-based least-squares solver, not explicit normal-equation inversion. There is no ridge, target feedback, rank search, optimizer or subsequent update. The emitted FP32 image is canonical; native bias and RoPE still run afterward.

This has an input-space interpretation. Let

\[
D=X^TZ(Z^TZ)^{-1},\qquad P_*=DB.
\]

Then F*=Wq D in ideal arithmetic, and `q*=Wq P* x`. Because `BX^T=Z^T`, we have BD=I, so

\[
BP_*=B,\qquad P_*^2=P_*.
\]

The new reconstruction preserves the retained code and infers source-correlated omitted components. It is an oblique projection, whose image depends on the source distribution, rather than the original orthogonal reconstruction. The stored basis/image, CPU target construction and native C execution have finite remainders; these ideal identities do not assert native byte equality.

Writing E=X−ZB gives

\[
D=B^T+E^TZ(Z^TZ)^{-1},
\]

and therefore

\[
q_*-q_0=Wq E^TZ(Z^TZ)^{-1}z
\]

up to the old image's finite construction profile. The added correction is folded into F*: it adds no runtime projection, MAC, coordinate, memory statistic or dynamic branch. It predicts missing response from the existing code. If two unseen receivers have equal Bx and different native queries, this fixed interface still cannot reproduce both exactly.

The source has193 rows in three correlated trajectories. Those same trajectories already constructed B. Source residuals and source-fit error are in-sample observations, not held-out evidence. The compiler reads only source X, B, the old query image and the original model weight. Fresh prompt text exists in the source payload and is integrity-hashed before compilation, but it is parsed only after the query image seal and is never supplied to fitting. None of the evaluated futures selects or changes F*.

## Query fidelity is a separate observable from its read

At fixed K/V history, attention is

\[
A(q,K,V)=\operatorname{softmax}(qK^T/\sqrt{128})V.
\]

Reducing Euclidean query error does not guarantee a closer read. Score shifts constant across history are softmax-invisible, and different directions encounter different keys and values. The subsequent native MLP and upper caches are additional nonlinear consumers. We measure native query reconstruction, actual attention/output differences, full logits, own generated tokens and the next user turn separately.

The final probability consumer has its own gauge and geometry. For native and candidate full-vocabulary logits lN/lF, set pN/pF=softmax(lN/lF). Then

\[
D_{KL}(p_N\|p_F)=\sum_i p_{N,i}(\log p_{N,i}-\log p_{F,i}).
\]

Uniform logit shifts leave p unchanged, and different coordinates have different probability sensitivity. Lower logit L2 therefore does not imply lower native-to-candidate KL. The saved study measures both, preserving their disagreement; it uses stable FP64 log-softmax rather than taking logs of underflowed probabilities. This is native-distribution fidelity, independent of answer truth.

For the development four-policy comparison, qualify common consumed IDs, exact selected normalized receivers, paired old/new queries across readers and identical K/V/compact-state histories. With old/fitted full probabilities P0/P1 and lifted compact weights Ptilde0/Ptilde1,

\[
I=A_{1,C}-A_{1,F}-A_{0,C}+A_{0,F}
=[(\widetilde P_1-\widetilde P_0)-(P_1-P_0)]V.
\]

Actual native full probabilities anchor the full side. The compact weights are reconstructed after candidate sealing using audit-only membership lists. All four saved-output-minus-weighted-V-mixture remainders remain explicit. Before row 33 there is no multi-member prototype, so the reader contrast isolates finite numerical profiles; merging adds information loss thereafter. This conditional local identity is not a formula for the downstream full-model logit interaction.

The prospective transfer panel has only native full, old compact and fitted compact paths. It tests the new compact program's outcomes without identifying a fresh full/compact interaction. After generated histories diverge, comparisons describe complete own-state policies rather than one local intervention on common whole-model state.

## Cost

Both old and fitted programs use

\[
1536\cdot128+128\cdot1536+128\cdot256+1536\cdot256
=819,200
\]

selected QKV MAC per update: one basis projection, a compiled query, a compiled key and native V. This saves 73.9583% of native selected QKV but only 0.150724% of full-token learned-linear work. It still requires 198 full-token linear calls, one more than ordinary native execution. The compact reader stays 98,464 bytes and has the same routing, nonlinear work and 14,336 CPU/GPU serving-transfer bytes per update.

The experiment retains both query images for comparison, adding 786,432 device bytes. A single fitted policy replaces its old image at unchanged parameter storage. Original full weights, dense lower receiver generation, other 27 cache layers and selected global cache allocation remain paid. CPU source-target construction costs 455,344,128 MAC, feature construction 37,945,344 MAC, with least-squares, diagnostics, hashing and copies additional. Historical native source acquisition and original image compilation are reused, not erased. No process-memory or latency saving follows from these nominal counts alone.
