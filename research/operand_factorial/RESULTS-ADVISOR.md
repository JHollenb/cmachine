# Astra: restored operands repair the current write, not the continuing state

2026-11-06. Independent saved-only interpretation of job `job-eb06becf984f`, report SHA256 `5ceab12e2ab62d20978269df55702bb4a26376d0db79752270df9ac5b74eaa0f`. `advisor_analyze.py` recomputes this account from 630 SHA-checked raw arrays, including FP32 bitwise operand pairing, Y11/H MLP16-output equality and new-cache comparisons. The reproducible output is `runs/job-eb06becf984f/advisor-independent-summary.json`. No model call was launched by this review, and no source report or raw evidence was modified.

The result clarifies two distinct missing computations. Restoring either operand improves the current MLP write at every fixed clock, and restoring both reproduces the native write exactly. But a locally better write can worsen the later reader. Even the exact native current write leaves a substantial difference from the native model on the same text, because the earlier approximate writes have already formed different memory. The new-cache timing locates where that remaining difference first enters the current computation.

The twelve fixed clocks are summarized below. The thirteenth, retrospectively selected wrong arithmetic digit, is kept separate. All later inputs are forced historical IDs; this is a causal suffix assay, not a newly generated conversation or autonomous repair.

## What the current operand factorial establishes

Let Y00 use approximate gate/up, Y10 native gate only, Y01 native up only, and Y11 both native. Zero deletes the current write on the same retained parent; it is not the earlier whole-history zero policy. For each cut, Y11 and H (native execution on identical historical IDs) have exactly the same current native write. The paired approximate gate values are byte-identical between Y00/Y01, native gates between Y10/Y11, approximate ups between Y00/Y10, and native ups between Y01/Y11, at all thirteen cuts.

Pooled write error uses the sum of squared errors divided by the sum of squared current native-write norms, then takes the square root. Immediate logit error analogously uses H as the reference and common denominator.

| Current policy | Fixed12 pooled write error | Writes improved versus Y00 | Fixed12 pooled H-logit error | Logit vectors improved versus Y00 |
|---|---:|---:|---:|---:|
| Y00, approximate both | 1.027882 | — | .337498 | — |
| Y10, native gate only | .974954 | 12/12 | .338835 | 8/12 |
| Y01, native up only | .935519 | 12/12 | .312609 | 9/12 |
| Y11, native both | 0 | 12/12 | .209360 | 12/12 |
| Current zero | 1 | 11/12 | .310187 | 8/12 |

Both-native matches H's immediate greedy token at all twelve fixed clocks; Y00 does so at9/12, gate-only9/12, up-only10/12 and current-zero8/12. This is bounded native-token fidelity, not task correctness. At the first arithmetic clock, native's own multiplication answer is wrong, so restoring its greedy decision is not an improvement in arithmetic truth.

Up restoration has the better pooled local metric, but it is not a universally preferred intervention. For recall after answer0, up-only improves write error from1.02553 to.97094 while worsening logit error from.32132 to.40579. Gate-only improves the first recall write from1.02204 to.96514 while worsening logit error from.29703 to.34405. The gate-only pooled logit error is slightly worse despite its12/12 better writes. A current write norm cannot certify the later reader.

## The interaction is a measured nonlinear conjunction

For Δs=SiLU(g)−SiLU(ĝ) and Δu=u−û, the real-arithmetic write identity is

\[
I_Y=Y_{11}-Y_{10}-Y_{01}+Y_{00}=D[\Delta s\odot\Delta u].
\]

The raw native-kernel projection of the right side differs from the measured four-write contrast by8.17e−7 to1.10e−6 of interaction norm across all thirteen cuts. This is the observed FP32 association remainder, not a claim of exact bitwise distributivity.

The fixed12 pooled squared-norm account illustrates why component norms cannot be called causal shares. With A=Y10−Y00, B=Y01−Y00 and I=I_Y, the summed energies are ||A||²=684.77, ||B||²=1961.54 and ||I||²=3628.60. Cross terms are2〈A,B〉=238.15,2〈A,I〉=−455.66 and2〈B,I〉=−1400.35; together they give ||Y11−Y00||²=4657.05. The interaction is substantial and partly cancels both baseline-conditional restoration vectors. Its norm is not a percentage of the failure. Averaged main effects already sum to the joint contrast and must not have I added a second time.

Native-suffix output interactions are separately measured finite contrasts. They are not the displayed D[Δs⊙Δu] expression, because the suffix is nonlinear and reads memory.

## Exact current repair leaves carried-state error

Write C for the Y00 logit vector, L for Y11 on the same candidate parent, and H for native execution on the same consumed history. For each cut and horizon,

\[
C-H=(C-L)+(L-H),
\]

\[
\|C-H\|^2=\|C-L\|^2+\|L-H\|^2+2\langle C-L,L-H\rangle.
\]

Across the fixed twelve immediate clocks, the common-denominator pooled terms are ||C−L||=.250578 and ||L−H||=.209360. The normalized signed cross contribution is+.00728424, reconstructing total squared error(.337498)². Six of the twelve individual cross terms are negative. Thus local and carried-state defects sometimes cancel; neither term can be reported as a fixed fraction of total error.

Nine fixed cuts have two available historical future IDs. On that common nine-cut set, pooled transported one-time error ||C−L||/||H|| changes from.257578 immediately to.163168 after one native step and.085935 after two. Residual ||L−H||/||H|| is.209023,.269531,.289319. Each pooled ratio uses that horizon's H energy. This does not establish contraction or monotonicity for every row, and it does not compare a twelve-row baseline with a nine-row future without qualification.

Both-native improves H-logit error at9/9 first futures and7/9 second futures relative to Y00. At horizon2, current-zero beats Y00 at9/9 and both-native beats current-zero at6/9. All arms match H's greedy token at all available fixed future clocks despite remaining vector differences. The next-token decisions alone therefore conceal the continuing state discrepancy.

## New K/V identifies the first later consumer

Bitwise comparison of every saved new K/V row gives a particularly clean timing result, across all thirteen cuts and all available future horizons:

- At the intervention token, Y10/Y01/Y11/current-zero first differ from Y00 in **new layer17 K/V**. Layer16 K/V was already formed before the changed MLP16 write.
- At that same token, Y11's new K/V matches H through layer17 and first differs at **layer18**. Its current native MLP16 write and the state entering layer17 agree with H; layer17 then reads older memory formed by different earlier writes.
- At the next two forced tokens, every non-Y00 intervention first differs from Y00 in **new layer18 K/V**. All policies now use native transitions, so lower computation and the new layer17 K/V agree. Their different historical layer17 cache rows are read at layer17, and the resulting difference can write new layer18 K/V. Every primary arm likewise first differs from H at layer18 on these futures.

These observations are precisely about new rows. They do not say the full prefix cache becomes equal through layer17: old changed layer17 rows remain and are what the next read can consume. The timing follows the architecture and confirms that this experiment actually exercised the expected writer/read boundary. A layer number alone does not identify the causal head or the individual historical rows.

## Retrospective digit: exact native write is not arithmetic rescue

At arithmetic position79 the model has consumed3 and should choose6 to form367. Y00 chooses5, correct6-minus-wrong5 margin−4.96921. Gate-only gives−5.04371, up-only−5.57688, both-native−6.05506 and current-zero−5.85004. Every arm still chooses5. Native H on this exact historical391 prefix also chooses5, margin−2.01657.

Both-native nevertheless reduces full-vector H error from.611516 to.469205. The arithmetic task margin worsens while native-vector fidelity improves. The correct conclusion is not that the reader has been repaired, or that substituting native history would necessarily solve the task. Native fidelity and truth are separate outcomes here. The subsequent supplied IDs remain historical5 and7, even if another intervention would have chosen6, so the two-step suffix cannot establish autonomous recovery.

## What gap closed, and the smallest next question

The cheap source program's failure is no longer described only as low weight energy or insufficient rank. We have direct evidence that both missing operands contribute, their conjunction matters, and the native current write is not the complete state needed by the next consumer. That is a mechanism map, not yet a cheap replacement algorithm.

The smallest next discriminating state cut is the **first layer17 read on a Y11 current state**. Its current incoming state and current K/V are already native-equal, while its historical layer17 K/V differs. Hold the current receiver, query, residual and all other historical layers fixed; substitute native versus candidate historical layer17 keys and values separately. Measure the layer17 attention read/write, native suffix, and at most the same bounded future. A common-parent key/value factorial could distinguish changed addresses from changed payload and their interaction without blaming all residual L−H on an undifferentiated cache. Restoring only this first reader need not fix later readers, because higher historical layers also differ.

This would again be a paid diagnostic, not a source-only inference proposal. If the goal instead becomes improving the cheap compiler immediately, these results support a consumer-conditioned joint operand objective rather than another unweighted rank increase. Such an objective still needs a source-only or separately calibrated way to obtain its receiver-dependent information. The present evidence does not supply that solution, compact sufficient-state closure, or a whole-model forward-pass-free chat emulator.
