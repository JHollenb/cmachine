# Astra review: substantial replacement, sampled coverage repair, failed chat transfer

November 6, 2026. Artifact-only independent review of `job-a6811af393aa`. No model execution, new fit, source mutation or native replay was performed for this review. `astra_geometry.py` reads the collected FP32 arrays into FP64 with one-thread numerical libraries. Its additive result is `runs/job-a6811af393aa/analysis/own_geometry.json`. The result binds the local report SHA. This review is not an independent streaming verification of the entire custody archive.

The experiment really removes substantial learned-linear execution: all-layer candidate paths execute 225 counted linear calls and 912,326,656 nominal MACs per consumed token, versus native 197 calls and 1,543,569,408 MACs. Both all-layer programs execute zero original gate/up calls; each compiled program owns 557,842,432 factor bytes. The difference is 631,242,752 MACs, approximately 40.895%. Original weights remain resident for compilation and controls. Memory removal, serving speedup and general chat preservation are not established.

The scientific result is more specific than “refresh failed.” The refresh markedly improves reconstruction of the p0-generated source receivers and improves many same-input local MLP writes. It nevertheless does not recover useful all-layer chat or improve the declared held-out policy-level fidelity score. The first result is real progress on the compiler's actual geometric objective; the latter is the essential limitation.

## Crossed source geometry

For each saved layer basis E and source receiver bank X, this audit computes the actual reconstruction residual X − (X Eᵀ)E. It does not replace this residual by an energy-subtraction formula assuming exact orthogonality. All entries below are residual **energy** divided by receiver energy, pooled across the 28 layers and 384 selected rows per layer; they are not relative L2 values or equal-layer means.

| Basis | Native source bank | p0-owned source bank |
|---|---:|---:|
| p0 | 0.629745% | 22.728922% |
| p1 | 2.744206% | 2.696622% |

At all 28 layers, p1 improves reconstruction of the same p0-owned source bank and worsens reconstruction of the same native source bank. Equal-layer means, retained separately in the JSON, are respectively 0.977575%, 34.722828%, 4.525378% and 3.959645%. Different layer norms explain why these differ from pooled fractions.

This is a genuine coverage tradeoff on the two recorded banks. It is not a comparison between p0's native training loss and p1's mixed training loss. It also is not a fixed-point result: the refreshed p1 program was not used to generate a third source bank and compile another program. The p0 source bank includes changes in continuous state, emitted tokens and selected clock locations; this cross cannot assign its difference solely to continuous state drift under fixed IDs.

## Same-input write observations and later behavior

Each saved candidate boundary supplies its actual normalized input to the original G/U/SiLU/D consumer after the primary path has closed. Comparing that native write with the candidate write therefore measures a local operator discrepancy on the candidate's input, without substituting native state into the primary trajectory.

Across all 24 held-out boundary observations and all 28 layers:

| Candidate | Pooled local write relative L2 | Mean per-row/layer relative L2 | Own-history shadow KL, equal case/turn |
|---|---:|---:|---:|
| p0, all layers | 58.6521% | 74.2085% | 5.2301656 nats |
| p1, all layers | 49.1031% | 69.7993% | 5.2908170 nats |

The pooled write quantity is sqrt(sum squared write differences / sum squared same-input native writes). The mean separately averages individual relative L2 values. p1 improves 25 of 28 per-layer pooled write comparisons; layers 9, 10 and 11 worsen. Middle-layer write errors remain particularly large. The 24 boundaries are correlated observations from six conversations, not 24 independent tasks.

The monitor favors p1: KL 6.0811300 → 4.3320552 nats. That selection is honestly frozen before the held-out panel, where the direction reverses slightly. Both full programs produce repetitive or malformed answers and score zero of eight strict binding/arithmetic answers. Native scores seven of eight, and p0 used only at layer16 scores six of eight. The native error is preserved: after exchanging Dena's red and Fenn's blue tickets, native answers Blue for Fenn instead of Red. The one-layer control additionally misses Ova's updated black cup and retains amber. Prose is not assigned an automatic truth score; its native and one-layer continuations are coherent but several hit the declared token cap.

The selected-layer-only control has shadow KL 0.1410497 nats and a 55.0196% pooled same-input write discrepancy at layer16. This is useful evidence that isolated local write distortion and whole-stack behavior are not interchangeable metrics. It does not by itself apportion full-stack failure among particular layers, accumulation, changed histories or approximation interactions.

Native same-input write comparisons are exactly zero in these saved arrays. This supports the local instrument and matched arithmetic geometry. It does not turn the candidate's changed receiver into a native receiver.

## Why coverage and local improvement do not certify the next computation

In real arithmetic, let P = EᵀE, a = Gx, b = Ux, da = G(P−I)x and db = U(P−I)x. Ignoring separately measurable finite image-formation/evaluation differences, the candidate write defect factors exactly as

    dw = D[(SiLU(a+da)−SiLU(a)) ⊙ b
           + SiLU(a) ⊙ db
           + (SiLU(a+da)−SiLU(a)) ⊙ db].

The compiler minimizes receiver reconstruction approximately; it does not directly minimize this write expression or its later consumer. A small unweighted receiver-energy residual need not preserve the G/U-sensitive directions, their product, or the downstream D-weighted response. This interpretation is consistent with the observations, but the equation is not a measured causal decomposition of the held-out failure.

Across transitions, candidate-minus-native state error has the exact abstract split

    e_next = [Fhat(Shat) − F(Shat)] + [F(Shat) − F(S)].

The local write diagnostics inspect part of the first term at saved boundaries. The complete future also depends on inherited state and the later consumer. Local improvement alone does not bound those terms, especially after token choices differ.

The KL objective also changes its history distribution with the policy. Both candidates receive their own generated histories; native shadows independently consume those IDs. p0 matches the shadow argmax on 107/184 decisions (58.15%) despite meaningless output; p1 matches 83/168 (49.40%). A native model can predict a continuation of an already repetitive supplied history. These counts do not mean 58% or 49% useful chat, and they do not establish agreement with native-owned conversations. The full-distribution discrepancy and actual answers must remain visible.

## Evidence and reporting boundaries

- This is one randomized receiver-covariance bootstrap, with Lg = G Eᵀ and Lu = U Eᵀ. Monitor selection does not make it a response-optimized or end-to-end future-trained compiler.
- A failed result for this rank, source panel and one refresh does not exclude economical nonlinear transition laws, response-aware factors, channel programs or a different allocation. Earlier all-layer programs are prior art, not discoveries made by this run.
- Primary plus shadow execution, compilation and local dense diagnostics remain paid. The saved physical ledger reports 14,336 compilation matvecs and 197,300,060,160 MACs per factor bank. Instrumented job time is not a matched serving benchmark.
- The earlier `job-1df6797e0f96` was an admission failure during compilation, with no completed candidate comparison. Its failed execution remains separate from this completed result.
- Raw per-turn `budget` fields are zero because the C TurnInfo field was not populated. The executing loop uses the path budget; the source/monitor/held-out manifest and actual decision counts identify the real 8/16/32-token caps. Preserve the raw field and record any corrected budget interpretation additively. This metadata defect does not explain the observed degradation.

The bounded conclusion is: substantial original map execution has now been omitted under the declared C profile, and one own-state coverage refresh repairs the sampled covariance problem, but neither full-stack program preserves useful chat on this panel. The gap is the relation between affordable receiver representation and the future-relevant response—not missing token-loop mechanics, and not a general impossibility result.
