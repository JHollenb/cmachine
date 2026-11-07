# A better query image at the same runtime cost

The source-only query refit improves the compact program's native-logit fidelity at **23 of 24 saved readouts**, including every first and followup-prefill readout across six conversations. It changes no generated token relative to the baseline. The gain is small but consistent at the initial common-input boundary: query reconstruction, actual attention and full logits improve in all six cases, with the same 128 coordinates, selected MAC budget and memory size.

This is one bounded allocation improvement, not a solved query interface. Source error falls more than target error, substantial query error remains, native-query controls show larger initial headroom, and one later readout regresses. Three prospective transfer prompts were not used for fitting or candidate selection; they do not establish general chat accuracy.

Main Beast job **`job-bd80dd16c1d5`**, report SHA256 `d3f13ffdc2061e5457792ec0947277ef7a68bac0e4685de64bad116400c19f64`. [Report](runs/job-bd80dd16c1d5/report.json), [verification](runs/job-bd80dd16c1d5/verification.json), [study and costs](runs/job-bd80dd16c1d5/study.json), [Astra review](REVIEW.md), [frozen protocol](runs/job-bd80dd16c1d5/payload/PLAN.md).

## What changes mathematically

The preceding CN program computes z=Bx and q0=F0z, retaining128 coordinates and the original source-derived key image, native V and native bias/RoPE. F0 is the original compiled weight projection. On193 authenticated source rows in three correlated trajectories, form Z=XB^T and Y=XWq^T. Replace only the query image by

\[
F_*^T=\arg\min_C\|ZC-Y\|_F^2.
\]

The CPU solve uses FP64 ordinary least squares and emits one canonical FP32 image. No ridge, dose/rank search, gradient loop, new source model advance or target fit occurs. Source coordinate condition number 32.7984 supports the numerical solve. Source relative Q error falls **7.310351%→5.776989%**. This is in-sample error; B already used all three trajectories.

In ideal arithmetic, F*=WqD with D=X^TZ(Z^TZ)^−1. BD=I and P*=DB is idempotent. The program preserves the existing code and reconstructs source-correlated omitted directions with an oblique projection. It cannot distinguish arbitrary unseen receivers with the same code. [MATH.md](MATH.md) gives the derivation and finite-profile qualifications.

The original bundle body, selected native Q tensor, source inputs, basis and old image are hash-bound. The new query image hash is `99e6fa43c6fbed8d01a78f2255b7a3ba12d57b49393f0a32392e3c292656e1c6`. It is sealed before parsing fresh prompts or tokenizing targets and bound into the invocation receipt. Fresh text exists and is integrity-hashed in the payload; the compiler never reads it as fitting data. No evaluated outcome changes the candidate.

## Actual repeated execution

One original Qwen2.5-1.5B-Instruct checkpoint runs in the frozen C FP32 CUDA/cuBLAS profile at attention17, capacity512, batch1/CPU1. The development panel crosses old/fitted/native-query controls with full/compact memory, plus ordinary native full execution: seven policies in each of three conversations. The native-query control still compiles K and pays native V; it is not ordinary native QKV.

Three prospective cases evaluate native full, old compact and fitted compact only. They vary material binding, multiplication/subtraction and kitchen-label prose. This asymmetric panel tests compact-program transfer, without identifying a fresh full/compact interaction. All 24 candidate paths seal before the six native references.

Each path starts empty at token0, produces its own queries, updates/reads its own state, consumes every generated token/EOS and receives the second user turn. Native MLP and suffix recompute on actual receivers. The 30 paths advance 3,024 tokens. Nine historical control paths reproduce 126 distinct arrays plus all four original bank arrays exactly. All 695 generated tokens follow their path's pending-token authority; all 120 full readouts agree with recorded argmax.

Old and fitted compact policies consume identical full histories in all six cases. Their selected normalized receivers, K/V rows, coordinates and 24 paired compact memory snapshots are byte-identical. Their queries and resulting attention differ. Upper-layer caches remain policy-specific, so later full-logit changes include accumulated consequences of earlier different reads.

## Fidelity results

These are full-logit relative L2 differences against each case's ordinary native full-memory execution, not answer-error probabilities.

| Case | Initial old compact | Initial fitted compact | Followup prefill old | Followup prefill fitted |
|---|---:|---:|---:|---:|
| Development recall |12.083%|11.781%|13.373%|13.100%|
| Development math |12.713%|12.357%|24.319%|24.152%|
| Development prose |6.292%|6.175%|30.267%|30.141%|
| Prospective material recall |17.501%|17.340%|14.665%|14.557%|
| Prospective math |9.037%|8.658%|26.628%|26.535%|
| Prospective kitchen prose |4.648%|4.569%|22.835%|22.548%|

All sixteen boundaries where every evaluated policy shares the native consumed history improve in compact execution. Across all 24 compact boundaries, 23 improve; development math after the followup answer slightly regresses **19.359856%→19.364275%**. Those are correlated clocks from six short conversations, not 24 independent tests. In development full-memory execution, 11/12 readouts improve; first prose answer is the small opposite effect.

The corrected FP64 initial query errors old→fitted are 22.137→22.053%, 23.119→23.005%, 15.070→14.833%, 23.068→22.988%, 22.545→22.427%, 15.150→14.903%. Initial compact attention errors also improve in all six cases. These modest receiver-level improvements support a useful calibration, while most query discrepancy remains. They do not show the code is sufficient for broad inputs.

A saved-only probability check is less uniform than logit L2. Native-to-candidate KL improves at **19/24** compact boundaries and **13/16** boundaries with histories matching native. Native-top-token log probability improves at 21/24. Initial development-math KL worsens 0.000356555→0.000382877 nats; initial prospective-prose KL worsens 0.029125575→0.032360183 nats, despite their improved query/attention/logit L2. Other initial KL changes improve, including prospective recall0.038174994→0.036193806 and prospective math0.000075969→0.000065652. These are native-distribution fidelity measurements, not truth or calibration scores. After native histories diverge, they compare operational policies. [All probability rows](runs/job-bd80dd16c1d5/study.json) preserve signed effects. A decoder that reconstructs queries better is not uniformly better under the actual probability consumer.

![Same-cost query prediction and consumer outcomes](runs/job-bd80dd16c1d5/query-program.png)

## Paid native-query headroom and a reversal

At initial development readouts, native-Q/compiled-K/native-V compact execution gives 7.956/9.396/4.297% logit differences, versus fitted11.781/12.357/6.175%. That is appreciable initial headroom beyond this source-only refit. It pays 2,981,888 selected MAC per update, about 3.64 times the fitted program, and retains the approximate K image.

Native Q is not uniformly better as a recurring policy. At the development math final boundary, native-Q compact/full discrepancies reach86.317/84.224%, versus old19.360/18.861%, after their generated explanations differ. Every policy still says 286 for266+13. The expensive control diagnoses a consequential query interface and a changed trajectory; it is not an automatic serving choice or a universal upper bound on fidelity.

## Outputs and truth

The fitted program emits exactly its baseline's tokens in every compared path, including full-memory development paths. Recall remains blue/green and prospective recall silver/wooden. Development math still gives 266 then wrong 286, instead of 279. On the new16×27 case, all policies first give 432; old and fitted compact give the correct 414 after subtracting18, while native gives 254. This correctness advantage already exists in the baseline and is not created by the query refit. Arithmetic explanations are truncated at their16-token budget. Output-only arithmetic evaluation is retained separately in study.json.

Prose remains coherent but differs from native. This panel establishes small native-computation fidelity gains, not a new behavioral capability, stronger math, semantic necessity of this attention layer or accurate unrestricted conversation. Historical deletion insensitivity remains a limitation of the basic-answer probes.

## Query × memory account

Development old/fitted full/compact paths share94/95/60-token prefixes. Actual receiver, paired queries, K/V and compact-state identities qualify

\[
I=[(\widetilde P_* -\widetilde P_0)-(P_*-P_0)]V_N.
\]

Full probabilities are actual native FP32 captures; compact weights/memberships are reconstructed post-seal only. Four finite mixing remainders close the common-prefix aggregate to 2.28–2.82×10^-16; maximum per-row expansion remainder is6.24×10^-16. Initial interaction norms are 1.122/1.060/1.174% of old/full attention. Before prototype merging at row 33, interaction stays at the finite-profile scale. The identity accounts for the conditional selected read, not the nonlinear suffix or a fresh-case factorial absent from the panel.

## Numerical reporting correction and custody

The first local strict recomputation refused96 scalar fields, confined to duplicated new query/attention/output diagnostics. Their numerator was FP64 but their reference norm used the incoming FP32 dtype, whose reduction differs between Mac and Beast. All primary raw hashes, source fitting, memory replays and full-logit measurements remain valid. The [original refusal](runs/job-bd80dd16c1d5/verification-refusal-diagnostic.json) and frozen report are preserved.

Zero-model CPU job **`job-7d8b02674ac8`** replays the exact frozen analyzer in its original Linux/NumPy environment. It reproduces canonical analysis bytes exactly and verifies all 74 source/467 raw files. [Combined verification](runs/job-bd80dd16c1d5/verification.json) binds both scheduler logs, original report/driver and the whole C-invocation Saturn receipt. No tolerance is relaxed. [FP64 corrected metrics](runs/job-bd80dd16c1d5/fp64-query-metrics.json) are an additive sidecar. The current working analyzer fixes both operands' dtype before normalization; this later source change does not rewrite the executed payload. Ten current model-free tests pass, including the dtype regression; the C driver passes strict compiler checks. All 60 compact snapshots and every declared compact read replay byte-exactly in the original profile. Evidence coverage runs/verifies 3,417 observations cleanly, without terminal certification.

## Cost and the next computation

Old and fitted CN each pay819,200 selected QKV MAC/update and 198 whole-token linear calls. The fitted image replaces its old image with no per-policy coordinate, parameter-storage or mutable-state increase. The experiment loads both images for comparison, adding 786,432 device bytes. Logical compact state stays 98,464 bytes, with the same at-most32 nodes/head, routing and 14,336 CPU/GPU serving-transfer bytes per update.

Whole-token nominal linear saving remains only 0.150724%. The native lower layers still generate x; other 27 cache layers, O/MLPs, vocabulary readout, full original weights and selected global-cache allocation remain paid. The C invocation counts604,458 learned-linear calls and 4,712,818,868,224 nominal MAC including 32 native parity advances. CPU source compilation adds455,344,128 target-construction MAC and 37,945,344 feature MAC, plus least-squares and diagnostics. Historical native source acquisition and original image construction remain paid.

Main worker 53.53s, C driver 42.57s, CPU compiler 2.04s; mrun elapsed71.63s, scheduler timestamp interval71.26s. Main peakRSS 4,010.5MB/VRAM6,406MB. The separate exact CPU audit costs 47.70s elapsed,181.3MB peakRSS and zero GPU/model work. Raw primary files total 171,443,331B, below the 201,134,940B geometric bound and 256MiB transport limit. These are instrumented runs, not serving-speed measurements.

Keep the fitted image as a source-bound experimental candidate. Its positive transfer shows the old fixed code was used imperfectly; the small gain and large residual show that refitting this decoder is not enough. The next informative comparison should target score/value response on declared source data at a matched budget, or explicitly test additional query information with its cost counted. Do not enlarge memory to compensate automatically. Any next program still needs its own repeated-state and later-consumer evaluation, with fresh data kept separate from feedback.

The probability countertrends make response-aware fitting the cleaner next objective before growing the interface. A source-only attention-response loss can compare `softmax(q_F K_C^T/sqrt128)V_N` to the declared native response, while keeping the old/fitted query programs as fixed-budget controls. The approximate key bank belongs in that objective: native Q is a diagnostic reference, not guaranteed optimal addressing for approximate K. Actual later logits, probability KL and own-state continuation must then judge whether the local objective transfers. No such response-aware candidate was fitted or selected in this run.
