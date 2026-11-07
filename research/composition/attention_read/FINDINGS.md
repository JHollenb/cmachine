# A correction-local score interaction becomes a distributed attention read

2026-11-06. The [planned layer17 cut](PLAN.md) localizes how the two episode states meet. Current query depends on question policy; retained correction KV depends on formation policy. Their bilinear score interaction is exactly zero outside the correction span in captured native FP32 scores. Softmax spreads interaction onto unchanged source and question positions. A fixed-query denominator/numerator update accounts for the native context change to2.81×10⁻⁷–6.45×10⁻⁷ relative discrepancy. Receiving MLP17 opposes the incoming interaction, but the net magnitude still grows before answers and shrinks after EOS.

The [original report](runs/job-812a0486995d/report.json), [strict offline verification](runs/job-812a0486995d/verification.json), [post-hoc signed MLP audit](runs/job-812a0486995d/mlp-interaction-audit.json), and [Astra review](REVIEW.md) retain the measured boundary. This is a conditional account of a native read given captured operands. It does not produce those operands from compact conversational state, establish semantic necessity, or replace full weights/KV.

## Specimen and execution

Same Qwen2.5-1.5B-Instruct, historical C FP32/cuBLAS profile, frozen276536-byte B3 at zero-based MLP16. A native source parent at cursor59 forms native or PROJECT correction/acknowledgment parents at93. Each forks native or PROJECT full question/answer episodes for Niko and Mira independently. All10turns consume their own generated IDs, including EOS. Every policy acknowledges OK and answers Niko green/Mira red.

Sixteen traces capture four worlds ×two questions ×pre-answer/consumed-EOS clocks. Input positions116/118 correspond to cache lengths117/119, authenticated from actual histories. Source occupies0–58, correction/acknowledgment59–92, and question/own reply93onward. These selected endpoints do not establish first temporal onset.

The isolated CUDA wrapper preserves native QK GEMMs, original softmax kernel and value GEMMs. It adds a selected read-only copy of unscaled scores before their buffer is overwritten by probabilities. Q/K ports include actual biases and RoPE. Twelve query heads map to two KVheads by floor(head/6), dimension128. All valid historical/current KV at layer17 is captured; complete all-layer KV is retained as boundary hashes. O17 exports once for saved-only linear projection checks.

All20prefill/final full-logit files and valid-cache hashes reproduce the authenticated uninstrumented historical endpoints exactly, including both correction parents. Histories, every selected-layer transition and native residual-add bytes also agree. This qualifies the instrument without supplying historical vectors to the executing C process.

## The score interaction has the predicted support

Write NN/PN/NP/PP for native/PROJECT correction policy first and native/PROJECT question policy second. Let I_X=X_PP−X_PN−X_NP+X_NN. All 32 dependency observations across the four boundaries hold: entry/query fixed by question policy; source KV fixed; correction KV fixed by formation; question KV fixed by question policy. Current observed K/V matches the appended cache row exactly.

The real-arithmetic unscaled score interaction is

\[
I_S=(q_P-q_N)(K_C^P-K_C^N)^\top,
\]

on correction columns C, with zero support elsewhere. Captured native unscaled FP32 score interaction is exactly zero on both source and question bands at all four boundaries. The full native versus ideal bilinear relative differences are3.84×10⁻⁶–4.49×10⁻⁶. Retaining the four native QK reduction remainders closes the vector account to7.64×10⁻¹⁴–1.02×10⁻¹³ L2 in the FP64 diagnostic calculation.

| Reader / clock | Native unscaled score interaction L2 | Native-minus-bilinear relative L2 |
|---|---:|---:|
| Niko / pre-answer |38.80604|4.49×10⁻⁶|
| Niko / consumed EOS |55.85872|3.84×10⁻⁶|
| Mira / pre-answer |42.47758|3.93×10⁻⁶|
| Mira / consumed EOS |56.09208|3.84×10⁻⁶|

The actual FP32 scale is0.0883883461356163, little-endian bytesf304b53d. The report separately retains the mathematical1/√128 factor, scale quantization, scaled-product rounding and QK reduction remainder. Tiny off-band FP64 factorial subtraction residuals are arithmetic diagnostics; the captured native off-band interaction is exactly zero.

## Softmax distributes the effect

Let P denote captured native probabilities. Its interaction extends outside the score-interaction support:

| Reader / clock | Source-band I_P L2 | Correction-band I_P L2 | Question-band I_P L2 |
|---|---:|---:|---:|
| Niko / pre-answer |0.16573|0.16530|0.08013|
| Niko / consumed EOS |0.28403|0.22230|0.40003|
| Mira / pre-answer |0.10508|0.12453|0.07805|
| Mira / consumed EOS |0.31293|0.19319|0.45163|

The source band remains consequential despite its unchanged scores at fixed question policy. Probability normalization couples all permitted columns in each head. The absolute L1 interaction mass summed over heads outside the correction span is50.48%,70.88%,48.45%,73.94% respectively. These are vector-support fractions, not semantic importance or fractions of causal responsibility. The unchanged source/value support cannot be discarded on the basis of localized score interaction.

## Values and allocation form the context together

The full value factorial I_V is exactly zero. This does not make value changes irrelevant: correction and question values have separate nonzero changes, which interact with allocation. With ΔαV=V_PN−V_NN and ΔβV=V_NP−V_NN,

\[
I_c=I_PV_{NN}+(P_{PP}-P_{PN})\Delta_\alpha V
+(P_{PP}-P_{NP})\Delta_\beta V+P_{PP}I_V+I_\varepsilon.
\]

Here ε is native value-GEMM output minus the declared FP64 product of captured probabilities/values. The saved account closes to1.76×10⁻¹⁵–2.05×10⁻¹⁵ L2. Native roundoff-interaction norms are2.82×10⁻⁶–3.22×10⁻⁶; the actual context-interaction norms are3.00723/6.13642for Niko and2.22200/6.57404for Mira, pre-answer/EOS respectively.

All three nonzero ideal terms contribute. The baseline-value allocation term has negative cross terms with each changed-value term, while those two changed-value terms align positively, at all four boundaries. The report retains their signed projections and ideal-term cross terms separately from the native remainder. Norms cannot be added as independent error budgets.

## A query-conditioned update account closes

For a fixed question policy, changing formation affects only correction-band scores/values. In a common exponential gauge,

\[
Z=\sum_i e^{s_i},\quad N=\sum_i e^{s_i}v_i,\quad c=N/Z,
\]

\[
\Delta c=\frac{\Delta N_C-c_{\rm before}\Delta Z_C}{Z_{\rm after}}.
\]

The second term renormalizes the whole previous read. Eight fixed-policy contrasts reproduce this identity to1.80×10⁻¹⁵–2.73×10⁻¹⁵ L2 on ideal softmax of native-rounded scaled scores. Comparing that prediction to native FP32 attention changes gives2.81×10⁻⁷–6.45×10⁻⁷ relative L2, with prediction norm as denominator. Off-band native probability ratio residuals retain softmax arithmetic separately.

For Niko's pre-answer native question, query head9 has Z_before/Z_after=0.40259402. Every unchanged off-band probability is multiplied by that common ratio in the ideal account. PROJECT-question head9 instead gives0.60083218. The correction's denominator effect is therefore conditional on which query the later computation produces.

This is a compact sufficient interface for a supplied query and selected read in real arithmetic. An incrementally stable implementation also carries its log-sum-exp gauge and numerical contract. New queries reweight the old keys, so one Z/N pair does not supply an arbitrary future read. Restricting to a finite declared query family would permit a bank of these signatures; no such small family or chat-state closure is established here.

## The receiving MLP opposes interaction without always reducing it

Entry and query factorials are exactly zero. Attention introduces the interaction; O17 maps it to the actual attention-write interaction, with aggregate native projection discrepancy5.34×10⁻⁶–5.92×10⁻⁶ L2. Per-world residual additions reproduce byte-for-byte. The receiving native MLP17 then changes with the actual receiver.

| Reader / clock | Receiver I L2 | MLP-return I L2 | Receiver/MLP signed cross power | Output I L2 |
|---|---:|---:|---:|---:|
| Niko / pre-answer |2.68602|2.65353|−6.39662|2.80344|
| Niko / consumed EOS |5.28679|3.94286|−18.71548|4.97804|
| Mira / pre-answer |2.04276|2.15243|−2.97741|2.41421|
| Mira / consumed EOS |5.62357|4.32586|−21.43210|5.37638|

The MLP interaction opposes the receiver interaction in every case. Yet its added power outweighs that opposition before both answers; the output interaction grows. After both EOS tokens, the net interaction shrinks. Negative cross terms alone do not establish a corrective computation, semantic repair, or universal contraction. The [post-hoc source-bound audit](runs/job-812a0486995d/mlp-interaction-audit.json) retains all powers, remainder cross terms and exact pointwise vector closure, without changing the executed report.

## Custody, resources and next question

Completed job `job-812a0486995d`: 29.49 scheduler seconds, 20.70 worker seconds, 12.72 C driver seconds; RSS 3914.7 MB / VRAM 6378 MB. One resident model performs 59 source advances plus 276 observed transitions, 335 full token computations. MLP16 performs 276 G/U calls and 276 down calls; 138 PROJECT transitions skip 276 G/U calls, with zero compile calls. Capture/export and analysis costs are included, so this is not a serving-speed benchmark.

Strict saved-only verification authenticates73sources and301raw files totaling27,694,210bytes, both predecessor projections, scheduler/log/report joins, pinned original runtime/bank and whole-invocation Saturn receipt. The prospective raw/payload bound is46,709,856bytes under the48MiB cap. Fourteen model-free math/token/import checks and C syntax checking pass. The [standard evidence sweep](index/job-812a0486995d/evidence-run.json) and [verification](index/job-812a0486995d/evidence-verify.json) cover14source-bound rows with5applicable predicates, no fires/claims/addresses. This is stream coverage, not semantic certification. Astra independently recomputes hashes, GQA products, support, normalizer responses and signed MLP account.

The initial payload import failure is preserved as null evidence in [ATTEMPTS.md](ATTEMPTS.md); only the corrected job executes the model. Original C serving and historical reports remain unchanged. A native typed pre-softmax score port is requested in saturn/request.jsonl so future assays can avoid isolated wrappers.

The next research question is whether a compact response representation built from an earlier memory span can predict its Z_C(q) and N_C(q) for genuinely later queries without consulting the span's original KV during prediction. Freeze the candidate before revealing those queries and keep that memory span fixed while testing query transfer. The present reader-policy comparison also changes question-span KV, so it does not alone isolate that transfer. Measure numerator and denominator errors separately, their coupled effect through native attention/MLP, and storage/evaluation cost against the original span. Count any dense complement read explicitly: success there would establish a reusable span response, not compact sufficient chat state.
