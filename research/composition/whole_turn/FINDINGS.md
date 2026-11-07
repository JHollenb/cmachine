# A frozen MLP program ingests a correction and reads its own updated context

2026-11-06. The C machine now applies its frozen B3 MLP16 program throughout a complete later user turn, its own acknowledgment, and two independent follow-up questions and answers. It retains Niko's changed green badge and Mira's unchanged red badge. Deleting that same writer also preserves both answers. The new result closes the missing ingestion boundary while retaining a mixed numerical trend; it does not establish compact sufficient chat state or a semantic advantage over deletion.

The [prospective plan](PLAN.md) and [source-bound report](runs/job-a071c7a43a38/report.json) define one update in one supplied-fact conversation. The [Astra review](REVIEW.md), [offline verification](runs/job-a071c7a43a38/verification.json), and [additive saved-data audit](runs/job-a071c7a43a38/additive-audit-v2.json) preserve the distinction between behavior, local fidelity and continuing state.

## What changed from the previous assay

The [earlier composition panel](../FINDINGS.md) substitutes only at the final query token and generated tokens; earlier question ingestion remains native. Here every appended template token, user token and generated token uses the selected policy at zero-based MLP16. The authenticated source and its own red+EOS answer remain native, with cursor59 and the historical complete valid-KV hash verified before candidates begin. No target calibration, training, state refresh or fallback occurs.

The original Qwen2.5-1.5B-Instruct checkpoint executes in the historical C FP32/cuBLAS profile. The supplied276536-byte B3 bank is imported unchanged. PROJECT approximates the current receiver by its three basis coordinates, skips two G/U projections and retains native RMS scale, nonlinear consumer, down projection and actual residual skip. CORRECT adds the two original-weight complement projections before the nonlinear consumer. Zero supplies a6144-byte owned zero write through the existing held-write API and executes neither G/U nor down at that site. Original weights, residuals and all-layer KV remain.

## Actual conversation and authoritative tokens

The later user message is:

> Correction: Niko's badge is now green. Keep the other person's badge unchanged. Reply only with OK.

After each branch's own acknowledgment, two independent children ask:

> What color is Niko's badge now? Reply with the color only.

> What badge color does Mira still have? Reply with the color only.

The second reader never consumes the first reader's answer. Both begin from that policy's actual post-correction state.

| Policy throughout ingestion and generation | Acknowledgment | Later Niko | Later Mira |
|---|---|---|---|
| Native | OK + EOS | green + EOS | red + EOS |
| Recurring B3 PROJECT | OK + EOS | green + EOS | red + EOS |
| Recurring zero-write | OK + EOS | green + EOS | red + EOS |
| Recurring CORRECT | OK + EOS | green + EOS | red + EOS |

Every answer has two tokens, including EOS; IDs are3925,13250,1151 followed by151645, respectively. Every emitted token is consumed. All replies end within their8-token budget, so no host-forced closing delimiter is needed in this specimen. Unit checks cover that framing path and distinguish151643 from the template's151645. Generated IDs are never decoded and re-tokenized to rebuild history; decoding occurs only afterward for reporting.

The append suffixes contain33/25/25 tokens including the already consumed151645. Skipping that exact duplicate leaves32/24/24 ingested tokens, followed by two answer tokens each. Each policy therefore advances86 observed transitions:34correction,26target and26collateral. All policies happen to generate identical histories here, so their corresponding state contrasts have no different-history confound.

The behavior demonstrates processing and retaining this update through a changed writer policy. It does not distinguish B3 content from tolerance of writer deletion, and the unchanged source context remains available in full.

## Same-text dense replay qualifies the local reference

All12primary turn records seal before diagnostic execution. The seal covers the primary JSON prefix, including its token histories, full boundary-logit hashes and complete valid-KV hashes. Individual per-token tensor-file hashes are collected afterward; they are not all part of that pre-diagnostic seal.

Three paid dense diagnostic paths replay the nonnative policies' actual tokens, each with one correction replay and two independent reader children. They are forced same-text diagnostics, not autonomous native answers. Their values never feed the primary candidates.

The architecture predicts a triangular dependency at this site: changing MLP16 can change higher state but cannot change the earlier computation under an identical consumed history. Across all258candidate/dense comparisons, captured layer16 entry, raw receiver, normalized MLP input and KV16 are byte-identical. This qualifies the dense MLP return as a same-input local reference. The supplemental audit also finds the dense diagnostics equal the native path on all258steps' six captured fields and all18full boundary readouts/cache hashes, because all histories match. Its1584checks include those selected arrays and boundaries; they are not independent scientific samples.

| Policy | Same-input local write comparisons | Relative error range | Mean |
|---|---:|---:|---:|
| B3 PROJECT |86|62.30–118.22%|101.85%|
| Zero |86|100%|100%|
| CORRECT |86|5.56×10⁻⁷–7.46×10⁻⁷|6.60×10⁻⁷|

At69/86transitions, PROJECT's local write is farther from the native write than zero. CORRECT restores the expected arithmetic floor while paying dense G/U work. Corrected full states are not byte-exact: finite reassociation remains measurable.

## The numerical benefit depends on the consumer boundary

For a boundary logit vector, the relative difference is \(\|Y_\pi-Y_N\|_2/\|Y_N\|_2\). Prefill is the actual readout that selects the first answer token. The final boundary follows consumption of EOS; it diagnoses the next continuation state, rather than the logits that selected the delivered EOS.

| Turn | PROJECT pre-answer | Zero pre-answer | PROJECT after consumed EOS | Zero after consumed EOS |
|---|---:|---:|---:|---:|
| Correction acknowledgment |36.92%|39.65%|21.05%|20.19%|
| Niko read |17.07%|21.02%|22.92%|20.53%|
| Mira read |20.49%|25.48%|23.12%|20.85%|

PROJECT is closer to native at all three pre-answer readouts and farther away at all three post-EOS readouts. Its content has a measurable numerical effect even though deletion preserves the same answers. These dependent examples do not establish a general quality ranking. CORRECT's relative full-logit differences remain approximately6.24×10⁻⁷–9.73×10⁻⁷across the six boundaries.

The raw margins retain another distinction. Mira's first-token top-versus-rest margin is6.658native,6.125PROJECT and0.887zero. For Niko it is4.382native,5.719PROJECT and4.677zero: zero is closer by that margin even though PROJECT is closer over the full vocabulary. Both select the correct token. No one scalar captures every aspect of the consumer.

## Factoring the complete policy effect

Astra proposed a saved-only signed account before another run. Define

\[
e_Z=Y_Z-Y_N,\qquad \Delta=Y_P-Y_Z,\qquad e_P=Y_P-Y_N=e_Z+\Delta.
\]

Then

\[
\|e_P\|^2=\|e_Z\|^2+\|\Delta\|^2+2\langle e_Z,\Delta\rangle.
\]

The [additive audit v2](runs/job-a071c7a43a38/additive-audit-v2.json) evaluates complete saved logits in FP64. Each following term is divided by that boundary's zero-error power:

| Turn / boundary | Added effect power | Signed cross term | PROJECT error power |
|---|---:|---:|---:|
| Correction / pre-answer |0.2853|−0.4181|0.8672|
| Correction / post-EOS |0.3130|−0.2263|1.0867|
| Niko / pre-answer |0.6443|−0.9851|0.6592|
| Niko / post-EOS |0.5117|−0.2658|1.2459|
| Mira / pre-answer |0.4561|−0.8092|0.6468|
| Mira / post-EOS |0.4881|−0.2587|1.2294|

The cross term is negative at every boundary: the added PROJECT effect opposes part of zero's error. At the pre-answer readouts this cancellation outweighs its added power; after EOS it does not. Thus the reversal is not simply a change from cancellation to reinforcement. The pointwise vector identity closes exactly in FP64; summed-power remainders are at most3.02×10⁻⁹from reduction ordering.

This is the effect of the complete PROJECT policy, including the cache it formed. It does not isolate one MLP write or assign causal percentages to modules. The original executed report and first additive audit remain unchanged, with both audit-source versions retained.

## Resources and evidence

Single mrun job `job-a071c7a43a38` succeeds in31.22scheduler seconds; worker26.05seconds, Cdriver18.26seconds. Sampled peak RSS3993.1MB and VRAM6378MB. One resident model, CUDA batch1,512capacity, one CPU thread, no retries. It performs59historical-source token advances plus344primary and258diagnostic advances,661full token computations in total.

The602observed selected-layer transitions execute860original G/U calls and516down calls, with zero compilation calls. PROJECT skips172G/U calls across its86transitions; zero skips another172G/U and86down calls. CORRECT pays172G/U calls. Dense diagnostic work is included in the totals. Capture, hashing, file output and analysis make this unsuitable as a serving-speed benchmark. The installed development chat build is untouched.

The exact tokenizer-derived bound is749observed transitions and65,440,768raw bytes, below the96MiB archive cap. Actual retained raw evidence has4215files totaling51,165,949bytes. Offline verification authenticates the scheduler log/report joins,67frozen sources, every raw hash/extent, the whole-invocation Saturn receipt and the exact executed analyzer. The additive audit supplies21declared-question/template joins, native diagnostic replay and the signed account. Astra independently checks all source/raw hashes, histories, parents, seal and numerical summaries.

Eight token-authority tests and Csyntax checking pass before launch. The standard Saturn [run](index/job-a071c7a43a38/evidence-run.json) and [verify](index/job-a071c7a43a38/evidence-verify.json) cover270source-bound rows with8applicable predicates, no fires/claims/addresses. That is target-stream coverage, separate from the custom array audit and scientific interpretation. No terminal certification is requested.

## What the next cut should separate

Whole-turn own-state execution is now measured. Compact-state sufficiency, useful semantic necessity, unbounded recall, arbitrary math/chat and serving savings remain open.

The useful next experiment is an update-episode × read-episode factorial. Let \(\alpha\) select dense or PROJECT throughout the correction and its acknowledgment, and \(\beta\) select dense or PROJECT throughout each later question and its answer:

\[
Y_{\alpha\beta}=O\!\left(T^{\rm read}_\beta\big(T^{\rm update}_\alpha(S_0,U)\big)\right).
\]

Dense/dense and PROJECT/PROJECT are already observed. The two new mixed policies would ask whether the pre-answer advantage comes from formed context, current reader computation, or their interaction. A dense reader on PROJECT-formed memory is an explicit hybrid; it does not restore native memory. Existing same-text dense diagnostics regenerate the correction densely and therefore do not constitute that mixed arm. Freeze any such follow-up before executing targets, preserve own generated IDs and keep deletion's successful behavior visible. This is a proposed cut, not a second run performed here.
