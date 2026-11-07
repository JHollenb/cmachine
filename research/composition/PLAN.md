# Frozen computation through its own continuation

Exploratory plan, 2026-11-06. This panel complements the separately implemented actual-writer intervention. It changes no shared runtime source. The executable uses an authenticated snapshot of the completed formation runtime and imports the original frozen B3 program and QR. Original model weights, full residual state, full KV and native attention/SiLU/down/readout remain.

Question: does repeated application of the existing approximate MLP16 program behave differently from one transported approximation, when every later transition consumes its own actual state?

The source conversation remains: “Mira's badge is red. Niko's badge is blue. What color is Mira's badge? Answer with the color only.” Generate and fully consume its native answer. Fork each new question independently from this same source-own-answer parent:

| Question | Budget | External task observation |
|---|---:|---|
| What badge color was assigned to Niko? Reply with just the color. | 8 | Supplied fact: blue |
| Compute 17 * 23. Reply with only the integer. | 8 | Arithmetic truth: 391 |
| Write two short sentences explaining why a reader needs context to interpret a message. | 48 | Retain actual prose and termination; no automatic semantic grade |

Prefill the new question natively through its penultimate token. Four policies share that exact token parent:

1. Native: ordinary original C equations at every transition.
2. Once: frozen B3 PROJECT at MLP16 on the final query token, then ordinary equations.
3. Recurring: PROJECT at MLP16 on the final query token and every subsequently consumed generated token.
4. Corrected: recurring original B3 CORRECT, adding the actual current omitted operand through two original G/U projections. This is an arithmetic control, with no projection-saving claim.

Every emitted token, including EOS or the last budget token, is consumed into each arm's own history. A token that changes later input is never replaced by a native token. Full state restoration occurs only at token parents. No candidate state, bank, dose or policy is selected from target answers.

The source program has real-arithmetic form G(γr)≈Σc_jG(γb_j), U(γr)≈Σc_jU(γb_j), with actual current RMS scale and skip retained, then original SiLU, product and down projection. PROJECT omits r−Σc_jb_j. The frozen columns were supplied by the earlier source/Mira calibration; this panel does not discover a semantic state or calibrate on the new questions.

Retain full first and final logits; actual token IDs, top margins, stop/truncation status and physical G/U/down counts; selected input, receiver, normalized MLP input, write and output at every transition; actual reconstruction/omission when a substitution runs; current K/V16/17; complete valid-KV hashes at the first and final boundaries; and immutable source/program/runtime identities. First-token once and recurring states must match exactly. Native and corrected differences remain measurements, not quality gates.

Compare same-input fidelity only while complete consumed token histories match. After their first different consumed token, equal step numbers describe different-history trajectories. A small omitted receiver norm does not bound the missing nonlinear write. Correctness, native agreement and termination are separate observations. No native mistake or divergent branch is discarded.

Use one finite mrun CUDA lease on Beast, one resident C FP32 model, batch1,512-token capacity, one CPU thread, no retries or training. The C runtime is a measured custom-backend exception: prior formation peak RSS4005.1MB/VRAM6378MB; use6144MB RAM/VRAM reservation with6758.4MB enforced VRAM ceiling. Original frozen B3 payload is276,536bytes; no compile/calibration model calls occur here. Cap the evidence archive at32MiB. Bound execution at600seconds; stop after this12-path panel. Instrumented wall time includes capture and hashing and is not a serving benchmark.

Only identity, finite values, shape/custody, replay closure and resource contracts may stop execution. Outcome quality, omission magnitude, token agreement and relative errors are diagnostics. Terminal certification is not requested.
