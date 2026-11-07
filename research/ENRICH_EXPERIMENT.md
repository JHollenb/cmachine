# One explicit contextual operand

Created 2026-11-05 before the prospective run. This is an exploratory computation-transfer assay, not certification of general chat or a learned semantic variable.

The preceding frozen B2 assay, `job-96e7f0b9e3ae`, omitted 33–37% of two natural later receivers and incurred 72–83% MLP-write error. Dense residual correction repaired the write to about 6e-7. A posthoc analysis found cosine 0.87258 between the two omitted operands. That motivates one new calibration direction; it does not establish transfer to the following queries.

The specimen remains Qwen2.5-1.5B-Instruct, checkpoint 989aa798, original weights widened once to FP32, native tokenizer, C/cuBLAS FP32 on Beast. The selected site is zero-based layer 16 at the final generation-prompt token, with the unchanged later layers, full tied output head, own generated tokens, and own KV as consumers. Use one mrun lease, one model bank, one CPU thread, CUDA admission, 512-token capacity, eight-token answer budgets, and no retries. There is no target refresh, fallback, or answer-correctness gate.

First compile B2 = [x0,a0] and its four G/U images from:

> Mira's badge is red. Niko's badge is blue. What color is Mira's badge? Answer with the color only.

Run the four source controls already used in the preceding assay. Then rebuild the native source and generate and consume its own answer, including EOS. All later question parents descend from that own-generated state.

The one declared calibration question is:

> Tell me Mira's badge color, just the color.

Capture its actual entry x, actual pre-add attention output write a, receiver r = FP32(x+a), and B2 omitted operand e. Copy e into a separate calibration buffer before generation. Orthogonalize it twice against the frozen FP64 QR columns; round to the actual stored FP32 operand b. Compute a new QR column from that rounded b, then explicitly compile G(gamma*b) and U(gamma*b), costing two GEMVs. Publish rank three only after finite-value checks. This changes the computational program; rolling back the calibration question restores the session while intentionally retaining the explicitly compiled operand. Preserve the original B2 basis, images, QR, and baseline write byte-for-byte.

Freeze B3 = [x0,a0,b] before either fresh target:

1. What is the color of Niko's badge? Reply with one color.
2. Which color badge does Niko have? Give only its color.

Each fresh target uses independent common-parent dense, B2 projection, B3 projection, and B3 residual-correction arms. B2 uses its original two-column solve, not a truncated three-column solution. Solve in FP64 and execute the recorded FP32 coefficients. Materialize rhat = FP32(FP32(c0*x0+c1*a0)+c2*b), and e = FP32(r-rhat). Use rho from the full live receiver. Combine unscaled images in the same order, multiply by rho once, and retain exact SiLU, elementwise multiplication, down projection, and the actual residual skip r. Both projection candidates use zero target G/U GEMVs. The correction arm adds two dense projections of gamma*e, scaled by the same rho, without overwriting the candidate result.

Save all raw first logits, actual executed MLP ports, x/a/r/rhat/e, frozen programs and QR, current upper-layer KV, and one common-next-token upper-layer KV per row. Save own generated answer IDs and text. Each target arm then independently asks the original Mira and Niko questions from its own delivered answer state, with separate eight-token budgets. Common-next full logits are hashed and summarized; first full logits are retained. Hashes of retained KV prefixes and both frozen program components must remain unchanged across branches. The whole process uses the public Saturn receipt bridge; this is not per-CUDA-operation receipt coverage.

The required observations are B2/B3 omitted norm, G/U/write error, first-logit and later-KV error, own-path behavior and later margins. Report signed carried-versus-attention residual attribution separately from semantic interpretation. Correct colors alone cannot establish computational fidelity, as the preceding held-write controls already demonstrated. Report calibration cost separately from target skipped work: four initial plus two extra compilation projections versus four skipped projections per candidate across these two fresh questions. No net speedup claim follows from this panel.

Only source/model/tokenizer identity, finite/schema/shape contracts, program immutability, actual work instrumentation, replay/rollback, and native-engineering parity may stop this finite run. Scientific errors, target disagreement, and attribution magnitudes remain observations. Do not classify a poor transfer result as absence of reusable computation.
