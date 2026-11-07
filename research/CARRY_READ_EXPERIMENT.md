# Which omitted operand causes the consequential error?

Status: completed by [job-00038c17cfcd](results/job-00038c17cfcd/report.json), 2026-11-05. The prospective version is frozen in that job's payload. This follows [partial prospective B3 transfer](FINDINGS.md#one-calibrated-operand-transfers-partially), reviewed with Astra. The pure nonlinear/down diagnostic below was added before submission, when no model result had been observed for this question. [Findings](FINDINGS.md#carried-omission-and-read-complement-through-the-native-consumer) retain the measured corrections, interaction and limits.

The B3 result improves local write fidelity without uniformly improving logits. Its remaining carry/read components partly cancel. Their geometric sizes cannot decide which component matters through the nonlinear consumer. Test them by explicit corrections before expanding the bank again.

Keep the same original Qwen checkpoint, native tokenizer, C FP32/cuBLAS numerical contract, selected layer 16 and 512-token capacity. Use the same initial Mira/red, Niko/blue source, its own generated and consumed source answer, and the one declared Mira calibration. Freeze B3 before the new question:

> Niko is wearing which badge color? Respond with the color alone.

Use independent common-parent dense, B3, B3-plus-carried-omission, B3-plus-read-omission and full-residual-correction arms. No new basis column, automatic enrichment, target fitting or fallback is permitted. Capture actual x, pre-add attention write a, rounded receiver r, executed reconstruction rhat and omitted operand e = FP32(r−rhat). Preserve the complete selected nonlinear write, first logits, current/next upper KV, own eight-token answer and independent later original Mira/Niko questions with eight-token budgets. Consume delivered EOS before later questions. Keep whole-process Saturn receipt coverage explicit.

Let the frozen FP64 QR define P3 = QQᵀ. Form the carried omission from (I−P3)x, recording its FP32 rounding. Define a complementary operand ea = FP32(e−ex) from the actual executed omitted operand, rather than assuming that separately projected operands add exactly in FP32. Record the FP64 remainder e−ex−ea and compare ea to the independently projected actual attention write. The full correction must still use the original actual e.

For z in {ex,ea,e}, calculate the diagnostic original-weight projections

\[
 \delta g_z=\rho(r)G(\gamma\odot z),\qquad
 \delta u_z=\rho(r)U(\gamma\odot z).
\]

Each one-component correction spends two dense target GEMVs. The baseline candidate spends none. Use the identical full receiver rho, actual residual skip r, SiLU, product, down projection and native suffix in every arm. The selected normalization input remains an observation even if its scratch buffer is reused for the diagnostic. Record actual work; this is causal localization, not an efficient replacement.

Measure signed effects through the write, current upper KV, common-next KV, complete first logits, own answer and later target/collateral margins. In exact arithmetic, the interaction contrast is

\[
 I_m=m_{x+a}-m_x-m_a+m_{B3}.
\]

After all continuation branches return to the source-answer parent, run one pure diagnostic from captured candidate G/U and the two actual scaled correction arrays. Assemble ((g0+delta_gx)+delta_ga) on GPU, analogously for u, then execute SiLU, the product and one original layer-16 down projection. Save its summed G/U and write. This costs zero extra G/U and one down call. It changes scratch only, with session metadata, retained KV and both frozen program components checked unchanged. It is not a sixth continuation arm.

Use that diagnostic write for m(x+a) in the local interaction contrast. Compare it to the full-e correction write separately, retaining operand-splitting and projection-rounding discrepancies. The diagnostic supplies an executed finite nonlinear/down consumer; it does not supply a joint-arm full-model continuation. Do not attribute a rounding-floor interaction to a learned nonlinear effect. Do not require the carried correction to win: geometry nominates the intervention but cannot decide its result. Opposing effects and a useful joint correction are scientifically informative.

The B2 program must retain SHA256 1a12d8a22bfd00575039b9409d28bbdd44bdaa8f8b207fe1f5f4f5bfb4424d3e, and its third-column extension must retain d33759a30b44473612f219ba00df332ea0330db4b7100c7d7a22e0519c43ddad from job-c034107916a2. Check these before the new target executes. The source four controls, own-source rebuild, declared Mira calibration, five target arms and pure diagnostic all retain raw outputs. Ordinary engineering parity compares the new binary to the tested 8a1f58d28aa01ab4 development build.

If carried correction removes most consequential error, trace its formation through the actual layer-15 attention/MLP writes on a further predeclared query. If read correction dominates, localize the current query's relevant address/payload computation. If the joint effect dominates, preserve the carry/read coupling as part of the sufficient interface. A new reduced program must then freeze an explanation before another new context and consume its own generated state.

Use one short mrun CUDA lease on Beast, one model bank and one CPU thread. Retain raw branches and counter-trends. Integrity, finite/schema contracts, native parity, immutability, actual work counts and replay/rollback are engineering requirements; approximation errors and semantic outcomes remain observations. No training, semantic-register claim, broad chat accuracy claim or speedup claim is part of this panel.
