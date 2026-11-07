# Source-derived gate/up operands through a conversation

November 6, 2026. User-authorized exploratory execution of Astra's proposed operand-formation cut. Qwen2.5-1.5B-Instruct in the existing C FP32 CUDA/cuBLAS profile, layer16, batch1, context512, CPU1. No training, target calibration, source-anchor refresh or automatic fallback. Original weights and full all-layer K/V remain. Greedy decisions use the complete vocabulary.

## Question and competing explanations

Can one weight-derived shared representation form useful gate/up operands more economically on the receivers produced by its own execution? The scalar gate law retained the expensive operands; B3 skipped their projections but lost consequential input information. This test changes operand formation while retaining the exact existing C SiLU/up-product kernel and dense down map.

Alternatives include useful weight-selected coordinates, generic benefit from adding dimensions, insufficient rank/representation, and locally improved writes whose effects accumulate adversely in context. Finite incorrect trajectories, losing deletion controls, token divergence and truncations are observations, not reasons to discard a branch. No universal-rank, compact-chat, arithmetic-truth, speedup or serving-promotion claim is requested.

## Frozen coordinate and compiler contract

Let r be the actual raw post-attention MLP receiver, gamma its learned post-attention RMS scale, and rho(r) the actual native scalar RMS inverse. Define A=G diag(gamma), B=U diag(gamma). Compile a rank64 orthonormal input basis Q from the original joint operator [A;B], with no activation samples. Use fixed-seed randomized right-subspace iteration with oversampling16 and two power iterations; QR and compilation precision are recorded. The same rank and coordinate convention apply to a fixed-seed random basis.

The candidate computes z=Q^T r, ghat=rho(AQ)z, uhat=rho(BQ)z and mhat=D[SiLU(ghat) times uhat]. Freeze original export hashes, compiler source/options and all compiled factor bytes before any primary trajectory. No candidate invocation may call the selected original G/U matrices. The three derived matvecs, native scalar norm, exact scalar nonlinear kernel and dense D are paid. Reassociation of native (r*rho)*gamma is an approximation, not an FP32 parity promise.

The original selected gate/up pair costs27,525,120 nominal MACs. Rank64 derived projection/images cost1,245,184 MACs and4,980,736 factor bytes per program. That is approximately1.70% of the reported whole-token linear work removed before overhead; it is not a measured latency saving. The compiled source and random banks both remain resident and all original parameters remain present for other operations and post-seal controls.

## Primary worlds and inputs

Four candidate worlds are source64, random64, unchanged imported B3 PROJECT and whole-MLP16 deletion. Apply each selected policy from the FIRST supplied prompt token, including all later user/header/generated tokens. The weight-derived programs have no native prefix bootstrap. B3 retains its historical supplied source-activation calibration from job-0e69e4a9b62e; do not conflate its dependency with the source-only bases. Native comparisons run after primary sealing.

Three development conversations, each with16 tokens per answer and two answers:

1. Recall: “Mira keeps a red badge and Niko keeps a blue badge. Which badge color belongs to Niko? Answer only the color.” Follow-up: “Now Mira trades her red badge for a green badge. What color does Mira have now? Answer only the color.”
2. Arithmetic: “Compute 17 * 23. Reply with only the integer.” Follow-up: “Subtract 24 from that result. Reply with the integer, then one short sentence explaining the calculation.” Historical native/deletion results differ on this input under a different source boundary; their necessity at this new boundary is an observation to retain, not an assumed success condition. Correct multiplication is391; native-output fidelity and mathematical truth are separate.
3. Prose: “Write two short sentences explaining why a reader needs context to interpret a message.” Follow-up: “Now give one short example using the word bank.” This can truncate at16 tokens; never report it as a complete explanation unless EOS is actually emitted.

Freeze template/tokenizer and actual IDs before driver execution. Every delivered token, including EOS and the last budget token, is consumed exactly once. For later user ingestion, skip an already consumed identical EOS; if the prior answer was externally truncated, append the explicit template closing event. No answer from one world enters another.

## Native consumers and diagnostics

Seal all twelve primary two-turn trajectories, their readouts and compiled identities before native references or target diagnostics. Then execute native own-history continuations, native reconstruction on each candidate's actual consumed IDs, and exact selected-site MLP evaluation on saved actual candidate receivers. Candidate replay and same-cache conditional probes may be used only after sealing; bind any such output to its exact primary parent and keep it out of the primary program.

Retain complete-vocabulary readouts and full valid-prefix cache hashes at initial/follow-up answer and final boundaries, actual emitted/consumed histories, local gate/up/write errors over the selected saved receiver trajectory, signed trends and first divergence. Full-vector losses compare common conditioning only where that conditioning is declared. Native own-history, same-ID reconstruction and exact local same-receiver evaluation answer different questions and must remain separate.

Count actual original selected G/U calls and derived projection/image calls by world. Compilation, bootstrap, dense down maps, all other layers, attention, vocabulary readout, capture/copy/hash cost and post-seal reference calls remain explicit. Native SiLU is evaluated on candidate approximate operands; that does not make those operands native. Source-spectrum energy is diagnostic and does not certify consumer adequacy.

## Mechanics, resources and evidence

Hard stops protect only model/input/source identity, typed shapes, device/finite state, token authority, immutable primary seals and resource/custody limits. An unsafe nonfinite partial arm is preserved as incomplete and must not be continued. No scientific accuracy threshold stops optional evidence collection.

Use an isolated experiment driver and frozen copies of the existing runtime, leaving ordinary serving source unchanged. Model-free local tests check compiler geometry, source dependence, coordinate consistency, counters/seals and analysis arithmetic. Native mechanics smoke uses mrun before the finite panel. Check ordinary native parity under the isolated adapter and source-file identities; synthetic/engineering inputs stay separate from primary scientific comparisons.

One finite CUDA mrun lease per lane, retries disabled, one resident C model; compile source factors in a bounded CPU subprocess while that model remains resident. No second HF model is loaded. Source weights exported for compilation remain near compute, with their hashes and reconstruction path in the collected manifest; compiled factors and scientific observations are collected directly from the exact job. Keep the archive bounded and precompute its worst-case size. Reuse measured C FP32 capacity512 resource evidence (about4GB RSS/6,378MB VRAM), add declared fixed bank/compiler working geometry, and inspect actual telemetry. Do not overlap duplicate full panels.

After collection, authenticate scheduler/log/payload/raw joins, independently recompute saved statistics, run the standard Saturn evidence sweep and verify it, and take custody of durable reports/programs. Preserve failed smokes and raw reports unchanged. Interpret convergence, disagreements and control outcomes before proposing another cut. Update the blog and capability map with the executed boundaries and paid work.
