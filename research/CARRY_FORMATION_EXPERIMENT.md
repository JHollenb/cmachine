# Inherited carry or newly written carry?

Status: completed2026-11-06by`job-0e69e4a9b62e`; [findings and correction history](FINDINGS.md#inherited-and-newly-written-carry-form-a-coupled-input). Both isolated corrections worsen fidelity while their joint carried correction improves it. Original protocol and its prospective definitions follow. The next unexecuted cut is the [actual native-writer intervention](WRITER_INTERVENTION_EXPERIMENT.md).

The preceding query shows consequential omitted carry and a substantial nonlinear interaction with the read complement. It does not show where that carried component originated. An immediate layer-15 trace must distinguish inherited state from newly added writes before naming layer 15 as the producer.

Retain the original checkpoint, native tokenizer, C FP32 contract, layer-16 B3 bank and its two predecessor hashes. The initial source, its own generated answer and declared Mira calibration stay the same. Freeze the bank before one further fresh question:

> Name the color of the badge belonging to Niko. Return just that color.

Add two selected read-only observation ports for the actual post-output-projection attention write and raw post-attention receiver. Append their enum entries so existing port identities remain stable. Preserve ordinary kernel order and keep tracing off by default. At the chosen prompt token, capture actual layer-15 entry x15, attention write a15, post-attention receiver r15, MLP write m15, output x16 and the existing layer-16 entry/receiver ports. Reach them by replay from the same token parent; these are observations, not new interior checkpoints.

The native state relation is

\[
 r_{15}=\mathrm{FP32}(x_{15}+a_{15}),\qquad
 x_{16}=\mathrm{FP32}(r_{15}+m_{15}).
\]

With the existing frozen P3 projector and the actual layer-16 carried omission ex16, define

\[
 z_I=\mathrm{FP32}((I-P_3)x_{15}),\qquad
 z_N=\mathrm{FP32}(e_{x,16}-z_I).
\]

The second operand is a new-write complement. Compare it independently with (I−P3)(a15+m15) and retain all addition, projection and split-rounding discrepancies. Do not label the complement as a pure native write or silently assign its roundoff to a learned mechanism.

Use five common-parent continuation arms: dense, frozen B3, inherited-only correction, new-write-complement correction and full-carried-omission correction. Each correction uses its original-weight layer-16 G/U images, the full live receiver's rho, exact SiLU/down consumer and actual residual skip. The full-carried arm is a reference for this decomposition; it still omits the layer-16 read complement and is not an exact-native correction. Compare each arm both to that reference effect and to dense native execution.

Retain the signed write, first-logit, current/next KV and own-state continuation effects with equal eight-token answer and later Mira/Niko followup budgets. Reuse the pure joint SiLU/down diagnostic if needed to separate finite interaction from split arithmetic; it changes no continuation state and requires no extra G/U projections. Keep original source native-engineering parity, frozen program identities, actual-work counts and rollback custody as mechanics requirements. No approximation or semantic outcome blocks exploration.

### Compare conditional effects at each actual consumer

This experiment traces the formation of a consequential omission under the frozen B3 interface. It does not yet identify the origin of a semantic badge variable. Let B be the frozen candidate, I inherited-only correction, N new-write-complement correction and F full-carried correction. At each retained consumer h, keep

\[
\Delta_I^h=Y_I^h-Y_B^h,\quad \Delta_N^h=Y_N^h-Y_B^h,\quad
J^h=Y_F^h-Y_B^h,\quad K^h=Y_F^h-Y_I^h-Y_N^h+Y_B^h.
\]

Retain both conditional effects, YF−YI and YF−YN, along with the isolated effects. Compare native-error changes and fidelity to the full-carried reference separately; F still omits the layer-16 read complement. Keep tensor norms, signed projections, every energy cross term and undefined zero-response ratios. The exact tensor identity is J=ΔI+ΔN+K; norm fractions are not independent causal percentages. If the split-joint correction is evaluated only as a pure local diagnostic, its local discrepancy does not bound an unexecuted split-joint continuation.

Apply these accounts at the local MLP write, residual exit, full first logits and each captured key/value layer at current and common-next token clocks. Bind each occurrence to the checkpoint/numerical profile, frozen bank, common parent, actual receiving operands, correction definition, position and token input. The common-next arm holds the input token fixed. Own-state followups retain their generated histories; after histories differ, their outputs are no longer the same-token-history numerical contrast.

The preceding panel demonstrates why this matters: read correction alone worsens downstream fidelity while the full correction after carry closes the remaining native error. Its operational interaction is 29.99% of the joint correction at the write, 53.76% at first logits and 64.24% at common-next upper K/V. These have different denominators and do not establish cross-port amplification. The [derived consumer records](results/job-00038c17cfcd/consumer-contrasts.json) preserve this new posthoc analysis without changing the original prospective report.

Select the next reversible cut using the combined consumer evidence. A new-write effect that remains consequential conditionally motivates splitting attention15 from MLP15; an inherited effect motivates moving the boundary earlier. Retain coupled routes when neither isolated arm explains their joint consequence. No component is discarded because its isolated correction worsens a score. No rank growth, automatic routing, training, target fit or accurate-free-chat claim is part of this experiment.

### Frozen implementation and execution geometry

The actual driver captures five layer15 ports in each target arm during ordinary sequential stepping, then uses the existing layer16 entry and receiver. The raw-receiver port precedes any experimental plane replacement. Old port IDs remain unchanged. The explicit correction API validates model, layer, phase, extent and finite values, then stages its supplied device operand in an owned buffer before attention or normalization scratch can overwrite an alias. It reuses the same two G/U calls and nonlinear/down consumer.

Expected physical panel: one real-model lifecycle check, two ordinary source paths, four source controls, one Mira calibration and five fresh target arms, ten own-history followups and one pure joint down diagnostic. All replies/followups have eight-token budgets and512-token capacity. One CUDA model is resident at a time. The new owned operand is6144GPU bytes; assay split buffers add18,432bytes. Frozen B3 program bytes remain unchanged. Expected lease runtime is about one minute based on the predecessor, not a speed claim.

Use one mrun CUDA lease on Beast, one CPU thread, strict preflight and no retries. Preserve the measured C-backend reservation exception:6144MB RAM/VRAM, based on predecessor peak4001.1MB RSS/6378MB VRAM and6758.4MB enforced VRAM ceiling. Full first logits are retained because the consumer analysis needs their complete numerical effect; selected-color scoring would omit that observable. Static selected-score campaigns do not implement this persistent-KV/custom-C continuation contract. The panel reuses one in-process immutable parent and resident weights instead.
