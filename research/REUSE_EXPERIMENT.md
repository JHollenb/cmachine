# Reusing a frozen receiver program at a later natural question

Prospective plan, November 5, 2026. This follows the successful [controlled receiver plane](PLANE_EXPERIMENT.md) and [its findings](FINDINGS.md#compiled-nonlinear-receiver-plane-and-its-continuing-state). The question is whether a frozen interface to one receiver predicts a later receiver formed naturally by the same model. No target-dependent refresh, fit, donor state or silent native fallback is allowed.

## Frozen source and live target

Use the same Qwen2.5-1.5B-Instruct checkpoint, C FP32/cuBLAS contract and zero-based layer 16. The source prompt supplies Mira's red badge and Niko's blue badge, then asks for Mira's color. At the final prompt token, freeze the incoming carry x0, attention write a0, native MLP write, and four coefficient vectors:

    B = [x0, a0]
    Ax = G(gamma*x0), Aa = G(gamma*a0)
    Ux = U(gamma*x0), Ua = U(gamma*a0).

Compile four extra projections once. Freeze a float64 two-column QR, including the small triangular factor and conditioning diagnostic. Record/hash the complete numerical program before targets. The previous specimen has condition approximately 3.76; the implementation needs a finite nonzero two-column QR, not a semantic applicability threshold.

Consume the model's own source answer, including its end token. From that immutable parent, independently append the native template for two predeclared questions:

1. “Tell me Niko's badge color, just the color.”
2. “Tell me Mira's badge color, just the color.”

These are two correlated probes from one context. The different requested binding and repeated target are separate observations, not independent specimens. The selected receiving token is the final generation-prompt token, with all preceding question tokens computed normally. Native full-token concatenation must match prompt IDs plus the appended suffix, accounting for an already consumed assistant-end token.

## The computation and controls

From each target's actual raw receiver r, compute c=B†r by two float64 dot products and a QR backsolve. Explicitly cast and record the two FP32 coefficients used by CUDA. Materialize rhat=cx*x0+ca*a0 in the declared FP32 multiply/add order, then e=r-rhat from those same materialized values. Record r, rhat, e, the double and executed coefficients, and the actual RMS scale rho. The receiving-state residual skip remains the complete live r.

The candidate uses

    ghat = rho*(cx*Ax+ca*Aa)
    uhat = rho*(cx*Ux+ca*Ua)
    write = D[SiLU(ghat)*uhat]
    output = r+write.

Normalization uses the complete live receiver throughout. It must not use RMS(rhat). Compare five arms from each identical token parent:

- Dense native: G[(r*rho)*gamma], U[(r*rho)*gamma].
- Frozen-subspace candidate: skip both target G/U matrix-vector calls.
- Residual correction: add rho*G(gamma*e), rho*U(gamma*e) to the candidate before SiLU. It spends two dense projections and earns no savings claim.
- Reassociated dense arithmetic control: rho*G(gamma*r), rho*U(gamma*r). This isolates the normalization/projection reassociation without discarding an operand.
- Held source write: keep r but use the frozen source MLP write, with no fabricated gate/up ports.

Also run the source receiver itself through dense compilation, frozen-subspace reuse, residual correction and reassociated dense execution. These source controls expose the arithmetic floor and confirm that the new program reconstructs its own source before interpreting transfer. All modes are explicit operation arguments. The earlier exact guarded plane API remains unchanged.

## State and evidence

Save actual selected MLP input/gate/up/write/output ports where computed, full first logits, current higher-layer KV17–27, and short eight-token own-generated answers. Also save one common-next-token upper-KV row and its full-logit hash, top and margin; its full logits are not retained for magnitude-error analysis. Consume every delivered token. On independent children of each later arm's own answer, ask the original Mira and Niko color questions with eight-token budgets and record token IDs, decoded text, lowercase red-minus-blue margins and truncation. Preserve failed/wrong/empty candidate answers and their state. Answer correctness and omitted-state norm are observations, never reasons to replace the candidate.

Hash the retained KV prefixes and frozen program before and after each branch. KV-prefix hashes do not seal pending/delivery/readout metadata; native checkpoint mechanics check those separately. Physical projection counters are outside rollback. Guard model/layer identity and finite values; a late instrumentation failure restores the saved entry residual for explicit recomputation/reset. This reusable numerical operator is not bound to one session/position, unlike the previous forced receiver-plane act. It changes no retained prefix KV.

One finite mrun CUDA lease on Beast; one resident weight bank, 512-token capacity, one CPU thread, no training, no retries. Use the measured C profile: 6,144 MB RAM and 6,144 MB VRAM reservations, 6,758.4 MB enforced ceilings, previous peak 6,378 MB VRAM. The new program/workspace is less than 1 MiB. Expected execution is under five minutes. Compare the ordinary path to the preserved installed binary. Raw execution, sources and public whole-process receipt remain bound to the exact job; no per-CUDA-operation Saturn receipt is claimed.

The terminal question is not requested. Mechanics and instrument validity are separate from transfer quality. The smallest useful outcome is an actual live-receiver reuse result with an explicit omitted operand and its native consumer consequence. This does not establish a semantic frame, general question coverage, memory compression or unrestricted chat accuracy.
