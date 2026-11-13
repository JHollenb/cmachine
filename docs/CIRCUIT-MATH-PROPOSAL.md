# Native circuit mathematics in cmachine

Status: implementation proposal, November 12, 2026. Base: `8822497`. Work is isolated from the canonical checkout's ongoing, uncommitted integration work.

Implementation update: all six mathematical families now have C APIs, with an opt-in native Qwen capture/consumer adapter and CUDA inspector frontend. See [the runbook and proofs](CIRCUIT-MATH.md) and [recorded local qualification](../qualification/circuit-math-20261112/README.md). This is model-free qualification; pretrained CUDA execution and direct MDB UI integration remain pending.

## Purpose and user flow

Understand a computation by selecting its occurrence, hydrating its operands, expanding the original equations, and following the result into a declared consumer. Speed is not an objective. No learned representation, new predictor, weight export, or scientific quality threshold is required.

The first implementation has two layers: a framework-free C mathematical library, and an opt-in observer/analysis adapter for the existing Qwen executor. The adapter captures one consumed token through its native layers. Captures are temporary inspection caches. Durable records identify model digest, session/reset/parent, token position, layer, port, parameter name and selector. A pointer is a borrowed live address, not a durable replay root.

1. Attach at a token boundary, consume one token normally, and seal the capture.
2. Choose a vocabulary token or token contrast; pull that consumer back through every remaining native layer. Historical K/V is held fixed; the current token's Q/K/V paths all participate.
3. Inspect a selected source/head's probability, centered value, routing coefficient and payload direction, or a neuron's gate/up arguments and signed consumer sensitivity.
4. Compare two compatible captures with exact finite MLP and attention accounting, including observed-minus-ideal numerical remainders. Endpoint secants explain changes; they do not forecast unknown endpoints.
5. Expand selected parameter relationships and frequency coefficients on demand. Test proposed state interfaces against repeated arguments with differing outcomes.

## Six capabilities

### Finite response

For `m(n)=D[SiLU(Gn)*Un]`, let `s=SiLU(g)`, `c=delta(s)/delta(g)` (the derivative when the gate endpoints coincide). Then

`delta(m)=D[diag(mean(u)*c)G + diag(mean(s))U]delta(n)`.

Evaluate this as factored operator actions, never as a hidden-by-hidden matrix. Preserve gate, up, interaction and native arithmetic remainder terms. RMS normalization has its own exact finite action: `delta(n)=gamma*(rho1*delta(r)+(rho1-rho0)*r0)`. Compute the scale difference by rationalization, retaining epsilon, baseline alignment and edit energy.

### Writer/reader links

`W J_RMS(r) D = rho W gamma D - rho^3/d (W gamma r)(r^T D)`.

Inspect requested writer columns and reader rows only. The first factor is reusable; the rank-one correction is contextual. This is a local relationship, not proof of a semantic edge.

### Consumer-connected attention

`dA=sum p*dV + sum p*(V-A)*dL`; for downstream adjoint `lambda`, `dM/dL_j=p_j dot(lambda,V_j-A)`. Extend the existing prototype-reader pullback to actual full-memory grouped-query attention, O, residual, RMS, MLP, final normalization and the original vocabulary head. No gradient of greedy token selection or floating-point rounding is claimed.

For arbitrary same-support finite changes, evaluate the stable equivalent of

`delta(A)=sum p*((exp(delta(L))-1)*(V-A)+exp(delta(L))*delta(V)) / sum p*exp(delta(L))`.

### Curvature and coalitions

One neuron's normalized-port Hessian action is determined by its two original read rows: `b*SiLU''(a)*g*g^T + SiLU'(a)*(g*u^T+u*g^T)`. Provide directional actions, exact even/odd and quadratic read geometry, and Boolean-lattice interaction coefficients. A full consumer Hessian also includes upstream and downstream curvature; a local product term is not the whole output interaction.

### Frequency coordinates

Provide explicit unnormalized token-axis DFT and inverse, circular convolution, product accounting, and cyclic-shift commutator diagnostics. One fixed receiver uses a source-axis correlation, not a square token-transport spectrum. Keep signed real/imaginary coefficients, phase, cancellation and unlisted terms. Finite causal boundaries and contextual tokenwise gains also mix frequencies; mixing alone has no semantic authority.

### Interface witnesses

Compare exact typed argument identities only within the same declared model, execution profile, program, consumer and horizon. Retain same-arguments/different-result witnesses and incomplete coverage. Near matches are diagnostics, not exact collisions. No universal sufficiency claim follows from a finite sample.

## Integration and evidence boundaries

Ordinary serving remains the existing native program. Observation is explicit and must leave logits, K/V, clocks and pending-token ownership byte-identical. Unsupported compact caches or replacement programs must refuse this original-equation adjoint rather than silently differentiate another computation. Native FP32 ports anchor an FP64 smooth-derivative inspection lane; its primal/remainder diagnostics are separate from C/CUDA parity.

No persistent parameter bank or full Jacobian/Hessian is created. Existing source checkpoint/bundle and exact token/edit history remain replay requirements; these captures do not add portable interior StateCuts or universal cache eviction. MDB can consume C reports through stable metadata; direct MDB UI wiring is a separate integration and must not be described as implemented here.

## Verification

Use sanitizer-backed model-free native-executor fixtures, independent directional differences, no-op observation/replay checks, finite-change closure, GQA source controls, tied-token contrast controls, zero/coincident secants, numerical extremes, malformed inputs and interface witnesses. Run the existing lifecycle/operand/cache regressions. No Beast or pretrained campaign runs while Beast is unavailable; a later Qwen2.5 qualification must retain the native numerical profile, full logits, signed dose curves and actual continuation.

## Mathematical sources

- [Expanded native chain](../../obsidian/blog/2026-11-12-152729-expanding-native-computation-without-dropping-variables.md)
- [Native read geometry](../../obsidian/blog/2026-11-12-161144-causal-attention-and-native-mlp-read-geometry.md)
- [Finite changes and systems/signals proofs](../../obsidian/blog/2026-11-12-165523-when-circuits-mix-frequencies-and-how-finite-edits-propagate.md)
- [Composition and semantic proof requirements](../../obsidian/blog/2026-11-12-172148-composing-native-mathematics-into-semantic-circuit-proofs.md)
- [Existing state-interface mathematics](../research/closure/MATH.md)
- [Existing attention response mathematics](../research/query_response/MATH.md)
