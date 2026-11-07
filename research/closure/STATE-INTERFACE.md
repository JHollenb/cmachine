# What is retained and where the next operands come from

2026-11-06. Read/write inventory for the original Qwen2.5-1.5B-Instruct C FP32 machine and the local B3 interface tested in [PLAN.md](PLAN.md). This is a state contract, not a claim that the full chat state has been compressed.

## Complete token boundary

The executable state retains all valid layer/head K/V rows, their cursor/capacity, a pending token, whether it was delivered, position, session/reset/lineage ownership, greedy top-two metadata, and the branch stack. Fixed model parameters, tokenizer/template, numerical profile and deterministic tie policy are program inputs. External user text supplies new token IDs and turn events. It supplies no hidden states or future cache rows.

At an interior layer boundary the live residual, layer/phase and necessary temporary operands must also be retained. A token-boundary cursor does not represent that interior continuation. The existing C stack is session-local and restores complete token boundaries over immutable retained prefix rows; it does not install arbitrary independently predicted caches.

## Native dependency ledger

| Computation | Reads | Writes and later consumer | Authority |
|---|---|---|---|
| Embed pending token | Actual pending token; original embedding | Initial residual for this position | Branch-owned token or supplied external user token |
| Input RMS | Current residual; RMS weights/epsilon | Normalized input to Q/K/V maps | Derived from current branch |
| Q/K/V maps and RoPE | Normalized residual; original affine parameters; actual position/theta | Positioned query/key and value | Derived from current branch |
| Cache append | Current K/V; layer/head/cursor | New row, visible to current and later reads | Branch-owned row; immutable earlier prefix |
| Attention | Current positioned Q; visible retained/current K/V; mask and support | Head outputs, then original output projection | Complete declared context support |
| Post-attention addition | Incoming residual; actual attention write | Receiver r for RMS/MLP and the residual skip | Actual finite addition |
| MLP input RMS | r; learned gamma; epsilon | rho and normalized input | Derived from current branch |
| Gate/up maps | Normalized input; original dense weights | Gate and up vectors consumed by SwiGLU | Original computation, or explicit qualified replacement |
| SwiGLU/down | Gate/up; original down weights | MLP write M(r) | Declared finite primitive/profile |
| Residual addition | r; actual consumed MLP write | Next layer's residual | Actual finite addition |
| Higher layers | Their residual and their own K/V history | Later residuals, current higher K/V, final state | Full branch-owned continuation |
| Readout and selection | Final norm/head; greedy/tie policy | Actual logits and next pending token | Branch-owned output |
| Delivery/commit | Pending/delivered/consumed flags; EOS and budget policy | Next input event and cache cursor | Session contract, distinct from displayed text |

This ledger answers executability: the complete native machine can form every operand it actually reads. It does not imply that a chosen smaller interface retains those operands or their necessary distinctions.

## The local16byte proposal

The local code contains three executed FP32 QR coefficients and one native FP32 RMS scale. The static B3 bank, original parameters and execution profile are additional inputs. PROJECT combines the frozen gate/up basis images with those coefficients/rho, evaluates SwiGLU/down and emits a write. It discards the orientation of the receiving remainder.

The full receiving r remains available to the experimental host for its skip, diagnostics and correction controls. Therefore the entire running model is a hybrid full-state host, not a16byte chat machine. A fixed source basis also does not provide future attention support or new-token query/key/value construction.

For one invocation the missing information can be supplied by e=r−Bc, or by its two original-weight images Wg(gamma⊙e) and Wu(gamma⊙e), alongside the other operands. These are sufficient paid inputs to the local factorization. The experiment does not infer a cheap transition for them; retaining two intermediate-width images is not a storage improvement over a hidden-width vector.

## What is observed and what is supplied

The source conversation, old calibration basis, three source receivers, candidate construction and intervention boundaries are supplied. No weights or compact representations are learned in this assay. The native MLP and suffix determine the measured writes, logits, cache rows and generated tokens. Spherical geometry guarantees real-arithmetic projection/norm identities; actual encoder equality is measured in the unchanged finite runtime.

The selected alternate is chosen by exact code equality before its native write is evaluated. Its write is then installed through the existing explicit held-write operation, with source residual skip fixed. This is a conditional-input/write intervention, not a claim that the alternate receiver arose from a natural conversation.

PROJECT versus native measures the compact local law. Native-write sham checks installation/replay. The alternate complement repair predicts the alternate write; it is not meant to restore the source write. Corrected recurrence separately restores each branch's actual receiving complement and pays its dense projections.

## The next-token dependency boundary

MLP16 occurs after current K/V16 formation. Changing its write can first change current K/V17. On the next matching input token, current K/V17 is formed before layer17 reads its changed old memory. Thus a newly appended K/V17 row can equal native while the retained prefix and later logits remain different. A future residual effect through that old layer17 memory can first write current K/V18; this assay does not capture K/V18.

Once emitted tokens diverge and are consumed, lower-layer computations may differ too. No individual matching token ID or matching new cache row restores a common history. The complete retained prefix and causal clock belong in the state description.

## What remains to be discovered

The practical reduction target is a representation that preserves the permitted future reads and has its own update law. The existing affine positive control demonstrates this under a supplied operation algebra. Here the proposed B3 code is challenged on actual nonlinear primitives. A new encoding may retain more directions, projected complement information or a restricted query/operation family; its bytes, computation and domain must be declared before it is called a reduction.
