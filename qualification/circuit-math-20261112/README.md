# Circuit mathematics: local qualification

November 12, 2026. Implemented on isolated branch `codex/cmachine-circuit-math-20261112`, from committed cmachine `8822497`. The canonical checkout contains unrelated ongoing changes; those were neither modified nor silently included in this worktree.

## Scope

The actual original C Qwen equations execute with the repository's CPU GPU stub and supplied tiny parameters: width eight, three layers, four query heads, two KV heads, head width two, MLP width twelve, vocabulary sixteen. No pretrained model, learned sparse model, training, Beast lease or CUDA execution is used. The separately recorded source hashes identify this implementation. Exact archived command outputs and exit codes are beside this file.

`mechanics_status`: locally qualified. `instrument_status`: native-equation tiny fixtures and independent numerical checks completed. `trend_status`: computational examples, no semantic inference. `terminal_status`: no pretrained or semantic certification.

## Measurements

- Passive capture and subsequent analysis preserve full native logits, all valid K/V, clocks and pending ownership byte-for-byte against an ordinary branch.
- All eight first-layer input derivatives of a final token-3 minus token-4 margin are checked by central native FP32 differences. Maximum absolute discrepancy: `0.00025065756268305428`, dose `0.002` on each side.
- Historical key and value source derivatives sum the two query heads sharing KV group one and follow every upper native consumer. Maximum absolute discrepancy: `0.00001336324285816648`.
- The selected source-axis frequency coefficients sum to `0.043859201231422673`; the corresponding native scalar is `0.04385919925268103`. Keeping its `-1.9787416427607951e-9` native arithmetic remainder closes the account.
- One finite endpoint comparison projects the MLP write through its baseline block-output adjoint. Gate: `+0.027799880369095097`; up: `-0.26137026093995935`; interaction: `+0.13067075913363924`; arithmetic remainder: `-7.722914556279893e-9`; total: `-0.10289962916013955`. This is a local write projection, not an additive decomposition of the full finite output change. Every neuron's product partial is independently checked against its original down-column dotted with this same output adjoint.
- The same comparison retains maximum absolute normalization/projection/product-secant discrepancies of approximately `1.96e-7`, `6.95e-8`, `2.02e-8`. These explicitly separate FP64 identities from native arithmetic.
- Identical endpoints produce zero finite write components. A same-token contrast gives zero adjoints. Stale session states, unsupported bindings and existing observers refuse. A nonfinite captured operand aborts the open token and an ordinary retry reproduces native continuation.

## Checks

`make all circuit-mechanics state-mechanics operand-mechanics bounded-mechanics sanitize` builds native/model-free executables and runs all four sanitizer-backed mechanics lanes. Existing bounded-cache compilation emits its preexisting unused-function warning. The new circuit modules compile with `-Werror`.

The Python unittest suite passes all 26 tests, including eight new independent compiled-C/NumPy tests covering randomized finite MLP/RMS identities, RMS adjoints and curvature, prompt-free read geometry/even-odd identities, attention Q/K/V VJP, query curvature, large common score offsets and the Fourier commutator identity. The C mechanics lane additionally checks DFT inversion/convolution, coalition inversion, interface witnesses, selected parameter hydration, writer/reader relationships, captured finite attention changes and consumer-relative frequency closure.

The native CLI frontend is compiled/linked against the CPU stub and malformed invocations are checked; the actual `circuit-cuda` build and pretrained model loading are unqualified here. No CUDA header/toolchain is assumed on this MBP.

The [original-core comparison](verify_native_reference.py) builds the committed base Qwen core and stub independently of the modified versions. Across eight native steps, all full logits, valid all-layer K/V, positions and pending tokens are byte-identical: 4,032 bytes, SHA256 `819e7acd26b552fdafe83c0ecda19034a61f4b89533d0ace4ae085691bf2d1cf`. This checks the instrumentation/stub change against the old implementation, beyond comparing two branches using the new implementation.

## Instrument defects found and corrected

Initial smoke runs caught an incorrect weight-info API assumption, followed by the CPU stub's historical score scratch differing from CUDA's packed normalized layout. The inspection uses an explicit weight-profile getter; the stub now publishes CUDA's score layout while preserving its original attention-return arithmetic. A derivative test initially replayed one token too late; moving its reference parent to the exact captured prefix corrected the instrument. None of these failed development checks is presented as model evidence.

Final review also corrected the finite-write example's selected projection from the pre-MLP receiver adjoint to the block-output adjoint. Both are mathematically valid projections, but only the latter is the derivative at the MLP write's actual consumer port. The final numerical values above and archived logs use that port. The earlier exploratory projection is not presented as a downstream-write decomposition.

## Next native qualification

When Beast is available, admit one sequential Qwen2.5-1.5B original-FP32/CUDA specimen through mrun. Verify the original bundle, capture/sham/replay parity and declared numerical profile. Preserve finite dose curves for both current-input and historical-source edits; compare local predictions with native margins and full logits, keeping rounding and nonlinear departures visible. Test source/head controls and own-token continuation separately. Qualification should precede claims about pretrained circuit behavior. No broad benchmark or default scientific promotion gate is required to collect those observations.

Direct MDB graph/symbol selection should invoke these APIs through a versioned report bridge, retain original parameter/replay references, and hydrate temporary operands. That UI/IPC bridge and arbitrary edited-session manifest replay are future work; this branch does not claim to implement them.
