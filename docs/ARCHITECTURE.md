# Architecture and state contracts

## Execution ownership

`cmachine` makes the execution boundary explicit. Immutable model parameters define the program. The session owns the mutable carrier and control state. The caller supplies input tokens or instruction bytes, and C computes outputs and the state that the next operation consumes.

| File | Responsibility |
| --- | --- |
| [model.c](../src/model.c) | mmap bundle loading, SHA256, descriptor extents and profile checks. |
| [isa.c](../src/isa.c) | ISA model equations, row advancement, checkpoints and architectural commits. |
| [program.c](../src/program.c) | Shared vector-graph dispatcher, owned convolution/scan state, greedy continuation, suffix replay and explicit trace/intervention controls. |
| [qwen.c](../src/qwen.c) | Qwen layers, token clock, K/V ownership, experimental operations and greedy readout. |
| [qwen_cuda.cu](../src/qwen_cuda.cu) | cuBLAS maps, RMSNorm, RoPE, attention, SwiGLU, cache and top-two primitives. |
| [qwen_serve.c](../src/qwen_serve.c) | Persistent single-session token protocol. |
| [qwen.h](../src/qwen.h) | Session, checkpoint, observation and operation APIs. |
| [qwen_operands.inc](../src/qwen_operands.inc) | Explicit model/layer-qualified Q/K/V programs and finite token-local first-layer tables. |
| [prototype_attention.h](../src/prototype_attention.h) | Experimental bounded-memory law and streaming query-response pullback; caller owns rewind. |
| [bundle.py](../tools/bundle.py) | Offline authenticated bundle export. |
| [program.py](../tools/program.py) | Trusted FX or canonical Mamba1 lowering, parameter/state allocation and sealed source maps. |
| [inspect_program.py](../tools/inspect_program.py) | Derived producer/consumer symbols and optional trace-row joins. |
| [chat.py](../tools/chat.py) | Tokenization, transcript bookkeeping and prefix reuse outside numerical execution. |
| [program_chat.py](../tools/program_chat.py) | Source-bound native tokenization and a fresh C graph process per request. |

The [supported-model inventory](SUPPORTED-MODELS.md) distinguishes the dedicated ISA/Qwen paths from the graph path. The latter compiles PAN and canonical Mamba1 operations to one 14-op schedule. Its bundle digest covers the schedule, vector descriptors, parameters and numerical configuration. A source map joins each instruction and recurrent slot to an exact source reference; semantic roles remain separate hypotheses. Unsupported operations or shapes refuse.

Graph execution advances one input row at a time. C owns the recurrent floats and greedy readout, leaving the last selected token pending. Suffix replay restores prompt-boundary state and logits and compares the resulting state, logits and tokens exactly within C. Consumed rows and pending-token ownership are reported separately; this is not the complete canonical StateIR transaction API or Qwen's persistent conversation protocol. [Measured source/C logit error](SUPPORTED-MODELS.md#measured-logit-error) qualifies numerical fidelity separately from replay.

## Two clocks and one causal history

At an interior Qwen boundary, the state includes the current residual, token position, layer and phase, valid K/V rows, pending/delivered-token status, session/reset identity, parent lineage and continuation stack. Model identity and numerical profile qualify that state. A complete token boundary has all current rows formed and a pending prediction; interior observation pointers alone cannot resume execution.

For layer ℓ, the order is:

\[
n=\operatorname{RMSNorm}_{in}(x),\qquad
q=\operatorname{RoPE}(W_qn+b_q),\quad
k=\operatorname{RoPE}(W_kn+b_k),\quad v=W_vn+b_v,
\]

\[
(K_{\ell,t},V_{\ell,t})\leftarrow(k,v),\quad
a=W_o\operatorname{Attention}(q,K_{\ell,\le t},V_{\ell,\le t}),
\]

\[
r=x+a,\quad z=\operatorname{RMSNorm}_{post}(r),\quad
m=W_d[\operatorname{SiLU}(W_gz)\odot W_uz],\quad x^+=r+m.
\]

The current layer's K/V forms before its attention read and MLP write. An MLP16 intervention therefore leaves that token's KV16 unchanged while potentially changing later layers. Later equal current rows do not establish equal prefix memory. The completed formation/read and attention studies demonstrate this distinction in actual execution.

The decoder uses 12 query heads and two K/V heads of dimension 128. Query head h uses K/V head floor(h/6). All-layer FP32 K/V costs 57,344 bytes, or 56 KiB, per token. That numerical carrier is additional to weights, program coefficients, scratch and metadata.

## Token ownership and continuations

`qwen_begin_token`, `qwen_advance_layer` and `qwen_finish_token` expose the existing operation order. `qwen_push` consumes a supplied token; `qwen_advance` consumes the session's pending prediction. Delivery to a client and consumption into cache are distinct events.

The client tokenizes the complete proposed transcript and checks whether its processed token history is still an exact prefix. It appends only a matching suffix. If an existing prefix changes, it explicitly resets and recomputes. It checks capacity before mutating client or C state, and transport failures end the session rather than blindly repeating a possibly applied mutation.

Checkpoint handles are bound to model/session/reset and parent identity. Equal position is insufficient. Nested rollback restores the retained parent's cursor and metadata; commit keeps the child and removes the parent marker. Prefix K/V remains immutable and children append above it. The 64-frame array costs 9,216 host bytes in the imported version, with no extra K/V copy per marker.

This cursor stack does not restore arbitrarily overwritten old cache rows or serialize portable sessions. A child whose result is returned as text can be handled by restoring the parent and explicitly ingesting that text. Raw sibling caches cannot be merged as a general call/return operation.

## Observation and replacement

The optional [circuit inspector](CIRCUIT-MATH.md) captures original native operands and connects finite-response, writer/reader, attention, curvature and frequency mathematics to a first-readout consumer. It adds no trained predictor or serving replacement. The [proposal](CIRCUIT-MATH-PROPOSAL.md) separates implemented APIs from remaining MDB UI and pretrained qualification work.

Observers receive selected borrowed device pointers during callbacks. They are read-only ports, not mutation callbacks or resumable StateCuts. Tracing is off by default. Captures add synchronization and transfer cost and are not neutral timing instrumentation.

The experimental APIs distinguish computed writes from the operands actually added to the residual. Their binding scopes differ:

- `QwenPlane` binds one guarded receiver-plane invocation and compiled gate/up images.
- `QwenSubspace` binds a model/layer and frozen two- or three-column bank across changed receivers.
- `QwenWriter` binds a supplied projector to one actual session/parent/clock and modifies selected writes.
- `QwenGateLaw` binds frozen source coefficients; each invocation binds its actual parent and live dense operands.
- `QwenQkvProgram` binds frozen basis/images to a model and layer; it encodes the actual live receiver once for selected Q/K/V maps.
- `QwenLexicalProgram` binds a finite first-layer token table to a model; missing IDs refuse before input mutation.

These APIs do not automatically route normal chat through a discovered law. Corrected modes may pay original projections, and every such call remains part of the computation cost.

## Numerical contract and failure

The C model uses the qualified FP32 CUDA/cuBLAS profile, original bias/RoPE order, explicit normalization and deterministic greedy ties. Reassociating equations can change finite results. Byte parity with another backend is a measured property for a named specimen and boundary, not an architectural assumption.

Original model and bundle checks, dimensions, device ownership, finite values and valid continuation state protect trustworthy execution. Scientific quality metrics remain observations during exploration. A failed optional approximation does not establish that its mechanism is absent.

Some experimental failures currently leave a partial token; failure-atomic continuation is not a general property of every replacement API. Callers must follow the returned state contract and replay/reset where required. The proposed interface improvements are documented in [VISION.md](VISION.md); they are not implemented by this extraction.
