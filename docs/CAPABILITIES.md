# Cmachine capabilities

Assessed November 6, 2026, with an independent Astra review of the source and experiment evidence. The reviewed checkout is `ae0fcd4`, including the operand APIs introduced in `5078b5d`. Each numerical qualification below retains its own executed source identity; this inventory does not extend an older receipt to newer code.

`cmachine` is an explicit model state machine. C owns the numerical computation, mutable state, output selection and continuation. It directly executes two nominated model profiles: the R3 direct-effect ISA Mamba specimen on CPU and Qwen2.5-1.5B-Instruct on CUDA. It also exposes opt-in replacement operations and owns the relocated research programs under [research/](../research/README.md).

For these qualified native paths, the major numerical computations are explicit, including the computations that obtain operands and advance state. This is stronger than observing tensors or predicting from a supplied final activation. It is distinct from recovering a general semantic instruction set or a substantially cheaper implementation. Ordinary Qwen serving still evaluates the dense learned model.

## Coverage at a glance

| Surface | Available capability | Scope |
|---|---|---|
| Native ISA execution | Instruction-byte decoding, contextual reads, three Mamba mixers, post MLPs, output heads and architectural commits | One R3 profile; CPU C/libm |
| Native language execution | Prompt ingestion, all decoder layers, full vocabulary readout and greedy generation from owned K/V | Qwen2.5-1.5B-Instruct; C/CUDA/cuBLAS |
| Persistent chat | Later user turns, exact token-prefix reuse, pending/delivered/consumed ownership, reset and continuation | Python tokenization outside numerical inference |
| State control | ISA row checkpoints; Qwen token parents, nested rollback/commit and token abort | Profile/backend-specific recovery contracts |
| Storage policies | Native FP32 cache, full byte cache, bounded retained rows and original BF16 weight storage with FP32 tiles | Explicit alternatives; approximation and storage changes remain separate |
| Inspection | Selected Qwen operand/write ports, full readouts, state hashes and work/allocation counters | Optional tracing; borrowed pointers are read-only |
| Public experimental operations | Q/K/V programs, first-layer lexical tables, receiver planes/subspaces, writer projections and gate laws | Explicit C API selection; no automatic semantic dispatcher |
| Experimental compact reader | Recent rows plus counted prototypes, and a query-response pullback | Fixed geometry; caller owns integration and rewind |
| Research compilers | Larger MLP factors, connected MLP/attention replacements, memory experiments and causal replay | Separate drivers and frozen run profiles |
| Broader model understanding | FLUX, its Qwen3 conditioner/VAE, smaller-Qwen readout laws and learned multiplication | External workspace implementations, not additional C backends |

The canonical runtime is this repository. `saturn/cmachine` is a workspace compatibility link to `cmachine/research`, whose live runtime aliases resolve back here. [Provenance](PROVENANCE.md) and [integration boundaries](INTEGRATION.md) describe the migration and external dependencies.

## Native Qwen computations

The adapter checks the following geometry and numerical configuration in [qwen.c](../src/qwen.c) and [the exporter](../tools/bundle.py). It is not a general Hugging Face model loader.

| Property | Supported profile |
|---|---|
| Tested checkpoint family | Qwen2.5-1.5B-Instruct |
| Residual width / layers | 1,536 / 28 |
| Query heads / K/V heads / head width | 12 / 2 / 128 |
| MLP intermediate width | 8,960 |
| Vocabulary | 151,936, tied embedding/readout |
| RMS epsilon / RoPE theta | `1e-6` / `1e6` |
| Configuration position limit | 32,768; actual allocation and qualified horizons are separate |
| Parameters | 338 named tensors; supported original BF16 or FP32 bytes |
| Execution | FP32; cuBLAS pedantic; C/CUDA contraction disabled |

Scaled RoPE, sliding-window attention, untied heads and different geometries are outside this adapter. The CUDA build currently targets NVIDIA `sm_89`, matching the qualified RTX 4080 profile. The CPU GPU stub is a mechanics instrument, not a CPU Qwen backend.

For a current residual `x`, each layer performs:

1. Learned input RMSNorm and biased Q/K/V projections.
2. Half-split rotary position transformation of Q and K at the absolute token position.
3. Append of the actual current K/V before the attention read.
4. Full causal grouped-query attention. Query head `h` uses K/V group `floor(h/6)`; scores use the `1/sqrt(128)` scale, stable softmax and weighted value mixing.
5. Output projection and residual addition.
6. Learned post-attention RMSNorm, gate/up projections, SiLU/product, down projection and residual addition.
7. After the last layer, learned final RMSNorm, tied full-vocabulary projection and deterministic top-two/greedy selection. Equal logits prefer the lower token ID; nonfinite readouts refuse.

For normalized MLP receiver `z`, the implemented write is

$$
M(z)=D[\operatorname{SiLU}(Gz)\odot Uz].
$$

For one head, the attention computation is

$$
A(q)=\frac{\sum_j e^{q^Tk_j/\sqrt{128}}v_j}
{\sum_j e^{q^Tk_j/\sqrt{128}}}.
$$

Per-head results concatenate before the output projection. These formulas summarize real arithmetic; [the CUDA implementation](../src/qwen_cuda.cu) defines reduction order, casts and finite execution. The runtime computes these operations from its own state, rather than obtaining a native-framework activation at each boundary.

Native learned-linear work is 197 map calls and 1,543,569,408 nominal MACs per completed token, before attention products and scalar operations. The vocabulary map alone is 233,373,696 MACs, about 15.12% of that ledger. These shape-derived counts are not hardware counters or latency measurements. [Costed replacement comparison](../research/transition_laws/FINDINGS.md).

## ISA Mamba computations

[isa.c](../src/isa.c) implements the nominated R3 direct-effect architecture: width 48, three mixers, 96 inner channels per mixer, four scan coordinates per inner channel, three-tap depthwise convolution and 87 FP32 parameter tensors. Its input is up to 15 instruction bytes, eight 64-bit registers and four flag bits. It advances 64 bit-position rows per instruction.

The complete path includes:

- A `120 -> 256 -> 256` GELU instruction decoder; destination/source probabilities, width/length/syscall predictions and contextual register reads.
- Positional/query-dependent source and immediate reads, flag queries and a projection into the 48-coordinate working state.
- Each mixer's input split, causal convolution, SiLU, selection parameters, softplus timestep, exponential decay/write recurrence, skip, gated read and output projection.
- Residual addition and each post block's LayerNorm, `48 -> 96 -> 48` GELU branch and residual write.
- Eleven grouped output-head families: register bits, flags, destination, source, width, length, syscall, immediate, store flag, store address and store value.
- Architectural commit: thresholded register/flag predictions and predicted instruction length advance the next instruction's register/flag/byte-offset state.

An individual scan-coordinate transition has the affine form

$$
H^+=D_t\odot H+W_t.
$$

The learned decay/write coefficients, gates and post MLPs are computed live. Scan and convolution histories reset at each instruction; committed architectural state persists between calls. Store/address/value/syscall heads remain visible outputs. The continuation consumer does not implement byte-memory writes, operating-system effects or a complete hardware ISA.

The [public ISA API](../src/machine.h) provides `isa_begin`, `isa_advance`, `isa_commit`, `isa_checkpoint` and `isa_restore`. Row checkpoints copy the complete mutable `IsaState` and bind model, law and policy. CPU execution uses sequential FP32 reductions, libm nonlinear functions and disabled contraction; it is not a port of an official fused Mamba backend.

A separately loaded `TCM_ISA_LAW` bundle supports an explicit anchored response substitution at post layer 2, row 8. It contains two operand directions and five FP64 linear/quadratic response arrays. Dense and quadratic arms operate on the supplied anchored receiver; the predicted write feeds ordinary heads and subsequent instructions. This is a bounded response program, not automatic discovery of natural applicability or replacement of every ISA MLP.

The initial [C/native ISA comparison](../research/FINDINGS.md#complete-isa-execution-and-a-substituted-mlp-law), `job-709fe816a540`, matched all sixteen tested architectural commits with maximum same-input error `6.10e-5` across the eleven head families. Row-31 restore was exact within C. Five quadratic trials each omitted one dense MLP row, and all ten resulting committed states matched native anchored continuations. These dependent cases qualify the numerical path and local law; they do not certify arbitrary instruction correctness.

## State ownership and conversation control

[qwen.h](../src/qwen.h) exposes layer-by-layer execution through `qwen_begin_token`, `qwen_advance_layer` and `qwen_finish_token`. `qwen_push` consumes a supplied ID; `qwen_advance` consumes the pending predicted ID. A delivered prediction and a token already consumed into K/V are separate events.

The persistent [server](../src/qwen_serve.c) accepts token IDs over a line protocol. The [text client](../tools/chat.py) uses the local tokenizer/chat template, verifies available tokenizer bindings, reuses an exact processed prefix and resets/recomputes when rendering changes that prefix. It supports later user turns, explicit supplied memory text and interactive parent/rollback commands. Tokenization and presentation remain outside C numerical execution.

| Control | Native prefix cache | Full-byte / bounded serving |
|---|---|---|
| Stored history | Complete FP32 prefix up to configured capacity | Encoded complete prefix, or declared retained support |
| Nested parents | Up to 64 token-boundary markers | Two snapshot-backed parents |
| Recovery mechanism | Restore cursor, pending/top metadata and lineage; retained prefix stays immutable | Restore encoded bytes, addresses, eviction population/RNG, logits and control metadata |
| Token abort | Restore the compatible pre-token parent | Restore the complete pre-token snapshot, including overwritten rows |
| Disk persistence | No portable session serializer | No portable session serializer |

Native cursor rollback does not restore an old full-vocabulary logits buffer; the server marks it invalid until a new token completes. Pending/top metadata is restored. Cursor markers cannot undo arbitrary overwritten prefix rows. `QwenCacheStep`, `QwenCacheGuard` and `QwenCacheRewind` let explicit backends own that mutable closure; callers must implement compatible restoration.

Some experimental layer failures can leave a partial token or changed current cache. The runtime exposes abort and specific retry contracts; it does not promise universal failure-atomic execution of arbitrary replacement code. Equal current layer outputs also do not imply equal historical higher-layer K/V. [State contracts](ARCHITECTURE.md).

## Memory and parameter storage policies

The [integrated runtime](INTEGRATED-RUNTIME.md) uses the same `build/tcm-cuda` executable for native, `full-byte` and `bounded` serving. Native remains the default. Alternatives require explicit selection.

| Policy | Computation and representation | Important boundary |
|---|---|---|
| Native FP32 | Original matrices widened at load; full FP32 K/V | All-layer K/V payload is 57,344 bytes per token, plus workspace/metadata |
| Full byte | Current K/V encoded as signed bytes with per-row scales; complete retained prefix decoded for attention | Lossy cache representation; storage still grows with context |
| Bounded | Anchors, recent ring and reservoir-sampled older rows, with declared weighting and absolute positions | Fixed physical support, not a semantic summary or unlimited recall |
| Tiled FP32 weights | Widened matrices consumed through a shared 1,024-row FP32 tile | Matched arithmetic control for source-storage comparisons |
| Source-BF16 weights | Original BF16 matrices retained and widened into the same FP32 tile | All learned parameters remain; arithmetic is FP32, not a BF16 forward |

The weight-tile profile uses process-global ownership and a shared tile; it is not a concurrent multi-model weight manager. Tiling changes finite map geometry relative to untiled native execution, so storage comparisons use the matched tiled-FP32 control.

The qualified bounded policy retains 16 anchors, 64 recent rows and 32 sampled rows. Its 112 physical rows use 9,279,620 tracked GPU state bytes including recovery. Original model/helpers/tile bring tracked allocation to 3,133,714,584 bytes. These are requested allocations, excluding CUDA/library overhead and host transcript storage. The integrated real-model qualification uses logical context 768; the source configuration limit does not imply every horizon has been tested.

Bounded state advancement, occupied-sample rollback and retry are qualified. Delayed recall still fails on the retained panel; the full-byte reference itself gives an incorrect arithmetic sum. Storage reduction, useful memory, numerical fidelity and task truth remain separate measurements. [Evidence and observed failures](STATUS.md).

## Public experimental computation interfaces

These operations are implemented in the canonical runtime but are not enabled by ordinary chat. See [the API](../src/qwen.h), [operand implementation](../src/qwen_operands.inc) and [query-program guide](QUERY-PROGRAMS.md).

| Interface | Computation available | What it does not supply |
|---|---|---|
| `QwenQkvProgram` | Encode the actual learned-RMS receiver once, `c=B n`; selected Q/K/V maps become `H_j c`; unselected maps remain native | Automatic basis discovery, semantic routing or quality guarantees |
| `QwenLexicalProgram` | Compile/load/save finite first-layer biased pre-RoPE Q/K/V atoms; lookup bypasses that layer's original RMS/QKV online | Contextual lookup for later layers, online fill or removal of later learned work |
| `QwenPlane` | Compile three receiver/carry/read basis directions into G/U images; dense, compiled and held-write comparisons at the guarded parent | General applicability outside its bound receiver/session/clock |
| `QwenSubspace` | Reuse two columns, optionally admit one explicitly calibrated third column, compute live coordinates/RMS, inspect omitted operands | An arbitrary-rank general MLP compiler or an exact sufficient chat state |
| Complement corrections | Correct the full omitted operand, carried component, read complement or supplied operand before the nonlinear write | Free repair: corrections can pay original G/U projections |
| `QwenWriter` | Project actual attention and/or MLP writes into a supplied three-column basis, with computed/consumed ports | Avoiding original writer computation or discovering semantic operations |
| `QwenGateLaw` | Source-anchored sigmoid polynomial, native and zero modes on actual branch operands | Removal of the dense G/U/down maps in polynomial mode |

QKV programs are immutable model/layer-qualified objects with explicit selected-map masks. Different layers can use different programs. Binding changes require an empty session without retained parents; the caller must keep attached program storage alive. Original biases and RoPE execute exactly once after the compiled maps. Work counters survive rewind and cover attached operand-program work, not the complete model ledger.

First-layer lexical tables exploit token locality before RoPE. A missing ID refuses before token mutation; a full-vocabulary table is implementable, but the real-model qualification covers 15 IDs and sixteen full-logit steps. Table persistence checks model/shape/content and assumes the supported little-endian FP32 format. Offline dense compilation, host storage and recurring row transfers remain paid.

### Compact reader and local response derivative

[prototype_attention.h](../src/prototype_attention.h) implements 16 recent rows plus 16 counted older prototypes per K/V group. Its carrier occupies 98,464 host bytes for this geometry. It appends actual new K/V, computes count-weighted softmax reads and exposes the streaming pullback

$$
J_q^T e=\frac1{\sqrt{128}}
\sum_i p_i k_i\big((v_i-a)^T e\big).
$$

This is the smooth FP64 query derivative at fixed existing memory, excluding FP32 output rounding. It does not differentiate prototype assignment/merging, historical state formation, the rest of the model or future argmax decisions. The kernel owns no session or checkpoint stack. An attached caller must restore all mutable prototype bytes; a native cursor cannot undo a merge.

## Qualified research operations and results

The relocated research tree contains executable compilers and diagnostic drivers beyond the public serving ABI. Their sources, supplied inputs, consumers and result limitations matter as much as their operation names.

| Program family | Demonstrated computation | Result boundary |
|---|---|---|
| [Contextual query compiler](../research/context_query/FINDINGS.md) | Source-only response update through actual compact memory; frozen transfer through fresh two-turn own-token histories | Mean conditional KL improves 13.59% across two tasks at two prompt lengths; some followups worsen; most model work remains native |
| [Query/key/history cut](../research/query_key_cut/FINDINGS.md) | Same-parent Q/K crosses, current-layer exact repair and from-zero selected-map policies | Native keys rescue one long native-parent binding; exact current repair cannot erase inherited higher-layer memory differences |
| [Connected MLP and reader](../research/operand_composition/FINDINGS.md) | A rank-64 MLP writes the actual residual consumed by a rank-128 Q/K program, followed by own tokens and another turn | About 1.85% whole-token nominal linear saving, with mixed fidelity and material interactions |
| [All-layer MLP factors](../research/transition_laws/FINDINGS.md) | Rank-256 gate/up programs across all 28 MLPs, with one candidate-state coverage refresh | 40.895% nominal learned-linear MAC omission; both all-layer policies give unusable replies and 0/8 strict answers versus native 7/8 |
| [Attention read account](../research/composition/attention_read/FINDINGS.md) | Fixed-query numerator/partition updates and actual receiving-MLP continuation | Qualified on supplied native operands; does not generate arbitrary future queries cheaply |
| [Span response](../research/composition/span_response/FINDINGS.md) and [recurrent bytes](../research/composition/span_recurrent/FINDINGS.md) | Prospective span summaries and recurring reads without accessing retired selected rows | Specific approximation and storage results; some simple answers also survive deletion |
| [Nonlinear state quotient](../research/state_quotient/FINDINGS.md) | A supplied compact state advances its own sixteen-update trajectory through native primitives; an off-space write exposes omitted orientation; another coordinate repairs it | Controlled known-wiring closure, not learned compact Qwen state; rank constraint applies to one strong common-linear-space construction |
| [Owned append and retention](../research/composition/README.md) | Live encoded append, current reads, eviction and complete overwrite recovery | Integrated mechanics are available; sufficient long-context memory is not established |

The later Q-only layer18 policy preserves two known inventory trajectories but worsens explanations; its four-conversation mean KL is 20.62% worse than updated17 alone. These are development contexts, not new held-out success. The all-layer covariance refresh improves source coverage and local writes without restoring conversation. Both countertrends remain part of the capability record.

The query/compiler experiments retain original lower-layer receiver formation, many original projections, readout and parameters. Larger rank-64/256 MLP factors are research-driver implementations; the public `QwenSubspace` API remains the bounded two/three-column interface. A successful research arm is not automatically an installed serving policy.

## Computations understood outside this C runtime

The broader statement that we understand major computations across several models refers to explicit, qualified numerical accounts in the research stack. Those implementations are useful evidence and potential porting targets. They are not additional architectures accepted by `tcm`.

The following links are workspace references and require the surrounding `domains` checkout. The corresponding MDB reference documentation is also in the public [saturn-pub repository](https://github.com/JHollenb/saturn-pub).

| External specimen | Computations explicitly accounted for | Supplied boundary and portability limit |
|---|---|---|
| FLUX.2 Klein 4B | Working image/text projection, time modulation, five joint and twenty single blocks per pass, Q/K normalization and RoPE, attention, activated MLPs, fused/residual writes, flow readout and finite Euler latent commit | [Complete functional pass](../../saturn-pub/packages/mdb/docs/FLUX-FUNCTIONAL-PASS.md); original parameters and pinned Torch/CUDA primitives remain |
| FLUX Qwen3 conditioner | Lexical input, demanded causal decoder prefix and conditioning assembly `E=concat(h9,h18,h27)` from the first 27 blocks of the 36-layer, width-2,560 conditioner | [Raw prompt to image](../../saturn-pub/packages/mdb/docs/FLUX-RAW-TO-IMAGE.md); this demanded prefix is not a C Qwen3 chat backend |
| FLUX image decoder | Latent unpack/scatter and normalization plus explicit learned VAE decoding into pixels | Same raw-prompt account: conditioning, all 112 denoiser transitions and VAE reach the original pixel consumer; checkpoint, seed, masks, casts, scheduler and decoder profile qualify equality |
| Qwen2.5-0.5B-Instruct | Supplied final-MLP receiver through RMSNorm, SwiGLU, residual, final norm/head; full-vocabulary dose envelope and first-token prediction | [Forecasts](../../saturn-pub/packages/mdb/docs/QWEN-COMPUTATION-FORECASTS.md), [fresh contexts](../../saturn-pub/packages/mdb/docs/QWEN-PROMPT-GENERALIZATION.md); observed prefix and original weights supplied; later tokens are native observations |
| ISA/Mamba contextual mechanisms | Affine scan composition, contextual state reader, gate/projection lowering, LayerNorm/GELU response and own-state architectural continuation | [ISA mechanism documentation](../../isa/docs/consumer-geometry-and-mlp-context.md); each source/receiver/intervention has its own applicability |
| Learned bilinear multiplication | `W((Ua) ⊙ (Vb))` compiled to an exact three-product Karatsuba-style identity, with recursive integer composition proved by induction | [Results and scope](../../model-experiments/experiments/2026-10-16-mlp-law-composition/RESULTS.md); bilinear family, recursion and exact compiler supplied; external Python arithmetic executor |

The complete raw-to-image specimen reports 1,011 bit-exact field comparisons across 614,793,216 scalar positions under its shared primitive profile. That establishes a finite numerical composition result, not an independent C/CUDA FLUX backend, a general semantic decomposition or a speedup.

Semantic findings add another level: particular routes maintain bindings, carry appearance changes or expose arithmetic state under declared interfaces. A head, row or MLP can change its causal role with its receiver. The symbol table records qualified computations and applicability; the time circuit records dependencies, clocks and lifetimes. Neither currently induces an automatic universal semantic program for chat.

## Evidence and verification

There are distinct qualifications:

| Evidence | What it establishes |
|---|---|
| [Initial independent Qwen port](../research/FINDINGS.md#independent-qwen-prefill-and-generation), `job-52f2231facaf` | Three own-state C trajectories match native FP32 token sequences; maximum first-readout error is about `2.67e-4`. Native BF16 can choose different sequences; floating-state equality is not claimed |
| [Installed integration](evidence/integration-7a6824d/README.md), source `7a6824d`, `job-f645fcd76a75` | 84 original/integrated C protocol entries agree exactly; compressed-policy behavior, actual clients and overwrite/abort/retry recovery pass in the declared panel |
| [Context-query source record](evidence/context-query-20261106/README.md), `job-fe51e7f4f0ee` | New public interfaces replay 420 historical source clocks; lexical subset and source-derived query transfer have dedicated numerical audits |
| [Query/key causal records](evidence/query-key-cut-20261106/README.md), `job-1555c27c080f` and `job-1335a953cd08` | Same-parent branch closure and known-context from-zero mask comparisons under their frozen source manifests |

The installed integration proof is for `7a6824d`; it does not certify every later operand API. Later experiments bind their actual executed sources separately, and do not change the installed serving default by virtue of a successful run.

The [bundle loader](../src/model.c) validates identities, tensor descriptors and extents. Experiment verifiers additionally join source manifests, model/input identity, raw hashes, parent lineage, scheduler results and saved numerical analyses. Standard evidence sweeps currently have `target-streams-only` coverage. Their clean status does not independently certify internal numerical operations, semantic meanings or general capabilities.

From the repository root, the existing model-free lanes are:

```sh
make
python3 -m unittest discover -s tests -v
make sanitize
make state-mechanics
make bounded-mechanics
make operand-mechanics
python3 validation/run.py verify docs/evidence/integration-7a6824d
```

`make sanitize` builds the ISA sanitizer binary. The state, bounded and operand targets execute model-free fixtures under sanitizers; they do not load a pretrained model or qualify CUDA parity. CUDA build/primitive tests and real-model scientific runs are separate lanes. Real-model or multi-step scientific work in the domains workspace uses its configured mrun/shared environment and declared CUDA admission. [Validation protocol](../validation/README.md).

Core numerical serving does not require Saturn or Torch. The text client needs tokenizer dependencies; offline export, analysis and fleet helpers have their own dependencies. Pretrained weights are not distributed. Large raw research arrays and many historical bundles are local ignored artifacts; compact tracked records do not make those inputs available in a standalone clone.

## Current limits and the target

Current capabilities do not include arbitrary model/configuration loading, a native C FLUX/VAE or smaller-Qwen backend, a CPU/MPS Qwen executor, stochastic/beam decoding, a general optimizer or whole-model autograd, portable interior session serialization, unlimited-context memory, an automatic law-discovery/router service, or a broadly faithful cheap whole-chat transition. The prototype query pullback is narrower than autograd, and saved lexical tables are narrower than serialized conversation state.

Unused prompt vocabulary-readout elimination is proposed in [VISION.md](VISION.md); `qwen_finish_token` currently pays the full head. General partial-token recovery and broader state installation also require explicit contracts. No serving speedup follows from nominal MAC omission, reduced cache bytes or avoiding a framework `forward()` call.

The supported simplifying target is an economical representation `E` and transition `G` with

$$
E(T(S,u))\approx G(E(S),u),\qquad
O(S)\approx H(E(S)),
$$

over the candidate's admitted states and future inputs, including its own generated tokens and later user turns. Full retained state already provides an executable transition; the unresolved work is finding cheaper reusable operand/read/write laws with adequate coverage. State size, native fidelity, task correctness, useful continuation and total execution cost remain separate outcomes.

Related workspace syntheses: [what we know how to compute](../../obsidian/blog/2026-11-06-124240-what-we-know-how-to-compute-inside-a-model.md), [model computation and remaining chat work](../../obsidian/blog/2026-11-06-200500-model-computation-and-the-remaining-work-for-free-chat.md), and [producing versus consuming state](../../obsidian/blog/2026-11-06-185200-producing-a-state-and-consuming-it-are-different-computations.md).
