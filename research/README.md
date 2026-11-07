# C model state machine

The [completed drawer query/reader/history cut](../../artifacts/qwen-relational-reader-cut-20261106/RESULTS.md) now rescues two supplied first bindings with native Q through the unchanged bounded reader. Exact current native attention still fails the shorter compact history because higher-layer historical consumers differ. The [independent audit and compiler revision](drawer_suffix/FINDINGS.md) reuses that run with zero new model calls, authenticates all generated choices, and prepares162source-only native-Q responses through actual compact memory. No new fitted image or serving policy is installed; [revised next assay](query_own_history/NEXT.md).

This research tree has moved into the canonical `domains/cmachine` repository. `domains/saturn/cmachine` is a compatibility symlink to this directory. Its `src`, `tests`, `tools` and Makefile resolve to the canonical runtime one level above; historical frozen payloads and raw evidence stay local and immutable. See the [integrated runtime](../docs/INTEGRATED-RUNTIME.md), [migration provenance](../docs/PROVENANCE.md) and [commit verification](../validation/README.md). The sections below retain the research chronology and original evidence boundaries.

The [matched own-history query experiment](query_own_history/FINDINGS.md) now measures one fixed program under native-generated and its own generated source histories, with independently built native shadows. Same prompts/followup/budgets/image/memory: mean conditional KL rises26.46%, entirely from a card context whose shorter paraphrase preserves both correct bindings; two other contexts improve. No new image or semantic failure follows.12trajectories/1,718advances,33historical arrays/423full-clock pairs,843exact reads and24snapshots verify. [Saved Q/key/aggregation factors](query_own_history/runs/job-47cc27324445/study.json) retain interactions and finite-profile rounding; they do not attribute task correctness. [Review](query_own_history/REVIEW.md), [next cuts](query_own_history/NEXT.md). Serving remains819,200selected MAC/98,464Bmemory with dense lower computation paid.

The [longer query-future experiment](query_horizon/FINDINGS.md) extends the same sources through32-token answers and a followup. A declared future objective selects the unchanged dose2 image, improving source KL1.76% versus dose1; all six images nevertheless reach the same first greedy disagreements. New own-history targets give KL8/12 and L211/12 improvements with identical compact answers. Both compact images answer418−17with401 while native starts241; explanations are capped. Fidelity and task truth need separate consumers.3,898advances,24exact snapshots,89sources/385rawfiles verify. Same819,200 selected MAC/98,464Bmemory; dense model remains paid. [Frozen executed protocol](query_horizon/EXECUTED_PROTOCOL.md), [Astra review](query_horizon/REVIEW.md), [next own-history/shadow cut](query_horizon/NEXT.md).

The runtime now has its own public repository: [JHollenb/cmachine](https://github.com/JHollenb/cmachine), with an independent checkout at `../../cmachine`. Its [architecture](https://github.com/JHollenb/cmachine/blob/main/docs/ARCHITECTURE.md), [current evidence](https://github.com/JHollenb/cmachine/blob/main/docs/STATUS.md) and [vision and gaps](https://github.com/JHollenb/cmachine/blob/main/docs/VISION.md) cover autonomous execution, operand formation, query-qualified memory, the completed attention trace and nonlinear quotient study, and the remaining forward-pass-free chat goal.

This Saturn directory remains the working location for existing campaigns and their raw evidence. Extraction preserves its source and does not move active experiments or replace the installed runtime. The public repository records exact import hashes; later changes here require an explicit source import rather than automatic synchronization.

This prototype executes the R3 direct-effect ISA Mamba model and Qwen2.5-1.5B-Instruct from independently loaded weights. C owns the live state, model equations and output selection. The Qwen backend uses a small CUDA helper and cuBLAS for matrix operations. Python handles offline conversion, native comparison and optional text tokenization.

The new isolated [query-memory program](query_memory/README.md) compiles one Qwen attention component, produces Q/K/V from each branch's actual receiver and repeatedly updates/reads98,464bytes of bounded memory. Three completed Beast experiments retain own-token continuations and exact compact-law replay. Source-basis expansion improves initial attention error from125–128%to24–47%, while follow-ups remain mixed. Full native receiver formation, other layers, weights and selected cache allocation remain paid; this is an experimental component boundary, with no serving change or whole-model speedup. [Results and resource account](query_memory/FINDINGS.md).

The [native-consumer query selector](query_consumer/FINDINGS.md) now evaluates that frozen image family through actual C source distributions and candidate-owned upper caches. It chooses dose2 rather than response-MSE dose1: source mean KL improves0.985%, but context and clock rankings disagree, target gains are mixed and answers unchanged. An eight-token pending audit rules out forced-history mismatch inside that exact window. Same serving budget;1,351 added source advances are paid. The completed C assay's final-summary error is recovered offline without altering the failed report, verified over80sources/651rawfiles and72target snapshots. [Math](query_consumer/MATH.md), [Astra review](query_consumer/REVIEW.md).

The subsequent [response-aware query program](query_response/FINDINGS.md) fits the actual compact-reader response at the same serving budget. One source-only gradient/dose panel lowers source response loss8.36% while slightly worsening native-query reconstruction. All six initial attention/layer-output/logit comparisons improve;22/24 compact L2 and18/24 KL boundaries improve, with three initial KL countertrends and unchanged answers.24paths/2,412advances and48snapshots independently verify. Same B128/819,200 selected MAC/98,464-byte memory; remaining dense computation stays paid. [Derivation](query_response/MATH.md), [Astra critique](query_response/REVIEW.md).

The [same-cost query refit](query_program/FINDINGS.md) now uses the existing 128-coordinate code more effectively. One source-only least-squares query image improves compact native-logit fidelity at 23/24 readouts across three development and three prospective conversations, with identical generated tokens and unchanged 819,200 selected MAC/98,464-byte memory. Gains are modest; the paid native-Q control has larger initial headroom and a later reversal. Native-distribution KL improves at 19/24 readouts, with two initial countertrends despite improved L2. Thirty paths/3,024 advances, 60 exact memory snapshots and nine historical paths/126 distinct arrays verify. A preserved FP32 diagnostic-normalization refusal is resolved by exact zero-model Linux replay plus an additive FP64 correction, with no relaxed tolerance. Dense receiver generation and general chat accuracy remain open. [Math](query_program/MATH.md), [Astra review](query_program/REVIEW.md).

The completed [QK-versus-value cut](query_value_cut/FINDINGS.md) then restores addressing and content separately. Both improve all first readouts, while a prose ranking reversal shows that local attention error does not order full policies. Compiled Q/K with native V removes 73.96% of native selected-QKV products and improves first-logit fidelity at modest local cost. This is a mixed native-map experimental policy; full-token linear savings are only 0.1507% and later outcomes are mixed. [Astra audit](query_value_cut/REVIEW.md).

The [bounded-memory transfer test](query_memory_transfer/FINDINGS.md) now executes that mixed policy from token zero through its own two-turn histories. Initial compact-read full-logit differences improve from 14.47/16.27/6.98% to 12.08/12.71/6.29%; all eight readouts with matching histories improve under native V in both readers. Later countertrends remain, arithmetic stays wrong, and native-QKV compact memory is still closer than the mixed compact policy. Eighteen paths/1,799 advances, 36 exact snapshots, 15 historical paths/213 arrays and four source arrays verify. This adds one experimental policy, with no serving change, compact-stack integration or whole-model forward-free claim. [Math](query_memory_transfer/MATH.md), [Astra audit](query_memory_transfer/REVIEW.md).

The [proposal](../../obsidian/blog/2026-11-05-164648-a-c-state-machine-for-executable-model-computation.md) connects the implementation to operational variables, MLP response laws, context and time circuits. [FINDINGS.md](FINDINGS.md) records measured behavior and limitations.

## Implemented boundaries

| Adapter | Persistent state | Execution |
|---|---|---|
| ISA | Registers, four flags and predicted byte offset between instructions; complete row state during an instruction | Three Mamba mixers, five MLPs, all eleven native heads |
| ISA quadratic law | Frozen anchor, branch baseline, two basis directions and five coefficient vectors | Explicit anchored intervention at final post row 8, replacing one dense branch evaluation |
| Qwen | All-layer K/V, position, pending token and whether it was delivered | Own incremental prompt prefill, full attention, SwiGLU, tied readout and greedy output |
| Qwen receiver plane | One frozen receiver/read/carry basis, six projected coefficient vectors and a native baseline write | Explicit experimental bypass of selected gate/up projections; native RMS reduction, SiLU and down projection |
| Qwen live-receiver subspace | Frozen two-column bank/QR, with one optional explicitly calibrated third operand | Separate approximate API; current receiver projection, actual skip and RMS scale; residual-correction diagnostic |
| Qwen source gate law | Frozen source gate, sigmoid coefficients and source SiLU | Explicit source-anchored nonlinear replacement on branch-owned operands; full dense projections and state retained |

The isolated [nonlinear quotient assay](state_quotient/README.md) tests a supplied compact state through16own updates using these native primitives: closed and repaired paths are byte-exact; one off-space write makes discarded orientation consequential at update2. An original-Qwen down-weight minor proves rank≥256 for one strong common-space construction, with no Qwen forward. This is a controlled state-interface result, not a compressed Qwen adapter or serving speedup. [Findings and recent-post synthesis](state_quotient/FINDINGS.md).

The current ISA continuation consumer commits registers, flags and predicted instruction length. Store, address, value and syscall heads remain visible outputs; byte-memory and OS execution are outside this consumer.

The ISA recurrence and convolution histories reset per instruction. Qwen K/V persists per session. Qwen supports a legacy saved parent and a strict stack of up to 64 nested token-boundary parents. Legacy saving replaces its previous checkpoint; legacy rewind is refused while nested frames remain. Stack rollback restores the top parent and discards its child; stack commit removes the parent marker while retaining the child state. These handles are session-local and do not serialize a portable Qwen session to disk.

Every retained KV prefix stays immutable. Children append above it, so nested markers introduce no additional KV copies. The current marker array is 9,216 host bytes including parent lineage, compared with the historical 8,704-byte array; KV still costs 56 KiB per token in this FP32 profile. Reset drops all markers, including after an unfinished token. Prefix-memory editing and durable sibling checkout require separately restorable storage and are outside this cursor-only stack.

The quadratic law is an explicitly anchored research experiment. It does not claim natural live applicability or general replacement of the MLP. Its final residual is `proposed_input + float32(frozen_branch + quadratic_response)`. Outside this selected intervention, the ordinary equations execute.

## Files

| File | Responsibility |
|---|---|
| `src/model.c` | Read-only mmap, SHA256, tensor bounds and identity checks |
| `src/isa.c` | ISA equations, row advancement, complete local checkpoints and commits |
| `src/qwen.c` | Qwen equations, layer/token clock, cache lifecycle and greedy policy |
| `src/qwen_cuda.cu` | cuBLAS, normalization, RoPE, cache writes, softmax, SwiGLU and top-two selection |
| `src/qwen_serve.c` | Persistent single-session stdin protocol |
| `tools/bundle.py` | Offline tensor/law exports with source bindings |
| `tools/chat.py` | Verified native tokenizer, exact prefix reuse, optional explicit memory text |
| `tools/worker.py` | Finite native/C experiments and retained observations |
| `tools/saturn_bridge.py` | Public Saturn receipt at a whole-process/controller boundary |
| `tools/fleet.py` | Single-lease mrun submission and exact-job result collection |

The executor has no Python/framework inference dependency. Text encoding in `chat.py` uses the checkpoint's native tokenizer outside numerical execution. The API accepts token IDs directly.

## Build and model-free checks

From this directory:

```sh
rtk proxy make
rtk proxy python3 -m unittest discover -s tests -v
rtk proxy make sanitize
rtk proxy make state-mechanics
```

On Beast, inside the execution environment:

```sh
make cuda
make gpu-mechanics
./build/gpu-mechanics
```

`make sanitize` produces a separate `build/tcm-sanitize`. GPU mechanics checks matrix orientation, grouped-query attention, RoPE, deterministic ties and checkpoint ownership. Numerical kernels use FP32; compiler contraction is disabled in our C/CUDA source, and cuBLAS uses pedantic math mode. This does not promise native Torch bit equality or prohibit internal cuBLAS fused operations.

## Export a model

From the domains workspace, using the shared environment:

```sh
rtk proxy common/bin/common exec python saturn/cmachine/tools/bundle.py isa \
  isa/runs/native-forward-symbols-20261029T090450.332745Z/extracted/stream-program-result/stream-program.pt \
  saturn/cmachine/models/isa-r3.tcm

rtk proxy common/bin/common exec python saturn/cmachine/tools/bundle.py law \
  isa/outputs/post2-quadratic-plan-r3-mamba/plan.json \
  saturn/cmachine/models/post2-quadratic.tcm
```

The Qwen worker exports the existing Beast checkpoint once into `/mnt/big/llm-models/tcm/qwen25-1.5b-fp32-execution-v2.tcm`. Original BF16 parameter bytes remain unchanged in this bundle. The backend widens them to FP32 once on GPU, retains one FP32 bank, and releases its host mapping. Qwen tokenizer/configuration hashes accompany the bundle.

The bundle contains a fixed little-endian header, configuration, tensor table and aligned payload. SHA256 covers configuration, model identity, tensor descriptors and all payload bytes. Unsupported profiles, shapes, dtypes and invalid extents are refused. These checks establish file integrity and declared shape compatibility; source provenance is retained by the exporter and run manifest.

## Run comparisons on Beast

From the domains workspace:

```sh
rtk proxy common/bin/common exec python saturn/cmachine/tools/fleet.py submit isa
rtk proxy common/bin/common exec python saturn/cmachine/tools/fleet.py collect JOB_ID

rtk proxy common/bin/common exec python saturn/cmachine/tools/fleet.py submit qwen
rtk proxy common/bin/common exec python saturn/cmachine/tools/fleet.py submit continuation
rtk proxy common/bin/common exec python saturn/cmachine/tools/fleet.py submit stack
rtk proxy common/bin/common exec python saturn/cmachine/tools/fleet.py submit plane
rtk proxy common/bin/common exec python saturn/cmachine/tools/fleet.py submit reuse
rtk proxy common/bin/common exec python saturn/cmachine/tools/fleet.py submit enrich
rtk proxy common/bin/common exec python saturn/cmachine/tools/fleet.py submit components
rtk proxy common/bin/common exec python saturn/cmachine/tools/fleet.py submit writer
rtk proxy common/bin/common exec python saturn/cmachine/tools/fleet.py submit lineage
```

Each submission is one finite lease with retries disabled and strict preflight. Qwen requests CUDA. Native and candidate models execute sequentially. The continuation assay uses a measured reservation exception from the first Qwen panel: 6,144 MB RAM and 8,192 MB VRAM versus observed peaks of 5,076.3 MB and 6,718 MB at context capacity 1,024.

ISA comparison preserves all heads, each path's own before/after state, exact same-C row-31 replay and five anchored law trials. Qwen comparison preserves original FP32 and BF16 native outputs, independent C output, prompt IDs, first-readout errors, checkpoint results and timing. Different arithmetic modes remain separate trajectories.

Timing scopes are explicit: loading, incremental C prefill, native parallel prefill and generation. Generation timing covers `generated_tokens - 1` forward passes because the first token's logits come from prefill. Allocation accounting excludes CUDA/cuBLAS overhead and the temporary BF16 widening buffer; scheduler telemetry reports measured job peaks.

## Persistent chat

An explicit [bounded chat mode](composition/bounded_chat/README.md) is now installed alongside the native build at `/mnt/big/saturn-cmachine/bounded-current`. Select `--cache-policy bounded` with that installation's client to use 16 anchors, 64 recent rows and 32 sampled rows per layer; `full-byte` selects its matched control. The completed 124-second Beast trial verifies persistent ownership, exact historical readouts, two snapshot-backed parents and unchanged refusal paths. Short prose, explicit 17+29 and recent recall worked; four delayed bounded answers remained wrong. [Actual replies and costs](composition/bounded_chat/FINDINGS.md), [Astra audit](composition/bounded_chat/REVIEW.md), [installed-file receipt](composition/bounded_chat/runs/job-f53e7d9c952f/raw/installation.json). The client default remains native.

The tested development build is installed on Beast at `/mnt/big/saturn-cmachine/current`, with the executable at `build/tcm-cuda` and client at `tools/chat.py`. The [build record](results/installed-build.json) pins the binary and source-manifest hashes. No background service is started. The source workspace remains the place to submit experiments with `tools/fleet.py`.

Inside an admitted Beast GPU lease, run the text client with the exported model and its original tokenizer directory:

```sh
python tools/chat.py \
  --model /mnt/big/llm-models/tcm/qwen25-1.5b-fp32-execution-v2.tcm \
  --tokenizer /mnt/big/llm-models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306
```

`/save` retains one legacy parent and `/rewind` restores it. `/push`, `/rollback` and `/commit` control nested continuations. Commit keeps the child's tokens readable; rollback restores the parent's transcript and KV boundary. To return only a child result, roll back and supply that result explicitly in the next input. `/quit` exits. `--prompt TEXT` runs one request. `--memory FILE` visibly inserts the file's text on the first turn; this is an explicitly assisted input, not native recall. Memory is ordinary text, so no vector database or extra model is required.

The client compares the complete new prompt's token IDs with the processed prefix. It appends only a matching suffix. If the template changes an already processed prefix, it resets and recomputes the prompt, explicitly invalidating the old parent and every nested frame. It checks capacity before changing either the client history or C state. Transport errors end the session rather than blindly repeating a possibly applied mutation.

The direct C protocol emits one JSON result per command:

| Command | Behavior |
|---|---|
| `A ID ID ...` | Append to consumed cache history and replace the unconsumed prediction |
| `G N` | Deliver up to N new tokens, continuing after any previously delivered token |
| `C` / `R` | Save / restore the latest parent |
| `P` | Push a nested frame; return its opaque ID |
| `B ID` / `K ID` | Roll back / commit the current top frame; reject stale or non-top IDs |
| `S` | Report clock, pending-token policy, stack depth, top frame and reset epoch |
| `Z` | Reset valid history without reloading weights; invalidate all parents |
| `Q` | Exit |

When a previously displayed pending token belongs in the next supplied history, an `A` caller must include it. The text client maintains this invariant automatically. Native context is bounded without silent eviction or summary replacement. The explicitly selected bounded backend evicts according to its declared policy and reports resident support through `/state`; its fixed GPU storage does not remove the logical context limit or bound the client's host transcript.

## Selected operation observations

`qwen_begin_token`, `qwen_advance_layer` and `qwen_finish_token` expose the existing token computation without changing its kernel order. Only complete token boundaries support checkpoints. The selected observer receives borrowed, read-only device pointers; its observations are not resumable interior cuts. Restore a token parent and replay to inspect a layer.

The CLI can capture fourteen ports at one layer and one token position, including the computed and consumed attention/MLP writes and raw post-attention receiver:

```sh
./build/tcm-cuda qwen MODEL --tokens-file INPUT_IDS --new-tokens 32 \
  --trace-prefix results/trace --trace-layer 16
```

Position defaults to the last prefill token; `--trace-position N` selects another zero-based position. Captures contain layer input, positioned Q/K/V, attention return, normalized MLP input, pre-SiLU gate projection, up projection, MLP return and layer exit. The JSONL descriptors bind model digest, position, layer, port, occurrence, element count and FP32 dtype. Matching `.f32` files retain packed vectors. The normal path has no observer transfers; selected captures synchronize and are not timing-neutral.

The explicit `QwenWriter` experiment copies a supplied frozen rank-three projector and binds one model/session/epoch/parent-lineage/token-position/edit-layer. `qwen_advance_layer_writer` selects ordinary, attention projection, MLP projection or both in sequence. Attention projection runs before its residual addition, so the actual MLP recomputes on the edited receiver; both projects that new MLP return. Subsequent tokens execute ordinary equations. Computed ports retain their old IDs; appended consumed-write ports distinguish the actual added operands. Ordinary tracing and serving never enable a writer policy automatically.

`QwenGateLaw` is a separate source-anchored scalar program. `qwen_gate_law_init` compiles five FP32 channel vectors from one finite source gate; `qwen_advance_layer_gate` explicitly selects native, polynomial or whole-write-zero execution and checks the invocation's retained parent. Source-law identity is reusable across clocks; each application binds its actual context. The [two-turn experiment](gate_law/FINDINGS.md) runs through own tokens and later user/header ingestion before native comparisons. All 93 evaluated law writes are closer to their same-input native write than zero, while accumulated cache error and native arithmetic mistakes remain. Native/polynomial execution both retain 197 dense matvecs per token. [Protocol, code and reproduction](gate_law/README.md) retain the numerical contract, work counts, error decomposition and tested binaries. The installed development release is unchanged.

The separate [source-derived operand experiment](operand_basis/README.md) replaces both original MLP16 G/U maps with three rank-64 derived factors from the first prompt token through two turns. It keeps the existing SiLU and dense down map. All 1,128 primary receivers/norm inputs match native same-ID replay byte for byte, and actual selected original G/U calls are zero. Source64 pooled write error is 1.008661 versus random64 0.996941 and deletion 1; paired identical-history comparisons retain the same unfavorable aggregate ordering. This is an executed cheaper operand program with mixed behavior, not a useful native-write reconstruction advantage. [Findings and work accounting](operand_basis/FINDINGS.md) retain source provenance, complete state, unchanged original weight residency, 4,980,736 factor bytes per program, approximately 1.70% nominal whole-token linear work removed and the absence of a serving timing claim. Its isolated driver does not change ordinary serving policy.

The completed [gate/up restoration factorial](operand_factorial/FINDINGS.md) diagnoses that program at twelve fixed clocks plus one retrospective digit. Both single-operand restorations improve all twelve local writes, while gate/up-only worsen four/three readouts. Both-native gives exact current writes and layer16 exits but retains20.936%pooled same-ID full-logit error. Newly written K/V differences move from17at the intervention to18on native fixed-ID futures; old cache differences remain. The56.34-second job verifies162sources/1,801raw files,65parent restores and2,000full token steps. These are paid native controls, with full weights/KV and supplied future IDs, not an autonomous repair or cheaper serving policy. [Protocol, independent audit and reproduction](operand_factorial/README.md) retain the execution boundary.

The [parent-lineage guard](PARENT_LINEAGE.md) distinguishes another history rebuilt to the same session/epoch/position. Actual retained-parent rollback restores its identity and permits replay; a different parent refuses before GPU access. The [Beast engineering result](results/job-11cdb86ad29b/report.json) verifies pretrained lifecycle mechanics and exact historical/current source logits, own IDs and checkpoint replay. This adds metadata ownership, not semantic compression or a new scientific writer sample. The installed development release remains separately pinned by its existing build record.

The [completed writer panel](FINDINGS.md#actual-writer-projections-remove-a-native-cancellation) shows that projecting both writes removes their cancellation with inherited state: outside-P norm rises13.7545→18.8595, although all short colors remain correct. The added program needs6144GPU and43008host bytes. [Protocol](WRITER_INTERVENTION_EXPERIMENT.md), [raw report](results/job-87eed6acd554/report.json) and [review](../../artifacts/cmachine-actual-writers-20261106/REVIEW.md) retain the full four-policy experiment and ordinary control. This is an explicit diagnostic API, not an automatic serving approximation or semantic projector.

## Experimental nonlinear receiver program

The [receiver-plane plan](PLANE_EXPERIMENT.md) tests a restricted executable MLP computation. `qwen_plane_init` allocates a program. At a selected partial-token layer, `qwen_advance_layer_compile` executes the native layer and compiles its three weighted raw operands into gate/up coefficients. `qwen_advance_layer_plane` then explicitly selects dense, compiled or held-write execution on a declared dose after replay from the same token parent. The program has no implicit serving policy and adds nothing to checkpoint closure.

For raw receiver `r0`, attention write `a` and incoming carry `x`, the controlled input is `r=(r0+s*a)+t*x`. With `Aj=G(gamma*rj)` and `Bj=U(gamma*rj)`, the compiled path evaluates `g=rho*((A0+s*Aa)+t*Ax)`, `u=rho*((B0+s*Ba)+t*Bx)`, then the actual `D[SiLU(g)*u]`. The existing RMS kernel exports `rho`; its reduction is retained. This identity holds in real arithmetic on the declared plane. Native floating-point multiplication/accumulation is reassociated, so zero-dose drift and downstream errors must be measured.

The frozen numerical payload is 239,616 bytes in this profile, plus four bytes of execution scratch. Compilation adds six matrix-vector calls; every compiled target skips two. All original weights remain resident. Neither this accounting nor the instrumented transition timing establishes whole-model speedup or weight compression. Exact finite live operands and session/model/epoch/position/layer ownership are checked. Read-guard and later finite failures leave a partial token with its entry residual intact/restored, requiring explicit layer recomputation or reset. Physical-work counters are not rolled back. Held-write execution emits no fabricated gate/up observations.

The model-free sanitizer assay covers non-unit RMS weights, all six doses, zero-dose native replay, actual bypass counts, held-write identity, incoming/read/session/epoch refusals, and safe ordinary retries after failed receiver materialization or compilation. The real-model panel additionally checks persistent higher-layer KV, same-history continuation and followups from each branch's own generated state. These controls establish an instrument and bounded computation behavior; reusable semantic frames and unrestricted chat accuracy remain separate research targets.

The separate `QwenSubspace` experiment binds model and layer while allowing a changed session/position. `qwen_advance_layer_subspace` explicitly selects native compilation, dense execution, frozen projection, residual correction, arithmetic reassociation, held source write, original two-column projection, carried-omission correction or read-complement correction. It never falls back or changes ordinary chat execution. Compilation freezes the incoming residual and attention write, four G/U images and an FP64 QR. Coordinates are solved in FP64 and executed as recorded FP32 values. The actual live residual determines rho and the final residual skip. Raw entry, attention write, reconstruction and omitted operand remain inspectable.

`qwen_subspace_extend` accepts one explicitly supplied calibration operand. It removes the old basis, rounds to the stored FP32 vector, computes QR from those actual bytes, and compiles two images before admitting rank three. The original B2 bank and QR remain immutable. PROJECT2 preserves their original arithmetic; PROJECT uses all admitted columns. Failed extension leaves rank two usable. This is a fixed bounded experiment, not a dynamic rank policy or automatic memory mechanism.

The [natural reuse result](FINDINGS.md#reusing-a-frozen-basis-on-natural-later-receivers) exposes 72–83% write errors despite correct short answers. [One calibrated operand](FINDINGS.md#one-calibrated-operand-transfers-partially) improves fresh write errors 74.6%→61.8% and 81.0%→60.7%, while one downstream logit error worsens. [Component correction](FINDINGS.md#carried-omission-and-read-complement-through-the-native-consumer) then reduces a fresh query's full-logit error norm 58.8% by correcting omitted carry, while read-only correction worsens it. Their executed nonlinear write interaction is 19.13% of native-write norm, versus a 4.55e-7 split/full arithmetic discrepancy. Residual correction recovers the arithmetic floor. B3 frozen device/QR storage totals 276,536 bytes; that component panel used 354,308 GPU workspace/program bytes and 43,008 host bytes, including its two split buffers. The current explicit-operand API adds one hidden-sized owned buffer, bringing GPU workspace/program arrays to 360,452 bytes. Original model weights remain resident, and no speedup or free-chat accuracy improvement is established.

Component corrections require rank three and spend two original-weight G/U calls each. The read-side complement is tied to the actual executed residual and retains its difference from independently projected attention. A pure diagnostic uses captured projections for one actual SiLU/down evaluation without advancing the session. Its measured joint write separates finite interaction from split arithmetic. The [formation protocol](CARRY_FORMATION_EXPERIMENT.md) is now completed: two appended read-only ports and a staged explicit-operand correction API preserve native ordinary execution while exposing a coupled inherited/new-write input. Both isolated corrections worsen fidelity; their joint carried correction improves it. The [next actual-writer intervention](WRITER_INTERVENTION_EXPERIMENT.md) remains proposed.

Use `tools/fleet.py submit formation` for the finite native formation panel and `tools/verify.py results/JOB_ID` for its custody/consumer audit. Experimental explicit operands require rank three and exactly hidden-sized finite device values; they are copied into owned storage before layer scratch can overwrite aliases. Bounded program/workspace arrays now cost360,452GPU bytes; formation-only assay buffers add18,432bytes. Frozen numerical payload remains276,536bytes. Consumer schema v2 retains the correct native-versus-full-carried reference error; the first producer's v1 cell and original verifier refusal remain in its explicit correction ledger.

The [consumer contrast analyzer](tools/consumer_contrasts.py) is applied automatically by offline verification to component panels. It adds49source-bound observations across local writes, full logits and key/value layers at current/common-next token clocks. Read-only correction can worsen fidelity alone while helping conditionally after carry; retain both effects and their interaction. The executed full-correction arm defines downstream contrasts. The separate split-joint write diagnostic supplies no downstream continuation or error bound. Derived records preserve original reports and require their custody verification. Reproduce the latest analysis with:

```bash
rtk proxy common/bin/common exec python saturn/cmachine/tools/verify.py \
  saturn/cmachine/results/job-00038c17cfcd
```

The [blog and holistic review](../../artifacts/cmachine-holistic-review-20261106/REVIEW.md) connect this to recent symbol-table, diffusion, operational-variable and learned-ISA evidence. The new analysis changes experimental interpretation and source-bound observations; ordinary native generation remains the same computation.

The `stack` fleet panel preserves the historical binary's output, own-state generation, raw first logits, selected-port arrays, a real-model lifecycle assay and the actual nested text client. The `reuse`, `enrich` and `components` panels preserve calibration/program bytes, all first logits, actual ports, selected current/next KV and own answers/followups. Their small raw/source archives are collected from scheduler-sealed logs. Historical-C compatibility is separate from native-framework parity, semantic operation recovery and answer correctness.

## Frozen computation through generated state

The separate [composition assay](composition/README.md) imports the authenticated formation runtime and frozen B3 program, then compares native, one-shot, recurring and corrected MLP16 execution on independent recall, arithmetic and prose questions. A subsequent three-path zero-write control tests whether successful answers distinguish the program's content from writer omission. All policies retain blue/391; compact prose finishes within48tokens while zero is truncated. Later shared-input B3 write errors remain roughly100%, and correction recovers native tokens across all three questions. The [findings and mathematics](composition/FINDINGS.md) qualify the captured layer16/KV16 dependency boundary, distinguish token histories, and retain all native/edited outputs. Full weights/KV remain; no compact sufficient state or serving speedup is established. This isolated study leaves the shared writer implementation and installed development release unchanged.

## Scope

The subsequent [whole-turn experiment](composition/whole_turn/FINDINGS.md) advances a complete correction, acknowledgment and two independent readers under native/PROJECT/zero/CORRECT policies atMLP16. All answerOK/green/red; zero preserves the same behavior. Across258same-text comparisons, lower captured ports stay exact, while B3 write errors average101.85%. Pre-answer full-logit fidelity improves over zero, then reverses after consumed EOS. This extends own-state execution through user ingestion without changing installed serving or reducing full weights/KV.

The [formation × question factorial](composition/formation_read/FINDINGS.md) then separates the correction policy from the full later question/answer policy with two new hybrids and authenticated historical endpoints. PROJECT formation with dense questions retains about 10% pre-answer full-logit difference; native formation with PROJECT questions gives 13–17%, versus 17–20% for PROJECT throughout. All short answers remain correct. Across 104 same-history transitions, local MLP16 ports and new KV16/17 rows stay exact for a fixed question policy while later logits differ. Formation and question computation therefore contribute through retained history and subsequent consumption, with substantial signed interaction. The 23.40-second Beast job retains full weights/cache; compact sufficient state and serving speedup remain open.

The subsequent [layer17 attention read](composition/attention_read/FINDINGS.md) captures sixteen native traces across those worlds and two clocks. Score interaction is exactly zero outside correction history; softmax distributes probability interaction onto unchanged source/question positions. A fixed-query numerator/normalizer update accounts for native context changes to 2.81e−7–6.45e−7 relative discrepancy using captured operands. Receiving MLP17 opposes the receiver interaction throughout, while net interaction grows before answers and shrinks after EOS. All 20 historical full-logit/cache-hash boundaries remain exact; 73 sources and 301 raw files verify. This establishes a conditional read account with full weights/KV, not a query-independent compact chat state or serving speedup.

The [prospective span-response cut](composition/span_response/FINDINGS.md) now freezes a memory-only program before two later queries exist. Eight key/value centroids per KV head occupy 16,480 serialized bytes versus 69,632 native span bytes. Substituting one layer17 read gives recall/prose first-logit differences of 1.592%/0.783%, versus deletion's 3.640%/1.733%; exact34 reproduces native bytes. All own answers match, including deletion. Isolated span numerator errors remain 47–49%; a saved-only factorization identifies substantial lost dispersion and finite score/value dependence. The 23.64-second Beast job verifies 72 sources and 402 raw files, independently audited by Astra. Dense complement, later native reads, full original cache and weights remain; this establishes local prospective approximation, not ongoing compact chat or a serving resource reduction.

The [recurring paired-byte follow-up](composition/span_recurrent/FINDINGS.md) keeps all 34 atoms in 17,984 serialized bytes and uses that representation at every later layer17 read through three linked turns. Original C17 K/V remains poisoned and native fallback is refused. Mean numerator error is 2.086% versus centroids' 48.438%; all six measured byte logit differences are below 0.1%. All answers also match deletion. All Q17 queries remain native on matching histories, so changed-query feedback is not exercised. The 23.78-second Beast run binds 76 sources and 1,107 raw files; an additional zero-model CPU replay exactly reproduces frozen analysis bytes after preserving a local FP32 log-count portability refusal. Newly written memory, other layers, full allocation/weights and dense work remain. [C runtime findings/requests ledger](requests.jsonl) records the architectural finding and next own-state append contract with blog backlinks.

The [owned compressed-append experiment](composition/compact_append/FINDINGS.md) now removes selected dense cache slots, encodes each actual new K/V row before the current read, and rewinds encoded cursors with conversation state. Native/exact/byte17/byte17+18 advance three linked turns;267main byte-layer appends verify. Partial abort restores[94,93]to root93after17already published and consumed. All89Q18arrays change from native, mean0.241851%, while byte17andbyte17+18have identical upstream queries. Stacked allocation saves1,032,188bytes including decode/status,3.43%of state; six logit differences are0.370–1.363%, worse than byte17at every boundary despite identical answers. The32.17-second Beast job verifies81sources/2,978raw files; Astra independently audits snapshots, controls and allocation. Full weights/26dense cache layers and linear context growth remain. [Math and protocol](composition/compact_append/README.md).

The C equation machine is independently executable model computation. The bounded ISA substitution is a recovered response law. Qwen's baseline executes full equations, with isolated selected-operation and cache approximations documented above; a compact semantic algebra for general conversation remains research. The nested stack provides external continuation control, rather than establishing learned recursion or semantic scope isolation. The existing durable Saturn SourceFrameStack remains a separate source/custody system; these C cursor handles are not durable StateCuts. No speedup from recovered Qwen laws, arbitrary architecture discovery, training, production service or whole-model compact sufficient context is claimed.

Exploration retains errors, native mistakes and divergent trajectories. Same-history comparisons localize numerical differences; independently generated histories measure behavior. Returning to a dense equation from an approximate state does not repair earlier drift.

The [all-layer experiment](composition/compact_all/FINDINGS.md) now constructs all28 byte caches from token zero, removing every dense cache slot. Native/exact/byte sessions advance 546 main tokens across five turns each, with exact controls and last-layer abort closure. State allocation falls 70.66% to 8,830,468 bytes; answers match while logit differences remain 3.732–9.839% and early routing errors occur. The original post-processing failure is preserved; a narrowly authenticated saved-only field-name repair verifies all evidence without another model run.

The separate [weight-storage experiment](composition/weight_tiles/FINDINGS.md) retains original BF16 matrices and uses one shared FP32 decode tile. All 258 tiled FP32/BF16 trajectory artifacts match exactly; original-versus-tiled arithmetic changes later logits 1.334–4.147%. Tracked model-plus-state allocation is 3,133,265,432 bytes, 49.50% below the original dense-cache baseline. Both jobs have independent Astra audits and offline evidence sweeps. Full learned parameters and forward work remain; that experiment retains linear memory. The [subsequent retention contract](composition/compact_all/RETENTION.md) counts recovery storage as well as active rows; its fixed-snapshot option now executes below.

The [bounded-retention implementation](composition/bounded_retention/FINDINGS.md) now runs anchors16/recent64/reservoir32 beyond physical support with fixed score/decode scratch and three counted recovery snapshots. Its 9,279,620-byte tracked GPU state restores occupied sample replacement, RNG, addresses and logits exactly. Five own-state arms advance 2,626 main tokens; every bounded arm fails all four delayed questions, while native/full-byte recover the three facts but give40 instead of46 for the sum. Saved-only factoring separates omission, weighting, changed K/V and changed queries; Astra independently reproduces every signed term. Verification binds73 sources/1,060 raw files, and74 evidence rows verify. Bounded storage is implemented; sufficient chat and forward elimination remain open. [Protocol and review](composition/bounded_retention/README.md), [architectural explanation](../../obsidian/blog/2026-11-06-171900-what-bounded-memory-changes-in-the-c-state-machine.md).

## Evidence verification

`tools/verify.py RESULT_DIRECTORY` checks the report against the scheduler-sealed log, verifies copied source hashes and public receipt seals, and recomputes Qwen first-readout errors from retained raw arrays. It performs no model execution. Each successful main assay retains its raw outputs and source snapshot under its result directory.

The standard Saturn evidence sweep and verification are recorded in `index/evidence-run.json` and `index/evidence-verify.json`. Its current coverage is the derived observation streams; it does not automatically interpret these custom reports as terminal claims or index the internal C ports. The two public invocation receipts are verified directly by `tools/verify.py`.

For the nested-stack panel, use the hash-bound `index/stack-20261105/target-streams` projection and its qualified run/verify records. The frozen payload also contains historical package fixtures; the broad sweep's four accepted fixture receipts are explicitly excluded from this job's results. [Scope correction](index/stack-20261105/projection-incident.json) preserves both the original sweep and the corrected interpretation.
