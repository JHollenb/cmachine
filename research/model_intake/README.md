# Automated model intake and computation discovery

This directory contains the structural intake tool, a working graph-to-C path, and the [architecture plan](PLAN.md) for connecting model execution to computation discovery.

The repository's [supported-model inventory](../../docs/SUPPORTED-MODELS.md) lists dedicated and compiled graph profiles, measured maximum/mean logit errors and their execution boundaries.

The [math extension](MATH.md) adds five learned matrix cells, two neural rewrite scorers, live projected FieldLM reads and binary-bound polynomial analysis. Its [finite panel](evidence/math-20261106/README.md) retains 127 compiled programs and independently verifies numerical errors, query results, score rankings, work counts and C-fed continuation chains. Host reasoning semantics remain explicit.

`intake.py` imports local declarative checkpoint sources through `mrun.decompiler`, validates coordinated IR, and uses `manalysis.decompiler_bridge` to produce an exact graph with stable source-bound references. Mathematical role priors remain a separate hypothesis-only artifact. That structural command performs no numerical execution. The new `tools/program.py` compiler and `tcm program` executor implement the next boundary; existing ISA and Qwen protocols retain their behavior.

## Use the prototype

From the domains workspace:

```sh
rtk proxy common/bin/common exec python cmachine/research/model_intake/intake.py \
  --source /path/to/local/checkpoint \
  --output /path/to/new/intake-directory
```

To inspect a previously emitted canonical IR bundle or full decompiler report:

```sh
rtk proxy common/bin/common exec python cmachine/research/model_intake/intake.py \
  --ir /path/to/ir.json \
  --output /path/to/new/intake-directory
```

`--source-id` and `--revision` optionally bind the supplied source identity. Only already registered mrun architecture adapters are used. This command does not authorize custom model code or load adapter plugins. Sources outside the supported declarative grammar retain the canonical refusal instead of receiving a guessed executable graph.

Each new output directory contains `input.json`, `intake.json`, a readable `REPORT.md`, and a content-addressed workspace with the canonical bundle, exact graph and role priors when decoding succeeds. `intake.json` is published atomically last as the complete-intake marker. Existing output directories are refused. Output cannot be placed inside a frozen source directory.

The `--ir` path retains normalized coordinated IR only, so its source-validation scope is deliberately weaker than a fresh `--source` import, even when the incoming JSON was a full source report. The report never infers a qualified C backend from an architecture name or operation inventory. Its `c_execution_qualified` and `semantic_claim` fields remain false.

## Mechanics checks

```sh
rtk proxy common/bin/common exec python -m unittest discover \
  -s cmachine/research/model_intake -p 'test_intake.py' -v
```

The nine tests use tiny synthetic Qwen2/Qwen3 source fixtures with inert parameter bytes. They check exact graph references and hypothesis separation, raw-IR provenance, changed-parameter identity, unknown-family refusal, tampered IR, output/source protection, retained mutation refusals, late-mutation rejection and atomic completion publication. They do not exercise trained models or numerical serving.

All nine passed in the configured shared environment on November 6, 2026. A separate CLI smoke returned zero for a supported synthetic source and two for an unknown family, preserving its refusal report. The [mechanics record](evidence/mechanics.json) binds the test result to the prototype/test source hashes. It is a local test record, not a real-model execution receipt.

## Compile and execute a graph

From a cmachine checkout, using an already normalized MambaSan source (EOS ID 31999):

```sh
rtk proxy make
rtk proxy ../common/bin/common exec python tools/program.py \
  /path/to/normalized-mambasan-source /path/to/new/model.tcm \
  --retain-ir /path/to/new/source-ir.json
rtk proxy ./build/tcm program /path/to/new/model.tcm \
  --input tokens.f32 --output logits.f32 --generate 12 --eos 31999 --replay
```

The input file contains little-endian FP32 rows. A token-model row is one exactly representable integer ID. The C executable verifies the bundle digest, instruction table, vector shapes, def-use order, parameter bindings, state extents and token domain. C consumes the prompt, selects its own greedy outputs, advances its own recurrence, and leaves the final delivered token pending. `--replay` restores the prompt-boundary recurrent state and logits and compares the continuation exactly within that executable/profile. `--state-file FILE` exports recurrent floats; the JSON receipt separately reports consumed rows and pending token.

Two front ends use the same 17-op C dispatcher:

| Front end | Automated work | Explicit boundary |
| --- | --- | --- |
| `lower_fx(module, input_width, identity, destination)` | Trace trusted source, check a single-row shape probe, lower primitive operations, export coefficients and exact FX-node bindings. | Caller supplies the trusted constructor/checkpoint loader. No automatic import or execution of repository Python. Unsupported operators and shapes refuse. |
| `tools/program.py SOURCE DESTINATION` | Canonical source/weight/state/IO decompilation, Mamba1 operation lowering, physical-allocation sharing, source-to-C references and state-slot allocation. | Existing mrun Mamba1 adapter supplies the architecture grammar. Other IR families currently refuse. |

The primitive set includes linear, add, multiply, positive remainder, cosine, affine, concat, slice, RMSNorm, embedding, causal depthwise convolution with SiLU, selective scan, SiLU, constant, ReLU, vector division and projected record matching. Convolution/scan are explicit reusable stateful operators. C does not dispatch an entire forward pass by model name. Source coefficients convert to FP32 offline with their original dtype retained; FP64 narrowing is explicit for approximate math cells. Tied parameter views share one exported bank. The schedule and vector descriptors live in the digest-covered bundle. The adjacent `.map.json` joins each instruction and state slot to its source, and includes the source's declared IO/state contracts.

`--trace FILE` records the complete SSA arena after each input step. `--zero OP` zeros a selected instruction output before downstream consumers. These are explicit research controls; they do not run automatically during serving. The source map makes an observation or intervention addressable, but a source name or operation pattern does not establish a mathematical role.

Generate the symbol/consumer table, optionally joining one trace row:

```sh
rtk proxy ../common/bin/common exec python tools/inspect_program.py model.map.json \
  --output symbols.json --trace trace.f32 --row 0 --job-id JOB_ID
rtk proxy ../common/bin/common exec python tools/verify_intake.py results/JOB_ID
```

The symbol table distinguishes the source program from a run/row/instruction occurrence. The verifier checks artifact hashes, executed-source bindings and source-map seals, then recomputes raw numerical/task observations without an accuracy threshold. The latest PAN payload is also retained locally for reproduction. Recorded means can differ slightly when recomputed on a different NumPy reduction implementation; the raw arrays and exact hashes remain authoritative.

The scalar build uses ordered FP32 C reductions with contraction disabled and host libm. `make program-blas` uses CBLAS for linear reductions, with `BLAS_CPPFLAGS` and `BLAS_LDLIBS` overrides. The fleet harness pins the build wheel `scipy-openblas32==0.3.34.481.3`, records its configuration and links its native library. C's compiler flags do not constrain OpenBLAS's own assembly/FMA choices. Cross-profile numerical differences remain observations.

For a decoded one-request chat client:

```sh
rtk proxy ../common/bin/common exec python tools/program_chat.py \
  --binary build/tcm-blas --model /path/to/model.tcm \
  --tokenizer /path/to/normalized-mambasan-source \
  --prompt '7足す5はいくつですか？' --new-tokens 12
```

Tokenization occurs outside C, using the checkpoint's native chat template. The client checks the model/map join and every bound non-weight source asset, including auxiliary tokenizer files, before and after encoding. Keep the source metadata alongside the tokenizer assets; weights may be omitted from that directory. `--zephyr-roles` explicitly selects the documented HavenHQ format when that tokenizer has no template. This client starts a fresh C process/prefix per request; it is not the persistent Qwen chat protocol. Model-sized scientific runs belong on mrun, including use of this client.

## Fleet specimens and evidence

```sh
rtk proxy ../common/bin/common exec python tools/intake_fleet.py submit pan
rtk proxy ../common/bin/common exec python tools/intake_fleet.py submit mamba-chat
rtk proxy ../common/bin/common exec python tools/intake_fleet.py collect JOB_ID
```

Each launch is a finite guarded lease with strict preflight, no nested submission, no training and no automatic job retry. The PAN source is the existing P113 seed-0 checkpoint. The default chat source is [`loiccabannes/MambaSan-130m-instruct`](https://huggingface.co/loiccabannes/MambaSan-130m-instruct), pinned to safetensors revision `b2eb0b6f88b5904a6166afc69e9bd7180920c63b` and SHA-256 `a33f55c181141bade9280388354751830a35204fcef0d3b3a9d7c2a05ad1e007`. Its author documents Japanese JaQuAD instruction tuning; this is a Japanese QA specimen. The original Mamba source has 24 layers, width 768, 32,000 lexical rows and a metadata-proven tied readout.

Audited normalization copies tensor payload bytes unchanged, renames the original singular embedding key, preserves the tied alias and emits a canonical HF configuration. Offline export then widens BF16 coefficients to FP32. The built-in `GPTNeoXJapaneseTokenizer` runs with local assets and no remote code, using the checkpoint's `### User:` / `### Assistant:` template. The receipt binds every asset, including `emoji.json`; canonical IOIR currently inventories that auxiliary file as metadata rather than as a tokenizer asset. This is native Python tokenization plus recorded token IDs, not a self-contained C tokenizer or a fully qualified IOIR codec.

The larger English `havenhq/mamba-chat` profile is retained with its pinned revision, tied alias and separate semantic/physical vocabulary sizes. Its first Xet acquisition stalled, and the HTTP attempt's measured transfer rate could not fit the declared budget. Both were cancelled with their partial source artifacts preserved. Neither produced a model forward or a scientific result. The smaller documented instruction model was selected to complete the first chat execution experiment.

The worker first compares independent source execution, releases that weight bank, then compiles/runs C. It retains numerical differences only for common token histories, divergent own-token trajectories separately, per-slot state observations, exact C suffix replay, and a mapped zero-writer intervention. Completed baselines are published before optional interventions; an arm error is retained separately and cannot erase its baseline. `results/JOB_ID` contains raw arrays, reports, staged source payloads and scheduler records. Historical failures remain failures. Compact tracked records live under [evidence](evidence/).

The first PAN run, `job-35ab966d9a16`, lowered 19 instructions and matched source argmax on all 12,769 input pairs. The strengthened importer reproduced it in `job-8f28ea1fa041`, with the same bundle digest and observations. Both engines answered 7,382 pairs correctly (57.8119%); maximum logit error was `9.15527e-5`. Zeroing the gate output before its decoder left 113 correct pairs (0.884956%). This is evidence of execution fidelity and a consumed dependency for this checkpoint. It does not establish a fully learned addition algorithm or an improvement in model capability.

The final PAN run `job-0143a2072cce` repeats those observations with baseline preservation enabled. The final chat run `job-1b568233159e` follows the initial successful `job-ff786f975581`, adds complete tokenizer-asset binding and baseline preservation, and produces the same program digest and numerical results:

| Specimen | Compiled instructions / symbols / state slots | Observed execution fidelity |
| --- | --- | --- |
| PAN P113 seed 0 | 19 / 20 / 0 | 12,769/12,769 source/C argmax agreement; maximum logit error `9.15527e-5`. |
| MambaSan-130m-instruct | 363 / 364 / 48 | Same 6-token and 12-token own continuations; 86 comparable input rows all agree on argmax; maximum logit error `1.43051e-5`. |

The Mamba program owns 737,280 recurrent floats (2,949,120 bytes). All three C continuations, including the intervention, replay exactly from their prompt boundaries. Native/C final-state comparisons cover all 48 slots when their consumed histories agree; maximum convolution-state difference is `6.19888e-6`, and maximum scan-state difference is `4.17233e-7`. Raw state arrays retain the first and last layers; other slot metrics are worker observations. Cached versus full-prefix native readout differs by at most `8.58307e-6`.

These prompts expose weak capability: the question asking Japan's capital produces `1990年`, and the 7+5 question produces `5足の指の指の指の指の指` in both source and C. Zeroing the mapped first-layer mixer writer changes its own continuation and changes prefill logits by up to `16.1873`. That establishes a consumed numerical dependency for this prompt; it does not identify a mathematical concept or improve either answer. [The evidence guide](evidence/README.md) links exact receipts, source maps, symbol tables, verification outputs and preserved acquisition failures.

## What remains manual

An unsupported family still needs a source grammar or trusted capture wrapper; a new primitive needs a C implementation and numerical/state checks. This implementation does not infer architectures from opaque weights, infer semantics from names, compress recurrent state, implement the full canonical StateIR transaction/lease/epoch API, or establish general math/chat improvement. The [math panel](MATH.md) analyzes admitted bilinear graphs against supplied specifications and reduces emitted programs using retrieved rules; arbitrary symbol-role discovery remains open. The C replay state covers recurrent floats and readout, while consumed-length/pending-token ownership is reported separately. See [the plan](PLAN.md).

The nine model-free program tests cover primitive parity, source-qualified FX identities, late source mutation, canonical tiny Mamba continuation, invalid shapes, semantic vocabulary padding, unsupported-operation refusal, failed-arm preservation and tokenizer/model mismatches. The seven additional math mechanics tests exercise generated circuits/scorers, polynomial refusal, resealed-sidecar attacks, scoped/mutable projected reads, invalid keys, nonfinite/division failure and vector-ABI guards. Together with nine structural intake tests and 18 runtime/client regressions, 43 tests pass. Math mechanics also pass under ASan/UBSan. Historical mechanics records retain the sources and scope actually tested.
