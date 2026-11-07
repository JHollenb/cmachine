# Model intake evidence

The [math expansion](math-20261106/README.md), `job-2a7aae82cec8`, adds learned matrix cells, neural rewrite scorers and actual projected FieldLM read consumers. Its binary-bound audit verifies 127 programs, 4,359 C invocations and 16,241 native-equal reads. The [findings](../MATH.md) separate formal polynomial identity, numerical error, computation reduction and learned scheduler quality. The PAN/Mamba receipts below retain their original sources and scope.

Both specimens execute through the shared C program dispatcher. PAN uses a trusted FX capture; Mamba uses the existing canonical Mamba1 grammar. Import produces the operation schedule, parameter bindings, state allocation and source map. The symbol table is derived automatically from that map. Numerical execution does not require semantic role labels.

| Final run | Source and result | Retained records |
| --- | --- | --- |
| `job-0143a2072cce` | Existing P113 seed-0 PAN checkpoint; all 12,769 pairs; 100% source/C argmax agreement; both 57.8119% correct. | [Report](pan/report.json), [scheduler and executed payload](pan/scheduler.json), [source map](pan/program.map.json), [symbols](pan/symbols.json), [verification](pan/verification.json). |
| `job-1b568233159e` | Pinned MambaSan-130m-instruct; 363 instructions, 48 slots; 18 matching generated tokens, 86 comparable rows; maximum logit error `1.43051e-5`. | [Report](mamba/report.json), [scheduler and executed payload](mamba/scheduler.json), [source map](mamba/program.map.json), [symbols](mamba/symbols.json), [verification](mamba/verification.json). |

Each final run used one finite strict-preflight mrun lease on Beast CPU, without CUDA, training, automatic job retries or nested submissions. Native FP32 execution provides the independent reference. The worker releases that model bank before compiling and running C. C selects and consumes its own generated tokens. It uses the original learned coefficients, widened offline to FP32, and performs the dense computations explicitly. No framework forward runs inside the C executable.

The Japanese instruction model is documented by its [author's model card](https://huggingface.co/loiccabannes/MambaSan-130m-instruct). The acquisition receipt pins revision `b2eb0b6f88b5904a6166afc69e9bd7180920c63b`, source safetensors SHA-256 `a33f55c181141bade9280388354751830a35204fcef0d3b3a9d7c2a05ad1e007`, configuration and tokenizer assets. Header normalization preserves tensor payload bytes and the tied embedding/readout allocation. The fleet reference and reusable client use the native Japanese tokenizer and source chat template. The complete asset binding includes `emoji.json`, although canonical IOIR currently classifies it as generic metadata.

PAN zero-gate intervention reduces correct predictions from 7,382 to 113. The Mamba first-layer writer intervention changes the response and prefill logits. These are bounded dependency observations. The task labels, source constructors, architecture grammar, prompt choices and intervention selection are supplied; meaningful mathematical roles have not been autonomously discovered. The Mamba responses are incorrect on both questions. No capability improvement, speedup, native BF16 parity, compact sufficient state or full StateIR transaction contract is claimed.

## Verification and custody

The tracked files are compact reports, exact source/payload hashes, maps, symbol tables and verification outputs. Raw numerical arrays, inputs, complete canonical IR and our staged source payloads remain in the ignored local directories `results/job-0143a2072cce` and `results/job-1b568233159e`. They are not bundled into this compact tracked folder. Compiled model bundles remain in the corresponding fleet work directories and can be rebuilt from the pinned sources. This folder alone is not a complete numerical reproduction archive.

From the worktree, verify collected raw evidence without loading or running a model:

```sh
rtk proxy ../common/bin/common exec python tools/verify_intake.py results/job-0143a2072cce
rtk proxy ../common/bin/common exec python tools/verify_intake.py results/job-1b568233159e
```

The verifier checks each declared raw file's size/hash, executed C sources against the scheduler's payload manifest, retained staged sources and the source-map fingerprint. It recomputes raw task/logit observations without an accuracy threshold. Current worktree sources also matched all 25 locally present staged files for each final run when these records were curated. The historical [structural mechanics record](mechanics.json) is preserved separately from the [new program mechanics](program-mechanics.json).

Mamba final-state metrics cover every slot when the consumed native/C histories agree. Raw state arrays retain the first and last layers; intermediate-layer metrics are recorded worker observations. Replay checks recurrent floats, final logits and selected tokens within C. Consumed-length/pending-token ownership is reported separately. Compiler flags disable C contraction; OpenBLAS retains its own reduction/FMA implementation. Cross-profile numerical differences remain visible.

## Preserved history and advisor review

[History](history.json) retains all collected attempts, including failed builds, an overlay-cache filesystem error, inherited offline acquisition, a stalled Xet download and an HTTP download outside the declared budget. The two cancelled HavenHQ acquisitions never reached a model forward. They are incomplete acquisition evidence and make no claim about that model's behavior. The metadata-only diagnostic is also distinct from a numerical run. Earlier successful PAN and Mamba runs remain retained locally with their original reports and hashes.

Astra reviewed source provenance, vocabulary/tied allocation, Japanese tokenizer/template, state/continuation semantics and evidence claims. Its final review found two material issues: optional intervention failure could obscure a completed baseline summary, and the reusable client's tokenizer needed an explicit model binding. Both are fixed. Model-free fault injection confirms baseline preservation; tokenizer checks reject auxiliary-file changes, extra unbound assets and model swaps. The final fleet runs exercise the revised worker and asset binding.

The next research step is automatic selection of discriminating experiments and consumer-tested computation hypotheses. The current implementation removes repeated source mapping and supported numerical bring-up work. It supplies the executable and evidence foundation for that investigation.
