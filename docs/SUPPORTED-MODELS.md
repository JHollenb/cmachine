# Supported models and numerical profiles

The following model profiles and math components have completed numerical execution in cmachine. Each result is scoped to its checkpoint, tested inputs and executed source manifest. A supported architecture grammar can accept compatible new checkpoints, but those checkpoints require their own execution evidence. FieldLM's host reasoning boundary is explicit below.

| Tested profile | Import and run surface | Numerical backend and state | Recorded evidence |
| --- | --- | --- | --- |
| R3 direct-effect ISA Mamba | `tools/bundle.py isa` and `tcm isa`; the nominated width-48, three-mixer ISA specimen. | CPU sequential FP32 C/libm; convolution/scan history resets per instruction, while register/flag/instruction-length commits continue. | [Initial comparison](../research/FINDINGS.md#complete-isa-execution-and-a-substituted-mlp-law), `job-709fe816a540`: all 16 architectural commits match native; maximum same-input head-value error `6.09996e-5`; exact C row replay. |
| Qwen2.5-1.5B-Instruct | `tools/bundle.py qwen`, `build/tcm-cuda serve` and the [chat client](../tools/chat.py); fixed 28-layer, width-1,536 profile. | FP32 C/CUDA/cuBLAS on the qualified `sm_89` host; persistent all-layer K/V, pending-token ownership and nested continuations. | [Initial independent port](../research/FINDINGS.md#independent-qwen-prefill-and-generation), `job-52f2231facaf`: three source/C own-token trajectories agree against native FP32; maximum first-readout logit error `2.6727e-4`. [Integration](evidence/integration-7a6824d/README.md) and [public demos](../demos/results/pretrained/README.md) separately qualify C lifecycle and source-covered operand replacements. |
| PAN P113 seed 0 | Trusted source constructor and checkpoint through `lower_fx(...)`, then `build/tcm program`; two-input modular-addition checkpoint with 1,473 parameters. | CPU sequential FP32 C/libm; 19 instructions, 20 symbols, no recurrent slots. | [Report](../research/model_intake/evidence/pan/report.json), `job-0143a2072cce`: all 12,769 source/C argmax predictions agree; both answer 7,382 pairs correctly (57.8119%). |
| MambaSan-130m-instruct | Pinned, normalized source through `tools/program.py`, then `build/tcm-blas program` or the [one-request chat client](../tools/program_chat.py); Japanese instruction model, 24 layers, width 768, 32,000 lexical rows. | CPU FP32 C/libm/OpenBLAS; 363 instructions, 364 symbols and 48 recurrent slots holding 737,280 floats. | [Report](../research/model_intake/evidence/mamba/report.json), `job-1b568233159e`: 18 matching generated tokens across two prompts, 86 comparable rows with matching argmax, and exact C suffix replay for both baseline paths and the zero-writer arm. |

## Math components and FieldLM consumers

The [math intake report](../research/model_intake/MATH.md) and [verified receipt](../research/model_intake/evidence/math-20261106/README.md) qualify `job-2a7aae82cec8` on the shared CPU sequential FP32 executor:

| Tested component | Import/run surface | Observed result and limit |
| --- | --- | --- |
| Exact learned matrix2, seeds 811/1597 | Original `ProductMLP`, trusted FX, `tcm program`; six instructions and 84 coefficients each | All 64 formal ordered coefficients match matrix multiplication; zero source/C and truth error on 272 bounded input pairs per cell. Three own-output steps are retained. This is not unrestricted FP32/uint64 correctness. |
| Approximate matrix2, seeds 137/149/163 | Original FP64 exports explicitly converted to FP32; same capture/executor | Maximum source FP32/C errors `4.76837e-6`, `9.05991e-6`, `8.10623e-6`. All 64 coefficient residuals remain nonzero in each original and deployed audit; no integer rounding or exact-algorithm claim. |
| Concept Graph V1/V2a | Frozen 21→32→1 / 23→32→1 networks; mean/std normalization and ReLU; eight C instructions each | 45 live frontiers/987 rows; maximum score error `4.57763671875e-5`. One within-selected-set order difference, no selected-set differences. V2a uses named V1 base-library features, without its acquired divide/expand library. |
| FieldLM concept/law reads | Generic C projected-record filter through native FieldFabric/index consumers | All 16,241 reads match native. Two acquired concepts, three supplied primitives and 13 laws reach before/after C numerical consumers. Parsing, frame extension, rules, graph mutation, features and extraction remain on the host. This is not a complete standalone C FieldLM runtime. |
| Concept Graph choice replay | Saturn public adapter, sealed prefix reconstruction and typed choice edits; original search/extractor and emitted C consumer | [117-branch causal panel](../research/model_intake/MATH-CAUSAL.md): a second-round substitution repairs scalar DAG cost `9→5` and emitted C instructions `15→11`, preserving the function and tested outputs. Later swaps can hurt; no retraining/general reasoning claim. Same-process reconstruction, with host reasoning semantics retained. |

The ordered matrix expression goes from four learned-cell calls to one, `23→10` emitted instructions and `336→84` nominal linear MACs, with identical tested outputs. Every scheduler reaches that reduction. On one scalar goal V1 beats matched random/enumeration but ties the heuristic; V2a is weaker there. General mathematical improvement and speedup remain unestablished. These numeric outputs and scores are distinct from vocabulary logits.

## Measured logit error

These are absolute differences between the independent native FP32 reference and C FP32 logits. They are measured observations, not configured tolerances or percentages. The complete raw arrays and source identities remain bound by the [intake receipts](../research/model_intake/evidence/README.md).

| Checkpoint and panel | Comparable input rows | Maximum absolute logit error | Mean absolute logit error | Generated-token agreement |
| --- | ---: | ---: | ---: | --- |
| PAN P113, all input pairs | 12,769 | `9.1552734375e-5` | `4.77931e-6` | Stateless classifier; 12,769/12,769 argmax agreement. |
| MambaSan, Japanese capital question | 39 | `1.430511474609375e-5` | `1.80386e-6` | 6/6 selected tokens, including EOS. |
| MambaSan, Japanese 7+5 question | 47 | `1.3828277587890625e-5` | `1.92780e-6` | 12/12 selected tokens within the declared budget. |

Mamba comparisons include prompt ingestion and each model's own consumed continuation. Logits are compared only while the input histories agree; after a divergence, distinct trajectories remain distinct evidence. The earlier Qwen maximum is for its three initial readouts, not every later generated step. The ISA measurement covers head values rather than a language-model vocabulary readout.

Exact replay is within one C executable/profile. It does not mean C and the native framework are bit-identical. C source disables contraction; OpenBLAS and cuBLAS retain their own numerical implementations. Source BF16 coefficients widen to FP32 for the new graph path, and these comparisons do not qualify native BF16 execution.

## Import a supported graph

The [intake guide](../research/model_intake/README.md) documents the shared 17-op dispatcher, exact source bindings, symbol/consumer tables, state allocation and intervention hooks. Its two front ends have different responsibilities:

- **Canonical Mamba1:** the existing mrun grammar supplies source, weight, state and I/O declarations. `tools/program.py` binds those operations to C. The tested MambaSan source normalization pins tensor/configuration/tokenizer identities and preserves its tied readout.
- **Trusted FX:** the caller supplies a reviewed constructor and checkpoint loader. Capture admits a bounded single-row FP32 vector graph. Unsupported operations or shapes refuse. PAN, the matrix cells and the rewrite scorers have completed trained-artifact panels for this route.

From a cmachine checkout in the domains workspace, compile an already normalized MambaSan source and prepare the C executable:

```sh
rtk proxy make program-blas
rtk proxy ../common/bin/common exec python tools/program.py \
  /path/to/normalized-mambasan-source /path/to/new/model.tcm \
  --retain-ir /path/to/new/source-ir.json
```

`make program-blas` needs CBLAS headers and a native BLAS library. `BLAS_CPPFLAGS` and `BLAS_LDLIBS` select the installation; the fleet worker pins and records its OpenBLAS build. The standalone C executor uses C/libm/native BLAS for inference. Python dependencies belong to offline import, native tokenization and research comparison. The configured shared workspace environment supplies mrun, manalysis, PyTorch, safetensors and Transformers.

For real-model work in that workspace, use the finite admitted fleet route:

```sh
rtk proxy ../common/bin/common exec python tools/intake_fleet.py submit pan
rtk proxy ../common/bin/common exec python tools/intake_fleet.py submit mamba-chat
rtk proxy ../common/bin/common exec python tools/intake_fleet.py collect JOB_ID
rtk proxy ../common/bin/common exec python tools/verify_intake.py results/JOB_ID
```

These commands submit jobs; they do not run model inference on the submitting machine. Outside the domains workspace, arrange the import dependencies and model assets explicitly. Weights are not distributed in this repository.

## Boundaries of support

The MambaSan client uses the native Japanese tokenizer and chat template outside C. Its source-map binding checks every non-weight asset, including `emoji.json`, before and after encoding. It starts a new C process/prefix per request. Qwen's persistent multi-turn chat and nested checkpoint protocol are a separate surface; the graph executor provides bounded suffix replay and reports consumed rows and the pending token separately from recurrent floats.

The retained English `havenhq/mamba-chat` acquisition profile has no completed numerical run here. Its cancelled downloads are preserved as incomplete acquisition evidence, and it is not listed as a tested supported model. Cached base Mamba weights are not evidence of instruction/chat behavior. Other model families accepted by an upstream structural importer are not automatically accepted by this C target. Qwen3, FLUX/VAE, smaller-Qwen and the separate three-product bilinear specimen remain [external computation accounts](CAPABILITIES.md#computations-understood-outside-this-c-runtime), rather than additional qualified C backends.

PAN remains partially correct, and MambaSan answers both tested questions incorrectly. All learned dense maps remain in the new graph path. Structural import, numerical fidelity, semantic discovery, capability improvement and speed are separate claims. [Current evidence](STATUS.md) and [capabilities](CAPABILITIES.md) retain their individual limits.
