# Math models through automated C intake

Frozen learned matrix cells and Concept Graph scorers now execute through the shared C dispatcher. Actual C scores and projected FieldLM record selections feed the original host consumers; their resulting arithmetic expressions compile back to C. The final guarded Beast CPU panel, `job-2a7aae82cec8`, completed all 18 search arms, retained 127 executable programs, and verified 4,359 C invocations. [Evidence and exact source identities](evidence/math-20261106/README.md).

This establishes bounded execution and useful computation reduction. It does not establish general mathematical reasoning improvement. The strongest learned scheduler ties the heuristic on the informative scalar goal, and every scheduler reaches the matrix reduction.

The subsequent [causal debugger](MATH-CAUSAL.md) localizes a useful repair to the second scalar rewrite round: one choice replacement can reduce final cost `9→5` through the unchanged V2a suffix, while preserving the exact function and C outputs. Its controls and later harmful swaps keep that conclusion local to the tested search context.

## Components and boundaries

| Component | Imported artifact | Automated C work | Supplied host computation |
| --- | --- | --- | --- |
| Learned 2×2 multiplication | October 17 frozen integer-factor rules, seeds 811/1597; 84 coefficients and seven bilinear products each | Trusted FX captures slices, parameter transposes, three linear maps and elementwise products; six C instructions | Original reviewed constructor; ordered row-major input ports |
| Approximate 2×2 multiplication | October 30 FP64 JSON exports, seeds 137/149/163 | Same capture, explicit FP32 deployment, separate original/deployed coefficient audits | Original coefficient precision and mathematical specification |
| Concept Graph V1/V2a | Exact archived frozen scorers; 21→32→1 and 23→32→1 | Eight instructions each: mean subtraction, vector division, linear/ReLU/linear; live candidate batches | Feature definitions/order, candidate generation and stable descending TopB selection |
| Learned concept memory | Frozen FieldLM store: two acquired rewrites, three supplied primitives, including a compound with a parent dependency | Content-bound string coordinates and relation/namespace/scope/role filters; C-selected records reach native two-read id→address→body consumers | Native parser/charts, FieldFabric frames, body verification, concept expansion and rewriting |
| Dimensional law library | Frozen 13-rule scalar/ordered-matrix library | Catalog and AST projected reads execute in C; retrieved bindings produce emitted C expressions | Domain semantics, repeated-placeholder frame joins, graph mutation and rule application |

The native library files and frozen exports come from the existing `model-experiments` studies. The fleet launcher stages reviewed source files and pins every archived object by SHA-256; it refuses a missing or changed artifact. It does not retrain, round an approximate rule into an integer algorithm, or invent a replacement checkpoint.

The historical learned scorer studies used `DictIndex`. This panel explicitly connects their frozen scorers to the native FieldLM index and its C read bridge. V2a runs on named V1 base-library features, with its additional rule flags zero; its separately acquired divide/expand library is not integrated here.

## What the analyzer understands

`tools/math_program.py` verifies the source-map seal and bundle digest, then compares the map with the binary's actual header, input/output/state contract, value table, instructions and parameter references. A resealed sidecar cannot change the graph being analyzed.

The bounded analyzer propagates exact rational polynomials through admitted instructions. It first establishes bilinearity under the two declared four-value ports, then compares all 64 ordered coefficients with the supplied matrix-multiplication specification. It does not classify from the sixteen basis evaluations alone. Unsupported nonlinear instructions produce an explicit instrument refusal.

Both integer-factor cells have zero coefficient residuals in the original and deployed programs. Each approximate export has 64 nonzero residuals in both audits. Their original FP64 coefficients remain available beside the explicitly narrowed FP32 execution. Formal polynomial identity is separate from floating-point execution: this is not unrestricted FP32 or uint64 correctness.

On 16 basis pairs plus 256 fresh bounded integer pairs:

| Cell | Maximum source FP32/C error | Maximum C/matrix-truth error |
| --- | ---: | ---: |
| learned811 / learned1597 | `0` | `0` |
| approx137 | `4.76837158203125e-6` | `5.7220458984375e-6` |
| approx149 | `9.059906005859375e-6` | `2.3126602172851562e-5` |
| approx163 | `8.106231689453125e-6` | `5.7220458984375e-6` |

Swapped ordered ports and three own-output recurrence steps are retained for each cell. The verifier reconstructs each C continuation input from the preceding C output and binds it to the invocation ledger. Source continuation errors are recomputed from retained native/C observations; it does not rerun the reference model locally.

## What the consumers achieved

All 16,241 projected reads agree with the native reader. There are 4,217 distinct executed queries; repeated immutable queries reuse their prior C results. Record mutation changes the compiled bank identity. Wrong namespace, scope, missing IDs, repeated placeholders and reversed matrix ports remain explicit observations. Host frame extension receives C-selected records, without a native lookup fallback; the native lookup runs separately for auditing.

The two learned concept cases reduce primitive DAG costs `3→2` and `7→3`, and emitted C instruction counts `6→5` and `11→7`. Primitive cases stay unchanged. All 13 dimensional laws retrieve/match through the actual frozen library; their emitted before/after outputs match exactly on the bounded numerical panel.

For search, every arm shares six rounds, at most eight selections per round, 48 total selections, 768 nodes and 1,024 matches per round. Final extraction uses one pass with a 0.5-second budget. A 120-second search bound protects continuation; it is not a quality gate. Every arm uses the same FieldLM representation, feature construction and V1 base rules. The native-V1 arm audits C scores but chooses native scores; C arms consume C scores and C projected reads.

| Goal | Native V1 | C V1 | C V2a/base | Heuristic | Random | Enumeration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Expanded scalar product, final DAG cost | 5 | 5 | 9 | 5 | 9 | 9 |
| Planted scalar circuit, final DAG cost | 3 | 3 | 3 | 3 | 3 | 3 |
| Ordered matrix expression, emitted C instructions | 10 | 10 | 10 | 10 | 10 | 10 |

The scalar product starts at 25 emitted C instructions: V1 and heuristic reach 11, while V2a, random and enumeration reach 15. The ordered matrix expression starts at 23 instructions and four learned-cell calls; every arm reaches 10 instructions and one call. Its nominal dense linear work falls `336→84` MACs per row, with zero observed output difference. This is useful symbolic-to-C integration, not evidence that learned selection caused the matrix saving.

The 45 scorer batches contain 987 actual candidate rows. Maximum source/C score error is `4.57763671875e-5`. One V2a frontier changes order inside the same selected set; no selected-set mismatch occurs. Full scores, orders and cutoff margins remain retained. These are scorer errors, not vocabulary-logit errors; the earlier PAN/Mamba logit measurements remain in the [model inventory](../../docs/SUPPORTED-MODELS.md).

The final worker took 12.87 seconds; the scheduler measured 16.61 seconds and 640.4 MB peak RSS. mrun selected a 1,024 MB reservation from measured history. CPU arithmetic uses one thread, with four cores reserved by the planner. There is no CUDA, training, nested submission or automatic retry. Complete per-arm wall times charge compilation, native auditing, extraction, C calls and process overhead. C-assisted arms take longer in this harness, which does not isolate lookup or serving speed.

## Run and inspect

From a cmachine checkout in the domains workspace:

```sh
rtk proxy ../common/bin/common exec python tools/math_fleet.py submit
rtk proxy ../common/bin/common exec python tools/math_fleet.py collect JOB_ID
rtk proxy ../common/bin/common exec python tools/verify_math.py results/JOB_ID \
  --output results/JOB_ID/verification.json
rtk proxy ../common/bin/common exec python tools/math_program.py \
  results/JOB_ID/raw/learned811.tcm
```

The guarded worker stages local frozen exports, reads hash-pinned archive objects on Beast, builds C, captures the models, executes the finite panel and collects bounded raw evidence. Small compiled programs are included; the archive still has a 64 MiB expanded limit. No model forward occurs on the submitting machine. Verification and polynomial analysis are offline artifact analysis.

For a new trusted vector model, call `lower_fx(module, input_width, checkpoint_sha256, destination, source_record=...)` in an admitted worker. The existing compiler generates schedule, parameters, symbols and source references. Static slices/transposes, normalization, linear layers and ReLU need no per-model C dispatcher. `ProjectedField` similarly compiles the records supplied by an existing native view; unknown string keys return no records, and dictionaries beyond exact FP32 integer capacity refuse.

The caller still supplies a reviewed source loader, input layout, feature/domain declarations and any candidate mathematical specification. New operations need implementations and numerical checks. This does not infer an arbitrary architecture or concept language from opaque weights.

## Next automation cuts

The next useful engineering step is a resident C call interface or batched projected-read ABI, removing per-query process and compilation overhead while preserving identical query/record identities. The next research step is a broader matched-budget set of new goals/seeds, charging all search work and retaining heuristic, random and V2a counter-trends. Structural candidates can then propose discriminating interventions against actual consumers; a supplied polynomial specification should stay distinct from an autonomously discovered role.

Astra reviewed provenance, feature versions, FP64 narrowing, ordered-domain controls, budget fairness and independent-verifier coverage. The final fixes bind certificates to binary instructions, match enumeration's round budget, recompute work counts and verify C continuation chains. Seven generated math mechanics tests pass in scalar and ASan/UBSan builds; the existing 36 runtime/intake regressions also pass. The standard Saturn sweep has stream-only coverage and makes no semantic certification claim.
