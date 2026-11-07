# Evidence status

This is a bounded research inventory as of November 6, 2026. This checkout now owns both the integrated runtime and the relocated research tree. Large original model banks, raw results and historical payloads live locally under `research` and are ignored by Git. Research sources, protocols and findings are tracked. The old `saturn/cmachine` address is a compatibility link. A separate tracked proof bundle below records the fresh integration run; migration did not rerun every historical experiment.

## Verified canonical integration

Source commit `7a6824dd9b640cfa785fa56cc4e7cfae14193f1c` integrates compressed append, bounded retention, overwritten-row recovery, source-BF16 weight tiling and persistent serving into `build/tcm-cuda`. Native remains the default. The original writer, gate-law, observation and parent-lineage APIs remain available. Research runtime paths resolve to these same canonical sources.

Guarded CUDA job `job-f645fcd76a75` built exact Git objects and ran twelve sequential model processes on Qwen2.5-1.5B-Instruct. All 84 protocol entries match a separately built original `b81e928` native core exactly. Three encoded-memory/weight profiles reproduce eight historical turns and sixteen full-logit hashes each, plus their delayed-root cache state. Native lifecycle checks, bounded occupied-sample abort, exact replay and a post-cache refusal/retry comparison pass. The retry test covers a merge-time bug fixed by sourcing evicted rows from the immutable pre-token snapshot.

Five actual text-client processes reproduce prior output with zero prefix recomputations. Short prose, explicit addition and recent recall work on the tested inputs. Bounded delayed replies remain `3`, `1234567890`, `Green`, `15`; the full-byte reference returns `17`, `29`, `amber`, `40`, with the correct sum being 46. This verifies preservation rather than new answer quality.

The bounded policy retains 9,279,620 tracked GPU state bytes at 112 physical rows, including its fixed recovery copies; model/helpers/tile bring the total to 3,133,714,584 bytes. These are requested allocation bytes, distinct from the entire validation job's measured peaks of 4,206.6 MB RSS and 6,396 MB VRAM, which include the larger reference/control processes and CUDA overhead. Logical context is qualified at 768, two bounded parents are available, and sessions last within their running process. Full parameters and dense learned computation remain.

The tested version is installed at `/mnt/big/cmachine/current`; both old current aliases resolve to it. No background service was started. The [proof bundle](evidence/integration-7a6824d/README.md) joins 120 shipped source files, 36 raw artifacts, sealed scheduler payload, installed binary and commit. An independent Astra saved-evidence audit found no material blocker. Offline evidence run/verify both return 0 with `target-streams-only` coverage; they do not certify semantic claims. The model-free lane also passes 18 Python tests and the sanitizer/lifecycle fixtures.

## PAN and instruction-Mamba graph intake

The [supported-model inventory](SUPPORTED-MODELS.md) includes the two graph profiles from [model intake](../research/model_intake/README.md). Both used the original 14-op version of the shared C dispatcher, with generated source/symbol maps, parameter bindings and recurrent state layouts. Real-model jobs ran sequentially through guarded Beast CPU leases, with native FP32 reference execution followed by independently initialized C execution. All dense learned computations remain in these profiles.

| Final job and specimen | Comparison | Maximum absolute logit error |
| --- | --- | ---: |
| `job-0143a2072cce`, PAN P113 seed 0 | All 12,769 source/C argmax predictions agree; both answer 7,382 pairs correctly (57.8119%). | `9.1552734375e-5` |
| `job-1b568233159e`, MambaSan-130m-instruct | Same 18 generated tokens across two Japanese prompts; all 86 comparable rows agree on argmax; both C baselines and the zero-writer arm replay exactly. | `1.430511474609375e-5` |

PAN's mean absolute logit error is `4.77931e-6`. The Mamba capital/addition panels have mean errors `1.80386e-6` / `1.92780e-6`; final state comparisons cover all 48 slots when consumed histories agree. The Mamba question asking Japan's capital produces `1990年`; its 7+5 response is incorrect and repetitive. Zeroing the supplied PAN gate or mapped first-layer Mamba writer changes downstream behavior. These are bounded execution and dependency observations, without semantic-role discovery, improved answers or speedup.

The [receipts](../research/model_intake/evidence/README.md) retain exact executed-source hashes, scheduler/payload identity, source maps, symbol tables and raw verification outputs. Large raw arrays and staged sources remain in ignored local result directories. Historical failed/cancelled acquisition attempts remain preserved, including the larger English Mamba-Chat downloads that never reached a forward. The graph executor is available in this source tree; these jobs do not update the separately installed Qwen release described above.

## Math intake and host reasoning consumers

The [math extension](../research/model_intake/MATH.md) adds ReLU, vector division and projected record matching to the shared dispatcher, now 17 operations. Guarded CPU job `job-2a7aae82cec8` imports two exact matrix2 circuits, three approximate FP64 exports explicitly deployed in FP32, and the frozen V1/V2a rewrite scorers. Its 127 retained programs and 4,359 C invocations pass the [independent artifact audit](../research/model_intake/evidence/math-20261106/verification.json), including binary/map agreement, all 64 polynomial coefficients, numeric rows, score rankings, C continuation chains and work counts.

All 16,241 actual projected FieldLM reads match native. Two learned concepts, three supplied primitives and 13 law bodies reach original host consumers and emitted C numerical programs. The matrix expression falls from four cell calls to one, `23→10` instructions and `336→84` nominal linear MACs, with zero tested output difference. Every scheduler reaches it. On one scalar goal C V1 beats schedule-matched random/enumeration but ties the heuristic; V2a is weaker. Scorer error reaches `4.57763671875e-5`, with one order difference inside an unchanged selected set. Approximate cell source/C errors reach `9.059906005859375e-6`.

Parsing, domains, rules, features, graph mutation and extraction remain supplied host computation. Formal identity is distinct from unrestricted FP32 correctness. C-assisted arms take longer in this harness, whose times include compilation and auditing as well as C subprocesses. General reasoning improvement and speedup remain open. The final panel uses fixed Python hash order and matched round budgets. Earlier completed receipts retain their different settings and weaker audit coverage; they are not rewritten as the final protocol.

## Contextual operand programs and a later-layer gap

The [operand interfaces](QUERY-PROGRAMS.md) now support immutable Q/K/V programs at several model-qualified layers, finite first-layer token tables, and the shared prototype-reader law/covariance pullback. Native remains the default. These newer sources were tested by guarded job `job-fe51e7f4f0ee`; they have not replaced the separately installed `7a6824d` release.

The new interfaces replay 420 archived source clocks exactly at Q/K/V/attention and token choice. A 15-token atom subset reproduces 16 steps of full logits exactly. A source-only same-memory response direction then lowers mean own-history source KL 34.54% and fresh two-turn KL 13.59% across two task families at 128/1024 prompt tokens, with no extra serving maps or state. Two inventory followup means worsen slightly despite all four conversation means improving.

A recipient-specific rank-128 layer-18 Q/K program raises fresh error even with full layer-18 memory and misplaces the whistle in both initial inventory replies. Its source receiver basis retains 95.22% energy; this does not establish future relational sufficiency. Prototype memory there partly compensates in one case and worsens three. That joint assay does not distinguish Q versus K error or identify consumer-relevant coordinates under subsequent writes. Full residual formation, other learned maps and readout remain dense. See [findings](../research/context_query/FINDINGS.md) and the [portable result record](evidence/context-query-20261106/README.md).

The [subsequent Q/K cut](../research/query_key_cut/FINDINGS.md), job `job-1555c27c080f`, now separates those terms on the known failures. At 1024 tokens, native keys with the compiled query restore silver on a native-layer18 historical parent; native query alone does not. On the compiled parent, even native current Q/K leaves amber. Current layer18 closes byte exactly across histories and attention19 immediately differs despite equal current input. All 32 physical branches restore the same owned parent and four shams are exact. This identifies current key error and inherited higher-layer history, with the 128-token case still unrepaired. Native diagnostic projections remain paid; no new serving law or speedup is claimed.

The [from-zero mask follow-up](../research/query_key_cut/ON_POLICY_PLAN.md), job `job-1335a953cd08`, keeps the frozen layer18 query but uses native keys throughout each own-generated conversation. Both known inventory bindings and their follow-up moves are correct at128/1024tokens; the reciprocal key-only program still misbinds. Query-only improves inventory KL36.64%/32.41% relative to updated17 alone but worsens explanation KL73.46%/70.28%, leaving the four-conversation mean20.62% worse. These eight paths reuse known development inputs; no new heldout or general-chat claim follows. The nominal whole-model learned-linear saving for query-only17/18 is0.2781%; most dense computation remains. See [portable records](evidence/query-key-cut-20261106/README.md).

## Execution and source gate law

The C executor owns incremental input, greedy outputs, persistent cache, nested continuations and later user/header ingestion. B3 replacement runs through complete correction and question episodes on its own generated IDs. All tested short badge answers also survive deletion, so those answers do not establish semantic necessity.

`job-91ba52c5fe9a` evaluates a frozen source-anchored scalar gate law through two development conversations. All 93 evaluated write errors are below the zero-write distance. Recurring means are 24.42% and 30.59%; follow-up full-logit discrepancies against native reconstruction on the same consumed IDs are 4.108% and 10.391%. These are dependent uses within two conversations, not 93 independent chat trials.

The gate law still executes all 197 dense matvecs and 1,543,569,408 nominal learned-linear MACs per token. It replaces selected-site exponential evaluation with a polynomial and retains full weights and K/V. No clean serving speedup or general answer-quality improvement is established. The native arithmetic specimen answers 17×23 as 401 rather than 391.

Report SHA256: `d8173d0206fd2a52cffa44f8f17bd5a2fd9dd1c7b33a8432d2fc133415627ee4`.

## Exact local-interface insufficiency

`job-7b8a5933f1dd` constructs three supplied receiver pairs whose executed B3 coefficient/RMS code is exactly the same 16 bytes. Their native MLP-write separations are 10.6575, 17.8135 and 20.8831 L2, respectively 61.75%, 84.13% and 118.66% of the source-write norm.

For any deterministic code-only predictor h and equal-code pair, at least one endpoint error is at least half their write separation. Refitting that unchanged interface cannot recover both writes. Dense omitted gate/up images repair the alternate write to roughly 7×10⁻⁷ relative error, while paying two original projections.

This is a local supplied-intervention counterexample, not a proof that natural chat cannot use compact state. Native full-logit and prefix-cache effects persist even where the short generated IDs agree. Captured new-row equality must not be confused with equality of all valid prefix memory.

Report SHA256: `b693fc8b12dbf0e79f93a9337136517aae1d020f267467a74357ce84ed58208b`.

## Formed memory and later reading

`job-a071c7a43a38` runs the complete badge correction and two independent later questions under native, PROJECT, corrected and zero policies. PROJECT improves all three pre-answer full-logit distances over deletion and worsens all three post-EOS distances. Signed cancellation changes its net usefulness at different consumers.

`job-c593b2c6e4a6` completes the NN/PN/NP/PP formation-by-question factorial. Dense question execution on PROJECT-formed context still differs from native. Matched question transitions can have identical local MLP16 computation and newly formed KV16/17 while later logits differ because historical higher memory remains different. Native execution now does not erase past cache error.

These are episode policies, not isolated one-step readers. All actual short answer IDs remain equal in this specimen. Relative norms, interactions and task correctness are separate observations.

## Completed attention mechanism trace

`job-812a0486995d` follows the factorial into native layer17 attention at pre-answer and consumed-EOS boundaries. All selected full-logit and valid-cache endpoints reproduce authenticated uninstrumented runs.

The query depends on question policy, while correction-band K/V depends on formation policy. Captured score interaction is exactly zero outside correction columns. Softmax normalization distributes probability interaction onto unchanged source and question spans: 48.45–73.94% of its absolute L1 mass lies outside the correction band. These are vector-support fractions, not semantic causal percentages.

For a fixed supplied query, changing correction memory satisfies

\[
\Delta c=\frac{\Delta N_C-c_{before}\Delta Z_C}{Z_{after}}.
\]

The account differs from native FP32 context change by 2.81×10⁻⁷–6.45×10⁻⁷ relative L2. The receiving MLP opposes the incoming interaction but increases the final interaction norm before answers and decreases it after EOS. Opposition alone does not establish repair or contraction.

This closes a conditional native read account. It does not generate future queries or N/Z signatures from compact memory, remove the original attention computation, or establish semantic necessity.

## Constructive nonlinear quotient and a rank constraint

`job-d57473b075cc` executes a supplied eight-coordinate nonlinear residual organism through unchanged C CUDA primitives. A two-coordinate state plus invariant energy advances its own exact 16-update trajectory. Adding one off-space write makes omitted sign consequential; retaining that coordinate restores byte-exact execution over the same horizon.

The respective nominal affine MAC counts per update are 96 for the full reference, 24 for the closed two-coordinate program and 36 for the repaired three-coordinate program. Full-width normalization representatives, constants, scratch and diagnostics remain paid. This known-wiring result is not learned compact Qwen state or a measured latency saving.

Separately, an exact nonzero determinant of an original layer16 down-map 256×256 minor proves rank(D)≥256. This excludes smaller retained spaces for the strong condition that every possible D write lies in a common linear space. It does not prove full rank, that reachable SwiGLU writes span all columns, or that useful rank-64 approximation is impossible.

Report SHA256: `10dbec1fc475d10366a9ce33c8d0b0fcdbff4e3ecf010458b59fca3d8dbca384`.

## Broader compiler and prediction evidence

The source-derived even-SwiGLU/readout campaign predicts 100 immediate Qwen token choices and twenty complete FLUX edited-state image futures under their qualified numerical profiles. Later Qwen cached steps are native observations. The complete image predictor retains original learned maps and shared primitives. The 0.5B CPU Qwen numerical results do not automatically qualify the C 1.5B CUDA profile.

The time-circuit image lane supplies complementary evidence:

- Compiling four supported clock-only operand bundles removes six learned tensors from serving: approximately 341.3 MB of weights become 442 KB of operands. Tested images remain exact. Only 0.00361245% of nominal transformer linear MACs is removed.
- A selective rank-one reader has 10.72% signed PNG response error on one paired specimen, compared with 104.17% for deletion.
- An 8,208-byte uniform reader advances exactly but merges the objects; response error is 122.68%. This rejects that summary's usefulness on the pair rather than every compact reader.
- A selected-only variant removes 88.93% of comparable reader numerical state while retaining the tested objects and swap; response error rises to 19.37%. Full K/V scanning and most learned work remain.

These results separate state-law mechanics, consumer fidelity, useful approximation, parameter storage and dynamic computation savings. They motivate operation-specific compilers rather than an assumed universal tiny state.

## Open status

General cheaper operand formation, a reusable response to the queries generated by the candidate itself, raw new-input bootstrap, whole-model coverage, useful long chat and serving speedup remain open. The Astra-advised operand experiment running separately is not part of this extraction and earns no result label here. See [the vision and next cuts](VISION.md).
