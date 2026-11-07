# Independent review of checkpoint-only query compilation

Reviewed 2026-11-06. This is an additive interpretation of immutable jobs
`job-077becbef4c9` and `job-2230480e1c18`; it does not alter either experiment.

The bounded memory program repeatedly produces answers using its own queries,
causal appends, and retained memory. The new checkpoint-only compiler does not
improve fidelity consistently: at equal rank it loses more of the actual Q/K/V
responses than the earlier source-activation compiler, and increasing its rank
improves local Q/K/V reconstruction without consistently improving attention or
downstream logits. These are mechanically valid observations, not an exclusion
of compact emulation.

## Evidence and independent checks

The follow-up report SHA256 is
`4cde6efb3720e784271b447a7efcb94f1d512db788503782df6cfa1a05622de5`.
The earlier report SHA256 is
`c4897a9c69bd8c644e026ac92dcccc959cd65c9e25f885873de459ce53ac51e2`.
Inspection used the saved payloads and raw arrays, not the mutable working
analyzer.

- Checked all 63 frozen source hashes, all 314 raw hashes and extents, and all
  263 candidate-seal entries. Raw files total 116,074,117 bytes.
- The nine repeated native/memory/zero paths have identical consumed IDs, turn
  records, retained common arrays, full logits, and 36 boundary cache hashes.
  Coefficient-array widening preserves the old 32 columns and adds zeros.
- Independently reconstructed all 36 compact-memory snapshots byte for byte.
  Independently recomputed compact attention with `math.fsum` for distances,
  scores, normalization, and value mixing: every resulting FP32 output was
  byte-exact. This covers nine compact trajectories, not merely one replay row.
- Checked all 84 full-logit argmax readouts and all 495 generated-token choices
  against their own pending-token streams.
- The emitted rank32 basis is the exact prefix of rank128. The maximum rank128
  Gram defect is 5.926e-9. The compiler's right-space SVD orientation and C image
  layouts agree. Authenticating the saved source recipes and checkpoint identity
  is not an independent recomputation of the original weight matrices locally.

The follow-up contains 21 paths and 2,119 own advances. Scheduler duration was
49.41 seconds; worker duration 41.38 seconds; driver duration 31.19 seconds.
The source compiler took 2.96 seconds. These are different timing scopes.

## Compiler effects and countertrends

The new basis comes only from original layer17 Q/K/V weights, with each family
normalized by its Frobenius norm. It uses no target rows or source-calibration
forward passes. This objective represents equally weighted isotropic input
directions; it does not minimize error under the model's actual receiver
distribution or under the nonlinear attention consumer.

For the entire common initial prompt, the selected normalized input rows are
byte-identical to native. Consequently the following pooled relative errors
compare genuinely common operands. Entries span the three cases.

| Compiler | Q error | K error | V error | Attention-output error |
|---|---:|---:|---:|---:|
| Earlier activation32 | 28.60–33.62% | 33.78–39.00% | 61.06–72.28% | 124.84–128.05% |
| Checkpoint weight32 | 54.89–55.69% | 63.13–66.05% | 95.86–96.68% | 113.84–114.00% |
| Checkpoint weight128 | 45.05–46.03% | 44.23–47.25% | 80.26–82.59% | 125.99–128.31% |

Increasing the weight basis rank improves all three local Q/K/V errors, yet
increases the pooled attention-output error in all three cases. The lower
attention error of weight32 therefore cannot certify a better query compiler.
Nor does better Q/K/V reconstruction alone order attention fidelity.

Initial full-logit differences for the projection-only arm are:

| Case | Activation32 | Weight32 | Weight128 | Memory-only | Zero attention |
|---|---:|---:|---:|---:|---:|
| Recall | 17.33% | 20.41% | 15.02% | 2.04% | 17.35% |
| Arithmetic | 18.43% | 19.17% | 25.20% | 1.71% | 18.69% |
| Prose | 14.49% | 15.31% | 20.83% | 0.80% | 19.98% |

Weight32 worsens all three initial readouts relative to activation32. Weight128
improves recall but worsens arithmetic and prose relative to weight32. There
are countertrends at later common-history boundaries: arithmetic follow-up
prefill improves from 30.52% for activation32 to 22.56% for weight32 (25.24% for
weight128). Comparisons after generated IDs diverge describe whole trajectories,
not a local intervention on the same cache.

The mean compact-reader error against full attention on each candidate's own
Q/K/V history is about 16–19% for memory-only, 4–5% for both32, and 8–9% for
both128. These use different queries and memories. Lower both32 error does not
show recovery of native attention; weakened or less selective queries can make
centroid approximation easier. Candidate own-history fidelity and original-model
fidelity must remain separate.

## What the behavior establishes

All arms, including deletion, answer the recall case blue then green. All answer
the first multiplication 266 correctly and begin the follow-up with 286, whereas
266 + 13 is 279. The arithmetic follow-up is budget-truncated. Prose remains
coherent across all arms, including deletion, with changed wording and examples.
The selected component therefore has measurable causal effects on logits and
some generated text, but these three cases do not establish that retaining its
approximate memory is necessary for task success or that it repairs reasoning.

The compact reader is an operational autonomous component: it starts at token0,
accepts its own current query, appends its own K/V, and reads only its bounded
state. The full diagnostic K/V history is isolated from that reader. Biases,
RoPE, grouped-query mapping, the output projection, and downstream consumers
retain their declared native implementations. The selected full-cache arena is
still allocated but remains untouched in compact arms.

Autonomy is bounded to this component. The earlier layers still produce its
receiver, 27 other layers retain full native caches, and the native suffix
produces each token decision. This is not a compact recursively closed
representation of the whole model, nor forward-pass-free generation.

## Work and state accounting

The memory occupies 98,464 bytes versus 1,048,576 bytes for the selected layer's
512-row cache capacity. All original weights and global cache allocations remain
resident. The CPU64 centroid reader, transfers, and centroid-update work are paid;
the full-cache scan is absent only from the selected compact reader.

Rank32 costs 114,688 selected-QKV MACs per advance; rank128 costs 458,752 versus
3,145,728 native. These save 96.35% and 85.42% of selected QKV work, respectively,
but only 0.1964% and 0.1741% of whole-token learned-linear MACs. Each factorized
advance has four matvec calls instead of three. These counts do not establish a
measured speedup.

The follow-up's static accounted floor is 4,658,823,168 MACs: randomized joint
maps, final rotation, family diagnostics, and both CUDA image compilations.
QR/SVD, normalization, and scalar work are additional. The compiled coefficient
storage is 458,752 bytes for rank32 and 1,835,008 for rank128. Offline diagnostic
cost should be distinguished from a hypothetical minimal production compiler,
but retained in the actual experiment ledger.

## Best next cut

A frozen nested activation32/128 compiler from the first job's 193 source-input
rows is a useful smallest next contrast. Preserve the original32 prefix,
deterministic pivot rule, source identity, and all target-independent choices;
inspect source numerical rank and orthogonality before execution. This separates
rank from the source distribution used to choose directions without paying new
calibration forwards. The choice of this follow-up is informed by these target
observations and should be described as exploratory, not a fresh held-out
confirmation. Target coverage is a diagnostic, not a criterion for fitting or
choosing the basis.

Judge that contrast through both the common-operand Q/K/V→attention chain and
own-trajectory native-consumer effects. If source geometry helps but value
formation remains dominant, a subsequent separate Q/K versus V compiler cut is
more informative than another undifferentiated rank increase. Longer-range or
deletion-sensitive examples are eventually needed to establish useful retained
memory, but are unnecessary to declare this mechanics result valid.

## Addendum: completed source-span refinement

The proposed source-only contrast has now executed as `job-c7334bbc5bb8`.
Its immutable report SHA256 is
`f033d61f0e02a047af0991bfa89845600eaafd3afd96d34fe41218caaffd933e`.
This addendum preserves the earlier interpretation and adds the new evidence.

Independent checks passed for all 67 frozen source hashes, 314 raw file hashes
and extents, and 263 candidate-seal entries. The raw files total 115,283,192
bytes. All nine native/memory/zero paths and all six activation32 paths exactly
match the original experiment in histories, turn records including cache hashes,
seven retained ports, and four full-logit files per path. All four compiled
rank32 program arrays are byte-identical. The authenticated 193 source rows also
match the first experiment byte for byte.

An independent saved-only reconstruction of the source Gram–Schmidt expansion
recovered all 96 extension pivots and the full emitted FP32 rank128 basis exactly.
No target rows entered that computation. Independent memory reconstruction again
matched all 36 snapshots; an `fsum` attention implementation produced the exact
saved FP32 outputs throughout all nine compact paths. All 84 full-logit argmax
readouts and all 466 generated-token choices matched their own authority streams.
The run has 21 paths and 2,090 own advances.

The improvement is substantial and consistently visible on the initial common
operands. Selected normalized receivers are byte-identical across the initial
prompt; these are not comparisons between different generated histories.

| Case | Source32 attention error | Source128 attention error | Source128 plus memory error | Source128 first logits | Source128 plus memory first logits |
|---|---:|---:|---:|---:|---:|
| Recall | 124.84% | 46.77% | 49.72% | 13.84% | 14.47% |
| Arithmetic | 126.72% | 36.99% | 39.52% | 15.30% | 16.27% |
| Prose | 128.05% | 24.48% | 28.46% | 7.09% | 6.98% |

Attention errors in this table pool the common initial-prompt vectors against
native. Logit errors compare the complete first readout against native. At
source128, pooled Q/K/V errors are respectively 15.07–23.12%, 17.15–25.92%, and
35.18–51.36%. All improve relative to source32 and to checkpoint-only128. Both
source128 arms improve initial logit fidelity over source32 and deletion in all
three cases. This supports a working inference that the receiver distribution
used to select the compiler's directions matters materially; increasing an
isotropic weight basis to the same rank did not yield the same improvement.

The improvement does not make the full trajectory uniformly closer. Recall
follow-up prefill changes from 16.50% to 16.32% for projection alone, but worsens
from 15.24% to 17.90% with compact memory. Arithmetic follow-up prefill is still
26.01% and 27.25%; its answer remains 286 instead of 279. Prose follow-up prefills
are about 35%, but their histories have already diverged. They cannot be
interpreted as a local error on native's cache. The projection-only and combined
source128 arms emit identical answers in all three cases despite nonzero internal
differences. Their coherent text, like deletion's coherent text, is not evidence
of semantic necessity or mathematical correctness.

The final compiler's counted static floor is 701,988,864 nominal MACs:
151,781,376 source-residual operations, 46,891,008 extension reorthogonalization,
and 503,316,480 image compilation for both programs. Pivot norms, normalization,
and scalar work are additional. Compilation took 0.075 seconds. No new source
acquisition forwards were paid in this job; the 193 original native source
advances remain part of the historical acquisition cost, rather than becoming
free because their arrays were reused. Scheduler/worker/driver durations were
52.57/43.03/34.24 seconds. Dynamic savings, memory allocations, extra matvec call,
transfers, and full native scaffold have the same boundaries described above;
these runs do not establish a latency speedup.

The current `MATH.md` and `README.md` correctly distinguish a positive finite
response function from exact native quotient closure, explain the lost
exponential mass and key/value covariance, and retain the paid native scaffold.
The covariance decomposition and Jensen direction are correct in the declared
real-arithmetic idealization. A minor wording correction is that the displayed
Taylor expansion is around q=0, equivalently in small centered scores, rather
than an expansion "around q." The positional/capacity bound and absence of a
standalone compact-state checkpoint lifecycle are appropriately explicit.

No further model run is needed to support this bounded result. The next design
question is how to preserve the query-sensitive response with fewer directions
or moments, while owning the rest of the transition state. Low-rank covariance
corrections or query-sensitive cluster splitting are concrete hypotheses, not
already validated repairs. Any later comparison should keep source compiler
fidelity, memory approximation, native-consumer usefulness, and whole-model work
as separate axes. Exact execution of the approximate loop is established here;
general native equivalence and forward-pass-free whole-model emulation remain
open.
