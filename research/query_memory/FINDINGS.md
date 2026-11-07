# An own-query memory component, with a measured compilation refinement

November 6, 2026. We implemented and ran an updating attention program on original Qwen2.5-1.5B-Instruct. It forms future Q/K/V from each branch's actual receiver, appends its own bounded memory, reads that memory and passes the result through the original native consumer and own-token continuation. Expanding the source-activation compiler improves initial attention and full-logit fidelity in all three development cases. The checkpoint-only compiler is a preserved negative/mixed result.

The remaining boundary is substantial: the program receives its normalized receiver from full native lower-layer computation. Its 98,464-byte memory alone does not generate that receiver. It retains the full original model, 27 other layers' caches, O/MLPs and vocabulary readout. This establishes a working component approximation and a useful refinement, not compact whole-model state or forward-pass-free chat.

## Executed experiment and iteration

| Job | Compiler | Paths / own token advances | Saved memory snapshots | Scheduler seconds / peak RSS / VRAM |
|---|---|---:|---:|---|
| [2230480e1c18](runs/job-2230480e1c18/report.json) |193 source receivers →activation 32 |15 /1488 |24 byte-exact |38.73 /3994.6 MB /6406 MB |
| [077becbef4c9](runs/job-077becbef4c9/report.json) |Original QKV weights →nested 32/128 |21 /2119 |36 byte-exact |49.68 /4085.5 MB /6406 MB |
| [c7334bbc5bb8](runs/job-c7334bbc5bb8/report.json) |Same authenticated 193 rows, original 32 prefix +96 directions |21 /2090 |36 byte-exact |52.67 /4083.6 MB /6406 MB |

Each job loads one original C model in one mrun CUDA lease on Beast, uses one CPU thread, strict preflight, one process and no retries. Each native-wrapper control compares 16 tokens' full logits and final all-layer K/V byte-exactly. Every generated token is consumed, including EOS; exact later user/header IDs are appended. All candidates seal before the native reference paths. The 21-path studies have 18 candidate paths and 3 native references. None receives native target state or query refresh.

The initial 193 source-activation advances and their dense readouts are paid. Subsequent compilers acquire no new calibration forwards. The final 128 basis uses only those historical source rows, keeps the original rank-32 basis bytes exactly and adds 96 maximum-source-residual directions with explicit reorthogonalization. Its selection follows development results and an initial-prompt geometry diagnostic. These repeated three cases are development evidence, with no held-out generalization claim.

The policies are native, bounded memory with native QKV, zero attention, compiled QKV with full attention at 32/128, and compiled QKV with bounded memory at 32/128. This separates query formation from memory approximation and preserves their interaction.

## What the small memory actually does

Each of two KV groups retains 16 recent exact post-RoPE key/value rows and 16 older counted prototypes. A retired row fills an unused slot, then merges into the nearest key mean. Counts and FP64 key/value sums determine prototype means. Twelve query heads read their original GQA group. The stable softmax score includes log(count), so the state represents a positive query-dependent response function over all retained history.

No original retired K/V rows are accessible to the candidate reader. The wrapper skips selected-layer global cache writes; that arena stays zero, verified after every compact path. Independent reconstruction reproduces every saved state and every recorded compact attention return byte-exactly. The reader handles each newly produced query, rather than reusing a numerator/denominator bound to one earlier query.

Prototype averaging deliberately omits within-cluster exponential dispersion and query-selected key/value covariance. Before the first merge, its FP64 reader differs from the saved mathematical full-history read at roughly 2–3×10⁻⁸ relative after FP32 rounding. After merging, errors become material. With native QKV, mean own-query attention errors are 18.17%/18.79%/16.40% across recall/math/prose, with maxima 58.41%/52.15%/92.09%. These are compression/profile diagnostics on each candidate's own recorded K/V, not native-CUDA same-cache model recovery. [Derivation](MATH.md).

## The compiler objective mattered more than increasing weight rank

The checkpoint-only compiler normalizes Q/K/V families equally by Frobenius norm and constructs a randomized joint right subspace. At rank 32, actual initial Q/K/V errors are worse than activation 32 across all three cases. Increasing this weight basis to 128 improves Q/K/V errors while making attention-output error worse. First full-logit differences improve for recall but worsen for math/prose. This is a nonmonotonic consumer result, not a broken instrument or evidence against all weight-based compilation.

The final source expansion gives a clear local improvement. Every comparison below pools exactly the same initial prompt positions; normalized receiver bytes match native before any Q/K/V comparison. Full attention is retained in this table to isolate compilation.

| Case | Source 32 Q / K / V error | Source 128 Q / K / V error | Attention error 32→128 |
|---|---|---|---|
| Recall |33.62 /39.00 /72.28% |22.14 /24.92 /51.36% |124.84→46.77% |
| Arithmetic |32.71 /38.80 /68.34% |23.12 /25.92 /50.35% |126.72→36.99% |
| Prose |28.60 /33.78 /61.06% |15.07 /17.15 /35.18% |128.05→24.48% |

Zero attention has 100% error at this local attention observable. The expanded program is below it on all three initial prompts; the original 32 program was above it. Native layer-exit relative errors also improve, from 0.773–0.862% to 0.144–0.309%. Their smaller scale includes the retained residual and native receiving MLP, so it cannot be interpreted as the same causal error fraction.

## Native consumers and own continuation

| First generation readout | Recall | Arithmetic | Prose |
|---|---:|---:|---:|
| Source 32, full attention |17.333% |18.428% |14.492% |
| Source 128, full attention |13.843% |15.296% |7.087% |
| Source 128 +bounded memory |14.471% |16.271% |6.980% |
| Native QKV +bounded memory |2.035% |1.705% |0.800% |
| Zero attention |17.350% |18.694% |19.984% |

These are full-vocabulary logit L 2 differences relative to the native array on identical consumed histories. They are numerical discrepancies, not semantic scores or probabilities of correctness.

Follow-ups retain a countertrend. For source 128+memory versus source 32+memory, recall-prefill difference worsens 15.241→17.900%; arithmetic improves 28.831→27.249%. Those histories still match. Prose differences are 35.707% versus 38.968%, but generated histories have diverged, so this is an own-policy outcome comparison rather than a controlled same-cache fidelity improvement.

All policies, including deletion, answer supplied colors blue then green. All answer 14×19 as 266 and then add 13 as 286; the correct second answer is 279. The source 128+memory arithmetic explanation is budget truncated. Native makes that same arithmetic error. No tools, external memory or arithmetic repair are supplied.

The source 128+memory prose path completes both turns:

> Labeling a notebook helps to organize information and keep track of important notes, making it easier to find and access the information later.

Its example is `"Important meeting notes: 2023-04-15"`. Native instead produces a doctor's appointment example. These are plausible own continuations; the date is model-generated example text. Deletion also gives coherent prose. This panel establishes numerical consequences of the attention computation, with no necessary semantic role for this selected layer or reliable general-chat capability.

The [figure](runs/job-c7334bbc5bb8/query-memory.png) shows the source-compiler improvement, native readouts, remaining compression loss and logical versus physically retained memory. Its own-query compression curves follow different branches' Q/K/V and histories; their relative heights do not rank native-query accuracy.

## Computation and memory account

| Quantity | Native selected component | Compiled 32 | Compiled 128 |
|---|---:|---:|---:|
| QKV nominal MAC/update |3,145,728 |114,688 |458,752 |
| Selected QKV reduction |— |96.354% |85.417% |
| Whole-token learned-linear reduction |— |0.1964% |0.1741% |
| Frozen basis +three images |Original weights retained |458,752 B |1,835,008 B |
| Coordinate scratch |— |128 B |512 B |

The whole native token retains 1,543,569,408 nominal learned-linear MAC and 197 matvec calls. A compiled path makes 198 calls due to the extra coefficient projection. The two-program experiment keeps both compiled banks resident, totaling 2,294,400 numeric GPU bytes including scratch. The current wrapper does not unload original QKV weights.

The complete numerical memory structure is 98,464 B. Native selected FP32 KV uses 2048 B per valid token, so this is about 49–61% less than the tested 94–124 token valid prefixes and 90.61% less than 512-token capacity. The full 1,048,576 B selected arena remains physically allocated. Diagnostic complete histories, readouts and snapshots are additional paid evidence storage; the full process still peaks near 6.4 GB VRAM. This is a logical serving-reader reduction, with no measured process-memory saving.

The selected score/value product count changes from 3072t to at most 3072×32 per update. Over final coupled paths it removes 43.74%/44.21%/51.20% of those products at their own 94/95/112 token lengths. Prototype nearest-mean routing adds 4096 products after 32 positions. CPU exp/log/divisions, sums and 14,336 CPU↔GPU serving bytes per update remain explicit in [study.json](runs/job-c7334bbc5bb8/study.json). Their precision/profile differs from native attention. Instrumented per-path timing is roughly 1.04–1.36 seconds depending on length, with one 1.53-second memory-only outlier; the panel establishes no speedup.

The checkpoint-only two-program compiler pays 4,658,823,168 counted matrix MAC, plus QR/SVD/scalars. The source expansion pays 151,781,376 residual MAC, 46,891,008 direction-reorthogonalization MAC and 503,316,480 C image-compilation MAC: 701,988,864 new counted MAC, plus scalar work. Reusing prior source rows avoids new dense acquisition, but the historical 193 native advances cost 297,908,895,744 nominal learned-linear MAC. Original basis construction and the first image compiler were also paid. Amortization must distinguish a reused existing source corpus from a fresh deployment's acquisition cost; compilation is not free.

## Integrity, review and next implementation boundary

Strict offline verification authenticates 62/63/67 sources and 224/314/314 raw files across the three jobs, scheduler seals, raw extents, candidate seals, input/model identities, whole-invocation receipts and recomputed analyses. Reports are immutable. The individual [initial](runs/job-2230480e1c18/verification.json), [checkpoint-only](runs/job-077becbef4c9/verification.json) and [source-expansion](runs/job-c7334bbc5bb8/verification.json) verification records bind each complete report SHA 256.

The checkpoint-only follow-up matches all 9 unaffected historical native/memory/deletion paths across 111 arrays, exact histories/cache-bound boundaries and coefficient padding. The final source refinement additionally matches all 6 original rank 32 paths:15 paths/189 arrays, with four program 32 arrays exact and padded coefficient prefixes exact. Astra independently reproduces the source expansion's 96 pivots and complete emitted 128 basis bytes, along with source/raw/seal identities and memory/readout/own-token audits. [Review](REVIEW.md).

The standard isolated evidence sweeps run/verify with 0 exit status over 1548/2215/2186 rows, six applicable watch predicates each, no fires and no promoted claims. Scope is `target-streams-only`; this checks evidence plumbing and does not certify scientific generalization or semantic quality.

The next useful engineering change is an explicit attention-state participant: a component can own its allocation, append/read transition and checkpoint/rollback state without allocating the replaced full cache. It must retain the declared model/profile/parent identity and preserve ordinary dense execution. The current cursor-only stack is insufficient for a mutable centroid memory because merges rewrite old aggregates. Do not attach this memory to serving rollback without copying or undoing those changes.

The next scientific boundary is the receiver and query response together. Keep source-derived operands, form future queries without a dense lower-layer refresh if claiming a compact whole-model program, and preserve enough key/value covariance or query-sensitive support for the consumer. Measure a bounded correction under new own-state contexts before enlarging the system. Faithful repeated state transition, task truth and global cost reduction remain separate targets.

## Reconciliation with the newest parallel work

The [prospective span-response experiment](../composition/span_response/FINDINGS.md) already constructs a memory-only bank before two future queries and feeds its prediction through the actual native consumer. Eight Ward centroids occupy 16,480 B versus 69,632 B of original selected-span K/V, with 1.592%/0.783% first-logit differences. Its separately audited omission decomposes into lost exponential dispersion and score/value dependence, both substantial. It changes one read and retains dense complement/later original-memory access. Our experiment adds continually appended selected-layer memory, recurring reads and compiled query formation, with the same fundamental first-moment limitation. These are complementary boundaries rather than competing claims of complete chat closure.

The [source-weight MLP operand experiment](../operand_basis/FINDINGS.md) supplies an independent warning about our failed weight-only query objective. Its source 64 basis captures more folded operator energy than random 64 yet has pooled native-write error 1.008661 versus 0.996941 for random and 1 for deletion. Separate operand reconstruction does not order a nonlinear consumer in either experiment. Our source-activation 128 result supports an improved domain-specific query interface; it does not validate energy capture as a general compiler objective. A receiving-consumer-weighted source objective remains a hypothesis requiring new own-state tests.
