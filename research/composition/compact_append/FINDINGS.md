# Growing compressed memory composes across two attention layers

The C experiment now owns growing signed-byte K/V at layers 17 and 18. Every actual new row is encoded, published and consumed by the current attention read; selected layers have **no dense cache slot**. Rollback truncates encoded cursors with conversation state, including abort after layer17 has already appended and read while layer18 fails before publication. The stacked arm advances three linked user turns from cursor 93 to 182 and preserves native answers.

This addresses the missing append and physical-ownership contracts in the [previous fixed-span study](../span_recurrent/FINDINGS.md). It also exercises changed queries: all 89 Q18 arrays differ from native because layer17's altered computation feeds them. The additional layer18 compression consumes those actual changed queries, without a native donor or query refresh.

Tracked state allocation falls by **1,032,188 bytes** with two selected layers, including byte arenas, decode workspace and status storage. The six measured stacked full-logit differences are **0.370–1.363%**. This remains approximate selected-layer K/V compression in one supplied-fact conversation. Full weights, 26 dense cache layers and all dense projections remain; context storage still grows linearly. General compact sufficient chat and a speedup are not established.

[Plan](PLAN.md), [runtime and harness](README.md), [math](MATH.md), [report](runs/job-2a28feeb7e5b/report.json), [verification](runs/job-2a28feeb7e5b/verification.json), and [Astra review](REVIEW.md) preserve the implementation and evidence boundaries. The blog links this result into the [MLP/context/compressed-response account](../../../../obsidian/blog/2026-11-06-125544-a-frozen-mlp-program-updates-a-conversation.md).

## Construction and current-row consumption

Same pinned Qwen2.5-1.5B-Instruct C FP32/cuBLAS specimen, capacity 512, 12 query heads, 2 KV heads and width 128. The native source/correction produces the authenticated cursor 93 prefix. A saved audit/hydration artifact contains the original positioned K/V. Before the C driver opens the future question files, it freezes layer17 and 18 byte-prefix programs, 49,136 bytes each. Constructor query fitting, training, rank search and semantic parsing are absent.

Four sessions run sequentially against one weight bank: ordinary native, exact-FP32 callbacks 17+18, byte17, and byte17+18. Each preceding session is destroyed before the next. Initial hydration buffers are freed before recording steady allocation. In byte sessions the selected dense layer slot is absent; only codes/scales and shared scratch are reachable from that reader. The original prefix remains on disk as an audit/control artifact, not an attention fallback.

Unlike the previous experiment's 34-row correction-span replacement, initial encoding covers **all 93 prefix rows** at selected layers. Each later row is the candidate's own current post-bias/post-RoPE key and actual value, encoded with per-row FP32 maxabs/127 scales and signed codes. The zero row uses scale 1; FP32 division and ties-away rounding are declared. The callback validates finite rows and scales before advancing its valid cursor. It then attends over that encoded prefix, **including the new current row**.

Key decode, native-shape score GEMMs and native softmax run before the same float head-plane is overwritten with decoded values for value GEMMs. Explicit default-stream checks make the shared workspace ordering part of the contract. No decoded float bank stays resident, and no selected native cache fallback exists.

Every new user/template and own generated token, including EOS, takes this path. Each session executes 89 main token transitions; four sessions give 356. There are 267 main selected byte-layer appends: 89 at byte17 plus 178 in the stacked arm. Together with 93 shared source/correction transitions and 22 successful mechanics transitions, the C run completes 471 full token transitions, plus two deliberately aborted partial tokens. These counts describe one conversation and its controls, not independent examples.

## Rollback, sibling replacement and partial-token abort

At root 93, each session advances two tokens, rolls back a child, overwrites a different sibling suffix, rolls back root, and replays the original two-token branch. Valid logical cache hashes and full output logits reproduce exactly. A restore changes reachability through the valid cursor; invalid suffix bytes need not be zeroed.

Byte sessions inject an error after encoding a new row but before publishing its cursor. At layer17 the partial cursors are [93,0], where the second entry is unselected. At layer18 they are [94,93]: layer17 has already published and consumed its row. Private token abort restores metadata, phase 0 and every selected cursor to 93. The valid root hash matches; a subsequent successful token executes and rolls back normally. This tests a partial multi-layer transaction rather than merely rewinding after successful tokens.

The exact callback arm makes 188 cache-step calls including mechanics; byte17 makes 96 and byte17+18 makes 192. There are 476 calls total, 474 published rows and two refused publications. Callback rewinds are 3 for exact and 6 for each byte arm. The saved analyzer checks these counts, histories, clocks, own greedy-token authority and EOS closure.

## Exact layout control and changed-query composition

The new slot map and callbacks are an instrument change. Exact-FP32 callback execution matches native at every captured Q/K/V, every sparse port, all six full-vocabulary readouts and logical cache boundaries. Ordinary native also reproduces all six historical native readout/cache boundaries from the preceding experiment. All 81 source hashes and the deterministic transform of the three authenticated runtime files verify.

All 89 Q17 arrays stay native under matched token histories. At 18, byte17 and byte17+18 have the same changed Q/K/V, because both use identical layer17 approximation and lower computation. Q18 differs from native at every position: relative L2 mean **0.241851%**, range **0.134914–0.549311%**. Adding byte 18 therefore tests compression under the actual changed upstream query, with no token-history confound here.

This is cross-layer changed-query composition. It is not continuous feedback from layer18's output into its own next query. The causal query inputs come from lower layers; changed emitted tokens would provide another return path. No emitted token diverges in this run.

## Fidelity, adverse trends and behavior

Independent FP64 response diagnostics compare each encoded cache against that same candidate's original own-row history at the same current query. They do not substitute native rows. Each summary contains 12 heads×89 positions=1,068 correlated reads.

| Selected response | Mean Z ratio | Mean relative N error | Worst relative N error | Mean conditional-mean error | Worst conditional-mean error |
|---|---:|---:|---:|---:|---:|
| Byte17, layer17 | 1.003558 | 2.100% | 9.946% | 1.735% | 5.296% |
| Byte17+18, layer17 | 1.003558 | 2.100% | 9.946% | 1.735% | 5.296% |
| Byte17+18, layer18 | 0.992295 | 3.672% | 21.998% | 2.460% | 10.867% |

Layer18 Z ratios span 0.82558–1.16621. Its worse response fidelity is retained; near-unity mean partition ratio is not a uniform guarantee. Mean own-row key/value codec errors are 1.091%/0.798% at 17 and 1.287%/0.820% at 18, measured separately from native-trajectory drift. Sparse GPU codes/scales and decoded valid-prefix snapshots exactly match independent encoding of all original prefix rows plus each candidate's actual newly written rows.

| Turn / boundary | Byte17 full-logit relative L2 | Byte17+18 full-logit relative L2 |
|---|---:|---:|
| Recall / before answer | 0.28631% | 0.93255% |
| Recall / consumed EOS | 0.20595% | 0.37011% |
| Summary / before answer | 0.29282% | 0.49685% |
| Summary / consumed EOS | 0.18557% | 0.48228% |
| Count / before answer | 0.60027% | 1.36339% |
| Count / consumed EOS | 0.20306% | 0.67887% |

Adding layer18 worsens every measured full-logit discrepancy here. All complete histories match, so these are same-history state comparisons. They are not directly the previous fixed-span intervention's error: this run compresses a larger initial prefix and all new selected memory.

All four sessions answer `Niko's`, `Mira's badge is red, and Niko's badge is green.`, and `1`, each ending with EOS. Earlier deletion controls also preserved these answers. This study adds no new necessity control and provides no broad arithmetic/free-form-chat accuracy claim. Its evidence is execution closure, represented-response fidelity and actual selected allocation savings.

## Actual allocation and execution cost

At capacity 512, native per-layer FP32 K/V costs 1,048,576 bytes; one byte arena costs 270,336 bytes. All byte layers share 524,288 bytes of float decode workspace and a 4-byte status flag.

| Policy | Dense cache layers | Session + owned cache + decode/status GPU bytes | Saving versus native |
|---|---:|---:|---:|
| Native | 28 | 30,096,896 | — |
| Exact callbacks 17+18 | 26 | 30,096,896 | 0 |
| Byte17 | 27 | 29,842,948 | 253,948 |
| Byte17+18 | 26 | 29,064,708 | **1,032,188** |

The stacked saving is 3.43% of native state allocation and about 0.0166% of model-plus-state allocation. Model weights remain 6,174,857,216 bytes; model allocation including tracked CUDA helper storage is 6,174,873,620 bytes. Steady model-plus-state allocation drops from 6,204,970,516 to 6,203,938,328 bytes. This is exact accounting of successful cudaMalloc request extents routed through the instrumented allocator, excluding CUDA/cuBLAS internal allocations and host metadata. The run verifies return to the model baseline between sessions and zero tracked bytes after final release.

Job [`job-2a28feeb7e5b`](runs/job-2a28feeb7e5b/job.json) takes 32.17 scheduler seconds, 26.20 worker seconds and 16.31 instrumented C seconds. Peak RSS is 4,060.1 MB and process VRAM 6,378 MB. The native control is deliberately allocated earlier and sets the tracked experiment peak; unchanged whole-run peak does not refute smaller candidate steady allocation. No isolated latency benchmark or speedup is claimed.

## Verification and remaining work

Twelve model-free checks pass, including ASAN/UBSAN on the actual transformed runtime's layout, current-row consumption, rollback/abort/replay, codec edge cases and token authority. C syntax passes. [Strict local verification](runs/job-2a28feeb7e5b/verification.json) authenticates 81 sources and 2,978 raw files totaling 38,165,938 bytes, the predecessor/model, scheduler log/report seal and public whole-invocation Saturn receipt. It recomputes the frozen analyzer with zero model calls under the existing strict arithmetic comparison. No portability repair or quality tolerance change is needed in this experiment. Report SHA256 is `5d046f5b56574dafadf4422af9cd855089da1c2df99a8039efb2573db042ab4a`.

Astra independently verifies custody, every saved byte scale/code, token/cursor chains, callback/native/historical parity and allocation sums. Its separate FP64 `fsum` reconstruction of 18 sparse encoded attention contexts differs from actual CUDA by 1.55e−7–5.39e−7 relative error. The standard [evidence sweep](index/job-2a28feeb7e5b/evidence-run.json) and [verification](index/job-2a28feeb7e5b/evidence-verify.json) complete over 41 source-bound observations and five applicable predicates, with no fires, claims or addresses; this is stream coverage, not semantic certification.

The next useful cut is a bounded retention/update experiment: create distinct facts or dependencies that later queries actually distinguish, append under the same ownership contract, then compare retaining all rows, evicting rows, and a source-only merged representation. Retain native and deletion outputs, measure how error changes with context and query direction, and allow changed own tokens to expose the return path. Wider layer coverage and a tiled decode kernel can then address memory and compute separately. Terminal certification is not requested; arbitrary chat sufficiency, fixed-size memory and forward-pass elimination remain open.
