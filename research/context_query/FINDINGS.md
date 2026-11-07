# A source-derived query transfers, while the next layer loses a binding

The updated layer17 query improves mean own-history conditional native-distribution KL in all four fresh conversations. The mean drops from 0.0271048724 to 0.0234218206 nats, 13.59%. Both prompt lengths and both task families contribute. This is a bounded trend over two semantic tasks and two context lengths, not four independent task families or general chat closure.

The next-layer extension is an informative countertrend. A source-only rank128 basis compiled with layer18's original Q/K weights raises mean KL to 0.0493900722 with full layer18 memory. Both inventory first answers now wrongly place the whistle in amber rather than silver. Adding prototype18 memory gives mean0.0518231336, worsening three cases and partly compensating in the long explanation. These observations do not isolate query versus key error or prove that another compact representation cannot work.

## Shared runtime integration

`QwenQkvProgram` now encodes the actual learned-RMS receiver once per attached layer and dispatches selected Q/K/V maps through immutable, model/layer-qualified images. Original biases and RoPE follow exactly once. Multiple layers can attach different programs; physical counters are not rewound. Programs cannot change during an established history.

`QwenLexicalProgram` compiles finite first-layer pre-RoPE Q/K/V atoms from the original native primitives. Missing IDs refuse before input mutation; table load verifies model, shape and contents. The real-model check compiles15 IDs, pays47,185,920 linear MACs and122,940host bytes, then reproduces16 steps of full logits exactly with131,072transfer bytes. This is subset qualification, not a complete vocabulary table.

The prototype memory and streaming covariance VJP now live in `src/prototype_attention.h`. The model-free lane qualifies the VJP and multilayer full-rank parity under sanitizers. The real-model integration replays all420 archived compact-source clocks through the new public interfaces: Q/K/V/attention bytes and420 top IDs match. The experiment's prototype backend refuses checkpoint restores; it owns no snapshot stack. Image-only Saturn restoration is a separate exact contract. No new serving default or installed release is enabled here.

## One reversible query direction

The compiler uses the prepared native-Q/actual-compact-memory source targets: three compact-generated histories,162 answer decisions and equal source/turn weighting. The basis and keys remain frozen. It differentiates the counted reader response, fits one ridge-smoothed image direction, and evaluates four signed doses. No new dense model calls are needed to form the direction from these retained operands. Source feedback then uses actual C own-token32+32 continuations and independently built native same-ID shadows.

| Dose | Same-memory source response loss | Source own-history future KL |
| --- | ---: | ---: |
| 0 | 17.337201 | 0.0299529843 |
| −1 | 38.673901 | 0.0415120129 |
| 0.5 | 10.831109 | 0.0255640469 |
| 1 | 8.133182 | 0.0196064384 |

The source-only controller selects dose1: 53.09% less local response loss and34.54% lower source future KL. All three source means improve at dose1. The negative direction worsens the aggregate but slightly improves the arithmetic source; it is not uniformly harmful. All four image proposals and exact Saturn parent restorations remain in evidence. The imported transaction helper labels index0 as the accepted parent during proposal generation; the later `selection-seal.json` is the sole authority for the selected serving image.

Selection seals before fresh text is parsed/tokenized. Layer18's source basis is then derived from421 actual receiver rows visited by the selected program's own source conversations. It retains95.22% of their squared receiver energy. The resident model compiles recipient-specific Q/K images with256 original-weight matvecs,352,321,536paid MACs. Layer17's images are not copied across layers. Fresh input parsing follows both seals.

## Fresh conversations and subsequent writes

All paths generate their own tokens and consume them, including EOS, before the followup is ingested. Original receiver formation, V, other layers, MLPs and readout remain dense. Prefill KL is sampled every32 supplied-token indices; future KL includes every answer decision before its chosen token is consumed. Some short followup prefill intervals have no sample after the duplicate-EOS skip; their missing per-turn prefill statistic stays null. Future coverage is complete.

| Fresh conversation | Frozen17 | Updated17 | Updated17 + Q/K18/full18 | Updated17 + Q/K18/prototype18 |
| --- | ---: | ---: | ---: | ---: |
| Inventory128 | .02387566 | .02088936 | .03636589 | .03743205 |
| Explanation128 | .02525375 | .02219389 | .04547816 | .04655750 |
| Inventory1024 | .02551361 | .02424161 | .06378876 | .08400733 |
| Explanation1024 | .03377647 | .02636242 | .05192747 | .03929566 |

The query update improves conversation means by12.51%,12.12%,4.99% and21.95%. Mean first-turn KL falls .01957366→.01550168; mean followup KL falls .03463609→.03134196. Both inventory followups worsen slightly (.02543641→.02564151 and .02872130→.02950979). Better aggregate fidelity is not uniform contraction through later writes.

Native and both layer17-only policies preserve the compass/violet and whistle/silver facts in each first inventory reply, then put the whistle in amber after the update. Both later-layer policies already put the whistle in amber before the update at both lengths. Their later answer happens to match the new requested location; that agreement does not establish correct preservation of the earlier binding. All inventory replies terminate. The explanation first replies reach the32-token cap, including native, while their followups terminate. Complete first explanations are not established.

Each policy is compared with a native shadow on its own actual IDs. After answer divergence, the conditional domains differ. These comparisons support a policy trend, not an isolated same-state causal attribution or an improvement in every semantic fact.

## Costs, verification and remaining gap

Serving selected Q/K/native-V costs819,200 learned-linear MACs per layer versus3,145,728native selected QKV, a73.96% boundary reduction. This saves only0.1507% of original whole-token nominal linear MACs per layer. The query update adds no serving maps, coordinates or memory. Two compiled layers save0.3014% nominal whole-token linear MACs and add one matvec dispatch per selected layer. Each prototype carrier is98,464bytes and each host reader invocation transfers14,336bytes. Original parameters remain resident. Timing is instrumented; no serving speedup follows.

The finite job runs29,900physical token advances in348.09C seconds and354.99worker seconds, one resident C model, one lease, CPU1/batch1. Scheduler peaks are4033.6MB RSS and6472MB VRAM. The generic HF planner metadata reports CPU/batch16/threads4 despite the explicitly declared and observed C CUDA path; the measured8GiB custom-executor reservation governs admission. The existing typed custom-executor envelope request was upvoted with this discrepancy.

Verification authenticates100payload sources and490raw files, four exact declared image updates/restorations and64turn records. All2,054 sampled full-distribution comparisons were independently computed in C/Python in-worker; maximum discrepancy is1.23e−12. A separate saved-only audit checks128retained full-logit pairs,64prototype snapshots,420exact historical reads, source response losses and layer18 basis geometry. Standard evidence run/verify both return0 on the curated64-row stream, with `target-streams-only` coverage and no automatically recognized scientific receipts or addresses. Dedicated audits supply the numerical authority.

Report SHA256: `d97a8dedb3a3e187e958c56ecf81bf59d8f2b45f3b69c4c27a76925507d9e9e2`.

The constructive result is an economical contextual query updated against actual compact histories and transferred, without refitting, into longer fresh two-turn histories. Closure across remaining layers is still open. The smallest next causal cut holds a layer18 parent fixed, crosses compiled/native Q with compiled/native K under full memory, then follows the earliest wrong-binding write. Only after that attribution should we fit a source-only consumer-weighted layer18 query or extend its code. A larger receiver-energy basis by itself would not identify which relationship the consumer needs.
