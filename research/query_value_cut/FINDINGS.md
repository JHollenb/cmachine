# Both the query/key reader and the value payload matter

November 6, 2026. The next causal cut improves fidelity to the native consumer in both restoration directions. Keeping native values while compiling Q/K improves all three initial full-logit readouts; keeping native Q/K while compiling values also improves all three. Neither operand family is the sole defect. The actual attention interaction is material, partly cancels the separate errors, and closes against captured native probabilities with an explicit finite mixing remainder.

The practical local tradeoff favors testing compiled Q/K with exact native values next: its selected QKV work remains 73.96% below native, while its first full-logit differences improve from 13.84/15.30/7.09% to 11.76/11.56/6.43%. This is an exploratory operating point on three development conversations, with remaining substantial discrepancy. It is not a promoted serving mode, a semantic-quality improvement or a compact whole-model successor.

## Executed boundary

Beast job [`job-27a597ebd5f8`](runs/job-27a597ebd5f8/report.json) uses the exact original Qwen2.5-1.5B-Instruct C FP32/cuBLAS numerical profile and layer 17 runtime from verified predecessor `job-c7334bbc5bb8`. It reuses that run's four frozen source 128 basis/image arrays byte-exactly. No new source acquisition, fitting, calibration forward or image-compilation forward occurs. Historical source acquisition and compilation remain paid.

Four policies form the QK×V factorial with full selected K/V memory: NN is native; CC compiles Q/K/V; CN compiles Q/K and retains native V; NC retains native Q/K and compiles V. Three additional policies retain native-QKV/bounded-memory, compiled-QKV/bounded-memory and zero attention. Each of 21 paths starts from empty state, consumes its own emitted IDs including EOS, and appends exact later user/header tokens. All 18 candidate paths seal before the 3 native reference paths.

Mixed policies call their actual native maps on their actual receiver. They are paid restoration ceilings and local efficiency/fidelity candidates. Original bias/RoPE, GQA mapping, O projection, receiving MLP and all later consumers are unchanged. Dense lower layers still form the receiver, and the full model bank and other 27 cache layers remain paid.

The native wrapper control matches 16 tokens' complete logits and final valid K/V byte-exactly. The new value-only compiler computes fresh coordinates lazily at V after native Q/K; counters confirm one coordinate projection per factored token and no cross-token coordinate reuse. [Prospective plan](PLAN.md), [design advice](ADVICE.md), [pre-run review](PRE_RUN_REVIEW.md), [code](driver.c).

## Local restoration and the actual later reader

All entries below use the same initial consumed history. The selected normalized receivers and expected Q/K/V and probability pairs match byte-exactly. Local attention errors pool the entire initial prompt; first full logits are the later recurring-policy outcomes after that prompt, including each policy's own historical upper-layer state.

| Case | CC attention | CN/native values | NC/native QK | CC first logits | CN first logits | NC first logits |
|---|---:|---:|---:|---:|---:|---:|
| Recall |46.77% |35.88% |35.63% |13.843% |11.761% |11.110% |
| Arithmetic |36.99% |27.32% |30.11% |15.296% |11.562% |12.311% |
| Prose |24.48% |16.85% |19.29% |7.087% |6.432% |4.418% |

Each restoration improves local attention and the first full-logit discrepancy in all three cases. Their ranking is context dependent. Prose gives a clear consumer reversal: CN is closer locally and slightly closer at the pooled selected layer exit, yet NC is closer at the actual full readout. This establishes that the local error metric does not order complete policies. It does not isolate the receiving MLP versus later upper-cache history as the cause of that reversal.

At follow-up prefill, exact consumed IDs still match across all four full-memory policies in recall and arithmetic:

| Case | CC | CN/native values | NC/native QK |
|---|---:|---:|---:|
| Recall |16.324% |11.975% |13.475% |
| Arithmetic |26.006% |21.359% |19.230% |

Both restorations remain favorable there, with their order changing. Later prose-prefill differences are 35.309%/29.819%/25.058%, but own generated histories have diverged, so those are trajectory outcomes rather than controlled same-cache restoration effects. No single current native map repairs accumulated upper state.

Later endpoints preserve countertrends. After the divergent arithmetic explanations, final logit differences are 64.69% for CC,18.86% for CN and 121.13% for NC. After the divergent first prose answer, CN is worse than CC at 9.68% versus 8.08%. These whole-policy outcomes rule out a blanket claim that every later readout improves; they do not isolate a current-reader defect on one shared cache.

All seven policies answer colors blue then green. All answer 14×19 as 266 and then add 13 as 286 instead of 279. The arithmetic explanations are budget truncated. Native makes that same arithmetic error. Prose completes its two turns; native-V produces a shorter notebook explanation, while native-QK retains the compiled first sentence but changes its example. Deletion also produces coherent prose and the same basic answers. These examples do not establish semantic necessity or reliable free-form chat.

## The observed interaction has an exact conditional form

For the actual attention matrices and value histories,

$$
I=a_{CC}-a_{CN}-a_{NC}+a_{NN}=(A_C-A_N)(V_C-V_N).
$$

The common-history prefix is 94 positions for recall,85 for arithmetic and 48 for prose:227 selected receiver positions, not 227 independent tasks. Initial prompt lengths are 56/44/43. The four-way factorization is restricted to these exact shared histories, with all expected paired operands and probabilities byte-exact. After the first consumed-token difference, only the operational policy contrast remains claimed.

Initial interaction norms are 20.65%/17.11%/7.72% of native attention norm. The QK/value main-effect cross terms are positive, while the interaction has negative inner products with both main effects in all three initial prompts. It therefore cancels part of both errors in the combined program. Norms cannot be added as causal shares, and removing one approximation also changes this cancellation.

The runtime records the actual native FP32 softmax probabilities for all four full-memory policies. Saved-only FP64 mixes of those probabilities and actual values reproduce the bilinear account. Before accounting for native mixing, the maximal-prefix interaction-law discrepancies are 9.10×10⁻⁸/8.62×10⁻⁸/8.17×10⁻⁸relative to native attention. Retaining all four native value-mixing remainders reduces the per-row account remainder below 4.83×10⁻¹⁶with NumPy; Astra's independent `fsum` account closes below 9.1×10⁻¹⁷on its aggregate account. These are different accumulation/aggregation profiles, with the same finite law. This is captured-probability closure, not a replacement of native probabilities by an independently recomputed softmax. [Mathematics](MATH.md), [frozen analysis](runs/job-27a597ebd5f8/payload/analysis.py), [Astra audit](REVIEW.md).

The four-corner identity does not predict the receiving nonlinear MLP or later full logits. Their interactions are measured directly, preserving signs, vector norms, cross terms, histories and native-scaled finite remainders. Mechanical replay and algebraic closure do not certify semantic interpretation.

## Work and memory comparison

| Selected policy | QKV MAC/update | Selected reduction | Whole-token linear reduction |
|---|---:|---:|---:|
| NN/native |3,145,728 |— |— |
| CC/compiled both |458,752 |85.417% |0.1741% |
| CN/native values |819,200 |73.958% |0.1507% |
| NC/native QK |2,981,888 |5.208% |0.0106% |

CN adds 360,448 MAC over CC; NC adds 2,523,136. Restoring native values therefore preserves much more of the selected projection saving than restoring native Q/K. The initial consumer comparisons identify a local cost/fidelity tradeoff; they establish neither a universal optimal policy nor a speedup. Each factored policy makes 198 learned-linear calls versus 197 native. Native full-token learned-linear work remains 1,543,569,408 MAC before attention/scalars.

The single frozen bank occupies 1,835,008 numeric GPU bytes plus 512 coordinate bytes. Original weights stay resident, including maps bypassed in an arm. Each compact reader still uses 98,464 host bytes, and its complete global selected-cache arena remains allocated but zero/unread. The 12 full-policy probability arrays are diagnostic O(n²)storage, with a 6,303,744-byte temporary capacity 512 host recorder, freed between paths. This is not compact serving state or an additional cache available to the reader. Full ports, snapshots, logits, hashing and CPU/GPU transfers are paid evidence work.

The one-model, batch 1, one-CPU thread, strict-preflight/no-retry lease reports 54.02 seconds elapsed through mrun; its scheduler start/finish interval is 52.33 seconds, worker 43.36 seconds and C driver 35.00 seconds. Peaks are 4127.1 MB RSS/6380 MB VRAM. Per-path time is instrumented and fixed-order, with slow native-reference outliers; it does not measure serving speed. The physical learned-linear ledger pays 423,764 calls and 3,308,603,015,168 nominal MAC, including 32 native-wrapper-control advances. All 2113 panel token advances and 489 generated-token choices are retained.

## Evidence and what changes in the design

The immutable report SHA 256 is`da86f60e282cfdf93cd72dffabd98c91cabb7bfb9c3b46450421cc08ca207052`. [Strict verification](runs/job-27a597ebd5f8/verification.json) authenticates 68 sources,308 raw files totaling 116,210,637 bytes, scheduler report/log/raw joins, a whole-invocation Saturn receipt and exact offline analysis recomputation. The 254 candidate-seal entries precede native release. All 24 compact snapshots and their own read outputs replay exactly.

The [historical join](runs/job-27a597ebd5f8/study.json) authenticates 15 repeated endpoints across 204 arrays, full histories, cache-bound turn records and all four frozen program arrays. The five repeated policies per task preserve the earlier result exactly. This qualifies the new dispatch/probability instrument while leaving the three conversations as repeated development evidence.

The isolated standard evidence run and verify return 0 over 2424 rows with six applicable predicates, no fires and no promoted claims. Scope remains `target-streams-only`, not semantic or terminal certification. The saved-only precursor, pre-run review, completed results and later interpretation remain separate; original payload/raw/report bytes are not rewritten.

The design changes in three ways. First, factor addressing and value content as explicit independently testable interfaces; raw V error did not identify a sole V bottleneck. Second, retain native-V/compiled-QK as a simple candidate for the next memory-response experiment, avoiding a router or blanket increase in rank. Third, select future source directions through their actual reader/consumer response rather than weight energy or one local reconstruction norm. For small score perturbations, δa≈Σ_i p_i(v_i−a)δs_i already shows why value geometry weights addressing errors.

That candidate has not yet been tested with bounded memory; the current compact arms use native-all or compiled-all QKV. Do not import its full-cache benefit into a centroid-memory claim. Dense receiver formation, the complete other-layer transition, physical cache replacement and mutable-state checkpoint ownership remain open. The next smallest scientific cut is the native-value/compiled-QK memory response and its missing query-sensitive moments; a later whole-model program must also generate the receiver it currently receives from native lower layers.
