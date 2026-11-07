# Source BF16 residence halves matrix storage under a matched consumer

The C experiment now stores all 197 learned matrices in their original BF16 representation and widens them only into one shared FP32 tile. Matched tiled FP32 and tiled BF16 execution produce **byte-identical captured state, logits, caches and tokens** through the five-turn conversation and lifecycle tests. Both independently construct all 28 byte caches from token zero. No new weight quantization, model training or native-state refresh occurs.

Tracked model-plus-state allocation falls from 6,183,704,088 to 3,133,265,432 bytes versus the ordinary FP32/all-byte baseline: **3,050,438,656 bytes /49.33% saved**. Including the earlier all-layer cache reduction, the difference from the original native model/state allocation is 3,071,705,084 bytes, 49.50%. These are successful cudaMalloc request extents, excluding CUDA/cuBLAS internals and host metadata. The run's overall peak includes the full-size controls and is not a compact-only process peak.

This is a storage result with a necessary arithmetic qualification. Tiled GEMV geometry itself changes later logits 1.334–4.147% relative to ordinary full-shape GEMV with identical parameter values and matching token histories. That is not BF16 storage error: the two tiled profiles are exact to each other. Full learned parameter information, dense forward work, linear cache retention and general chat sufficiency remain separate questions.

[Plan](PLAN.md), [implementation](README.md), [report](runs/job-02559c65b72e/report.json), [strict verification](runs/job-02559c65b72e/verification.json) and [Astra audit](REVIEW.md) bind this bounded result to Qwen2.5-1.5B-Instruct, C FP32/pedantic-cuBLAS, B 1 CPU 1 and capacity 512.

## Ownership and numerical controls

The original bundle already contains BF16 matrix bytes. The prior loader widened every tensor permanently to FP32. The new owned-weight API records each rank 2 matrix's input/output shape, storage type and allocated extent. The source-BF16 profile refuses any matrix whose authenticated source dtype is not BF16. All 141 rank 1 norm/bias vectors stay FP32. Together there are 338 tensors, including 197 matrices with 1,543,569,408 elements. The tied input embedding/output head is one registered matrix and is counted once.

An input token decodes its embedding row directly into the session's current residual. Each matvec decodes at most 1024 output rows into a shared 1024×8960 FP32 workspace, then runs the same native-shape tile GEMV as the tiled FP32 control. Both use FP32 inputs and decoded entries, pedantic cuBLAS and the same default stream. The BF16 path makes no extra rounding of the original parameter values. It keeps no permanent FP32 matrix bank. The original source mapping is released before conversation execution.

Three sequential processes load ordinary FP32, tiled FP32 and tiled BF16 profiles. Each uses the same byte-cache policy and independently consumes its own source answer, correction acknowledgement and three later replies. The source/correction/future user inputs are fixed; generated tokens are never supplied from a reference. Each profile performs 182 main transitions, six successful lifecycle transitions and one injected partial token. Cache ownership, rollback, sibling replay and last-layer abort use the existing all-layer contract.

The original FP32 profile exactly reproduces all 258 saved byte-trajectory files from the all-layer experiment: every own K/V/query bulk file, sparse consumer port, full readout, encoded snapshot and replay reference. Tiled FP32 and BF16 match all 258 such files byte for byte, together with semantic driver metadata after separately checking their different allocation fields. All five answers remain `red`, `OK`, `Niko's`, the correct two-badge summary and `1`, each with EOS.

## Actual resource and work account

| Profile | Permanent matrix bytes | FP32 vectors | Weight decode workspace | Model plus cache/state bytes |
|---|---: |---: |---: |---: |
| Ordinary FP32 |6,174,277,632 |579,584 |0 |6,183,704,088 |
| Tiled FP32 |6,174,277,632 |579,584 |36,700,160 |6,220,404,248 |
| Tiled source BF16 |3,087,138,816 |579,584 |36,700,160 |**3,133,265,432** |

Each profile also pays 16,404 tracked model-helper bytes and 8,830,468 cache/session bytes. The latter includes all 28 byte arenas, score scratch, cache decode workspace and status. Thus this source-storage change removes 3,087,138,816 permanent matrix bytes, then pays 36,700,160 shared decode bytes compared with ordinary execution. Host descriptors and evidence capture are additional.

Each profile makes 37,228 logical learned matvec calls, including mechanics. Tiled profiles execute 165,582 physical tile GEMVs. A completed token has 197 learned matvecs and 877 tile calls; the injected last-layer failure stops after 192 matvecs/706 tiles. Embedding lookup is separately counted 189 times. All counts reproduce the declared geometry. No dense learned mapping or learned-linear MAC is eliminated.

Instrumented C times are 10.733 s/11.322 s/9.924 s. They include model loading, the fixed order, audit capture and lifecycle work. No repeated/alternating serving benchmark is performed, so these times establish neither speedup nor a robust ranking. The complete scheduler lease takes 49.19 s, worker 40.63 s, peak RSS 4,242.9 MB and aggregate VRAM 6,386 MB. The full-size earlier controls set the experiment peak; report the smaller BF16 steady allocation separately.

## Changed arithmetic is visible despite exact source storage

| Turn | Tiled versus ordinary before-answer logits | After consumed EOS |
|---|---: |---: |
| Source color |1.3344% |1.5244% |
| Correction |2.2280% |2.4475% |
| Recall |2.3545% |1.6337% |
| Summary |1.4949% |1.4698% |
| Count |4.1465% |2.3905% |

Both tiled profiles give these same relative-L2 differences. Complete histories match ordinary execution. Astra finds an early saved arithmetic perturbation at the first token's Q0: relative error 8.81747e−8, maximum absolute 7.62939e−6. The first K0/V0 are exact; by layer1 some K/V scales differ while codes still match. This supports changed reduction geometry entering the trajectory, but does not apportion later effects between quantization boundaries, attention sensitivity and accumulated state. The [all-layer routing evidence](../compact_all/FINDINGS.md) supplies relevant sensitivity, not a completed causal decomposition of this new profile difference.

Lossless stored parameters therefore do not imply identical machine execution after changing arithmetic geometry. A future alternative can decode a full projection into one shared largest-matrix workspace and retain ordinary GEMV dimensions, paying substantially more scratch. That is a separate parity/cost experiment; it is not needed to validate the present matched-profile storage claim.

## Verification and scope

Eleven local checks pass, including actual-runtime ASAN/UBSAN after owned-embedding dispatch, exhaustive widening of all 65,536 BF16 bit patterns (with finite-pattern classification), public three-command panel receipt binding and token-authority tests. C syntax passes with inherited unused helpers; Python syntax passes. Astra reviews both deterministic source transforms before execution and the complete saved evidence afterward.

Strict verification checks 83 source files and 789 raw files totaling 96,501,721 bytes, the source model, the original all-layer report and its qualified saved-only reanalysis, scheduler log/report seal, original 258 raw joins and the public three-process panel receipt. It recomputes the frozen analyzer with zero model calls. Report SHA256 is `1b09e1ebb205c5dac1dc5b08c3684f3e5f08082606c00774e90f6f942d97d85d`. Astra independently confirms the source/raw seals, exact controls, 68,096 tiled snapshot row instances and resource/counter sums. The [evidence sweep](index/job-02559c65b72e/evidence-verify.json) covers 35 observations and two applicable predicates, with no fires, claims or addresses. The scope is experimental storage and execution; installed serving is unchanged.

This addresses resident weight cost without discarding the learned information. The [bounded-retention design](../compact_all/RETENTION.md) addresses the remaining dynamic-memory contract: active rows, score/decode workspace and rollback history must all be bounded together, and delayed queries must test what was lost. The layer0 block-key diagnostic is an independent accuracy lead. General compact sufficient chat, fixed-size exact arbitrary recall, forward elimination and measured serving speedup remain unestablished. Terminal certification is not requested.
