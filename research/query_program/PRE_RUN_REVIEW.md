# Independent pre-run review

Reviewed the staged `20261107T000349.851332Z` payload before execution. No launch-blocking mechanics or integrity issue found. This is a source-only query-prediction experiment; fit quality, native fidelity and semantic outcomes remain observations, not admission gates. No model was loaded or run for this review.

All 74 staged source hashes independently matched the manifest. The reviewed compiler, driver, worker, launcher, analyzer, audit, plan and fresh-input files matched their working copies. All six staged runtime files matched the authenticated `job-b94ac2f8a6c9` source hashes, and `memory.h` was byte-identical to that predecessor. The four bank hashes matched the prior source128 bank. Source acquisition is bound to report `c4897a9c69bd8c644e026ac92dcccc959cd65c9e25f885873de459ce53ac51e2`; runtime and bank provenance are bound to report `2ad6b50ea1a88fdd73c2d99978bf2ccce826f13c1eb042362e5cfbe2acca17a3`.

## Compiler and visibility

The compiler constructs `Z = X B^T` and the unbiased pre-RoPE target `Y = X Wq^T` in FP64 from the authenticated 193 source rows, unchanged B128, and original layer17 query weight. The tensor orientation, BF16 widening, tensor dimensions/extent and bundle-body authentication agree with the intended map. `lstsq(Z,Y)` returns a 128-by-1536 coefficient matrix; its transpose is stored as the 1536-by-128 FP32 query image expected by C. Solving directly for the image is algebraically equivalent to fitting and folding the baseline residual. There is no target input, rank search, ridge search or dose selection in this compiler.

The full-rank/finite checks protect the declared solve, not scientific quality. FP32 image finiteness is also checked before C uploads it. The emitted FP64 targets, compiler report, image and image seal provide an inspectable source fit. Post-run stationarity and stored-image residual diagnostics do not independently regenerate targets from model weights; the target construction is supported by the frozen compiler plus authenticated weight recipe. It should not be described as an independent second weight evaluation.

One wording precision: the worker hashes every payload source before compilation, including `fresh.json`. Therefore it reads that file's bytes for integrity before the image seal. It parses fresh prompt content and tokenizes targets only afterward, and the compiler reads neither. Use **before parsing fresh prompts or tokenizing targets** for the visibility boundary. Fresh text is also present in the source-visible payload, so this is a prospective operational quarantine, not concealed-text evaluation.

## Runtime and controls

Only the selected query weight image changes between old and fitted policies. Bias and RoPE still execute in the unchanged runtime. K uses the original compiled image; V uses the original native map; O, MLP and subsequent layers consume each path's actual receiver. Fitted Q and old Q use the same B coordinates and no additional serving operation. Native-Q controls compute Q natively, then lazily compute coordinates at K; `coordinates_ready` resets on every advance, and expected coordinate/map counters cover this dispatch.

The six development candidates per case and two fresh candidates per case total 24 candidate paths. All finish and their numerical arrays are sealed before six native references. The seal includes `.f64` source targets. The analyzer independently specifies the same 30-path order and seal membership. Each trajectory begins empty, owns its generated IDs and follow-up history, and compact paths never write selected-layer full KV. The diagnostic histories and membership reconstruction do not serve the reader.

The frozen fresh budgets 8/16/32 match the driver's task-modulo-three dispatch. This dispatch is adequate for this fixed payload, but a future change to case order or budgets must also change the driver. It is not a generic budget interface.

Nine historical native/old-CN endpoints are sufficient controls for the unchanged paths. The asymmetric fresh panel can estimate old-versus-fitted compact-program transfer relative to native full attention. It cannot isolate a fresh query-by-memory interaction. The development full/compact pairs provide that contrast. Native-Q controls are paid diagnostics, not an assumed optimum.

The query-memory algebra has the correct four-corner signs:

`I = fit_compact - fit_full - old_compact + old_full`

`I = [(Ptilde_fit - Ptilde_old) - (P_fit - P_old)] V_native`

The audit qualifies identical lower receivers, K/V histories, paired queries and compact states. Qualification accumulates over previous rows, correctly accounting for historical values. It retains four finite value-mixing remainders and marks mismatched rows inapplicable without discarding trajectory observations. Identical consumed IDs do not make upper caches identical; later logit differences remain own-policy effects.

## Cost and execution envelope

Old and fitted CN both require 819,200 selected QKV MACs and 198 whole-token linear calls. Native-Q/compiled-K/native-V requires 2,981,888 selected MACs and the same 198 calls. The fitted image replaces a 786,432-byte image in an individual program; this comparison job retains both and pays the extra allocation. The 98,464-byte compact state and its recurrence are unchanged. Original weights, other layers and full global KV arena remain allocated.

The compiler separately declares 455,344,128 target-construction and 37,945,344 coordinate-construction MACs. Least squares, diagnostic products, hashing and copies are additional paid work. This is not a total static-cost estimate or a latency saving claim.

The archive preflight accounts for 27,140 bytes per recorded advance, triangular full-reader probability streams for four development full paths and one fresh full path, 120 full-logit arrays, 60 compact snapshots, banks/image/source operands and a 16 MiB allowance. The final emitter additionally checks the actual uncompressed total against 256 MiB. Candidate-seal and collection bounds are consistent with the 30-path geometry.

The launcher requests CUDA, one CPU thread, one C model, physical batch one, strict preflight and no retries. Its 6144 MB RAM/VRAM reservation is explicitly the measured exception from the same backend, with the documented 1.1 ceiling above the prior 6406 MB device peak. The additional query image is small relative to that measured envelope. One bounded lease with a 180-second estimate and finite subprocess timeout is consistent with the proposed assay. No acceleration or semantic-success claim follows from these mechanics checks.
