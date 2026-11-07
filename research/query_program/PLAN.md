# Same-cost source-only query prediction

Frozen exploratory protocol. One conditional query image replaces only the old query image on unchanged B128 coordinates. No new source model advances, input dimensions, memory capacity, routing, damping search or target fitting. No terminal claim, serving installation or latency certification.

Model/profile: original Qwen2.5-1.5B-Instruct bundle and C FP32 CUDA/cuBLAS runtime, attention17, capacity512, one C model resident, batch1/CPU1. Reuse verified `query_memory_transfer/runs/job-b94ac2f8a6c9` (report SHA256 `2ad6b50ea1a88fdd73c2d99978bf2ccce826f13c1eb042362e5cfbe2acca17a3`) for exact frozen runtime and four source128 bank arrays. Historical source inputs are the authenticated193 normalized receiver rows from `job-2230480e1c18`, report `c4897a9c69bd8c644e026ac92dcccc959cd65c9e25f885873de459ce53ac51e2`. They are three correlated66/63/64-step trajectories, including eight own tokens each, and B already used all three.

Let Z=XB^T and Y=XWq^T using actual stored source input/basis bytes and the original model's Q weight in FP64. Solve one ordinary full-rank least-squares system `F^T=argmin_C ||ZC−Y||_F²` using NumPy lstsq, no ridge, rank search or target feedback. Store F as FP32. Original F0 is the compiled orthogonal weight projection; F is a conditional query predictor that can infer only omitted responses predictable from Z. It cannot distinguish arbitrary unseen equal-code inputs. Native query bias and RoPE remain unchanged.

Hash the exact bundle body and selected tensor recipe. Emit the fitted image, its hash seal, FP64 source Q targets and compiler report before parsing fresh.json or tokenizing any target. Fresh text is present in the payload and visible in this protocol, but compiler code reads no target. Declare actual compiler inputs and bind their hashes and the image into the invocation receipt. Source target construction costs455,344,128 CPU MAC and feature construction37,945,344 MAC; QR/SVD, diagnostics, hashing and copies are additional paid work. This is one bounded proposal with all outcomes retained, not iterative fitting on test futures.

Thirty own trajectories:

- Three established development cases each run seven policies: NN/full; old CN/full and compact; fitted-query/compiled-K/native-V full and compact; native-Q/compiled-K/native-V full and compact. Their budgets remain8/16/32 per answer.
- Three prospective transfer cases in fresh.json each run NN/full, old CN/compact and fitted-query/compact. No outcome chooses the candidate or changes its factors. This asymmetric panel tests compact-program transfer; it cannot isolate a fresh-case query×memory interaction.
- Execute all24 candidate trajectories, seal their raw arrays, then execute six NN/full references. Start each empty at token0 and follow its own queries, memory, generated tokens/EOS and subsequent user turn. Full suffix and MLP recompute on actual receivers. Retain history matching; identical token IDs do not imply common upper caches.

Nine development NN/old-CN paths must join the preceding run across recorded ports, probabilities, histories, cache-bound endpoints and snapshots. Native-Q controls diagnose query restoration in the actual consumer at a paid cost; they are not assumed to be the best policy or an efficient serving replacement. Earlier native-all compact and deletion results remain historical limitations rather than being repeated for panel symmetry.

Record all four full-logit boundaries, native-reference signed squared-error gains, answers, truth where unambiguous, completion/truncation, per-step own memory reads and selected native/compiled map calls/MACs. Preserve opposite effects and post-divergence own-policy outcomes. Source fit quality and local query error have no authority to suppress downstream paths. Every compact state and read must replay under the declared memory law; this validates the instrument, not fidelity.

On each development maximal common four-policy prefix, qualify equal lower receivers, unchanged K/V histories, paired old/new queries across full/compact and exact compact states. Lift compact weights only in the post-seal audit. The conditional query×memory identity is

`I=Y_fit,C−Y_fit,F−Y_old,C+Y_old,F = [(Ptilde_fit−Ptilde_old)−(P_fit−P_old)] V_N`

in real arithmetic. Native full probabilities are captured. Retain all four finite value-mixing remainders and audit expansion residual; before row33 the reader contrast is numerical-profile only. A mismatch marks that row's formula inapplicable and preserves policy outcomes.

One strict-preflight Beast lease,6144MB RAM/VRAM measured exception from the same backend (previous peakRSS4040.3MB/VRAM6406MB; ceiling6758MB fits16GB device after margin). One additional786,432-byte query image for comparison is negligible relative to residency, but counted. No retries, no accelerator duplicate suite. Expected under3min; stop after this finite panel. Collect the exact job, recompute the frozen analyzer without a model, independently audit and run evidence coverage. Archive bound256MiB with geometry checked before C execution.

Old and fitted CN each use819,200 selected QKV MAC/update,198 whole-token linear calls, the same rank128 coordinate buffer and98,464-byte compact state. No per-program storage or work increase: the fitted image replaces the old one. The comparison experiment retains both image banks and the entire original weights. Native-Q/compiled-K/native-V costs2,981,888 selected MAC, only5.21% below native QKV. Full native receiver generation, other27layers/caches, O/MLPs/readout and selected global cache allocation remain paid. No full-model forward-free or accurate-chat claim follows.
