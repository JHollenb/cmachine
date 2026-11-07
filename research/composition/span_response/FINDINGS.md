# A frozen span program answers queries it did not see

Eight memory-only key/value centroids per KV head approximate a correction span's response to two later questions without consulting that span's original K/V at the substituted read. The serialized program occupies **16,480 bytes**, compared with **69,632 bytes** for the selected native span. It improves the measured native attention context and first full-vocabulary logits over one centroid and deletion in both questions. This is prospective reuse of a local memory representation; the experiment retains the complete original model and cache and substitutes only one attention read per branch.

The [prospective plan](PLAN.md), [primary report](runs/job-524febce43e5/report.json), [strict verification](runs/job-524febce43e5/verification.json), [saved-only omission audit](runs/job-524febce43e5/omission-audit.json), and [independent Astra review](REVIEW.md) preserve construction order, native controls, signed errors and custody. This follows the [conditional native attention account](../attention_read/FINDINGS.md). The new program is constructed before the actual later queries exist, rather than evaluating an identity from an already captured query and original memory.

## Specimen, construction and actual consumer

The specimen is the pinned Qwen2.5-1.5B-Instruct C FP32/cuBLAS runtime: 28 layers, hidden width 1,536, 12 query heads, two KV heads and head width 128. Source formation and the correction are native throughout; no B3 MLP approximation participates in this cut. The authenticated source ends at token cursor 59, and the correction plus the runtime's own acknowledgment and EOS ends at 93. The fixed memory span is therefore C=[59,93), containing 34 post-RoPE key/value rows per KV head at layer17.

At cursor93, [the C constructor](span_bank.h) receives only these source-span keys and values. It uses deterministic Ward clustering in key space, with FP64 merge calculations and fixed index-order ties. A cluster stores its count, mean key and associated mean value, rounded to FP32. One-cluster, eight-cluster and exact34 programs are serialized and fingerprinted before [the driver](driver.c) opens either future question's token file. Exact34 retains native row order with count one. Membership masks are audit metadata; the bank/query-only prediction function does not receive them or the original span. The construction input copies are zeroed afterward.

Two independent later prompts ask:

- “Whose badge became green? Reply with just the name.”
- “In one sentence, summarize the current badge assignments for both people.”

Each question's native prefix is computed once through the penultimate question/template token. Five children share that parent and the actual final token's lower-layer computation: native, exact34, one, eight and delete. Only the final question token's layer17 attention read changes. All other layers and all subsequent generated tokens use native equations on their branch's own state. Every emitted token, including EOS, is consumed.

The selected candidate read packs the dense complement plus virtual bank entries into a private GPU table. It never reads original C keys or values. The complement includes unchanged source history and the new question, including the current allowed row. Its cost remains paid. One global softmax normalizes both the complement and bank, with log(cluster count) added to each virtual score. Separately normalizing the bank and adding its conditional mean would implement a different computation.

The original cache remains available for native controls and subsequent native generation. Thus even matching own answers cannot establish a continually compressed chat loop. This is a common-parent, single-read test of response transfer and its actual downstream consumer.

## The compact response and its mathematical limitation

For one head, let a=q/sqrt(128). A cluster j contains n_j entries with means kbar_j and vbar_j. The bank predicts

\[
\widehat Z_C(q)=\sum_j n_j\exp(a^\top\bar k_j),\qquad
\widehat N_C(q)=\sum_j n_j\exp(a^\top\bar k_j)\bar v_j.
\]

The actual C predictor uses the pinned FP32 attention scale promoted to FP64, a log-sum-exp gauge and FP64 accumulation. The primary saved output is log Z and N/Z per query head. It executes before the Python audit consults original memory; no refitting, repair or fallback occurs.

For exact real cluster means, convexity of exp gives a query-uniform result. Refining the single cluster into its eight descendants increases the denominator estimate:

\[
Z_1(q)\le Z_8(q)\le Z_C(q).
\]

This holds for every q and is a useful one-sided prediction. Stored FP32 means and native score arithmetic require their own numerical accounts. There is no corresponding ordering for the vector numerator, conditional span mean, globally normalized attention context or logits. Values can change sign and correlate with within-cluster scores.

Let the complement stay exact and let c denote the native full attention context. With delta defined as predicted minus exact,

\[
\widehat c-c=
\frac{\delta N_C-c\,\delta Z_C}{Z_{\bar C}+\widehat Z_C}.
\]

The primary analyzer retains both vector terms, their signed cross term and actual CUDA discrepancy. A small full-context difference can reflect the complement's contribution, opposing numerator/normalizer terms and the native consumer's response. It does not establish accurate recovery of the isolated span or a general contraction theorem.

## What the native consumer does

Relative L2 differences from native execution at the selected read and its first logits are:

| Later question | Program | Attention context | MLP17 return | Layer17 exit | First full logits |
|---|---|---:|---:|---:|---:|
| Recall | One | 14.635% | 8.508% | 4.236% | 3.063% |
| Recall | Eight | **7.648%** | **4.569%** | **2.256%** | **1.592%** |
| Recall | Delete C | 16.255% | 9.569% | 4.592% | 3.640% |
| Prose | One | 8.267% | 4.508% | 1.593% | 1.334% |
| Prose | Eight | **4.910%** | **2.779%** | **0.971%** | **0.783%** |
| Prose | Delete C | 11.538% | 6.221% | 2.260% | 1.733% |
| Both | Exact34 | 0 | 0 | 0 | 0 |

Exact34 matches native at every captured port, native probabilities, current KV17/18/27, full first/final logits, complete valid-cache hashes and own generated IDs. This establishes that the packed read can reproduce the original numerical computation at native geometry. The compact programs also change reduction geometry; their measured differences retain execution effects rather than assuming an ideal implementation.

All five arms produce the same own answer in each question, including EOS:

- Recall: `Niko's`.
- Prose: `Mira has a red badge, and Niko has a green badge.`

The retained exact-string recall diagnostic compares against `Niko` and is false for every arm. This is a formatting difference in a response that identifies Niko, not evidence of a bank-specific recall failure. No raw flag is changed. Deletion removes C from one layer17 read, while other layers and later tokens retain original memory access. Its matching output makes this behavioral panel nondiscriminating about necessity or semantic quality.

Final post-EOS arm/native logit comparisons have the same own histories and clocks in this run. Recall differences are one 0.13639%, eight 0.07354%, delete 0.13496%; prose differences are one 0.05784%, eight 0.03390%, delete 0.08735%. Retain the recall countertrend: one centroid is slightly worse than deletion after EOS despite being better at the initial read. No monotonic continuation theorem follows from the initial improvement.

## The isolated span remains substantially approximate

Mean per-head response measurements are:

| Question | Bank | Predicted/exact span Z | Relative span N error | Relative conditional span-mean error |
|---|---|---:|---:|---:|
| Recall | One | 0.3062 | 82.868% | 69.411% |
| Recall | Eight | **0.6210** | **48.715%** | **34.594%** |
| Prose | One | 0.3366 | 79.664% | 62.770% |
| Prose | Eight | **0.6352** | **47.427%** | **32.722%** |

Each row averages 12 correlated heads within one prompt. There are two prompts, not 24 independent generalization examples. Eight-centroid Z ratios range 0.3718–0.8450 for recall and 0.5111–0.8498 for prose. All 24 ideal-mean denominator orderings hold; stored-versus-ideal log Z differences are at most1.61e-7. Eight improves N, conditional span mean and full context over one at every observed head; this is an observation, not a vector monotonicity law.

Native attention assigns C an average12.633% probability mass in recall and6.929% in prose, ranging1.066–28.992% and1.680–18.143%, respectively. These denominators help interpret why the entire context and logits can look much closer than the isolated span response. They do not convert the error ratios into semantic responsibility or guaranteed attenuation factors.

## Factoring the missing computation

The [post-hoc audit](audit.py) uses saved original K/V and queries only after primary predictions are sealed. It fits nothing and calls no model. For each ideal cluster, write t_i=a^T(k_i-kbar_j) and take a uniform average E_j over its members. Exactly,

\[
Z_C-\widehat Z_{\rm ideal}
=\sum_j n_j e^{a^\top\bar k_j}\left(E_j[e^t]-1\right),
\]

\[
N_C-\widehat N_{\rm ideal}=D+C_v,
\]

\[
D=\sum_j n_j e^{a^\top\bar k_j}\bar v_j\left(E_j[e^t]-1\right),\qquad
C_v=\sum_j n_j e^{a^\top\bar k_j}E_j[e^t(v-\bar v_j)].
\]

D is lost score dispersion multiplying the cluster's mean value. C_v is finite exponential-score/value dependence. It contains more than linear key–value covariance. For the actual stored primary program, add R=Nhat_ideal−Nhat_stored, so the measured omission is D+C_v+R. R covers centroid storage and CPU prediction arithmetic; the native CUDA remainder belongs to the separate consumer account.

For gauge-independent diagnostics, divide every head's numerator terms by that head's exact span Z, then concatenate heads. In these value units:

| Eight-centroid question | Omission norm | Dispersion norm | Dependence norm | Storage/CPU remainder norm | 2 inner product(D,C_v) |
|---|---:|---:|---:|---:|---:|
| Recall | 12.0463 | 9.3365 | 8.5461 | 1.19e-6 | −15.0946 |
| Prose | 11.0926 | 8.6212 | 6.8643 | 8.60e-7 | +1.6019 |

Both omitted terms are substantial. They partly cancel in recall and mildly reinforce in prose. Their norms are not additive error shares or measures of full-attention importance. The vector account closes below1.20e-15 in the saved-only calculation. Astra independently recomputed the same terms using a different common-gauge exponential form, with vector closure below5.41e-16 and denominator closure below3.44e-16.

The working inference is specific: a bank of counts and first moments loses two identifiable pieces of the response law. Correcting only the denominator would leave the value-dependence term. A second-moment or higher-moment program is a plausible next source-only approximation, not an established sufficient statistic. Full key/key and value/key matrices are costly at width128 and can exceed this short span's original payload, so a proposed improvement must count its storage before running it.

## Resource and evidence account

| Representation | Count/key/value payload | Serialized bytes including header | Fraction of native span bytes |
|---|---:|---:|---:|
| Native 34-row K/V | 69,632 | 69,632 | 1 |
| One | 2,056 | 2,088 | 0.02999 |
| Eight | 16,448 | **16,480** | **0.23667** |
| Exact34 | 69,904 | 69,936 | 1.00437 |

The eight program is 4.225× smaller than the selected native span representation. This does not describe total process memory. The prototype retains full original weights/cache, all controls, 88,408 additional GPU bytes for bank arrays and 1,048,576 GPU bytes of private packed K/V scratch. Selected span QK/value multiply-accumulate counts fall from 104,448 to 24,576 for eight; complement work, packing and all dense model projections remain. There is no measured serving latency improvement or actual global cache removal.

The single finite CUDA lease, `job-524febce43e5`, completes in23.64scheduler seconds; worker time20.92seconds, instrumented C driver12.78seconds. Construction/export accounts for about0.00210seconds, not a standalone kernel benchmark. Peak RSS is4,083.3MB and VRAM6,378MB. The declared CUDA FP32 run uses batch1, one CPU thread and capacity512, with strict preflight and no automatic retry.

Strict verification authenticates72payload/source files and402raw files totaling16,605,193bytes, scheduler log/report bindings, predecessor identity and the public whole-invocation Saturn receipt. Executed analysis recomputes with the strict comparator; no numerical tolerance policy was loosened. Fifteen model-free checks include the actual C constructor, partition and exact34 controls, ideal Jensen ordering, global normalization, original-input overwrite invariance and token/import authority. C syntax checking passes with three inherited unused helpers. A local staging syntax typo was corrected before the only submitted model job; the earlier local stage remains.

The [evidence sweep](index/job-524febce43e5/evidence-run.json) and [verification](index/job-524febce43e5/evidence-verify.json) complete with42source-bound stream rows, seven applicable predicates and no fires, claims or addresses. This is stream coverage, not semantic certification. Report SHA256 is `1bc0692bc3a2243dc4031387ac152bd9b8d98713ef0931251fb6c4ddcc7f93fa`. The omission audit preserves its source and dependencies separately and leaves primary artifacts immutable.

This cut establishes a bounded prospective local response approximation. Ongoing compact sufficient chat still requires repeated use on its own evolving queries, an update/composition rule for changing memory, physical removal of superseded K/V, wider consumers and measured resource benefit. None is inferred from unchanged answers in these two branches.
