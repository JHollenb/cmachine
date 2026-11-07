# Independent review of job-bd80dd16c1d5

The same-cost fitted query image produces a small, consistent numerical improvement on this bounded panel, including all three prospective cases. It does not change any old-versus-fitted compact answer or consumed token history. The much larger native-query restoration effect leaves substantial query-formation headroom, but that control is expensive. This supports improving the cheap query predictor before increasing memory; it does not establish general accurate chat or whole-model emulation.

Reviewed immutable report SHA256 `d3f13ffdc2061e5457792ec0947277ef7a68bac0e4685de64bad116400c19f64`, its executed payload, and saved arrays. No model rerun or change to primary evidence was performed.

## Integrity and the numerical verification discrepancy

Independent hashing found no mismatch among all 74 payload sources, 467 recorded raw files, or 365 candidate-seal entries. Recorded raw evidence totals 171,443,331 bytes. All four historical program arrays match their predecessor. Nine historical native/old-CN paths match 126 distinct numerical files, including captured native probability streams. An initial broad-prefix count of 174 included the 48 compact files twice; a second exact-suffix join confirms the distinct count and all byte comparisons. Independent token-authority checks cover 695 generated IDs and all 120 boundary-logit argmax readouts. All 30 per-path linear-MAC ledgers match their dispatched policies.

Re-executing the frozen analyzer reconstructs all 60 compact snapshots byte-for-byte, with zero saved compact-output replay error across 3,024 total own advances. Its common-prefix checks retain the declared key/value/state identities and account for the four finite mixing remainders. These are instrument and execution checks, not fidelity guarantees.

The original strict cross-platform numerical verifier refuses 96 scalar comparisons. They are confined to the new operand-comparison metrics and their duplicate observation entries. The frozen `rel(a,b)` casts `a` to FP64 but leaves `np.linalg.norm(b)` in incoming FP32 for these calls. Linux and Mac FP32 norm reductions therefore give slightly different denominators. This is a diagnostic arithmetic-profile defect, not a raw-array discrepancy or compact-memory replay failure. Source-fit diagnostics and the earlier analysis families recompute within the original verifier policy.

Preserve that refusal and the immutable report. A separate correction should define both operands and denominator in FP64, with independent `math.fsum` checks. A blanket verifier tolerance relaxation is unnecessary. The independent numbers below use double-precision differences and `math.fsum` squared norms; the trends survive this correction. At this review's creation, this diagnosis does not itself mean the unmodified strict verifier passed.

## Source fit and actual-query instrument

An independent solve against the saved FP64 source targets produces the identical FP32 query image, SHA256 `99e6fa43c6fbed8d01a78f2255b7a3ba12d57b49393f0a32392e3c292656e1c6`. The coordinate system has rank 128 and condition number 32.7984. Stored-image source relative Q error decreases from 7.310351% to 5.776989%; normalized stationarity residual is approximately `8.18e-9`. This checks the fit against authenticated saved targets; it is not an independent second evaluation of original model weights.

All three development native-query full controls and all three native-query compact controls have byte-identical initial Q arrays to native execution on the same initial lower receivers. The fitted and baseline programs share B128, compiled K, native V, bias/RoPE, and the unchanged compact-memory recurrence. The compiler changes only one query image, so the contrast identifies the query predictor's effect on this program.

Initial-prompt compact metrics below compare each program with native execution. Entries are old → fitted relative errors, in percent. Query and attention errors pool the initial prompt's actual rows; logit errors use its final readout.

| Case | Query | Attention | First logits |
|---|---:|---:|---:|
| Development recall | 22.137 → 22.053 | 39.542 → 39.179 | 12.083 → 11.781 |
| Development math | 23.119 → 23.005 | 30.602 → 29.953 | 12.713 → 12.357 |
| Development prose | 15.070 → 14.833 | 22.162 → 21.793 | 6.292 → 6.175 |
| Prospective recall | 23.068 → 22.988 | 42.047 → 41.656 | 17.501 → 17.340 |
| Prospective math | 22.545 → 22.427 | 30.751 → 30.083 | 9.037 → 8.658 |
| Prospective prose | 15.150 → 14.903 | 22.120 → 21.626 | 4.648 → 4.569 |

The source fit improves more than the actual target query errors. That gap is a useful limit: source conditional prediction transfers weakly, and omitted target directions remain unavailable. The three prospective cases resemble the development task families; this is limited transfer evidence, not a broad held-out benchmark. The 193 fit rows remain three correlated trajectories.

## Continuation, countertrend and truth

All six old/fitted compact pairs consume identical token histories through both turns. Fitted compact logits are closer to native at 23 of 24 recorded boundaries. The exception is development math after its second answer: 19.359856% becomes 19.364275%. These are repeated boundaries within six conversations, not 24 independent samples. Identical IDs also do not make policy-specific upper caches identical.

Prospective follow-up prefill logit errors decrease from 14.665% to 14.557%, 26.628% to 26.535%, and 22.835% to 22.548%. The first two tasks preserve shared native history at the relevant earlier boundaries; prose diverges from native and later comparisons concern each policy's own accumulated state.

The fitted image changes no compact answer. Color/material recall remains correct. Development math remains `266`, then the incorrect `286` instead of `279`. Prospective math gives `432`, then `414` for both old and fitted compact programs, while native gives the incorrect `254`. That correctness advantage already existed before the fitted query image and cannot be credited to it. Numerical agreement with native and task truth are separate measurements. Prose remains coherent and old/fitted compact prose is identical under the tested budgets.

Native-query controls reduce first compact logit errors further to 7.956%, 9.396% and 4.297%, compared with fitted compact 11.781%, 12.357% and 6.175%. A strong countertrend appears after the development math follow-up: native-Q compact reaches 86.317% logit difference versus 19.360% for old compact; native-Q full similarly reaches 84.224% versus old full 18.861%. Those paths consume different generated explanations, so this is an accumulated own-trajectory difference, not evidence that exact Q is locally worse on one common state. Native-query controls also alter some prose trajectories. They demonstrate first-readout headroom on this panel, not a universal ceiling, monotonic continuation improvement or a cheap serving solution.

## Computation and next interpretation

Old and fitted CN each use 819,200 selected QKV MACs and 198 whole-token linear calls, with the same 98,464-byte compact state. The fitted image replaces the original image for a standalone candidate; this comparison retains both and pays an additional 786,432 bytes. Native-Q/compiled-K/native-V uses 2,981,888 selected MACs, leaving only about 5.21% of native QKV work removed. It should remain a diagnostic.

All paths plus parity checks total 4,712,818,868,224 nominal linear MACs and 604,458 linear calls. Static fitting costs are additional: declared target/feature products, least squares, diagnostics, body hashing and copies. The compiler took 2.04 seconds, driver 42.57 seconds, worker 53.53 seconds, and scheduler result 71.63 seconds; peak RSS was 4010.5 MB and VRAM 6406 MB. These scopes differ and do not establish an end-to-end speedup.

The result advances the narrow question: better source-only query formation can improve a recurrent compact reader without increasing serving work or memory. The gain is small and does not repair answer behavior. A subsequent query-compiler change should target the remaining actual reader sensitivity and preserve this candidate as a fixed baseline. More capacity or a larger memory is not yet justified merely by a nonzero query-memory interaction. Source-only objective changes must remain distinct from target-informed tuning; accumulated development observations cannot become an undisclosed fitting set.

All original model weights, full lower/upper receiver computation, other 27 attention layers and caches, O/MLPs/readout, and the selected full-cache arena remain paid. This is a bounded selected-component program inside a dense model, not forward-pass-free whole-model emulation.

## Saved-logit consumer qualification

An independent saved-logit calculation using FP64 stable log-softmax and `math.fsum` confirms the study's orientation and normalization: `KL(softmax(native) || softmax(candidate))`, in nats over all 151,936 vocabulary entries. Fitted compact improves this native-distribution KL at 19 of 24 boundaries, including 13 of the 16 boundaries where both compact histories match native. The native top token's log probability increases at 21 of 24 boundaries. These are distinct observables from the 23-of-24 logit-L2 trend.

Two clear initial countertrends occur despite improved query, attention and logit L2: development math KL rises from `0.00035655545` to `0.00038287720` nats, and prospective prose rises from `0.02912557464` to `0.03236018302`. The other three KL reversals are smaller: development recall first-answer final (`2.44567e-7` → `2.54240e-7`), development math follow-up final (`1.65777e-5` → `1.74642e-5`), and development prose first-answer final (`6.48358e-9` → `7.45878e-9`). The latter two compare differing native histories and therefore remain operational-policy observations.

The fitted query is a modest improvement under several measures, not uniformly better at the final consumer. Query MSE and unweighted logit L2 cannot substitute for distributional fidelity or task truth. This sharpens the next proposed objective toward the query's response through the actual compiled K and native V, followed by native-consumer evaluation. It does not authorize fitting these saved target outcomes or establish that such an objective will succeed. No new fit or model execution was needed for this qualification.

## Final verification status

The verification issue is now resolved through a separate zero-model CPU replay, `job-7d8b02674ac8`, without relaxing numerical tolerances or changing the primary payload, report or raw evidence. Its canonical analysis matches exactly, SHA256 `8878d174c57dba24a21b2e9a6a86587b2eee18c82d0a0fbffe990bf71c1cb9d5`. The saved primary verification binds all 74 source files, 467 raw files, scheduler/report custody and the Saturn invocation receipt. Canonical replay took 45.15 seconds of CPU worker time and evaluated no model.

The original Mac FP32-denominator refusal remains preserved as a diagnostic. The separate `fp64-query-metrics.json` sidecar supplies 24 stable operand-metric rows; the original reported metrics remain unchanged. The working analyzer now casts both operands to FP64 for future experiments only. These corrections distinguish a reproducible canonical execution from a portable diagnostic arithmetic profile.

The additional saved-only study confirms nine historical paths across 126 distinct arrays plus four bank arrays. All six old/fitted compact pairs have identical normalized receiver, K, V and coordinate streams, and their 24 corresponding memory snapshots match. The fitted query therefore changes reader weights and downstream state while preserving this panel's compact memory contents and token trajectories. These final mechanics checks strengthen the bounded attribution above without adding a semantic or generalization claim.
