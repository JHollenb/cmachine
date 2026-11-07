# A useful query and a compatible historical consumer

The requested six-branch256-token drawer cut was already executing as `job-935667de6e2f`. We reused its completed immutable evidence, including its native-parent and1024-token controls, rather than submit duplicate model work. This directory adds independent saved-only verification and a source-only compiler target. The [primary results](../../../artifacts/qwen-relational-reader-cut-20261106/RESULTS.md) and [blog](../../../obsidian/blog/2026-11-06-185301-the-right-query-can-rescue-a-memory-that-an-exact-read-cannot.md) describe the original experiment.

## What the requested branches show

On the fixed compact parent, with common native V and the unchanged native suffix:

| Current Q / reader | Complete first answer | Willow−Cobalt first-token margin | KL to native initial distribution |
|---|---|---:|---:|
| Native / full native | Cobalt | −0.168880 | 0.661030 |
| Native / full compiled keys | Willow | +0.536739 | 0.192457 |
| Native / existing bounded memory | Willow | +1.909513 | 0.019641 |
| Compiled / full native | Cobalt | −5.352301 | 4.257546 |
| Compiled / full compiled keys | Cobalt | −4.095587 | 3.187857 |
| Compiled / existing bounded memory | Cobalt | −3.476372 | 2.681219 |

The paid native query with the unchanged98,464-byte reader restores Willow and reduces initial KL99.27%. The existing1024-token contrast shows the same rescue, with98.41%lower KL. Restoring full native keys does not rescue the compiled query in either compact parent. These are two supplied, related diagnostic contexts, not held-out generalization samples.

An exact current native read still fails at256tokens under compact history. Under native history, that exact current read answers Willow. Fixed-current normalized receiver, attention and completed layer17residual are byte-identical across the two parents; different earlier K/V in later layers remain consequential. This localizes the initial contrast to the historical consumer above the repaired layer, without identifying a particular later layer or cache row. Later cross-parent continuations also differ in recurring policy, so their comparison has a broader interpretation.

All compact-parent followups retain obsolete tape/Willow after the user moves it to Cedar, including repaired first answers. Untouched native branches also miss that move. Five native-parent branches initially answer incorrectly and later give the corrected relation. Thus initial native fidelity, initial task truth and subsequent task truth must remain separate. All second responses hit32tokens.

## What changes in the compiler design

Keep the serving image, rank, compiled keys, native values and memory fixed for the next bounded query proposal. The useful new target is native-Q response **through the actual compact memory**, rather than automatically demanding full-native attention. An approximate read can compensate for a changed historical consumer; local native-vector fidelity is not always its best action. Native Q is a paid source teacher and is not a demonstrated global optimum or cheap serving computation.

[response_targets.py](response_targets.py) constructs that target from the three original autonomous source histories in `job-47cc27324445`, excluding every drawer operand. Its [manifest](source-targets-v1/manifest.json) binds420exactly replayed actual reads,162answer decisions, retained coefficient rows, source hashes, teacher vectors, baseline responses and full-native responses. Equal source/turn/decision weights sum to one. No fitting, optimizer, new image, model calls or suffix continuation occurs in this preparation.

The weighted squared-response account is72.531326=17.337201+8.859181+46.334944: same-memory query mismatch, signed cross term, and teacher displacement from full-native attention. This is an exact change-of-objective account, **not an observed loss reduction from an update**. Native-Q source teachers have not been consumed downstream in these source histories. See [the mathematical derivation](MATH.md).

[Astra independently regenerated every target](REVIEW.md). The [source/turn profile](source-targets-v1/profile.json) retains separate losses, energies and cross-term signs. Equal turn probability mass does not equalize response energy or gradient size. Teacher displacement is not irreducible error, and the scalar magnitudes are absolute attention units rather than normalized loss or KL.

The next compiler assay should compare the unchanged image with one bounded signed direction, then use the actual recomputed suffix and candidate-owned history to judge it. Local response loss remains diagnostic; no drawer fitting or automatic memory expansion follows. Before attributing the rescue uniquely to retained memory payload, the distinct diagnostic cut holds native Q fixed and compares its read with zero-read and value-permuted controls. Native Q itself may carry task information, so current results establish compatibility and a useful intervention, not a unique answer carrier.

## Evidence and cost

[Our audit](audit-v1/audit.json) authenticates99executed sources,510raw files,1,278generated-token decisions, eight exact controls, eight exact bounded diagnostic reads and14fixed-current cross-history conditions. [Custody](audit-v1/custody.json) binds our code/output to the primary analysis; all36answers and followups agree exactly, margins agree exactly and the largest independent KL difference is1.87e−14. An initial saved-only flat-versus-head-shaped comparison error is retained separately in [analysis-error-v1.json](analysis-error-v1.json); flattening the comparison corrected it without rerunning or altering raw evidence.

The reused broader run paid5,238physical advances and8.0791e12nominal model-linear MACs;85.616sworker,97.72srunner,3834.7MBRSS and6470MBVRAM. The linear ledger excludes diagnostic reader products and earlier paid native-operand acquisition, so it is not total work or a speedup result. Our audit and target preparation add zero model calls. The serving baseline remains819,200selected QKV MAC/update and0.1507%whole-token nominal linear savings; the dense receiver and most of the model remain paid. Recurring native Q would raise selected QKV work to an estimated2,981,888MAC, erasing most local savings.

Mechanics: verified. Instrument: applicable with explicit numerical profile. Trend: current-query rescue plus consequential historical consumer, with later task failures retained. Terminal status: no general-chat, memory-sufficiency, whole-model or speedup claim requested or established.
