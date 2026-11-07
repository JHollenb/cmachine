# The same query program on its own histories — November 6

The completed `job-47cc27324445` isolates the history distribution while keeping the model, query image, memory, prompts, followup and32-token budgets unchanged. Equal-source/equal-turn conditional native KL rises0.023686297→0.029952984, or26.46%, on compact-generated rather than native-generated histories. Two contexts improve; the card context alone drives the mean increase. Its compact answer is a shorter correct paraphrase preserving Iris/yellow and Pavel/orange. The larger KL is a fidelity shift, not demonstrated semantic failure.

[Protocol](PLAN.md), [math](MATH.md), [immutable report](runs/job-47cc27324445/report.json), [verification](runs/job-47cc27324445/verification.json), [saved-only study](runs/job-47cc27324445/study.json) and [Astra review](REVIEW.md). Executed sources are in the run's payload; additive analyses and documentation remain separate from that frozen evidence. The Saturn path is now a compatibility alias to `cmachine/research` following the concurrent workspace relocation.

## Controlled histories and independent evaluators

The specimen remains Qwen2.5-1.5B-Instruct, C FP32/pedantic cuBLAS, capacity512, selected layer17, B128, fixed dose2 image SHA7b0d047635f17effc91f408ca3b8691269169a0709ffd2fa78fd55623a9f005f, compiled K/native V and16recent+16counted prototypes per KV group. No image selection, new direction, target panel or memory expansion occurs.

History generator (native/compact) crosses evaluator (native/compact). The archived native IDs replay through compact and a fresh native reference. Separately, compact generates its own two answers before a new native shadow consumes those exact IDs. Compact state is destroyed and its arrays, trace and full-stream hash seal before native shadow creation. The shadow owns its native caches from token zero, never choosing or repairing candidate tokens, receivers or caches. Native O/MLP and later layers recompute normally under each evaluator's actual state.

The primary score uses distributions immediately before each generated answer token, includes the EOS choice, excludes unused post-answer rows, and weights sources/turns equally. Actual answer lengths remain visible. This is conditional fidelity on identical text within each pair; the two history conditions differ after their first generated mismatch.

| Source | Native-history KL | Own-history KL | Direction |
|---|---:|---:|---|
|Notebook separation|0.035966185|0.030402763|Better|
|Card correction|0.023682091|0.048440503|Worse|
|Arithmetic explanation|0.011410614|0.011015687|Better|
|Equal-source/turn mean|0.023686297|0.029952984|Worse26.46%|

Cross-history consumed prefixes match133/106/82tokens. Every retained selected operand and pending top is byte-identical within those prefixes under the corresponding evaluator. These include the state choosing the first differing token. Later decision-index comparisons are summaries over different linguistic histories, not paired identical states.

## The larger discrepancy preserves the card relationship

The first card answer matches native exactly. The second native answer is “Iris still has a yellow card, and Pavel now has an orange card.” The compact branch independently generates “Iris keeps a yellow card and Pavel keeps an orange card.” It ends at14tokens including EOS, versus17native. Both relationships are correct on this specimen. Other source replies are coherent partial explanations; four of six answers hit the32-token cap. There is no completed arithmetic example or general free-form correctness result.

Card followup KL rises0.043238916→0.092755739. Restricting native to its first14decision indices gives0.047310825, so the effect is not solely shorter-answer normalization. The exact mean difference0.049516823 splits into0.045444914 over the common-budget summaries plus0.004071909 from unequal-tail weighting. These are arithmetic terms, not causal shares: the text histories differ after token3. Large own-history KL also occurs at decisions where native and compact argmax agree. A changed probability distribution, a changed top token and a changed task binding are different observables.

The raw field `native_greedy_prefix_before_decision` is a legacy misnomer: it means cumulative agreement of the two evaluators' generated-token choices. In the native-generated condition, native remains on its own greedy path even after this flag becomes false. The [additive study qualification](runs/job-47cc27324445/study.json) preserves the original field and evidence rather than rewriting it.

## Separate Q, key projection and aggregation before changing memory

At327answer decisions, a saved-only diagnostic crosses native/compiled Q with native full K/V, compiled-key full K/V and compiled-key bounded memory. Native V arrays match byte-exactly on each same-text pair. Full-memory diagnostic reads use FP64; the actual bounded law emits FP32 and replays exactly. Maximum native attention reconstruction discrepancy is3.48e−7relative, retained as a numerical-profile remainder.

| Equal-source/turn mean relative attention norm | Native-generated history | Own-generated history |
|---|---:|---:|
|Query substitution alone|33.13%|34.06%|
|Key substitution alone|27.86%|27.60%|
|Query×key interaction|17.87%|18.05%|
|Both substitutions, full memory|40.69%|41.21%|
|Aggregation at actual compact Q|24.34%|24.08%|
|Actual bounded reader|52.18%|52.70%|

The aggregation term includes the bounded reader's final FP32 rounding. The vector account is Q+K+Q×K+aggregation-at-compact-Q+native-profile remainder; maximum relative closure remainder7.77e−17. Closure is an accounting identity, while exact compact replay and native reconstruction qualify the instrument. Norms overlap/cancel and cannot be added as percentages of blame. These hypothetical read outputs do not run the native suffix or establish which restoration changes an answer.

In the card followup, actual relative attention discrepancy changes67.22→68.47% while conditional KL more than doubles. Local norm magnitude therefore does not itself explain the probability change. The retained state still supports a correct paraphrase. This gives no memory-capacity lower bound and no reason to enlarge memory from the aggregate alone.

## Execution and cost

One finite Beast CUDA lease, one resident model, CPU1/B1, no retries:12trajectories and1,718physical advances, comprising846native-history evaluator advances,840own-history evaluator advances and32shams. The compact branch generates162answer tokens and takes420total own-history advances; the corresponding native shadow pays another420. Worker time52.427s, peak observed RSS4044.4MB/VRAM6380MB. These include capture and audits, not serving latency or speedup.

Verification authenticates75sources/215rawfiles, exactly reproduces33historical arrays and423full-clock metric pairs, verifies843compact reads and24memory snapshots, and recomputes the frozen analysis without a model. Contemporary C/Python audits cover843full-vocabulary pairs;48full vectors support24durable metric replays. Full streams were intentionally removed after their audits. Astra independently checks all162autonomous token choices, common-prefix bytes and24durable KL pairs, with maximum C/independent discrepancy2.67e−13.

The evidence sweep returns3with57relative native-entropy spike diagnostics, then verify0with an intact ledger. The [watch audit](runs/job-47cc27324445/watch-audit.json) finds56within-role token transitions and one between-pair boundary. Finite/shape/custody contracts pass; entropy ratios alone establish neither corruption nor semantic failure. Original fires remain unchanged. Mechanics/instrument complete; trends heterogeneous; terminal certification not requested.

Serving is unchanged:819,200selected QKV MAC/update versus3,145,728native,98,464selected logical memory bytes,73.96%selected-QKV and0.1507%whole-token nominal linear savings. Full native lower receiver, other27layers, original weights/other caches and vocabulary head remain paid. This is new feedback coverage, not a new cheaper query image, installed policy, accurate general chat or whole-model emulator.

The next [compiler cut](NEXT.md) uses one response direction from these actually visited receiver/memory states, with signed controls and native-consumer validation before fresh autonomous evaluation. The newer [long-context binding result](../../../obsidian/blog/2026-11-06-183300-longer-chat-context-still-needs-the-right-relationships.md) supplies a separate task-aligned reader intervention target. Its changed followups/budgets do not replace this matched-schedule contrast.
