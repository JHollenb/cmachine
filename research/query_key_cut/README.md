# Query and key errors at the first wrong binding

Completed guarded CUDA job `job-1555c27c080f`: 36 logical / 32 physical single-read branches across the known 128/1024-token inventory failures and two matched historical policies. At 1024 tokens, native key history with the compiled query restores silver on the native-layer18 parent; native query alone does not. On the compiled parent, even native current Q/K leaves amber. Current layer18 closes byte exactly across parents, while attention19 differs immediately.

[Findings](FINDINGS.md), [math](MATH.md), [frozen protocol](PLAN.md), [C executor](driver.c), [saved verifier](analyze.py), [independent prior/greedy-choice audit](lineage_audit.py). [Portable result records](../../docs/evidence/query-key-cut-20261106/README.md) retain the scope and identities; full arrays remain local.

The constructive follow-up `job-1335a953cd08` runs frozen query-only versus key-only layer18 programs from token zero. Query-only/native keys preserves both tested inventory bindings and their user updates; key-only does not. Query-only also worsens explanation fidelity relative to layer17 alone, leaving the four-conversation mean20.62% worse. These are known development paths, not a fresh heldout claim. See the [frozen follow-up plan](ON_POLICY_PLAN.md), [executor](policy_driver.c), [verification](policy_analysis.py) and [preserved staging failure](POLICY_MECHANICS.md).

Use the workspace shared public MDB environment:

```sh
rtk proxy common/bin/common exec /usr/bin/env PYTHONPATH= saturn-pub/packages/mdb/.venv/bin/python cmachine/research/query_key_cut/run.py stage
rtk proxy common/bin/common exec /usr/bin/env PYTHONPATH= saturn-pub/packages/mdb/.venv/bin/python cmachine/research/query_key_cut/run.py submit PAYLOAD
rtk proxy common/bin/common exec /usr/bin/env PYTHONPATH= saturn-pub/packages/mdb/.venv/bin/python cmachine/research/query_key_cut/run.py collect JOB
rtk proxy common/bin/common exec /usr/bin/env PYTHONPATH= saturn-pub/packages/mdb/.venv/bin/python cmachine/research/query_key_cut/analyze.py RUN_DIRECTORY
rtk proxy common/bin/common exec /usr/bin/env PYTHONPATH= saturn-pub/packages/mdb/.venv/bin/python cmachine/research/query_key_cut/lineage_audit.py RUN_DIRECTORY
rtk proxy common/bin/common exec /usr/bin/env PYTHONPATH= saturn-pub/packages/mdb/.venv/bin/python cmachine/research/query_key_cut/evidence.py JOB
```

The mask follow-up uses `policy_run.py stage|submit|collect` and `policy_analysis.py RUN_DIRECTORY`; run `evidence.py JOB` afterward. `policy_plot.py RUN_DIRECTORY` uses the shared environment's matplotlib. Its archived engine is staged from the authenticated original context-query run, not dynamically imported from a changed experiment checkout.

Staging requires the authenticated local context-query archive. The native operands are paid diagnostic projections on the actual compact17-produced receiver; they are not supplied by a separate full-native model, a new economical compiler or oracle-correct answer tokens. The canonical runtime is unchanged. Its existing cache callbacks and token checkpoints own the execution; this harness adds a complete prototype snapshot and explicit logits restoration for its one common parent. The public Saturn receipt covers the whole C invocation; inner branch closure is separately checked by C and retained-array audits.
