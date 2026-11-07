# Contextual query programs on their own evolving states

Completed guarded CUDA job `job-fe51e7f4f0ee` integrates the useful operand/reader computations into the canonical C runtime, then tests a source-only query update and a recipient-specific later-layer compiler. The updated layer17 query lowers mean fresh own-history conditional KL by 13.59% across four two-turn conversations at 128/1024 prompt tokens. Adding a rank128 Q/K program at layer18 raises error and misbinds the whistle in both inventory contexts, even with full layer18 memory. General closure across layers remains open.

[Findings](FINDINGS.md), [mathematics](MATH.md), [frozen plan](PLAN.md), [runtime API](../../docs/QUERY-PROGRAMS.md), [local full report](runs/job-fe51e7f4f0ee/report.json), [saved verifier](analyze.py), [independent retained-array audit](audit_saved.py). Large captures and images remain local/ignored; the tracked evidence summary under `docs/evidence/context-query-20261106` is a portable record, not the full raw archive.

Stage and submit through the workspace's shared public MDB environment:

```sh
rtk proxy common/bin/common exec /usr/bin/env PYTHONPATH= saturn-pub/packages/mdb/.venv/bin/python cmachine/research/context_query/run.py stage
rtk proxy common/bin/common exec /usr/bin/env PYTHONPATH= saturn-pub/packages/mdb/.venv/bin/python cmachine/research/context_query/run.py submit PAYLOAD
rtk proxy common/bin/common exec /usr/bin/env PYTHONPATH= saturn-pub/packages/mdb/.venv/bin/python cmachine/research/context_query/run.py collect JOB
rtk proxy common/bin/common exec /usr/bin/env PYTHONPATH= saturn-pub/packages/mdb/.venv/bin/python cmachine/research/context_query/analyze.py RUN_DIRECTORY
rtk proxy common/bin/common exec /usr/bin/env PYTHONPATH= saturn-pub/packages/mdb/.venv/bin/python cmachine/research/context_query/audit_saved.py RUN_DIRECTORY
rtk proxy common/bin/common exec /usr/bin/env PYTHONPATH= saturn-pub/packages/mdb/.venv/bin/python cmachine/research/context_query/evidence.py JOB
```

Staging requires the authenticated local own-history source archive and prepared same-memory targets. The payload copies actual canonical runtime sources and all required headers; make runs inside the shipped payload. There are no private macro replacements of Q/K/V or attention. The ordinary model weights remain frozen and resident. Source feedback, selected fresh inputs, generated states and full native shadows remain separate.
