# Preserved staging failure and narrow repair

The first from-zero mask followup, `job-7b9c5ae82f54`, stops before any token advance. The C executor loads the model and then cannot open its first coefficient file: the wrapper had left all six program arrays in the payload root while the executor resolves files relative to `results/`. Stderr is `context-query 21: f`. No policy trajectory JSON exists. This is a mechanics failure and supplies no scientific null.

The collected report SHA256 is `aeadff92ac0ca9f73ef4a1e43e4fb829c356bd1c0d07cc27aefa0337aacd6917`; its scheduler log hash verifies. The immutable failed source/payload/raw archive remains under `runs/job-7b9c5ae82f54`.

The repair copies the same authenticated arrays into the executor root, checks every exact byte count and hash before launching a model process, and includes driver stderr in protocol errors. Model, inputs, masks, images, rank, budgets and scientific plan are unchanged. A new explicit finite lease, `job-1335a953cd08`, runs the corrected harness; no automatic scheduler retry is enabled. Its result and scientific status are recorded separately in the findings.
