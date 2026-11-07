# Fresh C/CUDA demonstrations

Guarded mrun job **`job-995c8e81e7ca`** ran the public [chat](../../chat_branch/README.md) and [compiled operand](../../compiled_operands/README.md) examples sequentially on Beast's RTX 4080. Runtime base: `c6c1a9b`. The complete frozen source payload binds the demo scripts and canonical C/CUDA runtime independently of a later documentation commit.

The chat session answered `OK.`, then `Green` on branch A. After rollback, it independently regenerated `Green` with identical generated IDs, valid cache hash, complete FP32 logit hash, pending token and position. After a second rollback, branch B answered `Yellow` and committed. C selected the generated IDs; the client never injected archived answer IDs.

The compiled program completed **44 token advances**, including **eight own generated tokens**, with all **6,685,184 FP32 logits** directly equal to native, and matching valid cache hashes and selected tokens. Its 33-ID table survived save/load with the same model-qualified fingerprint. An uncovered token was refused before mutating session state, valid cache or logits.

| Work or storage | Recorded value |
| --- | ---: |
| Native source learned affine work | 67,917,053,952 nominal MACs |
| Offline table compilation | 103,809,024 nominal MACs |
| Candidate learned affine work | 67,778,641,920 nominal MACs |
| Original first-layer Q/K/V work omitted online | 138,412,032 nominal MACs (0.204%) |
| Host table | 270,468 bytes |
| Candidate transfers | 360,448 bytes |
| Original resident model weights | 6,174,857,216 bytes |

Discovery and compilation are paid; this is source-covered reuse. Remaining dense operations, nonlinear work, attention products, transfers and auditing are additional costs. No speedup or held-out generalization is claimed. The admitted lease, including the build and instrumentation, took 31.91 seconds; observed peaks were 6,394 MB VRAM and 4,967.8 MB RSS. This is a run record, not a timing comparison.

## Inspect or verify

- [Concise summary](summary.json), [complete worker report](report.json), [verification record](verification.json).
- [Chat text, token IDs, states and protocol events](raw/chat/report.json).
- [Per-step lexical comparisons and accounting](raw/operands/report.json), [source seal written before candidate creation](raw/operands/source-seal.json).
- [Frozen source manifest](source-manifest.json), [executed C driver](payload/demos/compiled_operands/driver.c), [executed chat client](payload/demos/chat_branch/chat_branch.py), [executed worker](payload/demos/fleet_worker.py).
- [Public file index](public-manifest.json), [redacted scheduler receipt](guarded-receipt.json).

From the repository root, without a model, GPU or Python dependencies:

```sh
python3 demos/verify.py --public demos/results/pretrained
```

The public verifier checks the indexed bytes, source/raw joins, model and binary identities, whole-invocation Saturn receipt, chat comparisons and lexical rows, then recomputes the summary. The full collector-side verification additionally checked the scheduler-sealed log and complete guarded custody before publication. The public scheduler receipt is a redacted attestation; public consistency verification does not independently contact or authenticate the scheduler. Binary and weight bytes are not distributed. The `.bin` lexical table remains near compute as derived model data.

The earlier attempt, `job-34c51e8fb6bb`, stopped before generation because the new client rejected standard Hugging Face tokenizer-cache symlinks. Its raw failure evidence remains in the local collection. The client now validates relative file addresses and the actual linked content hashes; three targeted tests cover accepted cache links, changed content and invalid addresses. The successful run above used a newly frozen payload. No scientific result was discarded or reclassified.

This is evidence for finite execution and replacement mechanics. Conversation quality outside these prompts, sufficient compact memory, economical broad operand generation and forward-pass-free chat remain open.
