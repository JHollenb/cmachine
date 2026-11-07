# Opt-in bounded chat

This integration exposes the [bounded-retention backend](../bounded_retention/README.md) through a persistent C process and the ordinary text client. It reuses the frozen encoder, row sampler, full learned decoder and snapshot implementation. The new boundary is interactive append/generation, state inspection and branching.

**Completed:** Beast job `job-f53e7d9c952f` installed and exercised the opt-in mode. Fresh prose, explicit addition and short recall worked in this small conversation; all four delayed questions remained wrong. Both persistent policies reproduced the prior assay's token IDs and full readout hashes. [Findings](FINDINGS.md), [verification](runs/job-f53e7d9c952f/verification.json), [installation receipt](runs/job-f53e7d9c952f/raw/installation.json).

The bounded policy retains 16 anchors, 64 recent rows and 32 sampled evicted rows at every layer. Its fixed GPU state is 9,279,620 bytes including two branch snapshots and a token-abort snapshot. The matched full-byte policy retains 768 rows with the same recovery machinery. Original BF16 parameters and FP32 tiled forward computation remain paid.

## Installation and use

The finite trial builds a versioned installation under `/mnt/big/saturn-cmachine/experimental/` and publishes `/mnt/big/saturn-cmachine/bounded-current` after protocol and client checks. The stable `current` link remains independently available. No background service is started.

Inside an admitted Beast GPU lease, use:

```sh
python /mnt/big/saturn-cmachine/bounded-current/tools/chat.py \
  --cache-policy bounded --context 768 --new-tokens 32 \
  --model /mnt/big/llm-models/tcm/qwen25-1.5b-fp32-execution-v2.tcm \
  --tokenizer /mnt/big/llm-models/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306
```

Select `--cache-policy full-byte` for the matched arithmetic/encoding control. The client's default policy is `native`, served by a byte-identical copy of the existing stable binary. Bounded support can be configured with `--anchors`, `--recent` and `--reservoir`; the published trial validates the default 16/64/32 geometry.

`/state` shows logical position, occupied support, eviction population, pending output, branch depth, cache/readout hashes and tracked GPU bytes. `/push`, `/rollback` and `/commit` support two snapshot-backed parents. A third push refuses locally. `/save` and `/rewind` explain that this backend requires the snapshot stack. Reset invalidates all markers. `/quit` closes the process.

Conversation state persists while the client/server run. The client retains the complete transcript on the host for display and template construction; that storage is additional to fixed GPU memory. If tokenization changes an already processed prefix, the client announces a complete prompt recomputation and invalidates saved branches. Neither disk-restorable sessions nor unbounded logical context are implemented. The trial uses a 768-token logical limit.

## Validation and evidence

[PLAN.md](PLAN.md) defines the finite comparison. [server.c](server.c) implements the persistent token protocol; [controller.py](controller.py) tests actual process ownership, refusals and replay; [analysis.py](analysis.py) checks historical token/readout parity and separately preserves text observations. [install.py](install.py) records installed files and publishes the separate alias. [worker.py](worker.py) runs seven sequential processes in one CUDA lease.

Stage and submit through `run.py stage` and `run.py submit PAYLOAD`. Collect with `run.py collect JOB`, then run `run.py verify runs/JOB` and `evidence.py JOB`. Use the shared `common exec` environment. Eight model-free tests cover client selection and snapshot budgets plus inherited codec/ring mechanics under sanitizers. Real-model evidence includes split versus combined generation, repeated EOS, rollback, reset, unchanged error paths and the prior delayed-recall specimen.

Semantic quality is an observation. The previous bounded experiment missed all four delayed questions; this integration does not add a new retention law or a semantic repair system. See the [architectural post](../../../../obsidian/blog/2026-11-06-171900-what-bounded-memory-changes-in-the-c-state-machine.md) for the distinction between bounded executable state and sufficient chat state.
