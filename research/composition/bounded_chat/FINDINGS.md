# Bounded memory now serves an actual chat client

Beast job `job-f53e7d9c952f` completed in 124.18 seconds and installed an explicit bounded mode at `/mnt/big/saturn-cmachine/bounded-current`. The persistent C server accepts real text-client conversations, continuously updates the same 112-row memory and supports two snapshot-backed parents. The stable `current` installation and the client's default native policy remain unchanged.

This integrates the existing memory law. It does not add a learned summary, answer repair or a new scientific retention policy. Its [frozen memory and arithmetic sources](runs/job-f53e7d9c952f/payload/source-manifest.json) are identical to the [bounded-retention predecessor](../bounded_retention/FINDINGS.md).

## Actual replies

Five actual text-client processes ran sequentially: full-byte and bounded ordinary conversations, full-byte and bounded delayed ledgers, then the existing native binary through the updated client. All retained the model's own outputs and completed without transport error. All five recorded **zero prompt recomputations**. The bounded delayed client reached cursor 402 while retaining 112 physical rows and 9,279,620 GPU-state bytes throughout reported boundaries.

| Request | Full-byte text client | Bounded text client |
|---|---|---|
| Explain yeast and rising bread in two short sentences | Sensible explanation, stopped at the 32-token budget | Same text, same budget stop |
| Explicit `17 + 29`, reply only integer |46|46|
| Remember Eno=17, then ask twice with rollback |17,17|17,17|
| Delayed Eno, after source/correction/neutral delay |17|3|
| Delayed current Tavi |29|1234567890|
| Delayed Luma |amber|Green|
| Delayed Eno plus current Tavi; truth 46 |40|15|

The prose answer ends at “causing it to rise” without an EOS; the client successfully branches and continues from that delivered-but-unconsumed token. The ordinary native binary gives a closely worded complete prose answer under its own arithmetic/storage profile. This is a tiny development conversation, with no general chat-quality claim.

The actual client uses 32 output tokens per turn. The historical assay uses 8 on the three individual late questions and 16 on the sum. Its truncated digit string and the client's longer digit string therefore should not be read as a new retention effect. The [raw client logs](runs/job-f53e7d9c952f/raw/bounded-delayed.stdout.txt) preserve the complete text, state reports and commands; the [full-byte control](runs/job-f53e7d9c952f/raw/full-byte-delayed.stdout.txt) preserves the wrong 40 as well as correct fact retrieval.

## Persistent ownership and replay

Both full-byte and bounded direct-protocol processes reproduce all eight historical turns, including 16 prefill/final full-vocabulary readout hashes per policy and the delayed root cache hash at cursor 364. Each also replays an identical generated child twice after overwriting ring slots. Eleven refusals per policy leave observable state unchanged: malformed/out-of-range/capacity append, invalid generation, unsupported legacy saves, a third snapshot, non-top/expired/reset-invalidated handles.

From the same pre-generation parent, `G 1; G 1` equals `G 2` in token IDs and complete reported state, excluding the deliberately distinct marker ID. A further generation after delivered EOS returns an empty list without mutation. Reset clears reachable rows, evictions and stack depth. Eight local tests cover client policy selection, bounded geometry, branch budget and inherited codec/ring mechanics with sanitizers.

The public client ordinarily leaves its final emitted token pending. The historical assay explicitly consumes that token, including EOS. Thus the actual delayed client is at 363 before its next prompt while the direct historical replay is at 364. The next append consumes the pending token once, before ingesting the new header. These are different ownership boundaries. A blanket comparison of those two cursors or hashes would be incorrect.

## Resource and installation account

| Tracked GPU allocation | Full-byte,768 rows | Bounded,112 rows |
|---|---:|---:|
| Mutable state including two parents and abort snapshot |48,781,316|9,279,620|
| Model, helpers and shared weight tile |3,124,434,964|3,124,434,964|
| Total |3,173,216,280|3,133,714,584|

The bounded state is 80.98% smaller than the same-recovery control. Total tracked GPU allocation falls 1.245% because the full learned parameters and dense forward computation remain. Successful allocation requests exclude CUDA/cuBLAS overhead and host memory. The complete seven-process lease peaks at 4,202.5 MB RSS and 6,394 MB VRAM; this aggregate includes the original native binary, so it is not a bounded-only peak or a speed comparison.

The versioned installation is `/mnt/big/saturn-cmachine/experimental/bounded-chat-bcf17701b893fe1b`. Its [79-file manifest](runs/job-f53e7d9c952f/raw/installed-files.json) binds the new binary, client, frozen sources and unchanged native copy. Publication checks the stable path/native hash and previous opt-in alias before atomically replacing the separate alias. No background service starts. The stable native SHA remains `d4ba05c0161a737d9d8c227a679ecb72ea6673773045bf638ca65df888995613`.

Conversation persistence lasts for the running processes. The host client retains the transcript for display and prompt construction, so total application storage is not fixed. No disk session archive is added. Logical capacity remains 768 in this trial; bounded GPU allocation does not establish unlimited context or arbitrary exact recall.

## Evidence and next question

[Saved-only verification](runs/job-f53e7d9c952f/verification.json) authenticates 75 sources, 23 raw files/171,960 bytes, scheduler log, whole-invocation Saturn receipt, installed-file manifest and exact recomputed analysis. Report SHA is `d20748bc4b9f95ac2ea9ea610bcc6d04b276724119263b16f6c6ef97b6ab041f`. The [evidence sweep](index/job-f53e7d9c952f/evidence-verify.json) covers seven observation rows and eight predicates without granting semantic claims. The receipt covers declared process invocations rather than every internal CUDA operation.

The runtime integration works, and the delayed-memory failure survives it. The next useful scientific change remains a better representation/update law tested through its own future queries and writes. The [architectural post](../../../../obsidian/blog/2026-11-06-171900-what-bounded-memory-changes-in-the-c-state-machine.md) places that question beside symbol tables, MLP laws and time circuits. The [usage guide](README.md) makes the installed trial directly accessible.
