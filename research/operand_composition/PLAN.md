# Frozen MLP16 → attention17 composition

Question: can the rank-128 query/key program consume the receiver and persistent
carriers produced by the rank-64 MLP program, at their combined measured cost?

Use Qwen2.5-1.5B-Instruct, original checkpoint bytes and the authenticated C
FP32/pedantic runtime. Freeze contextual-union's final coupled64 factors and
query_response's source-selected response query plus B128/K128. No new fitting,
optimizer, or selection sees the six new two-turn conversations. The query
objective historically used a prototype reader; full-memory use is a measured
transfer, not an assertion that this image is optimal for full memory.

Four arms: 00 native maps, 10 cheap MLP16, 01 cheap Q/K17, 11 both. Every arm
starts with empty full FP32 K/V at all 28 layers, advances from its own residuals
and memory, selects and consumes its own tokens, and receives the same second
user message. All original D/O/V, other layers, embedding and head remain paid.
Memory precision, support and allocation do not change across arms. Six cases
cover two correction/recall, two arithmetic and two prose contexts. Token budgets
are 8/8/16/16/32/32 per answer, including EOS consumption; capped completions
remain observations with their incomplete coverage visible.

Seal all 24 own trajectories before replaying the joint arm's actual consumed
IDs through 00, 10 and 01 (18 diagnostic paths). The fourth corner is the saved
11 trajectory. Matched-history interaction I=Y11−Y10−Y01+Y00 describes the
recurring policies on that history, not an isolated instantaneous intervention.
Own-history differences include generated-history effects. Retain signed
interaction vectors and their cross terms, full-vocabulary discrepancies/KL,
actual answers, and MLP/receiver/Q/K/V/read/write trajectories and selected KV.

Mechanics: authenticate source reports, factors, checkpoint and tokenizer;
check native sham logits and valid KV byte equality; independently evaluate the
ported factors from supplied normalized inputs; verify bias/RoPE placement,
token authority, shape/finite contracts, program immutability, call/MAC counts,
and all same-ID replay boundaries. Under matching IDs, MLP16 input/KV16 must
remain unchanged; MLP16 output and attention17 normalized receiver must be
reader-independent. Numerical replay discrepancies are recorded as instrument
diagnostics with arithmetic limitations, not arbitrary fidelity gates.

Only integrity or unsafe execution stops collection. Answer correctness, local
fit, same-top-token, cancellation and interaction size are observations. No
terminal emulation, speed, arbitrary chat or compressed-memory claim is sought.

One resident C model, CUDA/B1/one CPU thread, one mrun lease, no retries. Small
model-free mechanics checks precede the lease. Expected run is minutes, bounded
by 840 seconds and a 512 MiB raw archive. Preserve failures and immutable raw
artifacts, collect the exact job, independently verify and run Saturn evidence.
