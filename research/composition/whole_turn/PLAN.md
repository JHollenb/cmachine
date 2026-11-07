# Whole-turn ingestion and later reads of a frozen MLP program

2026-11-06. Exploratory next cut after the [composition findings](../FINDINGS.md) and [compact-chat synthesis](../../../../obsidian/blog/2026-11-06-115551-from-a-c-chat-runtime-to-compact-sufficient-state.md). Freeze this plan, all source bytes and the exact tokenized append grammar before any new model execution. Astra reviewed the design before submission.

## Question and specimen

Can the existing frozen B3 MLP16 program ingest a complete correction turn and subsequently read its own updated context? Original Qwen2.5-1.5B-Instruct, authenticated historical formation runtime, C FP32/cuBLAS numerical profile, CUDA Beast, one resident model, batch1, capacity512, one CPU thread. The original276536-byte bank is imported byte-for-byte without calibration or training. Original dense weights, residuals and full KV remain. This is a local computation experiment, not compact sufficient total chat state.

Rebuild the historical source conversation and its own red+EOS answer. Its consumed IDs, position59 and complete valid-KV hash must match the frozen historical receipt before candidate execution. All primary policies fork that same complete-token parent.

Correction: “Correction: Niko's badge is now green. Keep the other person's badge unchanged. Reply only with OK.” Reply budget8.

Two independent later user questions fork each policy's own post-correction parent:

- Target: “What color is Niko's badge now? Reply with the color only.” Budget8; application truth green.
- Collateral: “What badge color does Mira still have? Reply with the color only.” Budget8; application truth red.

The target answer never supplies the collateral branch. Generated IDs are authoritative and are not decoded/re-tokenized to rebuild context.

## Four policies

Native, recurring PROJECT, recurring zero-write and recurring CORRECT all process every appended template/user/generated token through their selected MLP16 policy. PROJECT omits the live complement and skips its two G/U calls. Zero uses the existing held-write API with an owned positive-zero6144-byte write and executes neither G/U nor down at that site. CORRECT pays the two original G/U projections and restores the omitted affine operands before the nonlinear consumer. All other operations execute normally on each branch's own state. Bank bytes are checked before/after the entire assay.

Every generated token is consumed, including EOS or the final budget token. If an acknowledgment reaches its budget without the template's151645 closing token, the next append explicitly supplies that delimiter through the same selected policy. This host-supplied token is logged separately from model generation. An already consumed151645 is skipped once;151643 is a distinct stop and does not qualify for that skip. Exact append grammar, including newline and role headers, is checked against native apply_chat_template before the C model starts. An incorrect acknowledgment, wrong color or truncation is evidence, not a scientific stopping condition.

## Primary seal and paid diagnostics

Complete all four policies and their12 turn results before any diagnostic replay. Seal the exact consumer-prefix bytes of driver.json, including output IDs, full boundary-logit hashes and complete valid-KV hashes, and publish that seal before diagnostic execution.

Then densely replay the three nonnative policies' exact consumed token sequences, using one correction replay plus two independent reader children per policy. These nine segments are forced same-text diagnostics, not autonomous native answers. They may inspect sealed candidate-selected IDs; they never write candidate state or supply candidate operands. Do not replay the native arm redundantly.

Because interventions occur only at MLP16, same-history dense replay is predicted to preserve captured layer16 entry, raw receiver, normalized input and current KV16 exactly. Check those actual arrays before describing its MLP write as a same-input reference. Upper KV and full logits measure accumulated same-text state effects. Cross-policy comparisons with different histories retain their scope.

## Observations and execution budget

Capture five1536-wide FP32 local ports and current K/V16–17 at each transition. Active policies additionally retain reconstruction/omission and their executed coefficients/scale. First-transition G/U ports are inherited observation diagnostics. Full151936-logit vectors and complete valid-KV hashes are retained at the prefill and final boundary of every turn. Record pending-token authority, clocks, skipped/supplied closing token, stop/censoring, exact parent histories, physical G/U/down counts and the primary seal. Retain raw arrays and immutable reports.

Before submission, the conservative envelope permits at most64 tokens per appended template/user suffix and8 generated tokens per reply: at most1512 selected-layer transitions across all primary/diagnostic segments, excluding the historical source replay. Including five local ports, KV16–17, three active-policy reconstruction/omission pairs, first-transition G/U arrays,42 full boundary readouts and8MiB non-array overhead gives96,023,552bytes (91.58MiB), within the96MiB archive cap. The worker verifies each suffix is at most64 tokens and calculates the tighter tokenizer-derived bound before starting the C model. Refuse only an invalid/incomplete instrument or unsafe resource bound, not an unfavorable result.

One finite mrun lease, one model load, no automatic retries, timeout600s, estimated180s. Use the measured same-backend reservation exception6144MB RAM/6144MB VRAM, from earlier512-context C jobs (RSS about4GB, VRAM6378MB, enforced VRAM ceiling6758.4MB), instead of generic HF geometry. No duplicate full suite, no serving speedup claim. Verify scheduler/log/report/source joins, all raw hashes/extents, primary seal, replay token histories, local skip addition and exact lower-port predictions offline. Run Saturn evidence sweep and verify after collection. Scientific outcomes remain observations/trends; terminal certification is not requested.
