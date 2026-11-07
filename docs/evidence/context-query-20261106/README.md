# Contextual query integration and own-state result record

Guarded CUDA job `job-fe51e7f4f0ee` ran the canonical C operand programs on Qwen2.5-1.5B-Instruct, source-only response proposals and fresh two-turn continuations. The [full findings](../../../research/context_query/FINDINGS.md) describe the positive trend and the later-layer failure.

The updated layer-17 query reduces mean fresh conditional KL from 0.0271048724 to 0.0234218206, 13.59%, without extra serving maps or memory. A separately compiled layer-18 Q/K bank increases error and loses an inventory binding even with full memory there. This is a bounded exploratory result, not general chat closure or a speed qualification. The runtime interfaces remain opt-in and have not replaced the separately installed release.

This directory retains the original report, its 100-file executed source manifest, saved-only analyses, proposal/selection seals, integration and execution records, and standard evidence summaries. Report SHA256 is `d97a8dedb3a3e187e958c56ecf81bf59d8f2b45f3b69c4c27a76925507d9e9e2`. The manifest binds the exact executed C/CUDA sources independently of a later source commit.

The local full archive lives at `research/context_query/runs/job-fe51e7f4f0ee`, including its sealed scheduler log, executed payload, all 490 raw files and retained logits. Those large arrays, model weights and derived weight images are not included in this small portable record. Run `analyze.py` and `audit_saved.py` against that full archive to reproduce the saved checks; these record copies alone cannot rerun the numerical audit or model experiment.

The dedicated audit authenticates 128 retained full-logit pairs, 64 prototype snapshots, 420 exact historical reads, declared image updates and layer-18 basis geometry. Full 2,054-row sampled C/Python stream comparisons ran inside the worker before their temporary arrays were removed. The standard evidence sweep has `target-streams-only` coverage; it recognizes 64 curated turn observations and no typed internal scientific receipts/addresses. Its clean status does not independently certify the numerical or semantic result.
