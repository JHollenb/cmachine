# Owned compressed append and stacked attention

This isolated experiment extends [recurring fixed-span compression](../span_recurrent/FINDINGS.md) into growing selected-layer cache ownership. Read the [prospective plan](PLAN.md) for the frozen execution contract. The installed C runtime is unchanged.

The completed [findings](FINDINGS.md), [mathematical contract](MATH.md), [Astra review](REVIEW.md), [report](runs/job-2a28feeb7e5b/report.json), [verification](runs/job-2a28feeb7e5b/verification.json) and [evidence sweep](index/job-2a28feeb7e5b/evidence-verify.json) bind the result. Across three linked turns, the stacked arm performs 178 main encoded appends/reads, restores partial cursors [94,93]to 93, consumes changedQ18 at all 89 positions, saves 1,032,188 tracked allocation bytes, and preserves native answers. Sparse full-logit differences range 0.370–1.363%; all model weights and 26 other dense caches remain.

`runtime_patch.py` reproducibly transforms three authenticated runtime files. Selected layers have no dense cache slot. A cache-step callback owns their memory; a rewind callback truncates it alongside session restore. A private token snapshot supports abort after partial layer execution. Ordinary native execution retains its original equations and allocation layout as a control.

`codec.h`, `cache.h` and `cache_cuda.cu` implement per-row signed-byte K/V with separate FP32 scales. The candidate's actual new row is encoded and published before its current attention read. Attention decodes keys into one counted float head-plane, scores and normalizes, then overwrites that workspace with decoded values. There is no selected-layer native fallback or persistent decoded float bank.

`driver.c` runs native, exact-FP32 callbacks at layers 17+18, byte17, and byte17+18 sequentially against one weight bank. Each fresh session starts from the authenticated native 93-token prefix, consumes three linked user turns and its own greedy tokens through EOS, and exercises stack rollback, sibling overwrite and replay. Byte sessions also inject an error after encoding and before cursor publication; the stacked case aborts after layer17 has already published and consumed its row.

`analysis.py` checks exact callback parity, independently reconstructs every encoded valid row, verifies sparse snapshots, tracks actual allocation extents, and separates local codec error from native-trajectory differences. Layer18 queries are compared only at matching consumed histories and clocks. All scientific magnitudes remain observations rather than discovery gates.

The bound is four sequential sessions, three linked turns each, capacity 512, one finite CUDA mrun lease and no automatic retries. No training, model-weight compression, semantic memory extraction, oracle answer repair or speed claim is part of this experiment.

Local model-free checks operate on a staged payload:

```sh
common exec python -m unittest -v test_append test_token_authority
cc -std=c11 -Wall -Wextra -Wpedantic -ffp-contract=off -Iruntime -fsyntax-only driver.c
```

`run.py` stages, submits, collects and verifies the immutable sources and raw evidence. `evidence.py JOB_ID` indexes verified observation rows through the standard offline evidence plane. Successful verification is a mechanics and custody result, not general chat certification.
