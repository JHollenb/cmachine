# Frozen correction-span responses to later queries

The [completed findings](FINDINGS.md) implement the [prospective plan](PLAN.md), turning the prior [conditional attention identity](../attention_read/FINDINGS.md) into a source-only construction. Deterministic Ward clustering stores counts, mean post-RoPE keys and mean values from one native correction span before later queries exist. Eight clusters per KV head use 16,448 payload bytes plus a 32-byte header, versus 69,632 raw span-KV bytes. One-cluster, exact34 and deletion controls distinguish aggregation quality, execution correctness and sensitivity to omission at the selected read.

Two fresh questions share native prefixes across five last-token read arms. Candidate layer17 attention packs only the dense complement and compiled entries into one globally normalized read; it does not consult original correction-span KV. Later generation is native and can consult original memory again. This is a one-use response/consumer assay, not an ongoing compact-chat system or whole-runtime memory reduction.

[span_bank.h](span_bank.h) contains the memory-only constructor and bank+query-only predictor. [span_cuda.cu](span_cuda.cu) performs the selected packed read. [driver.c](driver.c) freezes programs before opening question token files, records primary predictions and consumes each branch's own output. [analysis.py](analysis.py) checks exact34/native parity and audits original-span responses only after primary records exist.

Job [job-524febce43e5](runs/job-524febce43e5/report.json) completes in 23.64 scheduler seconds at 6,378 MB peak VRAM. Eight-centroid first-logit differences are 1.592% for recall and 0.783% for prose, versus deletion's 3.640% and 1.733%. Exact34 matches native bytes; every arm produces the same own answers. Isolated eight-centroid span numerator errors remain 47–49%. The [saved-only omission audit](runs/job-524febce43e5/omission-audit.json) separates lost score dispersion from finite score/value dependence. [Strict verification](runs/job-524febce43e5/verification.json) authenticates 72 sources and 402 raw files; [Astra](REVIEW.md) independently reproduces clustering, primary results and the omission account. This is local prospective approximation, with full original state and model still retained.

The [recurring follow-up](../span_recurrent/FINDINGS.md) now preserves all paired atoms as signed-byte rows and substitutes the fixed span throughout three linked turns, with original C17 K/V poisoned and native fallback refused. At 17,984 bytes, mean numerator error falls to 2.086% versus centroids' 48.438%; all six sparse byte logit differences stay below 0.1%. Query feedback, new-memory compression and physical allocation savings remain open.

From the domains workspace:

```sh
rtk proxy python3 saturn/cmachine/composition/span_response/run.py stage
rtk proxy common/bin/common exec python saturn/cmachine/composition/span_response/run.py submit PAYLOAD
rtk proxy common/bin/common exec python saturn/cmachine/composition/span_response/run.py collect JOB_ID
rtk proxy common/bin/common exec python saturn/cmachine/composition/span_response/run.py verify RESULT_DIRECTORY
rtk proxy common/bin/common exec python saturn/cmachine/composition/span_response/audit.py RESULT_DIRECTORY
rtk proxy common/bin/common exec python saturn/cmachine/composition/span_response/evidence.py JOB_ID
```

Fifteen model-free constructor/visibility/math/import/token tests pass. The actual C predictor gives identical outputs after the original construction operands are overwritten. C syntax checking passes with three inherited unused-helper warnings. These are instrument checks; no answer-quality threshold stops evidence collection.
