# Recurring compressed memory-span reads

The later [owned compressed-append experiment](../compact_append/FINDINGS.md) addresses this study's explicit next steps: actual candidate-row encoding and immediate consumption, partial-token rollback, absent selected dense slots and two-layer changed-query composition. It retains this fixed-span result and its original evidence unchanged.

The [completed findings](FINDINGS.md) and [Astra review](REVIEW.md) report 89 recurring reads per policy through three linked turns. Byte-bank mean span numerator error is 2.086%, versus centroids' 48.438%, at 17,984 versus 16,480 serialized bytes. All six measured byte full-logit differences are below 0.1%; all answers also match deletion. Original selected-span K/V17 stays poisoned. All Q17 queries stay native on matching token histories, and original allocation/weights remain. This establishes recurring use of a compressed fixed span, without compressed append, changed-query feedback or serving savings.

The [plan](PLAN.md) keeps all 34 paired key/value atoms in a source-only signed-byte representation, instead of collapsing their score dispersion and score/value dependence into centroids. The serialized bank is 17,984 bytes versus 69,632 native span bytes and 16,480 bytes for the prior eight-centroid representation. This is a standard row-quantized K/V baseline, not a recovered semantic law.

Five policies advance a linked three-turn conversation from one native correction parent. Every subsequent layer17 read uses its selected representation, including user/template ingestion, own generated tokens and EOS. After the native control, original correction-span K/V17 is overwritten with quiet NaNs. A new owned root invalidates old checkpoints, and a CUDA guard refuses native fallback at that layer. Original cache allocation, other layers, dense complement and model weights remain; physical allocation savings and arbitrary compact chat are not claimed.

[byte_bank.h](byte_bank.h) contains the memory-only C constructor and bank/query-only predictor. [span_cuda.cu](span_cuda.cu) decodes bytes directly into the explicitly counted packed-attention workspace. [driver.c](driver.c) owns construction order, poison roots, recurring execution and own-token authority. [analysis.py](analysis.py) audits sealed primary predictions and each policy's own future queries against original saved span operands only afterward.

From the domains workspace:

```sh
rtk proxy python3 saturn/cmachine/composition/span_recurrent/run.py stage
rtk proxy common/bin/common exec python saturn/cmachine/composition/span_recurrent/run.py submit PAYLOAD
rtk proxy common/bin/common exec python saturn/cmachine/composition/span_recurrent/run.py collect JOB_ID
rtk proxy common/bin/common exec python saturn/cmachine/composition/span_recurrent/run.py verify RESULT_DIRECTORY
rtk proxy common/bin/common exec python saturn/cmachine/composition/span_recurrent/evidence.py JOB_ID
```

The successful result is [job-d12ce01f244f](runs/job-d12ce01f244f/report.json). Its [verification](runs/job-d12ce01f244f/verification.json) binds 76 sources, 1,107 raw files and exact same-environment analysis reproduction. A local strict replay refusal from FP32 NumPy log-count differences is [preserved](runs/job-d12ce01f244f/verification-incident.json); no tolerance was loosened. For this named incident, [audit_replay.py](audit_replay.py) submits/collects a zero-model CPU replay of the frozen analyzer, and [verify_replay.py](verify_replay.py) binds both scheduler seals and the exact canonical analysis digest. `run.py verify` uses that qualified path when its authenticated receipt is present:

```sh
rtk proxy common/bin/common exec python saturn/cmachine/composition/span_recurrent/audit_replay.py submit RESULT_DIRECTORY
rtk proxy common/bin/common exec python saturn/cmachine/composition/span_recurrent/audit_replay.py collect RESULT_DIRECTORY
rtk proxy common/bin/common exec python saturn/cmachine/composition/span_recurrent/run.py verify RESULT_DIRECTORY
```

The original [failed boundary assertion](ATTEMPTS.md) and original frozen mathematical note remain in their source snapshots. [MATH.md](MATH.md) includes Astra's two later wording clarifications. Main execution takes 23.78 scheduler seconds; the saved-only verification replay takes 3.26 seconds with 85.6 MB RSS and no GPU/model calls.

In the staged payload, 13 model-free checks pass with `python -m unittest -v test_recurrent test_token_authority`. They compile the actual C byte constructor/predictor, verify tie handling and zero rows, check prediction after original inputs are erased, compare an independent response, and retain token/import authority checks. C syntax checking passes with four inherited unused helpers. No quality gate blocks evidence collection.
