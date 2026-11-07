# Source-only gate laws in the C state machine

The C runtime now has an explicit `QwenGateLaw` interface. A frozen source gate supplies per-channel sigmoid-Taylor2 coefficients and a finite anchored cubic expression for SiLU. Each invocation consumes the branch's actual dense gate/up operands, preserves its residual and writes through the original down map. It binds the actual retained parent separately from the reusable law. This is a new analytic program; the earlier B3 API instead approximates two input projections.

The [protocol](PLAN.md), [driver](driver.c), [worker](worker.py) and [offline analyzer](analysis.py) define the experiment. The primary comparison uses once, recurring and zero-write paths through two complete turns, including user/header ingestion. Native references and exact selected-site diagnostics run after all primary paths are sealed. Candidate replay, native reconstruction on the same consumed IDs and native single-step execution on the candidate's actual cache remain separate comparisons.

## Run and inspect

From the workspace root:

```sh
rtk proxy make -C saturn/cmachine state-mechanics
rtk proxy common/bin/common exec python -m unittest discover -s saturn/cmachine/gate_law -p test_analysis.py
rtk proxy common/bin/common exec python saturn/cmachine/gate_law/run.py stage-smoke
rtk proxy common/bin/common exec python saturn/cmachine/gate_law/run.py submit PAYLOAD
rtk proxy common/bin/common exec python saturn/cmachine/gate_law/run.py collect JOB
rtk proxy common/bin/common exec python saturn/cmachine/gate_law/run.py verify RESULT_DIRECTORY
rtk proxy common/bin/common exec python saturn/cmachine/gate_law/evidence.py JOB
```

Use `stage-panel` only after a successful smoke. Staging freezes current source and a hash-checked public Saturn package. One mrun lease uses one resident model, batch1, one CPU thread, CUDA and context512, with retries disabled. The declared RAM/VRAM exception is measured from the same C FP32 implementation. A pre-execution bound accounts for maximum reply lengths and raw evidence capacity. References omit unused intermediate captures to keep custody bounded; they still execute the complete native equations.

## Computation boundary

For hidden dimension1536, MLP width8960, 28 layers and vocabulary151936, native learned-linear work per consumed token is

$$
28[2(1536)^2+2(256)(1536)+3(8960)(1536)]+151936(1536)
=1{,}543{,}569{,}408\ \text{MACs}.
$$

This excludes attention score/value products, normalization, RoPE, elementwise work and reductions. C executes197 dense matrix-vector products per token. The polynomial law retains all197 and all these learned-linear MACs. It replaces8,960 selected-site exponential evaluations with polynomial work and five coefficient-array reads; 27 native SwiGLU evaluations remain. The law occupies179,200 GPU bytes per frozen anchor, in addition to full original weights/state. Temporary source-control/rebuild anchors are paid separately.

Zeroing the selected whole MLP skips three matrix-vector products and41,287,680 MACs, about2.6748% of learned-linear work. The earlier B3 PROJECT program skips only two of those matrices, about1.7832%, while retaining its down map and paying basis/QR work. Neither statistic is a measured latency speedup. Per-path timings include capture, synchronization, hashing and file I/O; candidate replay timings additionally include native local probes.

## Limits and mechanics history

The model-free lifecycle exercises native no-op equivalence, exact source-anchor reconstruction, wrong-parent refusal and actual zero-work deletion under ASan/UBSan. The complete real-model lifecycle exercises the same assertions on synthetic tokens. Gate-law nonfinite refusal currently leaves a partial token after attention work; the finite assay exits and preserves partial evidence rather than claiming failure-atomic branch recovery. Ordinary serving never enables this law automatically.

The first smoke, [`job-56b28564a9b3`](runs/job-56b28564a9b3/report.json), passes the pretrained lifecycle but fails before the ordinary comparison because a tokenizer structured return was incorrectly written as token IDs. Its report and payload remain immutable. Explicit flat-ID handling fixes that harness error. The corrected [`job-a149cfcf8097`](runs/job-a149cfcf8097/report.json) succeeds in40.46scheduler seconds, with4,239.6MB RSS and6,378MB VRAM. All151,936 ordinary first logits match the preserved prior binary byte for byte; generated tokens and checkpoint replay match. The [offline verification](runs/job-a149cfcf8097/verification.json) authenticates68sources,7raw files and the whole-invocation Saturn receipt. This is engineering evidence, not a scientific result for the future-law approximation.

The [research comparison](../../../obsidian/blog/2026-11-06-115221-how-we-get-from-state-prediction-to-autonomous-chat.md) explains why own-token execution, useful replacement content and compact sufficient state are different achievements. The actual scientific results are recorded separately after collection. The active installed Beast release is unchanged.
