# Testing a retained state against the next native computation

The completed [native assay](FINDINGS.md) finds three exact collisions of a proposed16byte MLP interface with substantially different native writes. The interface retains three B3 coefficients and native RMS scale. A supplied omitted-component rotation preserves the executed code, but the native nonlinear write, logits and cached state distinguish the receivers. Short own-token continuations remain identical. This is a local finite-interface counterexample, not a compact whole-chat implementation.

Start with the [blog](../../../obsidian/blog/2026-11-06-125119-what-the-compact-mlp-state-forgot.md), [state/read ledger](STATE-INTERFACE.md), [mathematics](MATH.md) and [bound native plan](PLAN.md). The [independent Astra review](ADVISOR-RESULTS.md) authenticates the evidence and challenges its interpretation.

## Execution and analysis

- [driver.c](driver.c) includes the authenticated original runtime without modifying its source, validates actual encoder bytes, selects candidates before outcomes and runs fixed-skip native consumers plus own-token branches.
- [geometry.py](geometry.py) generates the fixed source-only spherical candidates. Its source coefficient counts are diagnostics; GPU rho equality is required for a finite code witness.
- [worker.py](worker.py) builds and executes one C process within one finite mrun CUDA lease. Full model weights, residuals and K/V remain.
- [run.py](run.py) stages authenticated inputs, submits once, collects the source/raw archive and independently verifies identity plus numerical analysis. Its post-collection model check was added after submission; executed payload copies remain unchanged.
- [derive.py](derive.py) and [study/study.json](study/study.json) add explicit-denominator error profiles, current-row versus historical-memory timing, all-step sham checks and costs without editing the native report.
- [plot.py](plot.py) produces the two static scientific figures from retained authenticated arrays.
- [evidence.py](evidence.py) projects227source-bound scalar rows to the target-local Saturn evidence plane. Complete vectors/continuations remain in the raw tree and independent audit.

Use the shared environment:

```sh
rtk proxy common/bin/common exec python saturn/cmachine/closure/run.py stage
rtk proxy common/bin/common exec python saturn/cmachine/closure/run.py submit PAYLOAD_DIRECTORY
rtk proxy common/bin/common exec python saturn/cmachine/closure/run.py collect JOB_ID
rtk proxy common/bin/common exec python saturn/cmachine/closure/run.py verify RESULT_DIRECTORY
```

The historical completed run is [job-7b8a5933f1dd](runs/job-7b8a5933f1dd/report.json):25.36scheduler seconds,14.23C seconds,3790MB RSS and6378MB VRAM. It retains81payload sources,966raw records and the subsequently recovered exact [executed binary](runs/job-7b8a5933f1dd/binary/receipt.json). Reproduction is a new run; the existing artifacts suffice for reanalysis.

## Interpretation

Equal code with unequal next write proves this encoding insufficient for the declared intervention domain. The two-point triangle-inequality bound applies to every deterministic predictor from that same code and static inputs. It does not estimate natural-chat frequency or exclude richer compact states.

Paid complementary G/U information predicts the alternate write; it does not restore the source write. Native corrected recurrence is an arithmetic control with dense calls counted. Native attention and other layers continue to form the operands of the full-state host. A genuinely reduced next-state law remains a different claim.
