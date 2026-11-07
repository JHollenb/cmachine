# Composition of a frozen MLP program

The subsequent [whole-turn assay](whole_turn/README.md) now applies the same frozen program while ingesting a complete user correction and two independent later questions. All policies, including zero, retain green/red. PROJECT improves pre-answer full-logit fidelity over zero and reverses after consumed EOS. Exact same-text lower-port comparisons qualify the local reference; [new findings](whole_turn/FINDINGS.md) retain those mixed trends and the remaining compact-state gap.

This exploratory C assay compares one application of the supplied B3 MLP16 program with repeated application through its own generated continuation. Native equations and an arithmetic correction control run alongside both. The [completed findings](FINDINGS.md) include a subsequent zero-write control: all policies retain blue/391; compact prose completes, zero is truncated at48tokens, and large local write errors remain. The [plan](PLAN.md) fixes independent recall, arithmetic and prose questions, with8/8/48-token budgets and full consumption of every emitted token.

The experiment preserves the completed formation runtime under a content-addressed payload. It deliberately leaves the concurrently developed writer API untouched. The runtime source and276,536-byte B3 bank come from authenticated [formation evidence](../results/job-0e69e4a9b62e/report.json). The bank is imported once; source/calibration weight projections are not repeated. Native source generation must reproduce the historical token IDs, post-answer cursor and complete valid-KV hash before candidate execution begins.

For each actual receiver r, the program solves coefficients in the supplied three-column basis, forms original-weight G/U images from those coefficients, applies the current native RMS scale, and evaluates original SiLU, product and down projection. PROJECT omits the live residual complement. CORRECT explicitly evaluates its G/U images, pays the original two dense projections, and adds them before the nonlinear consumer.

The native lane uses ordinary C equations. Once and recurring share exactly the same first substitution; full logits and valid-KV state verify that equality. Recurring then applies the approximation to every later generated input, including final EOS consumption. The corrected lane follows the same recursive schedule with its current omission restored. Dense residuals, full KV, original model weights and most learned computations remain, so this does not establish a compact sufficient state or general chat fidelity.

`analysis.py` retains the distinction between identical token histories and diverged histories. For identical histories, the measured layer16 entry, receiver, normalized input and KV16 must remain byte-identical: the changed MLP lies after those computations. Higher-layer memory may differ. Once an emitted token changes and is consumed, later equal step numbers describe different input histories. Task correctness, native agreement and stopping are reported separately.

The [model-free checks](mechanics/qualification.json) and [import audit](mechanics/import-verification.json) cover lifecycle restoration, token authority and complete frozen-operand byte retention. The local importer uses ASan/UBSan with a malloc-backed GPU stub; it evaluates no model. Native execution uses one finite mrun CUDA lease, one resident FP32 C model and batch1. Captures and complete-state hashes make its wall time unsuitable as a serving benchmark.

Run through the shared environment:

```sh
rtk proxy python3 saturn/cmachine/composition/run.py stage
rtk proxy common/bin/common exec python saturn/cmachine/composition/run.py submit PAYLOAD_DIRECTORY
rtk proxy common/bin/common exec python saturn/cmachine/composition/run.py collect JOB_ID
rtk proxy common/bin/common exec python saturn/cmachine/composition/run.py verify RESULT_DIRECTORY
```

The collector retains the scheduler-sealed report/log, all selected arrays, source snapshot, whole-invocation Saturn receipt and frozen program. Verification imports the exact executed analyzer and does not evaluate a model. A completed run's observations then pass through the standard Saturn evidence sweep with target-stream coverage kept separate from scientific interpretation. `stage-zero` prepares only the three [posthoc deletion controls](ZERO_PLAN.md). `derive.py` builds the additive [cross-policy study](study/study.json) from both authenticated reports, and snapshots its analysis dependencies without changing original evidence.
