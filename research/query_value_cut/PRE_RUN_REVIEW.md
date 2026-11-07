# Pre-run implementation review

2026-11-06. Read-only review of the seven-arm QK/value assay before staging.
No model run or scientific-source edit was performed by the reviewer.

No execution blocker was found in the reviewed driver, analyzer, worker, or
launcher. The proposed evidence supports the stated component-local experiment;
it does not add a full-model closure or semantic-quality claim.

## Checked mechanics

- `advance` clears the coordinate-ready flag for every token. `qm_matvec` lazily
  computes Bn at the first selected factored map. In native-QK/compiled-V, this
  correctly occurs at V; the frozen native runtime still passes the same
  normalized receiver to Q, K, and V. The flag and independent coordinate-call
  counter prevent silent coordinate reuse across token advances.
- The policy dispatch matches NN, CC, CN, and NC. Selected affine MAC totals are
  3,145,728; 458,752; 819,200; and 2,981,888 respectively. Factored arms have one
  coordinate call and three output calls. Both per-path and whole-invocation
  counters include the 32 advances of native-wrapper parity.
- The frozen CUDA attention implementation leaves post-softmax probabilities
  contiguous as twelve rows of current length. At zero-based step t the capture
  offset is 6t(t+1), the block has 12(t+1) floats, and the total through n tokens
  is 6n(n+1) floats. C capture and Python slicing agree. Compact and zero arms
  correctly avoid reading the stale score buffer.
- Probability readback occurs after the native value mixture, before later
  layers reuse scratch. It reads actual FP32 softmax probabilities without
  recomputing or changing attention. Additional diagnostic synchronization and
  transfers remain paid work.
- The compact memory implementation and its C mechanics test match the prior
  authenticated payload byte for byte. Its selected-cache sentinel, causal
  append-before-read order, and replay remain in place.
- All eighteen candidates execute and seal before the three native references.
  Program arrays and candidate probabilities are included in the seal; native
  reference arrays are excluded by exact path prefix. Source128 bank loading
  replaces compilation, and model/report/file identities are checked before use.
- The source runtime and public support files are copied from the authenticated
  predecessor payload, not from mutable working runtime files. Source-manifest
  validation and the previous report's fixed SHA bind staging and submission.

## Factorial analysis and its qualification

The analyzer finds the maximal common consumed-ID prefix, then checks normalized
receiver bytes on that prefix. It also records the expected pairwise Q/K/V and
probability identities. Its order NN,CN,NC,CC and sign convention are correct.
The predicted interaction uses actual saved probabilities and FP64 value mixes;
the four native FP32 mixing remainders are retained separately. Initial-prompt
and maximal-common-prefix aggregate scopes are reported independently.

The exact `(P_C−P_N)(V_C−V_N)` interpretation requires both
`operand_pairs_byte_exact` and `probability_pairs_byte_exact`. If either is false,
preserve all data and report an instrument mismatch for that identity. The
remaining discrepancy could then contain pair mismatches and must not be called
only native value-mixing roundoff. The current report exposes these flags and
the residual; the final interpretation must use them. This is a qualification
of a specific algebraic claim, not a numerical-quality gate on the experiment.

The probability normalization discrepancy is correctly retained as a diagnostic.
Likewise, a large relative formula error when the true interaction is near zero
does not imply a large absolute defect. The native-scaled remainder and signed
cross terms supply the needed context. Layer-exit and logit interactions are
policy measurements, not applications of the linear value-mixture identity.

## Resource and custody geometry

Using the existing authenticated tokenized inputs and full budgets, maximum path
lengths are 107, 108, and 132. The worker's bound independently recomputes to
133,485,172 bytes versus a 150,994,944-byte (144 MiB) archive limit, leaving
17,509,772 bytes of headroom. The twelve maximum probability arrays contribute
3,924,864 bytes. Their temporary per-path capacity512 host allocation is
6,303,744 bytes, freed between paths. Raw arrays, full logits, compact snapshots,
the program bank, and an 8 MiB metadata/payload allowance are included in the
bound; the emitter separately enforces the complete physical archive size.

The launcher requests CUDA, one resident model, one CPU thread, physical batch1,
strict preflight, no retries, and direct-job collection. The explicit 6144 MB
RAM/VRAM reservations cite the measured same-profile predecessor and its 1.1
enforcement ceiling. The slightly smaller bank and small host recorder do not
change the material GPU geometry. Beast pinning is justified by the local model
bundle/tokenizer. The 120-second estimate and 840-second hard timeout fit the
bounded 21-path workload; an unrelated progressing sound job should remain
untouched.

The historical joining step must retain all fifteen repeated endpoints,
including exact histories, full logits, normalized/QKV/attention/layer-exit
ports, compact snapshots, and cache-bound readout records. These joins establish
that instrumentation and bank reuse preserved the old endpoints. They do not
turn the repeated development cases into new held-out specimens.

## Reviewed source identities

These hashes identify the files inspected before staging, not a substitute for
the eventual frozen payload manifest:

| File | SHA256 |
|---|---|
| driver.c | f15b92db08eee672a1b308dbd0ff4461666538b00797260a08498238c98aa412 |
| analysis.py | 26f77650abd3cee49bbc0848522b0d0928cd695e40d654a109e30dce1fe6545e |
| worker.py | 322ba227316c218a375749d3195eb23d47d48fa1c67f5b95175ae1c265b45e45 |
| run.py | bdb84d1169659fde208a6b16ae3fad76b379e50f4a22faf6b38e6f6a2b732448 |
| PLAN.md | 03b21509f257ff418794307f6d37285db0aaf6dcbe4c80a43cc1aaa05bea9fbc |

The reused program hashes and saved-only diagnostic provenance are recorded in
[ADVICE.md](ADVICE.md). No scientific input was changed by this review.
