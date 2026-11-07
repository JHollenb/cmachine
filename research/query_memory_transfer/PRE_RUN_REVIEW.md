# Pre-run review: native-value gain through compact memory

2026-11-06. Reviewed the eighteen-path implementation and staged payload
`20261106T233847.157182Z/payload`. No model run or scientific-source edit was
performed by the reviewer. This review is additive documentation after staging.

No execution blocker remains. One formula-applicability correction identified
during review was applied before the final stage: a prior operand-pair mismatch
now invalidates subsequent per-row identity claims cumulatively, while keeping
all observations. This is necessary because every earlier V row participates in
later weighted mixtures.

## Primary execution

Policy7 correctly selects compiled Q/K, native V, and the original compact
reader. Its coordinate-ready flag resets every token; it pays one coordinate
projection, two compiled maps, and one native V map. Its selected affine count
is 819,200 MAC/update and its whole-token matvec count is 198. The same compact
append/read, sentinel, mass, transfer, snapshot, and token-authority machinery is
used as in the verified predecessor.

The loop runs policies 1,2,3,5,7 for each of three cases, seals these fifteen
candidates, then runs the three native references. The analyzer expects that
exact order. This implements the requested NN/CC/CN × full/compact panel without
accidentally retaining NC or zero branches. Nine compact paths yield thirty-six
boundary snapshots; nine full-attention paths retain packed native probabilities.

The serving `memory.h` is byte-identical to the predecessor. No membership lists,
lifted weights, retired-row reference scans, or extra corrections enter the C
compact reader. `audit.py` is invoked by analysis only after execution and
candidate sealing. Probability capture remains restricted to full-attention
paths, with the previously verified triangular layout.

## Membership and interaction audit

The audit records the evicted ring position before replacing its slot. It follows
the same key-only nearest-centroid rule and first-index tie behavior, appends the
position to the corresponding group's membership list, and retains the original
numerical state update through `Memory.append`. Recent rows and retired cluster
members then lift to a positive response weight over original positions.

As an independent model-free check on actual recorded model histories, the new
audit was applied to all 600 compact steps of `job-27a597ebd5f8`. It reproduced
all 24 saved binary snapshots, and every original row appeared exactly once in
each group's partition. Lifted weights summed to one within 6.66e-16. The largest
relative discrepancy between a lifted-original-V FP64 mixture and the saved C
output was 3.70e-8, retained as finite-profile remainder. This check exercises
actual mergers and ring reuse rather than only the synthetic unit test.

The four-corner ordering, interaction signs, and `(Ptilde−P)(Vc−Vn)` formula are
correct. The qualification checks normalized receivers, Q/K, full probability
pairs, native/compiled V pairs, compact key-state signatures, memberships, and
lifted weights. Applicability is now cumulative through the common prefix.
The four finite remainders include centroid accumulation and ordered mixing,
not just the final FP32 cast.

The analyzer separates the first 32 positions, where only numerical profiles
differ, from positions with multi-member merges. Full and compact native-V gains
use the same original NN/full logit reference and retain signed squared-error
changes. The interaction norm is not presented as a signed gain. Later divergent
histories remain marked as recurring-policy outcomes. The summary's all-six
history flag is conservative: a mismatch in the independent NN/compact control
can make it false even if the four value-transfer arms still share a history;
the separate common-prefix audit supplies their precise local scope.

## Source and resource geometry

The runtime/public support sources are copied from the authenticated predecessor
payload, and the four rank128 arrays are copied from its hashed raw bank. Staging
binds the prior report
`da86f60e282cfdf93cd72dffabd98c91cabb7bfb9c3b46450421cc08ca207052`,
the model identity, and each source/bank hash. The current job performs no new
source acquisition, basis fit, or image compilation. Historical costs remain
declared as paid.

Maximum token counts remain 107/108/132. The worker's adjusted physical-evidence
bound is 116,975,016 bytes versus a 134,217,728-byte limit, leaving 17,242,712
bytes. It includes all eighteen array streams, full readouts, nine compact paths'
snapshots, nine triangular probability streams, the bank, and an 8 MiB metadata
allowance. Lifted matrices are reconstructed transiently rather than added to
the archive. The emitter separately enforces the complete archive's file-byte
total.

CUDA declaration, one resident model, batch1, CPU thread1, strict preflight, no
retries, direct-job collection, and the measured 6144 MB reservation exception
are preserved. The predecessor's 4127.1 MB RSS/6380 MB VRAM peaks fit the stated
1.1 enforcement ceiling. This panel has fewer paths and unchanged material GPU
geometry. The 120-second estimate and 840-second timeout are bounded consistently
with the earlier measured panels.

## Final staged source identities

The following current files were checked byte-for-byte against the final staged
payload and its manifest after the cumulative-applicability correction:

| File | SHA256 |
|---|---|
| driver.c | 2284e8c81003ece63ad2fb6c36e3128c4f3f15392031397522453f52ebc32882 |
| audit.py | 8ba7bc367cf33091129d77faacd439740333164495677db88aa281374eeb9197 |
| analysis.py | d232067012ff2c4c06c18c2bed951d1c3328b2f7f1b9329da31cdd6c373b2f5b |
| worker.py | bbbe0f2a6e71ac796c08b7898734d47647c6e6afc4ff2b86ff77f6957f464f00 |
| run.py | d856fe396d01929b5ec37bd3bae3df47136c0821e73df52d1f9e3f6c322a6f8f |
| test_analysis.py | fcdd6df50ecd04377731e7178a9def7ee92ba1553595ea59d99dc7562e091683 |

Final numerical quality, preservation or reversal of the value gain, task truth,
and latency remain observations to collect. None is a pre-run discovery gate.
