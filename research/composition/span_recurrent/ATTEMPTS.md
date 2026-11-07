# Execution attempts

## job-a4800d0cf34e — harness boundary refusal

The [first submitted job](runs/job-a4800d0cf34e/report.json) compiled and completed native source/correction plus all three native linked turns, then stopped before poisoning or any candidate execution. It is incomplete evidence for the requested recurring comparison. Sources, raw partial output, scheduler record and sealed log are retained under that run directory.

The new root check incorrectly required phase1 after native rollback. The pinned runtime's `restore` deliberately sets phase0; its actual `at_boundary` contract accepts either phase0 or phase1 at a positive cursor with a valid pending token. The root check now accepts those two completed-token phases while retaining depth0, cursor93, invalidated legacy checkpoint and epoch-overflow checks. No scientific threshold or exact34 parity requirement changes. The C runtime and model computation remain pinned.

The first job is not retried automatically. A new source-bound finite submission records the corrected harness separately. The earlier local stages and failed source snapshot remain immutable.
