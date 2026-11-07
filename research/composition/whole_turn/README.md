# Whole-turn ingestion with a frozen MLP program

The [completed findings](FINDINGS.md) extend the earlier [generated-token composition assay](../FINDINGS.md) to every token of a later user correction and two independent follow-ups. Native, recurring B3 PROJECT, recurring zero-write and recurring CORRECT all produce OK, green and red with EOS. The zero control retains the same behavior. B3 improves pre-answer full-logit fidelity relative to zero but has larger differences after consumed EOS; local write errors remain large. Full weights and KV remain.

[Plan](PLAN.md), [driver](driver.c), [worker](worker.py), [saved-only analysis](analysis.py), [additive audit](audit.py), and [Astra review](REVIEW.md) define the experiment. The model runtime, import helpers and frozen bank are inherited from authenticated historical payloads, leaving the shared runtime and installed chat build untouched. All primary outputs seal before nine paid dense same-token diagnostics.

The single Beast job is [job-a071c7a43a38](runs/job-a071c7a43a38/report.json). Its [offline verification](runs/job-a071c7a43a38/verification.json) authenticates67sources and4215raw files; the [additive audit](runs/job-a071c7a43a38/additive-audit-v2.json) adds declared-template joins, exact native diagnostic replay and signed policy effects without changing the executed report. The [evidence sweep](index/job-a071c7a43a38/evidence-run.json) and [verification](index/job-a071c7a43a38/evidence-verify.json) cover270source-bound rows, with stream coverage separate from scientific claims.

The subsequent [formation × question factorial](../formation_read/FINDINGS.md) adds only the two missing hybrid policies, authenticating their correction parents against this run. PROJECT formation followed by dense questions retains about 10% pre-answer full-logit difference; dense formation followed by PROJECT questions gives 13–17%. All answers remain green/red. Exact local controls separate changed retained history from current MLP16 computation, and signed interaction partly cancels the two episode effects. The [standalone blog](../../../../obsidian/blog/2026-11-06-125544-a-frozen-mlp-program-updates-a-conversation.md) connects both experiments to compact sufficient chat.

From the domains workspace:

```sh
rtk proxy python3 saturn/cmachine/composition/whole_turn/run.py stage
rtk proxy common/bin/common exec python saturn/cmachine/composition/whole_turn/run.py submit PAYLOAD
rtk proxy common/bin/common exec python saturn/cmachine/composition/whole_turn/run.py collect JOB_ID
rtk proxy common/bin/common exec python saturn/cmachine/composition/whole_turn/run.py verify RESULT_DIRECTORY
rtk proxy common/bin/common exec python saturn/cmachine/composition/whole_turn/audit.py RESULT_DIRECTORY
rtk proxy common/bin/common exec python saturn/cmachine/composition/whole_turn/evidence.py JOB_ID
```

Model-free token-authority tests run from the staged payload with `python -m unittest -v test_analysis`. Local C syntax checking uses `cc -std=c11 -Wall -Wextra -Wpedantic -ffp-contract=off -Iruntime -fsyntax-only driver.c`. Eight tests and syntax checking pass before submission; the three warnings concern inherited unused helpers. An initial missing zero-buffer declaration was corrected before the only model job, with earlier staging retained. This is an exploratory execution instrument, not a serving policy or compressed-state implementation.
