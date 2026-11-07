# Frozen math intake evidence

Final run: **`job-2a7aae82cec8`**, Beast CPU, guarded shipped payload, strict preflight and kernel memory enforcement. [Full findings and reproduction](../../MATH.md).

| Record | Scope |
| --- | --- |
| [Report](report.json) | Original complete worker report, raw file hashes, five cells, frozen FieldLM consumers and all 18 search arms |
| [Scheduler](scheduler.json) / [source manifest](source-manifest.json) | Exact source/input hashes, declared/sealed/executed payload identity, command, reservation and measured resources |
| [Independent verification](verification.json) | 434 raw files, 127 binary/map pairs and 4,359 C invocations; coefficient audits, query masks, score rankings, numerical observations, work counts and C continuation inputs |
| [Exact cell map](learned811.map.json) / [symbols](learned811-symbols.json) | Actual six-instruction captured computation with producer/consumer/source bindings |
| [V1 map](scorer-v1.map.json) / [V2a map](scorer-v2a.map.json) | Separate feature order, frozen checkpoint identities, normalization and eight-instruction scorers |
| [811](learned811-algebra.json), [1597](learned1597-algebra.json), [137](approx137-algebra.json), [149](approx149-algebra.json), [163](approx163-algebra.json) | All 64 original and deployed ordered coefficients per cell; exact and approximate results remain distinct |
| [History](history.json) | Four completed development panels with their original payload/report hashes and differing protocol coverage |
| [Mechanics](mechanics.json) | Generated local regression fixtures and sanitizer checks, separate from trained-artifact evidence |
| [Saturn run](evidence-run.json) / [verify](evidence-verify.json) | Standard offline evidence instrument; target-streams-only coverage, no semantic certification |

Both exact cells match source and matrix truth on all 272 bounded pairs; three approximate cells retain nonzero residuals and source/C errors. All 16,241 actual FieldLM reads match native. Two acquired concepts and 13 law bodies feed original host consumers; three supplied primitive records remain separate. V1's scalar cost of 5 beats matched random/enumeration at 9 and ties the heuristic at 5; V2a gives 9. The matrix reduction is shared by every arm: four learned-cell calls become one, 23 C instructions become 10, and nominal linear MACs fall 336→84 with zero bounded output difference. These observations do not establish general reasoning improvement or a speedup.

Maximum scorer error is `4.57763671875e-5`, with one ordering difference inside the same selected set. Parsing/charts, frame extension, domains, rewrite rules, feature construction, candidate ordering policy, graph mutation and extraction remain host supplied. The exact polynomial certificate establishes formal coefficient identity, not unrestricted floating-point or uint64 execution. Source continuation differences are recomputed from stored reference observations; C continuation chains additionally bind to actual invocation inputs/outputs.

The compact tracked folder is **not a complete reproduction archive**. The ignored local directory `results/job-2a7aae82cec8` retains about 14 MB of collected raw evidence, all 127 small `.tcm` binaries, arrays, projected-record banks, checkpoint JSON and the staged source payload. The four original local result directories remain preserved. Old worker reports and verification records were not rewritten when the protocol improved. Early enumeration used a different breadth schedule; final enumeration shares the eight-per-round cap. Only the final run includes the strengthened in-worker binary audit and retained native continuation arrays.

From this worktree, offline verification needs no model forwards:

```sh
rtk proxy ../common/bin/common exec python tools/verify_math.py results/job-2a7aae82cec8
rtk proxy ../common/bin/common exec python tools/math_program.py \
  results/job-2a7aae82cec8/raw/learned811.tcm
```

`verify_math.py` compares graph/header/parameter tables with binary bytes before polynomial analysis, reconstructs record-query masks, joins arrays to C calls by input/output hashes, recomputes rankings and MACs, and preserves effect sizes without a quality threshold. New source loaders, mathematical specifications and operations still need explicit declarations and checks. Astra's final static review found no remaining must-fix defect after binary/map binding, matched budgets, work/continuation verification and exact-coordinate guards were added.
