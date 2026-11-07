# Math choice intervention evidence

Final expanded panel: **`job-a117c5a7b06c`**, guarded Beast CPU worker. [Findings and reproduction](../../MATH-CAUSAL.md).

| Record | Scope |
| --- | --- |
| [Report](report.json) | Original complete worker report and all 2,790 raw file identities |
| [Scheduler](scheduler.json) and [source manifest](source-manifest.json) | Source/input/SDK hashes, declared/sealed/executed payload identity, admission and measured resources |
| [Independent verification](verification.json) | Cut/receipt state fingerprints, graph/features/choice identities, source-map/binary checks, C memo/invocation links and exact/numerical consumers |
| [Scalar product](scalar-product.json) | All six boundary swaps, 64 single exchanges, doses, alternatives, restoration/order controls and reciprocal suffix |
| [Planted scalar](scalar-planted.json) and [ordered matrix](ordered-matrix.json) | Equal-cost contexts across all reached boundaries |
| [Repair capsule](repair-cut.json), [frontier](repair-frontier.json), [graph and features](repair-epoch.json) | The reproduced second-round parent and candidate addresses |
| [One repaired branch](single-repair.json) and [Saturn receipts](single-repair-receipts.json) | One replacement reaches the fully factored expression through the unchanged V2a suffix and emitted C consumer |
| [Numerical example](consumer-example.json) and [reference](reference-program.map.json), [baseline](baseline-program.map.json), [repair](repair-program.map.json) maps | All 32 input/source/C rows and physical invocation receipts; emitted instruction counts 25, 15 and 11 |
| [History](history.json) | Initial panel and expanded control panel with their distinct staged sources and protocol coverage |
| [Mechanics](mechanics.json) | Generated local adapter/collector regression scope, separate from learned-artifact evidence |

The round-2 batch swap lowers cost 9→5. Of 64 single exchanges, 48 reach cost 5, 14 reach 7 and two stay at 9. All nine tested alternative-membership controls stay at 9. Later batch swaps reach cost 11, and the reciprocal V1 suffix recovers cost 5. All completed endpoints preserve the exact supplied polynomial and tested C numerical outputs. No training or general mathematical improvement is claimed.

All 52 unique Python tests pass locally. The two discovery commands overlap on five generated choice fixtures, which are counted once. The C execution sources are unchanged by the causal extension; their existing model-free CI lane remains required.

The full ignored `results/job-a117c5a7b06c` directory retains about 133 MiB of immutable raw evidence, all 299 executable programs, arrays, record banks, frozen checkpoints and the staged source payload. This compact tracked folder is not a full replay archive: the repair epoch references raw feature bytes held in that result directory. Initial `results/job-a72757b03ba8` remains preserved with its original 108-branch protocol; later controls and source fixes are not attributed to it.

```sh
rtk proxy ../common/bin/common exec python tools/verify_math_causal.py \
  results/job-a117c5a7b06c
```
