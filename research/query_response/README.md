# Same-cost response-aware query image

One bounded source-only query update on the existing B128 receiver code. The objective follows the actual compiled-key/native-value compact reader rather than minimizing query-vector error. [Protocol](PLAN.md), [math](MATH.md), [Astra advice](ADVICE.md), [compiler](compiler.py), [C execution](driver.c), [independent replay](analysis.py).

Baseline is the authenticated least-squares query image from [job-bd80dd16c1d5](../query_program/FINDINGS.md). Source rows and B already used the same three calibration trajectories. Source measurements are in-sample construction evidence; target outcomes never enter fitting or source-only dose selection.

Run through `common exec python saturn/cmachine/query_response/run.py stage`, then `submit PAYLOAD`, `collect JOB`, `verify RUN_DIRECTORY`. All real execution uses mrun. Tiny math mechanics: `common exec python -m unittest discover -s saturn/cmachine/query_response -p 'test_*.py'`.

Driver policy labels are retained for continuity with the previous wrapper: `values-native` means LS-query/compiled-K/native-V full memory; `values-native-memory` is its compact counterpart; `query-fit`/`query-fit-memory` use the new response image. Names do not mean the baseline query is still the original orthogonal image. Program provenance binds the exact LS bytes.

Serving budget is unchanged: 819,200 selected QKV MACs/update and 98,464-byte selected logical memory. CPU source construction, gradient/line-search work, diagnostic captures, the unchanged dense receiver and remaining model stay paid. No serving default is changed by the experiment.
