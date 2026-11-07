# Own-query program and bounded causal memory

An isolated C experiment replaces one real Qwen attention component with a frozen QKV program and an updating 98,464-byte memory. It evaluates future queries produced from its branch's actual receiver, consumes its own tokens and feeds the original native consumer. Full-model computation, weights and 27 other cache layers remain paid. This is an executable approximation boundary, with measured substantial fidelity gaps.

- [Frozen protocols](PLAN.md), [mathematics](MATH.md), [results](FINDINGS.md), [Astra review](REVIEW.md).
- [C execution wrapper](driver.c), [memory implementation](memory.h), [checkpoint-only compiler](weight_basis.py), [expanded source compiler](source_basis.py).
- [Independent law replay](analysis.py), [saved-only controls/costs](study.py), [evidence integration](evidence.py).
- [Initial activation 32 result](runs/job-2230480e1c18/report.json), [checkpoint-only 32/128 result](runs/job-077becbef4c9/report.json), [source-span refinement](runs/job-c7334bbc5bb8/report.json).

Each collected result preserves its actual executed `payload/`. Working sources follow the newest experiment and must not be substituted for earlier payloads. Source and raw manifests, candidate-before-native seals, whole-invocation Saturn receipts and offline recomputation bind the reports. Subsequent study/evidence/review artifacts are additive and separately identified.

The next [QK-versus-V cut](../query_value_cut/FINDINGS.md) is now completed. Both native-map restorations improve first readouts, with an exactly accounted address/content interaction and a later ranking reversal. Native values with compiled Q/K offer a useful local work/fidelity tradeoff; bounded-memory transfer of that mixed policy remains untested. Full-model receiver closure remains open.

From the domains root:

```sh
rtk proxy common/bin/common exec python -m unittest discover -s saturn/cmachine/query_memory -p 'test_*.py'
rtk proxy cc -O1 -g -Wall -Wextra -Wpedantic -Werror -fsanitize=address,undefined -ffp-contract=off saturn/cmachine/query_memory/test_memory.c -lm -o /tmp/query-memory-mechanics
rtk proxy /tmp/query-memory-mechanics
rtk proxy common/bin/common exec python saturn/cmachine/query_memory/run.py stage
rtk proxy common/bin/common exec python saturn/cmachine/query_memory/run.py submit ABSOLUTE_STAGED_PAYLOAD
rtk proxy common/bin/common exec python saturn/cmachine/query_memory/run.py collect JOB
rtk proxy common/bin/common exec python saturn/cmachine/query_memory/run.py verify RESULT_DIRECTORY
rtk proxy common/bin/common exec python saturn/cmachine/query_memory/study.py RESULT_DIRECTORY --previous saturn/cmachine/query_memory/runs/job-2230480e1c18
rtk proxy common/bin/common exec python saturn/cmachine/query_memory/evidence.py JOB
```

Real-model execution is one source-frozen mrun CUDA lease on Beast, one resident model, one CPU thread, strict preflight and no retries. These commands do not install a serving policy. The public `domains/cmachine` extraction is a separate checkout; these experimental additions have not been imported there.
