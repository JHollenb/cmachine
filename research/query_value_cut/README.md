# Query/key addressing versus value content

The next cut after the [query-memory program](../query_memory/FINDINGS.md) restores Q/K and V separately inside the real Qwen consumer. It distinguishes errors in attention allocation from errors in the values being read before designing another compiler or adding memory moments.

| Policy | Q/K maps | V map | Selected memory |
|---|---|---|---|
| NN/native | Original | Original | Full |
| CC/compiled | Frozen source 128 | Frozen source 128 | Full |
| CN/values-native | Frozen source 128 | Original | Full |
| NC/address-native | Original | Frozen source 128 | Full |
| Memory | Original | Original | Bounded recent/prototypes |
| Compiled-memory | Frozen source 128 | Frozen source 128 | Bounded recent/prototypes |
| Zero | Original | Original | Attention return zeroed |

Mixed policies pay native component computations. Every branch owns its actual receiver, memory and token continuation. The full native lower-layer receiver and upper consumer remain paid. Only common exact consumed-ID prefixes support the selected-attention QK×V factorization; full logits remain outcomes of recurring policies with their own upper cache history.

**Completed:** [job-27a597ebd5f8](runs/job-27a597ebd5f8/report.json) executes 21 paths/2113 token advances. Both native-map restorations improve all first readouts, with a prose ranking reversal between local attention and full logits. Fifteen historical endpoints/204 arrays match, and the actual captured-probability interaction closes with native mixing remainders. Later outcomes remain mixed and arithmetic stays wrong. [Findings and costs](FINDINGS.md), [Astra audit](REVIEW.md), [figure](runs/job-27a597ebd5f8/query-value-cut.png).

[PLAN.md](PLAN.md) freezes scope and economics. [ADVICE.md](ADVICE.md) contains Astra's design critique and a saved-only precursor; [MATH.md](MATH.md) defines the interaction and numerical remainder. [driver.c](driver.c) is an isolated wrapper around the exact predecessor runtime. [analysis.py](analysis.py) reconstructs memory, validates token authority and checks actual probability/value mixtures.

From the domains root:

```sh
rtk proxy common/bin/common exec python -m unittest discover -s saturn/cmachine/query_value_cut -p 'test_*.py'
rtk proxy common/bin/common exec python saturn/cmachine/query_value_cut/run.py stage
rtk proxy common/bin/common exec python saturn/cmachine/query_value_cut/run.py submit ABSOLUTE_STAGED_PAYLOAD
rtk proxy common/bin/common exec python saturn/cmachine/query_value_cut/run.py collect JOB
rtk proxy common/bin/common exec python saturn/cmachine/query_value_cut/run.py verify RESULT_DIRECTORY
```

One strict-preflight mrun CUDA lease, one model, batch 1, one CPU thread, no retries and no installed serving change. Executed payloads and raw results remain immutable; later interpretations and audits are additive.
