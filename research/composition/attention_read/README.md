# Native attention reads the two episode states

The [completed findings](FINDINGS.md) follow the [formation × question factorial](../formation_read/FINDINGS.md) through layer17 attention on the same Qwen2.5-1.5B-Instruct C FP32 specimen. Four worlds, two questions and pre-answer/consumed-EOS clocks give sixteen native traces. Score interaction is confined to correction history; softmax spreads probability interaction across unchanged source/question columns. A query-conditioned numerator/normalizer update accounts for native context changes to2.81×10⁻⁷–6.45×10⁻⁷ relative discrepancy. Receiving MLP interaction opposes the receiver throughout, while net interaction grows pre-answer and shrinks post-EOS. Full weights and KV remain.

[driver.c](driver.c) captures biased post-RoPE Q/K, valid K/V, native probabilities, context, projected write and receiving MLP17. [attention_cuda.cu](attention_cuda.cu) adds an explicitly selected pre-softmax score copy around the pinned arithmetic. [analysis.py](analysis.py) tests bilinear score support, softmax spread and the probability/value factorial, separating finite dot-product, scale, projection and addition effects. [run.py](run.py) authenticates both historical reports and the original runtime/bank. No serving build is changed.

Job [job-812a0486995d](runs/job-812a0486995d/report.json) completes in29.49scheduler seconds at6378MB peak VRAM. All20historical full-readout/cache joins are exact; [verification](runs/job-812a0486995d/verification.json) authenticates73sources and301raw files. The [prospective plan](PLAN.md), [independent Astra audit](REVIEW.md), [saved-only signed MLP account](runs/job-812a0486995d/mlp-interaction-audit.json), and [evidence sweep](index/job-812a0486995d/evidence-run.json) retain scope and custody. The result is a conditional native-read account, not compact chat state or a serving-speed benchmark.

The completed [prospective span-response follow-up](../span_response/FINDINGS.md) constructs an eight-centroid memory-only bank before two new queries exist. Its 16,480-byte program replaces original span K/V at one later read, improving first-logit fidelity over deletion while retaining the dense complement and full original state. The saved-only omission account identifies lost score dispersion and finite score/value dependence; ongoing compact chat remains open.

From the domains workspace:

```sh
rtk proxy python3 saturn/cmachine/composition/attention_read/run.py stage
rtk proxy common/bin/common exec python saturn/cmachine/composition/attention_read/run.py submit PAYLOAD
rtk proxy common/bin/common exec python saturn/cmachine/composition/attention_read/run.py collect JOB_ID
rtk proxy common/bin/common exec python saturn/cmachine/composition/attention_read/run.py verify RESULT_DIRECTORY
rtk proxy common/bin/common exec python saturn/cmachine/composition/attention_read/audit.py RESULT_DIRECTORY
rtk proxy common/bin/common exec python saturn/cmachine/composition/attention_read/evidence.py JOB_ID
```

The staged payload supports model-free `python -m unittest -v test_attention test_token_authority` and `cc -std=c11 -Wall -Wextra -Wpedantic -ffp-contract=off -Iruntime -fsyntax-only driver.c`. Fourteen checks pass, including the frozen payload's actual receipt-bridge import closure. Four inherited unused-helper warnings are retained. An initial singleton-axis error in the diagnostic value expansion was caught by the toy matrix test and corrected before model submission; the initial staging remains. A first worker attempt stopped before model execution because the stage omitted saturn_bridge.py; [attempt history](ATTEMPTS.md) retains that null evidence and the corrected submission separately.
