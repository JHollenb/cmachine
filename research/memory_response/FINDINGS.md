# All-layer owned memory: retention survives before exchange is solved

The full frozen development panel `job-ceab8d5c9869` completed all four initial-binding/update histories, five policies and two independent question forks: 40 outcomes, 80 first/final readouts and 80 own-query attention packets. The original [hypothesis snapshot](runs/job-ceab8d5c9869/payload/HYPOTHESIS.md), [report](runs/job-ceab8d5c9869/report.json), [presentation tables](runs/job-ceab8d5c9869/presentation.json), [verification](runs/job-ceab8d5c9869/verification.json) and [updated blog](../../../obsidian/blog/2026-11-06-154253-a-hypothesis-for-cheaper-model-state-transitions.md) preserve protocol and results.

| Policy | Correct answers | Observed behavior |
|---|---:|---|
| Native | 4/8 | Initial binding retained in every answer; exchange moves all four margins correctly without changing decisions. |
| Byte full | 5/8 | One additional individual answer correct; neither exchanged pair correct. |
| Byte top four | 5/8 | Same answer pattern as byte full, with a different generated acknowledgement history in one corner. |
| Byte recent four | 0/8 | All responses malformed whitespace/number continuations. |
| History deleted | 4/8 | Both people answered blue; identical full logits across the four histories for each question. |

Native and deletion have the same accuracy but different context dependence. Byte policies' extra answer is not demonstrated rebinding or evidence of better native emulation. Native itself shows a signed update trend that does not cross the argmax boundary. Scientific truth, native fidelity and runtime closure remain separate.

On the sixteen native-query packets at layers 17/18, mean conditional context error is 2.091% for byte-full versus original full memory, 30.649% for top-four versus decoded-byte full memory and 144.915% for recent-four versus decoded-byte full memory. Byte-full own-query means are 2.132%, 31.444% and 144.146%. All 48 byte-serving packets have exact live-cache codec identity; native/deletion use FP32 serving memory, with byte reconstruction as an audit counterfactual. Actual CUDA context versus the corresponding saved FP64 replay differs by less than 6.7e-7 relative L2 across all 80 packets.

Byte-full same-ID first-readout full-logit error is 4.236–5.540% across 8 reads. Top-four has 37.145–59.665% error across 6 matched reads; two question inputs at different clocks remain own-history outcomes. All cases, emitted/consumed IDs, signed color contrasts and local tail decompositions are retained.

Native session allocation is 30,096,896 GPU bytes; byte-full/top-four use 9,354,756 each, saving 68.9%. Recent-four uses 804,100 GPU bytes plus 59,136 host snapshot bytes. Top-four still retains/scans all keys, decodes all values and scans all probabilities. All original learned maps remain paid: 197 learned linear calls and 1,543,569,408 MACs per complete token, plus measured partial attempts. Weights remain about 6.175 GB. There is no process-memory, throughput or forward-pass-free claim.

Exact native callback parity, sibling replay, query-parent rollback and partial-token abort passed. The first failed smoke and the later observer negative-zero diagnostic remain preserved in [the repair history](INSTRUMENT_REPAIRS.md). Source/raw/log/receipt joins and discrete saved-analysis replay agree. Tiny Mac/Linux FP64 differences and the original overly strict scalar-policy refusal remain explicit diagnostics in verification; derived arithmetic is not claimed bit-identical across platforms.

The [independent raw audit](runs/job-ceab8d5c9869/INDEPENDENT_AUDIT.md) reproduces all 80 packet readers and the work ledger, including the unchanged case0 trajectories across the observer repair. Standard Saturn [evidence run](runs/job-ceab8d5c9869/evidence-run.json) and [verify](runs/job-ceab8d5c9869/evidence-verify.json) both return 0, with target-streams-only scope and 688 exported records containing repeated observation views. This is evidence-plane mechanics coverage, not semantic or serving certification.

The constructive feature-state closure theorem and positive-kernel error bound are proved in [MATH.md](MATH.md). This experiment does not discover the feature maps or prove bounded rank. [Astra's review](REVIEW.md) proposes a future implicit/explicit assignment by immediate/delayed query square to test when the behavioral weakness appears and how explicit assignment changes it. Internal writer/reader attribution would need further interventions. That follow-up was not run here.
