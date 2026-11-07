# Measuring math capability, scorer efficiency and unions

The frozen V1 scorer can be physically reduced from 32 to 14 ReLU units while matching full V1's extracted-program cost on all 30 tested cases. The reduced scorer has 323 rather than 737 learned coefficients and 308 rather than 704 linear MACs per candidate. Adding one supplied broader-factor proposal improves both tested trinomial layouts from cost 60 to 8, with 28 ties and no observed cost losses. These two layouts share one mathematical construction. This is an exploratory optimization capability result under fixed search budgets.

Combining specialist neuron sets behaves differently: their 14-unit union worsens both binomial4 layouts from cost 7 to 18, with 22 other original cases tied. More retained coordinates need not improve a nonlinear downstream search. [Compact receipts](evidence/math-capability-20261107/README.md) preserve both outcomes and the incomplete earlier attempts.

## Methods and scope

The paper search connected earlier workspace work on [blind reasoning discovery](../../../obsidian/blog/2026-07-30-123452-the-decompiler-found-the-reasoning-spine.md), [capability vectors](../../../obsidian/whitepapers/capability-vectors-not-scalar-intelligence.md) and [contextual unions](../../../saturn/experiments/2026-11-06-contextual-union/FINDINGS.md) to these primary sources:

| Paper | Method used here |
| --- | --- |
| [Circuit Compositions, ACL 2025](https://aclanthology.org/2025.acl-long.727/) | Test unions of task-specific subsets in their actual consumers. Our zero-deleted frozen neuron subsets differ from its learned activation masks and mean replacement. |
| [Towards Automated Circuit Discovery](https://arxiv.org/abs/2304.14997) | Use explicit interventions and a declared behavioral observable. We do not reproduce ACDC's discovery algorithm. |
| [Have Faith in Faithfulness](https://arxiv.org/abs/2403.17806) | Assess consumer behavior separately from component overlap and score fidelity. |
| [Causal Abstractions of Neural Networks](https://papers.nips.cc/paper/2021/hash/4f5c422f4d49a5a807eda27434231040-Abstract.html) | Guide the separation of supplied mathematical variables, physical state and downstream effects. No complete causal abstraction is established. |
| [Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations](https://proceedings.mlr.press/v236/geiger24a.html) | A future method for distributed role alignment; DAS was not run in this panel. |
| [Wanda](https://arxiv.org/abs/2306.11695) | Motivate explicit pruning controls and parameter accounting. Our weight-energy control is not Wanda's magnitude-times-activation criterion. |

The actual discovery algorithm is **frozen-coordinate score-fidelity selection followed by consumer testing**. On 12 discovery goals, the full V1 C policy generates 50 frontiers and 2,347 candidate feature rows. The exported C hidden-state program supplies post-ReLU activations. A greedy FP64 diagnostic selects original coordinates by minimizing within-frontier centered score error, copying their original coefficients without fitting. Each frontier has equal total weight; the global sample has 30 binomial and 20 planted frontiers. Thirty-two separate C zero-output-weight ablations measure discovery score sensitivity, not downstream necessity.

Both checkpoints score the V1 base-library frontier, using its named features and explicit zero fill for absent V2a columns. The complete V2a acquired divide/expand library is outside this comparison.

The canonical pinned `manalysis.bench.conditional_family.build_family_coalition_catalog` builds the bounded pairwise neuron unions. Binomial and planted top-eight subsets overlap in two units, giving a 14-unit union. V1 and V2a neuron numbers are never treated as aligned. Global selection, matched width, energy and two random masks remain separate controls. The discovery seal fixes variants and their case assignments before test execution. The follow-up explicitly records that its additional policies and leads were chosen after inspecting the first panel.

The original 24 cases include binomial layouts, fresh planted seeds, trinomials, scalar held shapes and ordered matrices. Six follow-up cases add a trinomial layout, two planted seeds, a mixed-arity product, nested factoring and matrix subtraction. These are known constructions; no exclusion from original checkpoint training data is claimed. Four full-V1 cases perform no rewrite selections. Thirteen of its 30 cases encounter more candidates than available batch slots; the remaining cases still test execution and possible order effects.

## Two distinct unions

**Neuron union:** preserve the set union of selected V1 post-ReLU units, physically shorten both linear layers, and retain the original output bias and normalization. This can change candidate rankings.

**Proposal union:** retain neural ordering but replace up to one, two or four final batch slots with unselected broader-factor candidates from the current recipient frontier. The supplied rule uses `rule_factor`, `log_group` and `target_on_route`, then neural ordering to break ties. It never reads a donor continuation, task answer or future outcome. The scorer-free control uses the same structural keys with ordinal ties and ordinal fill, consuming no neural scores. An alternative-membership control inserts non-broader candidates under the same one-slot budget.

All policies retain the original host matcher, e-graph mutation and extractor: six rounds, eight choices per round, 48 total selections, 768 nodes and 1,024 matches per round. A public Saturn adapter captures the first reached choice and reconstructs its sealed prefix before unchanged native continuation. Search and extraction clocks were conservatively proven unhit in all recorded attempts. This panel uses host `DictIndex` to isolate scorer/search behavior; the separate [FieldLM C projected-read integration](MATH.md) remains separately qualified.

## Paired results

| Candidate versus reference | Common cases | Better / same / worse final DAG cost |
| --- | ---: | ---: |
| Global14 versus full V1 | 30 | 0 / 30 / 0 |
| Global14 plus proposal1 versus global14 | 30 | 2 / 28 / 0 |
| Full V1 plus proposal1 versus full V1 | 30 | 2 / 28 / 0 |
| Specialist union14 versus global14 | 24 | 0 / 22 / 2 |
| Planted8 versus full V1 | 25 | 2 / 23 / 0 |
| Random14 seed29 versus full V1 | 25 | 2 / 23 / 0 |
| V2a plus proposal1 versus full V2a | 30 | 6 / 24 / 0 |
| V2a plus alternative1 versus full V2a | 24 | 3 / 20 / 1 |
| Scorer-free broad rule versus full V2a | 30 | 4 / 25 / 1 |

The planted8 and random14 gains are solely the two trinomial3 layouts, both cost `60→8`. Their second layout is explicitly a post-selection follow-up. The global14 plus proposal rule also reaches 8 on both; full V1 plus the rule reaches 8 and 10; V2a plus the rule reaches 8 and 20; broad-only reaches 22 and 22. The five other follow-up cases tie across the seven declared policies. Meeting a supplied constructive/historical reference is not proof of global optimality. Mean costs across scalar and domain-weighted matrix cases are retained in raw summaries but are not a general reasoning score.

All 602 declared goal-policy arms complete. Every endpoint has the same exact supplied polynomial as its input, and all 16 bounded integer input rows per emitted expression have zero C reference/output difference. Each emitted C program also has zero source/C difference on those rows. Formal polynomial equivalence and sampled FP32 execution equality are different evidence.

The maximum scorer error is `3.0517578125e-5`. Across the recorded decisions, 322 C/native ordering differences include 136 selected-set differences; these counts repeat across policies and layouts and are not independent error samples. The full V1 and planted8/random14 native-scored controls reproduce `60/8/8` on both trinomial layouts. That qualifies the specific compression lead, without claiming universal native/C search-policy equivalence. These are rewrite scores, not vocabulary logits; [PAN/Mamba logit errors](../../docs/SUPPORTED-MODELS.md#measured-logit-error) remain unchanged.

## Parameter and computation accounting

For feature width `d` and hidden width `h`, learned coefficients are `(d+2)h+1`; normalization stores `2d` additional floats; linear work is `(d+1)h` MACs per candidate. The original and reduced V1 programs both have eight vector instructions, whose vector widths differ.

| V1 scorer | Learned coefficients | Normalization floats | MACs per candidate | Executable bundle bytes |
| --- | ---: | ---: | ---: | ---: |
| Full32 | 737 | 42 | 704 | 5,172 |
| Global14 | 323 | 42 | 308 | 3,516 |
| Specialist union14 | 323 | 42 | 308 | 3,516 |
| Planted8 | 185 | 42 | 176 | 2,964 |

Global14 reduces learned scorer coefficients by 56.2%, per-candidate linear MACs by 56.25%, and bundle bytes by 32.0%. On the 24 original cases, its logical scoring work is 703,164 MACs versus 1,609,344 for full V1. The reduced scorer plus proposal1 uses 711,172, reflecting changed frontiers, and reduces emitted-program instructions from 284 to 232 across those cases. Both trinomial programs individually shrink `69→17` instructions. These counts include input slices and explicit operations, not just algebraic operators.

The supplied proposal rule, host search, normalization, discovery and replay still cost work. Native reference banks remain resident in the audit process. This is physical scorer compression, not a total process memory or latency claim. The final guarded one-thread Beast CPU worker took 16.01 seconds with 659.3 MB scheduler peak RSS, including discovery, audits and all arms.

## Reproduce and verify

```sh
rtk proxy ../common/bin/common exec python tools/math_fleet.py submit --mode math-capability
rtk proxy ../common/bin/common exec python tools/math_fleet.py collect JOB_ID
rtk proxy ../common/bin/common exec python tools/verify_math_capability.py results/JOB_ID --output results/JOB_ID/verification.json
rtk proxy ../common/bin/common exec python research/model_intake/plot_capability.py results/JOB_ID results/JOB_ID/figures
```

The offline verifier checks payload/raw hashes, actual binary parameter subsets, discovery-only selection and all rejected losses, fixed union membership, scorer/feature/C-call bindings, proposal cardinality, Saturn cut/receipt closure, exact polynomials, emitted-program input identities, costs and signed paired results. It runs no learned forwards or search. Integrity failures may stop verification; unfavorable quality results remain observations.

Sixty unique Python mechanics tests pass locally, including seven new generated capability fixtures and a constant-left FX `matmul` regression. Eight math fixtures also pass through the ASan/UBSan C executable. The compiler fix lowers a method-form static-right matmul to the existing generic linear primitive; it adds no model-specific C operation. Historical attempt failures and their executed sources remain immutable in the evidence history.
