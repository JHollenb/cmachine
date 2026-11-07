# Future-focused selection on longer source conversations

Completed32-token result: [findings](FINDINGS.md), [Astra review](REVIEW.md), [executed protocol](EXECUTED_PROTOCOL.md), [verification](runs/job-309bf320370d/verification.json) and [saved-only study](runs/job-309bf320370d/study.json). Future KL selects the unchanged dose2 image,1.76% better than dose1; all six images share eventual first greedy disagreements. New targets give KL8/12 and L211/12 improvements versus dose1 with identical compact answers. Native241 versus compact401 on418−17 separates fidelity from task correctness; explanations are truncated. Same serving budget, no installed policy change.

**Source-version boundary:** use `runs/job-309bf320370d/payload/` for this result. Working PLAN.md, driver.c, worker.py, fresh.json and panel.py contain subsequent development beyond the executed32-token forced-source protocol. The invocation recipe below is not a claim that those newer files ran in this job. [Next smallest discriminator](NEXT.md): own-history candidate generation with an independent native shadow, keeping memory fixed.

Extend the same original source prompts to up to32 native answer tokens, one shared followup and up to32 further tokens. Replay the same six immutable query images on the native source IDs, recording each first greedy disagreement and candidate-owned caches/memory. Select on equal-source/equal-turn KL at the prediction clocks choosing answer tokens, including EOS.

[Protocol](PLAN.md), [clock math](MATH.md), [C source/target executor](driver.c), [full-clock metric audit](metrics.py), [source/memory/target verifier](analysis.py).

The two target controls are the historical all-clock native-KL image (dose2/index5) and the response-MSE image (dose1/index4). In the inherited driver names, `values-native-memory` now means the authenticated dose2 control, `response-reference-memory` means dose1 and `query-fit-memory` means the new future-selected image. Program128-q is explicitly bound to image5; source proposal0 remains the unchanged LS image. Exact selected-image aliases are declared and deduplicated. Three new target conversations suffice for this temporal cut; old development targets are not repeated.

Source evidence keeps Q/K/V/attention/pending IDs, native receiver arrays and candidate receiver hashes. It drops redundant source output/coordinate arrays. Full-vocabulary C/Python streaming agreement covers every source clock; durable full-logit replay is limited to first/last decision in each answer. Original193 source-prefix rows and their archived operand/logit hashes are checked before interpreting the longer future.

Run using `common exec python saturn/cmachine/query_horizon/run.py stage`, then `submit PAYLOAD`, `collect JOB`, `verify RUN_DIRECTORY`. Real models run through mrun. Tiny mechanics: `common exec python -m unittest discover -s saturn/cmachine/query_horizon -p 'test_*.py'` plus the C distribution fixture.

Serving remains819,200 selected QKV MACs and98,464-byte selected logical memory per update. Source feedback is newly paid; the dense receiver, other layers, original weights/caches and vocabulary head remain. No serving default changes.
