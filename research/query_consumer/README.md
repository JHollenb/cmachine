# Select a frozen query image through the native consumer

The six source-response query proposals from [query_response](../query_response/FINDINGS.md) are immutable inputs. This experiment chooses their dose using full native C next-token distributions on the original forced source histories, then tests the chosen image on new conversations. It adds no direction, parameter, query coordinate or memory node.

[Protocol](PLAN.md), [state/consumer math](MATH.md), [C selector/runtime](driver.c), [independent metric implementation](metrics.py), [saved-state verifier](analysis.py).

The C process loads the model once. Native source plus six candidate histories cost 1,351 new source advances. Every candidate has its own upper caches and compact memory under the same fixed source token IDs. C/Python independently calculate all full-vocabulary source metrics while temporary streams exist; nine source full-logit checkpoints per policy remain for durable replay. Selection seals before target parsing/tokenization. The final target manifest is bound after that boundary.

Target policy labels retain the older wrapper names: `values-native-memory` is the frozen LS query, `response-reference-memory` is the preceding source-response-selected image (index4), and `query-fit-memory` is the image selected by native-source KL. If selection returns index0 or4, that logical policy aliases the identical physical path. Native target references run after candidate array seals.

Serving stays at 819,200 selected QKV MACs per advance and 98,464 bytes of selected logical memory. Added source forwards, offline feedback, dense receiver production, remaining layers/weights/caches and full readout are paid separately. Fixed source IDs are not autonomous source generation. Native fidelity is distinct from answer correctness.

Use `common exec python saturn/cmachine/query_consumer/run.py stage`, then `submit PAYLOAD`, `collect JOB`, `verify RUN_DIRECTORY`. Real-model work runs through mrun. Tiny mechanics checks: `common exec python -m unittest discover -s saturn/cmachine/query_consumer -p 'test_*.py'` and the C distribution/memory sanitizer fixtures.
