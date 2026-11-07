# Verify a committed integration

The verification target is engineering preservation: a named Git commit builds the unified runtime, preserves native computation and reproduces the sealed compressed-memory behavior. Historical bounded delayed-recall failures remain expected observations. This protocol does not certify useful general chat or a speedup.

Run the model-free lane first: `python -m unittest discover -s tests -v`, `make sanitize`, `make state-mechanics` and `make bounded-mechanics`. The last two use ASan/UBSan. The integrated bounded fixture adds a wrong-parent-lineage refusal to the occupied-sample overwrite/abort checks. Local evidence records exact source hashes rather than making a circular claim about its own containing commit.

Commit the source, then use the domains shared environment:

```sh
common exec python validation/run.py stage COMMIT
common exec python validation/run.py submit PAYLOAD
common exec python validation/run.py collect JOB
common exec python validation/run.py verify validation/runs/JOB
```

Staging reads every source from its exact Git object. It never copies an uncommitted working source. A separately built native reference uses original commit `b81e928`. Its original model/CUDA core is observed through the same nonmutating server view, compiled against its original header ABI. This avoids conflating native untiled arithmetic with the historical tiled-native-cache arm.

One guarded CUDA mrun lease executes twelve sequential model processes: original and integrated native protocols; full-byte and bounded source-BF16 profiles; bounded tiled-FP32 control; native writer/lineage lifecycle; real bounded occupied-sample abort/replay; four actual compressed-policy conversations; and an actual default native client. Geometry is B1, CPU1, logical768, with fixed explicit generation budgets. No models run simultaneously, no retries are requested, and expected duration is under five minutes. The reservation follows the measured preceding native-inclusive trial.

The verifier authenticates commit/tree and every source blob, declared/sealed/executed scheduler payload, scheduler log, model/tokenizer binding, raw extents/hashes, typed whole-invocation Saturn receipt and saved-only analysis. Native paired histories and readouts must agree exactly. Each compressed profile must reproduce eight historical turns, sixteen full readout hashes, delayed-root state, repeated-generation ownership and nonmutating refusals. The actual clients must reproduce the prior text with zero transcript recomputations. The receipt records that negative behavioral evidence explicitly.

Only after all these checks does the worker publish the versioned installation under `/mnt/big/cmachine/builds/`. Canonical `/mnt/big/cmachine/current` and the two old `saturn-cmachine` current aliases point to the unified build. Prior version directories remain available for historical receipts and rollback. Publication restores changed aliases if it cannot finish. No background service starts.

Full local run custody is ignored under `validation/runs`. A compact tracked proof bundle is added after the tested source commit, with the model source commit and runtime-tree identity explicit. Validation's public Saturn receipt implementation is a frozen import from the preceding sealed payload; source hashes bind it to the tested commit. The receipt covers process invocations, while the protocol and C lifecycle checks audit internal ownership.
