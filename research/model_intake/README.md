# Automated model intake and computation discovery

This directory contains an offline intake prototype and the [architecture plan](PLAN.md) for connecting the existing model compiler, C runtime and computation-discovery tools.

The prototype imports local declarative checkpoint sources through `mrun.decompiler`, validates its coordinated IR, and uses `manalysis.decompiler_bridge` to produce an exact graph with stable source-bound references. Mathematical role priors remain a separate hypothesis-only artifact. It performs no model forward, C lowering, token generation or semantic certification. Existing C serving is unchanged.

## Use the prototype

From the domains workspace:

```sh
rtk proxy common/bin/common exec python cmachine/research/model_intake/intake.py \
  --source /path/to/local/checkpoint \
  --output /path/to/new/intake-directory
```

To inspect a previously emitted canonical IR bundle or full decompiler report:

```sh
rtk proxy common/bin/common exec python cmachine/research/model_intake/intake.py \
  --ir /path/to/ir.json \
  --output /path/to/new/intake-directory
```

`--source-id` and `--revision` optionally bind the supplied source identity. Only already registered mrun architecture adapters are used. This command does not authorize custom model code or load adapter plugins. Sources outside the supported declarative grammar retain the canonical refusal instead of receiving a guessed executable graph.

Each new output directory contains `input.json`, `intake.json`, a readable `REPORT.md`, and a content-addressed workspace with the canonical bundle, exact graph and role priors when decoding succeeds. `intake.json` is published atomically last as the complete-intake marker. Existing output directories are refused. Output cannot be placed inside a frozen source directory.

The `--ir` path retains normalized coordinated IR only, so its source-validation scope is deliberately weaker than a fresh `--source` import, even when the incoming JSON was a full source report. The report never infers a qualified C backend from an architecture name or operation inventory. Its `c_execution_qualified` and `semantic_claim` fields remain false.

## Mechanics checks

```sh
rtk proxy common/bin/common exec python -m unittest discover \
  -s cmachine/research/model_intake -p 'test_intake.py' -v
```

The nine tests use tiny synthetic Qwen2/Qwen3 source fixtures with inert parameter bytes. They check exact graph references and hypothesis separation, raw-IR provenance, changed-parameter identity, unknown-family refusal, tampered IR, output/source protection, retained mutation refusals, late-mutation rejection and atomic completion publication. They do not exercise trained models or numerical serving.

All nine passed in the configured shared environment on November 6, 2026. A separate CLI smoke returned zero for a supported synthetic source and two for an unknown family, preserving its refusal report. The [mechanics record](evidence/mechanics.json) binds the test result to the prototype/test source hashes. It is a local test record, not a real-model execution receipt.

## The next integration

The next executable milestone is a cmachine target lowerer for the existing qualified Qwen2 profile, using the canonical source/weight/state/I/O description instead of a separately authored checkpoint mapping. A small pure-operation C path for the bilinear cell and scorer then tests graph-driven execution across different mathematical structures. Broader stateful models and automatic causal discovery follow with their own numerical and continuation evidence. See [the plan](PLAN.md) for the contracts, existing tools and remaining work.
