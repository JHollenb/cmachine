# Committed integration: execution proof

**Tested source commit:** `7a6824dd9b640cfa785fa56cc4e7cfae14193f1c`

**Source tree:** `84a224b896b33d4240cc8581b14f5463831f5f9f`

**Native reference:** `b81e9286a3e8bd9e25172fdd3969bdad4ca5430a`

**Guarded Beast job:** `job-f645fcd76a75`, succeeded, November 6, 2026 Pacific time.

This bundle proves the bounded engineering claim: the named integration commit builds the unified runtime and preserves tested native and experimental behavior. It contains the exact shipped sources and raw results; weights and executables are identified by digest rather than distributed. The proof was collected after the source commit and committed separately to avoid a circular commit identity.

From the repository root, reverify with Python 3 and Git, without a model, GPU or service connection:

```sh
python3 validation/run.py verify docs/evidence/integration-7a6824d
```

The verifier checks the named Git objects, scheduler success/log seal, source manifests, declared/sealed/executed payload identity, raw hashes/extents, report joins and Saturn whole-invocation receipt, then recomputes the frozen analysis. It evaluates zero models. Use a checkout whose Git history includes both named commits. The imported Saturn receipt code records the process panel; the C tests and protocol traces separately exercise its internal lifecycle.

## What ran and passed

- Model-free evidence: 18 Python tests, ISA sanitizer build, native and bounded ASan/UBSan fixtures, and strict serving/header-ABI syntax checks. [Local receipt](payload/local-checks.json).
- Original and integrated native cores: all **84 wire-protocol entries** identical, including ten own-token turns and twenty prefill/final full-logit hashes. The original core uses its original header ABI with the same reporting-only server view.
- Full-byte/source-BF16, bounded/source-BF16 and bounded/tiled-FP32: eight historical turns and sixteen historical full-logit hashes each, delayed-root cache bytes, eleven nonmutating refusals per profile, exact sibling replay, split generation and repeated-EOS ownership. The tiled-FP32 arm invokes the legacy bounded executable name without a policy flag.
- Actual pretrained lifecycle: original writer/gate/lineage checks, occupied-reservoir last-layer abort, exact arena/address/RNG/logit restoration, sibling replay and post-cache refusal followed by an exact same-layer retry.
- Five actual clients: unchanged historical prose/arithmetic/recall outputs and **zero prefix recomputations**. The bounded delayed-recall failures are explicitly preserved.
- Installed files: 124 manifest entries bound to the source commit and binary. Canonical and legacy aliases published only after checks; no background service started.

The twelve sequential model processes took 181.44 scheduler seconds (180.49 worker seconds). Aggregate measured peaks were 4,206.6 MB RSS and 6,396 MB VRAM, including the larger native/control profiles. Bounded source-BF16 tracked allocation is 9,279,620 state bytes and 3,133,714,584 model-plus-state bytes. The experiment uses logical context 768 and bounded support 112; it does not qualify every context length or model.

## Receipt map

| File | Role |
|---|---|
| [verification.json](verification.json) | Offline verification result and payload triplet |
| [report.json](report.json) | Full results, model/tokenizer identities and typed invocation receipt |
| [payload/source-manifest.json](payload/source-manifest.json) | Exact Git commit/path/hash for all 120 shipped sources |
| [raw/installed-files.json](raw/installed-files.json) | Installed source/client/binary digests |
| [raw/installation.json](raw/installation.json) | Version path, publication and prior aliases |
| [job.json](job.json), [guarded-receipt.json](guarded-receipt.json) | Redacted scheduler success and custody records |
| [logs.txt](logs.txt) | Unchanged scheduler-sealed output, including exported evidence |
| [packaging.json](packaging.json) | Published-file hashes and exact receipt redactions |
| [evidence-plane/evidence-verify.json](evidence-plane/evidence-verify.json) | Offline stream sweep; run/verify return 0, `target-streams-only` |

Installed version: `/mnt/big/cmachine/builds/7a6824dd9b640cfa-8e608af7734c`. Binary SHA256: `8e608af7734ce351a3572f705fbcb21a84d37ba6fe8d2d4cb6532c5ae7860db0`. Report SHA256: `82ec2e64dbcd41b40279dd8f1fe73765752b356028eecdcd5e4870bad91a27f9`.

Published scheduler records remove only `admission.claim_key` and `admission.idempotency_key`, which are unused by verification. Original and published hashes and removed paths are recorded in `packaging.json`; complete originals remain in local ignored custody. Raw results, log, report and payload sources are unchanged. This is a receipt-backed provenance record within the trusted scheduler/worker boundary, not an independent hardware attestation.

Raw transcripts retain their original trailing prompt spaces and input whitespace because their byte hashes are evidence. They are intentionally exempt from formatting cleanup.

General chat accuracy, sufficient bounded memory, durable session persistence, speed improvements and elimination of learned forward computation remain open. See the [independent Astra audit](../../INTEGRATION-REVIEW.md) and [runtime contracts](../../INTEGRATED-RUNTIME.md).
