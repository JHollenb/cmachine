# Import provenance

This repository was extracted on November 6, 2026 from the working `saturn/cmachine` runtime. At extraction, that directory had no independent `.git` repository and its runtime files were not tracked in Saturn's Git index. A separate checkout was created to preserve ongoing work in the original directory.

The initial import copies 43 files: the Makefile, nine `src/` files, fourteen test files and nineteen Python helpers. Their exact paths, sizes and SHA256 values are recorded in [import-manifest.json](import-manifest.json). The source was hashed before and after copying; both source versions and the copied files agreed. Original runtime, helper and test bytes are preserved.

The surrounding Saturn HEAD is recorded only as workspace context. It does not claim to contain the uncommitted imported files. This repository begins an independent history; it does not manufacture earlier commits or rewrite the original repository's history.

The new README, architecture, vision, status, integration documentation, ignore rules and model-free CI are packaging additions. Weights, compiled binaries, environment credentials, raw run archives, custody stores and the other agent's active experiment are outside the initial publication. Existing raw report identities cited in the status document remain unchanged.

The original imported hashes remain preserved as lineage, not rewritten to match newer code.

## Canonical integration

The subsequent user-authorized integration moved the entire former working tree to `cmachine/research` using an atomic same-filesystem rename, preserving its directory inode and leaving `saturn/cmachine` as a compatibility link. The inventory contains 63,626 files/links and 9,807,398,925 file bytes. The full compressed byte manifest stays local under `research/migration`; the [small migration receipt](evidence/research-migration.json) and [immutable-evidence verification](evidence/migration-verification.json) are committed.

The former live baseline `src`, `tests`, `tools` and Makefile are preserved under local `research/migration/baseline`. Compatibility aliases now resolve those paths to the canonical runtime. Historical frozen payloads and raw reports remain byte-preserved. The integrated core combines cache callbacks and abort recovery with the original writer, gate-law and parent-lineage APIs. Weight tiling and cache kernels derive from the sealed bounded-chat payload `job-f53e7d9c952f`; its numerical comparisons are retained as reference fixtures rather than relabeled as evidence for the new commit.

New pretrained evidence must identify the actual integration commit. The [validation launcher](../validation/run.py) stages exact `git show COMMIT:path` bytes, including an independently built original native reference from commit `b81e928`. The offline verifier checks every shipped source against its Git object, the guarded declared/sealed/executed payload, scheduler log, raw artifacts, typed Saturn receipt and independently recomputed analysis.

Integration commit `7a6824dd9b640cfa785fa56cc4e7cfae14193f1c` has now passed that fresh protocol in guarded Beast job `job-f645fcd76a75`. The [tracked proof bundle](evidence/integration-7a6824d/README.md) includes 120 exact shipped sources and 36 raw artifacts. Its follow-up receipt commit changes documentation/evidence rather than the tested canonical runtime. Scheduler records are explicitly redacted only at two unused admission identifiers; original hashes and exact removed paths are recorded alongside the unchanged raw/log/source bytes.
