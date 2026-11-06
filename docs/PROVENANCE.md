# Import provenance

This repository was extracted on November 6, 2026 from the working `saturn/cmachine` runtime. At extraction, that directory had no independent `.git` repository and its runtime files were not tracked in Saturn's Git index. A separate checkout was created to preserve ongoing work in the original directory.

The initial import copies 43 files: the Makefile, nine `src/` files, fourteen test files and nineteen Python helpers. Their exact paths, sizes and SHA256 values are recorded in [import-manifest.json](import-manifest.json). The source was hashed before and after copying; both source versions and the copied files agreed. Original runtime, helper and test bytes are preserved.

The surrounding Saturn HEAD is recorded only as workspace context. It does not claim to contain the uncommitted imported files. This repository begins an independent history; it does not manufacture earlier commits or rewrite the original repository's history.

The new README, architecture, vision, status, integration documentation, ignore rules and model-free CI are packaging additions. Weights, compiled binaries, environment credentials, raw run archives, custody stores and the other agent's active experiment are outside the initial publication. Existing raw report identities cited in the status document remain unchanged.

Future changes in the original Saturn working directory are not automatically synchronized into this checkout. A later import should identify the exact changed files, bind its source version and run the checks appropriate to those changes. Historical imported hashes should remain preserved as lineage, not rewritten to match newer code.
