# Saturn and mrun integration

The core C executor, bundle exporter, text client and model-free regressions can be used from this checkout. They do not require the Saturn Python package for numerical inference.

The other imported Python files preserve research helpers from the original workspace. In particular, `tools/fleet.py`, `tools/worker.py` and `tools/saturn_bridge.py` belong to the Saturn/mrun research lane. Some modes expect the original domains layout, private ISA specimens, protocol files, the public Saturn adapter and existing model/result stores. Their presence here does not make that whole environment a standalone dependency-free experiment runner.

The entire working research tree now lives at `cmachine/research`. Its old `saturn/cmachine` address is a compatibility symlink, and its baseline runtime paths alias the canonical runtime here. Run existing campaigns from their relocated research directories. Their directory depth within the domains workspace is preserved. Historical payloads remain immutable; each changed runtime execution requires a newly bound source manifest and fresh checks. Do not relabel an old receipt as covering new code.

Public interfaces used by the workspace are available in [saturn-pub](https://github.com/JHollenb/saturn-pub) and [mrun-pub](https://github.com/JHollenb/mrun-pub). Use the workspace's configured shared environment and mrun admission for real-model/multi-step scientific runs. CUDA work must request CUDA, and the lease must account for the actual backend and physical work.

The original raw reports, saved arrays, payloads and custody capsules remain authoritative for their historical runs. They are physically under this local checkout, ignored by Git. Tracked research source and compact commit receipts do not make the private raw/model stores publicly fetchable.

The runtime now has explicit cache-step/guard/rewind hooks at its existing state owner. Complete bounded snapshots also preserve parent lineage. A broader immutable-program/typed-invocation/physical-work contract remains a research direction. [Current integration and contracts](INTEGRATED-RUNTIME.md).
