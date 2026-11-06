# Saturn and mrun integration

The core C executor, bundle exporter, text client and model-free regressions can be used from this checkout. They do not require the Saturn Python package for numerical inference.

The other imported Python files preserve research helpers from the original workspace. In particular, `tools/fleet.py`, `tools/worker.py` and `tools/saturn_bridge.py` belong to the Saturn/mrun research lane. Some modes expect the original domains layout, private ISA specimens, protocol files, the public Saturn adapter and existing model/result stores. Their presence here does not make that whole environment a standalone dependency-free experiment runner.

Run those existing campaigns from their original `saturn/cmachine` checkout. This extraction does not move that directory, modify its active experiments, change an installed serving binary or change fleet jobs. New runtime development can use this repository; integrating a changed runtime into an existing campaign requires a newly bound source manifest and its ordinary execution checks. Do not relabel an old receipt as covering new code.

Public interfaces used by the workspace are available in [saturn-pub](https://github.com/JHollenb/saturn-pub) and [mrun-pub](https://github.com/JHollenb/mrun-pub). Use the workspace's configured shared environment and mrun admission for real-model/multi-step scientific runs. CUDA work must request CUDA, and the lease must account for the actual backend and physical work.

The original raw reports, saved arrays, payloads and custody capsules remain authoritative for their historical runs. This repository contains source import hashes and a compact status account, not an implicit copy of those archives or a promise that private artifact paths are publicly fetchable.

The next integration step is a thin typed operation contract at existing C boundaries: immutable program, actual parent-bound invocation, declared reads/writes and continuation, optional evidence receipt and a physical-work ledger. This is proposed. It should build on the current executor rather than create a second state owner.
