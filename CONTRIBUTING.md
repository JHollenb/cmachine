# Contributing

Build with `make`, run `python3 -m unittest discover -s tests -v` with NumPy installed, and run `make state-mechanics` for model-free ASan/UBSan Qwen contracts. CUDA numerical checks and pretrained runs are separate lanes.

Keep ordinary execution order and token authority explicit. An experiment must identify its model, numerical profile, actual operands, parent state, writes and downstream consumer. Preserve full-state reference execution while investigating alternatives. Do not claim that wrapper-free execution removes the dense model computation.

For real-model work in the domains workspace, use its configured mrun/shared environment. Preserve immutable historical reports, raw results and rejected branches. Separate execution integrity, instrument status, scientific trends and terminal claims. Approximation errors and countertrends are evidence, rather than reasons to erase an exploratory arm.

Document actual work removed, compilation/resolution cost, memory, transfers and fallback/replay. Detailed tracing is optional and its overhead must be visible. Mark proposed APIs and unmeasured speedups as such.
