# Closed-state demo

This is a small, weight-free executable demonstration of a state-interface
condition. It uses the canonical `cmachine/src/gpu.h` operations through
`cmachine/tests/gpu_stub.c`, so it runs on a fresh clone with a C11 compiler and
Python 3. The stub is an explicit CPU mechanics profile; no CUDA kernel,
pretrained checkpoint, tokenizer, or model outcome is involved.

From the `cmachine` repository root:

```sh
python3 demos/closed_state/run.py --output demos/closed_state/results.json
```

The driver compiles the C organism with `-ffp-contract=off`, runs eight paths
for 16 own-state updates, checks the exact C trajectories, and writes the raw
states, normalization rows, gate/up vectors, products, writes, source hashes,
and validation metrics to the requested JSON file. The temporary executable is
removed after the run. The checked-in `results.json` is one result from that
command and is reproducible from the source files named in its manifest.

The supplied recurrence is

```text
x+ = x + D (SiLU(G rho(x) x) * U rho(x) x)
rho(x) = (||x||^2 / 8 + 1e-6)^(-1/2)
```

`x` has eight FP32 coordinates and the intermediate width is four. `G` and
`U` read coordinates 0 and 1. The closed control writes only coordinates 0 and
1, so a two-coordinate state plus the omitted energy `kappa` has all operands
needed by its next update. The two full references start with the same visible
state and omitted values `+1` and `-1`; their retained trajectories,
normalization scales, and every gate/up/product and retained write match the
compact path byte-for-byte for all 16 updates. The full normalization vectors
remain diagnostic arrays; their omitted signed coordinate is intentionally
different in the two full references.

The escape control changes only `D` by adding a write to coordinate 2. The
first write has a missing signed coupling to the omitted orientation,

```text
kappa+ = kappa + 2 e^T V w + w^T V^T V w.
```

The stale compact state therefore first differs from both full references at
update 2. Retaining coordinate 2 repairs the interface and again matches both
full references exactly. In this fixture the one-step omitted energies are
`0.862843036` and `1.14726925`, and the two full declared outputs end `0.437323869`
apart in L2.

The nominal retained-state and affine counts are 32 bytes/96 MACs for the full
state, 12 bytes/24 MACs for the closed candidate, and 16 bytes/36 MACs for the
repair. Every path still pays the width-eight normalization, nonlinear product,
transfers, and the CPU stub's implementation costs. These are geometry counts,
not a measured speedup.

This demo establishes a finite closure, counterexample, and repair for a
declared supplied organism. It does not establish learned-model compression,
CUDA numerical parity, serving speed, free-form chat accuracy, or a general
state representation theorem.
