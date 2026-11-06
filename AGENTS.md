# Working on cmachine

Read `README.md`, `docs/ARCHITECTURE.md`, `docs/VISION.md` and the relevant source before changing execution. Keep ordinary C/CUDA operation order, numerical profiles, model/input identity and pending-token ownership explicit.

Use the model-free build and regression lane for packaging and lifecycle work. Real-model scientific runs in the domains workspace use its configured mrun and shared environment; do not bypass its admission or duplicate an active accelerator run.

Preserve unrelated work and immutable historical evidence. Separate execution correctness, instrument validity, scientific trends and terminal certification. A missed exploratory metric is not proof that the mechanism is absent.

State compression, arithmetic replacement, framework independence, own-token continuation, task correctness and measured speed are distinct claims. Count original and derived operations, compilation, operand resolution, memory, transfers and replay. Experimental policies remain explicit and detailed tracing stays optional.

The imported workspace-specific helpers retain dependencies described in `docs/INTEGRATION.md`. Do not assume this checkout contains the original private models, campaign protocols or raw result stores.
