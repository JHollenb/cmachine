# Compile first-layer operands, then advance the candidate's own tokens

This small C example uses the **public `QwenLexicalProgram` API**. It harvests one short native trajectory, compiles biased first-layer Q/K/V atoms for the IDs consumed on that trajectory, saves/loads the frozen table, then creates a fresh candidate session. After the supplied prompt, the candidate consumes **its own pending token**. Native future IDs are never fed to it.

At every completed token, it compares the entire FP32 vocabulary vector directly and hashes every valid K/V row at all 28 layers. It also checks token identity and a missing-token refusal that leaves session metadata, valid cache and logits unchanged. The native reference and table seal close before candidate execution. There is no online table fill or fallback.

This is an exact source-covered reuse demonstration, not held-out generalization. The native discovery pass, original model, offline compilation, host table and per-step transfers are all paid. Only the first-layer RMS/Q/K/V computation is bypassed online. Remaining layers and the full vocabulary projection execute natively. No speedup is claimed.

**Fresh result:** guarded CUDA job `job-995c8e81e7ca` completed 44 exact token advances with 33 covered IDs, including eight candidate-owned generated tokens. Every full FP32 logit vector matched directly; valid all-layer cache hashes and selected tokens matched at every step. Compilation cost 103,809,024 linear MACs, the host table occupied 270,468 bytes, and candidate transfers totaled 360,448 bytes. The [public result bundle](../results/pretrained/README.md) contains the source seal, row-level report, frozen executed code and offline verification.

## Build and run

Requires the canonical CUDA build and a compatible exported Qwen2.5-1.5B-Instruct `.tcm` bundle. Run real-model execution through mrun in the domains workspace. Within its admitted worker, or an equivalently provisioned external machine:

```sh
python3 demos/compiled_operands/run.py --build-only
python3 demos/compiled_operands/run.py \
  --binary build/demo-compiled-operands \
  --model /path/to/qwen.tcm --tokens-file prompt.ids.txt \
  --output /path/to/new-result-directory --steps 8
```

The token file contains whitespace-separated integer IDs, including whatever chat-template prefix is desired. No tokenizer runs inside C. Prompt length plus generated-token consumption budget must be at most 64; each path is bounded by that limit. Generation stops after consuming EOS or reaching the budget. The two paths use one model and sequential sessions; at most 128 completed model token advances occur. The output directory must be new. Do not pass arbitrary strings through a shell tokenizer.

Build linkage is `build/model.o`, `build/qwen.o`, `build/qwen_cuda.o` plus this driver, linked with cuBLAS and libm. No independent model arithmetic is implemented here. `--binary` reuses a build; `invocation.json` hashes that binary and the driver, while the report binds the loaded model and frozen program. Broader source/scheduler custody belongs to the invoking mrun harness.

## Outputs and intended use

- `source-seal.json`: native consumed/pending IDs and every readout/cache hash, written before the candidate exists.
- `atoms.bin`: persisted, model-qualified table. It is derived model data; keep it out of Git.
- `report.json`: per-step exactness, refusal control and work/storage accounting.
- `invocation.json`, `stdout.txt`, `stderr.txt`: wrapper identity and process logs.

The complete native logit arrays are held only for direct comparison, then freed. Valid-cache comparison uses SHA-256 over layer/head/position-ordered FP32 bytes; unused cache capacity is excluded. `source_native_nominal_linear_macs` and `candidate_nominal_linear_macs` count learned affine work only: attention products, scalar operations, auditing, compilation and transfers are additional. Original matrices remain resident. Table hits are API counters, not independent GPU hardware counters.

For a weight-free introduction, run `make operand-mechanics` from the repository root. That canonical fixture tests the same public APIs, model/layer binding, missing-ID refusal, tamper detection and replay. Its identity-basis QKV example demonstrates dispatch parity, not compression.

The earlier real-model [context-query integration record](../../docs/evidence/context-query-20261106/README.md) reports 15 lexical IDs and 16 exact full-logit steps. That is historical evidence, **not** a result of this new demo. A fresh run's own report is the authority for this driver.
