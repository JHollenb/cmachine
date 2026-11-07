# Design review: transition-laws panel

## Scope

This is a prospective all-layer rank-256 gate/up factor experiment on the Qwen2.5-1.5B
profile. It is a new operator experiment, not a new all-layer execution claim. The earlier
[`composition/compact_all`](../composition/compact_all/FINDINGS.md) run already demonstrated
all-layer callback, cache, rollback, and own-token mechanics. The earlier
[`contextual-union`](../../experiments/2026-11-06-contextual-union/FINDINGS.md) run demonstrated
receiver-aware factorization at selected layers, but also measured own-state drift and separated
selected-layer MAC savings from whole-model cost. This panel should inherit those distinctions.

## Required interpretation

The serving graph replaces only the two gate/up projections. RMSNorm, SiLU, the Hadamard product,
the native down projection, residual additions, attention, all 28 K/V writes, and the vocabulary
readout remain native. Therefore the result can test a cheaper gate/up *operator* while preserving
the dense native consumer. It cannot be called a source-free, forward-pass-free, cache-compressed,
or autonomous chat emulator without a later experiment that removes those paid computations.

The proposed E0/E1 split is scientifically useful:

- E0 is an uncentered rank-256 randomized-SVD basis of the equal-weight native source sample,
  using the fixed (q=272), two-iteration, per-layer seed and sign convention. It is not an exact
  full-SVD optimum.
- E1 has the same rank, row budget, and serving graph, but fits native plus E0-own-source
  receivers at equal weight.
- Six source conversations fit the programs; three separate monitor conversations can choose a
  continuation policy; six fresh conversations are held out.

The fit boundary must be sealed before held-out tokenization/inspection. The manifest should bind
the 384 native and 384 E0-own rows, per-case membership, uncentered/centered convention, signs and
ordering of singular vectors, FP32 factor bytes, and the exact monitor-selection rule. If E1 uses
candidate-generated own rows, that is a declared source-stage refresh; it is not an independent
native calibration.

## Native shadow and token authority

The native shadow must consume the candidate's actual generated IDs with a separate full native
cache. Its KL is a conditional consumer discrepancy. It is not evidence that the native model
would have generated those IDs, and it must not supply candidate K/V or select candidate tokens.
Keep three paths distinct in every report:

1. native-owned IDs and native state;
2. candidate-owned IDs and factor state;
3. native shadow on candidate-owned IDs.

The same-ID shadow isolates operator/consumer error under a fixed discrete history. Divergence
between candidate and native-owned histories is a separate closed-loop effect. Compare both at
the same decision clock and preserve the full distributions, selected IDs, EOS/cap status,
prefix/KV hashes, and pending-readout policy.

## Independent mechanics checks

Before interpreting KL or token behavior, independently verify model/tokenizer/runtime identity,
28 layers, factor shapes \([256,1536]\) and \([8960,256]\), FP32 layout, source/own/held-out
quarantine, finite values, exact row counts, and factor sealing. Recompute the factor MAC identity

\[
 C_f=1536\cdot256+2(8960\cdot256)+1536\cdot8960=18{,}743{,}296
\]

against the declared native gate/up/down term

\[
 C_n=3(1536\cdot8960)=41{,}287{,}680.
\]

The 28-layer difference is 631,242,752 MACs, which is 40.895% of the declared
1,543,569,408 native per-token ledger only if that ledger includes the same MLP terms. Keep a
category-separated ledger and do not turn MAC arithmetic into latency or whole-model speedup.

At each layer retain normalized input, projected coordinates, native and factor gate/up values,
MLP write, post-residual state, K/V hash, downstream readout, and token decision. These are
diagnostics and trend evidence. Only identity, shape, finiteness, seal ordering, and replay
mechanics should stop collection during exploration.

## Main scientific risks

Rank-256 source reconstruction is not transition closure. The candidate can leave the source
span, and the nonlinear SiLU-times-up operation can amplify a small receiver projection error.
E1 can improve own-source fit while worsening a fresh context. Preserve signed per-layer and
per-clock values instead of reducing to a single mean KL. Report candidate local error, native
shadow conditional error, native-owned error, K/V drift, and own-token trajectory separately.

The phrase “all 28 MLPs” must not be mistaken for one shared cross-layer map: the natural object
is one \((E_\ell,L^g_\ell,L^u_\ell)\) triple per layer. If the implementation uses a single
cross-layer E, record that stronger hypothesis explicitly. A native D and original model weights
still resident also make this a hybrid exact-consumer test. The older all-layer cache result
cannot be used as evidence that this factor law exists; it is only the lifecycle precedent.

The smallest useful analysis table is paired E0/E1, source/own/held-out, and native-owned versus
shadow versus candidate-owned. Include local \(z,g,u,w\), K/V, logits/KL, token IDs, and cost.
This separates an approximation error at the current receiver from an error caused by advancing a
different state into the next token.
