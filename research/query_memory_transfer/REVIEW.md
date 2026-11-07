# Independent review: value restoration survives bounded memory

Reviewed 2026-11-06. Job `job-b94ac2f8a6c9`; report SHA256
`2ad6b50ea1a88fdd73c2d99978bf2ccce826f13c1eb042362e5cfbe2acca17a3`.
This review is additive. Frozen payloads, reports, and raw arrays were not edited.

Restoring native V improves the compiled program under both full and bounded
memory at all eight readout boundaries with shared consumed histories. The gain
survives recurrent compression, but its magnitude is not invariant: compression
strengthens it at four boundaries and weakens it at four. These are correlated
boundaries from three development cases, not eight independent generalization
tests. Task truth does not improve, and native QKV with compact memory remains
closer to the original model than the new compiled-QK/native-V compact program.

## Independent integrity and authority audit

- Verified 69 frozen source hashes, 281 raw hashes/extents, and exact membership
  and hashes of the 227-file candidate seal. Raw files total 100,826,415 bytes.
  Source-manifest digest, encoded scheduler report, raw report, and Saturn
  receipt fingerprint/parent/result bindings agree.
- All fifteen historical endpoint paths match the predecessor: 204 retained
  arrays, nine probability streams, exact consumed histories and cache-bound
  turn records, plus all four source128 bank arrays. There was no hidden change
  in the existing endpoints.
- All 72 full-vocabulary argmax readouts and 407 generated-token choices match
  each path's own authority stream. There are eighteen paths and 1,799 own
  advances. The three new CN+compact paths have 94, 95, and 112 advances.
- An independent state reconstruction, using `math.fsum` for centroid-distance
  decisions, reproduced all 36 compact snapshots byte for byte across 901
  compact steps. Every original position appeared exactly once in each group's
  membership partition. The maximum relative discrepancy between the lifted
  FP64 mixture of original V rows and the saved compact output was 3.70e-8,
  retained as a finite-profile remainder.
- Per-path and invocation work ledgers agree: 361,905 learned-linear calls and
  2,823,271,415,808 nominal MACs, including 32 native wrapper-parity advances.
  All twelve signed gain rows were independently recomputed from full logits.

These checks used the executed payload and raw data, not the mutable working
analyzer as the authority.

## Lifted memory law and finite arithmetic

The four-arm common prefixes have 94, 91, and 58 positions. Their receivers,
Q/K rows, native/compiled V pairs, full probabilities, key-state partitions,
and independently lifted compact weights agree across the required pairs.
The cumulative formula instrument is applicable throughout each prefix.

For CC=compiled Q/K/V and CN=compiled Q/K with native V, independent lifted
weights recover

    I = Y_CC,compact − Y_CC,full − Y_CN,compact + Y_CN,full
      = (Ptilde − P)(Vc − Vn),

up to the declared finite remainders. Here P is captured native full-attention
probability, and Ptilde distributes each prototype's response weight uniformly
over its original members. Memberships are reconstructed after sealing and do
not enter serving state.

Using independent `fsum` score/weight calculations and weighted mixtures, the
formula discrepancy relative to CN/full over each maximal prefix is
7.77e-8 / 7.59e-8 / 7.28e-8. Accounting for the four signed finite mixing
remainders leaves approximately 1.04–1.06e-16. The compact remainders include
centroid accumulation and ordered reduction, as well as FP32 output rounding.

Before any multi-member merge, interaction norms are approximately 6.90–7.06e-8
relative to CN/full, consistent with the numerical-profile difference. Over
the merged-memory intervals they are 9.87%, 9.21%, and 3.40%, well beyond the
observed pre-merge numerical-profile scale. Their norm supplies neither a signed
benefit nor a prescription to add covariance state. The full response and its
native consumer determine the direction of the effect.

## Measured gain and countertrends

All entries below are full-logit relative L2 differences against the same
NN/full reference. The first readout has identical consumed histories.

| Case | CC+compact | CN+compact | CN+full | Native QKV+compact |
|---|---:|---:|---:|---:|
| Recall | 14.471% | 12.083% | 11.761% | 2.035% |
| Arithmetic | 16.271% | 12.713% | 11.562% | 1.705% |
| Prose | 6.980% | 6.292% | 6.432% | 0.800% |

The recall follow-up prefill improves 17.900→13.373% and arithmetic improves
27.249→24.319%, on shared histories. The prose follow-up improves
35.707→30.267%, but its preceding generated history differs; that is a policy
outcome comparison rather than a local same-cache fidelity result.

The eight shared-history boundaries all retain positive native-V squared-error
gains under both readers. Their compact/full gain ratios range from 0.686 to
1.190. Arithmetic follow-up loses about 31% of the full-reader gain while still
improving over CC+compact. This rules out reporting that compression preserves
the repair magnitude unchanged.

There are meaningful countertrends. Prose's first-turn final discrepancy is
9.691% for CN+compact versus 8.861% for CC+compact after histories diverge. At
the initial prose readout, compression slightly improves CN's logits
6.432→6.292% despite worsening its local attention reconstruction. Neither a
single attention norm nor the interaction norm orders downstream usefulness.

All policies give blue then green, answer 14×19 as 266, and start the follow-up
with the incorrect 286 instead of 279. Arithmetic explanations remain truncated.
The new prose continuation gives a plausible meeting-label example with a
model-generated name and date. Coherence and numerical restoration do not
establish improved truth or semantic necessity. The prior deletion control's
behavior remains relevant, although it was correctly omitted from this smaller
panel.

## Formation versus memory: what to target next

A direct initial-prompt decomposition separates the residual CN formation error
F=CN/full−NN/full from the memory increment M=CN/compact−CN/full, using the same
native denominator:

| Case | Attention norm F | Attention norm M | First-logit norm F | First-logit norm M |
|---|---:|---:|---:|---:|
| Recall | 35.88% | 15.69% | 11.76% | 1.50% |
| Arithmetic | 27.32% | 13.94% | 11.56% | 2.27% |
| Prose | 16.85% | 14.33% | 6.43% | 0.60% |

At attention, F and M are nearly orthogonal: cosines +0.027, −0.005, and +0.004.
Memory error is substantial locally, but the initial native-consumer discrepancy
is still dominated by compiled Q/K formation. Native QKV+compact also remains
closer at every shared-history readout. These comparisons do not make the two
errors generally independent; their directions and the suffix's sensitivity
matter.

Do not grow the memory automatically because I is nonzero. A useful next
diagnostic can use these existing common-prefix arrays to separate query and
key contributions to score error, then examine their softmax/value effects:

    ΔS = (ΔQ Knᵀ + Qn ΔKᵀ + ΔQ ΔKᵀ) / sqrt(128).

The actual head scale and RoPE profile must remain included. This saved-only
analysis can prioritize a source-fitted addressing correction versus a memory
refinement; it cannot establish an autonomous native-consumer repair by itself.
Any later correction should be tested as its own recurring policy. Covariance
growth, receiver closure, and a Q-versus-K causal restoration are distinct next
questions rather than consequences of the interaction identity.

## Cost and remaining boundary

The new program retains native V and costs 819,200 selected-QKV MAC/update:
73.96% fewer than native selected QKV, but only 0.1507% less whole-token
learned-linear work. It still makes 198 matvec calls per token. The selected
memory is 98,464 bytes; full original weights, the allocated selected cache
arena, 27 other cache layers, dense receiver formation, and the native suffix
remain. Each new compact advance pays 14,336 CPU/GPU serving-transfer bytes,
bounded score/value products, routing, and CPU nonlinear/scalar work. Post-hoc
membership and full diagnostic histories are separate paid evidence work.

No new calibration or image compilation occurred; the authenticated source128
bank and historical acquisition/compilation were reused. The mrun-reported
elapsed/worker/driver times are 48.44/38.03/28.66 seconds; the scheduler timestamp
interval is 48.30 seconds. Peak RSS/VRAM are 4040.3/6406.0 MB. The experiment
establishes no latency or process-memory reduction.

The bounded conclusion is that native-value restoration survives this compact
memory law on the tested recurring trajectories, with heterogeneous gain size
and a verified finite interaction account. Whole-model native equivalence,
forward-pass-free chat, correctness improvement, and serving deployment remain
unestablished. No additional model run is required to support this result.

## Addendum: saved-only query versus key diagnosis

The proposed score diagnostic is now complete in [score_audit.py](score_audit.py)
and [score-audit.json](runs/job-b94ac2f8a6c9/score-audit.json). The latter's SHA256
is `ad7ceef5a6f67d0c03a527e12ddbd87990704751b087018e3b1dc2ff5f9416f2`.
It authenticates the original report and every consumed raw array, uses only
the three common initial prompts, and evaluates no model. The executed report
and payload remain unchanged.

The FP64 score decomposition closes relative to the native score norm within
6.3e-16, including after removing each head/clock's mean over causal positions.
The audit retains raw and row-centered terms and signed cross products. Centering
removes the softmax row-constant gauge and reduces the apparent query/key norm
ratio from roughly 1.80–1.96 to 1.50–1.59.

| Case | Centered query term | Centered key term | Centered interaction term |
|---|---:|---:|---:|
| Recall | 22.28% | 14.69% | 8.26% |
| Arithmetic | 26.37% | 16.63% | 10.59% |
| Prose | 15.09% | 10.09% | 4.58% |

These are norms relative to pooled centered native scores, not causal attribution
percentages. Query and key main-effect cross products are positive; interaction
cross products with each main effect are negative. The result is heterogeneous:
the query term exceeds the key term in only 364/672, 235/528, and 212/516
head/clock rows. Larger query defects dominate the pooled norm without dominating
every head or a majority of rows in the latter two cases.

With native V fixed, FP64 hybrid attention error is 25.39/21.43/10.91% for
native Q plus compiled K, compared with 29.19/25.08/13.42% for compiled Q plus
native K. Both compiled maps give 35.88/27.32/16.85%. The two already executed
endpoints agree with saved CUDA attention to about 1.7–2.0e-7 relative error.
The hybrid corners themselves are CPU64 calculations, not executed CUDA mixed
policies or autonomous continuations; no native-suffix or semantic repair is
established for them.

This prioritizes a better cheap query compiler objective as the next hypothesis,
while retaining key errors and nonlinear interaction. It does not recommend
native-Q serving: native Q with compiled K/native V costs 2,981,888 selected
MAC/update, saving only 5.21% there, versus 819,200 for the existing CN program.
Native Q would be a paid diagnostic ceiling. A useful replacement must preserve
consumer-relevant query directions cheaply and face new own-state tests.
