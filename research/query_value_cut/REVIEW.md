# Independent result review: QK addressing versus V payload

Reviewed 2026-11-06. Executed job: `job-27a597ebd5f8`.
Report SHA256:
`da86f60e282cfdf93cd72dffabd98c91cabb7bfb9c3b46450421cc08ca207052`.

Both exact-map restorations improve initial native-consumer fidelity. The errors
are shared between addressing and payload, with a consequential interaction;
neither a V-only bottleneck nor an independent additive-error account explains
the result. The algebra is verified using actual captured native probabilities.
The task answers do not improve: all arms retain native's incorrect follow-up
arithmetic. This is a mechanically valid localization result on repeated
development cases, not a new general-chat or correctness claim.

## Independent integrity and execution checks

This review used the executed `payload/`, saved report, and raw arrays. Current
working documentation was not substituted for the frozen scientific source.
The later provenance addition to working `ADVICE.md` is additive documentation,
not part of the executed payload.

- All 68 frozen source hashes, 308 raw file hashes/extents, and 254 candidate-seal
  entries pass. Raw files total 116,210,637 bytes. Source-manifest digest, the
  scheduler's encoded report, collected raw report, and Saturn receipt
  fingerprint/parent/result chain agree.
- All fifteen repeated historical endpoint paths match the source-span run:
  204 arrays, exact consumed histories and cache-bound turn records, plus all
  four program-bank arrays. This includes all 24 compact-memory snapshots and
  compact outputs previously independently replayed. Probability capture and
  program-bank reuse preserve those numerical endpoints.
- All 84 full-vocabulary argmax checks and all 489 generated-token choices agree
  with each path's own pending-token stream. The 21 paths make 2,113 own advances.
- All twelve full-attention probability files have the correct triangular
  extent, finite nonnegative entries, and actual value mixtures consistent with
  the saved attention outputs. Across all 1,200 recorded full-attention steps,
  the maximum relative FP64 reconstruction discrepancy is 2.35e-7, consistent
  with the native FP32 mixture. Maximum probability row-sum discrepancy is
  2.68e-7. These are observed finite-profile discrepancies, not quality gates.
- Coordinate-call and affine-MAC ledgers agree with the actual policies and
  token counts. The invocation pays 423,764 learned-linear calls and
  3,308,603,015,168 nominal MACs, including the 32 wrapper-parity advances.

## The interaction is directly accounted for

The maximal shared consumed-ID prefixes are 94, 85, and 48 positions for recall,
arithmetic, and prose. Their normalized receiver rows are byte-identical. On
those prefixes, both the expected Q/K/V operand pairs and the corresponding
native probability pairs are byte-identical. The formula instrument is therefore
applicable in all three cases.

With NN=native QK/native V, CN=compiled QK/native V, NC=native QK/compiled V, and
CC=compiled QK/compiled V, the ideal value-mixture identity is

    a_CC − a_CN − a_NC + a_NN = (P_C − P_N)(V_C − V_N).

An independent `math.fsum` implementation using the saved FP32 probabilities and
values recovers the right side. Its discrepancy from the actual four-output
interaction, relative to native attention over the maximal common prefix, is
9.10e-8 / 8.62e-8 / 8.17e-8. Subtracting the signed sum of the four native
value-mixing remainders reduces the remaining discrepancy to approximately
8.3–9.1e-17. Thus the account closes through the declared finite native mixture;
it is not merely agreement with a separately recomputed softmax.

The interaction's cross terms with both main effects are negative in all three
initial prompts, while the QK and V main-effect cross term is positive. The
interaction partly offsets their combined defect. Error norms therefore cannot
be added or independently ranked as if the two interfaces did not interact.

## Local and downstream effects

These attention errors pool the same initial prompt operands and use the native
attention norm as denominator. CN and NC reproduce the earlier saved-only
mixed-corner calculation to finite-profile roundoff.

| Case | CN attention error | NC attention error | CC attention error | Interaction norm |
|---|---:|---:|---:|---:|
| Recall | 35.88% | 35.63% | 46.77% | 20.65% |
| Arithmetic | 27.32% | 30.11% | 36.99% | 17.11% |
| Prose | 16.85% | 19.29% | 24.48% | 7.72% |

Both restorations also improve the complete first-readout logits:

| Case | CC logit error | CN logit error | NC logit error |
|---|---:|---:|---:|
| Recall | 13.84% | 11.76% | 11.11% |
| Arithmetic | 15.30% | 11.56% | 12.31% |
| Prose | 7.09% | 6.43% | 4.42% |

Prose provides a useful ranking reversal: CN has the smaller local attention
error, but NC has the smaller first-logit error. The unchanged O projection,
residual, receiving MLP, and later consumer determine which error directions
matter. A local unweighted attention norm does not fully order those effects.

Both restorations also improve recall and arithmetic follow-up prefills on
identical consumed histories. Recall CC/CN/NC errors are 16.32/11.98/13.47%;
arithmetic errors are 26.01/21.36/19.23%. Their relative ranking changes across
case and boundary. Upper-layer caches remain policy-specific even on identical
IDs; these are recurring-policy outcomes, not current-read interventions on one
shared native cache.

Later comparisons retain countertrends. Arithmetic NC's final logit discrepancy
is 121.13%, versus 64.69% for CC and 18.86% for CN, after the explanations have
diverged. Prose CN's final first-turn discrepancy is 9.68%, versus CC's 8.08%,
also on divergent histories. Preserve these as own-policy outcomes; they cannot
be used as isolated same-cache error estimates. The attention interaction law
does not extend unchanged through the nonlinear suffix or across different
token histories.

## Behavior, cost, and scope

Every arm answers the colors blue then green. Every arm gives 266 for 14×19 and
starts the follow-up with 286 instead of the correct 279. Follow-up arithmetic
explanations are budget-truncated. Prose remains coherent with changed wording
and label examples. Deletion also produces coherent outputs and the same core
answers. Numerical restoration is established here; improved truth and semantic
necessity of the selected component are not.

CN retains native V and costs 819,200 selected-QKV MACs per token, saving 73.96%
of that component's affine work. NC retains native Q/K and costs 2,981,888,
saving only 5.21%. CC costs 458,752, saving 85.42%. Against the full native
learned-linear denominator, the savings are respectively 0.1507%, 0.0106%, and
0.1741%. Each factored arm still adds a coordinate-projection call. CN is a
potentially more economical local fidelity tradeoff than NC, but these cases do
not establish a universally preferable arm or a measured speedup.

The job compiles no new basis or images; it loads the authenticated 1,835,008-byte
source128 bank. Historical source acquisition and compilation remain paid. Full
selected KV attention remains in the factorial arms; both compact controls keep
their prior 98,464-byte memory and allocated-but-unused selected cache arena.
All original model weights, dense receiver formation, 27 other layers, and the
native output consumer remain resident and active. Diagnostic probability
capture and transfers are additional paid work.

The mrun-reported elapsed/worker/driver times are 54.02/43.36/35.00 seconds;
the scheduler start-to-finish timestamp interval is 52.33 seconds. Reported resource
peaks are approximately 4,127 MB RSS and 6,380 MB VRAM. Per-path timings vary,
including slower native paths; no latency benefit follows from this single
instrumented panel.

No further model run is needed to substantiate this cut. It identifies two
comparable formation defects and an exactly accounted joint interaction, while
showing that the native suffix changes their practical ranking. A future cheap
repair should target their joint consumer response under source evidence,
rather than choosing directions solely by raw V error or treating QK and V
errors as independent. Separate Q versus K attribution, compact receiver
closure, and covariance-aware memory remain distinct open questions. These
mixed arms are paid native-map diagnostics, not forward-pass-free whole-model
emulators or installed serving policies.
