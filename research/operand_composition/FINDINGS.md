# Two cheap components execute together; consumer fidelity remains open

November 6, 2026. Frozen rank-64 MLP16 and rank-128 Q/K17 programs now execute as
one connected segment of Qwen2.5-1.5B-Instruct. Each arm starts empty, produces
its own residuals and full FP32 K/V, generates and consumes its own answers, and
continues through a second user turn. The interface executes correctly. The
joint program is generally less faithful to native readout than either isolated
replacement, with useful behavioral gains and adverse cases retained.

Main mrun job: `job-d9f9f2405fc4`. [Immutable report](runs/job-d9f9f2405fc4/report.json),
SHA256 `a89b911249672f37157de83d66510c343ae967df9d8f554f10e4ef3cc5cc9a50`.
[Verification](runs/job-d9f9f2405fc4/verification.json),
[scientific analysis](runs/job-d9f9f2405fc4/analysis/scientific-analysis.json),
[saved query/key crosses](runs/job-d9f9f2405fc4/analysis/reader-attribution.json),
[independent Astra check](runs/job-d9f9f2405fc4/analysis/astra-check.json),
[Astra review](ASTRA_REVIEW.md), [plan](PLAN.md), [equations](MATH.md).

## Frozen component and numerical boundary

Import final `coupled64.npz` from contextual-union `job-7a1a2715be48` and the
selected response query/B128/K128 from query_response `job-b141eadfe952`.
Source reports, selection seal, emitted row-major FP32 factors and original
checkpoint SHA are joined before execution. MLP factors consume the existing
post-RMSNorm/gamma input; Q/K retain native bias and half-split RoPE. Original
dense down/O/V, all other layers, embedding and vocabulary head remain.
No fitting or candidate selection uses these new conversations. The query's
historical fit used a prototype reader; full-memory serving is explicitly a
transfer, not an assertion that this query is optimal for the present reader.

All four arms use the same full FP32 cache geometry and the same C FP32/pedantic
runtime. Native C is the fidelity reference; this run does not certify byte
parity between C and the predecessor PyTorch execution. Source weights are the
same. The loaded model owns 6,174,857,216 FP32 weight bytes. Each session owns
30,096,896 GPU bytes. Imported factors/scratch add 6,685,440 GPU bytes shared
across the sequential panel. No memory compression or weight-residency saving
is measured in this experiment.

## Executed geometry and seam controls

Six fresh conversations cover two correction/recall, two arithmetic and two
prose cases. Budgets are 8/8/16/16/32/32 tokens per answer, including consumed
EOS. Four own policies per case: 00 native, 10 cheap MLP, 01 cheap Q/K, 11 both.
All 24 own trajectories and their metadata are sealed before 18 diagnostic
replays: 00/10/01 consume the actual joint11 IDs, with own11 supplying the fourth
corner. Those replay answer strings are supplied IDs, not replay-policy outputs.

There are 4,691 main transitions plus 32 native-sham advances. All own emitted
tokens are actual prior pending selections and are consumed once. On matched
histories, MLP16 normalized input and layer16 K/V are byte exact across arms.
MLP16 write/output and attention17 normalized receiver/V are reader-independent
(00=01 and 10=11) exactly. This confirms that the reader consumes the writer's
actual result without a native refresh. Shapes, finiteness, work counters,
fixed allocations, imported-program immutability and own/replay authority close.

Independent CPU64 equations reproduce the saved consumed attention to maximum
relative difference 2.994e-7. Gate/up map diagnostics reach at most 1.215e-7;
query/key bias-plus-RoPE diagnostics reach at most 7.069e-7. These are numerical
instrument observations under different reduction/libm profiles, not fidelity
thresholds or claims of CPU64/CUDA byte parity.

## Readout and interaction

Mean full-vocabulary relative L2 against native **on the joint arm's matched
consumed history**, with each policy retaining its own computed memory:

| Boundaries | MLP only | Q/K only | Both |
|---|---:|---:|---:|
| Initial prefill, six cases |14.9109%|6.1486%|15.8944%|
| All four boundaries per case |15.9078%|9.5641%|18.2227%|

The 24 boundaries are correlated clocks from six conversations. Joint improves
over MLP-only at 6/24 L2 boundaries and 11/24 native-to-candidate KL boundaries.
Mean native-to-candidate KL is .084758/.025048/.092909 for 10/01/11. All three
match native's pending top at 21/24 matched boundaries, illustrating why top
agreement alone does not certify emulation. Separate own-history comparisons
retain whether the consumed histories actually match.

For Y at a common consumed history, A=Y10−Y00, B=Y01−Y00 and
I=Y11−Y10−Y01+Y00. The raw four vectors retain every signed component. The mean
relative logit interaction is 7.5009%, maximum 42.2865% at case3 second-answer
end. Eight of 24 A/B cross terms are negative. Their norms are not additive
shares of responsibility, and these recurring-policy interactions do not
isolate a single instantaneous operator perturbation.

## Conditional reader fidelity and query/key diagnosis

At the same 24 boundaries, the Q/K replacement's attention discrepancy against
native addressing averages 53.6966% with the native writer and 53.6013% with the
cheap writer. The mean paired ratio is 1.0111, range .8827–1.2162. The cheap
writer therefore does not cause a broad mean increase in the reader's already
large discrepancy; individual increases and decreases remain visible.

Saved-only CPU64 query × key-bank crosses hold the writer, consumed IDs and V
fixed. They make no new model/future call:

| Writer | Query only | Key bank only | Both | Query/key interaction |
|---|---:|---:|---:|---:|
| Native |43.3561%|38.0861%|53.6966%|25.7820%|
| Compiled |41.5907%|38.4662%|53.6013%|24.5310%|

Entries are mean relative attention-context differences against the corresponding
native-addressing context. They are conditional numerical contrasts, not causal
shares of an answer. Actual query-vector errors average33.5461%/33.8540%; key-bank
errors29.1907%/29.7938%, for native/compiled writers. Both addressing maps are
rough on these cases; attributing the gap solely to writer-induced distribution
shift or solely to the query would miss observed terms. Smaller relative logit
numbers do not establish a contraction because the observables have different
norms and consumers.

## Actual answers and countertrends

- All arms answer the simple badge correction with `green`.
- In the card swap, native/MLP-only/joint answer incorrect `purple`. Q/K-only
  answers `Bo has a yellow card now.`, which contains the correct binding but
  violates the requested one-color format. Composition loses that local gain.
- Native, MLP-only and Q/K-only answer17×23 with401. Joint answers correct391.
  Its next answer is432, not391+19=410. The other arms answer442, also incorrect.
- All arms contain434 and407 in the31×14 then subtract27 case. Explanations
  differ, some contain inconsistent checks, and second replies are truncated.
- Prose remains readable with different wording and several capped outputs.
  Completion coverage remains part of the record; capped prose is not counted
  as a completed-task success.

Native fidelity and task truth are therefore distinct. Neither success on one
arithmetic answer nor21/24 top agreement establishes accurate chat emulation.

## Cost, custody and interpretation

The joint program executes199 logical linear calls rather than197, but replaces
expensive maps with smaller factors. Nominal linear MACs fall
1,543,569,408→1,514,962,944 per completed token:28,606,464 saved, or1.8533%.
Attention-over-history and all instrumentation are additional. The lease lasts
95.14s, worker69.40s and C invocation60.87s; peak RSS4126.7MB/VRAM6416MB. These
are instrumented execution timings, not a native-versus-candidate speed trial.

Saved verification authenticates79 payload sources and1210 raw files, joins the
scheduler log/report/raw receipt, and independently repeats strict mechanics.
Raw declared artifacts total360,052,878 bytes. Astra independently authenticates
120 selected raw files and recomputes all24 matched logit/attention boundaries
using stdlib `math.fsum`; means and L2/KL win counts agree.

The original verifier's CPU64 scalar-equality rule refuses a cross-platform
diagnostic difference of1.18e-14. Its [original refusal](runs/job-d9f9f2405fc4/corrections/original-numerical-verification-refusal.txt)
and executed source remain unchanged. [Additive verification](verify_portability.py)
retains schemas/finiteness and all custody/mechanics exactly while recording
every signed numerical scalar difference. Maximum drift is approximately2.70e-14; both
reported and recomputed values remain in the [portability record](runs/job-d9f9f2405fc4/corrections/numerical-portability-v1.json).
Custody and mechanics are verified; original cross-platform scalar equality is
not established under that rule. No raw observation is rewritten.

The isolated Saturn evidence run/verify both return0 over5816 observations,
29 applicable predicates, no fires, no claims and no address citations. This is
stream coverage, not terminal certification.

The working inference is that retaining the full residual and broad memory gives
a workable interface for connected cheap computations. The unresolved problem
is the accuracy and coverage of those computations and their stored futures.
The next bounded compiler experiment should fit Q/K addressing through actual
broad-memory consumption on designated native/cheap-writer source trajectories,
retain both immediate and future effects under reversible selection, then use
fresh sealed evaluation cases at unchanged serving budget. A Q-only repair or
cheap-writer-only adaptation is not justified by the two-map diagnosis.
