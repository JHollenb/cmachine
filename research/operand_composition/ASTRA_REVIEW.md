# Astra review: a connected approximate writer and reader

Reviewed November 6, 2026. This is an additive interpretation of completed job
`job-d9f9f2405fc4`; no native execution, fitting or changes to its raw evidence.
Primary report SHA256:
`a89b911249672f37157de83d66510c343ae967df9d8f554f10e4ef3cc5cc9a50`.

The experiment establishes a connected, self-fed replacement segment. Frozen
coupled64 MLP16 factors produce the actual residual that frozen B128 query/key
factors at attention17 consume. Each primary arm starts empty, writes its own
full FP32 K/V, chooses and consumes its tokens, and receives a second user turn.
The instrument completed 24 own trajectories and 18 diagnostic replays, totaling
4,691 main token transitions. Exact same-history seam controls establish the
declared dependency and prevent a native receiver refresh from masquerading as
composition. This is a bounded executable program, not improved whole-model
emulation or a general sufficient compact state.

## Independent numerical check

`astra_check.py` reads the immutable FP32 files with Python's standard library,
authenticates each loaded file against the report, and computes norms and KL
using double arithmetic and `math.fsum`. Its reproducible output is
`runs/job-d9f9f2405fc4/analysis/astra-check.json`. It does not import the main
analyzer or NumPy. Run it with:

```sh
rtk proxy python3 saturn/cmachine/operand_composition/astra_check.py saturn/cmachine/operand_composition/runs/job-d9f9f2405fc4
```

The independent logit and attention calculations agree with the recorded trends:

| Matched-history mean relative full-logit error | Writer only | Reader only | Joint |
|---|---:|---:|---:|
| Initial prompts, six cases |14.91%|6.15%|15.89%|
| All 24 boundaries |15.91%|9.56%|18.22%|

Joint improves over writer-only at 6/24 L2 boundaries and 11/24 native-to-candidate
KL boundaries. Mean logit interaction is 7.50%, maximum 42.29% at case3's second
answer endpoint. These are correlated observations from six conversations, not
24 independent tasks. The component and interaction vectors have signed cross
terms; their norms are not additive percentages of responsibility.

The reader's conditional attention error averages 53.70% with the native writer
and 53.60% with the cheap writer. Each is normalized by its corresponding
native-reader context at the same writer policy. Mean paired ratio is 1.011,
range 0.883–1.216. Thus this seam does not produce a broad average explosion of
the reader's already substantial error. It also does not establish equivalence,
robustness outside these cases, or absence of individual adverse effects.
These comparisons include each policy's historical keys on common token IDs;
they do not isolate the current query alone. Attention and logit percentages
have different denominators and do not establish contraction between layers.

## The saved Q × K cross sharpens the diagnosis

The additive `reader_attribution.py` uses CPU64 attention on actual saved
operands, crossing the current native/compiled query and complete
native/compiled historical key bank. At each writer and fixed token history,
the value bank is byte-identical between reader policies. No hybrid is advanced
through a new native suffix.

| Fixed writer | Q-only context error | K-only context error | Both | Q/K interaction |
|---|---:|---:|---:|---:|
| Native |43.356%|38.086%|53.697%|25.782%|
| Compiled |41.591%|38.466%|53.601%|24.531%|

These are conditional saved-reader effects, with each row's native Q/K response
as denominator. They are neither downstream causal shares nor independent
effects that can be added as norms. Query operand error is about 33.546% under
the native writer and 33.854% under the compiled writer; key-bank error is
29.191% and 29.794%. Both addressing maps are already rough on these cases,
before adding the cheap writer. A diagnosis solely blaming producer-induced
distribution shift is therefore unsupported. The cross is compatible with
query and key compensation, but does not prove which future compiler change
will improve a final consumer.

## Truth, fidelity and token authority

All policies answer the simple green correction correctly. Reader-only gives
Bo the correct yellow card in the swap case, while native, writer-only and joint
return purple. Its full sentence violates the requested one-color format; it
does terminate normally. Only joint changes native's incorrect 401 for
17×23 to correct 391. Joint then gives 432 rather than 410 after being asked to
add 19. The followup explanation is truncated, but the emitted integer is
already wrong. Prose remains understandable, with differing trajectories and
several budget-truncated outputs. This panel establishes neither general
correctness nor monotonic behavioral improvement.

Texts recorded under replay policies decode the supplied joint11 answer IDs;
they are not answers those policies generated. Matched-history comparisons
measure recurring policies on a common input stream. Comparisons across their
own divergent histories additionally contain generated-history effects.

## What follows for the architecture and compiler

Retaining the full candidate residual and broad paired memory was useful: it
removed the ambiguity of whether a tiny code could form the next operation's
input. The candidate writes the entire residual, and each next operator forms
its own coordinates from that actual residual. We should keep this boundary
while improving the operators, rather than simultaneously introducing the
known-failing bounded-retention policy.

Joint execution saves 28,606,464 nominal learned-linear MACs per token, 1.8533%
of the original total, while logical linear calls rise from 197 to 199. Static
factors, dense D/O/V, other layers, full learned weights, full K/V and vocabulary
computation remain paid. The instrumented runtime is not a serving-latency
comparison or forward-pass-free whole-model result.

The next compiler question is more specific than increasing rank: can an
addressing program fitted through its actual broad-memory consumer improve the
connected computation at the same serving budget? Retain this frozen baseline.
Use declared source histories with both native and cheap-writer receivers and
their own K/V, so adaptation to one parent does not hide a regression on the
other. The saved Q/K cross argues for evaluating both addressing factors rather
than treating Q-only retuning as the entire remaining problem. Any new proposal
must be selected on designated source/monitor consumers, with later-state and
probability effects retained, then sealed before new evaluations. This is a
proposed compiler experiment, not a result or authorization for another run.

## Numerical portability qualification

The original saved verifier refused a Mac-versus-Beast CPU64 diagnostic scalar:
1.4602355654099607e-5 versus 1.4602355642265408e-5, an absolute difference of about
1.18e-14, just beyond its 1e-14 absolute comparison threshold. Preserve that
refusal and both values. It does not imply changed raw arrays or native
execution. The independently recomputed main scientific trends agree.

An additive portability verifier can retain exact checks for file hashes,
identities, schemas, finite arrays, integer counts, token authority and
structural equality while recording floating diagnostic differences without an
uncalibrated boolean quality threshold. This is a changed verification contract,
not a retroactive pass of the original scalar-parity rule. Custody/mechanics
and cross-platform diagnostic arithmetic should be reported separately.
