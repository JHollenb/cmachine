# Equal compact codes do not determine the next native MLP write

2026-11-06. [Native job job-7b8a5933f1dd](runs/job-7b8a5933f1dd/report.json) supplies three exact finite-interface counterexamples. Different receiving vectors have identical 16-byte B3 codes—three executed FP32 coefficients and actual native FP32 RMS scale—but different native MLP16 writes. The differences reach full logits and persistent memory with the residual skip held fixed. All selected fiber continuations retain native token IDs within this short horizon.

This establishes insufficiency of this particular local interface on the supplied intervention domain. It does not establish impossibility of other compact states, natural-context collision frequency, unrestricted chat quality or a serving advantage. Native/full-state own-token execution already existed; the advance is a direct test of what the proposed interface discards.

## Declared state and exact witness

The [plan](PLAN.md), [dependency inventory](STATE-INTERFACE.md) and [mathematical account](MATH.md) define the interface and domain. All three source points are the previously observed recall, arithmetic and prose final-query receivers. A fixed seeded spherical construction generates 64 candidates per receiver, preserving QR projections and norm in real arithmetic. The actual unchanged GPU RMS and runtime QR code then determine exact finite equality.

The run finds 14, 38 and 44 exact source-code collisions among those 64 candidates. Before evaluating any candidate MLP output, it selects the first collision in the declared angle/direction/sign order. Each selected angle is −1 radian; directions are 3, 0 and 0. The selected receiving-vector separations are 14.947, 21.936 and 29.815 L2. Each selected 16-byte code is identical to its own source code; the three source codes need not equal one another.

| Known source | Native write separation L2 | Separation / source write norm | Any code-only predictor's two-point lower bound L2 | Alternate write vs paid complement repair, relative to alternate |
|---|---:|---:|---:|---:|
| Recall |10.657487|61.75%|5.328744|6.27×10⁻⁷|
| Arithmetic |17.813451|84.13%|8.906726|6.76×10⁻⁷|
| Prose |20.883055|118.66%|10.441528|7.01×10⁻⁷|

For two outputs Y0,Y1 sharing one code, the triangle inequality gives max_i||h(c)−Yi||≥||Y1−Y0||/2 for any deterministic code-only predictor. The measured separations provide that bounded lower bound independently of how PROJECT is implemented. The original report's generic relative_l2 field normalizes by its right-hand endpoint; this table explicitly normalizes separation by the source write instead. Raw values remain unchanged.

PROJECT's gate, up and write arrays are byte-identical across each exact-code pair. Its saved normalized-input arrays are diagnostics of the different actual receivers, not inputs consumed by the code-only combination. Keeping their difference visible avoids claiming that normalization itself reconstructed the omitted information.

## The missing operand and the positive control

Writing r=Bc+e, the native gate/up operands contain both basis images and the original-weight images of e. Equal coefficients and rho do not determine those complementary images. The nonlinearity consumes their actual values before the down map.

Adding the selected alternate's own complementary G/U images recovers its native write to 6.27×10⁻⁷–7.01×10⁻⁷ relative difference. This predicts the alternate computation; it does not turn it into the source computation. The correction pays two original dense projections. Full-state corrected recurrence is a separate control and retains native generated tokens in all three cases.

This answers a local sufficiency question with both a counterexample and a repair operand. A cheap update for that extra information remains open. Retaining full e restores the original receiver; retaining two intermediate-width images can cost more than the receiver. Neither is a demonstrated compact chat state.

## The unchanged native suffix consumes the difference

The held-write API installs the measured writes while preserving the actual source residual skip and all earlier prefix state. Native source receiver/write replay and native-write sham are exact. This prevents a direct change to the skip from explaining the consumer effect.

| Source | First full-logit L2 difference | Difference / native logit norm | Current K/V17 L2 difference | Fiber token agreement |
|---|---:|---:|---:|---|
| Recall |170.335080|8.95%|7.728313|2/2 including EOS|
| Arithmetic |203.918107|13.72%|12.217387|4/4 including EOS|
| Prose |238.578468|17.13%|13.656691|8/8; budget truncated|

At the intervention, current K/V16 is byte-identical and K/V17 changes. On every later matching-history fiber transition, the newly appended K/V16 and K/V17 rows are byte-identical to native, yet complete valid-prefix cache hashes and full-logit hashes remain different. Old edited memory remains readable. Layer17 can consume its changed historical row after forming its new current row; the first permitted new-cache consequence is then K/V18, which this assay does not capture.

Every displayed/generated token is consumed exactly once. The prose window does not finish within the eight-token budget, so its agreement concerns a prefix, not a complete response. Compact once/recurring already differ from native at the first prose token; the recurring zero-write path first differs at emitted index 2. Recall and multiplication remain blue/391 under every policy. Those answers remain insensitive consumers here.

The 24 paths contain 136 recorded transitions: 24 final-question steps plus 112 emitted-and-consumed tokens. Same-history and different-history comparisons remain separate. Own-state recurrence is operational; whole-model reduced-state closure is not established by these paths.

The separate [whole-turn result](../composition/whole_turn/FINDINGS.md) now executes B3 throughout a complete user correction and two independent later reads. All policies, including deletion, preserve the updated and collateral badge answers; numerical PROJECT-versus-zero trends reverse between pre-answer and post-EOS boundaries. This complementary ingestion result is a different job and does not enter this assay's counts or sufficiency proof.

## Attention has the same structural obstacle

A supplied two-key calibration executes the unchanged attention kernel. Two key assignments give identical q=0 summaries and outputs. With the same new query, their outputs are +0.761594057 and −0.761594057. Appending a common zero-key/value+1 row gives equal q=0 outputs of 0.333333343, while the changed-query outputs are 0.819938779 and −0.330481797.

The real-arithmetic result is the explicit tanh witness in [MATH.md](MATH.md). It shows why a summary for one query does not answer another. Fixed-query numerator/partition append has a closed update; arbitrary future queries need retained query-dependent information. These are known-wiring toy operands through the native primitive, not a learned semantic-attention result.

## Mechanics, cost and provenance

The scheduler lease takes 25.36 seconds, C driver 14.23 seconds, and worker 21.72 seconds, with peak RSS 3790 MB and VRAM 6378 MB. One original resident C FP32 model, batch 1, capacity 512 and one CPU thread execute all branches. The component panel additionally calculates 15 standalone MLP writes, and the 192 candidate encoders perform native RMS plus exact runtime-coordinate checks. These costs include dense computation, diagnostics and hashing; no inference speedup is claimed.

The [post-collection verifier](runs/job-7b8a5933f1dd/verification.json) authenticates 81 source files, 966 raw records, scheduler/log/report joins, actual loaded-model digest, unchanged bank and exact geometry binding. It reproduces the executed analysis without a model. The sealed worker omitted an explicit loaded-model comparison; the verifier performs that check against the authenticated predecessor before scientific interpretation. The exact [executed binary](runs/job-7b8a5933f1dd/binary/receipt.json) was subsequently retrieved read-only and matches its recorded hash. No native rerun or report rewrite was needed.

The [independent Astra audit](ADVISOR-RESULTS.md) verifies all 966 raw hashes, 958 finite FP32 arrays, 105 native/sham array counterparts and all 17 paired step logits/cache hashes. The [additive study](study/study.json) separates relative-error denominators, current cache rows, complete prefix memory and same-history trajectories. The [offline evidence sweep](index/job-7b8a5933f1dd/evidence-run.json) and [verification](index/job-7b8a5933f1dd/evidence-verify.json) cover 227 scalar rows: 192 candidate-code diagnostics, 3 primary writes, 24 query-step policy rows and 8 attention calibrations. They complete with no watch findings or drift. Full continuation vectors and histories are audited separately; the sweep is stream coverage, not semantic certification.

## A constructive simplifying assumption

The [final mathematical section](MATH.md#a-constructive-assumption-that-makes-nonlinear-closure-possible) gives an exact real-arithmetic nonlinear quotient for a fixed residual block when every permitted linear read and write direction shares a sufficiently small space. The omitted orientation then remains invariant and only its retained energy affects normalization. The state updates without replacing the nonlinear function by a global polynomial. This is a conditional derivation, not a measured rank reduction of Qwen; full-rank read/write families can make the construction non-reducing.

Sources: [driver](driver.c), [geometry](geometry.py), [executed analyzer](runs/job-7b8a5933f1dd/payload/analyze.py), [native report](runs/job-7b8a5933f1dd/report.json), [mathematical advice](../../experiments/2026-11-05-operational-variable-closure/ADVISOR-STATE-CLOSURE.md), and [prior exact affine positive control](../../../model-experiments/experiments/2026-10-08-future-qualified-state/FINDINGS.md). Independent review and standard evidence-stream coverage are recorded separately from scientific interpretation.
