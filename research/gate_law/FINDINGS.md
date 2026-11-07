# A source-anchored gate law advances through its own conversation

November 6, 2026. A frozen scalar law now substitutes for layer16 SiLU in the C Qwen2.5-1.5B-Instruct machine while consuming its own future tokens and complete cache, including subsequent user/header ingestion. The law is closer to the same-input native MLP write than zero at all93 evaluated uses. Repeated application accumulates state error, and native arithmetic mistakes remain visible. Original dense projections and full model state remain paid.

This is a bounded local computation result on two development conversations. The93 evaluations include91 recurring applications and two duplicate initial once observations; they are not93 independent trials. The earlier B3 program drops projection complements; this new program retains dense projections and approximates one nonlinearity. Their numerical errors are not a matched benchmark of the two methods.

## What was implemented

`QwenGateLaw` stores five immutable FP32 channel vectors: source gate g*, sigmoid p*, first/second sigmoid Taylor coefficients and source SiLU s*. For current branch-computed g and u,

$$
d=g-g_*,\quad \sigma_2=(p_*+c_1d)+(c_2d)d,\quad
\widetilde s=s_*+(g\sigma_2-g_*p_*),\quad
\widetilde m=D(\widetilde s\odot u).
$$

Contraction is disabled; the displayed grouping is executed. This sigmoid-Taylor2-derived cubic expression is not the full cubic Taylor expansion of SiLU. Source coefficient compilation uses native exponentials once per anchor. Its future invocation uses no refreshed target anchor or exact selected-site SiLU. Dense G/U/down, all other layers, attention, normalization, residuals, K/V and full-vocabulary readout remain native.

The source gate comes from the last consumed prompt row, at positions54/42. Each candidate first consumes the known final prompt newline at55/43; it receives no native answer token. Once applies only there. Recurring applies to every later consumed token, including follow-up templates. Zero skips G/U/down at this site while preserving the residual. Every emitted token is consumed once. Each answer has a16-token budget; truncation is external.

All six candidate paths and1,941 input/parameter/output files seal before native reference paths and exact diagnostics. Native own-history generation, native rebuild on the candidate's exact consumed IDs, and native one-step execution on the actual candidate cache remain distinct. The [protocol](PLAN.md), [driver](driver.c), [worker](worker.py) and [analyzer](analysis.py) preserve that order.

## Native behavior and truth

| Policy | Niko color | Mira after the supplied green update | 17 ×23 | Subsequent subtraction reply |
|---|---|---|---|---|
| Native | blue + EOS | green + EOS |401 + EOS|`157. The result of 401 minus 24 is` |
| Once | blue + EOS | green + EOS |401 + EOS|Same native token sequence |
| Recurring | blue + EOS | green + EOS |401 + EOS|`157. Subtracting 24 from 157 gives the` |
| Zero | blue + EOS | green + EOS |409 + EOS|`409 - 24 = 385. Subtracting ` |

All arithmetic follow-up replies hit the16-token limit; their unfinished explanations cannot be ranked as completed answers. Correct multiplication is391 and the intended correct-result subtraction is367. Native's157 is also inconsistent with its own prior401, whose subtraction should yield377. Zero's multiplication409 is wrong but its385 correctly subtracts24 from that own answer. Thus deletion loses model-output fidelity while improving this specific consistency observable. The experiment establishes no arithmetic truth improvement from the gate law.

Recurring arithmetic prose first differs from native at zero-based answer index4 on turn2. Once reproduces every tested native answer ID. All policies preserve the simple color answers, so those colors alone do not establish a necessary semantic computation.

## Local law and continuing-state error

| Measurement | Recall conversation | Arithmetic conversation |
|---|---:|---:|
| First law-write relative error |6.2233%|10.9851%|
| Recurring mean same-input write error |24.4219%|30.5931%|
| Recurring maximum same-input write error |69.8377%|79.0032%|
| Follow-up full-logit difference, native rebuilt on same IDs |4.1081%|10.3910%|
| Follow-up once difference, native rebuilt on same IDs |0.1265%|0.5437%|

All93 polynomial products independently reproduce the actual FP32 arrays, and all93 write errors are below the zero-vector relative distance1. This is a useful local trend under the two frozen source anchors. It gives no global accuracy domain or error bound for arbitrary future chat.

![Local law error, continuing-state discrepancy and remaining computation](runs/job-91ba52c5fe9a/gate-law-results.png)

The [posthoc continuation account](runs/job-91ba52c5fe9a/continuation-analysis.json) makes the different errors explicit. Let C be candidate logits, L a native step on the candidate's actual parent cache, H native reconstruction on its exact consumed IDs, and N native own-history logits:

$$
C-N=(C-L)+(L-H)+(H-N).
$$

These are outcome differences under explicit controls. Their norms do not add. The saved-vector identity closes exactly at FP64 vector precision here; squared norms retain all signed cross terms.

At the recurring follow-up boundary H=N in both conversations, since the consumed histories still agree. Using one denominator, the norm of N:

| Contribution | Recall | Arithmetic |
|---|---:|---:|
| Current-law defect C−L |0.8215%|3.5827%|
| Accumulated-state effect L−H |4.0768%|9.4407%|
| Total C−N |4.1081%|10.3910%|
| Cosine between current and accumulated effects |−0.06251|+0.08885|

The main report's local0.8356%/3.5395% uses L as its denominator instead; these numbers must not be subtracted from its same-ID percentages. For once, the current step is native and C=L exactly, yet the historical state effect persists at0.1265%/0.5437%. Executing an exact operation now does not erase earlier cache error.

All273 primary current-KV16 rows match their native same-ID reconstructions exactly; upper K/V differs. This observed boundary agrees with the architectural ordering: MLP16 writes after KV16 formation. The study does not capture every lower-layer array, establish compact sufficient memory or show within-chat memoization.

## Computation comparison

| Program | Dense matvecs per consumed token | Learned-linear MACs per token | Other paid work |
|---|---:|---:|---|
| Native C |197|1,543,569,408|28 native SwiGLU sites, attention, norms, RoPE, cache and readout |
| Recurring gate law |197|1,543,569,408|27 native SwiGLUs + one polynomial; full state/weights |
| Earlier B3 PROJECT |195 at its selected site policy|Nominal27,525,120 fewer, about1.7832%|Native nonlinear/down, QR/basis operations and complete remaining model; different assay |
| Zero MLP16 |194|1,502,281,728|Deletes the write, saves2.6748% of learned-linear MACs |

The gate law occupies179,200GPU bytes per anchor. Two source laws stay resident during the panel, with temporary rebuild/control anchors separately paid. It replaces8,960 selected-site exponential evaluations per recurring token, adds polynomial operations and coefficient reads, and retains the expensive learned affine maps. No clean speedup measurement is established. Instrumented recurring recall takes0.4753s versus native0.4716s; arithmetic takes2.8914s versus0.6145s. These timings include captures, finite checks, hashing and file I/O in different sequential positions. Candidate replay timing also includes native probe work.

The complete experiment performs1,216 C token transitions and239,099 dense matvecs including93 saved-operand down diagnostics; source coefficient compilation runs six times for original, reconstructed and altered-source controls. Those controls change the law bytes in both cases. The scientific driver takes30.24s, worker39.83s and scheduler lease51.53s, with peak RSS3,981.4MB and VRAM6,378MB. These are paid experimental costs.

## Evidence and implementation boundary

[`job-91ba52c5fe9a`](runs/job-91ba52c5fe9a/report.json) is immutable at SHA256 `d8173d0206fd2a52cffa44f8f17bd5a2fd9dd1c7b33a8432d2fc133415627ee4`. The [verifier](runs/job-91ba52c5fe9a/verification.json) authenticates68sources,6,041raw files totaling146,790,026bytes, the scheduler-log/report join and a public whole-invocation Saturn receipt. Candidate replay reproduces1,935 arrays exactly. Astra independently verifies all source/raw seals, primary prefix, full92readouts/argmaxes, own-token authority and FP32 polynomial products using a separate rounding implementation. [Review](REVIEW.md).

Six focused Python checks,13 existing checks and ASan/UBSan lifecycle pass. The preceding corrected real-model [smoke](runs/job-a149cfcf8097/report.json) retains151,936 byte-identical historical/current ordinary logits, matching IDs and checkpoint replay. The first tokenizer-harness failure remains preserved. The standard [evidence sweep](index/job-91ba52c5fe9a/evidence-run.json) and [verification](index/job-91ba52c5fe9a/evidence-verify.json) cover129rows with24applicable predicates and no fires/automatically admitted claims or addresses. This stream coverage is separate from scientific interpretation.

The hook is explicit and experimental; normal serving never enables it automatically. A nonfinite gate refusal leaves a partial token and exits this assay with partial evidence, so failure-atomic continuation is not claimed. The active installed Beast release is unchanged. The source and tested job binaries retain the implementation. No terminal certification, compact sufficient chat, general task accuracy or whole-model forward-pass elimination is established.

## What forward-pass-free emulation requires next

The current C engine already avoids a framework `forward()` API call, while calculating the dense model's equations. To remove that learned forward work, the next programs must replace the actual affine operand formation and update their own state across future invocations.

For an encoding c=E(S), the target on a declared reachable future domain is

$$
E(T_\theta(S,u))=\overline T(E(S),u),\qquad
R_\theta(S)=\overline R(E(S)).
$$

Approximate versions need a declared consumer, horizon and error budget. The domain must include the state produced by the emulator itself. The most useful assumption is a reusable operand interface that remains constructible after its own updates. A universal tiny subspace or global low-degree polynomial is unnecessary and currently unsupported.

The immediate next scientific cut is a composed gate/up/write program on its own receivers, compared with the native and deletion controls here. The existing B3 complement identity supplies a diagnostic of what it loses. Preserve native context while discovering which complementary operands make that cheaper program useful; do not refresh them from the target computation and count that as a saving.

The newer [finite-interface collision study](../closure/FINDINGS.md) makes that requirement concrete. Three receiver pairs have exactly equal executed B3 coefficient/RMS bytes and different native writes, separated by 10.6575, 17.8135 and 20.8831 L2. Any deterministic predictor using only that code must incur at least half the pair separation on one endpoint. Refitting the same interface cannot recover the missing distinction. Our gate law retains the full native gate/up operands and therefore pays for information that this code omits. This collision result concerns supplied interventions at one local interface, not a universal impossibility theorem for compact chat.

There is a constructive simplifying assumption. For a fixed block x⁺=x+D f(Ax,Bx,||x||²), suppose an orthonormal row basis R contains every admitted reader direction and every write direction. With z=Rx and κ=||(I−RᵀR)x||², the omitted orientation is invisible and unchanged:

$$
z^+=z+RD f(AR^\top z,BR^\top z,\|z\|^2+\kappa),\qquad \kappa^+=\kappa.
$$

This is an exact real-arithmetic state program under the assumption, retaining the actual nonlinearity. It becomes a reduction only when the required space is sufficiently small; a full-rank reader/write family can make it non-reducing. The [derivation](../closure/MATH.md#a-constructive-assumption-that-makes-nonlinear-closure-possible) is not a measured reduced space for Qwen or a finite-precision qualification. It identifies the structural condition to search for: discarded information must remain invisible to the declared future consumers except through explicitly retained quantities.

Then address the other large costs: attention's response over future queries and changing support, all-layer K/V formation, and vocabulary readout. One evaluated attention average does not determine the next query's average. A retained N(q),Z(q) response needs an update law over the queries actually reached. The source-only symbol table can bind those laws to operands, profile and validity domain; the time circuit supplies dependency clocks and own-state updates. Neither label supplies a missing numerical transition.

Exact dependency elimination can proceed alongside approximation. For example, prompt tokens whose readout is never consumed may permit omitting intermediate vocabulary projections, once that consumer boundary is implemented and qualified. It would remove actual paid work without claiming semantic compression. Automatic reuse, memory reduction or serving promotion requires its own evidence. Accurate arithmetic and factual chat also need a task-quality contract separate from native-model fidelity.
