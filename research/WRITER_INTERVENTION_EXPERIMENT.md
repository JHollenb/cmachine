# Follow an actual layer15 writer through its receiving computations

Status: completed2026-11-06 by[`job-87eed6acd554`](results/job-87eed6acd554/report.json), with [findings](FINDINGS.md#actual-writer-projections-remove-a-native-cancellation) and [independent review](../../artifacts/cmachine-actual-writers-20261106/REVIEW.md). The exact pre-run protocol remains frozen in the job's payload. Follows the completed [carry-formation panel](FINDINGS.md#inherited-and-newly-written-carry-form-a-coupled-input).

The preceding panel establishes coupled behavior when supplying projected correction operands to MLP16. It does not remove or change an originating layer15 writer. This next experiment changes actual layer15 writes and lets later receivers, attention, normalization and memory evolve from that edited state.

Retain the original checkpoint/C FP32 contract, source conversation, source own answer, Mira calibration and frozen B3 hashes. Freeze one new question before any target run:

> Which badge color belongs to Niko? Answer only that color.

Let P=Q3Q3ᵀ be the existing researcher-supplied projector in residual coordinates. Declare the sequential FP64 projection and FP32 materialization explicitly. Do not renormalize the projected writes. From one immutable token parent, run four policies:

$$
a=A_{15}(x,H),\qquad
a_\alpha=\begin{cases}a&\alpha=0\\\operatorname{FP32}(Pa)&\alpha=1,\end{cases}
$$

$$
r_\alpha=\operatorname{FP32}(x+a_\alpha),\qquad m_\alpha=M_{15}(r_\alpha),
$$

$$
m_{\alpha\beta}=\begin{cases}m_\alpha&\beta=0\\\operatorname{FP32}(Pm_\alpha)&\beta=1,\end{cases}
\qquad x_{16}^{\alpha\beta}=\operatorname{FP32}(r_\alpha+m_{\alpha\beta}).
$$

The policies are native00, attention-cut10, MLP-cut01 and both11. In attention-cut arms, MLP15 must compute its own changed return. In both11, project that newly computed return. This is a factorial of sequential intervention policies, not a decomposition into two fixed vectors. All later computation is dense/native from each arm's own state. There is no MLP16 surrogate or correction in this panel.

Implement only the nominated layer/write operations, with model/projector/phase binding and preallocated scratch. Preserve normal kernel order when no act is supplied. Record pre-edit and consumed attention/MLP writes separately, actual receivers and layer15 output. A zero-policy path must reproduce ordinary native execution before interpreting edits. Complete token parents and deterministic stepping remain the only restoration mechanism.

Retain the layer16 incoming vector and its omission under frozen P, full first logits, current and common-next K/V, and eight-token own answers plus later Mira/Niko followups. Hold the common-next token to native's pending ID and record it explicitly. Keep isolated and conditional effects and the policy interaction at every consumer, with both full-vector and omission measurements; an improvement in the omission need not improve downstream fidelity.

The clock predicts no current-token K/V15 change: both nominated writes occur after that row was formed. Current K/V can first change at16. On the common next token, newly formed K/V16 should remain unchanged; layer16 attention then reads changed past memory and can first affect K/V17. Extend the retained observation range to include15and16so these boundaries are actually observed. These are integrity/localization diagnostics, not semantic outcome gates.

Keep the projector frozen and retain signed norms, dose, removed components, collateral, differing histories and all completed outcomes. A projected state is an artificial intervention with causally coherent continuation. P is not an identified semantic subspace, and attention-cut effects include MLP mediation; this panel cannot uniquely separate attention's direct and mediated consequences.

Use one finite mrun CUDA lease with one resident C model and shared parents, no training/retries and strict preflight. Begin with this one four-arm panel before broadening contexts. Compare resource use with the measured formation panel; fewer arms alone is not a speedup claim.

## Frozen implementation details

The four scientific policies have a separate ordinary-execution target control. Ordinary and explicit zero-policy paths retain full first/common-next logits, all14 ports at current layers15/16 and common-next layer16, current/common-next KV15–27, own eight-token answers and two followups per path. Compare all retained arrays and continuations for exact zero-policy parity. The common-next input must equal native's actual first pending ID.

The projector copies the same frozen B3 FP64 columns into owned host storage, with6144GPU bytes for a recoverable entry and43008host bytes for columns and materialization. It binds model/session/epoch/token position/edit layer separately from compile layer16. Project each actual writer using sequential FP64 dot products, then sequential three-term FP64 coordinate sums and one FP32 materialization; do not implement P by subtracting a rounded omission. A failed projection leaves an explicit partial token with its entry restored; ordinary layer recomputation or reset is required. The experiment does not silently retry.

The existing source conversation, four source controls, native source answer and one Mira calibration reconstruct the exact predecessor program/extension hashes. Source controls remain separate from the four scientific target policies. All edited suffixes and later tokens use the ordinary C FP32 equations. No target-dependent projector refresh or surrogate MLP16 is used.

Report absolute layer16 input/omission norms alongside whole-vector effects. The omission diagnostic subtracts the three FP64 projector terms sequentially from the input and then rounds to FP32, matching the earlier formation diagnostic. It is not the same finite law as subtracting FP32(Px).

The [historical read-only audit](../../artifacts/cmachine-holistic-review-20261106-1046/historical-writer-audit.json) found large individual-write norm removals under P3. This qualifies the supplied intervention's interpretation, not its validity. No omission-size, answer-quality or semantic-selectivity gate stops a mechanically valid branch.
