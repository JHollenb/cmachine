# Astra: prospective gate/up restoration review

2026-11-06. Read-only mathematical/protocol review before this factorial's native outcomes. The reviewed `PLAN.md` adopts a common parent, frozen source64 factors and histories, native future transitions, and a separate native same-ID history reference. No model call was launched by this review. This note does not change historical evidence or certify future implementation.

## Why this is the next informative cut

The source-weight basis captured more linear operator energy and improved separate selected gate/up errors relative to random64, but failed to improve pooled native-write reconstruction over deletion. The relevant missing object is therefore the joint nonlinear operand computation, not simply more scalar operator energy. This experiment can localize which exact operand restoration changes the write and which changes the later consumer. It cannot by itself discover a cheap replacement for the restored information.

Use the exact source64 factor bytes from job `job-51df6b792d28`, with no recompile or target-dependent basis choice. Original native operands are intentionally paid diagnostic controls, computed on the actual receiver. Their presence must never be described as a source-only cheap inference path.

## Exact cut selection and prediction clock

The fixed historical token positions, in first / after-answer0 / after-followup / final order, are:

| Case | Positions |
|---|---|
| Recall | 55, 57, 91, 93 |
| Arithmetic | 43, 47, 78, 82 |
| Prose | 43, 59, 79, 95 |

The retrospective arithmetic answer is `[18,20,22,151645]`, or `3,5,7,EOS`. The first wrong digit is answer index1, token20 (`5`), instead of token21 (`6`). Its producer is the readout after consuming token18 (`3`) at **position79**, `case1-source64-s079`. Intervene while that input token is being processed at MLP16. A cut while consuming erroneous token20 at position80 occurs one decision too late to test the first wrong digit.

The task margin for that immediate readout is `logit[21]−logit[20]`. This truth comes from 391−24=367, independently of native own-text behavior. Native-own had answered 401 then 157 and cannot serve as arithmetic truth. The digit selection is retrospective; keep it separate from the twelve fixed clocks.

Up to two subsequent historical IDs are literal recorded events. At the digit cut these are20 and22. Forcing them after a hypothetical repair to6 deliberately follows the old history. At other cuts the next events may be a subsequent template/user event. At final cuts no next IDs exist. Do not fabricate extra tokens to fill a two-step horizon.

## The write factorial and what it does not assert

Let s=φ(g), ŝ=φ(ĝ), Δs=s−ŝ and Δu=u−û, with φ=SiLU. Define writes

\[
Y_{00}=D[\widehat s\odot\widehat u],\quad
Y_{10}=D[s\odot\widehat u],\quad
Y_{01}=D[\widehat s\odot u],\quad
Y_{11}=D[s\odot u].
\]

The baseline-conditional restorations and interaction are

\[
Y_{10}-Y_{00}=D[\Delta s\odot\widehat u],\quad
Y_{01}-Y_{00}=D[\widehat s\odot\Delta u],\quad
I_Y=Y_{11}-Y_{10}-Y_{01}+Y_{00}=D[\Delta s\odot\Delta u].
\]

The gate restoration conditioned on native up is Y11−Y01; the up restoration conditioned on native gate is Y11−Y10. Both differ from their baseline-conditional counterparts by I_Y. These are exact real-arithmetic identities. Retain the executed FP32 reconstruction discrepancy rather than asserting byte equality across differently associated computations.

For a shared residual skip/cache/clock parent, write Z_ab=F_parent(Y_ab) for a native suffix output. Its finite factorial interaction I_Z=Z11−Z10−Z01+Z00 must be measured. In general it is neither D[Δs⊙Δu] nor a fixed linear image of I_Y. A large conditional restoration or interaction is not a fraction of causal responsibility. The zero-write branch helps distinguish restoration from removal; its performance is an observation, not a hard gate.

## Current write versus carried state

At each measured horizon let C=Z00, L=Z11 on the candidate parent, and H be native execution on the identical historical IDs. Then

\[
C-H=(C-L)+(L-H),
\]

\[
\|C-H\|^2=\|C-L\|^2+\|L-H\|^2+2\langle C-L,L-H\rangle.
\]

Use the same denominator, for example ||H||, for all normalized terms, and record the signed cross term divided by ||H||². Treat a zero denominator explicitly if it occurs. Do not add normalized magnitudes or call the larger term the majority of the total error when cancellation is substantial.

At the immediate cut C−L isolates changing this write with the candidate parent fixed. L−H measures the remaining difference between candidate and native history construction. Call it cache-only only if the other retained quantities have been verified equal. At later horizons C−L is the transported consequence of the one-time write change; it is no longer a local current-write vector.

Every branch uses all-native transitions for the next up to two frozen historical IDs. H does likewise. Consequently **only immediate Y00 is expected to reproduce the old recurring-source64 trajectory**. Later Y00 follows a new policy, and a mismatch with later historical source64 is not a mechanics failure. This is a teacher-forced common future assay, not an autonomous repaired conversation. A changed immediate greedy token does not establish a recovered answer or coherent own-token continuation.

## Minimal integrity checks and claim limits

The declared immediate historical Y00 logit/cache replay, receiver/norm/write joins, exact factor identity, and native H replay are sufficient meaningful mechanics controls when paired with correct common-parent reconstruction. The native Y11 write should also join the existing paid native write on that receiver or an unchanged native MLP invocation. Shape, finiteness, token/cut clock, valid-cache range and restored-state independence must be preserved. No scalar reconstruction-quality threshold belongs in this discovery protocol.

Historical boundary parity is evidence about this implementation. It does not make these C in-memory checkpoints durable Saturn StateCuts. The process receipt and internal C reconstruction have different scopes. Failure of one optional branch should preserve completed branches unless the shared parent or instrument is invalid.

The two-cut mechanics lane and thirteen-cut panel are bounded and appropriate. No broader sweep or additional quality gates are needed. Report actual physical prefix reconstruction, diagnostics, suffix and future-token work. The two-step horizon is intentionally narrow. If current native operand restoration does little, the next question concerns carried state; if it helps, the next question is how to form that operand information cheaply without importing the paid native control. Neither outcome alone establishes compact state closure or autonomous chat fidelity.
