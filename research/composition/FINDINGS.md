# A frozen MLP program through its own generated state

2026-11-06. The C machine now runs a supplied compact MLP16 program repeatedly through its own generated tokens and KV state, alongside native execution, one-shot substitution, an arithmetic correction control and a subsequent zero-write control. Recall and multiplication survive every policy on these three questions. Complete prose survives both compact policies, despite large local write errors. The zero control shows why answer agreement alone cannot establish useful recovered computation.

The clearest computational result is a conditional boundary: under matching consumed token histories, the captured layer16 entry, receiver, normalized input and K/V16 remain byte-identical even when MLP16 and higher memory differ. Once a different token is consumed, those histories must be treated separately. This gives a concrete dependency rule for state composition.

## Policies and actual output

The [primary plan](PLAN.md) freezes three independent questions after the same reconstructed native source conversation and its own consumed answer. The276,536-byte B3 program comes from the earlier source/Mira calibration; no new target calibration occurs. The [posthoc zero plan](ZERO_PLAN.md) adds only three deletion paths after the first result exposes large local errors.

| Policy | At the last query token | At subsequent consumed tokens | Recall | 17 × 23 | Prose under48-token budget |
|---|---|---|---|---|---|
| Native | Dense | Dense | blue |391|45 tokens, EOS |
| Once | B3 PROJECT | Dense | blue |391|38 tokens, EOS |
| Recurring | B3 PROJECT | B3 PROJECT | blue |391|40 tokens, EOS |
| Corrected | B3 CORRECT | B3 CORRECT | blue |391|45 tokens, EOS; native token sequence |
| Zero control | Zero MLP16 write | Zero MLP16 write | blue |391|48 tokens, truncated |

Counts include EOS where it occurs. Every emitted token is consumed; the zero prose's48th token is also consumed. Each question starts independently from the same source-own-answer parent. These are fifteen paths over three questions in one supplied-fact context, not independent general-chat trials.

Native prose:

> 1. Context helps readers understand the meaning behind the words and the author's intentions.
> 2. Without context, a reader may interpret a message in a way that is not intended by the author, leading to misunderstandings.

Recurring compact prose:

> A reader needs context to interpret a message because the meaning of words can vary depending on the context in which they are used. Without context, a reader cannot understand the full meaning of a message.

Zero-control prose at the budget boundary:

> 1. Without context, a reader cannot understand the meaning behind the message, as the words used may have multiple interpretations.
> 2. Without context, a reader cannot grasp the tone or intent of the message, as the underlying emotions or intentions may

The zero output is right-censored by the declared budget. Its failure to finish within48tokens does not establish inability to finish or a general language-quality ranking. Both once and recurring complete, so completion does not establish a benefit from repeated application either. Correct color/integer answers survive deletion and therefore do not discriminate B3's content on these probes.

## The computation being reused

Let r be the actual post-attention receiver, B=[b0,b1,b2] the supplied basis, and c the FP64 QR solve executed as FP32 coefficients. Native RMS scale is retained:

$$
\rho(r)=\left(\|r\|^2/d+\epsilon\right)^{-1/2}.
$$

The program forms

$$
\widehat g=\rho\sum_jc_jG(\gamma\odot b_j),\qquad
\widehat u=\rho\sum_jc_jU(\gamma\odot b_j),
$$

$$
\widehat M(r)=D\left[\operatorname{SiLU}(\widehat g)\odot\widehat u\right].
$$

PROJECT omits e=r−Bc. CORRECT evaluates the two additional original-weight images ρG(γ⊙e) and ρU(γ⊙e), adds them before SiLU/product, and preserves the actual residual skip. In real arithmetic this restores the dense operands. The implemented FP32 ordering leaves a measured rounding floor.

Across all54 corrected transitions, same-input MLP-write relative differences remain **5.65×10⁻⁷–7.35×10⁻⁷**, and generated tokens match native on every question. CORRECT pays the two dense G/U calls; it supplies an arithmetic control rather than a savings result.

## Repeated use changes the continuation before inputs diverge

Once and recurring produce identical first complete states, verified by full first logits, complete valid-KV hashes and selected ports. The prose outputs first choose different tokens at emitted index9. Their actual consumed histories and captured layer16 operands remain equal through input steps0–9; the differing token is consumed at step10.

After the first step, once uses the actual dense MLP16 on those shared receivers. It therefore supplies a same-input reference for the recurring program on that common history:

| Task | Later shared-receiver transitions | Recurring write relative error versus once's dense write |
|---|---:|---:|
| Recall |2|100.51–100.76%|
| Arithmetic |4|99.95–102.84%|
| Prose |9|105.37–111.29%|

Large local error coexists with correct short answers and coherent prose. At14 of these15 transitions, the recurring write is farther from the same-input dense write than the hypothetical zero vector is. That comparison concerns local writes; the executed zero continuation supplies the separate behavioral control.

The first full-logit differences also retain a countertrend: B3 is closer than zero for recall and arithmetic, while zero is closer for prose. B3 first-write relative errors are67.36%,86.42% and108.98%, respectively. No single distance ranks all later behavior.

After token divergence, equal step indices represent different histories. The [sidecar](study/study.json) keeps those trajectory contrasts separate from comparisons on matching actual receivers. An omitted-receiver norm alone is not a missing-write estimate.

## A state factorization that the architecture supports

Split persistent state at this cut. Let P contain the KV history through layer16, and Q contain higher-layer KV and the output state. The lower computation can produce its new state and receiver from the previous lower state and supplied token:

$$
(P_{t+1},r_{16,t+1})=F_{\rm lower}(P_t,u_{t+1}),
$$

$$
Q_{t+1}=F_{\rm upper}\!\left(Q_t,\operatorname{FP32}[r_{16,t+1}+M_{16}^{\pi}(r_{16,t+1})]\right).
$$

The selected policy π changes MLP16 after K/V16 formation. With the same token history, it cannot alter the earlier causal computation. The panel directly verifies exactness at the captured layer16 ports and KV16; it does not capture every lower-layer array. Higher K/V can differ immediately. Generated-token selection feeds the upper result back into the next lower computation:

$$
u_{t+1}=\arg\max\operatorname{logits}(Q_t).
$$

That feedback explains the boundary. Matching tokens preserve shared lower dependencies; a changed consumed token invalidates their shared-history qualification. Later equality of individual token IDs does not restore equality of their prefix histories or caches.

This supplies a useful design rule for stacked state: preserve or share a state fragment according to its actual read dependencies and causal clock. A change elsewhere in the machine does not invalidate every fragment. A matching current token does not validate fragments built from a different history.

## What can be reused exactly

The primary panel contains204 normalized MLP input occurrences but123 distinct vectors:60 singleton groups,54 pairs and9 groups of four. There are **no repeated normalized inputs within any individual path**. The repetitions occur across counterfactual branches.

For the native G/U functions, identical normalized input bytes under fixed weights, shape and numerical profile are a sufficient direct operand-reuse condition. The data therefore identifies potential sharing across branches, not measured reuse within one chat. PROJECT additionally reads raw-receiver-derived coefficients and rho; a normalized-input hash alone is not its general cache key. No runtime memoizer or speedup is claimed.

A useful next runtime contract should distinguish those exact operand matches from a declared approximation. The present program is explicitly approximate whenever it drops a consequential complement; a finite coefficient solve is not evidence of future sufficiency.

## Execution and evidence

The primary job [job-86cce06d33b1](runs/job-86cce06d33b1/report.json) takes23.42scheduler seconds, with12.53seconds in the C driver, peak RSS3973.3MB and VRAM6378MB. The [zero job](runs/job-9337489b61af/report.json) takes18.77scheduler seconds, with10.51driver seconds, RSS4058.1MB and the same VRAM peak. Both use one resident original Qwen2.5-1.5B-Instruct C FP32 model, batch1,512context and one CPU thread. Original dense weights, full residuals/KV and most computation remain. Instrumentation and hashing are included in wall time.

The primary panel records204 selected-layer transitions,304 original G/U calls and204 down calls;52 PROJECT transitions avoid their two G/U calls. The54 CORRECT transitions pay those calls. The57 zero transitions execute no G/U or down call at that site while retaining normalization/subspace diagnostics. No calibration projections run in either job, and all original frozen bank bytes match before and after execution.

Verification authenticates both scheduler log/report joins,63/66 source files and1,498/470 raw records. The model-free importer preserves every frozen operand byte under ASan/UBSan; lifecycle checks and six token-authority/history tests pass. Astra independently checks all1,968 raw hashes/extents and critiques the scope. The additive sidecar snapshots all three of its analysis sources; original reports remain unchanged.

The narrow Saturn evidence sweeps cover150primary and57zero observation rows, with2and3applicable predicates, respectively. Run and verification succeed, with no watch fires or automatically ingested claims/addresses. This is stream coverage alongside the explicit numerical/custody audit, not semantic certification.

## Consequence for compact chat state

Repeated application through own tokens and memory is now executable and measured. It preserves these bounded behaviors while substantially changing local computation. The arithmetic control recovers native continuation, and the zero control exposes the limited discrimination of the short-answer probes.

The next scientific target is a representation that retains the consequential omitted computation at a consumer where its content matters. More basis columns, more answer matches or more memoized examples would not by themselves establish that representation. First qualify its read dependencies, compare against simple deletion, and follow its own state through a discriminating native consumer. General compact chat, reliable open-ended math/recall and a measured serving advantage remain open.
