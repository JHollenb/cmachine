# Independent Astra review: frozen span response

Advisor: existing user-requested Astra agent `/root/astra_c_design`, November6,2026. Astra reviewed the proposed boundary and implementation before submission, then independently audited the sealed result and saved-only omission analysis. No additional real-model run was requested for the review.

## Construction and execution review

The implementation freezes a source-only response bank before opening question token files, uses the correct post-RoPE keys and associated values, retains cluster counts inside one global normalizer and separates construction inputs from the bank/query-only predictor. Shared native question parents localize the selected read. Exact34 is an actual packed-kernel numerical control. Own-token consumption and final-history/clock labels avoid assigning same-input meaning to arbitrary independent continuations.

Astra found no execution blocker. The scope is a single last-question-token layer17 read, with dense complement and later original-memory access explicitly paid. It is not continual compact chat, full cache replacement, universal query transfer, a latency result or a semantic-necessity test.

## Independent primary-result audit

Astra verified report/source/raw/scheduler bindings and predecessor identity. An independent plain-Python Ward reconstruction exactly reproduced all memberships and stored FP32 count/key/value bytes for all three banks. The original correction-span K/V bytes join the preceding native attention experiment. All ten own token histories, EOS consumption and exact34/native port, probability, selected-KV, full-logit and complete-cache-hash comparisons agree.

Independent response recomputation confirms the predicted numerator/denominator account and ideal Jensen ordering. Eight improves initial context and logits over one and deletion for both later questions. The isolated span numerator remains about47–49% wrong in mean per-head relative L2, despite much smaller full-logit differences. All answers match, including deletion, and the literal `Niko` diagnostic is a formatting mismatch against the actual `Niko's` in every arm.

The review specifically retains a continuation countertrend: recall's one-centroid post-EOS logit difference is slightly larger than deletion's. First-read improvement does not imply monotonic downstream improvement. The two prompts and correlated heads do not support a general success rate.

## Independent omission-audit review

The [saved-only audit](runs/job-524febce43e5/omission-audit.json) binds the verified report, verification file, audit source and ten raw dependencies. Its [source snapshot](runs/job-524febce43e5/omission-audit-source.py) preserves the exact analyzer. Astra verified those bindings and independently recomputed dispersion and dependence using an alternative stable form:

\[
D_j=\left(\sum_i e^{s_i-g}-n_j e^{\bar s_j-g}\right)\bar v_j,
\qquad C_j=\sum_i e^{s_i-g}(v_i-\bar v_j).
\]

This avoids the analyzer's centered-exponential products. The signs in E_N=N_exact−N_stored=D+C+R, per-head exact-span-Z normalization, term norms and cross terms agree. Eight recall gives D9.33651481, C8.54612419, omission12.04625076 and cross−15.09458856. Eight prose gives D8.62118088, C6.86429808, omission11.09257681 and cross+1.60191470. Independent vector closure is at most5.41e-16 and denominator closure at most3.44e-16.

Two qualifications from the review are incorporated into the findings and blog. C measures finite exponential-score/value dependence, rather than merely linear key–value covariance. R measures centroid storage and primary CPU arithmetic; native CUDA discrepancies belong to the separate executed-consumer account. These value-unit norms are gauge-independent diagnostics, not semantic responsibility shares, full-attention weights or importance percentages.

The useful inference is that both lost dispersion and value dependence remain substantial and change alignment with the query. A denominator-only repair would not generally recover the missing numerator. Any next moment-based representation must be constructed from memory alone, price its extra bytes and work, and then be tested on future queries and their native consumers. No new terminal claim is requested.
