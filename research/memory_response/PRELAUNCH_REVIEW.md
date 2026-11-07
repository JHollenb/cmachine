# Astra prelaunch review

November 6, 2026. Read-only review before any model execution of this experiment.

Astra reviewed the frozen scientific plan, hypothesis post, launcher, worker, resource plan, cache kernels, driver and audit observer. No launch blocker remained after the recent-window start, metadata-only mark restoration, zero-sized session allocation and post-floor parent hash fixes.

The final driver review confirmed that partial abort distinguishes published layers 0–17 from failed layer 18, restores ring bytes and metadata, and reproduces the six-token suffix. Both question forks restore the same post-floor parent and history length. Audit K/V are the candidate's actual post-RoPE rows. The query is captured at the last supplied question token. GQA and compact active-window probability layouts are consistent. Audit buffers are not passed to serving readers.

Two-session control allocation adds 31,145,476 bytes above the measured prior one-session layout. Capture size is bounded before model loading. Top-four still decodes all value rows and scans all probabilities; its nonzero value MAC count is a nominal upper bound. Its custom reduction changes finite arithmetic in addition to selected support.

If `codec_audit_exact` is false for a byte policy, that row's reconstructed-byte diagnostic must be labelled instrument-broken or nonactual. This does not invalidate or suppress the required primary native and candidate outcomes.

This review is a code/protocol review, not evidence that the behavioral hypothesis succeeds. The submitted source snapshot preserves this note and the prospective post.
