# Retained parent identity for writer interventions

The actual-writer panel is valid: each arm explicitly restored one retained parent and verified its complete KV prefix hash. The reusable API previously bound session, epoch, position and layer, which did not itself distinguish a different history rebuilt to the same position within one epoch.

Add one 64-bit lineage identity to each token-boundary state. Session creation and reset obtain a fresh identity. Begin-token and layer stepping retain that parent identity. Successful token completion obtains a fresh identity for the new boundary. Checkpoints and stack frames save it; restoration restores it. Counter exhaustion refuses before GPU or live transition mutation and never wraps to an old identity.

`QwenWriter` binds the retained parent lineage as well as model, session, epoch, position and edit layer. A mismatching parent is refused before reading or writing GPU state. Restoring the actual parent permits reuse even when the legacy bookmark serial changed. Independently recomputing byte-identical history gets a new identity: this is object lineage, not a content hash or a claim that the two histories differ numerically.

The change adds eight bytes per checkpoint and 512 bytes across the fixed 64-frame marker array: 9,216 bytes in the tested ABI versus the historical 8,704. No KV copies, model operations or GPU allocations are added. `QwenSubspace` intentionally remains reusable across contexts; `QwenPlane` retains its separate exact live-input/read guards.

The model-free hostile check restores an earlier parent, constructs a different history at the same position and epoch, then verifies writer refusal with unchanged session metadata, residual and current KV row bytes. It separately verifies successful reuse after actual parent rollback. The finite Beast mechanics lane runs the pretrained lifecycle and one historical/current ordinary-output parity case; it adds no new scientific writer sample and does not rerun the four-arm panel.

## Completed verification

[`job-11cdb86ad29b`](results/job-11cdb86ad29b/report.json) succeeds on its first attempt: 36.04 scheduler seconds, 33.89 worker seconds, peak RSS 4,121.1 MB and VRAM 6,378 MB. The actual pretrained lifecycle uses synthetic token inputs; a separate ordinary source case compares against the preserved prior binary. All 151,936 first logits are byte-identical, generated IDs match and checkpoint replay is exact. The receipt explicitly reports the writer-parent lineage checks. Counter-exhaustion refusal is source-reviewed; the receipt's exercised overflow check concerns stack capacity.

The [offline verifier](results/job-11cdb86ad29b/verification.json) authenticates 87 frozen sources, the scheduler log/report join, all three raw input/readout files, two complete readouts and one public lifecycle receipt. Astra independently checks the source/raw hashes, input/output joins and byte equality. Local ASan/UBSan and 13 Python checks pass. The scientific actual-writer report remains unchanged; this is engineering validation only. The tested binary is retained in the job directory; the active installed development release is unchanged.

The standard isolated [evidence sweep](index/lineage-20261106/evidence-run.json) and [verification](index/lineage-20261106/evidence-verify.json) complete with one source-bound observation, two applicable stream predicates and no findings. This is stream coverage for an engineering receipt, not semantic certification or an additional scientific sample.
