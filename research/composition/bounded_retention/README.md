# Bounded C attention memory and delayed recall

This experimental implementation replaces linear retained rows with fixed anchors, a recent-row ring and an optional reservoir of actual evicted K/V pairs. It reuses the original BF16 weight residence and FP32 tile consumer from [weight storage](../weight_tiles/FINDINGS.md). All layers encode new candidate rows and consume them immediately. The logical clock remains absolute; eviction does not rebase RoPE.

The [architecture blog post](../../../../obsidian/blog/2026-11-06-171900-what-bounded-memory-changes-in-the-c-state-machine.md) explains the changed state transition, allocation and recovery contracts, and why the failed delayed questions matter for compact sufficient chat.

[ring.h](ring.h) owns support, sampling, transactions and two nested marker snapshots. A third snapshot is reserved for token abort. Each snapshot includes encoded data, absolute addresses, sample population/RNG and full logits; native session metadata remains bound by model, owner, epoch and marker. The [small runtime transform](runtime_patch.py) validates recovery before session metadata changes. [ring_cuda.cu](ring_cuda.cu) gathers sorted live support, reads paired keys/values and adds reservoir mass through the existing softmax arithmetic. No fact parser or target answer enters this read.

The [prospective plan](PLAN.md) declares native, full-byte, recent64, anchors16+recent64 and anchors16+recent64+reservoir32. One model load serves five sequential own-token sessions. Each includes an early question, a correction, neutral delay and four independent late children. Actual tokenized fact positions must lie outside anchors/recent support at the delayed root. The native lane uses the same weight tiling; answers are compared with declared facts separately from native fidelity.

The reservoir keeps M paired rows sampled uniformly from n evictions. With m=min(n,M), its attention approximation is

\[
\widehat Z(q)=\sum_{i\in A\cup R}e^{s_i(q)}+\frac{n}{m}\sum_{j\in S}e^{s_j(q)},\qquad
\widehat N(q)=\sum_{i\in A\cup R}e^{s_i(q)}v_i+\frac{n}{m}\sum_{j\in S}e^{s_j(q)}v_j.
\]

Empty samples contribute zero. The read is their ratio. This preserves an exponential mixture, key/value pairing and represented mass, while allowing rare rows to be missed. For an independently fixed query and population, numerator and denominator estimates are unbiased with respect to reservoir sampling. Their ratio is not unbiased; adaptive model queries/populations require separate analysis. The executed FP32 log-weight/bias path and ideal real weighting differ slightly; the saved analyzer records that remainder as a diagnostic.

Serving rows cost 28×528×C bytes for support capacity C. Two marker snapshots and one abort snapshot add three more arenas and three logit buffers. Score/decode/gather scratch scales with C. These explicit recovery copies are deliberately counted; no journal or historical token array serves attention. Experiment traces are audit artifacts and remain outside the model's read interface.

Run `run.py stage`, local `python -m unittest test_mechanics -v` inside the staged payload, then `run.py submit PAYLOAD`, `collect JOB`, and `verify RESULT_DIRECTORY`. The [mechanics fixture](mechanics.c) exercises actual transformed-session guards with a synthetic CPU cache consumer, including accepted sample replacement followed by a last-layer failure, nested sibling overwrite/replay, fixed allocation under doubled logical capacity and invalid-marker refusals. It qualifies lifecycle mechanics, not CUDA model numerics. The [saved analyzer](analysis.py) replays actual own-token histories, sampling, selected-layer encoding, root support, allocation and readouts without loading a model.

The [completed findings](FINDINGS.md) preserve bounded execution and the failed delayed consumer: all bounded arms answer zero of four late questions correctly. Native/full-byte preserve three factual answers but give 40 rather than 46 for the sum. The reservoir state costs 9,279,620 tracked GPU bytes including all recovery copies. [Astra's audit](REVIEW.md), the [report](runs/job-1d8106dee187/report.json), [strict verification](runs/job-1d8106dee187/verification.json) and [evidence sweep](index/job-1d8106dee187/evidence-verify.json) bind the result. The [v2 saved-data factorization](runs/job-1d8106dee187/diagnostic-v2.json) separates omission, multiplicity, changed carriers and changed queries without another model run. General compact sufficient chat, unrestricted exact recall, forward elimination and serving speedup remain unestablished.
