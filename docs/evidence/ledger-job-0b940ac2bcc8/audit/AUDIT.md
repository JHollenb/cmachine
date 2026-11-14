# Independent audit: per-token ledger, job-0b940ac2bcc8

Auditor: claude-opus-5-5. Date 2026-11-13. Read-only on the source tree. The only writes were to this directory, plus the gitignored `build/ledger-mechanics` binary that `make ledger-mechanics` produced.
Labels: **M** = I recomputed or ran it. **A** = asserted from reading code.

Raw inputs: `runs.jsonl`, `ledger-C0-on.f32`, `ledger-FORCE_F1-on.f32`, `states.f32`, `g3_static.f32`, and G3 records 0 and 63. I copied them read-only from `beast:/mnt/usb-storage/experiment-results/cmachine-ledger/full1-job-0b940ac2bcc8/` into the session scratchpad, not into the repo. The evidence `results.json`, `samples.csv` and `prompts.txt` are byte-identical to the Beast copies (sha256, M).

## Scripts (all take paths as arguments; NumPy and stdlib only)

| script | what it checks |
|---|---|
| `custody.py` | manifest sha256 vs the staged payload and vs the current tree, for full1, feas1 and prof1 |
| `timing_recompute.py` | medians, p10/p90, overheads, paired per-prompt deltas, arm order, and baseline ratios, from raw per-step `ms` |
| `ledger_read_recompute.py` | the "What the ledger shows" numbers, from the raw C0/on rows (own layout code) |
| `census_identity.py` | the header identity even + odd = SiLU(g)·u on raw rows, and whether "odd share" is a share of product energy |
| `g3_recheck.py` | an independent parser and float64 recompute of G3 records, plus the check `raw + mlp_write == output` (bitwise), which the worker does not make |
| `part4_arithmetic.py` | ‖n‖ of the dumped states, and the Cauchy–Schwarz half-width implied by the reported tails |

## Verdicts

1. **The numbers match (M).** Every timing median, p10/p90, overhead, kernel and readback figure in LEDGER.md recomputes exactly from the raw per-step samples. The same holds for every G2, G3 and G4 statistic, the prof1 and feas1 tables, and the Part 4 table. There is one sign slip (see must-fix). The off arms reproduce the speedups job, with ratios 0.9992–1.0039. Stronger evidence: the final-step logit banks of C0, F1, G1, G8, G64 and bf16_G1 are **bitwise identical** (sha256) to the HEAD-binary banks of job-1d03b313def2, for the off arms and also for the on arms.
2. **Custody is clean (M).** All 97 manifest files match both the staged payload and the current tree. The launcher-style `git diff HEAD` sha256 recomputes to the recorded `444b6db6…`. The worker sha `392c4551…` matches `results.json` meta and the current `worker.py`. The source manifest embedded in `results.json` equals the staged `sources.json`. Only `docs/` changed after the job, as the builder said. `build_ledger` `binary_sha256` and the mrun-side wall time and cgroup RSS could not be verified, because mrun was unreachable from here.
3. **The gates test what they claim, with two exceptions.**
   - G1 compares genuinely separate runs (separate reset, prefill and decode; banks written from different runs) (A), and the cross-binary match above confirms it (M).
   - G2 covers all 31 named fields (M). It has no coded pass threshold (A).
   - G3 recomputes with independent float64 arithmetic from observer ports, and my own recheck reproduces it (M). Exceptions:
     - **top1/top2:** the "truth" values are copied from the same `S.top` that fills the ledger row, so they are not checked.
     - **pfirst/pmax/argmax:** these read the same native-probability buffer, which is correct by the header's definition but only checks the reduction.
   - G4 is a real semantic test: the native margin from the logits is compared with the ledger's split of retained writes.
   - The tolerances are reasonable for FP32. Lean vs float64 truth agrees at FP32 storage rounding (dots ≤ 6e-8 rel).
4. **The ledger math matches the header in both executors (A, with M identity checks).**
   - Fast fused add: the dot uses x before the add, and the retained w is the value that was added. The last layer goes through `finish`, and prefill rows without a readout go through `lg_tail`.
   - softmax fields, rho exports, gate thresholds, even/odd, c, rho_F and the closure all match the header.
   - The identity even2 + odd2 + 2·even_odd = prod2 holds to 3e-7 of the cancelling scale (M).
   - I found no definition that lean and fast share but that contradicts the header.
5. **Off path and timing are fair (A, M).**
   - The native kernels do not appear in the diff; off-path changes are host-side branches only.
   - Arm order rotates per prompt, every step is bracketed by syncs, and the readback is inside the timed region for "on".
   - Paired per-prompt deltas agree with the pooled medians: G1 0.346–0.359 ms, C0 1.055–1.062 ms.
6. **The Part 4 negative is sound (A, M).**
   - The bound is implemented as stated, with consistent norms: rows of W·diag(γ) against n = N/γ.
   - The arithmetic: ‖n‖ = 39.19 for every state, so the half-width ‖n_tail‖·‖row_tail‖ is about 75 at k = 512 and about 120 at k = 32 (medians). Logit gaps here are at most 15. A full-vocabulary candidate set is therefore a property of the data, not a bug.
7. **"What the ledger shows" is mostly accurate and properly scoped.** Every number recomputes. Two interpretations are wrong (below). All 16 prompts agree in sign on the main facts: Σattention > 0 and last MLP < 0.

## Must-fix (each is a wrong claim; the fixes are wording or a small extra check)

- **`docs/LEDGER.md:175`: "Odd (nonlinear) share of the product energy".** odd2/(even2+odd2) is not a share of the product energy. In every layer, even and odd are strongly anti-correlated: the median of even_odd/√(even2·odd2) ranges from −0.56 (layer 0) to −1.00 (layers 1–3, 5). The median of (even2+odd2)/prod2 ranges from 2 to 1.25e5. The ~0.5 at layers 1–5 shows the cancellation in hard-off neurons (even ≈ −odd), not a nonlinearity holding half the energy (M, `census_identity.py`). `worker.py:461` (`odd_share_mean`) has the same naming problem.
- **`docs/LEDGER.md:172`: "Layers 1–3 are mostly hard-off".** Hard-off fractions are 73% at layer 1, 46% at layer 2 and 23% at layer 3. Layer 5 (2307 hard-off, 26%) and layer 4 (388) are not mentioned (M).
- **`docs/LEDGER.md:132` (G3): "top1/top2 … pfirst and pmax are exact".** For top1/top2 this cannot fail: `worker.py:503` copies `r["top1"]`/`r["top2"]` from the same `S.top` that writes the ledger row. top2 is never checked independently anywhere. Part 4's float64 top1 matching 1024/1024 does cover top1. Fix: say so, or dump the logits row and recompute the top-2.
- **`docs/LEDGER.md:150`: "The closure is exact up to float32 storage of the 57 terms".** The ledger split agrees with a float64 split to ≤ 7.8e-7 (G3 `margin_sum`). The native margin differs from the float64 margin by up to 9.3e-6 (G3 `margin`). So the remainder mostly measures the native FP32 logit error, not the term storage (M). Example: record 0 has native 0.1624241, float64 0.1624309, ledger sum 0.1624313.
- **`docs/LEDGER.md:103`: "First-step and final-step logits … for C0, F1, G1, G8, G64 and bf16_G1".** G8 and G64 have only final-step banks (`ledger_run.c:532`; the results list has no `logits_first_G8/G64`) (M).
- **`docs/LEDGER.md:81`, prof1 "everything skipped" row, B=64 "−0.007".** The value is +0.007 (12.857 skipped vs 12.850 off) (M).

## Minor

- G2, G3 and G4 have no predeclared pass threshold in `worker.py`; only G1 has `pass`. "All four gates pass" is a judgment after the fact. Either state the criterion or say "within FP32 expectations".
- In `determinism()` the key (prompt, step, seq) collapses prefill rows to the last one. 1040/1040 is 1024 decode rows plus 16 last-prefill rows; the other 309 prefill rows are not compared.
- The `rows_disagree` 1e-2 smoke check covers only prompt 0 (4 decode rows plus the last prefill row). "Never fired" is true but narrow.
- Fast-vs-truth and G2 are limited by real differences between the two executors' activations (about 1e-6 to 1e-5). They cannot isolate the FP32 per-element census and entropy change, so the claim "unchanged at about 3e-6" is a floor, not a measurement of that change.
- `docs/LEDGER.md:30–32`, "peak RSS (cgroup)" 3890/3991/3785 MB: the in-job sampler gives tree RSS of 4069/3998/3955 MB. These are unverified here because mrun was unreachable.
- Section 7's "matches its negative margin contribution" links cos(r, w) to the c-direction without a test. All per-layer means are over 1024 rows from 16 autocorrelated prompts, and the mean margin is pulled up by prompt 2 (code, 7.46).
- The lean BF16-residency margin path (`weight_mode==2`) and the tiled profile were not exercised by this job.

## Headline numbers (M, from raw per-step samples)

| arm | off ms | on ms | on − off | kernel / readback |
|---|---|---|---|---|
| G1 FP32 graph B=1 | 9.9772 | 10.3290 | **+0.3518 ms (+3.53%)** | 0.3437 / 0.0081 |
| G64 | 13.0389 | 13.7008 | +0.6619 ms (+5.08%) | 0.6022 / 0.0597 |
| bf16_G1 | 5.3619 | 5.7279 | +0.3660 ms (+6.83%) | 0.3607 / 0.0053 |
| C0 lean | 10.5370 | 11.5942 | **+1.0572 ms (+10.03%)** | — |

**Off vs speedups baseline: ratios 0.9992–1.0039, and the final logits are bitwise identical to the HEAD binary.**

`make ledger-mechanics` passed (bit-identical on/off, 26 lifecycle/refusal cases) (M). It tests `qwen.c` with the CPU stub, not the CUDA kernels.
