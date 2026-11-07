# Independent mathematical review of the finite state-closure assay

2026-11-06. Astra advisor; offline analysis only. Native job `job-7b8a5933f1dd`. This note is additive; executed sources and raw results were not changed.

The three selected pairs are exact collisions of the declared 16-byte executed interface and substantial disagreements of the native MLP write. This establishes that these three B3 coefficients plus the native RMS scale do not determine the local native MLP write on this intervention domain. The observed token agreement does not undo that counterexample: logits and complete cached state differ while these short greedy continuations remain the same.

## Independent integrity checks

I recomputed all 81 payload-source SHA-256 values and all 966 raw-file sizes and SHA-256 values against the saved manifests. All 958 raw FP32 array files are finite. The report hash is `b693fc8b12dbf0e79f93a9337136517aae1d020f267467a74357ce84ed58208b` and matches the saved verification. There are 24 branches and 136 token-consumption forwards, including each branch's final supplied question token and consumed EOS where reached.

The actual driver model digest, report model content digest, and Saturn request model digest agree at `8b0589b6f367c42e457d53f22af32882f4f0d807803481cdd65c675e33f12fbc`. The reported weights identity is `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`; these are distinct identity fields, not interchangeable hashes. The receipt request agrees with the binary identity. Geometry bytes match the receipt's geometry hash. The before/after bank fingerprints agree with the command-bound bank fingerprint. This independently verifies the recorded identity joins, not an additional local reload of the remote model.

The authenticated driver calls the unchanged runtime's `subspace_coordinates` on every encoded source and candidate, comparing its executed coefficients and RMS scale bytewise to the selector's code. Selection examines code equality before any candidate MLP evaluation. For each source, the selected index is the first actual code collision with different receiver bytes in the bound ordering. Supplied geometry source bytes and selected candidate bytes match the actual standalone calculation inputs.

All three rebuilt native receivers and writes match the supplied source receiver and standalone source write bytewise. All 105 saved native/sham counterpart arrays agree bytewise, as do their full-logit and complete-prefix cache hashes at all 17 paired steps. All eight policies have the same first native residual receiver for each task. This validates the fixed-skip comparison and rules out changing the skip as the explanation for the first effect.

## Exact pairs and unavoidable local prediction error

All selected angles are −1 radian. Recall selects direction 3/index 6; arithmetic and prose select direction 0/index 0. These are appreciable rotations, not tiny quantization perturbations: receiver displacement is 28.05%, 42.31%, and 55.85% of the respective source receiver L2 norm. The ideal spherical geometry motivates the search; actual encoder-byte equality establishes the finite claim.

| Task | Exact source-code collisions / 64 | Native write separation, L2 | Any code-only predictor: worst endpoint error at least | Separation / source write norm |
|---|---:|---:|---:|---:|
| Recall | 14 | 10.657487370 | 5.328743685 | 0.617483 |
| Arithmetic | 38 | 17.813451206 | 8.906725603 | 0.841302 |
| Prose | 44 | 20.883055111 | 10.441527556 | 1.186575 |

For each pair, `source-code == alternate-code` and the two PROJECT writes are byte identical. If a deterministic predictor produces the one value \(\widehat M(C)\), the triangle inequality gives

\[
\max\{\|\widehat M-M(r_0)\|_2,\|\widehat M-M(r_1)\|_2\}
\ge \tfrac12\|M(r_1)-M(r_0)\|_2.
\]

This lower bound applies to **any** deterministic predictor from this code and the same bound static program inputs, not merely the tested PROJECT implementation. It is an absolute L2 bound for the finite executed write. The last table column uses a common source denominator and must not be reinterpreted as a lower bound relative to whichever endpoint incurs the error. Three source neighborhoods do not estimate a population failure rate.

The paid complement correction approximates the **alternate fiber write**, with relative L2 errors `6.272664e-7`, `6.762866e-7`, and `7.013911e-7` and absolute errors around `1.2–1.5e-5`. It does not restore the source write. It supplies omitted gate/up information through dense work; its success identifies useful sufficient operands for this invocation, not a compact recursively updated state.

## Native consumer and the cache clock

With the source skip held fixed, first full-vocabulary logit differences are:

| Task | L2 difference | Difference / native logit norm | Maximum coordinate difference |
|---|---:|---:|---:|
| Recall | 170.335080 | 0.089538 | 2.881063 |
| Arithmetic | 203.918107 | 0.137173 | 2.702738 |
| Prose | 238.578468 | 0.171325 | 2.998755 |

These relative norms describe logit vectors, not probability errors or percentages of semantic change. Every native/fiber paired step has different full-logit and complete-prefix-cache hashes.

At the intervention token, current KV16 is byte identical and current KV17 differs, with concatenated K/V L2 distances `7.728313`, `12.217387`, and `13.656691`. This is the expected clock: the changed MLP16 write changes the input to layer17, which writes KV17.

On **every later same-history native/fiber token**, newly written KV16 **and KV17** are byte identical. The old differing KV17 remains in the cache; layer17 reads it only after forming that token's Q/K/V. Its resulting residual difference can first reach newly written KV18. KV18 was not captured, so that last statement is an architectural eligibility argument, not a measured KV18 difference. Complete-prefix cache hashes remain different throughout. Equal new KV17 therefore does not imply that the effect disappeared.

## What the continuation does and does not establish

Native and one-time fiber branches emit and consume identical tokens: recall `blue` then EOS, arithmetic `391` then EOS, and the same eight prose tokens. Prose ends by budget, not EOS. The fiber-repair branch has the same respective token sequences. This shows robustness of these greedy decisions over the measured horizon despite different writes, logits, and memory. It does not establish equality for future questions, longer continuations, or sampling.

All policies, including recurring zero-write, answer the two short questions correctly. Those answer scores are consequently poor evidence for the compact interface's adequacy. Prose is more discriminating: compact-once and compact-recurring diverge from native immediately and share their eight emitted tokens; zero diverges at emitted index 2; corrected-recurring matches native for all eight. None of these prose paths reaches EOS within the budget. Once/recurring token equality does not make their hidden computations identical or prove recurring approximation harmless.

The attention calibration also behaves as declared. The uniform query reads 0 in both two-row worlds and 1/3 after append. The distinguishing query reads approximately ±tanh(1); after appending the common zero-key/+1-value row its two reads are approximately `0.819938779` and `−0.330481797`. These match the known numerator/partition calculation to FP32 error. The calibration is supplied wiring, not learned semantic-state discovery.

## Relation to the earlier positive quotient result

The [October 8 executable-state post](../../../obsidian/blog/2026-10-08-183803-from-time-formed-circuits-to-editable-executable-state.md) already gives the transition-compatible quotient criterion and the supplied affine organism with five variable scalars rather than nine, matching 3,840/3,840 declared consumer outputs. This assay does not rediscover that theorem. It challenges a concrete nonlinear native interface and finds exactly which candidate equivalence fails: equal executed B3 code need not mean equal MLP write.

The next mathematical issue is whether a richer retained interface can update its missing consumer-relevant information from its **own** preceding state and admitted input. Dense complement correction answers how to compute the missing operands when the full receiver is supplied. It does not yet give that closed update law. No experiment here proves that every useful encoding must be large, or that a task-conditioned state cannot be smaller than the complete dense host state.
