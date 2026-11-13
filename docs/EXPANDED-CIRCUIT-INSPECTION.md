# Decision regimes, selective memory, GELU and spatial phases

These additions extend the [native circuit inspector](CIRCUIT-MATH.md). They use original equations, optional native observations and declared consumers. No new learned model, fitted predictor or sparse representation is required. Qualification is model-free and uses supplied tiny Qwen/ISA organisms; no new pretrained model run has occurred.

## Decision envelopes and equation-selected probes

For a fixed final state `r+alpha*m`, a bias-free vocabulary head after RMS has

\[
\lambda_v(\alpha)=(a_v+\alpha b_v)/\sqrt{A+2B\alpha+C\alpha^2},
\quad a_v=W_v\Gamma r,\ b_v=W_v\Gamma m.
\]

The common positive denominator preserves greedy ranking. `cm_qwen_final_readout_lines` hydrates every original tied-head row and returns temporary full-vocabulary coefficients for the final captured MLP receiver/write. `cm_qwen_write_readout_lines` serializes these coefficients with model, port, clock and lineage metadata. It exports two derived scalars per vocabulary row, not a parameter bank. These coefficients are optional inspection data; ordinary execution does not compute them.

Inside an admitted mrun job, the native inspector can write a new coefficient file:

```sh
make circuit-cuda
./build/tcm-circuit /path/to/qwen.tcm --target 123 --readout-lines /fresh/lines.json --tokens 1 2 3
```

Then, with no framework/model imports:

```sh
python3 tools/decision_envelope.py /fresh/lines.json --lower 0 --upper 1
```

The retained-coefficient demonstration works locally without any model:

```sh
python3 tools/decision_envelope.py tests/fixtures/translation-envelope.json
```

The Python tool constructs the sorted-slope upper hull with exact rational arithmetic over supplied coefficients. It keeps identical-line ties, point-only ties, endpoint ties, all positive-length winning intervals and a representable interior probe when one exists. A region narrower than binary64 spacing is retained rather than silently erased. Lowest token ID wins exact ties.

Full-vocabulary status requires all declared vocabulary IDs; an input containing retained hull lines is explicitly `provided-lines-only`. The tool does not certify that omitted rows cannot win. Bias after normalization, nonlinear suffixes, state-dependent edit directions and sampled decoding have different contracts. Native rounding can shift analytical boundaries.

The retained translation fixture reproduces middle token 650's interval `(324474813684256/4315771582720603,207832459388728/939467298396787)` and selects `1/8`. Quarter-spaced probes miss it. This checks retained coefficient arithmetic; no native pretrained probe at the new dose was executed.

## Selective memory: signed write → retention → read

`circuit_expansions.h` provides a generic per-channel diagonal SSM kernel, cross-cut factors and source-state paths. For fixed producers,

\[
H_t=R_t\odot H_{t-1}+\Delta_t B_tu_t,\qquad y_t=C_t^TH_t,
\]
\[
K_{t,s}=\Delta_s\sum_n C_{t,n}B_{s,n}\prod_{j=s+1}^tR_{j,n}.
\]

For a cut after `q` rows, `cm_ssm_cut` returns `L[T-q,N]` and `E[N,q]` with `K_future,past=L E`. Products are evaluated without dividing by retention, so zero/underflowed retention is supported. Rank is at most `N` for this strictly cross-cut conditional operator. This is not a rank bound for the complete nonlinear model, the full triangular kernel, or a count of semantic circuits. A nonzero initial state has an additional propagated boundary term; this kernel maps new inputs, and the supplied R3 scan starts at zero.

The ISA adapter captures the actual `u`, timestep, shared B/C selection, per-channel retention and injection, recurrent state, read, outer gate, post normalization, GELU operands and final write. It observes one selected layer over all 64 bit positions. The scan clock resets per instruction. Each channel has four scan coordinates and two preconvolution tail coordinates; retaining the scan alone is not a complete continuation interface.

`cm_isa_scan_path` uses actual observed injections; `cm_isa_scan_kernel` uses ideal `dt*B*u`. Their numerical differences remain distinct. Skip, outer gate, output projection, residual, post MLP and later layers are outside this scan-read kernel. Signed paths are not normalized attention probabilities.

`cm_ssm_finite` exposes the complete ordered state difference:

\[
\Delta H_t=R_{1,t}\Delta H_{t-1}+\Delta R_t H_{0,t-1}+\Delta I_t+\Delta e_t.
\]

This retains changed retention, injection and native arithmetic instead of attributing the full response to a frozen kernel. Producer arrays remain available for further decomposition; the generic kernel does not differentiate their formation.

### Native ISA workflow

```sh
make expansion-mechanics isa-circuit
./build/tcm-isa-circuit /path/to/r3.tcm --bytes 4889d8 --layer 2 --channel 0 --source 8 --receiver 31 --cut 31 --dose 0.25 --register 0
```

Real-model invocations belong inside admitted mrun jobs. This CLI's declared architectural root has zero registers, flags, RIP and instruction count. The C API accepts any valid caller-owned architectural root and keeps its exact input bytes/architecture in the transient capture. The original mmap bundle remains the weight owner.

`cm_isa_capture` runs ordinary `isa_begin` and all 64 `isa_advance` steps, leaving the original heads awaiting commit. Captures and queries do not commit an instruction. `cm_isa_state_effect` regenerates an exact row boundary, edits one scan coordinate, executes the complete remaining native program and reports a declared register-logit difference. It restores the original checkpoint on return. This is a finite native intervention, not a frozen-producer derivative; later producers and nonlinear consumers execute normally. Earlier-bit targets provide causal controls.

Observer pointers are session-local and are never serialized into `IsaState` or `IsaCheckpoint`. Existing numerical state layouts stay unchanged; C callers must rebuild for the extended session struct. Captures refuse replacement policies/existing observers, validate complete port coverage and bind queries to a hash of the completed native state. Failures restore the full parent state.

## GELU: explicit affine channel, retained nonlinear response

The original ISA path uses exact-erf GELU. The new scalar kernels provide value, first/second derivatives, secants and an observed-minus-ideal remainder. The exact decomposition is

\[
\operatorname{GELU}(a)=a/2+(a/2)\operatorname{erf}(a/\sqrt2),
\]
\[
f(n)=\tfrac12W_2W_1n+\tfrac12W_2b_1+b_2+
\tfrac12W_2[(W_1n+b_1)\odot\operatorname{erf}((W_1n+b_1)/\sqrt2)].
\]

`cm_isa_gelu_write` evaluates both terms through the original CPU-mapped parameters and retains the full native write remainder. Biases are retained. The affine term is affine in the actual normalized port; it is not affine in the raw residual before LayerNorm. Negating that residual does not negate a biased preactivation. Exact-erf and tanh-GELU require different derivative profiles; the native operator is not replaced.

For the last post block only, `cm_isa_register_account` traces the write into an actual register logit, keeping copy prior, head bias, residual carry, affine GELU part, nonlinear GELU part, MLP arithmetic remainder and readout arithmetic remainder. Earlier blocks have a nonlinear suffix and cannot use this direct-head account. The native state-effect helper can still measure their full downstream effects.

## Consumer-weighted diagnostics, without fitting or dropping variables

`cm_consumer_metric` takes declared consumer adjoint rows `J`, an omission/error vector `e` and nonnegative consumer weights, returning every signed `J e` and `sum w*(J e)^2`. It does not materialize `J^T W J` or select a compression bank. `cm_qwen_omission` connects this diagnostic to the existing original-model adjoints at input, output, receiver, attention, product, gate and up ports.

The tiny fixture contrasts a large direction orthogonal to the selected consumer (norm about 14.48, tangent effect zero) with a small aligned direction (norm 0.01, tangent effect about 0.03775). This is a local geometric example, not evidence that the large direction is dispensable. Finite nonlinear effects, other consumers and future state remain separate measurements. Nothing is fitted and no original variable is discarded.

## Polyphase audit utility

`cm_unpatchify` applies four separate phase scales/biases and restores the fine spatial grid. `cm_dft2` and `cm_polyphase_combine` verify

\[
\widehat X_{k,l}=\sum_{a,b=0}^1 e^{-\pi i(ka/H+lb/W)}
\widehat Y^{ab}_{k\bmod H,l\bmod W}.
\]

`cm_polyphase_audit` compares that prediction with the explicit spatial transform, and measures the loss caused by replacing denormalized phase planes with their mean. It keeps the four phase-specific statistics. On the supplied one-cell checkerboard `(1,-1,-1,1)`, phase averaging erases all spatial energy (4); the transform identity closes exactly in the recorded check.

The utility processes one latent channel at a time with that channel's four phase statistics. It evaluates FP64 mathematics on supplied arrays; it does not reproduce a VAE's BF16 denormalization/promotion profile.

This is a deterministic C utility for supplied arrays, not a FLUX denoiser or VAE implementation. No offloaded full latent archive was recovered and no image-quality result is claimed. MDB/Saturn can pass hydrated latent views to this utility later.

## Verification and mathematical provenance

See [the qualification record](../qualification/expanded-inspection-20261112/README.md) for exact logs, source identities, native-state parity, the nearby/long-gap dose rows and limitations. Independent tests compare C outputs with NumPy, finite differences, direct state iteration and exhaustive competition over supplied line sets. No scientific quality threshold stops collection.

The source chapters are [Decision Envelopes](../../obsidian/model-math/15-decision-envelopes.md), [Selective State Spaces](../../obsidian/model-math/20-selective-state-spaces.md), [GELU and MLPs](../../obsidian/model-math/19-gelu-and-mlps.md), [Approximate Operators](../../obsidian/model-math/16-approximate-operators.md) and [Scheduler and Pixels](../../obsidian/model-math/22-scheduler-and-pixels.md). These sibling-workspace links resolve in the canonical checkout. The additions extend the supplied equations into executable inspection. They do not promote report arithmetic, toy execution and pretrained semantic evidence into the same claim.
