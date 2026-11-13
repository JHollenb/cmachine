# Circuit inspection: original mathematics, live operands, declared consumers

The optional C circuit inspector expands native computations without fitting an auxiliary model. It implements the six families in [the proposal](CIRCUIT-MATH-PROPOSAL.md). The mathematical kernels are framework-free. A Qwen adapter connects them to actual native ports and a first-readout vocabulary consumer. Pretrained CUDA qualification remains pending; the completed verification uses the existing tiny native C executor with a CPU GPU stub.

The [expanded inspection tools](EXPANDED-CIRCUIT-INSPECTION.md) add exact decision envelopes/probe selection, native ISA selective-memory and GELU accounts, consumer-weighted omission diagnostics and a standalone polyphase audit.

## Run the completed local demonstration

```sh
make circuit-mechanics
```

This compiles with AddressSanitizer, UndefinedBehaviorSanitizer, strict warnings and floating-point contraction disabled. It runs a three-layer, width-eight, supplied-parameter fixture through the ordinary Qwen C equations. It checks passive capture parity, full-consumer derivatives, historical source key/value edits, finite changes, frequency accounting, state-interface witnesses and stale-state refusal. These are mechanics results, not language or semantic scores.

## Native inspector when CUDA execution is available

```sh
make circuit-cuda
./build/tcm-circuit /path/to/original-qwen.tcm \
  --target 123 --contrast 456 --layer 16 --head 6 --source 3 --neuron 20 \
  --frequency --tokens 1 2 3 4 5
```

IDs are exact vocabulary IDs, not words. All options precede `--tokens`; the last supplied token is the inspected consumed token. The tool emits newline-delimited JSON: a selected occurrence report, an empty-root replay recipe with the complete consumed IDs, and optionally a bounded source-frequency report. The recipe still requires the original exact bundle and execution profile. It does not serialize a resumable interior state or independently authenticate its own report.

Real-model execution of this command belongs inside an admitted workspace mrun job; the example specifies the executable invocation, not an admission bypass.

The selected source can be historical or current. The consumer is the original token logit or token-minus-contrast margin after the last layer and final RMS normalization. The inspector reads the original tied vocabulary rows, not a fitted lens. It differentiates every remaining native operation for this token, including current Q/K/V and residual paths. Historical K/V is held fixed; the report's historical routing partial is a sensitivity at that particular cache occurrence, not a derivative through its earlier formation.

`make circuit-cuda` and actual pretrained execution have not been qualified on the MBP. The frontend and C modules compile in the local model-free lane. The ordinary `tcm-cuda` serving path does not invoke the inspector.

## Library workflow

Include `qwen_circuit.h`; link `qwen_circuit.c`, `circuit_math.c` and the ordinary native executor/backend. Initialize owned objects to zero and release the adjoint before its capture, then the session and original model. Captures require exclusive synchronous access and stay valid only while their session remains at the captured result. They borrow the session/model; they do not prolong either object's lifetime.

```c
CmQwenCapture capture = {0};
CmQwenAdjoint consumer = {0};
if (cm_qwen_capture_token(&capture, &session, input_token) == 0 &&
    cm_qwen_consumer(&consumer, &capture, target_token, contrast_token) == 0) {
    cm_qwen_report(stdout, &capture, &consumer, layer, head, source, neuron);
}
cm_qwen_adjoint_free(&consumer);
cm_qwen_capture_free(&capture);
```

Capture consumes the supplied token exactly once, using `qwen_begin_token`, ordinary `qwen_advance_layer`, then `qwen_finish_token`. It is not an edit. Existing observers, compact cache backends, lexical programs, compiled Q/K/V programs and non-native weight residence profiles refuse before execution. Failure during capture aborts the open token through the existing native parent contract. No replacement path is silently analyzed as the original equations.

Two appended observer ports expose the actual packed attention probabilities and the executed post-SwiGLU product. Existing port numbers remain unchanged. An explicitly attached observer with `observe_layer=-1` now observes all layers at the selected token position. No observer means no added transfer or synchronization.

## Finite MLP response and exact secants

For endpoint gate/up operands `g0,u0` and `g1,u1`, set `si=SiLU(gi)`. The product change is

\[
\Delta f=u_0\Delta s+s_0\Delta u+\Delta s\Delta u+\Delta e.
\]

Here `e` is executed product minus its FP64 ideal product. `cm_neuron_delta` exposes all four terms. `cm_qwen_mlp_finite` projects the three ideal terms through the original down map, retains the full changed native write, and reports its arithmetic remainder. Output buffers retain every hidden coordinate; an optional consumer vector provides signed scalar projections. Summaries also expose normalization, projection and product-secant discrepancies. These are instrument diagnostics, not semantic failure gates.

The equally exact symmetric product identity yields an endpoint-dependent linear action:

\[
T_{01}=D\left[\operatorname{diag}(\bar u\odot c)G+
\operatorname{diag}(\bar s)U\right],\qquad
c_i=\frac{\operatorname{SiLU}(g_{1i})-\operatorname{SiLU}(g_{0i})}{g_{1i}-g_{0i}}.
\]

Coincident endpoints use the derivative; nearby endpoints use `expm1` to reduce sigmoid cancellation. `cm_qwen_mlp_secant_action` streams original rows and applies this factored operator to a supplied normalized-port direction. It never stores the full matrix. Native gate/up projections differ from FP64 products by measured arithmetic terms; these are not silently assigned to a circuit.

For RMS, write `qi=epsilon+||ri||²/d` and `rhoi=1/sqrt(qi)`. Rationalization gives

\[
\Delta\rho=-\frac{\Delta q}{\sqrt{q_0}\sqrt{q_1}(\sqrt{q_0}+\sqrt{q_1})},\quad
\Delta q=(r_1-r_0)^T(r_1+r_0)/d,
\]
\[
\Delta n=\Gamma[\rho_1\Delta r+\Delta\rho\,r_0].
\]

`cm_rms_finite` retains the complete norm and alignment. Composing this action with the MLP secant explains a finite residual edit. It uses both endpoints and is not a forecast of an unknown future. Chaining reports across diverged token histories requires the two actual programs; it cannot treat different greedy choices as a smooth path.

## Reusable writer/reader relationships

For a selected down-column `v`, reader row `w`, full receiver `r` and RMS gain `gamma`,

\[
w^TJ_N(r)v=\rho w^T\Gamma v-
\frac{\rho^3}{d}(w^T\Gamma r)(r^Tv).
\]

`cm_writer_reader` returns the static pairing, radial correction and total. `cm_qwen_writer_reader` hydrates only the requested original down-column, gate/up/Q/K/V reader row and gain. The caller declares the receiving state. This is a local coupling under the specified direct residual perturbation; it does not assert that an arbitrary distant writer reaches that reader unchanged through intervening operations.

`cm_qwen_parameter` gives transient complete selected rows or columns. Reports name the original checkpoint tensor and selector. No parameter matrix is exported or permanently copied. Reverse inspection streams parameter rows, which can be expensive; this is an explicit understanding tool, not a serving optimization.

`cm_qwen_static_link` computes the selected static pairing without a session or prompt. `cm_qwen_read_geometry` returns original gate/up norms, alignment and the two possible eigenvalues of the symmetric quadratic read form `(g u^T+u g^T)/2`. Neither function executes a model token. `cm_neuron_even_odd` also exposes the exact bias-free split `even=gu/2`, `odd=gu*tanh(g/2)/2`; the nonlinear part is retained, not truncated. These are algebraic coordinates, not automatically meaningful semantic names.

## Attention and the downstream consumer

The native reverse traversal computes final-norm/head, all residual and MLP paths, O projection, grouped-query attention, inverse rotary pullbacks, Q/K/V maps and input RMS. Native FP32 operands anchor an FP64 smooth-equation linearization. It is not a derivative of rounded machine code, argmax or future-token generation. Captured probability/return arithmetic and FD discrepancies qualify this lane separately from native execution parity.

For one head,

\[
dA=\sum_jp_jdv_j+\sum_jp_j(v_j-A)d\ell_j,\qquad
\partial M/\partial\ell_j=p_j\langle\lambda,v_j-A\rangle.
\]

`cm_attention_vjp` returns positioned Q/K, value and logit adjoints. The Qwen adapter sums shared K/V contributions across query heads. Its `routing` array exposes signed output sensitivity for every visible source; `attention` retains the channel-sensitive downstream consumer. The selected neuron report includes actual gate/up/product and the down-column's product partial. Activity times gradient is not asserted to be a finite additive attribution.

`cm_attention_finite` evaluates arbitrary same-support logit/value changes through stable reweighting. `cm_qwen_attention_finite` instead compares two actual native captures and exposes routing `delta(P)V0`, payload `P0 delta(V)`, interaction and native remainder. Captures must have the same model and position; their distinct parents and inputs are explicit.

The generic reweighting routine treats a supplied zero probability as zero mass; it cannot recover a probability lost to native underflow. Paired native captures retain newly nonzero probabilities and their complete finite effects.

## Curvature, interactions and frequency

`cm_neuron_curvature` computes a bilinear Hessian action at a normalized port from the original gate/up rows. `cm_rms_second` retains the full radial second-order normalization terms, permitting the complete chain rule instead of pretending the normalized read plane covers the raw residual. `cm_attention_query_second` exposes the centered score-squared/value relationship at fixed K/V. No full Hessian is allocated.

`cm_coalitions` computes/inverts the exact Boolean-lattice Mobius transform for at most twelve edit bits. A two-edit interaction equals an integrated mixed Hessian only for compatible fixed additive edit directions and a smooth rectangle. Ordered overwrite Acts do not automatically meet that condition. The function cannot establish that callers supplied common-parent experiments; callers retain that custody.

`cm_dft` uses an unnormalized forward transform and `1/N` inverse on an explicitly declared axis. `cm_circular_convolution` exposes product-frequency pair terms. `cm_shift_commutator` measures failure to commute with cyclic token shift. These direct inspection transforms are bounded to 4,096 positions; no fast FFT library or learned spectral model is required.

`cm_qwen_source_frequency` provides a consumer-connected view of one receiver: with `zj=dot(lambda,Vj)`, coefficient `k` is `conj(DFT(p))[k]*DFT(z)[k]/N`. All real coefficients sum to the ideal scalar read; native-return-minus-ideal is retained. The CLI prints sixteen bins plus the complete unlisted complex sum. This is source-axis correlation, not the two-frequency operator of a full square attention window. A tokenwise MLP needs a multi-occurrence token trajectory for a nontrivial token-axis spectrum. Neither frequency mixing nor phase alone proves semantics.

## State-interface witnesses

`cm_interface_compare` compares caller-supplied canonical typed argument and result bytes. Scope must bind exact model, numerical profile, native program, consumer, horizon, and any abstract intervention class. Different scopes/arguments are kept separate. Equal arguments with a different result produce a witness, not an averaged score. A successful pair agrees on that pair; it does not certify general sufficiency. Scope and byte canonicalization are caller responsibilities; the library does not infer them from tensors.

This is the bridge to semantic investigation: choose an independently meaningful task specification, nominate original variables, test counterfactuals, and refine the interface when a missing dependency is exposed. Native connectivity, local sensitivity, finite necessity and semantic meaning remain distinct evidence.

## Results and remaining integration

The sanitizer-backed tiny fixture preserves full native logits, K/V, clocks and token ownership. Its full-input consumer derivative has maximum absolute central-difference discrepancy `2.5066e-4`; shared historical K/V source derivatives have discrepancy `1.3364e-5`. The finite MLP report exposes opposing gate/up effects and interaction, with the numerical remainder explicitly retained. Source-frequency accounting reproduces the consumer's native scalar after adding a `-1.9787e-9` native arithmetic remainder.

Independent NumPy tests exercise RMS adjoints/finite changes/curvature, finite MLP secants, full attention Q/K/V VJP, query curvature, large positive/negative score offsets and the Fourier commutator identity. Exact data and execution logs are in [the qualification record](../qualification/circuit-math-20261112/README.md).

The C report and parameter references are available for MDB integration. Direct MDB graph selection/IPC/UI wiring, portable replay manifests for arbitrary edited sessions, full multi-token adjoints, recurrent-Mamba reverse inspection and pretrained CUDA validation remain future work. This implementation does not replace them with a semantic-certification claim.

The mathematical provenance is the November 12 workspace write-ups: *Expanding native computation without dropping variables*, *Causal attention and native MLP read geometry*, *When circuits mix frequencies and how finite edits propagate*, and *Composing native mathematics into semantic circuit proofs*. The proposal links those original posts and the preexisting cmachine derivations.
