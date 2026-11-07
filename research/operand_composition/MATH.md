# A connected writer–reader pair

Indices are zero based. For a matched consumed-token history, all computations
below MLP16 are unchanged. Let x_t denote MLP16's actual post-attention,
post-RMSNorm/gamma input. Native and compiled writers are

\[
w_N(x)=D[\operatorname{SiLU}(Gx)\odot Ux],\qquad
w_C(x)=D[\operatorname{SiLU}(L_gEx)\odot L_uEx].
\]

The imported coupled64 E/Lg/Lu are the actual independently fitted matrices;
gamma has already been applied to x and is not folded into E a second time.
Both writers retain the original full residual skip and dense down projection.

The resulting residual is normalized for attention17. Write this receiver as
y_t^W. The compiled addressing program forms

\[
z_t=By_t^W,\quad
q_t=R_t(Fz_t+b_q),\quad
k_t=R_t(Kz_t+b_k),\quad
v_t=V y_t^W+b_v.
\]

Native addressing instead uses the original Q/K maps. R_t is the original
half-split rotary operation at the absolute token clock. Each arm appends its
actual K/V before reading all available paired FP32 rows. Every layer's cache
starts empty; no native historical carriers are transplanted into serving.

At a matched history, MLP16's normalized input and same-token layer16 K/V are
identical across all four arms by induction over tokens/layers. MLP16's write,
attention17's normalized receiver and its V are reader-independent: 00=01 and
10=11 for these ports. These are controls of the stated computation graph. The
query/key program receives the actual altered receiver produced by the writer.
Its historical K/V also come from that combination's earlier computations.

For any exported vector Y at a matched consumed history, define

\[
A=Y_{10}-Y_{00},\quad B=Y_{01}-Y_{00},\quad
I=Y_{11}-Y_{10}-Y_{01}+Y_{00}.
\]

Then Y11−Y00=A+B+I exactly in real arithmetic. Squared norms include all signed
cross terms; none is a percentage of semantic responsibility. This recurring
policy interaction includes the history of those policies' writes. Across
different self-selected token histories, the same subtraction additionally
includes history changes. It is not an isolated single-operation derivative.

Six own-token squares (24 trajectories) are followed by three policy replays per
square on the recorded joint arm's IDs. The joint arm is the existing fourth
corner and is not rerun. Own execution seals before diagnostic replay. Full
logits at first/second prefill and first/second answer end retain discrepancies
on matching histories and those caused by diverging histories separately.

Nominal learned-linear MACs per completed token:

| Arm | MACs | Logical linear calls |
|---|---:|---:|
|00|1,543,569,408|197|
|10|1,517,289,472|198|
|01|1,541,242,880|198|
|11|1,514,962,944|199|

Joint reduction is 28,606,464 MACs (1.8533% of native linear work). It excludes
attention over history, nonlinear operations, tracing, synchronization, hashes
and setup; runtime measurements are instrumented observations, not a latency
comparison. All original model parameters remain resident. Imported numeric
factor storage is 6,684,672 bytes; separate mutable coordinate scratch is 768
bytes. Full-sized memory remains allocated at all layers for every arm.
