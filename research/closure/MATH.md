# State sufficiency is preservation of future distinctions

2026-11-06. Mathematical account for the [finite native-interface experiment](PLAN.md). The closure criterion and affine positive control already appear in the [future-qualified-state work](../../../obsidian/blog/2026-10-08-183803-from-time-formed-circuits-to-editable-executable-state.md#compression-must-preserve-allowed-futures). This experiment applies that question to an actual nonlinear model component.

## The criterion and its limits

Fix the model, numerical profile, phase/clock, admitted state domain D, input family U and consumer. Let S advance by F(S,u), with observable Y(S,u). For a proposed encoding c=C(S), exact laws G,H exist on the image of D precisely when

$$
C(S_1)=C(S_2)\Longrightarrow
\begin{cases}
C(F(S_1,u))=C(F(S_2,u)),\\
Y(S_1,u)=Y(S_2,u)
\end{cases}
$$

for every admitted pair and input. Necessity follows by applying the same G,H to the common code. For sufficiency define G(C(S),u)=C(F(S,u)) and H(C(S),u)=Y(S,u); the implication makes both independent of the representative S. Recursive equality follows by induction while the states and inputs remain in the admitted domain.

Every operand used by the replacement needs a declared source. Reconstructing every intermediate of the original implementation is unnecessary if the replacement computes the same permitted observations through another program.

This criterion distinguishes three questions. A fully retained native state already supports an executable decoder. A local approximate replacement inside that decoder can be autonomous while inaccurate. A reduced description earns sufficiency only for the consumers, transitions and inputs whose distinctions it preserves. One successful trajectory cannot establish invariance over all members of a code class.

## What a two-state witness proves

Suppose C(S0)=C(S1)=c but Y0 differs from Y1. Any deterministic code-only prediction h(c) must satisfy

$$
\|Y_1-Y_0\|\le\|Y_1-h(c)\|+\|h(c)-Y_0\|,
$$

$$
\max_i\|h(c)-Y_i\|\ge\tfrac12\|Y_1-Y_0\|.
$$

The lower bound applies to the declared pair and norm. It does not lower-bound error on naturally occurring chat histories or every alternative encoding. A raw-byte difference establishes finite executed-interface insufficiency when all identity/mechanics checks hold; its numerical and behavioral magnitude are separate observations.

## The local B3 interface

Let r be the post-attention receiver. The current compact MLP program retains the three executed FP32 QR coefficients c and native FP32 RMS scale rho. Its mutable input code is16bytes; static basis/images/QR, original weights and the full-state host are additional costs.

For source construction use the stored QR columns Q without assuming exact orthonormality. Put

$$
b=Q(Q^\top Q)^{-1}Q^\top r_0,\quad e=r_0-b,
$$

and choose v orthogonal to Q and e with ||v||=||e||. Then r(phi)=b+e cos(phi)+v sin(phi) has the same Qᵀr and norm in real arithmetic. FP32 materialization can change both executed coefficients and the GPU reduction. Therefore the actual encoder, including native rho, must produce identical16bytes before the pair is called an exact fiber witness. Geometry alone is not that proof.

The primary observable is M(r). In the full native suffix we add either measured write to the same actual residual skip r0. This isolates missing MLP-input information from the trivial direct addition of a different r. These supplied conditional-input interventions need not be naturally reachable conversations.

## What the complement supplies

For a linear coordinate map A and reconstruction B, define c=Ar and e=(I−BA)r. Then r=Bc+e exactly, regardless of idempotence of BA. With RMS weights Gamma=diag(gamma),

$$
\rho(r)=(\|r\|^2/d+\epsilon)^{-1/2},\quad
g=\rho W_g\Gamma(Bc+e),\quad u=\rho W_u\Gamma(Bc+e),
$$

$$
M(r)=W_d[\operatorname{SiLU}(g)\odot u].
$$

The basis code supplies the Bc terms; it does not supply Wg Gamma e or Wu Gamma e. Retaining e restores r. Retaining its two projected images is sufficient for this particular nonlinear invocation when combined with the other operands. Neither choice establishes a compact next-token state, and two intermediate-width images can cost more than the original hidden vector.

For one local residual write r+=r+M(r), the real-arithmetic coordinate update factors as

$$
c^+=c+A M(r),\qquad e^+=e+(I-BA)M(r).
$$

The omitted part is updated too. It cannot be treated as dead simply because the named coordinates are available. Finite execution additionally contains coefficient quantization, matrix/reduction ordering and residual-addition remainders. The experiment measures the original numerical program; these real-arithmetic identities do not erase those terms. This local layer update is also not the next token's receiver: attention and lower-layer cache dependencies must form that future receiver.

A differential diagnostic makes the same issue visible. In a smooth real-arithmetic neighborhood, exact factorization through a code requires DY[v]=0 whenever DC[v]=0. For C(r)=(Ar,rho(r)), tangent directions obey Av=0 and rᵀv=0. In such a direction

$$
Dg[v]=\rho W_g\Gamma v,\qquad Du[v]=\rho W_u\Gamma v,
$$

$$
DM[v]=W_d\left[(\operatorname{SiLU}'(g)\odot u)\odot Dg[v]
+\operatorname{SiLU}(g)\odot Du[v]\right].
$$

There is no algebraic reason for that expression to vanish merely because Av and rᵀv vanish. This is a derivation for the smooth equations, not an empirical Jacobian measurement or a derivative of the quantized16byte encoder.

## Why one attention read is not sufficient memory

Take two scalar-key memories. In world0 use (k1,k2)=(1,−1); in world1 use (−1,1). Both have (v1,v2)=(1,−1). At q=0, both numerator/partition summaries are (N,Z)=(0,2). At q=1,

$$
o_0(1)=\tanh(1),\qquad o_1(1)=-\tanh(1).
$$

Thus a summary evaluated at the old query does not determine the new read. The native-kernel toy embeds this example in128dimensions with the query scaled to cancel the usual sqrt(d) denominator.

For a fixed query, unchanged old rows/visibility and an appended (k,v), the update N+=N+exp(q·k/sqrt(d))v, Z+=Z+exp(q·k/sqrt(d)) is sufficient. Query changes expose the missing information. This separates append closure for a fixed read from closure over future query inputs.

One constructive assumption would restrict queries to a finite declared set, storing N(qj),Z(qj) for each. Another would supply an exact feature factorization of the admitted kernel, kappa(q,k)=phi(q)ᵀpsi(k). Then

$$
T=\sum_j v_j\psi(k_j)^\top,\quad z=\sum_j\psi(k_j),\qquad
o(q)=\frac{T\phi(q)}{z^\top\phi(q)},
$$

$$
T^+=T+v\psi(k)^\top,\qquad z^+=z+\psi(k).
$$

This is a closed memory update for that kernel/domain. Approximating the original kernel introduces a separate error and rollout question. It also assumes the rest of the machine can produce each new query/key/value from its own state. A finite query family or feature representation is a concrete simplifying assumption to test, rather than assuming that one current answer summarizes arbitrary context.

## How to interpret recurrence

Keep native own-history, candidate own-history, native rebuilt on candidate text and native one-step on candidate cache separate. Only the last isolates a local execution defect; same-text rebuild also includes accumulated candidate-state error. Once token histories diverge, later matching token IDs do not restore equal prefixes.

The local fiber result can identify an insufficient input interface immediately. Eight own-token steps then test whether the altered write persists into logits, memory and token decisions in the retained full-state host. They do not prove that the16byte code alone advances chat. The next reduced interface must carry or reconstruct whatever distinctions those consumers expose, and earn its own update law.

## A constructive assumption that makes nonlinear closure possible

There is a useful positive condition beyond a globally affine model. Consider a fixed residual block

$$
x^+=x+D f(Ax,Bx,\|x\|^2),
$$

where f can include RMS scaling, SiLU and elementwise products. Let L contain any additional permitted linear consumers. Suppose the complete family of linear reads A,B,Dᵀ,L has a common row space with orthonormal row basis R of rank r. This family and its fixed write map are supplied assumptions; finding them is a separate problem.

Write z=Rx and e=(I−RᵀR)x. The declared reads annihilate e, and the columns of D lie in range(Rᵀ). Consequently e+=e. Only its invariant energy kappa=||e||² enters the norm. We get the exact real-arithmetic closed program

$$
z^+=z+RD f(AR^\top z,BR^\top z,\|z\|^2+\kappa),\qquad
\kappa^+=\kappa,\qquad Lx=LR^\top z.
$$

This code retains r mutable coordinates plus one episode-dependent invariant scalar. Strict scalar-count reduction requires r+1<d unless kappa is fixed or derivable on the admitted domain. If D already has full row rank, the stacked family has rank d and this construction offers no reduction. No rank reduction of the actual Qwen block has been measured here.

Equivalently retain a=Ax,b=Bx,p=Dᵀx,s=||x||² and put w=f(a,b,s). Expanding the update gives

$$
a^+=a+ADw,\quad b^+=b+BDw,\quad p^+=p+D^\top Dw,
$$

$$
s^+=s+2p^\top w+w^\top D^\top Dw.
$$

The last equation uses old p. A row basis removes redundant linear registers. This derivation shows precisely why a discarded orientation may be safe: every admitted reader ignores it, no admitted write rotates it into visibility, and its norm contribution remains known. The B3 code did not establish those conditions. Adding a new reader or write map outside the declared family can invalidate the quotient.

This is a constructive mathematical result for the stated fixed block, not an executed neural compression or a full-transformer state law. Attention, changing support, different layer parameters and finite arithmetic require their own closure analysis. It does show that keeping the real nonlinear function is compatible with a simpler state program when its read/write geometry permits one.
