# The missing information is a future write coupling

Let native state x advance by T_u and emit O_u under an admitted future input u. An encoding C is sufficient on a declared domain when functions Tbar_u and Obar_u exist such that

$$
C(T_u(x))=\overline T_u(C(x)),\qquad O_u(x)=\overline O_u(C(x)).
$$

Equal codes must therefore remain equal after every admitted update and yield equal declared observations. Matching one current read does not suffice. By induction, the two equalities preserve all finite admitted futures. Conversely, two native states with the same code and different encoded successors or outputs disprove that exact interface on the stated domain. This is a statement about a particular encoding, not about the impossibility of other representations.

## A constructive nonlinear quotient

Consider

$$
x^+=x+Dw,\qquad w=\operatorname{SiLU}(G\,\rho(x)x)\odot U\,\rho(x)x,\qquad
\rho(x)=(\|x\|^2/d+\epsilon)^{-1/2}.
$$

The supplied organism has identity normalization weights; with a learned diagonal scale, absorb it into G/U in the real-arithmetic derivation and retain its finite operation order separately. Let R have orthonormal rows. Require its row space to contain the rows of G, U and the declared linear reader L, and every column of D. Write

$$
z=Rx,\quad e=(I-R^TR)x,\quad\kappa=\|e\|^2.
$$

Then Ge=Ue=Le=0 and (I−RᵀR)D=0. Consequently e and κ are constant, while

$$
\rho=(\{\|z\|^2+\kappa\}/d+\epsilon)^{-1/2},\quad
w=\operatorname{SiLU}(\rho GR^Tz)\odot(\rho UR^Tz),
$$

$$
z^+=z+RDw,\qquad\kappa^+=\kappa,\qquad Lx=LR^Tz.
$$

Nonlinearity remains explicit. Reduction comes from a closed operand interface and compiled small affine maps, without requiring the complete computation to be polynomial.

## Break only the write condition

Let V=(I−RᵀR)D be nonzero while G/U/L still ignore e. The initial read can remain identical, yet

$$
e^+=e+Vw,\qquad
\kappa^+=\kappa+2e^TVw+w^TV^TVw.
$$

The missing signed coupling is eᵀVw. Equal omitted energy does not determine it. For one escaping direction v and scalar amplitude a, two omitted states +v/||v|| and −v/||v|| produce opposite cross terms under the same first write. Their next normalization can differ, exposing the discarded orientation to later G/U and declared output.

The C assay uses R selecting coordinates0/1, e=±coordinate2, and an extra D[2,0]=.25. The first output update can still agree even though encoded energy already differs; the next normalization and subsequent output can distinguish the worlds. The repair expands R to include coordinate2. This is a human-supplied counterexample and repair, not automatic state discovery.

An alternative interface retains directional write couplings instead of explicitly storing each coordinate. For fixed v, a=eᵀv obeys a+=a+||v||²w and κ+=κ+2aw+||v||²w², using the old a. Multiple directions replace a by Vᵀe. Whether that is smaller depends on write rank and the other consumers. This alternative is algebraic here, not an additional executed implementation.

## Finite arithmetic is part of the state contract

Real-arithmetic dependence on ||e||² does not prove an arbitrary FP32 reduction depends only on an independently rounded κ. The original C RMS kernel has a specific256-lane strided/tree reduction and rsqrtf. The compact assay constructs the same eight-slot representative from retained coordinates and sqrt(κ), then calls that unchanged kernel. Its supplied omitted values ±1 or0 have matching squared reduction leaves. This qualifies a narrow finite construction; arbitrary rotated complements may require extra numerical information or a declared approximation.

Full-width normalization scratch, its constant gamma vector, original nonlinear kernels, static maps, transfers and recording remain paid. Nominal affine work changes from96 to24 or36 MACs per update. The tiny capture-heavy assay does not measure serving speedup.

## Test the strong construction on original Qwen weights

If R must contain all columns of D, rank(R)≥rank(D). A fixed nonsingular256×256 minor proves rank(D)≥256, excluding a common space of rank below256 for this sufficient construction. It does not prove full1536 rank, characterize the span of reachable SwiGLU products, or rule out nonlinear/restricted/approximate representations.

Original BF16 values widen exactly to binary32 and are dyadic rationals. Map each rational to the prime field modulo p=2³¹−1, where powers of two are invertible. For a binary32 normal with exponent bits E and fraction F,

$$
x=(-1)^s(2^{23}+F)2^{E-150}.
$$

Subnormals use F2⁻¹⁴⁹. Gaussian elimination over this field returns a determinant certificate. A nonzero image of the determinant implies the rational determinant was nonzero, hence a real-rank lower bound. Zero modulo this one prime would be inconclusive about nonsingularity over the reals. Independent verification uses Python float.as_integer_ratio and arbitrary-precision integer elimination, separately from the producer's bit mapping and NumPyint64 elimination.

## Consequence for the broader design

Tighten the admitted future domain when seeking an exact smaller state: specified readers, operations, positions and numerical profiles. Loosen exact native equality only when intentionally studying an approximate program, and measure its own baseline, signed response, accumulated history and actual consumer. Keep those choices explicit. Exact global invisibility is sufficient for exact emulation; it is not necessary for useful bounded approximation. Recent selected-only image results illustrate the latter while paying their full K/V scan and original learned suffix.
