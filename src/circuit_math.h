#ifndef TCM_CIRCUIT_MATH_H
#define TCM_CIRCUIT_MATH_H
/* Optional FP64 inspection lane. All arrays are borrowed host arrays; outputs
 * must not overlap inputs unless explicitly stated. No model/state mutation,
 * fitting, parameter persistence, or derivatives of rounded arithmetic.
 * Returns 0 on success, -1 for invalid/nonfinite geometry or arithmetic.
 * Outputs are unspecified after an error; native model/state is never touched. */
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct { double re,im; } CmComplex;
typedef struct { double gate,up,interaction,remainder,total,secant; } CmNeuronDelta;
typedef struct { double gate_norm,up_norm,dot,eigen_plus,eigen_minus; } CmReadGeometry;
int cm_read_geometry(size_t,const double *,const double *,CmReadGeometry *);
int cm_neuron_even_odd(double,double,double *,double *);
double cm_silu(double);
double cm_silu_first(double);
double cm_silu_second(double);
double cm_silu_secant(double,double);
int cm_neuron_delta(double,double,double,double,double,double,CmNeuronDelta *);
int cm_neuron_curvature(size_t,const double *,const double *,const double *,const double *,const double *,double *);
/* RMS differential and adjoint; exact finite endpoint action; no dense J. */
int cm_rms_jvp(size_t,const double *,const double *,double,const double *,double *);
int cm_rms_vjp(size_t,const double *,const double *,double,const double *,double *);
int cm_rms_finite(size_t,const double *,const double *,const double *,double,double *);
int cm_rms_second(size_t,const double *,const double *,double,const double *,const double *,double *);
/* Selected scalar entry/action of W J_RMS D. reader/writer are full residual
 * vectors, not truncated coordinates. static_part is W gamma D (without rho). */
int cm_writer_reader(size_t,const double *,const double *,const double *,const double *,double,double *,double *,double *);
/* One actual head, row-major K,V [sources,channels], p sums to one. Gradients
 * are at positioned Q/K; caller composes RoPE/projections and GQA summation. */
int cm_attention_vjp(size_t,size_t,const double *,const double *,const double *,const double *,const double *,const double *,double *,double *,double *,double *);
int cm_attention_query_second(size_t,size_t,const double *,const double *,const double *,const double *,const double *,double *);
/* Finite same-support change from baseline p,V and changed logits/values.
 * Stable even for very large finite logit changes; computes normalized ratios. */
int cm_attention_finite(size_t,size_t,const double *,const double *,const double *,const double *,double *);
/* Unnormalized forward / 1/N inverse. N<=4096: inspection, not serving FFT. */
int cm_dft(size_t,const CmComplex *,CmComplex *,int);
int cm_circular_convolution(size_t,const CmComplex *,const CmComplex *,CmComplex *);
/* Scalar token operator [N,N]; sum |J S-S J|^2, S cyclic backward shift. */
int cm_shift_commutator(size_t,const double *,double *);
/* In-place Boolean-lattice Mobius transform; inverse=1 reconstructs outcomes.
 * Supplied subset outcomes must share parent/order; no causal claim inferred. */
int cm_coalitions(unsigned,double *,int);
typedef struct {
    const char *scope; /* exact model/profile/program/consumer/horizon identity */
    const char *occurrence;
    const void *arguments; size_t argument_bytes; /* canonical typed bytes */
    const void *result; size_t result_bytes;
} CmInterfaceRow;
typedef enum { CM_INTERFACE_OTHER_SCOPE,CM_INTERFACE_OTHER_ARGUMENTS,
               CM_INTERFACE_AGREES,CM_INTERFACE_WITNESS } CmInterfaceRelation;
int cm_interface_compare(const CmInterfaceRow *,const CmInterfaceRow *,CmInterfaceRelation *);
#ifdef __cplusplus
}
#endif
#endif
