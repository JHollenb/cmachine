#ifndef TCM_CIRCUIT_EXPANSIONS_H
#define TCM_CIRCUIT_EXPANSIONS_H
#include "circuit_math.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Optional host FP64 inspection. Borrowed arrays, disjoint outputs, return 0
 * on success/-1 on invalid geometry or nonfinite arithmetic. No live edits. */
double cm_gelu(double);
double cm_gelu_first(double);
double cm_gelu_second(double);
double cm_gelu_secant(double,double);
int cm_gelu_split(double,double,double *,double *,double *);
/* Per-channel diagonal SSM; R,B,C [T,N], dt [T]. Producers fixed. Output
 * kernel [T,T] excludes skip, outer gate and output projection. No division
 * by retention, so rounded zero retention is supported. */
int cm_ssm_kernel(size_t,size_t,const double *,const double *,const double *,const double *,double *);
/* cut is number of past rows: L [T-cut,N], E [N,cut]; K_future,past=L E. */
int cm_ssm_cut(size_t,size_t,size_t,const double *,const double *,const double *,const double *,double *,double *);
/* Contribution of an actual observed injected write [T,N] to a later read. */
int cm_ssm_path(size_t,size_t,const double *,const double *,const double *,size_t,size_t,double *);
/* Ordered finite state split: R1*dHprev + dR*Hprev0 + dInjection + remainder.
 * Each vector length N; out [4,N]. Includes changed native arithmetic. */
int cm_ssm_finite(size_t,const double *,const double *,const double *,const double *,const double *,const double *,const double *,const double *,double *);
/* Supply consumer adjoint rows J [consumers,coordinates] and nonnegative
 * diagonal weights; report signed J e and sum w*(J e)^2. No fitting or
 * claim about finite outcomes or consumers absent from this set. */
int cm_consumer_metric(size_t,size_t,const double *,const double *,const double *,double *,double *);
/* Four row-major phase planes [4,H,W] to [2H,2W], after per-phase scale/bias.
 * This is a deterministic transform utility, not a FLUX/VAE executor. */
int cm_unpatchify(size_t,size_t,const double *,const double *,const double *,double *);
int cm_dft2(size_t,size_t,const CmComplex *,CmComplex *,int);
int cm_polyphase_combine(size_t,size_t,const CmComplex *,CmComplex *);
typedef struct { double max_spectral_error,phase_average_loss_l2,spatial_energy; } CmPolyphaseAudit;
int cm_polyphase_audit(size_t,size_t,const double *,const double *,const double *,CmPolyphaseAudit *);
#ifdef __cplusplus
}
#endif
#endif
