#ifndef TCM_GPU_H
#define TCM_GPU_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct TcmGpu TcmGpu;
typedef struct { int first, second; float first_value, second_value; } TcmTop2;
TcmGpu *gpu_open(void);
void gpu_close(TcmGpu *);
void *gpu_alloc(size_t);
void gpu_free(void *);
void gpu_zero(void *,size_t);
void gpu_copy(void *,const void *,size_t);
void gpu_read(void *,const void *,size_t);
void gpu_write(void *,const void *,size_t);
float *gpu_weight(const void *,uint32_t,size_t);
void gpu_matvec(TcmGpu *,const float *,int,int,const float *,float *);
void gpu_rms(float *,const float *,const float *,int,float);
/* Experimental receiver-plane primitives. RMS exports its existing reduction;
 * it does not recompute a differently ordered denominator. */
void gpu_rms_scale(float *,const float *,const float *,int,float,float *);
void gpu_multiply(float *,const float *,const float *,int);
void gpu_plane(float *,const float *,const float *,const float *,int,float,float);
void gpu_plane_scaled(float *,const float *,const float *,const float *,int,float,float,const float *);
void gpu_linear2(float *,const float *,const float *,int,float,float,const float *);
int gpu_identical_finite(TcmGpu *,const float *,const float *,int);
void gpu_add(float *,const float *,int);
void gpu_bias(float *,const float *,int);
void gpu_rope(float *,int,int,int,float);
void gpu_cache(float *,const float *,int,int,int,int);
void gpu_attention(TcmGpu *,float *,const float *,const float *,const float *,float *,int,int,int,int,int);
void gpu_swiglu(float *,const float *,int);
/* Source-only coefficients and a separately declared finite FP32 gate law. */
void gpu_gate_anchor(const float *,float *,float *,float *,float *,int);
void gpu_gate_swiglu(float *,const float *,const float *,const float *,const float *,const float *,const float *,int);
TcmTop2 gpu_top2(TcmGpu *,const float *,int);
/* Per-token ledger (src/qwen_ledger.h). Read-only reductions of native values;
 * every native output is produced by the same arithmetic as the plain calls.
 * gpu_attention_ledger: gpu_attention plus per-head softmax fields written to
 *   ledger[field*qheads+head] (field = QL_H_*), from the softmax's own pass.
 * gpu_ledger_dot(dot,norm2,x,w,n,retain): dot=x.w, norm2=w.w in double, stored
 *   as float; dot/x and norm2 and retain may be NULL; retain gets a copy of w.
 * gpu_ledger_gate(out,g,u,n,scratch): out[0..5] = |g|<1, g<=-5, g>=5 counts, sum
 *   even^2, sum odd^2, sum even*odd (QL_GATE_SMALL..QL_EVEN_ODD); call before
 *   swiglu. Per-element terms FP32 (tanhf), sums double.
 *   scratch: device, >= 6*((n+255)/256) doubles, owned by the caller.
 * gpu_ledger_margin(row,embedding,gamma,writes,layers,block_floats,n,a,b,scratch):
 *   margin decomposition of a ledger row from writes[(2*layers+1)*n] = h_0, then the
 *   attention and MLP write of each layer; rho_F is read from row[QL_RHO_FINAL].
 *   embedding is the model's owned handle under the active weight profile.
 *   scratch: device, >= 2*layers+2 doubles (may be the gate scratch). */
void gpu_attention_ledger(TcmGpu *,float *,const float *,const float *,const float *,float *,int,int,int,int,int,float *);
void gpu_ledger_dot(float *,float *,const float *,const float *,int,float *);
void gpu_ledger_gate(float *,const float *,const float *,int,double *);
void gpu_ledger_margin(float *,const float *,const float *,const float *,int,int,int,int,int,double *);
void gpu_sync(void);
/* Explicit process-wide weight profile: 0 native FP32, 1 tiled FP32,
 * 2 source BF16 with FP32 tiles. Tiled profiles own one loaded model; release
 * their shared tile after model destruction before selecting another profile. */
void gpu_weight_configure(int);
int gpu_weight_profile(void);
float *gpu_owned_weight(TcmGpu *,const void *,uint32_t,size_t,int,int,int);
size_t gpu_owned_weight_bytes(const float *,size_t);
void gpu_owned_embedding(float *,const float *,int,int);
void gpu_weight_release(void);
void gpu_weight_info(size_t *,size_t *,size_t *,int *,unsigned long long *,unsigned long long *,unsigned long long *);
size_t gpu_live_bytes(void);
size_t gpu_peak_bytes(void);
size_t gpu_free_bytes(void);
#ifdef __cplusplus
}
#endif
#endif
