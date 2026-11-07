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
void gpu_sync(void);
size_t gpu_free_bytes(void);
#ifdef __cplusplus
}
#endif
#endif
