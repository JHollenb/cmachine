#include "gpu.h"
#include "qwen_ledger.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA(call) do { cudaError_t e=(call);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);} }while(0)
#define BLAS(call) do { cublasStatus_t e=(call);if(e!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cuBLAS %s:%d: status %d\n",__FILE__,__LINE__,(int)e);exit(2);} }while(0)
struct TcmGpu { cublasHandle_t blas; float *partial_values; int *partial_ids,*invalid; TcmTop2 *result; };
extern "C" TcmGpu *gpu_open(void){TcmGpu *g=(TcmGpu*)calloc(1,sizeof(*g));if(!g)exit(2);BLAS(cublasCreate(&g->blas));BLAS(cublasSetMathMode(g->blas,CUBLAS_PEDANTIC_MATH));g->partial_values=(float*)gpu_alloc(2*1024*sizeof(float));g->partial_ids=(int*)gpu_alloc(2*1024*sizeof(int));g->invalid=(int*)gpu_alloc(sizeof(int));g->result=(TcmTop2*)gpu_alloc(sizeof(TcmTop2));return g;}
extern "C" void gpu_close(TcmGpu *g){if(g){gpu_free(g->partial_values);gpu_free(g->partial_ids);gpu_free(g->invalid);gpu_free(g->result);BLAS(cublasDestroy(g->blas));free(g);}}
struct Allocation {void *pointer;size_t bytes;Allocation *next;};
static Allocation *allocations;
static size_t live_allocation_bytes,peak_allocation_bytes;
extern "C" size_t gpu_live_bytes(void){return live_allocation_bytes;}
extern "C" size_t gpu_peak_bytes(void){return peak_allocation_bytes;}
extern "C" void *gpu_alloc(size_t n){void *p=NULL;CUDA(cudaMalloc(&p,n));Allocation *r=(Allocation*)malloc(sizeof(*r));if(!r)exit(2);*r={p,n,allocations};allocations=r;live_allocation_bytes+=n;if(live_allocation_bytes>peak_allocation_bytes)peak_allocation_bytes=live_allocation_bytes;return p;}
extern "C" void gpu_free(void *p){if(p){Allocation **r=&allocations;while(*r&&(*r)->pointer!=p)r=&(*r)->next;if(!*r)exit(2);Allocation *old=*r;CUDA(cudaFree(p));live_allocation_bytes-=old->bytes;*r=old->next;free(old);}}
extern "C" void gpu_zero(void *p,size_t n){CUDA(cudaMemset(p,0,n));}
extern "C" void gpu_copy(void *d,const void *s,size_t n){CUDA(cudaMemcpy(d,s,n,cudaMemcpyDeviceToDevice));}
extern "C" void gpu_read(void *d,const void *s,size_t n){CUDA(cudaMemcpy(d,s,n,cudaMemcpyDeviceToHost));}
extern "C" void gpu_write(void *d,const void *s,size_t n){CUDA(cudaMemcpy(d,s,n,cudaMemcpyHostToDevice));}
extern "C" void gpu_sync(void){CUDA(cudaDeviceSynchronize());}
extern "C" size_t gpu_free_bytes(void){size_t free,total;CUDA(cudaMemGetInfo(&free,&total));return free;}
__global__ static void widen(float *out,const unsigned short *in,size_t n){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=__uint_as_float((unsigned)in[i]<<16);}
extern "C" float *gpu_weight(const void *data,uint32_t dtype,size_t n){
    float *out=(float*)gpu_alloc(n*sizeof(float));
    if(dtype==1)CUDA(cudaMemcpy(out,data,n*sizeof(float),cudaMemcpyHostToDevice));
    else if(dtype==2){void *raw=gpu_alloc(n*2);CUDA(cudaMemcpy(raw,data,n*2,cudaMemcpyHostToDevice));widen<<<(unsigned)((n+255)/256),256>>>(out,(const unsigned short*)raw,n);CUDA(cudaGetLastError());gpu_free(raw);}
    else{fprintf(stderr,"unsupported GPU weight dtype\n");exit(2);}return out;
}

static void gpu_base_matvec(TcmGpu *g,const float *w,int in,int out,const float *x,float *y){const float one=1,zero=0;BLAS(cublasSgemv(g->blas,CUBLAS_OP_T,in,out,&one,w,in,x,1,&zero,y,1));}
__global__ static void rms_kernel(float *out,const float *x,const float *w,int n,float epsilon,float *export_scale){
    __shared__ float sums[256];float sum=0;for(int i=threadIdx.x;i<n;i+=blockDim.x)sum+=x[i]*x[i];sums[threadIdx.x]=sum;__syncthreads();
    for(int stride=128;stride;stride/=2){if(threadIdx.x<stride)sums[threadIdx.x]+=sums[threadIdx.x+stride];__syncthreads();}
    float scale=rsqrtf(sums[0]/n+epsilon);for(int i=threadIdx.x;i<n;i+=blockDim.x)out[i]=(x[i]*scale)*w[i];
    if(export_scale&&!threadIdx.x)*export_scale=scale;
}
extern "C" void gpu_rms(float *out,const float *x,const float *w,int n,float eps){rms_kernel<<<1,256>>>(out,x,w,n,eps,NULL);CUDA(cudaGetLastError());}
extern "C" void gpu_rms_scale(float *out,const float *x,const float *w,int n,float eps,float *scale){rms_kernel<<<1,256>>>(out,x,w,n,eps,scale);CUDA(cudaGetLastError());}
__global__ static void multiply_kernel(float *out,const float *x,const float *y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=x[i]*y[i];}
extern "C" void gpu_multiply(float *out,const float *x,const float *y,int n){multiply_kernel<<<(n+255)/256,256>>>(out,x,y,n);CUDA(cudaGetLastError());}
__global__ static void plane_kernel(float *out,const float *base,const float *a,const float *b,int n,float s,float t,const float *scale){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=(base[i]+s*a[i])+t*b[i];out[i]=scale?(*scale)*v:v;}}
extern "C" void gpu_plane(float *out,const float *base,const float *a,const float *b,int n,float s,float t){plane_kernel<<<(n+255)/256,256>>>(out,base,a,b,n,s,t,NULL);CUDA(cudaGetLastError());}
extern "C" void gpu_plane_scaled(float *out,const float *base,const float *a,const float *b,int n,float s,float t,const float *scale){plane_kernel<<<(n+255)/256,256>>>(out,base,a,b,n,s,t,scale);CUDA(cudaGetLastError());}
__global__ static void linear2_kernel(float *out,const float *a,const float *b,int n,float ca,float cb,const float *scale){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=ca*a[i];if(b)v+=cb*b[i];out[i]=scale?(*scale)*v:v;}}
extern "C" void gpu_linear2(float *out,const float *a,const float *b,int n,float ca,float cb,const float *scale){linear2_kernel<<<(n+255)/256,256>>>(out,a,b,n,ca,cb,scale);CUDA(cudaGetLastError());}
__global__ static void identical_kernel(const float *a,const float *b,int n,int *invalid){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&(!isfinite(a[i])||!isfinite(b[i])||__float_as_uint(a[i])!=__float_as_uint(b[i])))atomicExch(invalid,1);}
extern "C" int gpu_identical_finite(TcmGpu *g,const float *a,const float *b,int n){gpu_zero(g->invalid,sizeof(int));identical_kernel<<<(n+255)/256,256>>>(a,b,n,g->invalid);CUDA(cudaGetLastError());int invalid;gpu_read(&invalid,g->invalid,sizeof(int));return !invalid;}
__global__ static void add_kernel(float *x,const float *y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]+=y[i];}
extern "C" void gpu_add(float *x,const float *y,int n){add_kernel<<<(n+255)/256,256>>>(x,y,n);CUDA(cudaGetLastError());}
extern "C" void gpu_bias(float *x,const float *y,int n){gpu_add(x,y,n);}
__global__ static void rope_kernel(float *x,int heads,int dimension,int position,float theta){
    int i=blockIdx.x*blockDim.x+threadIdx.x,half=dimension/2;if(i>=heads*half)return;int h=i/half,d=i%half;
    // Same half-split rotation and FP32 frequency construction as Qwen2.5.
    float inverse=1.0f/powf(theta,(2.0f*d)/dimension);float angle=(float)position*inverse;
    float cosine=cosf(angle),sine=sinf(angle),a=x[h*dimension+d],b=x[h*dimension+d+half];
    x[h*dimension+d]=a*cosine+(-b)*sine;x[h*dimension+d+half]=b*cosine+a*sine;
}
extern "C" void gpu_rope(float *x,int heads,int dim,int position,float theta){rope_kernel<<<(heads*(dim/2)+255)/256,256>>>(x,heads,dim,position,theta);CUDA(cudaGetLastError());}
__global__ static void cache_kernel(float *cache,const float *x,int heads,int dim,int capacity,int position){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<heads*dim)cache[(size_t)(i/dim)*capacity*dim+(size_t)position*dim+i%dim]=x[i];}
extern "C" void gpu_cache(float *cache,const float *x,int heads,int dim,int capacity,int pos){cache_kernel<<<(heads*dim+255)/256,256>>>(cache,x,heads,dim,capacity,pos);CUDA(cudaGetLastError());}
/* One score row per block, ledger path only (ledger != NULL). The native statements
 * are those of softmax_kernel below, in the same order, and the ledger branch only reads
 * native values (scaled scores, this pass's maximum, native probabilities).
 * ledger[field*heads+blockIdx.x], field = QL_H_*. Entropy terms p*logf(p) are FP32,
 * summed in double; logZ sums double exp terms. */
__device__ static void softmax_row(float *scores,int length,float scale,float *ledger,int heads){
    float *x=scores+(size_t)blockIdx.x*length;__shared__ float reduction[256];float maximum=-INFINITY;
    __shared__ double ledger_partition[256],ledger_entropy[256];__shared__ float ledger_best[256];__shared__ int ledger_index[256];
    for(int i=threadIdx.x;i<length;i+=256)maximum=fmaxf(maximum,x[i]*scale);reduction[threadIdx.x]=maximum;__syncthreads();
    for(int stride=128;stride;stride/=2){if(threadIdx.x<stride)reduction[threadIdx.x]=fmaxf(reduction[threadIdx.x],reduction[threadIdx.x+stride]);__syncthreads();}
    maximum=reduction[0];
    /* Ledger only: every thread has read reduction[0] before it is reused below. */
    if(ledger)__syncthreads();
    double ledger_z=0;
    float sum=0;for(int i=threadIdx.x;i<length;i+=256){float e=expf(x[i]*scale-maximum);if(ledger)ledger_z+=exp((double)(x[i]*scale)-(double)maximum);x[i]=e;sum+=e;}reduction[threadIdx.x]=sum;__syncthreads();
    for(int stride=128;stride;stride/=2){if(threadIdx.x<stride)reduction[threadIdx.x]+=reduction[threadIdx.x+stride];__syncthreads();}
    double entropy=0;float best=-1.0f;int best_index=length;
    sum=reduction[0];for(int i=threadIdx.x;i<length;i+=256){x[i]/=sum;if(ledger){float p=x[i];if(p>0.0f)entropy-=(double)(p*logf(p));if(p>best){best=p;best_index=i;}}}
    if(ledger){
        int t=threadIdx.x;ledger_partition[t]=ledger_z;ledger_entropy[t]=entropy;ledger_best[t]=best;ledger_index[t]=best_index;__syncthreads();
        for(int stride=128;stride;stride/=2){
            if(t<stride){
                ledger_partition[t]+=ledger_partition[t+stride];ledger_entropy[t]+=ledger_entropy[t+stride];
                float b=ledger_best[t+stride];int bi=ledger_index[t+stride];
                if(b>ledger_best[t]||(b==ledger_best[t]&&bi<ledger_index[t])){ledger_best[t]=b;ledger_index[t]=bi;}
            }
            __syncthreads();
        }
        if(!t){
            int h=blockIdx.x;
            ledger[QL_H_LOGZ*heads+h]=(float)((double)maximum+log(ledger_partition[0]));
            ledger[QL_H_PFIRST*heads+h]=x[0];
            ledger[QL_H_ENTROPY*heads+h]=(float)ledger_entropy[0];
            ledger[QL_H_PMAX*heads+h]=ledger_best[0];
            ledger[QL_H_ARGMAX*heads+h]=(float)ledger_index[0];
        }
    }
}
/* The OFF path keeps the historical kernel verbatim (claude/speedups); softmax_row
 * is used only by the ledger kernel. */
__global__ static void softmax_kernel(float *scores,int length,float scale){
    float *x=scores+(size_t)blockIdx.x*length;__shared__ float reduction[256];float maximum=-INFINITY;
    for(int i=threadIdx.x;i<length;i+=256)maximum=fmaxf(maximum,x[i]*scale);reduction[threadIdx.x]=maximum;__syncthreads();
    for(int stride=128;stride;stride/=2){if(threadIdx.x<stride)reduction[threadIdx.x]=fmaxf(reduction[threadIdx.x],reduction[threadIdx.x+stride]);__syncthreads();}
    maximum=reduction[0];float sum=0;for(int i=threadIdx.x;i<length;i+=256){float e=expf(x[i]*scale-maximum);x[i]=e;sum+=e;}reduction[threadIdx.x]=sum;__syncthreads();
    for(int stride=128;stride;stride/=2){if(threadIdx.x<stride)reduction[threadIdx.x]+=reduction[threadIdx.x+stride];__syncthreads();}
    sum=reduction[0];for(int i=threadIdx.x;i<length;i+=256)x[i]/=sum;
}
__global__ static void softmax_ledger_kernel(float *scores,int length,float scale,float *ledger,int heads){softmax_row(scores,length,scale,ledger,heads);}
static void attention(TcmGpu *g,float *out,const float *q,const float *keys,const float *values,float *scores,int qheads,int kvheads,int dim,int capacity,int length,float *ledger){
    const float one=1,zero=0;int group=qheads/kvheads;
    for(int h=0;h<kvheads;h++)BLAS(cublasSgemm(g->blas,CUBLAS_OP_T,CUBLAS_OP_N,length,group,dim,&one,keys+(size_t)h*capacity*dim,dim,q+(size_t)h*group*dim,dim,&zero,scores+(size_t)h*group*length,length));
    if(ledger)softmax_ledger_kernel<<<qheads,256>>>(scores,length,1.0f/sqrtf((float)dim),ledger,qheads);
    else softmax_kernel<<<qheads,256>>>(scores,length,1.0f/sqrtf((float)dim));
    CUDA(cudaGetLastError());
    for(int h=0;h<kvheads;h++)BLAS(cublasSgemm(g->blas,CUBLAS_OP_N,CUBLAS_OP_N,dim,group,length,&one,values+(size_t)h*capacity*dim,dim,scores+(size_t)h*group*length,length,&zero,out+(size_t)h*group*dim,dim));
}
extern "C" void gpu_attention(TcmGpu *g,float *out,const float *q,const float *keys,const float *values,float *scores,int qheads,int kvheads,int dim,int capacity,int length){
    attention(g,out,q,keys,values,scores,qheads,kvheads,dim,capacity,length,NULL);
}
extern "C" void gpu_attention_ledger(TcmGpu *g,float *out,const float *q,const float *keys,const float *values,float *scores,int qheads,int kvheads,int dim,int capacity,int length,float *ledger){
    attention(g,out,q,keys,values,scores,qheads,kvheads,dim,capacity,length,ledger);
}
__global__ static void swiglu_kernel(float *gate,const float *up,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)gate[i]=(gate[i]/(1.0f+expf(-gate[i])))*up[i];}
extern "C" void gpu_swiglu(float *gate,const float *up,int n){swiglu_kernel<<<(n+255)/256,256>>>(gate,up,n);CUDA(cudaGetLastError());}
__global__ static void gate_anchor_kernel(const float *g,float *p,float *first,float *second,float *s,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){
        float denominator=1.0f+expf(-g[i]);p[i]=1.0f/denominator;
        first[i]=p[i]*(1.0f-p[i]);second[i]=((0.5f*first[i])*(1.0f-2.0f*p[i]));
        s[i]=g[i]/denominator;
    }
}
extern "C" void gpu_gate_anchor(const float *g,float *p,float *first,float *second,float *s,int n){
    gate_anchor_kernel<<<(n+255)/256,256>>>(g,p,first,second,s,n);CUDA(cudaGetLastError());
}
__global__ static void gate_swiglu_kernel(float *g,const float *u,const float *source,const float *p,const float *first,const float *second,const float *s,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){
        float d=g[i]-source[i];float sigma=(p[i]+first[i]*d)+(second[i]*d)*d;
        float predicted=s[i]+(g[i]*sigma-source[i]*p[i]);g[i]=predicted*u[i];
    }
}
extern "C" void gpu_gate_swiglu(float *g,const float *u,const float *source,const float *p,const float *first,const float *second,const float *s,int n){
    gate_swiglu_kernel<<<(n+255)/256,256>>>(g,u,source,p,first,second,s,n);CUDA(cudaGetLastError());
}
__device__ static bool better(float a,int ai,float b,int bi){return a>b||(a==b&&ai<bi);}
__device__ static void insert(float value,int id,float &a,int &ai,float &b,int &bi){if(id<0)return;if(better(value,id,a,ai)){b=a;bi=ai;a=value;ai=id;}else if(id!=ai&&better(value,id,b,bi)){b=value;bi=id;}}
__global__ static void top_kernel(const float *x,int n,float *values,int *ids,int *invalid){
    __shared__ float first[256],second[256];__shared__ int fi[256],si[256];int tid=threadIdx.x,i=blockIdx.x*256+tid;
    if(i<n&&!isfinite(x[i]))atomicExch(invalid,1);
    first[tid]=i<n?x[i]:-INFINITY;fi[tid]=i<n?i:2147483647;second[tid]=-INFINITY;si[tid]=2147483647;__syncthreads();
    for(int stride=128;stride;stride/=2){if(tid<stride){insert(first[tid+stride],fi[tid+stride],first[tid],fi[tid],second[tid],si[tid]);insert(second[tid+stride],si[tid+stride],first[tid],fi[tid],second[tid],si[tid]);}__syncthreads();}
    if(!tid){values[2*blockIdx.x]=first[0];values[2*blockIdx.x+1]=second[0];ids[2*blockIdx.x]=fi[0];ids[2*blockIdx.x+1]=si[0];}
}
__global__ static void finish_top(const float *values,const int *ids,int n,TcmTop2 *out){
    __shared__ float first[256],second[256];__shared__ int fi[256],si[256];int tid=threadIdx.x;float a=-INFINITY,b=-INFINITY;int ai=2147483647,bi=2147483647;
    for(int i=tid;i<n;i+=256)insert(values[i],ids[i],a,ai,b,bi);first[tid]=a;second[tid]=b;fi[tid]=ai;si[tid]=bi;__syncthreads();
    for(int stride=128;stride;stride/=2){if(tid<stride){insert(first[tid+stride],fi[tid+stride],first[tid],fi[tid],second[tid],si[tid]);insert(second[tid+stride],si[tid+stride],first[tid],fi[tid],second[tid],si[tid]);}__syncthreads();}
    if(!tid){out->first=fi[0];out->second=si[0];out->first_value=first[0];out->second_value=second[0];}
}
extern "C" TcmTop2 gpu_top2(TcmGpu *g,const float *x,int n){
    int blocks=(n+255)/256;if(blocks>1024){fprintf(stderr,"top2 vocabulary exceeds adapter capacity\n");exit(2);}gpu_zero(g->invalid,sizeof(int));top_kernel<<<blocks,256>>>(x,n,g->partial_values,g->partial_ids,g->invalid);CUDA(cudaGetLastError());
    finish_top<<<1,256>>>(g->partial_values,g->partial_ids,2*blocks,g->result);CUDA(cudaGetLastError());TcmTop2 result;gpu_read(&result,g->result,sizeof(result));int invalid;gpu_read(&invalid,g->invalid,sizeof(int));if(invalid){fprintf(stderr,"non-finite Qwen logits\n");exit(2);}return result;
}

/* These extensions consume the already-defined GEMV and softmax primitives. */
#include "qwen_weights.inc"
#include "qwen_cache.inc"

/* Per-token ledger reductions (src/qwen_ledger.h). One block each, fixed-order
 * double tree reductions (deterministic, no atomics); they only read native
 * buffers and write ledger storage. Placed after the weight extension so the
 * margin can read the embedding under every residency profile. */
__global__ static void ledger_dot_kernel(float *dot,float *norm2,const float *x,const float *w,int n,float *retain){
    __shared__ double a[256],b[256];int t=threadIdx.x;double sa=0,sb=0;
    for(int i=t;i<n;i+=256){double v=(double)w[i];if(x)sa+=(double)x[i]*v;sb+=v*v;if(retain)retain[i]=w[i];}
    a[t]=sa;b[t]=sb;__syncthreads();
    for(int stride=128;stride;stride/=2){if(t<stride){a[t]+=a[t+stride];b[t]+=b[t+stride];}__syncthreads();}
    if(!t){if(dot)*dot=(float)a[0];if(norm2)*norm2=(float)b[0];}
}
extern "C" void gpu_ledger_dot(float *dot,float *norm2,const float *x,const float *w,int n,float *retain){
    ledger_dot_kernel<<<1,256>>>(dot,norm2,x,w,n,retain);CUDA(cudaGetLastError());
}
/* Gate census: one element per thread across ceil(n/256) blocks; the per-element
 * terms are FP32 (tanhf), the block trees and the cross-block sum are double
 * (counts are exact in double); one thread sums the blocks in index order. */
__global__ static void ledger_gate_partial(double *partials,const float *g,const float *u,int n){
    __shared__ double sums[6][256];int t=threadIdx.x,i=blockIdx.x*256+t;double v[6]={0,0,0,0,0,0};
    if(i<n){
        float gi=g[i];v[0]=fabsf(gi)<1.0f?1.0:0.0;v[1]=gi<=-5.0f?1.0:0.0;v[2]=gi>=5.0f?1.0:0.0;
        float even=(gi*u[i])*0.5f,odd=even*tanhf(gi*0.5f);v[3]=(double)(even*even);v[4]=(double)(odd*odd);v[5]=(double)(even*odd);
    }
    for(int k=0;k<6;k++)sums[k][t]=v[k];__syncthreads();
    for(int stride=128;stride;stride/=2){if(t<stride)for(int k=0;k<6;k++)sums[k][t]+=sums[k][t+stride];__syncthreads();}
    if(!t)for(int k=0;k<6;k++)partials[(size_t)blockIdx.x*6+k]=sums[k][0];
}
__global__ static void ledger_gate_finish(float *out,const double *partials,int blocks){
    if(threadIdx.x||blockIdx.x)return;
    double total[6]={0,0,0,0,0,0};for(int b=0;b<blocks;b++)for(int k=0;k<6;k++)total[k]+=partials[(size_t)b*6+k];
    for(int k=0;k<6;k++)out[k]=(float)total[k];
}
extern "C" void gpu_ledger_gate(float *out,const float *g,const float *u,int n,double *scratch){
    int blocks=(n+255)/256;if(blocks<1)return;
    ledger_gate_partial<<<blocks,256>>>(scratch,g,u,n);CUDA(cudaGetLastError());
    ledger_gate_finish<<<1,1>>>(out,scratch,blocks);CUDA(cudaGetLastError());
}
/* c_i = gamma_i*(W_a,i - W_b,i) in double; vector -1 is ||c||^2, vector 0 is h_0,
 * then the attention and MLP write of each layer. One block per vector (2L+2 blocks,
 * same per-thread order and tree as the former single-block loop) writes its double
 * dot to scratch; one thread then forms the terms and sums them in that fixed order.
 * encoded != NULL: source-BF16 rows, widened exactly as `widen`. */
__global__ static void ledger_margin_partial(double *dots,const float *embedding,const unsigned short *encoded,const float *gamma,const float *writes,int n,int a,int b){
    __shared__ double partial[256];int t=threadIdx.x,j=(int)blockIdx.x-1;
    double sum=0;
    for(int i=t;i<n;i+=256){
        double wa,wb;
        if(encoded){wa=(double)__uint_as_float((unsigned)encoded[(size_t)a*n+i]<<16);wb=(double)__uint_as_float((unsigned)encoded[(size_t)b*n+i]<<16);}
        else{wa=(double)embedding[(size_t)a*n+i];wb=(double)embedding[(size_t)b*n+i];}
        double c=(double)gamma[i]*(wa-wb);
        sum+=j<0?c*c:c*(double)writes[(size_t)j*n+i];
    }
    partial[t]=sum;__syncthreads();
    for(int stride=128;stride;stride/=2){if(t<stride)partial[t]+=partial[t+stride];__syncthreads();}
    if(!t)dots[blockIdx.x]=partial[0];
}
__global__ static void ledger_margin_finish(float *row,const double *dots,int layers,int block_floats){
    if(threadIdx.x||blockIdx.x)return;
    double rho=(double)row[QL_RHO_FINAL],total=0;
    row[QL_C_NORM]=(float)sqrt(dots[0]);
    for(int j=0;j<2*layers+1;j++){
        double term=rho*dots[j+1];total+=term;
        if(!j)row[QL_MARGIN_EMBED]=(float)term;
        else row[QL_HEAD+(size_t)((j-1)/2)*block_floats+((j-1)%2?QL_MARGIN_MLP:QL_MARGIN_ATTN)]=(float)term;
    }
    row[QL_MARGIN_SUM]=(float)total;
}
extern "C" void gpu_ledger_margin(float *row,const float *embedding,const float *gamma,const float *writes,int layers,int block_floats,int n,int a,int b,double *scratch){
    /* Profile 0 and tiled FP32 (1) hold FP32 rows; source BF16 (2) holds encoded rows. */
    WeightRecord *r=weight_mode?find_weight(embedding):NULL;
    if(weight_mode==2&&(!r||!r->encoded||r->in!=n)){fprintf(stderr,"ledger: unrecognized BF16 embedding residency\n");exit(2);}
    const unsigned short *encoded=r&&r->encoded?(const unsigned short*)embedding:NULL;
    ledger_margin_partial<<<2*layers+2,256>>>(scratch,embedding,encoded,gamma,writes,n,a,b);CUDA(cudaGetLastError());
    ledger_margin_finish<<<1,1>>>(row,scratch,layers,block_floats);CUDA(cudaGetLastError());
}
