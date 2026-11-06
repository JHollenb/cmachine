#include "gpu.h"
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
extern "C" void *gpu_alloc(size_t n){void *p=NULL;CUDA(cudaMalloc(&p,n));return p;}
extern "C" void gpu_free(void *p){if(p)CUDA(cudaFree(p));}
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
extern "C" void gpu_matvec(TcmGpu *g,const float *w,int in,int out,const float *x,float *y){const float one=1,zero=0;BLAS(cublasSgemv(g->blas,CUBLAS_OP_T,in,out,&one,w,in,x,1,&zero,y,1));}
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
__global__ static void softmax_kernel(float *scores,int length,float scale){
    float *x=scores+(size_t)blockIdx.x*length;__shared__ float reduction[256];float maximum=-INFINITY;
    for(int i=threadIdx.x;i<length;i+=256)maximum=fmaxf(maximum,x[i]*scale);reduction[threadIdx.x]=maximum;__syncthreads();
    for(int stride=128;stride;stride/=2){if(threadIdx.x<stride)reduction[threadIdx.x]=fmaxf(reduction[threadIdx.x],reduction[threadIdx.x+stride]);__syncthreads();}
    maximum=reduction[0];float sum=0;for(int i=threadIdx.x;i<length;i+=256){float e=expf(x[i]*scale-maximum);x[i]=e;sum+=e;}reduction[threadIdx.x]=sum;__syncthreads();
    for(int stride=128;stride;stride/=2){if(threadIdx.x<stride)reduction[threadIdx.x]+=reduction[threadIdx.x+stride];__syncthreads();}
    sum=reduction[0];for(int i=threadIdx.x;i<length;i+=256)x[i]/=sum;
}
extern "C" void gpu_attention(TcmGpu *g,float *out,const float *q,const float *keys,const float *values,float *scores,int qheads,int kvheads,int dim,int capacity,int length){
    const float one=1,zero=0;int group=qheads/kvheads;
    for(int h=0;h<kvheads;h++)BLAS(cublasSgemm(g->blas,CUBLAS_OP_T,CUBLAS_OP_N,length,group,dim,&one,keys+(size_t)h*capacity*dim,dim,q+(size_t)h*group*dim,dim,&zero,scores+(size_t)h*group*length,length));
    softmax_kernel<<<qheads,256>>>(scores,length,1.0f/sqrtf((float)dim));CUDA(cudaGetLastError());
    for(int h=0;h<kvheads;h++)BLAS(cublasSgemm(g->blas,CUBLAS_OP_N,CUBLAS_OP_N,dim,group,length,&one,values+(size_t)h*capacity*dim,dim,scores+(size_t)h*group*length,length,&zero,out+(size_t)h*group*dim,dim));
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
