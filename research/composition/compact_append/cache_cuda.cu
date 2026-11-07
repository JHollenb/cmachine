#include "runtime/qwen_cuda.cu"
#include <stdint.h>
/* Each block owns one current (kind,KV-head) row, still outside published cursor. */
__global__ static void encode_current(signed char *key,signed char *value,float *scales,const float *k,const float *v,int capacity,int position,int *invalid){
    int kind=blockIdx.x/2,head=blockIdx.x%2,d=threadIdx.x;const float x=(kind?v:k)[head*128+d];__shared__ float maxima[128];
    if(!isfinite(x))atomicExch(invalid,1);maxima[d]=isfinite(x)?fabsf(x):0;__syncthreads();
    for(int stride=64;stride;stride/=2){if(d<stride)maxima[d]=fmaxf(maxima[d],maxima[d+stride]);__syncthreads();}
    float scale=maxima[0]>0?__fdiv_rn(maxima[0],127.0f):1.0f;
    if(!(scale>0)||!isfinite(scale)){if(!d)atomicExch(invalid,1);return;}
    float q=isfinite(x)?roundf(__fdiv_rn(x,scale)):0.0f;q=fminf(127.0f,fmaxf(-127.0f,q));
    (kind?value:key)[((size_t)head*capacity+position)*128+d]=(signed char)q;
    if(!d)scales[(kind*2+head)*capacity+position]=scale;
}
extern "C" int gpu_encode_current(signed char *key,signed char *value,float *scales,const float *k,const float *v,int capacity,int position,int *invalid){
    gpu_zero(invalid,4);encode_current<<<4,128>>>(key,value,scales,k,v,capacity,position,invalid);CUDA(cudaGetLastError());int status;gpu_read(&status,invalid,4);return status?-1:0;
}
__global__ static void decode_prefix(float *out,const signed char *codes,const float *scales,int capacity,int length,int kind){
    int index=blockIdx.x,head=index/length,row=index%length,d=threadIdx.x;
    out[((size_t)head*capacity+row)*128+d]=(float)codes[((size_t)head*capacity+row)*128+d]*scales[(kind*2+head)*capacity+row];
}
extern "C" void gpu_byte_attention(TcmGpu *g,float *out,const float *q,const signed char *key,const signed char *value,const float *scales,float *scores,float *workspace,int capacity,int length){
    cudaStream_t stream;BLAS(cublasGetStream(g->blas,&stream));if(stream!=0){fprintf(stderr,"byte cache requires default stream\n");exit(2);}
    const float one=1,zero=0;
    decode_prefix<<<2*length,128>>>(workspace,key,scales,capacity,length,0);CUDA(cudaGetLastError());
    for(int h=0;h<2;h++)BLAS(cublasSgemm(g->blas,CUBLAS_OP_T,CUBLAS_OP_N,length,6,128,&one,workspace+(size_t)h*capacity*128,128,q+(size_t)h*6*128,128,&zero,scores+(size_t)h*6*length,length));
    softmax_kernel<<<12,256>>>(scores,length,1.0f/sqrtf(128.0f));CUDA(cudaGetLastError());
    /* QK no longer needs keys; the same stream orders value overwrite after it. */
    decode_prefix<<<2*length,128>>>(workspace,value,scales,capacity,length,1);CUDA(cudaGetLastError());
    for(int h=0;h<2;h++)BLAS(cublasSgemm(g->blas,CUBLAS_OP_N,CUBLAS_OP_N,128,6,length,&one,workspace+(size_t)h*capacity*128,128,scores+(size_t)h*6*length,length,&zero,out+(size_t)h*6*128,128));
}
