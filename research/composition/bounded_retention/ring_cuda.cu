#include "weight_cuda.cu"
__global__ static void ring_decode(float *out,const signed char *codes,const float *scales,const int *order,int capacity,int length,int kind){
    int head=blockIdx.x/length,row=blockIdx.x%length,d=threadIdx.x,source=order[row];out[((size_t)head*capacity+row)*128+d]=(float)codes[((size_t)head*capacity+source)*128+d]*scales[(kind*2+head)*capacity+source];
}
__global__ static void ring_bias(float *scores,const float *weights,int length){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<12*length)scores[i]+=weights[i%length]*sqrtf(128.0f);}
extern "C" void gpu_ring_attention(TcmGpu *g,float *out,const float *q,const signed char *key,const signed char *value,const float *scales,float *scores,float *workspace,const int *order,const float *weights,int capacity,int length,int weighted){
    cudaStream_t stream;BLAS(cublasGetStream(g->blas,&stream));if(stream!=0||length<1||length>capacity)exit(2);const float one=1,zero=0;
    ring_decode<<<2*length,128>>>(workspace,key,scales,order,capacity,length,0);CUDA(cudaGetLastError());
    for(int h=0;h<2;h++)BLAS(cublasSgemm(g->blas,CUBLAS_OP_T,CUBLAS_OP_N,length,6,128,&one,workspace+(size_t)h*capacity*128,128,q+(size_t)h*6*128,128,&zero,scores+(size_t)h*6*length,length));
    if(weighted){ring_bias<<<(12*length+255)/256,256>>>(scores,weights,length);CUDA(cudaGetLastError());}
    softmax_kernel<<<12,256>>>(scores,length,1.0f/sqrtf(128.0f));CUDA(cudaGetLastError());
    ring_decode<<<2*length,128>>>(workspace,value,scales,order,capacity,length,1);CUDA(cudaGetLastError());
    for(int h=0;h<2;h++)BLAS(cublasSgemm(g->blas,CUBLAS_OP_N,CUBLAS_OP_N,128,6,length,&one,workspace+(size_t)h*capacity*128,128,scores+(size_t)h*6*length,length,&zero,out+(size_t)h*6*128,128));
}
