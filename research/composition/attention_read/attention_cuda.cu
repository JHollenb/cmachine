/* Pinned arithmetic, with one explicitly selected read-only pre-softmax copy. */
#define gpu_attention historical_gpu_attention
#include "runtime/qwen_cuda.cu"
#undef gpu_attention

static float *trace_scores_host;
static int trace_capacity,trace_hits;
static float trace_scale;
extern "C" void attention_trace_arm(float *host,int capacity){trace_scores_host=host;trace_capacity=capacity;trace_hits=0;}
extern "C" int attention_trace_finish(float *scale){int hits=trace_hits;*scale=trace_scale;trace_scores_host=NULL;trace_capacity=0;return hits;}
extern "C" void gpu_attention(TcmGpu *g,float *out,const float *q,const float *keys,const float *values,float *scores,int qheads,int kvheads,int dim,int capacity,int length){
    if(!trace_scores_host){historical_gpu_attention(g,out,q,keys,values,scores,qheads,kvheads,dim,capacity,length);return;}
    if(trace_hits||qheads*length>trace_capacity){fprintf(stderr,"invalid attention trace extent\n");exit(2);}
    const float one=1,zero=0;int group=qheads/kvheads;
    for(int h=0;h<kvheads;h++)BLAS(cublasSgemm(g->blas,CUBLAS_OP_T,CUBLAS_OP_N,length,group,dim,&one,keys+(size_t)h*capacity*dim,dim,q+(size_t)h*group*dim,dim,&zero,scores+(size_t)h*group*length,length));
    gpu_read(trace_scores_host,scores,(size_t)qheads*length*sizeof(float));
    trace_scale=1.0f/sqrtf((float)dim);trace_hits++;
    softmax_kernel<<<qheads,256>>>(scores,length,trace_scale);CUDA(cudaGetLastError());
    for(int h=0;h<kvheads;h++)BLAS(cublasSgemm(g->blas,CUBLAS_OP_N,CUBLAS_OP_N,dim,group,length,&one,values+(size_t)h*capacity*dim,dim,scores+(size_t)h*group*length,length,&zero,out+(size_t)h*group*dim,dim));
}
