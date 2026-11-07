/* Isolated selected-read substitution. Ordinary calls use the pinned function. */
#define gpu_attention historical_gpu_attention
#include "runtime/qwen_cuda.cu"
#undef gpu_attention

static float *packed_keys,*packed_values;
static int armed,clusters,trace_calls,packed_length;
static const float *bank_keys,*bank_values,*bank_logweights;
static float *host_scores;
static int byte_mode;
static const signed char *byte_keys,*byte_values;
static const float *byte_scales;
static const float *guarded_keys;
extern "C" void span_poison_guard(const float *keys){guarded_keys=keys;}
extern "C" void span_workspace_open(void){packed_keys=(float*)gpu_alloc((size_t)2*512*128*4);packed_values=(float*)gpu_alloc((size_t)2*512*128*4);}
extern "C" void span_workspace_close(void){gpu_free(packed_keys);gpu_free(packed_values);packed_keys=packed_values=NULL;}
extern "C" void span_read_arm(int m,const float *k,const float *v,const float *logw,float *scores){armed=1;byte_mode=0;clusters=m;bank_keys=k;bank_values=v;bank_logweights=logw;host_scores=scores;trace_calls=0;packed_length=0;}
extern "C" void span_byte_arm(const signed char *k,const signed char *v,const float *scales,float *scores){armed=1;byte_mode=1;clusters=34;byte_keys=k;byte_values=v;byte_scales=scales;bank_logweights=NULL;host_scores=scores;trace_calls=0;packed_length=0;}
__global__ static void decode_rows(float *target,const signed char *encoded,const float *scales,int kind){
    int index=blockIdx.x,d=threadIdx.x,head=index/34,row=index%34;
    target[((size_t)head*512+59+row)*128+d]=(float)encoded[(size_t)index*128+d]*scales[kind*68+index];
}
extern "C" int span_read_finish(int *length){*length=packed_length;int result=trace_calls;armed=0;return result;}
extern "C" int span_read_length(void){return packed_length;}
__global__ static void weighted_softmax(float *scores,int length,float scale,int m,const float *logw){
    float *x=scores+(size_t)blockIdx.x*length;int kvhead=blockIdx.x/6;__shared__ float reduction[256];float maximum=-INFINITY;
    for(int i=threadIdx.x;i<length;i+=256){float bias=logw&&i>=59&&i<59+m?logw[kvhead*m+i-59]:0.0f;maximum=fmaxf(maximum,x[i]*scale+bias);}reduction[threadIdx.x]=maximum;__syncthreads();
    for(int stride=128;stride;stride/=2){if(threadIdx.x<stride)reduction[threadIdx.x]=fmaxf(reduction[threadIdx.x],reduction[threadIdx.x+stride]);__syncthreads();}
    maximum=reduction[0];float sum=0;
    for(int i=threadIdx.x;i<length;i+=256){float bias=logw&&i>=59&&i<59+m?logw[kvhead*m+i-59]:0.0f;float e=expf((x[i]*scale+bias)-maximum);x[i]=e;sum+=e;}reduction[threadIdx.x]=sum;__syncthreads();
    for(int stride=128;stride;stride/=2){if(threadIdx.x<stride)reduction[threadIdx.x]+=reduction[threadIdx.x+stride];__syncthreads();}
    sum=reduction[0];for(int i=threadIdx.x;i<length;i+=256)x[i]/=sum;
}
extern "C" void gpu_attention(TcmGpu *g,float *out,const float *q,const float *keys,const float *values,float *scores,int qheads,int kvheads,int dim,int capacity,int length){
    if(guarded_keys&&keys==guarded_keys&&(!armed||clusters<0)){fprintf(stderr,"dense fallback on poisoned span refused\n");exit(2);}
    if(!armed){historical_gpu_attention(g,out,q,keys,values,scores,qheads,kvheads,dim,capacity,length);return;}
    if(trace_calls++||qheads!=12||kvheads!=2||dim!=128||capacity!=512||length<=93){fprintf(stderr,"invalid span read geometry\n");exit(2);}
    if(clusters<0){packed_length=length;historical_gpu_attention(g,out,q,keys,values,scores,qheads,kvheads,dim,capacity,length);return;}
    if(clusters>34){fprintf(stderr,"invalid span program\n");exit(2);}packed_length=length-34+clusters;
    for(int h=0;h<2;h++)for(int kind=0;kind<2;kind++){
        const float *source=(kind?values:keys)+(size_t)h*capacity*128;const float *bank=(kind?bank_values:bank_keys);float *target=(kind?packed_values:packed_keys)+(size_t)h*capacity*128;
        gpu_copy(target,source,(size_t)59*128*4);
        if(clusters&&!byte_mode)gpu_copy(target+(size_t)59*128,bank+(size_t)h*clusters*128,(size_t)clusters*128*4);
        gpu_copy(target+(size_t)(59+clusters)*128,source+(size_t)93*128,(size_t)(length-93)*128*4);
    }
    if(byte_mode){decode_rows<<<68,128>>>(packed_keys,byte_keys,byte_scales,0);decode_rows<<<68,128>>>(packed_values,byte_values,byte_scales,1);CUDA(cudaGetLastError());}
    const float one=1,zero=0;int group=6;
    for(int h=0;h<2;h++)BLAS(cublasSgemm(g->blas,CUBLAS_OP_T,CUBLAS_OP_N,packed_length,group,128,&one,packed_keys+(size_t)h*capacity*128,128,q+(size_t)h*group*128,128,&zero,scores+(size_t)h*group*packed_length,packed_length));
    if(host_scores)gpu_read(host_scores,scores,(size_t)12*packed_length*4);
    weighted_softmax<<<12,256>>>(scores,packed_length,1.0f/sqrtf(128.0f),clusters,bank_logweights);CUDA(cudaGetLastError());
    for(int h=0;h<2;h++)BLAS(cublasSgemm(g->blas,CUBLAS_OP_N,CUBLAS_OP_N,128,group,packed_length,&one,packed_values+(size_t)h*capacity*128,128,scores+(size_t)h*group*packed_length,packed_length,&zero,out+(size_t)h*group*128,128));
}
