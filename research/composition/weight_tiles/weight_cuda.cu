/* Matched GEMV geometry isolates lossless storage from reduction-profile changes. */
#define gpu_matvec gpu_base_matvec
#include "cache_cuda.cu"
#undef gpu_matvec
#include <string.h>
struct WeightRecord {float *handle;int in,out,encoded;size_t bytes;};
static WeightRecord weight_record[197];
static int weight_count,weight_mode;
static float *weight_tile;
static const size_t tile_bytes=(size_t)1024*8960*4;
static size_t permanent_matrix_bytes,rank1_bytes;
static unsigned long long matvec_calls,tile_calls,embedding_calls;
extern "C" void gpu_weight_configure(int mode){if(mode<0||mode>2||weight_count||weight_tile)exit(2);weight_mode=mode;}
extern "C" float *gpu_owned_weight(TcmGpu *g,const void *data,uint32_t dtype,size_t count,int rank,int rows,int columns){
    (void)g;
    if(rank!=2){rank1_bytes+=count*4;return gpu_weight(data,dtype,count);}
    if(weight_count>=197||count!=(size_t)rows*columns||columns>8960)exit(2);
    int encoded=weight_mode==2;float *p;
    if(encoded){if(dtype!=2){fprintf(stderr,"exact BF16 matrix source required\n");exit(2);}p=(float*)gpu_alloc(count*2);gpu_write(p,data,count*2);}
    else p=gpu_weight(data,dtype,count);
    size_t bytes=count*(encoded?2:4);weight_record[weight_count++]={p,columns,rows,encoded,bytes};permanent_matrix_bytes+=bytes;
    if(weight_mode&&!weight_tile)weight_tile=(float*)gpu_alloc(tile_bytes);return p;
}
static WeightRecord *find_weight(const float *p){for(int i=0;i<weight_count;i++)if(weight_record[i].handle==p)return &weight_record[i];return NULL;}
extern "C" size_t gpu_owned_weight_bytes(const float *p,size_t count){WeightRecord *r=find_weight(p);return r?r->bytes:count*4;}
extern "C" void gpu_owned_embedding(float *out,const float *embedding,int token,int width){
    WeightRecord *r=find_weight(embedding);if(!r||width!=r->in||token<0||token>=r->out)exit(2);embedding_calls++;
    if(r->encoded){const unsigned short *src=(const unsigned short*)embedding+(size_t)token*width;widen<<<(width+255)/256,256>>>(out,src,width);CUDA(cudaGetLastError());}
    else gpu_copy(out,embedding+(size_t)token*width,(size_t)width*4);
}
extern "C" void gpu_matvec(TcmGpu *g,const float *w,int in,int out,const float *x,float *y){
    WeightRecord *r=find_weight(w);if(!r||r->in!=in||r->out!=out){fprintf(stderr,"unregistered learned matrix\n");exit(2);}matvec_calls++;
    if(!weight_mode){gpu_base_matvec(g,w,in,out,x,y);return;}
    cudaStream_t stream;BLAS(cublasGetStream(g->blas,&stream));if(stream!=0)exit(2);
    for(int row=0;row<out;row+=1024){int n=out-row;if(n>1024)n=1024;size_t elements=(size_t)n*in;if(elements*4>tile_bytes)exit(2);
        if(r->encoded){const unsigned short *src=(const unsigned short*)w+(size_t)row*in;widen<<<(unsigned)((elements+255)/256),256>>>(weight_tile,src,elements);CUDA(cudaGetLastError());}
        else gpu_copy(weight_tile,w+(size_t)row*in,elements*4);
        gpu_base_matvec(g,weight_tile,in,n,x,y+row);tile_calls++;
    }
}
extern "C" void gpu_weight_info(size_t *matrix,size_t *vectors,size_t *scratch,int *count,unsigned long long *matvecs,unsigned long long *tiles,unsigned long long *embeddings){
    *matrix=permanent_matrix_bytes;*vectors=rank1_bytes;*scratch=weight_tile?tile_bytes:0;*count=weight_count;*matvecs=matvec_calls;*tiles=tile_calls;*embeddings=embedding_calls;
}
extern "C" void gpu_weight_release(void){gpu_free(weight_tile);weight_tile=NULL;weight_count=0;permanent_matrix_bytes=rank1_bytes=0;}
