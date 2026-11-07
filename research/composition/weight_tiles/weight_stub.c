#include "gpu.h"
float *gpu_owned_weight(TcmGpu *g,const void *data,uint32_t dtype,size_t count,int rank,int rows,int cols){(void)g;(void)rank;(void)rows;(void)cols;return gpu_weight(data,dtype,count);}
size_t gpu_owned_weight_bytes(const float *p,size_t count){(void)p;return count*4;}
void gpu_owned_embedding(float *out,const float *embedding,int token,int width){gpu_copy(out,embedding+(size_t)token*width,(size_t)width*4);}
