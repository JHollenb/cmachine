#ifndef TCM_BYTE_BANK_H
#define TCM_BYTE_BANK_H
#include "span_bank.h"
typedef struct {
    int8_t keys[2*34*128],values[2*34*128];
    float scales[2*2*34];
    int8_t *gpu_keys,*gpu_values;
    float *gpu_scales;
    char file_sha[65],gpu_before[65];
} ByteProgram;
/* Memory-only construction, with no query, future or scorer. */
static int byte_construct(ByteProgram *p,const float *keys,const float *values){
    memset(p,0,sizeof(*p));
    for(int kind=0;kind<2;kind++)for(int row=0;row<68;row++){
        const float *x=(kind?values:keys)+(size_t)row*128;int8_t *dest=(kind?p->values:p->keys)+(size_t)row*128;float maximum=0;
        for(int d=0;d<128;d++){if(!isfinite(x[d]))return -1;maximum=fmaxf(maximum,fabsf(x[d]));}
        float scale=maximum>0?maximum/127.0f:1.0f;if(!(scale>0)||!isfinite(scale))return -1;p->scales[kind*68+row]=scale;
        for(int d=0;d<128;d++){float q=roundf(x[d]/scale);if(q>127)q=127;if(q< -127)q= -127;dest[d]=(int8_t)q;}
    }return 0;
}
static float byte_decode(int8_t x,float scale){return (float)x*scale;}
/* Only program and query operands; never original span arrays. */
static int byte_predict(const ByteProgram *p,const float query[128],int kvhead,float scale,double *log_z,double mean[128]){
    if(kvhead<0||kvhead>=2)return -1;double scores[34],maximum=-INFINITY,total=0;
    for(int j=0;j<34;j++){int row=kvhead*34+j;double score=0;for(int d=0;d<128;d++)score+=(double)query[d]*byte_decode(p->keys[(size_t)row*128+d],p->scales[row]);score*=scale;scores[j]=score;if(score>maximum)maximum=score;}
    for(int d=0;d<128;d++)mean[d]=0;
    for(int j=0;j<34;j++){int row=kvhead*34+j;double w=exp(scores[j]-maximum);total+=w;for(int d=0;d<128;d++)mean[d]+=w*byte_decode(p->values[(size_t)row*128+d],p->scales[68+row]);}
    if(!(total>0)||!isfinite(total))return -1;*log_z=maximum+log(total);for(int d=0;d<128;d++)mean[d]/=total;return 0;
}
#endif
