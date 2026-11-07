#ifndef TCM_PROTOTYPE_ATTENTION_H
#define TCM_PROTOTYPE_ATTENTION_H
/* Fixed 2KV/12Q/128D numerical law imported from the authenticated Qwen reader.
 * The carrier layout remains 98,464 bytes. State ownership/rewind is external. */
#include <stdint.h>
#include <string.h>
#include <math.h>
typedef struct {
    double key_sum[2][16][128],value_sum[2][16][128];
    uint32_t count[2][16],clusters[2];
    float keys[2][16][128],values[2][16][128];
    uint64_t next,recent,head;
} QwenPrototypeMemory;
static inline int qwen_prototype_append(QwenPrototypeMemory *m,uint64_t position,const float *keys,const float *values){
    if(!m||!keys||!values||position!=m->next||position>=UINT32_MAX)return -1;
    for(int j=0;j<256;j++)if(!isfinite(keys[j])||!isfinite(values[j]))return -1;
    if(m->recent==16){
        for(int g=0;g<2;g++){
            const float *k=m->keys[g][m->head],*v=m->values[g][m->head];unsigned c;
            if(m->clusters[g]<16)c=m->clusters[g]++;
            else{c=0;double best=INFINITY;for(unsigned slot=0;slot<16;slot++){double distance=0;for(int j=0;j<128;j++){double e=(double)k[j]-m->key_sum[g][slot][j]/m->count[g][slot];distance+=e*e;}if(distance<best){best=distance;c=slot;}}}
            m->count[g][c]++;for(int j=0;j<128;j++){m->key_sum[g][c][j]+=k[j];m->value_sum[g][c][j]+=v[j];}
        }
    }else m->recent++;
    for(int g=0;g<2;g++){memcpy(m->keys[g][m->head],keys+g*128,128*4);memcpy(m->values[g][m->head],values+g*128,128*4);}
    m->head=(m->head+1)%16;m->next++;return 0;
}
static inline double qwen_prototype_k(const QwenPrototypeMemory *m,int g,unsigned row,int j){return row<m->clusters[g]?m->key_sum[g][row][j]/m->count[g][row]:m->keys[g][(m->head+16-m->recent+row-m->clusters[g])%16][j];}
static inline double qwen_prototype_v(const QwenPrototypeMemory *m,int g,unsigned row,int j){return row<m->clusters[g]?m->value_sum[g][row][j]/m->count[g][row]:m->values[g][(m->head+16-m->recent+row-m->clusters[g])%16][j];}
static inline int qwen_prototype_weights(const QwenPrototypeMemory *m,const float *q,int g,double *score,unsigned *n,double *den){
    unsigned clusters=m->clusters[g];*n=clusters+(unsigned)m->recent;double maximum=-INFINITY;*den=0;
    for(unsigned row=0;row<*n;row++){
        double dot=0;for(int j=0;j<128;j++)dot+=(double)q[j]*qwen_prototype_k(m,g,row,j);
        if(row<clusters&&!m->count[g][row])return -1;
        score[row]=dot/sqrt(128.)+(row<clusters?log((double)m->count[g][row]):0.);if(score[row]>maximum)maximum=score[row];
    }
    for(unsigned row=0;row<*n;row++){score[row]=exp(score[row]-maximum);*den+=score[row];}return !(*den>0)||!isfinite(*den)?-1:0;
}
static inline int qwen_prototype_read(const QwenPrototypeMemory *m,const float *queries,float *out){
    if(!m||!queries||!out||!m->next||m->recent>16||m->clusters[0]>16||m->clusters[1]>16)return -1;
    for(int j=0;j<1536;j++)if(!isfinite(queries[j]))return -1;
    for(int h=0;h<12;h++){
        int g=h/6;double score[32],den;unsigned n;if(qwen_prototype_weights(m,queries+h*128,g,score,&n,&den))return -1;
        for(int j=0;j<128;j++){double value=0;for(unsigned row=0;row<n;row++)value+=score[row]*qwen_prototype_v(m,g,row,j);out[h*128+j]=(float)(value/den);if(!isfinite(out[h*128+j]))return -1;}
    }return 0;
}
/* Smooth FP64 VJP at actual FP32 queries; output rounding is not differentiated.
 * J^T e=sum p*k*((v-a)^T e)/sqrt(d). No 128x128 covariance allocation. */
static inline int qwen_prototype_vjp(const QwenPrototypeMemory *m,const float *queries,const double *error,double *gradient){
    if(!m||!queries||!error||!gradient||!m->next||m->recent>16||m->clusters[0]>16||m->clusters[1]>16)return -1;
    for(int j=0;j<1536;j++)if(!isfinite(queries[j])||!isfinite(error[j]))return -1;
    for(int h=0;h<12;h++){
        int g=h/6;double score[32],den,mean[128]={0},influence[32];unsigned n;if(qwen_prototype_weights(m,queries+h*128,g,score,&n,&den))return -1;
        for(unsigned row=0;row<n;row++)for(int j=0;j<128;j++)mean[j]+=score[row]/den*qwen_prototype_v(m,g,row,j);
        for(unsigned row=0;row<n;row++){double dot=0;for(int j=0;j<128;j++)dot+=(qwen_prototype_v(m,g,row,j)-mean[j])*error[h*128+j];influence[row]=score[row]/den*dot;}
        for(int j=0;j<128;j++){double result=0;for(unsigned row=0;row<n;row++)result+=influence[row]*qwen_prototype_k(m,g,row,j);gradient[h*128+j]=result/sqrt(128.);if(!isfinite(gradient[h*128+j]))return -1;}
    }return 0;
}
#endif
