#ifndef TCM_SPAN_BANK_H
#define TCM_SPAN_BANK_H
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { SPAN_HEADS=2,SPAN_DIM=128,SPAN_ROWS=34,SPAN_START=59,SPAN_END=93 };
typedef struct {
    int clusters;
    float *counts,*keys,*values,*logweights;
    float *gpu_keys,*gpu_values,*gpu_logweights;
    uint64_t *members;
    char file_sha[65],gpu_before[65];
} SpanProgram;
typedef struct { int active,count;uint64_t members;double key[128],value[128]; } SpanCluster;

/* Constructor has only memory operands; no query, template, future or scorer. */
static int span_construct(SpanProgram *p,int clusters,const float *keys,const float *values){
    memset(p,0,sizeof(*p));if(clusters<1||clusters>34)return -1;p->clusters=clusters;size_t n=(size_t)2*clusters;
    p->counts=malloc(n*4);p->keys=malloc(n*128*4);p->values=malloc(n*128*4);p->logweights=malloc(n*4);p->members=malloc(n*sizeof(uint64_t));
    if(!p->counts||!p->keys||!p->values||!p->logweights||!p->members)return -1;
    for(int head=0;head<2;head++){
        SpanCluster c[34]={0};
        for(int i=0;i<34;i++){c[i].active=c[i].count=1;c[i].members=UINT64_C(1)<<i;for(int d=0;d<128;d++){c[i].key[d]=keys[((size_t)head*34+i)*128+d];c[i].value[d]=values[((size_t)head*34+i)*128+d];}}
        for(int remaining=34;remaining>clusters;remaining--){
            int ai=-1,bi=-1;double best=INFINITY;
            for(int a=0;a<34;a++)if(c[a].active)for(int b=a+1;b<34;b++)if(c[b].active){
                double squared=0;for(int d=0;d<128;d++){double difference=c[a].key[d]-c[b].key[d];squared+=difference*difference;}
                double cost=((double)c[a].count*c[b].count)/(c[a].count+c[b].count)*squared;
                if(cost<best){best=cost;ai=a;bi=b;}
            }
            if(ai<0||bi<0||!isfinite(best))return -1;SpanCluster *a=&c[ai],*b=&c[bi];int count=a->count+b->count;
            for(int d=0;d<128;d++){a->key[d]=(a->count*a->key[d]+b->count*b->key[d])/count;a->value[d]=(a->count*a->value[d]+b->count*b->value[d])/count;}
            a->count=count;a->members|=b->members;b->active=0;
        }
        int j=0;for(int i=0;i<34;i++)if(c[i].active){size_t index=(size_t)head*clusters+j;p->counts[index]=(float)c[i].count;p->logweights[index]=c[i].count==1?0.0f:logf((float)c[i].count);p->members[index]=c[i].members;
            for(int d=0;d<128;d++){p->keys[index*128+d]=(float)c[i].key[d];p->values[index*128+d]=(float)c[i].value[d];}j++;}
        if(j!=clusters)return -1;
    }return 0;
}

/* Bank-only prediction: the original span is not an argument or a global. */
static int span_predict(const SpanProgram *p,const float query[128],int kvhead,float scale,double *log_z,double mean[128]){
    if(kvhead<0||kvhead>=2||p->clusters<1)return -1;double scores[34],maximum=-INFINITY,total=0;
    for(int j=0;j<p->clusters;j++){size_t index=(size_t)kvhead*p->clusters+j;double score=0;for(int d=0;d<128;d++)score+=(double)query[d]*p->keys[index*128+d];score=score*(double)scale+log((double)p->counts[index]);scores[j]=score;if(score>maximum)maximum=score;}
    for(int d=0;d<128;d++)mean[d]=0;
    for(int j=0;j<p->clusters;j++){size_t index=(size_t)kvhead*p->clusters+j;double weight=exp(scores[j]-maximum);total+=weight;for(int d=0;d<128;d++)mean[d]+=weight*p->values[index*128+d];}
    if(!(total>0)||!isfinite(total))return -1;*log_z=maximum+log(total);for(int d=0;d<128;d++)mean[d]/=total;return 0;
}
static void span_host_free(SpanProgram *p){free(p->counts);free(p->keys);free(p->values);free(p->logweights);free(p->members);}
#endif
