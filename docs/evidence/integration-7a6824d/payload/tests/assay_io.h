#ifndef TCM_ASSAY_IO_H
#define TCM_ASSAY_IO_H
/* Small shared I/O for finite C assays. Caller supplies REQUIRE. */
#include "qwen.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
typedef struct { const char *directory,*name; } Capture;
static double clock_seconds(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void digest(const void *p,size_t n,char text[65]){unsigned char raw[32];tcm_sha256(p,n,raw);tcm_hex(raw,32,text);}
static float *read_floats(const float *p,size_t n){float *h=malloc(n*4);REQUIRE(h);gpu_read(h,p,n*4);for(size_t i=0;i<n;i++)REQUIRE(isfinite(h[i]));return h;}
static void save(const char *directory,const char *name,const void *data,size_t bytes){char path[4096];REQUIRE(snprintf(path,sizeof(path),"%s/%s",directory,name)<(int)sizeof(path));FILE *f=fopen(path,"wb");REQUIRE(f);REQUIRE(fwrite(data,1,bytes,f)==bytes);REQUIRE(!fclose(f));}
static void port(void *context,int position,int layer,QwenPort port_id,const float *data,size_t count){
    (void)position;(void)layer;Capture *c=context;if(port_id<QWEN_MLP_INPUT||port_id>QWEN_OUTPUT)return;
    char name[256];REQUIRE(snprintf(name,sizeof(name),"%s-%s.f32",c->name,qwen_port_name(port_id))<(int)sizeof(name));float *h=read_floats(data,count);save(c->directory,name,h,count*4);free(h);
}
static void prefix_digest(const QwenSession *s,int length,char hash[65]){
    const QwenModel *m=s->model;size_t row=(size_t)length*m->head,count=(size_t)2*m->layers*m->kvheads*row,offset=0;float *h=malloc(count*4);REQUIRE(h);
    for(int kind=0;kind<2;kind++)for(int l=0;l<m->layers;l++)for(int k=0;k<m->kvheads;k++){
        size_t address=((size_t)l*m->kvheads+k)*s->capacity*m->head;gpu_read(h+offset,(kind?s->values:s->keys)+address,row*4);offset+=row;
    }digest(h,count*4,hash);free(h);
}
static void kv_row(const QwenSession *s,int position,const char *directory,const char *name){
    const QwenModel *m=s->model;size_t count=(size_t)2*(m->layers-17)*m->kvheads*m->head,offset=0;float *h=malloc(count*4);REQUIRE(h);
    for(int kind=0;kind<2;kind++)for(int l=17;l<m->layers;l++)for(int k=0;k<m->kvheads;k++){
        size_t address=(((size_t)l*m->kvheads+k)*s->capacity+position)*m->head;gpu_read(h+offset,(kind?s->values:s->keys)+address,(size_t)m->head*4);offset+=m->head;
    }for(size_t i=0;i<count;i++)REQUIRE(isfinite(h[i]));save(directory,name,h,count*4);free(h);
}
static void readout(FILE *f,const QwenSession *s,int red,int blue,const char *directory,const char *name){
    REQUIRE(s->phase==1);float *h=read_floats(s->logits,(size_t)s->model->vocab);char hash[65];digest(h,(size_t)s->model->vocab*4,hash);if(name)save(directory,name,h,(size_t)s->model->vocab*4);
    fprintf(f,"{\"top\":%d,\"runner_up\":%d,\"top_value\":%.9g,\"top_margin\":%.9g,\"red_minus_blue\":%.9g,\"sha256\":\"%s\"}",s->pending,s->top.second,s->top.first_value,s->top.first_value-s->top.second_value,h[red]-h[blue],hash);free(h);
}
static int *tokens(const char *path,int *count){FILE *f=fopen(path,"r");REQUIRE(f);int *ids=malloc(512*sizeof(int));REQUIRE(ids);*count=0;int id;while(fscanf(f,"%d",&id)==1){REQUIRE(*count<512);ids[(*count)++]=id;}REQUIRE(feof(f)&&!fclose(f)&&*count>0);return ids;}
static int generated(FILE *f,QwenSession *s,int budget){
    int last=-1,stopped=0;fprintf(f,"{\"ids\":[");for(int i=0;i<budget;i++){
        last=s->pending;REQUIRE(last>=0);if(i)fputc(',',f);fprintf(f,"%d",last);stopped=last==s->model->eos||(s->model->eos_extra>0&&last==s->model->eos_extra);
        REQUIRE(!qwen_push(s,last));if(stopped)break;
    }fprintf(f,"],\"stopped\":%s,\"truncated\":%s,\"all_delivered_consumed\":true}",stopped?"true":"false",stopped?"false":"true");return last;
}
#endif
