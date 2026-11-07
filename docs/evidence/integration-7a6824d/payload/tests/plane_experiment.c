#define _POSIX_C_SOURCE 200809L
#include "qwen.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Finite scientific driver, not a serving policy. One resident model; every
 * target starts from the same immutable token-boundary prefix. */
#define REQUIRE(x) do{if(!(x)){fprintf(stderr,"plane driver %s:%d: %s\n",__FILE__,__LINE__,#x);exit(2);}}while(0)
typedef struct { const char *directory,*name; } Capture;
static double clock_seconds(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void digest(const void *p,size_t n,char text[65]){unsigned char raw[32];tcm_sha256(p,n,raw);tcm_hex(raw,32,text);}
static float *read_floats(const float *p,size_t n){float *h=malloc(n*4);REQUIRE(h);gpu_read(h,p,n*4);for(size_t i=0;i<n;i++)REQUIRE(isfinite(h[i]));return h;}
static void save(const char *directory,const char *name,const void *data,size_t bytes){char path[4096];REQUIRE(snprintf(path,sizeof(path),"%s/%s",directory,name)<(int)sizeof(path));FILE *f=fopen(path,"wb");REQUIRE(f);REQUIRE(fwrite(data,1,bytes,f)==bytes);REQUIRE(!fclose(f));}
static void port(void *context,int position,int layer,QwenPort port_id,const float *data,size_t count){
    (void)position;(void)layer;Capture *c=context;
    if(port_id<QWEN_MLP_INPUT||port_id>QWEN_OUTPUT)return;
    char name[256];REQUIRE(snprintf(name,sizeof(name),"%s-%s.f32",c->name,qwen_port_name(port_id))<(int)sizeof(name));float *h=read_floats(data,count);save(c->directory,name,h,count*4);free(h);
}
static void program(const QwenPlane *p,const char *directory,char hash[65]){
    int d=p->model->hidden,h=p->model->intermediate;size_t count=(size_t)4*d+(size_t)6*h,offset=0;float *raw=malloc(count*4);REQUIRE(raw);
    for(int j=0;j<3;j++){gpu_read(raw+offset,p->basis[j],(size_t)d*4);offset+=d;}
    for(int j=0;j<3;j++){gpu_read(raw+offset,p->gate[j],(size_t)h*4);offset+=h;}
    for(int j=0;j<3;j++){gpu_read(raw+offset,p->up[j],(size_t)h*4);offset+=h;}
    gpu_read(raw+offset,p->baseline_write,(size_t)d*4);offset+=d;REQUIRE(offset==count);
    for(size_t i=0;i<count;i++)REQUIRE(isfinite(raw[i]));digest(raw,count*4,hash);if(directory)save(directory,"plane-program.f32",raw,count*4);free(raw);
}
static void prefix_digest(const QwenSession *s,int length,char hash[65]){
    const QwenModel *m=s->model;size_t row=(size_t)length*m->head,count=(size_t)2*m->layers*m->kvheads*row,offset=0;float *h=malloc(count*4);REQUIRE(h);
    for(int kind=0;kind<2;kind++)for(int l=0;l<m->layers;l++)for(int k=0;k<m->kvheads;k++){
        size_t address=((size_t)l*m->kvheads+k)*s->capacity*m->head;
        gpu_read(h+offset,(kind?s->values:s->keys)+address,row*4);offset+=row;
    }digest(h,count*4,hash);free(h);
}
static void kv_row(const QwenSession *s,int position,const char *directory,const char *name){
    const QwenModel *m=s->model;size_t count=(size_t)2*(m->layers-17)*m->kvheads*m->head,offset=0;float *h=malloc(count*4);REQUIRE(h);
    for(int kind=0;kind<2;kind++)for(int l=17;l<m->layers;l++)for(int k=0;k<m->kvheads;k++){
        size_t address=(((size_t)l*m->kvheads+k)*s->capacity+position)*m->head;
        gpu_read(h+offset,(kind?s->values:s->keys)+address,(size_t)m->head*4);offset+=m->head;
    }for(size_t i=0;i<count;i++)REQUIRE(isfinite(h[i]));save(directory,name,h,count*4);free(h);
}
static void readout(FILE *f,const QwenSession *s,int red,int blue,const char *directory,const char *name){
    REQUIRE(s->phase==1);float *h=read_floats(s->logits,(size_t)s->model->vocab);char hash[65];digest(h,(size_t)s->model->vocab*4,hash);
    if(name)save(directory,name,h,(size_t)s->model->vocab*4);
    fprintf(f,"{\"top\":%d,\"runner_up\":%d,\"top_value\":%.9g,\"top_margin\":%.9g,\"red_minus_blue\":%.9g,\"sha256\":\"%s\"}",s->pending,s->top.second,s->top.first_value,s->top.first_value-s->top.second_value,h[red]-h[blue],hash);free(h);
}
static int *tokens(const char *path,int *count){
    FILE *f=fopen(path,"r");REQUIRE(f);int *ids=malloc(512*sizeof(int));REQUIRE(ids);*count=0;int id;
    while(fscanf(f,"%d",&id)==1){REQUIRE(*count<512);ids[(*count)++]=id;}REQUIRE(feof(f)&&!fclose(f)&&*count>0);return ids;
}
static int generated(FILE *f,QwenSession *s,int budget){
    int last=-1,stopped=0;fprintf(f,"{\"ids\":[");
    for(int i=0;i<budget;i++){
        last=s->pending;REQUIRE(last>=0);if(i)fputc(',',f);fprintf(f,"%d",last);
        stopped=last==s->model->eos||(s->model->eos_extra>0&&last==s->model->eos_extra);
        /* Consume every delivered token, including the last/EOS, so later
         * questions use the generated state rather than an oracle transcript. */
        REQUIRE(!qwen_push(s,last));if(stopped)break;
    }fprintf(f,"],\"stopped\":%s,\"truncated\":%s,\"all_delivered_consumed\":true}",stopped?"true":"false",stopped?"false":"true");return last;
}
int main(int argc,char **argv){
    REQUIRE(argc==8);const char *directory=argv[5];int red=atoi(argv[6]),blue=atoi(argv[7]),n,nt,nc;
    int *prompt=tokens(argv[2],&n),*target=tokens(argv[3],&nt),*collateral=tokens(argv[4],&nc);REQUIRE(n>1);
    TcmModel bundle;QwenModel m;QwenSession s;QwenPlane p;char error[256];double started=clock_seconds();
    REQUIRE(!tcm_load(&bundle,argv[1],error,sizeof(error)));REQUIRE(!qwen_model_init(&m,&bundle,error,sizeof(error)));tcm_unload(&bundle);
    REQUIRE(red>=0&&red<m.vocab&&blue>=0&&blue<m.vocab);REQUIRE(!qwen_session_init(&s,&m,512));REQUIRE(!qwen_plane_init(&p,&m));
    for(int i=0;i<n-1;i++)REQUIRE(!qwen_push(&s,prompt[i]));char prefix[65],compiled_hash[65];prefix_digest(&s,n-1,prefix);
    char report_path[4096];REQUIRE(snprintf(report_path,sizeof(report_path),"%s/plane-driver.json",directory)<(int)sizeof(report_path));FILE *f=fopen(report_path,"wb");REQUIRE(f);
    const float doses[][2]={{0,0},{.25f,0},{-.25f,0},{-1,0},{0,.25f},{.5f,.5f}};
    int common=-1;Capture capture={directory,NULL};s.observer=port;s.observer_context=&capture;s.observe_position=n-1;s.observe_layer=16;
    fprintf(f,"{\"schema\":\"tcm-receiver-plane-driver-v1\",\"model_digest\":\"%s\",\"position\":%d,\"layer\":16,\"capacity\":512,\"weight_bytes\":%zu,\"session_bytes\":%zu,\"program_bytes\":%zu,\"prefix_sha256\":\"%s\",\"rows\":[",m.digest,n-1,m.weight_bytes,s.allocated_bytes,p.allocated_bytes-4,prefix);
    for(int arm=0;arm<19;arm++){
        int dose=arm?(arm-1)/3:0,mode=arm?(arm-1)%3:-1;float a=doses[dose][0],b=doses[dose][1];char name[64],file[128];
        if(!arm)strcpy(name,"native");else snprintf(name,sizeof(name),"%s-%d",mode==0?"dense":mode==1?"compiled":"held",dose);capture.name=name;
        uint64_t outer;REQUIRE(!qwen_stack_push(&s,&outer));REQUIRE(!qwen_begin_token(&s,prompt[n-1]));while(s.layer<16)REQUIRE(!qwen_advance_layer(&s));
        QwenPlaneWork work;double transition_start=clock_seconds();
        if(!arm){REQUIRE(!qwen_advance_layer_compile(&s,&p,&work));program(&p,directory,compiled_hash);}
        else REQUIRE(!qwen_advance_layer_plane(&s,&p,(QwenPlaneMode)mode,a,b,&work));
        gpu_sync();double transition_seconds=clock_seconds()-transition_start;
        while(s.layer<m.layers)REQUIRE(!qwen_advance_layer(&s));REQUIRE(!qwen_finish_token(&s));if(!arm)common=s.pending;
        if(arm)fputc(',',f);fprintf(f,"{\"name\":\"%s\",\"dose\":[%.9g,%.9g],\"work\":{\"gate_up_matvecs\":%u,\"down_matvecs\":%u,\"compile_matvecs\":%u},\"selected_transition_seconds\":%.9g,\"first\":",name,a,b,work.gate_up_matvecs,work.down_matvecs,work.compile_matvecs,transition_seconds);
        snprintf(file,sizeof(file),"%s-first.f32",name);readout(f,&s,red,blue,directory,file);snprintf(file,sizeof(file),"%s-current-kv.f32",name);kv_row(&s,n-1,directory,file);
        uint64_t child;REQUIRE(!qwen_stack_push(&s,&child));REQUIRE(!qwen_push(&s,common));fprintf(f,",\"same_history_next\":");readout(f,&s,red,blue,NULL,NULL);snprintf(file,sizeof(file),"%s-next-kv.f32",name);kv_row(&s,n,directory,file);REQUIRE(!qwen_stack_rollback(&s,child));
        fprintf(f,",\"own\":");int last=generated(f,&s,8);fprintf(f,",\"questions\":[");
        for(int question=0;question<2;question++){
            REQUIRE(!qwen_stack_push(&s,&child));int *ids=question?collateral:target,count=question?nc:nt;
            int skip=last==m.eos?1:0;REQUIRE(!skip||ids[0]==m.eos);
            for(int j=skip;j<count;j++)REQUIRE(!qwen_push(&s,ids[j]));if(question)fputc(',',f);
            fprintf(f,"{\"name\":\"%s\",\"first\":",question?"collateral":"target");readout(f,&s,red,blue,NULL,NULL);fprintf(f,",\"output\":");generated(f,&s,8);fputc('}',f);REQUIRE(!qwen_stack_rollback(&s,child));
        }
        REQUIRE(!qwen_stack_rollback(&s,outer));char after[65],frozen[65];prefix_digest(&s,n-1,after);program(&p,NULL,frozen);REQUIRE(!strcmp(prefix,after)&&!strcmp(compiled_hash,frozen));
        fprintf(f,"],\"prefix_intact\":true,\"program_unchanged\":true}");REQUIRE(!fflush(f));
    }
    fprintf(f,"],\"program_sha256\":\"%s\",\"common_future_token\":%d,\"seconds\":%.9g}\n",compiled_hash,common,clock_seconds()-started);REQUIRE(!fclose(f));
    free(prompt);free(target);free(collateral);qwen_plane_free(&p);qwen_session_free(&s);qwen_model_free(&m);return 0;
}
