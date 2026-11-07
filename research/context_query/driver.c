#define _POSIX_C_SOURCE 200809L
#include "qwen.h"
#include "prototype_attention.h"
#include "distribution.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#define REQUIRE(x) do{if(!(x)){fprintf(stderr,"context-query %d: %s\n",__LINE__,#x);exit(2);}}while(0)
#define CAP 2048
#define D 1536
#define VOCAB 151936
#define R 128
static const char *root;
static QwenModel model;
static QwenQkvProgram programs[4],later;
static int selected;
static uint64_t advances;
static double began;
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static FILE *openfile(const char *name,const char *mode){char path[4096];REQUIRE(snprintf(path,sizeof(path),"%s/%s",root,name)<4096);FILE *f=fopen(path,mode);REQUIRE(f);return f;}
static void hash(const void *p,size_t n,char out[65]){unsigned char h[32];tcm_sha256(p,n,h);tcm_hex(h,32,out);}
static void save(const char *name,const void *p,size_t n){FILE *f=openfile(name,"wb");REQUIRE(fwrite(p,1,n,f)==n&&!fclose(f));}
static float *load(const char *name,size_t n){FILE *f=openfile(name,"rb");float *p=malloc(n*4);REQUIRE(p&&fread(p,4,n,f)==n&&fgetc(f)==EOF&&!fclose(f));for(size_t j=0;j<n;j++)REQUIRE(isfinite(p[j]));return p;}
static int tokens(const char *name,int *v){FILE *f=openfile(name,"r");int n=0,x;while(fscanf(f,"%d",&x)==1){REQUIRE(n<CAP&&x>=0&&x<VOCAB);v[n++]=x;}REQUIRE(feof(f)&&n&&!fclose(f));return n;}
static void read_gpu(float *h,const float *p,size_t n){gpu_read(h,p,n*4);for(size_t j=0;j<n;j++)REQUIRE(isfinite(h[j]));}
typedef struct {QwenSession s;QwenPrototypeMemory mem[2];float *norm18;int parity,count;float *expected[4];uint64_t reads,transfer;} Run;
static int cache(void *ctx,int layer,int pos,const float *q,const float *k,const float *v,float *out,float *scores){
    (void)scores;Run *r=ctx;REQUIRE(layer==17||layer==18);float hq[D],hk[256],hv[256],a[D];read_gpu(hq,q,D);read_gpu(hk,k,256);read_gpu(hv,v,256);
    REQUIRE(!qwen_prototype_append(r->mem+layer-17,(unsigned)pos,hk,hv)&&!qwen_prototype_read(r->mem+layer-17,hq,a));gpu_write(out,a,sizeof(a));r->reads++;r->transfer+=(D*2+512)*4;return 0;
}
static int guard(void *ctx,int restore,int pos,uint64_t sid,uint64_t epoch,uint64_t lineage){
    (void)sid;(void)epoch;(void)lineage;Run *r=ctx;if(restore)return -1;
    for(int j=0;j<2;j++)if((r->s.cache_mask&(UINT64_C(1)<<(17+j)))&&r->mem[j].next!=(uint64_t)pos)return -1;return 0;
}
static void rewind_cache(void *ctx,int pos,uint64_t sid,uint64_t epoch){(void)sid;(void)epoch;REQUIRE(!pos);Run *r=ctx;memset(r->mem,0,sizeof(r->mem));}
static void observe(void *ctx,int pos,int layer,QwenPort port,const float *data,size_t n){
    Run *r=ctx;REQUIRE(pos==r->count);
    if(r->parity&&layer==17){int j=port==QWEN_QUERY?0:port==QWEN_KEY?1:port==QWEN_VALUE?2:port==QWEN_ATTENTION?3:-1;if(j>=0){float a[D];read_gpu(a,data,n);REQUIRE(!memcmp(a,r->expected[j]+(size_t)pos*n,n*4));}}
    if(layer==18&&port==QWEN_QUERY&&r->norm18)read_gpu(r->norm18+(size_t)pos*D,r->s.norm,D);
}
static void init(Run *r,int policy,int capture){
    memset(r,0,sizeof(*r));uint64_t mask=policy<0?0:UINT64_C(1)<<17;if(policy==5)mask|=UINT64_C(1)<<18;
    REQUIRE(!qwen_session_init_layout(&r->s,&model,CAP,mask,mask?cache:NULL,mask?rewind_cache:NULL,r));r->s.cache_guard=mask?guard:NULL;
    if(policy>=0){int image=policy<4?policy:selected;REQUIRE(!qwen_qkv_bind(&r->s,17,programs+image));if(policy>=4)REQUIRE(!qwen_qkv_bind(&r->s,18,&later));}
    if(capture){r->norm18=calloc((size_t)CAP*D,4);REQUIRE(r->norm18);r->s.observer=observe;r->s.observer_context=r;r->s.observe_layer=18;}
}
static void push(Run *r,int token){REQUIRE(advances<36000&&now()-began<780&&r->count<CAP);r->s.observe_position=r->count;REQUIRE(!qwen_push(&r->s,token));r->count++;advances++;}
static void finish(Run *r){qwen_session_free(&r->s);free(r->norm18);}
static void integrate(void){
    float *b=load("program128-basis.f32",R*D),*k=load("program128-k.f32",R*256);
    for(int j=0;j<4;j++){char name[80];snprintf(name,sizeof(name),"query-proposal-%d.f32",j);float *q=load(name,R*D);REQUIRE(!qwen_qkv_program_init(programs+j,&model,17,R,3,b,q,k,NULL));free(q);}free(b);free(k);
    /* Replay the authenticated source's actual token clocks through public APIs. */
    for(int c=0;c<3;c++){
        char name[128];int ids[CAP];snprintf(name,sizeof(name),"parity%d.txt",c);int n=tokens(name,ids);Run *r=calloc(1,sizeof(*r));REQUIRE(r);init(r,0,0);r->parity=1;r->s.observer=observe;r->s.observer_context=r;r->s.observe_layer=17;
        const char *fields[]={"q","k","v","attention"};int widths[]={D,256,256,D};
        for(int j=0;j<4;j++){snprintf(name,sizeof(name),"parity%d-%s.f32",c,fields[j]);r->expected[j]=load(name,(size_t)n*widths[j]);}
        snprintf(name,sizeof(name),"parity%d-tops.i32",c);FILE *f=openfile(name,"rb");int *tops=malloc(n*4);REQUIRE(tops&&fread(tops,4,n,f)==(size_t)n&&fgetc(f)==EOF&&!fclose(f));
        for(int t=0;t<n;t++){push(r,ids[t]);REQUIRE(r->s.pending==tops[t]);}for(int j=0;j<4;j++)free(r->expected[j]);free(tops);finish(r);free(r);
    }
    int ids[CAP],uniq[CAP],nu=0,n=tokens("source0.txt",ids);REQUIRE(n>=16);for(int j=0;j<16;j++){int seen=0;for(int k0=0;k0<nu;k0++)seen|=uniq[k0]==ids[j];if(!seen)uniq[nu++]=ids[j];}
    for(int j=0;j<nu;j++)for(int k0=j+1;k0<nu;k0++)if(uniq[j]>uniq[k0]){int t=uniq[j];uniq[j]=uniq[k0];uniq[k0]=t;}
    QwenLexicalProgram lex={0};REQUIRE(!qwen_lexical_compile(&lex,&model,uniq,(size_t)nu));
    char lexpath[4096];snprintf(lexpath,sizeof(lexpath),"%s/lexical-subset.bin",root);REQUIRE(!qwen_lexical_save(&lex,lexpath));qwen_lexical_free(&lex);REQUIRE(!qwen_lexical_load(&lex,&model,lexpath));
    Run a,b0;init(&a,-1,0);init(&b0,-1,0);REQUIRE(!qwen_lexical_bind(&b0.s,&lex));float *ha=malloc(VOCAB*4),*hb=malloc(VOCAB*4);REQUIRE(ha&&hb);
    for(int t=0;t<16;t++){push(&a,ids[t]);push(&b0,ids[t]);read_gpu(ha,a.s.logits,VOCAB);read_gpu(hb,b0.s.logits,VOCAB);REQUIRE(!memcmp(ha,hb,VOCAB*4));}
    char text[1024];int size=snprintf(text,sizeof(text),"{\"source_clock_reads_exact\":420,\"source_full_logits_top_ids_exact\":420,\"lexical_replayed_steps\":16,\"lexical_covered_ids\":%d,\"lexical_full_logits_exact\":true,\"lexical_compile_linear_macs\":%llu,\"lexical_host_bytes\":%zu,\"lexical_transfer_bytes\":%llu,\"model_digest\":\"%s\",\"model_identity\":\"%s\"}\n",nu,(unsigned long long)((uint64_t)nu*D*(D+512)),lex.host_bytes,(unsigned long long)b0.s.operand_work.lexical_transfer_bytes,model.digest,model.identity);save("integration.json",text,(size_t)size);
    free(ha);free(hb);finish(&a);finish(&b0);qwen_lexical_free(&lex);
}
typedef struct {int clock,turn,phase,boundary;} Sample;
static void ints(FILE *f,int *ids,int n){fputc('[',f);for(int j=0;j<n;j++)fprintf(f,"%s%d",j?",":"",ids[j]);fputc(']',f);}
static void run_pair(const char *name,int policy,const char *promptfile,const char *followfile,int source){
    int prompt[CAP],follow[CAP],ids[CAP],pn=tokens(promptfile,prompt),fn=tokens(followfile,follow),n=0,phase[CAP],turn[CAP],answer[2][32],an[2]={0},stopped[2]={0};REQUIRE(pn+fn+64<CAP);
    for(int j=0;j<CAP;j++)turn[j]=-1;Sample samples[256];int ns=0;float *logits=malloc(VOCAB*4),*candidate=malloc(VOCAB*4);REQUIRE(logits&&candidate);FILE *cf=openfile("candidate-logits.tmp","w+b");Run *r=calloc(1,sizeof(*r));REQUIRE(r);init(r,policy,source);double start=now();
    for(int t=0;t<2;t++){
        int *input=t?follow:prompt,count=t?fn:pn,skip=t&&n&&ids[n-1]==input[0];
        for(int j=skip;j<count;j++){ids[n]=input[j];phase[n]=t?2:0;push(r,ids[n++]);if(j%32==0&&j!=count-1){samples[ns++]=(Sample){n-1,t,0,0};read_gpu(logits,r->s.logits,VOCAB);REQUIRE(fwrite(logits,4,VOCAB,cf)==VOCAB);}}
        for(int j=0;j<32;j++){
            int token=r->s.pending;turn[n-1]=t;answer[t][an[t]++]=token;stopped[t]=token==model.eos||token==model.eos_extra;
            samples[ns++]=(Sample){n-1,t,1,j==0?1:stopped[t]||j==31?2:0};read_gpu(logits,r->s.logits,VOCAB);REQUIRE(fwrite(logits,4,VOCAB,cf)==VOCAB);
            ids[n]=token;phase[n]=t?3:1;push(r,ids[n++]);if(stopped[t])break;
        }
        if(policy>=0){char file[160];snprintf(file,sizeof(file),"%s-turn%d-memory17.bin",name,t);save(file,r->mem,sizeof(r->mem[0]));if(policy==5){snprintf(file,sizeof(file),"%s-turn%d-memory18.bin",name,t);save(file,r->mem+1,sizeof(r->mem[1]));}}
    }
    REQUIRE(n==r->count&&ns<256&&!fflush(cf));QwenOperandWork work=r->s.operand_work;uint64_t reads=r->reads,transfer=r->transfer;size_t allocated=r->s.allocated_bytes;double candidate_s=now()-start;char file[160];
    if(source){snprintf(file,sizeof(file),"%s-norm18.f32",name);save(file,r->norm18,(size_t)n*D*4);}
    char tracehash[65];hash(ids,n*4,tracehash);finish(r);free(r); /* Candidate owns all tokens before a native shadow exists. */
    r=calloc(1,sizeof(*r));REQUIRE(r);init(r,-1,0);FILE *nf=openfile("native-logits.tmp","wb");snprintf(file,sizeof(file),"%s-metrics.json",name);FILE *mf=openfile(file,"w");REQUIRE(!fseek(cf,0,SEEK_SET));int cursor=0;start=now();fprintf(mf,"{\"samples\":[");
    for(int j=0;j<n;j++){
        push(r,ids[j]);if(cursor>=ns||samples[cursor].clock!=j)continue;Sample *s=samples+cursor;read_gpu(logits,r->s.logits,VOCAB);REQUIRE(fread(candidate,4,VOCAB,cf)==VOCAB&&fwrite(logits,4,VOCAB,nf)==VOCAB);Distribution metric;REQUIRE(!distribution(logits,candidate,VOCAB,&metric));char nh[65],ch[65];hash(logits,VOCAB*4,nh);hash(candidate,VOCAB*4,ch);
        fprintf(mf,"%s{\"clock\":%d,\"turn\":%d,\"future\":%s,\"boundary\":%d,\"kl\":%.17g,\"l2\":%.17g,\"native_top\":%d,\"candidate_top\":%d,\"native_sha256\":\"%s\",\"candidate_sha256\":\"%s\"}",cursor?",":"",j,s->turn,s->phase?"true":"false",s->boundary,metric.kl,metric.l2,metric.native_top,metric.candidate_top,nh,ch);
        if(s->phase)REQUIRE(j+1<n&&metric.candidate_top==ids[j+1]);if(s->boundary){snprintf(file,sizeof(file),"%s-turn%d-b%d-native.f32",name,s->turn,s->boundary);save(file,logits,VOCAB*4);snprintf(file,sizeof(file),"%s-turn%d-b%d-candidate.f32",name,s->turn,s->boundary);save(file,candidate,VOCAB*4);}cursor++;
    }
    REQUIRE(cursor==ns&&fgetc(cf)==EOF&&!fclose(cf)&&!fclose(nf));fprintf(mf,"]}\n");REQUIRE(!fclose(mf));double native_s=now()-start;finish(r);free(r);
    snprintf(file,sizeof(file),"%s.json",name);FILE *f=openfile(file,"w");fprintf(f,"{\"name\":\"%s\",\"policy\":%d,\"source\":%s,\"count\":%d,\"prompt_length\":%d,\"trace_sha256\":\"%s\",\"candidate_complete_before_native_shadow\":true,\"consumed_ids\":",name,policy,source?"true":"false",n,pn,tracehash);ints(f,ids,n);fprintf(f,",\"input_phases\":");ints(f,phase,n);fprintf(f,",\"prediction_turn\":");ints(f,turn,n);fprintf(f,",\"answers\":[");ints(f,answer[0],an[0]);fputc(',',f);ints(f,answer[1],an[1]);
    fprintf(f,"],\"stopped\":[%s,%s],\"candidate_seconds\":%.9g,\"shadow_seconds\":%.9g,\"allocated_session_bytes\":%zu,\"prototype_reads\":%llu,\"prototype_transfer_bytes\":%llu,\"operand_work\":{\"applications\":%llu,\"code_calls\":%llu,\"compiled_calls\":%llu,\"native_calls\":%llu,\"linear_macs\":%llu}}\n",stopped[0]?"true":"false",stopped[1]?"true":"false",candidate_s,native_s,allocated,(unsigned long long)reads,(unsigned long long)transfer,(unsigned long long)work.applications,(unsigned long long)work.code_calls,(unsigned long long)work.compiled_calls,(unsigned long long)work.native_calls,(unsigned long long)work.linear_macs);REQUIRE(!fclose(f));free(logits);free(candidate);
}
static void compile_later(void){
    float *b=load("basis18.f32",R*D),*q=malloc(R*D*4),*k=malloc(R*256*4),*x=gpu_alloc(D*4),*y=gpu_alloc(D*4),host[D];REQUIRE(q&&k);
    for(int j=0;j<R;j++){gpu_write(x,b+(size_t)j*D,D*4);gpu_matvec(model.gpu,model.layer[18].q,D,D,x,y);read_gpu(host,y,D);for(int d=0;d<D;d++)q[d*R+j]=host[d];gpu_matvec(model.gpu,model.layer[18].k,D,256,x,y);read_gpu(host,y,256);for(int d=0;d<256;d++)k[d*R+j]=host[d];}
    REQUIRE(!qwen_qkv_program_init(&later,&model,18,R,3,b,q,k,NULL));save("image18-q.f32",q,R*D*4);save("image18-k.f32",k,R*256*4);char text[300];int n=snprintf(text,sizeof(text),"{\"layer\":18,\"rank\":128,\"compile_linear_calls\":256,\"compile_linear_macs\":%llu,\"fingerprint\":\"%s\"}\n",(unsigned long long)((uint64_t)R*D*(D+256)),later.fingerprint);save("compile18.json",text,(size_t)n);free(b);free(q);free(k);gpu_free(x);gpu_free(y);
}
int main(int argc,char **argv){
    REQUIRE(argc==3);root=argv[2];began=now();TcmModel bundle;char error[256];REQUIRE(!tcm_load(&bundle,argv[1],error,sizeof(error))&&!qwen_model_init(&model,&bundle,error,sizeof(error)));tcm_unload(&bundle);integrate();puts("READY");fflush(stdout);
    char command[32],name[96],prompt[96],follow[96];int policy,source;
    while(scanf("%31s",command)==1){
        if(!strcmp(command,"RUN")){REQUIRE(scanf("%95s %d %95s %95s %d",name,&policy,prompt,follow,&source)==5&&policy>=-1&&policy<=5);run_pair(name,policy,prompt,follow,source);printf("DONE %s\n",name);}
        else if(!strcmp(command,"SELECT")){REQUIRE(scanf("%d",&selected)==1&&selected>=0&&selected<4);puts("SELECTED");}
        else if(!strcmp(command,"COMPILE18")){compile_later();puts("COMPILED18");}
        else if(!strcmp(command,"QUIT"))break;else REQUIRE(0);fflush(stdout);
    }
    char text[256];int n=snprintf(text,sizeof(text),"{\"advances\":%llu,\"seconds\":%.9g,\"gpu_peak_bytes\":%zu,\"selected\":%d}\n",(unsigned long long)advances,now()-began,gpu_peak_bytes(),selected);save("execution.json",text,(size_t)n);for(int j=0;j<4;j++)qwen_qkv_program_free(programs+j);qwen_qkv_program_free(&later);qwen_model_free(&model);return 0;
}
