#define _POSIX_C_SOURCE 200809L
#include "qwen.h"
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>
#define REQUIRE(x) do { if (!(x)) { fprintf(stderr,"operand-composition %s:%d %s\n",__FILE__,__LINE__,#x); exit(2); } } while (0)
#define CAPACITY 512
#define CASES 6
#define D 1536
#define H 8960
#define V 151936

static const char *directory;
static int budgets[CASES];
static double seconds(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static void digest(const void *p,size_t n,char out[65]) { unsigned char b[32]; tcm_sha256(p,n,b); tcm_hex(b,32,out); }
static void finite_read(float *out,const float *data,size_t n) { gpu_read(out,data,n*4); for(size_t j=0;j<n;j++) REQUIRE(isfinite(out[j])); }
static void save(const char *name,const void *data,size_t n) { char file[4096]; REQUIRE(snprintf(file,sizeof(file),"%s/%s",directory,name)<(int)sizeof(file)); FILE *f=fopen(file,"wb"); REQUIRE(f&&fwrite(data,1,n,f)==n&&!fclose(f)); }
static void ints(FILE *f,const int *v,int n) { fputc('[',f); for(int j=0;j<n;j++) fprintf(f,"%s%d",j?",":"",v[j]); fputc(']',f); }
static int *tokens(const char *stem,int *count) { char file[4096]; snprintf(file,sizeof(file),"%s/%s.txt",directory,stem); FILE *f=fopen(file,"r"); int *v=malloc(CAPACITY*sizeof(int)),value; REQUIRE(f&&v); *count=0; while(fscanf(f,"%d",&value)==1) { REQUIRE(*count<CAPACITY&&value>=0&&value<V); v[(*count)++]=value; } REQUIRE(feof(f)&&*count>0&&!fclose(f)); return v; }

typedef struct {
    float mlp_norm[D],mlp_write[D],mlp_output[D],receiver[D],q[D],k[256],v[256],attention[D],output[D];
    float mlp_code[64],query_code[128],kv16[512],kv18[512];
} Record;
typedef struct {
    QwenSession *session; int writer,reader,count,mlp_ready,query_ready,full_operands; Record *records;
    int ids[CAPACITY],tops[CAPACITY]; float first_gate[H],first_up[H],boundary_gate[H],boundary_up[H];
    unsigned capture_mask,full_capture_mask;
    uint64_t linear_calls,linear_macs,native_gate_up_calls,compiled_gate_up_calls,mlp_coordinate_calls;
    uint64_t native_qk_calls,compiled_qk_calls,query_coordinate_calls,native_attention_calls,reader_products;
} Run;
typedef struct { const char *name; int in,out; float *data; char hash[65]; } Matrix;
static Matrix matrices[]={
    {"mlp-E.f32",D,64,NULL,{0}}, {"mlp-Lg.f32",64,H,NULL,{0}}, {"mlp-Lu.f32",64,H,NULL,{0}},
    {"query-B.f32",D,128,NULL,{0}}, {"query-F.f32",128,D,NULL,{0}}, {"query-K.f32",128,256,NULL,{0}}
};
static float *mlp_code,*query_code;
static Run *active;
static uint64_t all_linear_calls,all_linear_macs;
static void linear(TcmGpu *gpu,const float *w,int in,int out,const float *x,float *y) {
    gpu_matvec(gpu,w,in,out,x,y); all_linear_calls++; all_linear_macs+=(uint64_t)in*out;
    if(active) { active->linear_calls++; active->linear_macs+=(uint64_t)in*out; }
}
static void composed_matvec(TcmGpu *gpu,const float *w,int in,int out,const float *x,float *y) {
    Run *r=active; const QwenModel *m=r?r->session->model:NULL;
    const QwenLayer *l=r?m->layer+r->session->layer:NULL;
    int selected_mlp=r&&r->session->layer==16&&(w==l->gate||w==l->up);
    int selected_qk=r&&r->session->layer==17&&(w==l->q||w==l->k);
    if(r&&r->session->layer==17&&w==l->q&&r->records) { finite_read(r->records[r->count].receiver,x,D); r->capture_mask|=8; }
    if(selected_mlp&&r->writer) {
        REQUIRE(in==D&&out==H);
        if(!r->mlp_ready) { linear(gpu,matrices[0].data,D,64,x,mlp_code); r->mlp_ready=1; r->mlp_coordinate_calls++; if(r->records) finite_read(r->records[r->count].mlp_code,mlp_code,64); }
        linear(gpu,matrices[w==l->gate?1:2].data,64,H,mlp_code,y); r->compiled_gate_up_calls++;
    } else if(selected_qk&&r->reader) {
        REQUIRE(in==D&&out==(w==l->q?D:256));
        if(!r->query_ready) { linear(gpu,matrices[3].data,D,128,x,query_code); r->query_ready=1; r->query_coordinate_calls++; if(r->records) finite_read(r->records[r->count].query_code,query_code,128); }
        linear(gpu,matrices[w==l->q?4:5].data,128,out,query_code,y); r->compiled_qk_calls++;
    } else {
        linear(gpu,w,in,out,x,y);
        if(selected_mlp) r->native_gate_up_calls++;
        if(selected_qk) r->native_qk_calls++;
    }
}
static void composed_attention(TcmGpu *gpu,float *out,const float *q,const float *keys,const float *values,float *scores,int qheads,int kvheads,int dimension,int capacity,int length) {
    if(active&&active->session->layer==17) { active->native_attention_calls++; active->reader_products+=(uint64_t)qheads*length*dimension*2; }
    gpu_attention(gpu,out,q,keys,values,scores,qheads,kvheads,dimension,capacity,length);
}
#define gpu_matvec composed_matvec
#define gpu_attention composed_attention
#include "runtime/qwen.c"
#undef gpu_matvec
#undef gpu_attention

static void observer(void *context,int position,int layer,QwenPort port,const float *data,size_t n) {
    Run *r=context; REQUIRE(position==r->count&&r->records&&(layer==16||layer==17)); Record *p=r->records+r->count; float *target=NULL; unsigned bit=0;
    if(layer==16) {
        if(port==QWEN_MLP_INPUT) { target=p->mlp_norm; bit=1; }
        else if(port==QWEN_MLP_RETURN) { target=p->mlp_write; bit=2; }
        else if(port==QWEN_OUTPUT) { target=p->mlp_output; bit=4; }
        else if(port==QWEN_MLP_GATE||port==QWEN_MLP_UP) {
            if(!r->count) finite_read(port==QWEN_MLP_GATE?r->first_gate:r->first_up,data,n);
            if(r->full_operands) { target=port==QWEN_MLP_GATE?r->boundary_gate:r->boundary_up; r->full_capture_mask|=port==QWEN_MLP_GATE?1:2; }
        }
    } else {
        if(port==QWEN_QUERY) { target=p->q; bit=16; }
        else if(port==QWEN_KEY) { target=p->k; bit=32; }
        else if(port==QWEN_VALUE) { target=p->v; bit=64; }
        else if(port==QWEN_ATTENTION) { target=p->attention; bit=128; }
        else if(port==QWEN_OUTPUT) { target=p->output; bit=256; }
    }
    if(target) { finite_read(target,data,n); r->capture_mask|=bit; }
}
static void selected_kv(const QwenSession *s,int layer,int position,float *out) {
    size_t n=0;
    for(int kind=0;kind<2;kind++) for(int group=0;group<2;group++) {
        size_t offset=(((size_t)layer*2+group)*s->capacity+position)*128;
        finite_read(out+n,(kind?s->values:s->keys)+offset,128); n+=128;
    }
    REQUIRE(n==512);
}
static void advance(Run *r,int token) {
    REQUIRE(r->count<CAPACITY&&r->session->position==r->count); r->mlp_ready=r->query_ready=0; r->capture_mask=r->full_capture_mask=0; active=r;
    QwenSession *s=r->session;
    REQUIRE(!qwen_begin_token(s,token));
    while(s->layer<s->model->layers) {
        if(r->records&&(s->layer==16||s->layer==17)) { s->observer=observer; s->observer_context=r; s->observe_position=r->count; s->observe_layer=s->layer; }
        else s->observe_layer=-1;
        REQUIRE(!qwen_advance_layer(s));
    }
    REQUIRE(!qwen_finish_token(s));
    REQUIRE(r->mlp_ready==r->writer&&r->query_ready==r->reader);
    if(r->records) { REQUIRE(r->capture_mask==511); selected_kv(s,16,r->count,r->records[r->count].kv16); selected_kv(s,18,r->count,r->records[r->count].kv18); }
    REQUIRE(!r->full_operands||r->full_capture_mask==3); r->full_operands=0;
    r->ids[r->count]=token; r->tops[r->count++]=s->pending; active=NULL;
}
static void fresh(QwenSession *s,const QwenModel *m) { REQUIRE(!qwen_session_init(s,m,CAPACITY)); size_t n=(size_t)m->layers*m->kvheads*CAPACITY*m->head; gpu_zero(s->keys,n*4); gpu_zero(s->values,n*4); }
static void cache_digest(const QwenSession *s,char hash[65]) {
    size_t n=(size_t)s->model->layers*s->model->kvheads*s->position*s->model->head,offset=0; float *data=malloc(2*n*4); REQUIRE(data);
    for(int kind=0;kind<2;kind++) for(int layer=0;layer<s->model->layers;layer++) for(int group=0;group<s->model->kvheads;group++) {
        const float *p=(kind?s->values:s->keys)+((size_t)layer*s->model->kvheads+group)*s->capacity*s->model->head; size_t count=(size_t)s->position*s->model->head; finite_read(data+offset,p,count); offset+=count;
    }
    digest(data,offset*4,hash); free(data);
}
static void boundary(FILE *f,Run *r,const char *name,int turn,const char *stage) {
    char file[256],hash[65]; float *logits=malloc(V*4); REQUIRE(logits); finite_read(logits,r->session->logits,V);
    snprintf(file,sizeof(file),"%s-t%d-%s-logits.f32",name,turn,stage); save(file,logits,V*4); free(logits); cache_digest(r->session,hash);
    snprintf(file,sizeof(file),"%s-t%d-%s-gate.f32",name,turn,stage); save(file,r->boundary_gate,H*4);
    snprintf(file,sizeof(file),"%s-t%d-%s-up.f32",name,turn,stage); save(file,r->boundary_up,H*4);
    fprintf(f,",\"%s\":{\"position\":%d,\"top\":%d,\"valid_global_kv_sha256\":\"%s\"}",stage,r->count,r->session->pending,hash);
}
static void record_arrays(const Run *r,const char *name) {
    struct { const char *name; size_t offset,n; } fields[]={
        {"mlp-norm",offsetof(Record,mlp_norm),D},{"mlp-write",offsetof(Record,mlp_write),D},{"mlp-output",offsetof(Record,mlp_output),D},
        {"receiver",offsetof(Record,receiver),D},{"q",offsetof(Record,q),D},{"k",offsetof(Record,k),256},{"v",offsetof(Record,v),256},
        {"attention",offsetof(Record,attention),D},{"output",offsetof(Record,output),D},{"mlp-code",offsetof(Record,mlp_code),64},
        {"query-code",offsetof(Record,query_code),128},{"kv16",offsetof(Record,kv16),512},{"kv18",offsetof(Record,kv18),512}
    };
    char file[256];
    for(unsigned j=0;j<sizeof(fields)/sizeof(*fields);j++) { float *v=malloc((size_t)r->count*fields[j].n*4); REQUIRE(v); for(int i=0;i<r->count;i++) memcpy(v+(size_t)i*fields[j].n,(const char*)(r->records+i)+fields[j].offset,fields[j].n*4); snprintf(file,sizeof(file),"%s-%s.f32",name,fields[j].name); save(file,v,(size_t)r->count*fields[j].n*4); free(v); }
    snprintf(file,sizeof(file),"%s-tops.i32",name); save(file,r->tops,(size_t)r->count*4);
    snprintf(file,sizeof(file),"%s-first-gate.f32",name); save(file,r->first_gate,H*4);
    snprintf(file,sizeof(file),"%s-first-up.f32",name); save(file,r->first_up,H*4);
}
static void load_programs(void) {
    for(unsigned j=0;j<sizeof(matrices)/sizeof(*matrices);j++) { Matrix *p=matrices+j; char file[4096]; snprintf(file,sizeof(file),"%s/%s",directory,p->name); FILE *f=fopen(file,"rb"); size_t n=(size_t)p->in*p->out; float *data=malloc(n*4); REQUIRE(f&&data&&fread(data,4,n,f)==n&&fgetc(f)==EOF&&!fclose(f)); for(size_t i=0;i<n;i++) REQUIRE(isfinite(data[i])); digest(data,n*4,p->hash); p->data=gpu_weight(data,TCM_F32,n); free(data); }
    mlp_code=gpu_alloc(64*4); query_code=gpu_alloc(128*4); REQUIRE(mlp_code&&query_code);
}
static void check_programs(void) { for(unsigned j=0;j<sizeof(matrices)/sizeof(*matrices);j++) { Matrix *p=matrices+j; size_t n=(size_t)p->in*p->out; float *v=malloc(n*4); char hash[65]; REQUIRE(v); finite_read(v,p->data,n); digest(v,n*4,hash); REQUIRE(!strcmp(hash,p->hash)); free(v); } }
static void export_constants(const QwenModel *m) {
    struct { const char *name; const float *data; int n; } constants[]={
        {"mlp16-post-norm.f32",m->layer[16].post_norm,D},{"attention17-input-norm.f32",m->layer[17].input_norm,D},
        {"attention17-q-bias.f32",m->layer[17].qb,D},{"attention17-k-bias.f32",m->layer[17].kb,256}
    };
    for(unsigned j=0;j<sizeof(constants)/sizeof(*constants);j++) { float *v=malloc((size_t)constants[j].n*4); REQUIRE(v); finite_read(v,constants[j].data,constants[j].n); save(constants[j].name,v,(size_t)constants[j].n*4); free(v); }
}
static void seal_own(FILE *driver) {
    /* Close an immutable copy of the own-only JSON before the diagnostic IDs
     * are consumed. This binds token authority and turn clocks as well as arrays. */
    REQUIRE(!fflush(driver)); char file[4096]; snprintf(file,sizeof(file),"%s/driver.json",directory); FILE *prefix=fopen(file,"rb"); REQUIRE(prefix&&!fseek(prefix,0,SEEK_END)); long length=ftell(prefix); REQUIRE(length>0&&!fseek(prefix,0,SEEK_SET)); unsigned char *data=malloc((size_t)length+3); REQUIRE(data&&fread(data,1,(size_t)length,prefix)==(size_t)length&&!fclose(prefix)); memcpy(data+length,"]}\n",3); save("own-driver.json",data,(size_t)length+3); free(data);
    snprintf(file,sizeof(file),"%s/own-seal.json",directory); FILE *f=fopen(file,"w"); DIR *dir=opendir(directory); REQUIRE(f&&dir); fprintf(f,"{\"phase\":\"all-own-trajectories-before-diagnostic-replay\",\"files\":["); struct dirent *entry; int comma=0;
    while((entry=readdir(dir))) { const char *name=entry->d_name; if(!strstr(name,".f32")&&!strstr(name,".i32")&&strcmp(name,"own-driver.json")) continue; snprintf(file,sizeof(file),"%s/%s",directory,name); struct stat st; REQUIRE(!stat(file,&st)&&S_ISREG(st.st_mode)); void *p=malloc((size_t)st.st_size); FILE *in=fopen(file,"rb"); char hash[65]; REQUIRE(p&&in&&fread(p,1,(size_t)st.st_size,in)==(size_t)st.st_size&&!fclose(in)); digest(p,(size_t)st.st_size,hash); free(p); fprintf(f,"%s{\"name\":\"%s\",\"bytes\":%llu,\"sha256\":\"%s\"}",comma++?",":"",name,(unsigned long long)st.st_size,hash); }
    REQUIRE(!closedir(dir)); fprintf(f,"]}\n"); REQUIRE(!fclose(f)); fprintf(stderr,"sealed all24 own trajectories before same-ID diagnostic replay\n");
}
typedef struct { int start,ingested,skip,prefill,final,budget,answers[32],length,stopped; } Turn;
typedef struct { int ids[CAPACITY],count; Turn turns[2]; } History;
static History joint[CASES];
static void path(FILE *f,const QwenModel *m,int task,int policy,int replay) {
    char name[128],stem[128]; snprintf(name,sizeof(name),"case%d-%s-%s",task,replay?"replay":"own",(const char*[]){"00","10","01","11"}[policy]);
    QwenSession s; fresh(&s,m); Run *r=calloc(1,sizeof(*r)); REQUIRE(r); r->session=&s; r->writer=policy&1; r->reader=!!(policy&2); r->records=calloc(CAPACITY,sizeof(Record)); REQUIRE(r->records); double start=seconds();
    int np,nf; snprintf(stem,sizeof(stem),"case%d-prompt",task); int *prompt=tokens(stem,&np); snprintf(stem,sizeof(stem),"case%d-followup",task); int *follow=tokens(stem,&nf); History *h=joint+task;
    fprintf(f,"{\"name\":\"%s\",\"task\":%d,\"policy\":%d,\"mode\":\"%s\",\"own_generation\":%s,\"writer\":%s,\"reader\":%s,",name,task,policy,replay?"replay":"own",replay?"false":"true",r->writer?"true":"false",r->reader?"true":"false");
    if(replay) fprintf(f,"\"replay_of\":\"case%d-own-11\",",task);
    fprintf(f,"\"turns\":[");
    for(int turn=0;turn<2;turn++) {
        Turn local={0},*t=replay?h->turns+turn:&local; if(turn) fputc(',',f);
        if(!replay) { t->start=r->count; int *input=turn?follow:prompt,n=turn?nf:np; t->skip=turn&&r->ids[r->count-1]==input[0]; REQUIRE(!turn||input[0]==151645); t->ingested=n-t->skip; for(int j=t->skip;j<n;j++) { r->full_operands=j==n-1; advance(r,input[j]); } t->prefill=r->count; t->budget=budgets[task]; }
        else { REQUIRE(r->count==t->start); while(r->count<t->prefill) { r->full_operands=r->count==t->prefill-1; advance(r,h->ids[r->count]); } }
        fprintf(f,"{\"turn\":%d,\"start\":%d,\"ingested\":%d,\"template_skip\":%d,\"budget\":%d",turn,t->start,t->ingested,t->skip,t->budget); boundary(f,r,name,turn,"prefill");
        if(!replay) { for(int j=0;j<t->budget;j++) { int token=s.pending; t->answers[t->length++]=token; t->stopped=token==m->eos||token==m->eos_extra; r->full_operands=t->stopped||j==t->budget-1; advance(r,token); if(t->stopped) break; } t->final=r->count; }
        else { while(r->count<t->final) { r->full_operands=r->count==t->final-1; advance(r,h->ids[r->count]); } }
        fprintf(f,",\"answer_ids\":"); ints(f,t->answers,t->length); fprintf(f,",\"stopped\":%s,\"truncated\":%s",t->stopped?"true":"false",t->stopped?"false":"true"); boundary(f,r,name,turn,"final"); fputc('}',f); REQUIRE(!fflush(f));
        if(!replay&&policy==3) h->turns[turn]=*t;
    }
    if(!replay&&policy==3) { h->count=r->count; memcpy(h->ids,r->ids,(size_t)r->count*4); }
    if(replay) REQUIRE(r->count==h->count&&!memcmp(r->ids,h->ids,(size_t)r->count*4));
    record_arrays(r,name);
    fprintf(f,"],\"consumed_ids\":"); ints(f,r->ids,r->count);
    fprintf(f,",\"count\":%d,\"linear_calls\":%llu,\"linear_macs\":%llu,\"native_gate_up_calls\":%llu,\"compiled_gate_up_calls\":%llu,\"mlp_coordinate_calls\":%llu,\"native_qk_calls\":%llu,\"compiled_qk_calls\":%llu,\"query_coordinate_calls\":%llu,\"native_attention_calls\":%llu,\"reader_products\":%llu,\"session_gpu_bytes\":%zu,\"seconds\":%.9g}",r->count,(unsigned long long)r->linear_calls,(unsigned long long)r->linear_macs,(unsigned long long)r->native_gate_up_calls,(unsigned long long)r->compiled_gate_up_calls,(unsigned long long)r->mlp_coordinate_calls,(unsigned long long)r->native_qk_calls,(unsigned long long)r->compiled_qk_calls,(unsigned long long)r->query_coordinate_calls,(unsigned long long)r->native_attention_calls,(unsigned long long)r->reader_products,s.allocated_bytes,seconds()-start);
    REQUIRE(!fflush(f)); fprintf(stderr,"completed %s: %d advances, %.3fs\n",name,r->count,seconds()-start); free(r->records); free(r); qwen_session_free(&s); free(prompt); free(follow);
}
static void native_sham(const QwenModel *m) {
    int n; int *ids=tokens("case0-prompt",&n); REQUIRE(n>=16); QwenSession a,b; fresh(&a,m); fresh(&b,m); Run *r=calloc(1,sizeof(*r)); float *x=malloc(V*4),*y=malloc(V*4); REQUIRE(r&&x&&y); r->session=&b;
    for(int j=0;j<16;j++) { REQUIRE(!qwen_push(&a,ids[j])); advance(r,ids[j]); finite_read(x,a.logits,V); finite_read(y,b.logits,V); REQUIRE(a.pending==b.pending&&!memcmp(x,y,V*4)); }
    char ha[65],hb[65]; cache_digest(&a,ha); cache_digest(&b,hb); REQUIRE(!strcmp(ha,hb)); save("native-sham-last-logits.f32",x,V*4); free(ids); free(r); free(x); free(y); qwen_session_free(&a); qwen_session_free(&b);
}
int main(int argc,char **argv) {
    REQUIRE(argc==3); directory=argv[2]; int nb; int *b=tokens("budgets",&nb); REQUIRE(nb==CASES); for(int j=0;j<CASES;j++) { REQUIRE(b[j]>0&&b[j]<=32); budgets[j]=b[j]; } free(b);
    TcmModel bundle; QwenModel m; char error[256],file[4096]; double start=seconds(); REQUIRE(!tcm_load(&bundle,argv[1],error,sizeof(error))&&!qwen_model_init(&m,&bundle,error,sizeof(error))); tcm_unload(&bundle); REQUIRE(m.hidden==D&&m.intermediate==H&&m.vocab==V&&m.layers==28&&m.qheads==12&&m.kvheads==2&&m.head==128);
    load_programs(); export_constants(&m); check_programs(); native_sham(&m);
    snprintf(file,sizeof(file),"%s/driver.json",directory); FILE *f=fopen(file,"w"); REQUIRE(f);
    fprintf(f,"{\"schema\":\"tcm-operand-composition-driver-v1\",\"model_digest\":\"%s\",\"model_identity\":\"%s\",\"writer_layer\":16,\"reader_layer\":17,\"capacity\":512,\"memory_policy\":\"full-fp32-all-layers\",\"native_sham_steps\":16,\"native_sham_logits_and_kv_byte_exact\":true,\"record_bytes\":%zu,\"weight_gpu_bytes\":%zu,\"program_gpu_bytes\":%zu,\"epsilon\":%.9g,\"theta\":%.9g,\"paths\":[",m.digest,m.identity,sizeof(Record),m.weight_bytes,(size_t)(64*D+2*H*64+128*D+D*128+256*128+64+128)*4,m.epsilon,m.theta);
    int comma=0; for(int task=0;task<CASES;task++) for(int policy=0;policy<4;policy++) { if(comma++) fputc(',',f); path(f,&m,task,policy,0); }
    seal_own(f);
    for(int task=0;task<CASES;task++) for(int policy=0;policy<3;policy++) { fputc(',',f); path(f,&m,task,policy,1); }
    check_programs(); fprintf(f,"],\"own_sealed_before_replay\":true,\"frozen_programs_byte_unchanged\":true,\"all_linear_calls\":%llu,\"all_linear_macs\":%llu,\"seconds\":%.9g}\n",(unsigned long long)all_linear_calls,(unsigned long long)all_linear_macs,seconds()-start); REQUIRE(!fclose(f));
    for(unsigned j=0;j<sizeof(matrices)/sizeof(*matrices);j++) gpu_free(matrices[j].data); gpu_free(mlp_code); gpu_free(query_code); qwen_model_free(&m); return 0;
}
