#define _POSIX_C_SOURCE 200809L
#include "qwen.h"
#include "prototype_attention.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define REQUIRE(x) do{if(!(x)){fprintf(stderr,"qk-cut line%d: %s\n",__LINE__,#x);exit(2);}}while(0)
#define CAP 2048
#define D 1536
#define K 256
#define VOCAB 151936
static const char *root;
static QwenModel model;
static QwenQkvProgram program17,program18;
static double began;
static uint64_t advances,diagnostic_calls,diagnostic_macs,hash_transfer_bytes;
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static FILE *file(const char *name,const char *mode){char path[4096];REQUIRE(snprintf(path,sizeof(path),"%s/%s",root,name)<4096);FILE *f=fopen(path,mode);REQUIRE(f);return f;}
static void save(const char *name,const void *p,size_t n){FILE *f=file(name,"wb");REQUIRE(fwrite(p,1,n,f)==n&&!fclose(f));}
static float *load(const char *name,size_t n){float *p=malloc(n*4);FILE *f=file(name,"rb");REQUIRE(p&&fread(p,4,n,f)==n&&fgetc(f)==EOF&&!fclose(f));for(size_t i=0;i<n;i++)REQUIRE(isfinite(p[i]));return p;}
static int tokens(const char *name,int *ids){FILE *f=file(name,"r");int n=0,v;while(fscanf(f,"%d",&v)==1){REQUIRE(n<CAP&&v>=0&&v<VOCAB);ids[n++]=v;}REQUIRE(n&&!fclose(f));return n;}
static void host(float *out,const float *p,size_t n){gpu_read(out,p,n*4);for(size_t i=0;i<n;i++)REQUIRE(isfinite(out[i]));}
static void hash(const void *p,size_t n,char h[65]){unsigned char digest[32];tcm_sha256(p,n,digest);tcm_hex(digest,32,h);}
static void ints(FILE *f,const int *ids,int n){fputc('[',f);for(int i=0;i<n;i++)fprintf(f,"%s%d",i?",":"",ids[i]);fputc(']',f);}
typedef struct {float norm[D],qc[D],qn[D],used[D],kc[K],kn[K],v[K],computed[D],consumed[D];} Reader;
typedef struct {float input[D],attention[D],attention_write[D],receiver[D],mlp_input[D],mlp_write[D],output[D];} Layer;
typedef struct {
    QwenSession s;QwenPrototypeMemory memory,parent_memory;QwenCheckpoint parent;
    float *keys[2],*values,*permuted,*native_q,*native_k,*parent_logits,*norms,*queries;
    int history,next,cut,arm,has_parent,recording;char name[128],parent_hash[65];
    Reader reader;Layer layers[10];float *probabilities;uint64_t reads17,reads18;
} Run;
static int guard(void *ctx,int restore,int pos,uint64_t sid,uint64_t epoch,uint64_t lineage){
    Run *r=ctx;if(sid!=r->s.session_id||epoch!=r->s.epoch)return -1;
    if(restore)return !r->has_parent||pos!=r->parent.position||lineage!=r->parent.lineage_id?-1:0;
    return r->memory.next!=(uint64_t)pos||r->next!=pos?-1:0;
}
static void rewind_cache(void *ctx,int pos,uint64_t sid,uint64_t epoch){Run *r=ctx;REQUIRE(r->has_parent&&pos==r->parent.position&&sid==r->parent.session_id&&epoch==r->parent.epoch);r->memory=r->parent_memory;r->next=pos;}
static void snapshot_gpu(const char *name,const float *p,size_t n){float *h=malloc(n*4);REQUIRE(h);host(h,p,n);save(name,h,n*4);free(h);}
static int cache(void *ctx,int layer,int pos,const float *q,const float *k,const float *v,float *out,float *scores){
    Run *r=ctx;
    if(layer==17){float hq[D],hk[K],hv[K],a[D];host(hq,q,D);host(hk,k,K);host(hv,v,K);REQUIRE(!qwen_prototype_append(&r->memory,(uint64_t)pos,hk,hv)&&!qwen_prototype_read(&r->memory,hq,a));gpu_write(out,a,D*4);r->reads17++;return 0;}
    REQUIRE(layer==18&&pos==r->next);const QwenLayer *l=model.layer+18;
    /* Paid diagnostic native operands, from this invocation's actual receiver. */
    gpu_matvec(model.gpu,l->q,D,D,r->s.norm,r->native_q);gpu_bias(r->native_q,l->qb,D);gpu_rope(r->native_q,12,128,pos,model.theta);
    gpu_matvec(model.gpu,l->k,D,K,r->s.norm,r->native_k);gpu_bias(r->native_k,l->kb,K);gpu_rope(r->native_k,2,128,pos,model.theta);diagnostic_calls+=2;diagnostic_macs+=(uint64_t)D*(D+K);
    gpu_cache(r->keys[0],k,2,128,CAP,pos);gpu_cache(r->keys[1],r->native_k,2,128,CAP,pos);gpu_cache(r->values,v,2,128,CAP,pos);r->next++;
    host(r->norms+(size_t)pos*D,r->s.norm,D);host(r->queries+(size_t)pos*D,r->native_q,D);
    int qt=r->history,kt=r->history,arm=pos==r->cut?r->arm:-1;
    if(arm>=2&&arm<=5){qt=(arm==3||arm==5);kt=(arm==4||arm==5);}if(arm==6||arm==8)qt=kt=1;
    const float *used=qt?r->native_q:q,*values=r->values;
    if(arm==6){REQUIRE(pos>=16);gpu_write(r->native_q,r->queries+(size_t)(pos-16)*D,D*4);used=r->native_q;}
    if(arm==8){for(int g=0;g<2;g++)for(int t=0;t<=pos;t++)gpu_copy(r->permuted+((size_t)g*CAP+t)*128,r->values+((size_t)g*CAP+pos-t)*128,128*4);values=r->permuted;}
    gpu_attention(model.gpu,out,used,r->keys[kt],values,scores,12,2,128,CAP,pos+1);r->reads18++;
    if(pos==r->cut&&r->recording){
        Reader *a=&r->reader;host(a->norm,r->s.norm,D);host(a->qc,q,D);memcpy(a->qn,r->queries+(size_t)pos*D,D*4);host(a->used,used,D);host(a->kc,k,K);host(a->kn,r->native_k,K);host(a->v,v,K);host(a->computed,out,D);host(r->probabilities,scores,(size_t)12*(pos+1));
        if(arm==7)gpu_zero(out,D*4);if(arm==1)gpu_write(out,a->computed,D*4);host(a->consumed,out,D);
    }
    return 0;
}
static void observe(void *ctx,int pos,int layer,QwenPort port,const float *data,size_t n){
    Run *r=ctx;if(pos!=r->cut||layer<18||layer>27||!r->recording)return;Layer *a=r->layers+layer-18;float *target=NULL;
    if(port==QWEN_INPUT)target=a->input;else if(port==QWEN_ATTENTION)target=a->attention;else if(port==QWEN_ATTENTION_WRITE)target=a->attention_write;else if(port==QWEN_RAW_RECEIVER)target=a->receiver;else if(port==QWEN_MLP_INPUT)target=a->mlp_input;else if(port==QWEN_MLP_RETURN)target=a->mlp_write;else if(port==QWEN_OUTPUT)target=a->output;if(target){REQUIRE(n==D);host(target,data,D);}
}
static void push(Run *r,int token){
    REQUIRE(advances<8000&&now()-began<360);REQUIRE(!qwen_begin_token(&r->s,token));r->s.observe_position=r->cut;
    while(r->s.layer<28){r->s.observe_layer=r->s.layer;REQUIRE(!qwen_advance_layer(&r->s));}REQUIRE(!qwen_finish_token(&r->s));advances++;
}
static void init(Run *r,int history,int cut){
    memset(r,0,sizeof(*r));r->history=history;r->cut=cut;r->arm=-1;REQUIRE(!qwen_session_init_layout(&r->s,&model,CAP,(UINT64_C(1)<<17)|(UINT64_C(1)<<18),cache,rewind_cache,r));r->s.cache_guard=guard;REQUIRE(!qwen_qkv_bind(&r->s,17,&program17)&&!qwen_qkv_bind(&r->s,18,&program18));
    for(int j=0;j<2;j++)r->keys[j]=gpu_alloc((size_t)CAP*K*4);r->values=gpu_alloc((size_t)CAP*K*4);r->permuted=gpu_alloc((size_t)CAP*K*4);r->native_q=gpu_alloc(D*4);r->native_k=gpu_alloc(K*4);r->parent_logits=malloc(VOCAB*4);r->norms=malloc((size_t)CAP*D*4);r->queries=malloc((size_t)CAP*D*4);r->probabilities=malloc((size_t)12*(cut+1)*4);REQUIRE(r->parent_logits&&r->norms&&r->queries&&r->probabilities);r->s.observer=observe;r->s.observer_context=r;
}
static void release(Run *r){qwen_session_free(&r->s);for(int j=0;j<2;j++)gpu_free(r->keys[j]);gpu_free(r->values);gpu_free(r->permuted);gpu_free(r->native_q);gpu_free(r->native_k);free(r->parent_logits);free(r->norms);free(r->queries);free(r->probabilities);}
static void memory_hash(Run *r,int count,int logits,char h[65]){
    size_t floats=(size_t)(r->s.dense_layers*2+3)*K*count+(logits?VOCAB:0),offset=0;unsigned char *bytes=malloc(floats*4+sizeof(r->memory));REQUIRE(bytes);float *p=(float*)bytes;
    for(int kind=0;kind<2;kind++)for(int layer=0;layer<28;layer++)if(r->s.cache_slots[layer]>=0)for(int g=0;g<2;g++){
        const float *src=(kind?r->s.values:r->s.keys)+((size_t)r->s.cache_slots[layer]*2+g)*CAP*128;host(p+offset,src,(size_t)count*128);offset+=(size_t)count*128;
    }
    for(int stream=0;stream<3;stream++)for(int g=0;g<2;g++){const float *src=(stream<2?r->keys[stream]:r->values)+(size_t)g*CAP*128;host(p+offset,src,(size_t)count*128);offset+=(size_t)count*128;}
    if(logits){host(p+offset,r->s.logits,VOCAB);offset+=VOCAB;}REQUIRE(offset==floats);memcpy(bytes+offset*4,&r->memory,sizeof(r->memory));hash(bytes,floats*4+sizeof(r->memory),h);hash_transfer_bytes+=floats*4;free(bytes);
}
static void restore_parent(Run *r){REQUIRE(!qwen_restore(&r->s,&r->parent));gpu_write(r->s.logits,r->parent_logits,VOCAB*4);REQUIRE(r->memory.next==(uint64_t)r->parent.position&&r->next==r->parent.position&&r->s.lineage_id==r->parent.lineage_id&&r->s.pending==r->parent.pending);char h[65];memory_hash(r,r->parent.position,1,h);REQUIRE(!strcmp(h,r->parent_hash));}
static void boundary(Run *r,const char *suffix){char name[180];snprintf(name,sizeof(name),"%s-%s.f32",r->name,suffix);snapshot_gpu(name,r->s.logits,VOCAB);}
static void expected_logits(Run *r,int context,int history,const char *suffix){char name[160];snprintf(name,sizeof(name),"case%d-h%d-%s.f32",context,history,suffix);float *expected=load(name,VOCAB),*actual=malloc(VOCAB*4);REQUIRE(actual);host(actual,r->s.logits,VOCAB);REQUIRE(!memcmp(expected,actual,VOCAB*4));free(expected);free(actual);}
static void save_cache(Run *r,int context,int history){char name[160];size_t n=(size_t)(r->cut+1)*K;float *h=malloc(n*4);REQUIRE(h);for(int j=0;j<3;j++){float *src=j<2?r->keys[j]:r->values;for(int g=0;g<2;g++)host(h+(size_t)g*(r->cut+1)*128,src+(size_t)g*CAP*128,(size_t)(r->cut+1)*128);snprintf(name,sizeof(name),"case%d-h%d-%s.f32",context,history,j==0?"keys-compiled":j==1?"keys-native":"values");save(name,h,n*4);}free(h);snprintf(name,sizeof(name),"case%d-h%d-norms.f32",context,history);save(name,r->norms,(size_t)(r->cut+1)*D*4);snprintf(name,sizeof(name),"case%d-h%d-native-queries.f32",context,history);save(name,r->queries,(size_t)(r->cut+1)*D*4);}
static void branch(FILE *out,Run *r,int context,int history,int arm,int cut_token,const int *parent_ids,int parent_n,const int *follow,int fn){
    int alias=history?5:2;if(arm==alias){fprintf(out,"{\"context\":%d,\"history\":%d,\"arm\":%d,\"alias_of\":0}",context,history,arm);return;}
    restore_parent(r);r->arm=arm;r->recording=1;memset(&r->reader,0,sizeof(r->reader));memset(r->layers,0,sizeof(r->layers));snprintf(r->name,sizeof(r->name),"case%d-h%d-arm%d",context,history,arm);uint64_t before=advances;push(r,cut_token);boundary(r,"cut-logits");
    char name[180];snprintf(name,sizeof(name),"%s-reader.f32",r->name);save(name,&r->reader,sizeof(r->reader));snprintf(name,sizeof(name),"%s-layers.f32",r->name);save(name,r->layers,sizeof(r->layers));snprintf(name,sizeof(name),"%s-probabilities.f32",r->name);save(name,r->probabilities,(size_t)12*(r->cut+1)*4);if(!arm)save_cache(r,context,history);
    r->recording=0;r->arm=-1;int ids[CAP],n=parent_n;memcpy(ids,parent_ids,(size_t)n*4);ids[n++]=cut_token;int answer[2][32],length[2]={0},stopped[2]={0};
    for(int t=0;t<2;t++){
        if(t){int skip=n&&ids[n-1]==follow[0];for(int i=skip;i<fn;i++){ids[n++]=follow[i];push(r,follow[i]);}boundary(r,"followup-logits");if(!history&&!arm)expected_logits(r,context,0,"turn1-b1");}
        int budget=t?32:24;for(int i=0;i<budget;i++){int token=r->s.pending;answer[t][length[t]++]=token;stopped[t]=token==model.eos||token==model.eos_extra;if(stopped[t]||i==budget-1){boundary(r,t?"followup-last-logits":"answer-last-logits");if(!history&&!arm)expected_logits(r,context,0,t?"turn1-b2":"turn0-b2");}ids[n++]=token;REQUIRE(n<CAP);push(r,token);if(stopped[t])break;}
    }
    if(!history&&!arm){snprintf(name,sizeof(name),"case%d-original.txt",context);int old[CAP],on=tokens(name,old);REQUIRE(on==n&&!memcmp(old,ids,(size_t)n*4));}
    char final_hash[65],ids_hash[65];memory_hash(r,r->s.position,0,final_hash);hash(ids,n*4,ids_hash);
    fprintf(out,"{\"context\":%d,\"history\":%d,\"arm\":%d,\"name\":\"%s\",\"cut_position\":%d,\"parent_position\":%d,\"parent_hash\":\"%s\",\"parent_lineage\":%llu,\"parent_replayed_exact\":true,\"cut_token\":%d,\"answers\":[",context,history,arm,r->name,r->cut,r->parent.position,r->parent_hash,(unsigned long long)r->parent.lineage_id,cut_token);ints(out,answer[0],length[0]);fputc(',',out);ints(out,answer[1],length[1]);fprintf(out,"],\"stopped\":[%s,%s],\"consumed_ids\":",stopped[0]?"true":"false",stopped[1]?"true":"false");ints(out,ids,n);fprintf(out,",\"consumed_ids_sha256\":\"%s\",\"final_cache_sha256\":\"%s\",\"physical_advances\":%llu,\"intervention_scope\":\"one attention18 read; original history policy resumes immediately\"}",ids_hash,final_hash,(unsigned long long)(advances-before));
}
static void case_run(FILE *out,int context,int history){
    char name[160];int ids[CAP],token[CAP],follow[CAP];snprintf(name,sizeof(name),"case%d-parent.txt",context);int n=tokens(name,ids);snprintf(name,sizeof(name),"case%d-token.txt",context);REQUIRE(tokens(name,token)==1);snprintf(name,sizeof(name),"case%d-follow.txt",context);int fn=tokens(name,follow);REQUIRE(n>context&&n+fn+57<CAP);
    Run *r=calloc(1,sizeof(*r));REQUIRE(r);init(r,history,n);for(int i=0;i<n;i++){push(r,ids[i]);if(i==context-1)expected_logits(r,context,history,"turn0-b1");}
    REQUIRE(!qwen_checkpoint(&r->s,&r->parent));r->parent_memory=r->memory;r->has_parent=1;host(r->parent_logits,r->s.logits,VOCAB);memory_hash(r,n,1,r->parent_hash);snprintf(name,sizeof(name),"case%d-h%d-parent-memory.bin",context,history);save(name,&r->parent_memory,sizeof(r->parent_memory));
    for(int arm=0;arm<9;arm++){if(context!=128||history||arm)fputc(',',out);branch(out,r,context,history,arm,token[0],ids,n,follow,fn);REQUIRE(!fflush(out));printf("BRANCH %d %d %d\n",context,history,arm);fflush(stdout);}
    release(r);free(r);
}
static void programs(void){float *b=load("basis17.f32",128*D),*q=load("query17.f32",128*D),*k=load("key17.f32",128*K);REQUIRE(!qwen_qkv_program_init(&program17,&model,17,128,3,b,q,k,NULL));free(b);free(q);free(k);b=load("basis18.f32",128*D);q=load("query18.f32",128*D);k=load("key18.f32",128*K);REQUIRE(!qwen_qkv_program_init(&program18,&model,18,128,3,b,q,k,NULL));free(b);free(q);free(k);}
int main(int argc,char **argv){
    REQUIRE(argc==3);root=argv[2];began=now();TcmModel bundle;char error[256];REQUIRE(!tcm_load(&bundle,argv[1],error,sizeof(error))&&!qwen_model_init(&model,&bundle,error,sizeof(error)));tcm_unload(&bundle);programs();FILE *f=file("driver.json","w");fprintf(f,"{\"schema\":\"qk18-cut-driver-v1\",\"model_digest\":\"%s\",\"model_identity\":\"%s\",\"capacity\":2048,\"rows\":[",model.digest,model.identity);
    for(int context=128;context<=1024;context+=896)for(int history=0;history<2;history++)case_run(f,context,history);
    fprintf(f,"],\"physical_advances\":%llu,\"diagnostic_native_qk_calls\":%llu,\"diagnostic_native_qk_macs\":%llu,\"cache_hash_transfer_bytes\":%llu,\"seconds\":%.9g,\"gpu_peak_bytes\":%zu}\n",(unsigned long long)advances,(unsigned long long)diagnostic_calls,(unsigned long long)diagnostic_macs,(unsigned long long)hash_transfer_bytes,now()-began,gpu_peak_bytes());REQUIRE(!fclose(f));qwen_qkv_program_free(&program17);qwen_qkv_program_free(&program18);qwen_model_free(&model);return 0;
}
