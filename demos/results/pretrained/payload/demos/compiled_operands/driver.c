#define _POSIX_C_SOURCE 200809L
#include "qwen.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>

/* Public-API example. Numerical inference remains entirely in canonical C/CUDA. */
#define LIMIT 64
typedef struct { int consumed, pending; char cache[65], logits[65]; } Row;
static void die(const char *s){fprintf(stderr,"compiled-operands: %s\n",s);exit(2);}
static void hash(const void *p,size_t n,char out[65]){unsigned char d[32];tcm_sha256(p,n,d);tcm_hex(d,32,out);}
static void path(char out[4096],const char *dir,const char *name){int n=snprintf(out,4096,"%s/%s",dir,name);if(n<0||n>=4096)die("output path too long");}
static FILE *create(const char *dir,const char *name){char p[4096];path(p,dir,name);FILE *f=fopen(p,"wx");if(!f)die("output already exists or cannot be created");return f;}
static void cache_hash(const QwenSession *s,char out[65]){
    const QwenModel *m=s->model;size_t row=(size_t)s->position*m->head;
    size_t n=2*(size_t)m->layers*m->kvheads*row,offset=0;
    float *host=malloc(n*4+1);if(!host)die("cache audit allocation");
    for(int kind=0;kind<2;kind++)for(int l=0;l<m->layers;l++)for(int h=0;h<m->kvheads;h++){
        size_t address=((size_t)l*m->kvheads+h)*s->capacity*m->head;
        if(row)gpu_read(host+offset,(kind?s->values:s->keys)+address,row*4);
        offset+=row;
    }
    hash(host,n*4,out);free(host);
}
static int eos(const QwenModel *m,int t){return t==m->eos||(m->eos_extra>0&&t==m->eos_extra);}
static int cmp(const void *a,const void *b){int x=*(const int*)a,y=*(const int*)b;return (x>y)-(x<y);}
static int number(const char *s){char *e;errno=0;long n=strtol(s,&e,10);if(errno||*e||n<1||n>LIMIT)die("steps must be 1..64");return (int)n;}
static void capture(QwenSession *s,int token,Row *r,float *logits){
    if(qwen_push(s,token))die("native token transition refused");
    r->consumed=token;r->pending=s->pending;gpu_read(logits,s->logits,(size_t)s->model->vocab*4);
    hash(logits,(size_t)s->model->vocab*4,r->logits);cache_hash(s,r->cache);
}
int main(int argc,char **argv){
    if(argc!=5){fprintf(stderr,"usage: %s MODEL.tcm TOKENS.txt OUTPUT_DIRECTORY GENERATED_STEPS\n",argv[0]);return 2;}
    int budget=number(argv[4]),prompt[LIMIT],n=0,t;FILE *input=fopen(argv[2],"r");if(!input)die("cannot open tokens");
    while(fscanf(input,"%d",&t)==1){if(n>=LIMIT)die("prompt too long");prompt[n++]=t;}
    if(!feof(input)||fclose(input)||!n||n+budget>LIMIT)die("need nonempty integer prompt; prompt + generated steps <=64");
    TcmModel bundle;QwenModel model;char error[256];
    if(tcm_load(&bundle,argv[1],error,sizeof(error)))die(error);
    if(qwen_model_init(&model,&bundle,error,sizeof(error)))die(error);
    tcm_unload(&bundle);for(int i=0;i<n;i++)if(prompt[i]<0||prompt[i]>=model.vocab)die("token outside vocabulary");
    int capacity=n+budget+1;size_t width=(size_t)model.vocab;
    float *reference=malloc((size_t)LIMIT*width*4),*actual=malloc(width*4);if(!reference||!actual)die("logit audit allocation");
    Row rows[LIMIT];memset(rows,0,sizeof(rows));QwenSession source;if(qwen_session_init(&source,&model,capacity))die("source init");
    int count=0,generated=0;
    for(int i=0;i<n;i++){capture(&source,prompt[i],rows+count,reference+(size_t)count*width);count++;}
    for(int i=0;i<budget;i++){int own=source.pending;capture(&source,own,rows+count,reference+(size_t)count*width);count++;generated++;if(eos(&model,own))break;}
    size_t session_bytes=source.allocated_bytes;qwen_session_free(&source);
    int ids[LIMIT];for(int i=0;i<count;i++)ids[i]=rows[i].consumed;qsort(ids,(size_t)count,sizeof(int),cmp);
    int unique=0;for(int i=0;i<count;i++)if(!i||ids[i]!=ids[i-1])ids[unique++]=ids[i];
    QwenLexicalProgram compiled={0},loaded={0};if(qwen_lexical_compile(&compiled,&model,ids,(size_t)unique))die("lexical compile");
    char table[4096];path(table,argv[3],"atoms.bin");FILE *guard=fopen(table,"wx");if(!guard)die("table output already exists");if(fclose(guard))die("table guard close");
    if(qwen_lexical_save(&compiled,table)||qwen_lexical_load(&loaded,&model,table)||strcmp(compiled.fingerprint,loaded.fingerprint))die("table save/load identity");
    FILE *seal=create(argv[3],"source-seal.json");
    fprintf(seal,"{\"schema\":\"compiled-operands-source-v1\",\"model_digest\":\"%s\",\"model_identity\":\"%s\",\"program_fingerprint\":\"%s\",\"prompt_tokens\":%d,\"generated_tokens_consumed\":%d,\"table_ids\":[",model.digest,model.identity,compiled.fingerprint,n,generated);
    for(int i=0;i<unique;i++)fprintf(seal,"%s%d",i?",":"",ids[i]);
    fprintf(seal,"],\"steps\":[");for(int i=0;i<count;i++)fprintf(seal,"%s{\"position\":%d,\"consumed\":%d,\"pending\":%d,\"logits_sha256\":\"%s\",\"valid_cache_sha256\":\"%s\"}",i?",":"",i+1,rows[i].consumed,rows[i].pending,rows[i].logits,rows[i].cache);
    fprintf(seal,"]}\n");if(fclose(seal))die("source seal close");
    /* Candidate is created only after native source, compilation and seal close. */
    QwenSession candidate;if(qwen_session_init(&candidate,&model,capacity)||qwen_lexical_bind(&candidate,&loaded))die("candidate init/bind");
    FILE *out=create(argv[3],"report.json");fprintf(out,"{\"schema\":\"compiled-operands-demo-v1\",\"model_digest\":\"%s\",\"program_fingerprint\":\"%s\",\"rows\":[",model.digest,loaded.fingerprint);
    int all=1,candidate_steps=0,candidate_generated=0,refused=0;
    for(int i=0;i<count;i++){
        /* The continuation takes the candidate's pending ID, never rows[i].consumed. */
        int own=i<n?prompt[i]:candidate.pending;
        if(qwen_push(&candidate,own)){refused=1;all=0;break;}
        gpu_read(actual,candidate.logits,width*4);char ch[65],lh[65];cache_hash(&candidate,ch);hash(actual,width*4,lh);
        int same_logits=!memcmp(actual,reference+(size_t)i*width,width*4),same_cache=!strcmp(ch,rows[i].cache),same_tokens=own==rows[i].consumed&&candidate.pending==rows[i].pending;
        fprintf(out,"%s{\"position\":%d,\"consumed\":%d,\"pending\":%d,\"full_logits_exact\":%s,\"valid_cache_hash_exact\":%s,\"token_exact\":%s,\"logits_sha256\":\"%s\",\"valid_cache_sha256\":\"%s\"}",candidate_steps?",":"",i+1,own,candidate.pending,same_logits?"true":"false",same_cache?"true":"false",same_tokens?"true":"false",lh,ch);
        candidate_steps++;all&=same_logits&&same_cache&&same_tokens;
        if(i>=n){candidate_generated++;if(eos(&model,own))break;}
    }
    all&=candidate_steps==count;
    int missing=0;while(missing<model.vocab){int present=0;for(int i=0;i<unique;i++)present|=ids[i]==missing;if(!present)break;missing++;}
    QwenSession before=candidate;char before_cache[65],after_cache[65],before_logits[65],after_logits[65];cache_hash(&candidate,before_cache);gpu_read(actual,candidate.logits,width*4);hash(actual,width*4,before_logits);
    int missing_rc=qwen_push(&candidate,missing);cache_hash(&candidate,after_cache);gpu_read(actual,candidate.logits,width*4);hash(actual,width*4,after_logits);
    int unchanged=missing_rc!=0&&!memcmp(&before,&candidate,sizeof(candidate))&&!strcmp(before_cache,after_cache)&&!strcmp(before_logits,after_logits);
    uint64_t per=(uint64_t)model.hidden*(model.hidden+2*model.kvheads*model.head);
    fprintf(out,"],\"source_steps\":%d,\"candidate_steps\":%d,\"candidate_generated_tokens_consumed\":%d,\"candidate_transition_refused\":%s,\"all_steps_exact\":%s,\"missing_id\":%d,\"missing_refusal_no_mutation\":%s,\"save_load_identity\":true,\"covered_ids\":%d,\"compile_linear_macs\":%llu,\"table_host_bytes\":%zu,\"candidate_lexical_hits\":%llu,\"candidate_transfer_bytes\":%llu,\"candidate_original_first_layer_qkv_macs_omitted\":%llu,\"source_native_nominal_linear_macs\":%llu,\"candidate_nominal_linear_macs\":%llu,\"native_model_gpu_bytes\":%zu,\"session_gpu_bytes\":%zu,\"scope\":\"source-covered own-token exact replay; original weights and remaining model maps retained; no heldout or speed claim\"}\n",count,candidate_steps,candidate_generated,refused?"true":"false",all?"true":"false",missing,unchanged?"true":"false",unique,(unsigned long long)compiled.compile_linear_macs,compiled.host_bytes,(unsigned long long)candidate.operand_work.lexical_hits,(unsigned long long)candidate.operand_work.lexical_transfer_bytes,(unsigned long long)(per*(uint64_t)candidate_steps),(unsigned long long)(UINT64_C(1543569408)*(uint64_t)count),(unsigned long long)((UINT64_C(1543569408)-per)*(uint64_t)candidate_steps),model.weight_bytes,session_bytes);
    if(fclose(out))die("report close");
    qwen_session_free(&candidate);qwen_lexical_free(&loaded);qwen_lexical_free(&compiled);qwen_model_free(&model);free(reference);free(actual);
    return all&&unchanged?0:1;
}
