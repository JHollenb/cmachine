#define _POSIX_C_SOURCE 200809L
#include "qwen.h"
#include "memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>
#include "distribution.h"
#define REQUIRE(x) do{if(!(x)){fprintf(stderr,"query-memory %s:%d %s\n",__FILE__,__LINE__,#x);exit(2);}}while(0)
#define SITE 17
#define RANK 128
#define LIMIT 512
#define SOURCE_LIMIT 512
static const char *directory;
static double seconds(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void hash(const void *p,size_t n,char out[65]){unsigned char raw[32];tcm_sha256(p,n,raw);tcm_hex(raw,32,out);}
static void save(const char *name,const void *p,size_t n){char path[4096];REQUIRE(snprintf(path,sizeof(path),"%s/%s",directory,name)<(int)sizeof(path));FILE *f=fopen(path,"wb");REQUIRE(f&&fwrite(p,1,n,f)==n&&!fclose(f));}
static void finite_read(float *h,const float *p,size_t n){gpu_read(h,p,n*4);for(size_t i=0;i<n;i++)REQUIRE(isfinite(h[i]));}
static void ints(FILE *f,const int *v,int n){fputc('[',f);for(int i=0;i<n;i++)fprintf(f,"%s%d",i?",":"",v[i]);fputc(']',f);}
static int *tokens(const char *stem,int *count){char path[4096];snprintf(path,sizeof(path),"%s/%s.txt",directory,stem);FILE *f=fopen(path,"r");REQUIRE(f);int *ids=malloc(LIMIT*sizeof(int));REQUIRE(ids);*count=0;int value;while(fscanf(f,"%d",&value)==1){REQUIRE(*count<LIMIT&&value>=0&&value<151936);ids[(*count)++]=value;}REQUIRE(feof(f)&&*count>0&&!fclose(f));return ids;}
typedef struct {float norm[1536],q[1536],k[256],v[256],attention[1536],output[1536],coeff[RANK];} Record;
typedef struct {
    QwenSession *session;int factor,qk_factor,q_factor,k_factor,v_factor,query_fit,compact,zero,rank,count,coordinates_ready;
    QMMemory memory;Record *records;int ids[LIMIT],tops[LIMIT];float current_k[256],current_v[256];unsigned cache_mask;
    uint64_t linear_calls,linear_macs,native_qkv_calls,compiled_qkv_calls,coordinate_calls,native_attention_calls,compact_reads,zero_reads,reader_products,merge_distance_products,serving_transfer_bytes;
    float *probabilities;
} Run;
static Run *active;
typedef struct {int rank;float *basis,*query,*key,*value,*coordinates;} Program;
static Program programs[1],*active_program;static float *fitted_query,*reference_query;
static uint64_t all_linear_calls,all_linear_macs;
static void linear(TcmGpu *gpu,const float *w,int in,int out,const float *x,float *y){
    gpu_matvec(gpu,w,in,out,x,y);all_linear_calls++;all_linear_macs+=(uint64_t)in*out;
    if(active){active->linear_calls++;active->linear_macs+=(uint64_t)in*out;}
}
static void qm_matvec(TcmGpu *gpu,const float *w,int in,int out,const float *x,float *y){
    Run *r=active;const QwenModel *m=r?r->session->model:NULL;
    int selected=r&&r->session->layer==SITE;const QwenLayer *l=selected?m->layer+SITE:NULL;
    int q=selected&&w==l->q,k=selected&&w==l->k,v=selected&&w==l->v;
    if(q){
        if(r->records)finite_read(r->records[r->count].norm,x,1536);
    }
    if((q&&r->q_factor)||(k&&r->k_factor)||(v&&r->v_factor)){
        REQUIRE(active_program&&active_program->rank==r->rank);
        if(!r->coordinates_ready){linear(gpu,active_program->basis,1536,r->rank,x,active_program->coordinates);r->coordinates_ready=1;r->coordinate_calls++;if(r->records)finite_read(r->records[r->count].coeff,active_program->coordinates,r->rank);}
        linear(gpu,q?(r->query_fit==2?reference_query:r->query_fit?fitted_query:active_program->query):k?active_program->key:active_program->value,r->rank,out,active_program->coordinates,y);r->compiled_qkv_calls++;
    }else{linear(gpu,w,in,out,x,y);if(q||k||v)r->native_qkv_calls++;}
}
static void qm_cache(float *cache,const float *row,int heads,int dimension,int capacity,int position){
    Run *r=active;if(r&&r->compact&&r->session->layer==SITE){
        const QwenModel *m=r->session->model;size_t base=(size_t)SITE*m->kvheads*capacity*m->head;
        if(cache==r->session->keys+base){REQUIRE(!(r->cache_mask&1));finite_read(r->current_k,row,256);r->cache_mask|=1;}
        else{REQUIRE(cache==r->session->values+base&&!(r->cache_mask&2));finite_read(r->current_v,row,256);r->cache_mask|=2;}
        REQUIRE(heads==2&&dimension==128&&position==r->count);r->serving_transfer_bytes+=256*4;
    }else gpu_cache(cache,row,heads,dimension,capacity,position);
}
static void qm_attention(TcmGpu *gpu,float *out,const float *q,const float *keys,const float *values,float *scores,int qheads,int kvheads,int dimension,int capacity,int length){
    Run *r=active;if(r&&r->session->layer==SITE){
        if(r->compact){
            REQUIRE(r->cache_mask==3&&length==r->count+1);float query[1536],result[1536];finite_read(query,q,1536);
            if(r->memory.recent==16&&r->memory.clusters[0]==16)r->merge_distance_products+=2*16*128;
            REQUIRE(!qm_append(&r->memory,r->count,r->current_k,r->current_v));r->cache_mask=0;
            REQUIRE(!qm_read(&r->memory,query,result));gpu_write(out,result,sizeof(result));r->serving_transfer_bytes+=sizeof(query)+sizeof(result);r->compact_reads++;
            r->reader_products+=(uint64_t)6*((uint64_t)r->memory.clusters[0]+r->memory.clusters[1]+2*r->memory.recent)*128*2;
        }else if(r->zero){gpu_zero(out,1536*4);r->zero_reads++;}
        else{gpu_attention(gpu,out,q,keys,values,scores,qheads,kvheads,dimension,capacity,length);r->native_attention_calls++;r->reader_products+=(uint64_t)12*length*128*2;if(r->probabilities)finite_read(r->probabilities+(size_t)6*length*(length-1),scores,(size_t)12*length);}
    }else gpu_attention(gpu,out,q,keys,values,scores,qheads,kvheads,dimension,capacity,length);
}
#define gpu_matvec qm_matvec
#define gpu_cache qm_cache
#define gpu_attention qm_attention
#include "runtime/qwen.c"
#undef gpu_matvec
#undef gpu_cache
#undef gpu_attention

static void observer(void *context,int position,int layer,QwenPort port,const float *data,size_t n){
    Run *r=context;REQUIRE(position==r->count&&layer==SITE&&r->records);Record *p=r->records+r->count;float *target=NULL;
    if(port==QWEN_QUERY)target=p->q;else if(port==QWEN_KEY)target=p->k;else if(port==QWEN_VALUE)target=p->v;else if(port==QWEN_ATTENTION)target=p->attention;else if(port==QWEN_OUTPUT)target=p->output;
    if(target)finite_read(target,data,n);
}
static void advance(Run *r,int token){
    REQUIRE(r->count<LIMIT&&r->session->position==r->count&&!r->cache_mask);r->coordinates_ready=0;active=r;
    if(r->records){r->session->observer=observer;r->session->observer_context=r;r->session->observe_layer=SITE;r->session->observe_position=r->count;}
    REQUIRE(!qwen_push(r->session,token));REQUIRE(r->coordinates_ready==r->factor);r->ids[r->count]=token;r->tops[r->count++]=r->session->pending;active=NULL;
}
static void cache_digest(const QwenSession *s,char digest[65]){
    size_t n=(size_t)s->model->layers*s->model->kvheads*s->position*s->model->head;float *data=malloc(2*n*4);REQUIRE(data);size_t offset=0;
    for(int kind=0;kind<2;kind++)for(int layer=0;layer<s->model->layers;layer++)for(int group=0;group<s->model->kvheads;group++){
        const float *p=(kind?s->values:s->keys)+((size_t)layer*s->model->kvheads+group)*s->capacity*s->model->head;
        size_t count=(size_t)s->position*s->model->head;finite_read(data+offset,p,count);offset+=count;
    }hash(data,offset*4,digest);free(data);
}
static void record_arrays(Run *r,const char *name){
    char top_file[192];snprintf(top_file,sizeof(top_file),"%s-tops.i32",name);save(top_file,r->tops,(size_t)r->count*4);
    if(r->probabilities){snprintf(top_file,sizeof(top_file),"%s-probabilities.f32",name);save(top_file,r->probabilities,(size_t)6*r->count*(r->count+1)*4);}
    struct {const char *name;size_t offset,n;} fields[]={{"norm",offsetof(Record,norm),1536},{"q",offsetof(Record,q),1536},{"k",offsetof(Record,k),256},{"v",offsetof(Record,v),256},{"attention",offsetof(Record,attention),1536},{"output",offsetof(Record,output),1536},{"coeff",offsetof(Record,coeff),RANK}};
    for(unsigned field=0;field<sizeof(fields)/sizeof(*fields);field++){
        float *host=malloc((size_t)r->count*fields[field].n*4);REQUIRE(host);
        for(int i=0;i<r->count;i++)memcpy(host+(size_t)i*fields[field].n,(char*)(r->records+i)+fields[field].offset,fields[field].n*4);
        char file[192];snprintf(file,sizeof(file),"%s-%s.f32",name,fields[field].name);save(file,host,(size_t)r->count*fields[field].n*4);free(host);
    }
}
static void load_program(FILE *f,Program *program){
    const char *labels[]={"basis","q","k","v"};float **outputs[]={&program->basis,&program->query,&program->key,&program->value};int widths[]={1536,1536,256,256};program->rank=128;
    for(int kind=0;kind<4;kind++){
        char file[4096];snprintf(file,sizeof(file),"%s/program128-%s.f32",directory,labels[kind]);FILE *source=fopen(file,"rb");size_t n=(size_t)widths[kind]*128;float *matrix=malloc(n*4);REQUIRE(source&&matrix&&fread(matrix,4,n,source)==n&&fgetc(source)==EOF&&!fclose(source));for(size_t j=0;j<n;j++)REQUIRE(isfinite(matrix[j]));*outputs[kind]=gpu_weight(matrix,TCM_F32,n);free(matrix);
    }
    program->coordinates=gpu_alloc(128*4);
    char fitfile[4096];snprintf(fitfile,sizeof(fitfile),"%s/query-fit.f32",directory);FILE *fit=fopen(fitfile,"rb");size_t fitn=(size_t)1536*128;float *fitdata=malloc(fitn*4);REQUIRE(fit&&fitdata&&fread(fitdata,4,fitn,fit)==fitn&&fgetc(fit)==EOF&&!fclose(fit));for(size_t j=0;j<fitn;j++)REQUIRE(isfinite(fitdata[j]));fitted_query=gpu_weight(fitdata,TCM_F32,fitn);free(fitdata);
    fprintf(f,"{\"rank\":128,\"source_rows\":193,\"basis_kind\":\"authenticated-source128-bank-reuse\",\"compile_linear_calls\":0,\"compile_linear_macs\":0,\"native_qkv_macs\":3145728,\"immutable_numeric_bytes\":1835008}");
}
static void fresh(QwenSession *s,const QwenModel *m){REQUIRE(!qwen_session_init(s,m,512));size_t n=(size_t)m->layers*m->kvheads*512*m->head;gpu_zero(s->keys,n*4);gpu_zero(s->values,n*4);}
static void native_sham(const QwenModel *m){
    int n,*input=tokens("source0",&n);REQUIRE(n>=16);QwenSession native,wrapped;fresh(&native,m);fresh(&wrapped,m);Run r={0};r.session=&wrapped;float *a=malloc(151936*4),*b=malloc(151936*4);REQUIRE(a&&b);
    for(int i=0;i<16;i++){REQUIRE(!qwen_push(&native,input[i]));advance(&r,input[i]);finite_read(a,native.logits,151936);finite_read(b,wrapped.logits,151936);REQUIRE(native.pending==wrapped.pending&&!memcmp(a,b,151936*4));}
    char ah[65],bh[65];cache_digest(&native,ah);cache_digest(&wrapped,bh);REQUIRE(!strcmp(ah,bh));save("native-sham-last-logits.f32",a,151936*4);free(a);free(b);free(input);qwen_session_free(&native);qwen_session_free(&wrapped);
}
static void file_hash(const char *name,char digest[65]){
    char file[4096];snprintf(file,sizeof(file),"%s/%s",directory,name);FILE *f=fopen(file,"rb");struct stat st;REQUIRE(f&&!stat(file,&st));void *p=malloc(st.st_size);REQUIRE(p&&fread(p,1,st.st_size,f)==(size_t)st.st_size&&!fclose(f));hash(p,st.st_size,digest);free(p);
}
typedef struct {int start,prefill_end,final,skip,length,stopped,ids[32],clocks[32];} Turn;
typedef struct {int n,prompt,follow_n,ids[LIMIT],phase[LIMIT],prediction_turn[LIMIT];Turn turns[2];} OHTrace;

static void trace_init(OHTrace *t,int prompt,int follow){
    memset(t,0,sizeof(*t));t->prompt=prompt;t->follow_n=follow;
    for(int j=0;j<LIMIT;j++)t->prediction_turn[j]=-1;
}
static int snapshot_clock(const OHTrace *t,int row){
    for(int j=0;j<2;j++)if(row==t->turns[j].clocks[0]||row==t->turns[j].clocks[t->turns[j].length-1])return 1;
    return 0;
}
static void oh_snapshot(const char *name,const OHTrace *trace,int row,const float *logits){
    if(snapshot_clock(trace,row)){char file[192];snprintf(file,sizeof(file),"%s-t%d-logits.f32",name,row);save(file,logits,151936*4);}
}
static void memory_boundary(Run *r,const char *name,int turn,const char *stage){
    char file[192];snprintf(file,sizeof(file),"%s-t%d-%s-memory.bin",name,turn,stage);save(file,&r->memory,sizeof(r->memory));
}
static void trace_write(const OHTrace *s,const char *name){
    char file[4096];snprintf(file,sizeof(file),"%s/%s-trace.json",directory,name);FILE *f=fopen(file,"w");REQUIRE(f);
    fprintf(f,"{\"count\":%d,\"prompt_length\":%d,\"followup_length\":%d,\"consumed_ids\":",s->n,s->prompt,s->follow_n);ints(f,s->ids,s->n);
    fprintf(f,",\"input_phases\":");ints(f,s->phase,s->n);fprintf(f,",\"prediction_turn\":");ints(f,s->prediction_turn,s->n);fprintf(f,",\"turns\":[");
    for(int j=0;j<2;j++){
        const Turn *a=s->turns+j;
        fprintf(f,"%s{\"turn\":%d,\"start\":%d,\"prefill_end\":%d,\"final\":%d,\"template_skip\":%d,\"stopped\":%s,\"truncated\":%s,\"answer_ids\":",j?",":"",j,a->start,a->prefill_end,a->final,a->skip,a->stopped?"true":"false",a->stopped?"false":"true");
        ints(f,a->ids,a->length);fprintf(f,",\"prediction_clocks\":");ints(f,a->clocks,a->length);fputc('}',f);
    }
    fprintf(f,"]}\n");REQUIRE(!fclose(f));
}
static void write_ledger(FILE *f,Run *r,const char *name,const char *role,double start){
    int sentinel=1;
    if(r->compact){
        size_t base=(size_t)SITE*2*LIMIT*128,n=(size_t)2*LIMIT*128;float *h=malloc(n*4);REQUIRE(h);
        for(int k=0;k<2;k++){finite_read(h,(k?r->session->values:r->session->keys)+base,n);for(size_t j=0;j<n;j++)if(h[j]!=0)sentinel=0;}
        free(h);REQUIRE(sentinel&&r->memory.next==(uint64_t)r->count);
    }
    char kv[65];cache_digest(r->session,kv);
    fprintf(f,"{\"name\":\"%s\",\"role\":\"%s\",\"compact\":%s,\"count\":%d,\"linear_calls\":%llu,\"linear_macs\":%llu,\"native_qkv_calls\":%llu,\"compiled_qkv_calls\":%llu,\"coordinate_calls\":%llu,\"compact_reads\":%llu,\"native_attention_calls\":%llu,\"reader_products\":%llu,\"merge_distance_products\":%llu,\"serving_transfer_bytes\":%llu,\"selected_full_kv_untouched\":%s,\"final_global_kv_sha256\":\"%s\",\"seconds\":%.9g}",name,role,r->compact?"true":"false",r->count,(unsigned long long)r->linear_calls,(unsigned long long)r->linear_macs,(unsigned long long)r->native_qkv_calls,(unsigned long long)r->compiled_qkv_calls,(unsigned long long)r->coordinate_calls,(unsigned long long)r->compact_reads,(unsigned long long)r->native_attention_calls,(unsigned long long)r->reader_products,(unsigned long long)r->merge_distance_products,(unsigned long long)r->serving_transfer_bytes,r->compact&&sentinel?"true":"false",kv,seconds()-start);
}
static void compact_add(Run *r,OHTrace *trace,int token,int phase,FILE *stream,float *logits){
    trace->phase[r->count]=phase;advance(r,token);finite_read(logits,r->session->logits,151936);REQUIRE(fwrite(logits,4,151936,stream)==151936);
}
static void candidate_seal(const char *name,int pair){
    char path[4096],file[192],digest[65];snprintf(path,sizeof(path),"%s/pair%d-candidate-seal.json",directory,pair);FILE *f=fopen(path,"w");REQUIRE(f);
    file_hash("candidate-logits.tmp",digest);fprintf(f,"{\"phase\":\"compact-complete-before-native-shadow\",\"candidate_stream_sha256\":\"%s\",\"files\":[",digest);
    DIR *d=opendir(directory);REQUIRE(d);struct dirent *e;int comma=0;size_t n=strlen(name);
    while((e=readdir(d))){
        if(strncmp(e->d_name,name,n)||e->d_name[n]!='-')continue;
        REQUIRE(strlen(e->d_name)<sizeof(file));strcpy(file,e->d_name);file_hash(file,digest);
        fprintf(f,"%s{\"name\":\"%s\",\"sha256\":\"%s\"}",comma++?",":"",file,digest);
    }
    fprintf(f,"]}\n");REQUIRE(!closedir(d)&&!fclose(f));
}
static void pair(FILE *out,const QwenModel *m,int source,int own){
    int n,fn,oldn;char stem[96],name[96],shadow[96],file[4096];int index=own*3+source;
    snprintf(stem,sizeof(stem),"source%d",source);int *prompt=tokens(stem,&n);
    snprintf(stem,sizeof(stem),"source%d-followup",source);int *follow=tokens(stem,&fn);
    snprintf(stem,sizeof(stem),"source%d-native-history",source);int *old=tokens(stem,&oldn);
    REQUIRE(n+fn+64<LIMIT&&follow[0]==151645);
    snprintf(name,sizeof(name),"pair%d-compact",index);snprintf(shadow,sizeof(shadow),"pair%d-native",index);
    QwenSession session;fresh(&session,m);Run *r=calloc(1,sizeof(*r));REQUIRE(r);r->session=&session;
    r->q_factor=r->k_factor=r->qk_factor=r->factor=r->compact=r->query_fit=1;r->rank=128;
    r->records=calloc(LIMIT,sizeof(Record));REQUIRE(r->records);active_program=programs;
    OHTrace trace;trace_init(&trace,n,fn);float *logits=malloc(151936*4),*candidate=malloc(151936*4);REQUIRE(logits&&candidate);
    snprintf(file,sizeof(file),"%s/candidate-logits.tmp",directory);FILE *cf=fopen(file,"w+b");REQUIRE(cf);double start=seconds();
    fprintf(out,"{\"pair\":%d,\"source\":%d,\"history\":\"%s\",\"evaluators\":[",index,source,own?"compact-generated":"native-generated");
    for(int turn=0;turn<2;turn++){
        Turn *t=trace.turns+turn;t->start=r->count;int *input=turn?follow:prompt,count=turn?fn:n;
        t->skip=turn&&r->ids[r->count-1]==input[0];
        for(int j=t->skip;j<count;j++){
            if(!own)REQUIRE(r->count<oldn&&old[r->count]==input[j]);
            compact_add(r,&trace,input[j],turn?2:0,cf,logits);
        }
        t->prefill_end=r->count;memory_boundary(r,name,turn,"prefill");
        for(int j=0;j<32;j++){
            REQUIRE(own||r->count<oldn);int token=own?session.pending:old[r->count],row=r->count-1;
            trace.prediction_turn[row]=turn;t->clocks[t->length]=row;t->ids[t->length++]=token;
            t->stopped=token==m->eos||token==m->eos_extra;
            compact_add(r,&trace,token,turn?3:1,cf,logits);if(t->stopped)break;
        }
        t->final=r->count;memory_boundary(r,name,turn,"final");
    }
    REQUIRE(own||r->count==oldn);trace.n=r->count;memcpy(trace.ids,r->ids,(size_t)r->count*4);
    REQUIRE(!fflush(cf));
    for(int t=0;t<trace.n;t++)if(snapshot_clock(&trace,t)){
        REQUIRE(!fseek(cf,(long)((size_t)t*151936*4),SEEK_SET)&&fread(logits,4,151936,cf)==151936);oh_snapshot(name,&trace,t,logits);
    }
    record_arrays(r,name);trace_write(&trace,name);write_ledger(out,r,name,own?"autonomous-generator":"native-ID-replay",start);
    free(r->records);free(r);qwen_session_free(&session);active_program=NULL;
    candidate_seal(name,index);
    /* The compact session is gone. This fresh native session owns every cache. */
    fresh(&session,m);r=calloc(1,sizeof(*r));REQUIRE(r);r->session=&session;r->records=calloc(LIMIT,sizeof(Record));REQUIRE(r->records);
    snprintf(file,sizeof(file),"%s/native-logits.tmp",directory);FILE *nf=fopen(file,"wb");REQUIRE(nf);
    snprintf(file,sizeof(file),"%s/pair%d-metrics.json",directory,index);FILE *mf=fopen(file,"w");REQUIRE(mf);fprintf(mf,"{\"pair\":%d,\"rows\":[",index);
    REQUIRE(!fseek(cf,0,SEEK_SET));double sums[2]={0};start=seconds();int prefix=1;
    for(int t=0;t<trace.n;t++){
        advance(r,trace.ids[t]);finite_read(logits,session.logits,151936);REQUIRE(fwrite(logits,4,151936,nf)==151936&&fread(candidate,4,151936,cf)==151936);
        Distribution metric;REQUIRE(!distribution(logits,candidate,151936,&metric));int turn=trace.prediction_turn[t],expected=turn>=0?trace.ids[t+1]:-1;
        if(turn>=0)sums[turn]+=metric.kl;
        char nh[65],ch[65];hash(logits,151936*4,nh);hash(candidate,151936*4,ch);oh_snapshot(shadow,&trace,t,logits);
        fprintf(mf,"%s{\"step\":%d,\"prediction_turn\":%d,\"expected_next_token\":%d,\"native_greedy_prefix_before_decision\":%s,\"kl_nats\":%.17g,\"relative_logit_l2\":%.17g,\"native_top_logp_delta_nats\":%.17g,\"native_entropy_nats\":%.17g,\"native_top\":%d,\"candidate_top\":%d,\"native_logits_sha256\":\"%s\",\"candidate_logits_sha256\":\"%s\"}",t?",":"",t,turn,expected,prefix?"true":"false",metric.kl,metric.l2,metric.top_logp_delta,metric.native_entropy,metric.native_top,metric.candidate_top,nh,ch);
        if(turn>=0){if(own)REQUIRE(metric.candidate_top==expected);else REQUIRE(metric.native_top==expected);if(metric.native_top!=metric.candidate_top)prefix=0;}
    }
    REQUIRE(fgetc(cf)==EOF);fprintf(mf,"],\"per_turn_kl_nats\":[%.17g,%.17g],\"future_mean_kl_nats\":%.17g}\n",sums[0]/trace.turns[0].length,sums[1]/trace.turns[1].length,.5*(sums[0]/trace.turns[0].length+sums[1]/trace.turns[1].length));
    REQUIRE(!fclose(cf)&&!fclose(nf)&&!fclose(mf));record_arrays(r,shadow);fputc(',',out);write_ledger(out,r,shadow,"independent-native-shadow",start);
    fprintf(out,"],\"candidate_sealed_before_native\":true}");REQUIRE(!fflush(out));
    free(r->records);free(r);qwen_session_free(&session);free(prompt);free(follow);free(old);free(logits);free(candidate);
    printf("PAIR_READY %d\n",index);REQUIRE(!fflush(stdout)&&getchar()=='\n');
}
int main(int argc,char **argv){
    REQUIRE(argc==3);directory=argv[2];TcmModel bundle;QwenModel m;char error[256],file[4096];double start=seconds();
    REQUIRE(!tcm_load(&bundle,argv[1],error,sizeof(error))&&!qwen_model_init(&m,&bundle,error,sizeof(error)));tcm_unload(&bundle);native_sham(&m);
    snprintf(file,sizeof(file),"%s/driver.json",directory);FILE *f=fopen(file,"w");REQUIRE(f);
    fprintf(f,"{\"schema\":\"tcm-query-own-history-driver-v1\",\"model_digest\":\"%s\",\"model_identity\":\"%s\",\"capacity\":512,\"layer\":17,\"memory_bytes\":%zu,\"native_sham_steps\":16,\"native_sham_logits_and_kv_byte_exact\":true,\"programs\":[",m.digest,m.identity,sizeof(QMMemory));load_program(f,programs);
    char image_sha[65],input_sha[65];file_hash("query-fit.f32",image_sha);file_hash("input-manifest.json",input_sha);
    fprintf(f,"],\"query_image_sha256\":\"%s\",\"input_manifest_sha256\":\"%s\",\"pairs\":[",image_sha,input_sha);
    for(int own=0;own<2;own++)for(int c=0;c<3;c++){if(own||c)fputc(',',f);pair(f,&m,c,own);}
    fprintf(f,"],\"all_linear_calls\":%llu,\"all_linear_macs\":%llu,\"seconds\":%.9g}\n",(unsigned long long)all_linear_calls,(unsigned long long)all_linear_macs,seconds()-start);REQUIRE(!fclose(f));
    gpu_free(programs[0].basis);gpu_free(programs[0].query);gpu_free(programs[0].key);gpu_free(programs[0].value);gpu_free(programs[0].coordinates);gpu_free(fitted_query);qwen_model_free(&m);return 0;
}
