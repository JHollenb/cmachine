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
static void boundary(FILE *f,Run *r,const char *name,int turn,const char *stage){
    char stem[192],file[256],digest[65];snprintf(stem,sizeof(stem),"%s-t%d-%s",name,turn,stage);float *logits=malloc(151936*4);REQUIRE(logits);finite_read(logits,r->session->logits,151936);snprintf(file,sizeof(file),"%s-logits.f32",stem);save(file,logits,151936*4);free(logits);
    cache_digest(r->session,digest);fprintf(f,",\"%s\":{\"position\":%d,\"top\":%d,\"valid_global_kv_sha256\":\"%s\"}",stage,r->count,r->session->pending,digest);
    if(r->compact){snprintf(file,sizeof(file),"%s-memory.bin",stem);save(file,&r->memory,sizeof(r->memory));}
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
static void seal(void){
    DIR *dir=opendir(directory);REQUIRE(dir);FILE *f;char file[4096];snprintf(file,sizeof(file),"%s/candidate-seal.json",directory);f=fopen(file,"w");REQUIRE(f);fprintf(f,"{\"files\":[");struct dirent *entry;int comma=0;
    while((entry=readdir(dir))){size_t n=strlen(entry->d_name);if(n<4||(!strstr(entry->d_name,".f32")&&!strstr(entry->d_name,".bin")&&!strstr(entry->d_name,".i32")&&!strstr(entry->d_name,".f64")))continue;
        snprintf(file,sizeof(file),"%s/%s",directory,entry->d_name);struct stat st;REQUIRE(!stat(file,&st)&&S_ISREG(st.st_mode));void *data=malloc(st.st_size);FILE *in=fopen(file,"rb");REQUIRE(data&&in&&fread(data,1,st.st_size,in)==(size_t)st.st_size&&!fclose(in));char digest[65];hash(data,st.st_size,digest);free(data);fprintf(f,"%s{\"name\":\"%s\",\"bytes\":%llu,\"sha256\":\"%s\"}",comma++?",":"",entry->d_name,(unsigned long long)st.st_size,digest);
    }fprintf(f,"]}\n");REQUIRE(!fclose(f)&&!closedir(dir));fprintf(stderr,"sealed candidate/source arrays before native references\n");
}
static void fresh(QwenSession *s,const QwenModel *m){REQUIRE(!qwen_session_init(s,m,512));size_t n=(size_t)m->layers*m->kvheads*512*m->head;gpu_zero(s->keys,n*4);gpu_zero(s->values,n*4);}
static void native_sham(const QwenModel *m){
    int n,*input=tokens("source0",&n);REQUIRE(n>=16);QwenSession native,wrapped;fresh(&native,m);fresh(&wrapped,m);Run r={0};r.session=&wrapped;float *a=malloc(151936*4),*b=malloc(151936*4);REQUIRE(a&&b);
    for(int i=0;i<16;i++){REQUIRE(!qwen_push(&native,input[i]));advance(&r,input[i]);finite_read(a,native.logits,151936);finite_read(b,wrapped.logits,151936);REQUIRE(native.pending==wrapped.pending&&!memcmp(a,b,151936*4));}
    char ah[65],bh[65];cache_digest(&native,ah);cache_digest(&wrapped,bh);REQUIRE(!strcmp(ah,bh));save("native-sham-last-logits.f32",a,151936*4);free(a);free(b);free(input);qwen_session_free(&native);qwen_session_free(&wrapped);
}
static void path(FILE *f,const QwenModel *m,int task,int policy){
    REQUIRE(policy==0||policy==5||policy==7||policy==8||policy==9||policy==10||policy==11||policy==12);
    static const char *labels[]={"native","compiled","memory","compiled-memory","zero","values-native","address-native","values-native-memory","query-native","query-fit","query-native-memory","query-fit-memory","response-reference-memory"};int n,follow_n;char stem[128],name[128];snprintf(stem,sizeof(stem),"case%d-prompt",task);int *prompt=tokens(stem,&n);snprintf(stem,sizeof(stem),"case%d-followup",task);int *follow=tokens(stem,&follow_n);snprintf(name,sizeof(name),"case%d-%s",task,labels[policy]);
    QwenSession s;fresh(&s,m);Run *r=calloc(1,sizeof(*r));REQUIRE(r);r->session=&s;r->q_factor=policy==5||policy==7||policy==9||policy==11||policy==12;r->k_factor=policy!=0;r->v_factor=0;r->qk_factor=r->q_factor&&r->k_factor;r->query_fit=policy==12?2:policy==9||policy==11;r->factor=r->q_factor||r->k_factor;r->compact=policy==7||policy==10||policy==11||policy==12;r->zero=0;r->rank=r->factor?128:0;active_program=r->factor?programs:NULL;r->records=calloc(LIMIT,sizeof(Record));REQUIRE(r->records);if(!r->compact&&!r->zero){r->probabilities=calloc((size_t)6*LIMIT*(LIMIT+1),4);REQUIRE(r->probabilities);}double start=seconds();
    fprintf(f,"{\"name\":\"%s\",\"task\":%d,\"policy\":%d,\"rank\":%d,\"factor\":%s,\"qk_factor\":%s,\"q_factor\":%s,\"k_factor\":%s,\"query_fit\":%s,\"v_factor\":%s,\"compact\":%s,\"zero\":%s,\"turns\":[",name,task,policy,r->rank,r->factor?"true":"false",r->qk_factor?"true":"false",r->q_factor?"true":"false",r->k_factor?"true":"false",r->query_fit?"true":"false",r->v_factor?"true":"false",r->compact?"true":"false",r->zero?"true":"false");
    for(int turn=0;turn<2;turn++){
        if(turn)fputc(',',f);int start_position=r->count,*input=turn?follow:prompt,count=turn?follow_n:n,skip=turn&&r->ids[r->count-1]==input[0];
        REQUIRE(!turn||input[0]==151645);for(int j=skip;j<count;j++)advance(r,input[j]);
        int budget=task%3==0?8:task%3==1?16:32;fprintf(f,"{\"turn\":%d,\"start\":%d,\"ingested\":%d,\"template_skip\":%d,\"budget\":%d",turn,start_position,count-skip,skip,budget);boundary(f,r,name,turn,"prefill");
        int answers[32],length=0,stopped=0;for(int j=0;j<budget;j++){int token=s.pending;answers[length++]=token;stopped=token==m->eos||token==m->eos_extra;advance(r,token);if(stopped)break;}
        fprintf(f,",\"answer_ids\":");ints(f,answers,length);fprintf(f,",\"stopped\":%s,\"truncated\":%s",stopped?"true":"false",stopped?"false":"true");boundary(f,r,name,turn,"final");fputc('}',f);REQUIRE(!fflush(f));
    }
    int sentinel=1;if(r->compact){size_t base=(size_t)SITE*2*512*128,ncache=(size_t)2*512*128;float *host=malloc(ncache*4);REQUIRE(host);for(int kind=0;kind<2;kind++){finite_read(host,(kind?s.values:s.keys)+base,ncache);for(size_t j=0;j<ncache;j++)if(host[j]!=0)sentinel=0;}free(host);REQUIRE(sentinel&&r->memory.next==(uint64_t)r->count);}
    record_arrays(r,name);fprintf(f,"],\"consumed_ids\":");ints(f,r->ids,r->count);fprintf(f,",\"count\":%d,\"linear_calls\":%llu,\"linear_macs\":%llu,\"native_qkv_calls\":%llu,\"compiled_qkv_calls\":%llu,\"coordinate_calls\":%llu,\"native_attention_calls\":%llu,\"compact_reads\":%llu,\"zero_reads\":%llu,\"reader_products\":%llu,\"merge_distance_products\":%llu,\"serving_transfer_bytes\":%llu,\"selected_full_kv_untouched\":%s,\"memory_bytes\":%zu,\"seconds\":%.9g}",r->count,(unsigned long long)r->linear_calls,(unsigned long long)r->linear_macs,(unsigned long long)r->native_qkv_calls,(unsigned long long)r->compiled_qkv_calls,(unsigned long long)r->coordinate_calls,(unsigned long long)r->native_attention_calls,(unsigned long long)r->compact_reads,(unsigned long long)r->zero_reads,(unsigned long long)r->reader_products,(unsigned long long)r->merge_distance_products,(unsigned long long)r->serving_transfer_bytes,r->compact&&sentinel?"true":"false",sizeof(QMMemory),seconds()-start);REQUIRE(!fflush(f));
    fprintf(stderr,"completed %s:%d own advances,%.3fs\n",name,r->count,seconds()-start);free(r->probabilities);free(r->records);free(r);qwen_session_free(&s);free(prompt);free(follow);
}
static void file_hash(const char *name,char digest[65]){
    char file[4096];snprintf(file,sizeof(file),"%s/%s",directory,name);FILE *f=fopen(file,"rb");struct stat st;REQUIRE(f&&!stat(file,&st));void *p=malloc(st.st_size);REQUIRE(p&&fread(p,1,st.st_size,f)==(size_t)st.st_size&&!fclose(f));hash(p,st.st_size,digest);free(p);
}
static float *image(const char *name,int install){
    char file[4096];snprintf(file,sizeof(file),"%s/%s",directory,name);FILE *f=fopen(file,"rb");size_t n=1536*128;float *p=malloc(n*4);REQUIRE(f&&p&&fread(p,4,n,f)==n&&fgetc(f)==EOF&&!fclose(f));for(size_t i=0;i<n;i++)REQUIRE(isfinite(p[i]));
    float *result=NULL;if(install)result=gpu_weight(p,TCM_F32,n);else save("selected-query.f32",p,n*4);free(p);return result;
}
static void source_snapshot(const char *name,int step,int prompt,int count,const float *logits){
    if(step!=15&&step!=prompt-1&&step!=count-1)return;char file[256];snprintf(file,sizeof(file),"%s-t%d-logits.f32",name,step);save(file,logits,151936*4);
}
static void handshake(const char *message){
    printf("%s\n",message);REQUIRE(!fflush(stdout));REQUIRE(getchar()=='\n');
}
static int select_source(const QwenModel *m,uint64_t *source_calls,uint64_t *source_macs){
    const size_t vocab=151936;const double doses[]={0,-1,.25,.5,1,2};int counts[3],*ids[3],pn,*prompts=tokens("source-prompt-lengths",&pn);REQUIRE(pn==3);size_t total=0;
    for(int c=0;c<3;c++){char label[32];snprintf(label,sizeof(label),"source%d",c);ids[c]=tokens(label,&counts[c]);REQUIRE(counts[c]==prompts[c]+8);total+=counts[c];}REQUIRE(total==193);
    float *native=malloc(total*vocab*4),*current=malloc(vocab*4);REQUIRE(native&&current);uint64_t calls_before=all_linear_calls,macs_before=all_linear_macs;double means[6]={0};char filename[4096],native_file[4096];
    snprintf(native_file,sizeof(native_file),"%s/source-native-logits.tmp",directory);FILE *nf=fopen(native_file,"wb");REQUIRE(nf);
    size_t offset=0;
    for(int c=0;c<3;c++){
        QwenSession session;fresh(&session,m);Run *r=calloc(1,sizeof(*r));REQUIRE(r);r->session=&session;r->records=calloc(LIMIT,sizeof(Record));REQUIRE(r->records);char name[80];snprintf(name,sizeof(name),"source%d-native",c);
        for(int t=0;t<counts[c];t++){advance(r,ids[c][t]);float *row=native+(offset+t)*vocab;finite_read(row,session.logits,vocab);REQUIRE(fwrite(row,4,vocab,nf)==vocab);source_snapshot(name,t,prompts[c],counts[c],row);}
        record_arrays(r,name);offset+=counts[c];free(r->records);free(r);qwen_session_free(&session);
    }REQUIRE(!fclose(nf));
    for(int proposal=0;proposal<6;proposal++){
        char label[80];snprintf(label,sizeof(label),"query-proposal-%d.f32",proposal);gpu_free(fitted_query);fitted_query=image(label,1);
        snprintf(filename,sizeof(filename),"%s/source-candidate-logits.tmp",directory);FILE *cf=fopen(filename,"wb");REQUIRE(cf);snprintf(filename,sizeof(filename),"%s/source-metrics-%d.json",directory,proposal);FILE *mf=fopen(filename,"w");REQUIRE(mf);fprintf(mf,"{\"proposal\":%d,\"dose\":%.17g,\"rows\":[",proposal,doses[proposal]);offset=0;int comma=0;
        for(int c=0;c<3;c++){
            QwenSession session;fresh(&session,m);Run *r=calloc(1,sizeof(*r));REQUIRE(r);r->session=&session;r->q_factor=r->k_factor=r->qk_factor=r->factor=r->compact=r->query_fit=1;r->rank=128;r->records=calloc(LIMIT,sizeof(Record));REQUIRE(r->records);active_program=programs;char name[80];snprintf(name,sizeof(name),"source%d-proposal%d",c,proposal);double aggregate=0;
            for(int t=0;t<counts[c];t++){
                advance(r,ids[c][t]);finite_read(current,session.logits,vocab);REQUIRE(fwrite(current,4,vocab,cf)==vocab);Distribution metric;REQUIRE(!distribution(native+(offset+t)*vocab,current,vocab,&metric));aggregate+=metric.kl;char nh[65],ch[65];hash(native+(offset+t)*vocab,vocab*4,nh);hash(current,vocab*4,ch);source_snapshot(name,t,prompts[c],counts[c],current);
                fprintf(mf,"%s{\"source\":%d,\"step\":%d,\"region\":\"%s\",\"kl_nats\":%.17g,\"relative_logit_l2\":%.17g,\"native_top_logp_delta_nats\":%.17g,\"native_entropy_nats\":%.17g,\"native_top\":%d,\"candidate_top\":%d,\"native_logits_sha256\":\"%s\",\"candidate_logits_sha256\":\"%s\"}",comma++?",":"",c,t,t>=prompts[c]?"historical-own-suffix":t<24?"shared-header":"prompt-body",metric.kl,metric.l2,metric.top_logp_delta,metric.native_entropy,metric.native_top,metric.candidate_top,nh,ch);
            }
            means[proposal]+=aggregate/counts[c]/3;record_arrays(r,name);offset+=counts[c];free(r->records);free(r);qwen_session_free(&session);
        }
        fprintf(mf,"],\"equal_trajectory_mean_kl_nats\":%.17g}\n",means[proposal]);REQUIRE(!fclose(mf)&&!fclose(cf));char message[128];snprintf(message,sizeof(message),"SOURCE_CANDIDATE_READY %d",proposal);handshake(message);
    }
    int selected=0;for(int i=1;i<6;i++)if(means[i]<means[selected]||(means[i]==means[selected]&&fabs(doses[i])<fabs(doses[selected])))selected=i;
    char chosen[80];snprintf(chosen,sizeof(chosen),"query-proposal-%d.f32",selected);image(chosen,0);char image_sha[65];file_hash("selected-query.f32",image_sha);snprintf(filename,sizeof(filename),"%s/source-selection.json",directory);FILE *f=fopen(filename,"w");REQUIRE(f);fprintf(f,"{\"schema\":\"tcm-native-consumer-query-selection-v1\",\"selected_index\":%d,\"selected_image_sha256\":\"%s\",\"source_advances\":1351,\"source_native_advances\":193,\"source_candidate_advances\":1158,\"source_lengths\":[%d,%d,%d],\"prompt_lengths\":[%d,%d,%d],\"selector\":\"equal-trajectory mean of all next-token KL(native||candidate); exact tie smaller abs dose then index\",\"mean_kl_nats\":[",selected,image_sha,counts[0],counts[1],counts[2],prompts[0],prompts[1],prompts[2]);for(int i=0;i<6;i++)fprintf(f,"%s%.17g",i?",":"",means[i]);fprintf(f,"]}\n");REQUIRE(!fclose(f));
    for(int c=0;c<3;c++)free(ids[c]);free(prompts);free(native);free(current);*source_calls=all_linear_calls-calls_before;*source_macs=all_linear_macs-macs_before;
    char seal_sha[65],message[256];file_hash("source-selection.json",seal_sha);snprintf(message,sizeof(message),"SOURCE_SELECTION_READY %d %s %s",selected,seal_sha,image_sha);handshake(message);gpu_free(fitted_query);fitted_query=image("selected-query.f32",1);return selected;
}
int main(int argc,char **argv){
    REQUIRE(argc==3);directory=argv[2];TcmModel bundle;QwenModel m;char error[256],file[4096];double started=seconds();REQUIRE(!tcm_load(&bundle,argv[1],error,sizeof(error))&&!qwen_model_init(&m,&bundle,error,sizeof(error)));tcm_unload(&bundle);
    native_sham(&m);snprintf(file,sizeof(file),"%s/driver.json",directory);FILE *f=fopen(file,"w");REQUIRE(f);fprintf(f,"{\"schema\":\"tcm-query-consumer-driver-v1\",\"model_digest\":\"%s\",\"model_identity\":\"%s\",\"layer\":17,\"capacity\":512,\"memory_bytes\":%zu,\"native_sham_steps\":16,\"native_sham_logits_and_kv_byte_exact\":true,\"sources\":[],\"programs\":[",m.digest,m.identity,sizeof(QMMemory));load_program(f,programs);reference_query=image("query-proposal-4.f32",1);uint64_t source_calls,source_macs;int selected=select_source(&m,&source_calls,&source_macs);char selection_sha[65],input_sha[65];file_hash("source-selection.json",selection_sha);file_hash("input-manifest.json",input_sha);
    fprintf(f,"],\"selected_index\":%d,\"source_selection_sha256\":\"%s\",\"target_input_manifest_sha256\":\"%s\",\"source_linear_calls\":%llu,\"source_linear_macs\":%llu,\"paths\":[",selected,selection_sha,input_sha,(unsigned long long)source_calls,(unsigned long long)source_macs);
    int comma=0;for(int task=0;task<6;task++){const int policies[]={7,12,11};for(unsigned i=0;i<3;i++){if(i==2&&(selected==0||selected==4))continue;if(comma++)fputc(',',f);path(f,&m,task,policies[i]);}}seal();
    for(int task=0;task<6;task++){fputc(',',f);path(f,&m,task,0);}fprintf(f,"],\"aliases\":[");
    if(selected==0||selected==4)for(int task=0;task<6;task++)fprintf(f,"%s{\"logical_name\":\"case%d-query-fit-memory\",\"physical_name\":\"case%d-%s\",\"reason\":\"identical selected query image and complete execution policy\"}",task?",":"",task,task,selected==0?"values-native-memory":"response-reference-memory");
    fprintf(f,"],\"candidates_sealed_before_native\":true,\"all_linear_calls\":%llu,\"all_linear_macs\":%llu,\"seconds\":%.9g}\n",(unsigned long long)all_linear_calls,(unsigned long long)all_linear_macs,seconds()-started);REQUIRE(!fclose(f));
    gpu_free(programs[0].basis);gpu_free(programs[0].query);gpu_free(programs[0].key);gpu_free(programs[0].value);gpu_free(programs[0].coordinates);gpu_free(fitted_query);gpu_free(reference_query);qwen_model_free(&m);return 0;
}
