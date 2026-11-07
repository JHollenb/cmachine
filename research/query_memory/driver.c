#define _POSIX_C_SOURCE 200809L
#include "qwen.h"
#include "memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>
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
    QwenSession *session;int factor,compact,zero,rank,count;
    QMMemory memory;Record *records;int ids[LIMIT],tops[LIMIT];float current_k[256],current_v[256];unsigned cache_mask;
    uint64_t linear_calls,linear_macs,native_qkv_calls,compiled_qkv_calls,native_attention_calls,compact_reads,zero_reads,reader_products,merge_distance_products,serving_transfer_bytes;
} Run;
static Run *active;
typedef struct {int rank;float *basis,*query,*key,*value,*coordinates;} Program;
static Program programs[2],*active_program;
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
    if((q||k||v)&&r->factor){
        REQUIRE(active_program&&active_program->rank==r->rank);
        if(q){linear(gpu,active_program->basis,1536,r->rank,x,active_program->coordinates);if(r->records)finite_read(r->records[r->count].coeff,active_program->coordinates,r->rank);}
        linear(gpu,q?active_program->query:k?active_program->key:active_program->value,r->rank,out,active_program->coordinates,y);r->compiled_qkv_calls++;
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
        else{gpu_attention(gpu,out,q,keys,values,scores,qheads,kvheads,dimension,capacity,length);r->native_attention_calls++;r->reader_products+=(uint64_t)12*length*128*2;}
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
    REQUIRE(r->count<LIMIT&&r->session->position==r->count&&!r->cache_mask);active=r;
    if(r->records){r->session->observer=observer;r->session->observer_context=r;r->session->observe_layer=SITE;r->session->observe_position=r->count;}
    REQUIRE(!qwen_push(r->session,token));r->ids[r->count]=token;r->tops[r->count++]=r->session->pending;active=NULL;
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
    struct {const char *name;size_t offset,n;} fields[]={{"norm",offsetof(Record,norm),1536},{"q",offsetof(Record,q),1536},{"k",offsetof(Record,k),256},{"v",offsetof(Record,v),256},{"attention",offsetof(Record,attention),1536},{"output",offsetof(Record,output),1536},{"coeff",offsetof(Record,coeff),RANK}};
    for(unsigned field=0;field<sizeof(fields)/sizeof(*fields);field++){
        float *host=malloc((size_t)r->count*fields[field].n*4);REQUIRE(host);
        for(int i=0;i<r->count;i++)memcpy(host+(size_t)i*fields[field].n,(char*)(r->records+i)+fields[field].offset,fields[field].n*4);
        char file[192];snprintf(file,sizeof(file),"%s-%s.f32",name,fields[field].name);save(file,host,(size_t)r->count*fields[field].n*4);free(host);
    }
}
static void compile_program(const QwenModel *m,FILE *f,Program *program,int rank){
    char file[4096];snprintf(file,sizeof(file),"%s/weight-basis%d.f32",directory,rank);FILE *source=fopen(file,"rb");float *basis=malloc((size_t)rank*1536*4);REQUIRE(source&&basis&&fread(basis,4,(size_t)rank*1536,source)==(size_t)rank*1536&&fgetc(source)==EOF&&!fclose(source));for(int j=0;j<rank*1536;j++)REQUIRE(isfinite(basis[j]));
    snprintf(file,sizeof(file),"program%d-basis.f32",rank);save(file,basis,(size_t)rank*1536*4);program->rank=rank;program->basis=gpu_weight(basis,TCM_F32,(size_t)rank*1536);program->coordinates=gpu_alloc((size_t)rank*4);
    uint64_t calls=all_linear_calls,macs=all_linear_macs;float *input=gpu_alloc(1536*4),*result=gpu_alloc(1536*4);float host[1536];
    const float *weights[]={m->layer[SITE].q,m->layer[SITE].k,m->layer[SITE].v};const char *labels[]={"q","k","v"};float **outputs[]={&program->query,&program->key,&program->value};
    for(int kind=0;kind<3;kind++){
        int width=kind?256:1536;float *matrix=malloc((size_t)width*rank*4);REQUIRE(matrix);
        for(int col=0;col<rank;col++){gpu_write(input,basis+(size_t)col*1536,1536*4);linear(m->gpu,weights[kind],1536,width,input,result);finite_read(host,result,width);for(int row=0;row<width;row++)matrix[(size_t)row*rank+col]=host[row];}
        char output[128];snprintf(output,sizeof(output),"program%d-%s.f32",rank,labels[kind]);save(output,matrix,(size_t)width*rank*4);*outputs[kind]=gpu_weight(matrix,TCM_F32,(size_t)width*rank);free(matrix);
    }
    gpu_free(input);gpu_free(result);free(basis);
    fprintf(f,"{\"rank\":%d,\"source_rows\":193,\"basis_kind\":\"authenticated-source-activation-prefix32-plus96\",\"compile_linear_calls\":%llu,\"compile_linear_macs\":%llu,\"dynamic_qkv_macs\":%d,\"native_qkv_macs\":3145728,\"immutable_numeric_bytes\":%d}",rank,(unsigned long long)(all_linear_calls-calls),(unsigned long long)(all_linear_macs-macs),3584*rank,4*3584*rank);
}
static void seal(void){
    DIR *dir=opendir(directory);REQUIRE(dir);FILE *f;char file[4096];snprintf(file,sizeof(file),"%s/candidate-seal.json",directory);f=fopen(file,"w");REQUIRE(f);fprintf(f,"{\"files\":[");struct dirent *entry;int comma=0;
    while((entry=readdir(dir))){size_t n=strlen(entry->d_name);if(n<4||(!strstr(entry->d_name,".f32")&&!strstr(entry->d_name,".bin")&&!strstr(entry->d_name,".i32")))continue;
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
    static const char *labels[]={"native","project32","memory","both32","zero","project128","both128"};int n,follow_n;char stem[128],name[128];snprintf(stem,sizeof(stem),"case%d-prompt",task);int *prompt=tokens(stem,&n);snprintf(stem,sizeof(stem),"case%d-followup",task);int *follow=tokens(stem,&follow_n);snprintf(name,sizeof(name),"case%d-%s",task,labels[policy]);
    QwenSession s;fresh(&s,m);Run *r=calloc(1,sizeof(*r));REQUIRE(r);r->session=&s;r->factor=policy==1||policy==3||policy==5||policy==6;r->compact=policy==2||policy==3||policy==6;r->zero=policy==4;r->rank=r->factor?(policy>=5?128:32):0;active_program=r->factor?programs+(r->rank==128):NULL;r->records=calloc(LIMIT,sizeof(Record));REQUIRE(r->records);double start=seconds();
    fprintf(f,"{\"name\":\"%s\",\"task\":%d,\"policy\":%d,\"rank\":%d,\"factor\":%s,\"compact\":%s,\"zero\":%s,\"turns\":[",name,task,policy,r->rank,r->factor?"true":"false",r->compact?"true":"false",r->zero?"true":"false");
    for(int turn=0;turn<2;turn++){
        if(turn)fputc(',',f);int start_position=r->count,*input=turn?follow:prompt,count=turn?follow_n:n,skip=turn&&r->ids[r->count-1]==input[0];
        REQUIRE(!turn||input[0]==151645);for(int j=skip;j<count;j++)advance(r,input[j]);
        int budget=task==0?8:task==1?16:32;fprintf(f,"{\"turn\":%d,\"start\":%d,\"ingested\":%d,\"template_skip\":%d,\"budget\":%d",turn,start_position,count-skip,skip,budget);boundary(f,r,name,turn,"prefill");
        int answers[32],length=0,stopped=0;for(int j=0;j<budget;j++){int token=s.pending;answers[length++]=token;stopped=token==m->eos||token==m->eos_extra;advance(r,token);if(stopped)break;}
        fprintf(f,",\"answer_ids\":");ints(f,answers,length);fprintf(f,",\"stopped\":%s,\"truncated\":%s",stopped?"true":"false",stopped?"false":"true");boundary(f,r,name,turn,"final");fputc('}',f);REQUIRE(!fflush(f));
    }
    int sentinel=1;if(r->compact){size_t base=(size_t)SITE*2*512*128,ncache=(size_t)2*512*128;float *host=malloc(ncache*4);REQUIRE(host);for(int kind=0;kind<2;kind++){finite_read(host,(kind?s.values:s.keys)+base,ncache);for(size_t j=0;j<ncache;j++)if(host[j]!=0)sentinel=0;}free(host);REQUIRE(sentinel&&r->memory.next==(uint64_t)r->count);}
    record_arrays(r,name);fprintf(f,"],\"consumed_ids\":");ints(f,r->ids,r->count);fprintf(f,",\"count\":%d,\"linear_calls\":%llu,\"linear_macs\":%llu,\"native_qkv_calls\":%llu,\"compiled_qkv_calls\":%llu,\"native_attention_calls\":%llu,\"compact_reads\":%llu,\"zero_reads\":%llu,\"reader_products\":%llu,\"merge_distance_products\":%llu,\"serving_transfer_bytes\":%llu,\"selected_full_kv_untouched\":%s,\"memory_bytes\":%zu,\"seconds\":%.9g}",r->count,(unsigned long long)r->linear_calls,(unsigned long long)r->linear_macs,(unsigned long long)r->native_qkv_calls,(unsigned long long)r->compiled_qkv_calls,(unsigned long long)r->native_attention_calls,(unsigned long long)r->compact_reads,(unsigned long long)r->zero_reads,(unsigned long long)r->reader_products,(unsigned long long)r->merge_distance_products,(unsigned long long)r->serving_transfer_bytes,r->compact&&sentinel?"true":"false",sizeof(QMMemory),seconds()-start);REQUIRE(!fflush(f));
    fprintf(stderr,"completed %s:%d own advances,%.3fs\n",name,r->count,seconds()-start);free(r->records);free(r);qwen_session_free(&s);free(prompt);free(follow);
}
int main(int argc,char **argv){
    REQUIRE(argc==3);directory=argv[2];TcmModel bundle;QwenModel m;char error[256],file[4096];double started=seconds();REQUIRE(!tcm_load(&bundle,argv[1],error,sizeof(error))&&!qwen_model_init(&m,&bundle,error,sizeof(error)));tcm_unload(&bundle);
    native_sham(&m);snprintf(file,sizeof(file),"%s/driver.json",directory);FILE *f=fopen(file,"w");REQUIRE(f);fprintf(f,"{\"schema\":\"tcm-query-memory-weight-driver-v2\",\"model_digest\":\"%s\",\"model_identity\":\"%s\",\"layer\":17,\"capacity\":512,\"memory_bytes\":%zu,\"native_sham_steps\":16,\"native_sham_logits_and_kv_byte_exact\":true,\"sources\":[],\"programs\":[",m.digest,m.identity,sizeof(QMMemory));
    compile_program(&m,f,programs,32);fputc(',',f);compile_program(&m,f,programs+1,128);fprintf(f,"],\"paths\":[");
    int comma=0;for(int task=0;task<3;task++)for(int policy=1;policy<=6;policy++){if(comma++)fputc(',',f);path(f,&m,task,policy);}seal();
    for(int task=0;task<3;task++){fputc(',',f);path(f,&m,task,0);}
    fprintf(f,"],\"candidates_sealed_before_native\":true,\"all_linear_calls\":%llu,\"all_linear_macs\":%llu,\"seconds\":%.9g}\n",(unsigned long long)all_linear_calls,(unsigned long long)all_linear_macs,seconds()-started);REQUIRE(!fclose(f));
    for(int i=0;i<2;i++){gpu_free(programs[i].basis);gpu_free(programs[i].query);gpu_free(programs[i].key);gpu_free(programs[i].value);gpu_free(programs[i].coordinates);}qwen_model_free(&m);return 0;
}
