#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#define REQUIRE(x) do{if(!(x)){fprintf(stderr,"retention %s:%d: %s\n",__FILE__,__LINE__,#x);exit(2);}}while(0)
#include "runtime/assay_io.h"
#include "ring.h"
extern void gpu_weight_configure(int);
extern void gpu_weight_release(void);
extern void gpu_weight_info(size_t*,size_t*,size_t*,int*,unsigned long long*,unsigned long long*,unsigned long long*);
static const char *directory;
static const int watched[]={0,17,27};
static const char *labels[]={"native","full_byte","recent","anchors","reservoir"};
static FILE *open_file(const char *name){char p[4096];REQUIRE(snprintf(p,sizeof(p),"%s/%s",directory,name)<(int)sizeof(p));FILE *f=fopen(p,"wb");REQUIRE(f);return f;}
static void ints(FILE *f,const int *x,int n){fputc('[',f);for(int i=0;i<n;i++)fprintf(f,"%s%d",i?",":"",x[i]);fputc(']',f);}
typedef struct {const char *stem;int position;} Boundary;
static void observe_boundary(void *context,int position,int layer,QwenPort port,const float *data,size_t n){
    Boundary *b=context;for(int i=0;i<3;i++)if(watched[i]==layer&&(port==QWEN_QUERY||port==QWEN_KEY||port==QWEN_VALUE||port==QWEN_ATTENTION)){
        REQUIRE(position==b->position);char name[192];REQUIRE(snprintf(name,sizeof(name),"%s-l%d-%s.f32",b->stem,layer,qwen_port_name(port))<(int)sizeof(name));float *h=read_floats(data,n);save(directory,name,h,n*4);free(h);
    }
}
static int step(QwenSession *s,Ring *c,int token,const char *spot){
    Boundary b={spot,s->position};if(spot){s->observer=observe_boundary;s->observer_context=&b;s->observe_position=s->position;}
    int e=c?ring_begin(c,token):qwen_begin_token(s,token);if(e)return e;
    while(s->layer<28){s->observe_layer=s->layer;e=qwen_advance_layer(s);if(e)break;}
    if(!e)e=c?ring_finish(c):qwen_finish_token(s);s->observer=NULL;s->observer_context=NULL;return e;
}
static void state_hash(QwenSession *s,Ring *c,char hash[65]){
    if(!c){prefix_digest(s,s->position,hash);return;}
    size_t n=c->arena_bytes+20+(size_t)c->capacity*4;unsigned char *h=malloc(n);REQUIRE(h);gpu_read(h,c->arena,c->arena_bytes);size_t p=c->arena_bytes;memcpy(h+p,&c->view.cursor,4);p+=4;memcpy(h+p,&c->view.population,8);p+=8;memcpy(h+p,&c->view.rng,8);p+=8;memcpy(h+p,c->positions,(size_t)c->capacity*4);digest(h,n,hash);free(h);
}
static int marker(QwenSession *s,Ring *c,uint64_t *id){return c?ring_mark(c,id):qwen_stack_push(s,id);}
static int rollback(QwenSession *s,Ring *c,uint64_t id){return c?ring_rollback(c,id):qwen_stack_rollback(s,id);}
static void boundary(FILE *f,QwenSession *s,const char *stem,const char *clock){char name[192];snprintf(name,sizeof(name),"%s-%s.f32",stem,clock);fprintf(f,",\"%s\":",clock);readout(f,s,1151,12203,directory,name);}
static int turn(FILE *f,QwenSession *s,Ring *c,const char *stem,int index,const int *ids,int count,int budget,int last){
    int skip=last==ids[0],start=s->position;char spot[192];
    fprintf(f,"{\"name\":\"%s\",\"input_index\":%d,\"parent_position\":%d,\"template_skip\":%d,\"steps\":[",stem,index,start,skip);
    for(int j=skip;j<count;j++){snprintf(spot,sizeof(spot),"%s-prefill",stem);int p=s->position;REQUIRE(!step(s,c,ids[j],j==count-1?spot:NULL));fprintf(f,"%s[%d,%d,%d]",j>skip?",":"",p,ids[j],s->pending);}
    fputc(']',f);boundary(f,s,stem,"prefill");fputs(",\"generation_steps\":[",f);int answer[32],n=0,stopped=0;
    for(int j=0;j<budget;j++){int token=s->pending,p=s->position;answer[n++]=token;stopped=token==s->model->eos||token==s->model->eos_extra;snprintf(spot,sizeof(spot),"%s-final",stem);REQUIRE(!step(s,c,token,stopped||j==budget-1?spot:NULL));fprintf(f,"%s[%d,%d,%d]",j?",":"",p,token,s->pending);if(stopped)break;}
    fputs("],\"answer_ids\":",f);ints(f,answer,n);fprintf(f,",\"stopped\":%s,\"truncated\":%s,\"all_delivered_consumed\":true,\"final_position\":%d",stopped?"true":"false",stopped?"false":"true",s->position);boundary(f,s,stem,"final");fputc('}',f);REQUIRE(!fflush(f));fprintf(stderr,"%s cursor%d answer%d stopped%d\n",stem,s->position,n,stopped);return answer[n-1];
}
static void snapshot_save(Ring *c,const char *name){
    FILE *f=open_file(name);uint32_t h[]={1,28,(uint32_t)c->capacity,(uint32_t)c->anchors,(uint32_t)c->recent,(uint32_t)c->sample,(uint32_t)c->view.cursor};REQUIRE(fwrite("TCMRING1",1,8,f)==8&&fwrite(h,4,7,f)==7&&fwrite(&c->view.population,8,1,f)==1&&fwrite(&c->view.rng,8,1,f)==1&&fwrite(c->positions,4,(size_t)c->capacity,f)==(size_t)c->capacity);void *raw=malloc(c->arena_bytes);REQUIRE(raw);gpu_read(raw,c->arena,c->arena_bytes);REQUIRE(fwrite(raw,1,c->arena_bytes,f)==c->arena_bytes);free(raw);REQUIRE(!fclose(f));
}
static void mechanics(FILE *f,QwenSession *s,Ring *c){
    char root[65],first[65],replay[65],after[65];uint64_t parent,child;int p=s->position,a=s->pending,b=17,sibling=23;state_hash(s,c,root);FILE *rows=c?c->rows:NULL;if(c)c->rows=NULL;
    REQUIRE(!marker(s,c,&parent)&&!step(s,c,a,NULL)&&!marker(s,c,&child)&&!step(s,c,b,NULL));state_hash(s,c,first);float *y=read_floats(s->logits,151936);REQUIRE(!rollback(s,c,child)&&!step(s,c,sibling,NULL)&&!rollback(s,c,parent));state_hash(s,c,after);REQUIRE(!strcmp(root,after));
    REQUIRE(!marker(s,c,&parent)&&!step(s,c,a,NULL)&&!step(s,c,b,NULL));state_hash(s,c,replay);float *z=read_floats(s->logits,151936);REQUIRE(!memcmp(y,z,151936*4)&&!strcmp(first,replay));free(y);free(z);REQUIRE(!rollback(s,c,parent));state_hash(s,c,after);REQUIRE(!strcmp(root,after));
    int refused=0,partial=0;if(c){
        QwenSession before=*s;REQUIRE(ring_rollback(c,parent));REQUIRE(!memcmp(&before,s,sizeof(*s)));state_hash(s,c,after);REQUIRE(!strcmp(root,after));refused++;
        QwenCheckpoint detached;REQUIRE(!qwen_checkpoint(s,&detached));before=*s;REQUIRE(qwen_restore(s,&detached));REQUIRE(!memcmp(&before,s,sizeof(*s)));state_hash(s,c,after);REQUIRE(!strcmp(root,after));refused++;
        REQUIRE(!ring_mark(c,&parent)&&!ring_mark(c,&child));before=*s;uint64_t third;REQUIRE(ring_mark(c,&third));REQUIRE(!memcmp(&before,s,sizeof(*s)));REQUIRE(ring_rollback(c,parent));REQUIRE(!memcmp(&before,s,sizeof(*s)));REQUIRE(!ring_commit(c,child)&&!ring_commit(c,parent));refused+=2;
        c->fail_layer=27;c->fail_position=p;REQUIRE(!ring_begin(c,a));int e=0;while(s->layer<28&&!e)e=qwen_advance_layer(s);REQUIRE(e==-77&&s->layer==27&&s->position==p);partial=1;REQUIRE(!ring_abort(c));state_hash(s,c,after);REQUIRE(!strcmp(root,after));REQUIRE(!ring_mark(c,&parent)&&!ring_push(c,a)&&!ring_rollback(c,parent));state_hash(s,c,after);REQUIRE(!strcmp(root,after));
    }
    int replacement_abort=0,replacement_position=-1;
    if(c&&c->sample){
        REQUIRE(!ring_mark(c,&parent));for(int trial=0;trial<128&&!replacement_abort;trial++){
            char prior_hash[65],aborted_hash[65];state_hash(s,c,prior_hash);int pending=s->pending;TcmTop2 top=s->top;float *prior_logits=read_floats(s->logits,151936);int position=s->position;REQUIRE(!ring_begin(c,a));
            int replace=c->sample_slot>=0&&c->cuts[0].positions[c->sample_slot]>=0;if(replace){c->fail_layer=27;c->fail_position=position;}
            int e=0;while(s->layer<28&&!e)e=qwen_advance_layer(s);
            if(replace){REQUIRE(e==-77&&!ring_abort(c));state_hash(s,c,aborted_hash);float *actual=read_floats(s->logits,151936);REQUIRE(!strcmp(prior_hash,aborted_hash)&&s->pending==pending&&!memcmp(&top,&s->top,sizeof(top))&&!memcmp(prior_logits,actual,151936*4));free(actual);replacement_abort=1;replacement_position=position;}
            else REQUIRE(!e&&!ring_finish(c));free(prior_logits);
        }REQUIRE(!ring_rollback(c,parent));state_hash(s,c,after);REQUIRE(!strcmp(root,after));
    }
    if(c)c->rows=rows;
    fprintf(f,"{\"root_position\":%d,\"root_sha256\":\"%s\",\"replayed_sha256\":\"%s\",\"exact_replay\":true,\"root_restored\":true,\"partial_last_layer_abort\":%s,\"nonmutating_refusals\":%d,\"occupied_sample_replacement_abort\":%s,\"replacement_position\":%d}",p,root,replay,partial?"true":"false",refused,replacement_abort?"true":"false",replacement_position);
}
int main(int argc,char **argv){
    REQUIRE(argc==4);directory=argv[3];int count[8],*ids[8];char path[4096];for(int i=0;i<8;i++){snprintf(path,sizeof(path),"%s/input-%d.txt",argv[2],i);ids[i]=tokens(path,&count[i]);}
    double started=clock_seconds();gpu_weight_configure(2);TcmModel bundle;QwenModel m;char error[256];REQUIRE(!tcm_load(&bundle,argv[1],error,sizeof(error))&&!qwen_model_init(&m,&bundle,error,sizeof(error)));tcm_unload(&bundle);REQUIRE(m.layers==28&&m.hidden==1536&&m.vocab==151936);
    FILE *f=open_file("driver.json");fprintf(f,"{\"schema\":\"tcm-bounded-retention-driver-v1\",\"model_digest\":\"%s\",\"logical_capacity\":768,\"arms\":[",m.digest);
    for(int mode=0;mode<5;mode++){
        QwenSession s;Ring cache,*c=mode?&cache:NULL;if(c)ring_open(c,&s,&m,768,mode>=3?16:0,mode==1?768:64,mode==4?32:0,mode==1);else REQUIRE(!qwen_session_init(&s,&m,768));
        size_t steady=gpu_live_bytes();char name[192];if(c){snprintf(name,sizeof(name),"%s-rows.bin",labels[mode]);c->rows=open_file(name);}
        fprintf(f,"%s{\"name\":\"%s\",\"state_gpu_bytes\":%zu,\"cache_gpu_bytes\":%zu,\"recovery_gpu_bytes\":%zu,\"cache_host_bytes\":%zu,\"tracked_model_state_gpu_bytes\":%zu,\"support_capacity\":%d,\"turns\":[",mode?",":"",labels[mode],s.allocated_bytes+(c?c->gpu_bytes:0),c?c->arena_bytes:(size_t)28*768*2048,c?3*(c->arena_bytes+(size_t)m.vocab*4):0,c?c->host_bytes:0,steady,c?c->capacity:768);
        snprintf(name,sizeof(name),"%s-source",labels[mode]);int last=turn(f,&s,c,name,0,ids[0],count[0],8,-1);uint64_t mark;REQUIRE(!marker(&s,c,&mark));fputc(',',f);snprintf(name,sizeof(name),"%s-early",labels[mode]);turn(f,&s,c,name,3,ids[3],count[3],8,last);REQUIRE(!rollback(&s,c,mark));
        fputc(',',f);snprintf(name,sizeof(name),"%s-correction",labels[mode]);last=turn(f,&s,c,name,1,ids[1],count[1],8,last);fputc(',',f);snprintf(name,sizeof(name),"%s-delay",labels[mode]);last=turn(f,&s,c,name,2,ids[2],count[2],8,last);
        int root=s.position;char root_hash[65];state_hash(&s,c,root_hash);if(c){snprintf(name,sizeof(name),"%s-root.bin",labels[mode]);snapshot_save(c,name);}
        for(int q=0;q<4;q++){REQUIRE(!marker(&s,c,&mark));fputc(',',f);snprintf(name,sizeof(name),"%s-late%d",labels[mode],q);turn(f,&s,c,name,q+4,ids[q+4],count[q+4],q==3?16:8,last);REQUIRE(!rollback(&s,c,mark));char h[65];state_hash(&s,c,h);REQUIRE(!strcmp(h,root_hash)&&s.position==root);}
        fprintf(f,"],\"root_position\":%d,\"root_sha256\":\"%s\",\"mechanics\":",root,root_hash);mechanics(f,&s,c);
        fprintf(f,",\"physical_cache_calls\":%llu,\"physical_cache_publishes\":%llu,\"cache_restores\":%llu}",c?c->calls:0,c?c->published:0,c?c->restores:0);REQUIRE(!fflush(f));
        if(c){REQUIRE(!fclose(c->rows));c->rows=NULL;REQUIRE(!qwen_reset(&s)&&c->view.cursor==0&&c->view.population==0);for(int i=0;i<c->capacity;i++)REQUIRE(c->positions[i]==-1);ring_close(c);}qwen_session_free(&s);
    }
    size_t matrices,vectors,tile;int matrix_count;unsigned long long matvecs,tiles,embeddings;gpu_weight_info(&matrices,&vectors,&tile,&matrix_count,&matvecs,&tiles,&embeddings);
    fprintf(f,"],\"matrix_bytes\":%zu,\"vector_bytes\":%zu,\"weight_tile_bytes\":%zu,\"matrix_count\":%d,\"matvecs\":%llu,\"tiles\":%llu,\"embeddings\":%llu,\"peak_tracked_gpu_bytes\":%zu,\"seconds\":%.9g}\n",matrices,vectors,tile,matrix_count,matvecs,tiles,embeddings,gpu_peak_bytes(),clock_seconds()-started);REQUIRE(!fclose(f));qwen_model_free(&m);gpu_weight_release();REQUIRE(!gpu_live_bytes());for(int i=0;i<8;i++)free(ids[i]);return 0;
}
