#include "turn_helpers.h"
#include "cache.h"
static const char *directory;
static const int watched[]={0,17,18,27};
static FILE *row_file,*query_file;
static void array_save(const char *name,const void *data,size_t bytes){save(directory,name,(void*)data,bytes);}
static void device_save(const char *name,const float *data,size_t n){float *host=read_floats(data,n);for(size_t i=0;i<n;i++)REQUIRE(isfinite(host[i]));array_save(name,host,n*4);free(host);}
static void logical_digest(const QwenSession *s,OwnedCache *cache,char out[65]){
    size_t plane=(size_t)2*s->position*128,total=2*28*plane,offset=0;float *host=malloc(total*4);REQUIRE(host);
    for(int kind=0;kind<2;kind++)for(int l=0;l<28;l++){cache_read_plane(s,cache,kind,l,s->position,host+offset);offset+=plane;}
    for(size_t i=0;i<total;i++)REQUIRE(isfinite(host[i]));digest(host,total*4,out);free(host);
}
static FILE *artifact_open(const char *name){char path[4096];REQUIRE(snprintf(path,sizeof(path),"%s/%s",directory,name)<(int)sizeof(path));FILE *f=fopen(path,"wb");REQUIRE(f);return f;}
typedef struct {QwenSession *s;const char *stem,*spot;unsigned ports[28];float rows[28][2][2][128],queries[4][12][128];} AllCapture;
static void all_observer(void *context,int position,int layer,QwenPort port,const float *data,size_t n){
    AllCapture *c=context;REQUIRE(layer>=0&&layer<28&&position==c->s->position&&!(c->ports[layer]&(1u<<port)));c->ports[layer]|=1u<<port;
    if(port==QWEN_KEY||port==QWEN_VALUE){REQUIRE(n==256);gpu_read(c->rows[layer][port==QWEN_VALUE],data,n*4);}
    for(int i=0;i<4;i++)if(watched[i]==layer){
        if(port==QWEN_QUERY){REQUIRE(n==1536);gpu_read(c->queries[i],data,n*4);}
        if(c->spot&&(port==QWEN_INPUT||port==QWEN_ATTENTION||port==QWEN_ATTENTION_WRITE||port==QWEN_MLP_RETURN||port==QWEN_OUTPUT||port==QWEN_RAW_RECEIVER)){
            char name[192];snprintf(name,sizeof(name),"%s-%s-l%d-%s.f32",c->stem,c->spot,layer,qwen_port_name(port));device_save(name,data,n);}
    }
}
static void all_step(FILE *f,QwenSession *s,OwnedCache *cache,int token,History *history,const char *stem,const char *spot,int index,const char *role,int comma){
    AllCapture c={.s=s,.stem=stem,.spot=spot};int position=s->position;s->observer=all_observer;s->observer_context=&c;s->observe_position=position;
    REQUIRE(!qwen_begin_token(s,token));while(s->layer<28){s->observe_layer=s->layer;REQUIRE(!qwen_advance_layer(s));}REQUIRE(!qwen_finish_token(s));s->observer=NULL;s->observer_context=NULL;
    for(int l=0;l<28;l++){REQUIRE(c.ports[l]==((1u<<QWEN_PORT_COUNT)-1));if(cache)REQUIRE(cache->layers[l].cursor==s->position);}
    REQUIRE(fwrite(c.rows,1,sizeof(c.rows),row_file)==sizeof(c.rows)&&fwrite(c.queries,1,sizeof(c.queries),query_file)==sizeof(c.queries));
    REQUIRE(history->count<512);history->ids[history->count++]=token;char hh[65];history_digest(history,hh);
    if(comma)fputc(',',f);fprintf(f,"{\"step\":%d,\"position\":%d,\"input_id\":%d,\"role\":\"%s\",\"top\":%d,\"history_sha256\":\"%s\",\"capture\":",index,position,token,role,s->pending,hh);if(spot)fprintf(f,"\"%s\"",spot);else fputs("null",f);
    fprintf(f,",\"encoded_cursor_count\":%d,\"all_encoded_cursors_at_position\":true}",cache?28:0);REQUIRE(!fflush(f));
}
static void all_boundary(FILE *f,const QwenSession *s,OwnedCache *cache,const char *name,const char *clock){
    char file[128],hash[65];snprintf(file,sizeof(file),"%s-%s.f32",name,clock);fprintf(f,",\"%s\":",clock);readout(f,s,1151,12203,directory,file);logical_digest(s,cache,hash);fprintf(f,",\"%s_logical_cache_sha256\":\"%s\"",clock,hash);
    if(cache&&cache->layers[0].byte&&!strcmp(clock,"final")){snprintf(file,sizeof(file),"%s-cache.bin",name);FILE *out=artifact_open(file);cache_snapshot(out,cache,s->position);REQUIRE(!fclose(out));}
}
static void dialog_turn(FILE *f,QwenSession *s,OwnedCache *cache,const char *label,int t,const int *ids,int count,int budget,History *history){
    int skip=history->count&&history->ids[history->count-1]==ids[0],prefill=count-skip;char stem[96];snprintf(stem,sizeof(stem),"%s-t%d",label,t);REQUIRE(prefill>0&&budget<=32);
    fprintf(f,"{\"name\":\"%s\",\"turn\":%d,\"budget\":%d,\"parent_position\":%d,\"parent_history_ids\":",stem,t,budget,s->position);ints(f,history->ids,history->count);fputs(",\"template_ids\":",f);ints(f,ids,count);fprintf(f,",\"template_skip\":%d,\"steps\":[",skip);
    for(int j=0;j<prefill;j++)all_step(f,s,cache,ids[j+skip],history,stem,j==prefill-1?"prefill":NULL,j,"template-user",j>0);fputc(']',f);all_boundary(f,s,cache,stem,"prefill");fputs(",\"generation_steps\":[",f);
    int answer[32],n=0,stopped=0;for(int j=0;j<budget;j++){int token=s->pending;answer[n++]=token;stopped=token==s->model->eos||(s->model->eos_extra>0&&token==s->model->eos_extra);all_step(f,s,cache,token,history,stem,stopped||j==budget-1?"final":NULL,prefill+j,"own-generated",j>0);if(stopped)break;}
    fputs("],\"answer_ids\":",f);ints(f,answer,n);fprintf(f,",\"stopped\":%s,\"truncated\":%s,\"all_delivered_consumed\":true,\"final_position\":%d",stopped?"true":"false",stopped?"false":"true",s->position);all_boundary(f,s,cache,stem,"final");fputc('}',f);REQUIRE(!fflush(f));fprintf(stderr,"completed %s cursor%d: %d own tokens,stopped=%d\n",stem,s->position,n,stopped);
}
static void mechanics(FILE *f,QwenSession *s,OwnedCache *cache,const char *label,int a,int b,int sibling){
    char root[65],first[65],replay[65],after[65],yhash[65],name[128];logical_digest(s,cache,root);int position=s->position;uint64_t parent,child;
    REQUIRE(!qwen_stack_push(s,&parent)&&!qwen_push(s,a)&&!qwen_stack_push(s,&child)&&!qwen_push(s,b));logical_digest(s,cache,first);float *y=read_floats(s->logits,151936);digest(y,151936*4,yhash);snprintf(name,sizeof(name),"%s-rollback-reference.f32",label);array_save(name,y,151936*4);
    REQUIRE(!qwen_stack_rollback(s,child)&&!qwen_push(s,sibling)&&!qwen_stack_rollback(s,parent));REQUIRE(!qwen_stack_push(s,&parent)&&!qwen_push(s,a)&&!qwen_push(s,b));logical_digest(s,cache,replay);float *z=read_floats(s->logits,151936);REQUIRE(!memcmp(y,z,151936*4)&&!strcmp(first,replay));free(y);free(z);REQUIRE(!qwen_stack_rollback(s,parent));logical_digest(s,cache,after);REQUIRE(!strcmp(root,after));
    int injected=cache&&cache->layers[0].byte,partial[28]={0};
    if(injected){REQUIRE(!qwen_stack_push(s,&parent));cache->fail_layer=27;cache->fail_position=position;REQUIRE(!qwen_begin_token(s,a));int error=0;while(s->layer<28&&!error)error=qwen_advance_layer(s);REQUIRE(error==-77&&s->phase==2);for(int i=0;i<28;i++)partial[i]=cache->layers[i].cursor;
        REQUIRE(!qwen_abort_token(s)&&s->phase==0&&s->position==position);logical_digest(s,cache,after);REQUIRE(!strcmp(root,after));for(int i=0;i<28;i++)REQUIRE(cache->layers[i].cursor==position);
        REQUIRE(!qwen_stack_rollback(s,parent)&&!qwen_stack_push(s,&parent)&&!qwen_push(s,a)&&!qwen_stack_rollback(s,parent));logical_digest(s,cache,after);REQUIRE(!strcmp(root,after));}
    fprintf(f,"{\"root_position\":%d,\"root_logical_cache_sha256\":\"%s\",\"two_token_logical_cache_sha256\":\"%s\",\"replayed_logical_cache_sha256\":\"%s\",\"full_logits_sha256\":\"%s\",\"suffix_replaced\":true,\"root_restored\":true,\"partial_abort\":%s,\"partial_cursors\":",position,root,first,replay,yhash,injected?"true":"false");ints(f,partial,28);fputc('}',f);
}
int main(int argc,char **argv){
    REQUIRE(argc==8);directory=argv[7];int counts[5],*question[5];for(int t=0;t<5;t++)question[t]=tokens(argv[t+2],&counts[t]);const int budgets[]={8,8,8,32,8};const char *names[]={"native","float_all","byte_all"};
    TcmModel bundle;QwenModel m;char error[256];double started=clock_seconds();REQUIRE(!tcm_load(&bundle,argv[1],error,sizeof(error))&&!qwen_model_init(&m,&bundle,error,sizeof(error)));tcm_unload(&bundle);REQUIRE(m.hidden==1536&&m.layers==28&&m.qheads==12&&m.kvheads==2&&m.head==128&&m.vocab==151936);size_t model_live=gpu_live_bytes();FILE *f=artifact_open("driver.json");
    fprintf(f,"{\"schema\":\"tcm-compact-all-driver-v1\",\"model_digest\":\"%s\",\"capacity\":512,\"starts_empty\":true,\"watched_layers\":[0,17,18,27],\"model_live_gpu_bytes\":%zu,\"chains\":[",m.digest,model_live);
    for(int mode=0;mode<3;mode++){
        QwenSession s;OwnedCache cache,*c=mode?&cache:NULL;REQUIRE(gpu_live_bytes()==model_live);
        if(mode){cache_open(&cache,&s,mode==2);REQUIRE(!qwen_session_init_layout(&s,&m,512,(UINT64_C(1)<<28)-1,owned_step,owned_rewind,&cache));cache_bind(&cache);REQUIRE(!s.keys&&!s.values&&!s.dense_layers);for(int l=0;l<28;l++)REQUIRE(s.cache_slots[l]==-1);}
        else REQUIRE(!qwen_session_init(&s,&m,512));size_t steady=gpu_live_bytes(),state=s.allocated_bytes+(c?c->arena_bytes+c->workspace_bytes+(c->invalid?4:0):0);REQUIRE(steady==model_live+state);
        char name[128];snprintf(name,sizeof(name),"%s-rows.f32",names[mode]);row_file=artifact_open(name);snprintf(name,sizeof(name),"%s-queries.f32",names[mode]);query_file=artifact_open(name);History history={0};
        if(mode)fputc(',',f);fprintf(f,"{\"policy\":\"%s\",\"mode\":%d,\"dense_layers\":%d,\"session_gpu_bytes\":%zu,\"cache_arena_bytes\":%zu,\"decode_workspace_bytes\":%zu,\"status_bytes\":%d,\"steady_state_gpu_bytes\":%zu,\"tracked_live_gpu_bytes\":%zu,\"turns\":[",names[mode],mode,s.dense_layers,s.allocated_bytes,c?c->arena_bytes:0,c?c->workspace_bytes:0,c&&c->invalid?4:0,state,steady);
        for(int t=0;t<5;t++){if(t==1||t>2)fputc(',',f);dialog_turn(f,&s,c,names[mode],t,question[t],counts[t],budgets[t],&history);if(t==1){fputs("],\"mechanics\":",f);mechanics(f,&s,c,names[mode],question[2][1],question[2][2],question[2][3]);fputs(",\"later_turns\":[",f);}}
        fprintf(f,"],\"main_transitions\":%d,\"owned_step_calls_including_mechanics\":%lu,\"published_rows_including_mechanics\":%lu,\"cursor_rewinds\":%lu}",s.position,c?c->calls:0,c?c->published:0,c?c->rewinds:0);REQUIRE(!fclose(row_file)&&!fclose(query_file));qwen_session_free(&s);if(c)cache_close(c);REQUIRE(gpu_live_bytes()==model_live);
    }
    fprintf(f,"],\"model_weight_bytes\":%zu,\"peak_tracked_gpu_bytes\":%zu,\"seconds\":%.9g}\n",m.weight_bytes,gpu_peak_bytes(),clock_seconds()-started);REQUIRE(!fclose(f));qwen_model_free(&m);REQUIRE(!gpu_live_bytes());for(int i=0;i<5;i++)free(question[i]);return 0;
}
