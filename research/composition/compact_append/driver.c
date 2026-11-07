#include "turn_helpers.h"
#include "cache.h"
static const char *directory;
enum {ROOT_ROWS=93,CAPACITY=512};
static void array_save(const char *name,const void *host,size_t bytes){save(directory,name,(void*)host,bytes);}
static void device_save(const char *name,const float *data,size_t n){float *host=read_floats(data,n);for(size_t i=0;i<n;i++)REQUIRE(isfinite(host[i]));array_save(name,host,n*4);free(host);}
static void logical_digest(const QwenSession *s,OwnedCache *cache,char out[65]){
    size_t plane=(size_t)2*s->position*128,total=(size_t)2*28*plane,offset=0;float *host=malloc(total*4);REQUIRE(host);
    for(int kind=0;kind<2;kind++)for(int l=0;l<28;l++){cache_read_plane(s,cache,kind,l,s->position,host+offset);offset+=plane;}
    for(size_t i=0;i<total;i++)REQUIRE(isfinite(host[i]));digest(host,total*4,out);free(host);
}
static unsigned char *file_read(const char *name,size_t *bytes){char path[4096];REQUIRE(snprintf(path,sizeof(path),"%s/%s",directory,name)<(int)sizeof(path));FILE *f=fopen(path,"rb");REQUIRE(f&&!fseek(f,0,SEEK_END));long n=ftell(f);REQUIRE(n>0&&!fseek(f,0,SEEK_SET));unsigned char *raw=malloc((size_t)n);REQUIRE(raw&&fread(raw,1,(size_t)n,f)==(size_t)n&&!fclose(f));*bytes=(size_t)n;return raw;}
static void root_export(const QwenSession *s){
    size_t plane=2*ROOT_ROWS*128,total=2*28*plane,offset=0;float *host=malloc(total*4);REQUIRE(host);
    for(int kind=0;kind<2;kind++)for(int l=0;l<28;l++){cache_read_plane(s,NULL,kind,l,ROOT_ROWS,host+offset);offset+=plane;}
    array_save("root-kv.f32",host,total*4);
    for(int layer=17;layer<=18;layer++){
        size_t bytes=32+ROOT_ROWS*528;unsigned char *raw=malloc(bytes);REQUIRE(raw);memcpy(raw,"TCMAPND1",8);uint32_t header[]={1,(uint32_t)layer,2,128,ROOT_ROWS,8};memcpy(raw+8,header,24);float scales[4*ROOT_ROWS];int8_t codes[4*ROOT_ROWS*128];
        for(int kind=0;kind<2;kind++)for(int h=0;h<2;h++)for(int p=0;p<ROOT_ROWS;p++){int row=(kind*2+h)*ROOT_ROWS+p;REQUIRE(!codec_row(host+((size_t)kind*28+layer)*plane+((size_t)h*ROOT_ROWS+p)*128,codes+(size_t)row*128,scales+row));}
        memcpy(raw+32,scales,sizeof(scales));memcpy(raw+32+sizeof(scales),codes,sizeof(codes));char name[64];snprintf(name,sizeof(name),"prefix-l%d.bin",layer);array_save(name,raw,bytes);free(raw);
    }memset(host,0,total*4);free(host);
}
static void root_import(QwenSession *s,OwnedCache *cache,const QwenCheckpoint *meta){
    size_t bytes;float *root=(float*)file_read("root-kv.f32",&bytes);size_t plane=2*ROOT_ROWS*128;REQUIRE(bytes==2*28*plane*4);
    for(int l=0;l<28;l++){
        CacheLayer *a=cache?cache_layer(cache,l):NULL;
        if(a){
            if(a->byte){char name[64];snprintf(name,sizeof(name),"prefix-l%d.bin",l);size_t n;unsigned char *raw=file_read(name,&n);cache_import_program(a,raw,n);memset(raw,0,n);free(raw);}
            else cache_import(cache,a,ROOT_ROWS,root+(size_t)l*plane,root+(size_t)(28+l)*plane);
        }else for(int kind=0;kind<2;kind++)for(int h=0;h<2;h++)gpu_write((kind?s->values:s->keys)+((size_t)s->cache_slots[l]*2+h)*512*128,root+((size_t)kind*28+l)*plane+(size_t)h*ROOT_ROWS*128,(size_t)ROOT_ROWS*128*4);
    }
    memset(root,0,bytes);free(root);s->position=ROOT_ROWS;s->pending=meta->pending;s->delivered=meta->delivered;s->top=meta->top;s->phase=0;s->layer=0;
}
typedef struct {QwenSession *s;const char *stem,*spot;int index;unsigned ports[2];} AppendCapture;
static void append_observer(void *context,int position,int layer,QwenPort port,const float *data,size_t n){
    AppendCapture *c=context;int li=layer-17;REQUIRE(li>=0&&li<2&&position==c->s->position&&!(c->ports[li]&(1u<<port)));c->ports[li]|=1u<<port;char file[192];
    if(port==QWEN_QUERY||port==QWEN_KEY||port==QWEN_VALUE){REQUIRE(snprintf(file,sizeof(file),"%s-s%03d-l%d-%s.f32",c->stem,c->index,layer,qwen_port_name(port))<(int)sizeof(file));device_save(file,data,n);}
    if(c->spot){REQUIRE(snprintf(file,sizeof(file),"%s-%s-l%d-%s.f32",c->stem,c->spot,layer,qwen_port_name(port))<(int)sizeof(file));device_save(file,data,n);}
}
static void compact_step(FILE *f,QwenSession *s,OwnedCache *cache,int token,History *history,const char *stem,const char *spot,int index,const char *role,int comma){
    int position=s->position;AppendCapture c={s,stem,spot,index,{0,0}};s->observer=append_observer;s->observer_context=&c;s->observe_position=position;REQUIRE(!qwen_begin_token(s,token));
    while(s->layer<28){s->observe_layer=s->layer==17||s->layer==18?s->layer:-1;REQUIRE(!qwen_advance_layer(s));}REQUIRE(!qwen_finish_token(s));s->observer=NULL;s->observer_context=NULL;
    REQUIRE(c.ports[0]==((1u<<QWEN_PORT_COUNT)-1)&&c.ports[1]==((1u<<QWEN_PORT_COUNT)-1));if(cache)for(int i=0;i<cache->count;i++)REQUIRE(cache->layers[i].cursor==s->position);
    REQUIRE(history->count<512);history->ids[history->count++]=token;char hh[65];history_digest(history,hh);
    if(comma)fputc(',',f);fprintf(f,"{\"step\":%d,\"position\":%d,\"input_id\":%d,\"role\":\"%s\",\"top\":%d,\"history_sha256\":\"%s\",\"capture\":",index,position,token,role,s->pending,hh);if(spot)fprintf(f,"\"%s\"",spot);else fputs("null",f);fprintf(f,",\"encoded_cursors\":[");if(cache)for(int i=0;i<cache->count;i++){if(i)fputc(',',f);fprintf(f,"%d",cache->layers[i].cursor);}fputs("]}",f);REQUIRE(!fflush(f));
}
static void compact_boundary(FILE *f,const QwenSession *s,OwnedCache *cache,const char *name,const char *clock){
    char file[128],hash[65];snprintf(file,sizeof(file),"%s-%s.f32",name,clock);fprintf(f,",\"%s\":",clock);readout(f,s,1151,12203,directory,file);logical_digest(s,cache,hash);fprintf(f,",\"%s_logical_cache_sha256\":\"%s\"",clock,hash);
    if(cache)for(int i=0;i<cache->count;i++)if(cache->layers[i].byte){size_t bytes;unsigned char *raw=cache_serialize(cache->layers+i,s->position,&bytes);snprintf(file,sizeof(file),"%s-%s-l%d.bin",name,clock,cache->layers[i].layer);array_save(file,raw,bytes);free(raw);}
}
static void compact_turn(FILE *f,QwenSession *s,OwnedCache *cache,const char *label,int t,const int *template_ids,int count,int budget,History *history){
    int skip=history->ids[history->count-1]==template_ids[0],prefill=count-skip;char stem[96];snprintf(stem,sizeof(stem),"%s-t%d",label,t);REQUIRE(template_ids[0]==151645&&prefill>0&&budget<=32);
    fprintf(f,"{\"name\":\"%s\",\"turn\":%d,\"budget\":%d,\"parent_position\":%d,\"parent_history_ids\":",stem,t,budget,s->position);ints(f,history->ids,history->count);fprintf(f,",\"template_ids\":");ints(f,template_ids,count);fprintf(f,",\"template_skip\":%d,\"steps\":[",skip);
    for(int j=0;j<prefill;j++)compact_step(f,s,cache,template_ids[j+skip],history,stem,j==prefill-1?"prefill":NULL,j,"template-user",j>0);fputc(']',f);compact_boundary(f,s,cache,stem,"prefill");fputs(",\"generation_steps\":[",f);
    int answer[32],n=0,stopped=0;for(int j=0;j<budget;j++){int token=s->pending;answer[n++]=token;stopped=token==s->model->eos||(s->model->eos_extra>0&&token==s->model->eos_extra);compact_step(f,s,cache,token,history,stem,stopped||j==budget-1?"final":NULL,prefill+j,"own-generated",j>0);if(stopped)break;}
    fputs("],\"answer_ids\":",f);ints(f,answer,n);fprintf(f,",\"stopped\":%s,\"truncated\":%s,\"all_delivered_consumed\":true,\"final_position\":%d",stopped?"true":"false",stopped?"false":"true",s->position);compact_boundary(f,s,cache,stem,"final");fputc('}',f);REQUIRE(!fflush(f));fprintf(stderr,"completed %s: %d ingested,%d own consumed,stopped=%d\n",stem,prefill,n,stopped);
}
static void mechanics(FILE *f,QwenSession *s,OwnedCache *cache,const char *label,int a,int b,int sibling){
    char root[65],first[65],replay[65],root_after[65],yhash[65],file[128];logical_digest(s,cache,root);uint64_t parent,child;REQUIRE(!qwen_stack_push(s,&parent));REQUIRE(!qwen_push(s,a));REQUIRE(!qwen_stack_push(s,&child));REQUIRE(!qwen_push(s,b));logical_digest(s,cache,first);float *y=read_floats(s->logits,151936);digest(y,151936*4,yhash);snprintf(file,sizeof(file),"%s-rollback-reference.f32",label);array_save(file,y,151936*4);int pending=s->pending;
    REQUIRE(!qwen_stack_rollback(s,child)&&!qwen_push(s,sibling)&&!qwen_stack_rollback(s,parent));REQUIRE(!qwen_stack_push(s,&parent));REQUIRE(!qwen_push(s,a)&&!qwen_push(s,b));logical_digest(s,cache,replay);float *z=read_floats(s->logits,151936);REQUIRE(!memcmp(y,z,151936*4)&&pending==s->pending&&!strcmp(first,replay));free(y);free(z);REQUIRE(!qwen_stack_rollback(s,parent));logical_digest(s,cache,root_after);REQUIRE(!strcmp(root,root_after));
    int injected=0,partial[2]={0,0};
    if(cache&&cache->layers[0].byte){
        REQUIRE(!qwen_stack_push(s,&parent));cache->fail_layer=cache->layers[cache->count-1].layer;cache->fail_position=s->position;REQUIRE(!qwen_begin_token(s,a));int error=0;while(s->layer<28&&!error)error=qwen_advance_layer(s);REQUIRE(error==-77&&s->phase==2);for(int i=0;i<cache->count;i++)partial[i]=cache->layers[i].cursor;
        REQUIRE(!qwen_abort_token(s)&&s->phase==0&&s->position==93);logical_digest(s,cache,root_after);REQUIRE(!strcmp(root,root_after));for(int i=0;i<cache->count;i++)REQUIRE(cache->layers[i].cursor==93);REQUIRE(!qwen_stack_rollback(s,parent));REQUIRE(!qwen_stack_push(s,&parent)&&!qwen_push(s,a)&&!qwen_stack_rollback(s,parent));logical_digest(s,cache,root_after);REQUIRE(!strcmp(root,root_after));injected=1;
    }
    fprintf(f,"{\"root_logical_cache_sha256\":\"%s\",\"two_token_logical_cache_sha256\":\"%s\",\"replayed_logical_cache_sha256\":\"%s\",\"full_logits_sha256\":\"%s\",\"suffix_replaced\":true,\"token_and_encoded_root_restored\":true,\"injected_unpublished_append_failure\":%s,\"partial_cursors\":[%d,%d],\"abort_restored_root\":%s}",root,first,replay,yhash,injected?"true":"false",partial[0],partial[1],injected?"true":"false");
}
int main(int argc,char **argv){
    REQUIRE(argc==12);directory=argv[7];int n,ns,nc;int *prompt=tokens(argv[2],&n),*source_own=tokens(argv[8],&ns),*correction_ids=tokens(argv[3],&nc);TcmModel bundle;QwenModel m;QwenSession s;OwnedCache cache;char error[256],path[4096],hash[65],source_hash[65];History source={0};double started=clock_seconds();
    REQUIRE(!tcm_load(&bundle,argv[1],error,sizeof(error))&&!qwen_model_init(&m,&bundle,error,sizeof(error)));tcm_unload(&bundle);REQUIRE(m.hidden==1536&&m.qheads==12&&m.kvheads==2&&m.head==128&&m.layers==28);size_t model_live=gpu_live_bytes();REQUIRE(!qwen_session_init(&s,&m,512));
    for(int j=0;j<n;j++){REQUIRE(!qwen_push(&s,prompt[j]));source.ids[source.count++]=prompt[j];}for(int j=0;j<ns;j++){REQUIRE(s.pending==source_own[j]&&!qwen_push(&s,source_own[j]));source.ids[source.count++]=source_own[j];}REQUIRE(source.count==59&&source_own[ns-1]==151645);prefix_digest(&s,s.position,source_hash);REQUIRE(!strcmp(source_hash,argv[9]));
    REQUIRE(snprintf(path,sizeof(path),"%s/driver.json",directory)<(int)sizeof(path));FILE *f=fopen(path,"wb");REQUIRE(f);Capture capture={directory,NULL};fprintf(f,"{\"schema\":\"tcm-compact-append-driver-v1\",\"model_digest\":\"%s\",\"capacity\":512,\"layers\":[17,18],\"source_kv_sha256\":\"%s\",\"source_history_ids\":",m.digest,source_hash);ints(f,source.ids,source.count);fputs(",\"correction\":",f);Turn correction;History corrected=source;run_turn(f,&s,NULL,0,"correction",correction_ids,nc,&corrected,&capture,&correction,NULL);
    REQUIRE(s.position==93);prefix_digest(&s,s.position,hash);REQUIRE(!strcmp(hash,argv[10]));history_digest(&corrected,hash);REQUIRE(!strcmp(hash,argv[11]));QwenCheckpoint meta;REQUIRE(!qwen_checkpoint(&s,&meta));root_export(&s);
    char initial_hash[2][65];for(int i=0;i<2;i++){char file[64];snprintf(file,sizeof(file),"prefix-l%d.bin",17+i);size_t size;unsigned char *raw=file_read(file,&size);digest(raw,size,initial_hash[i]);free(raw);}
    fprintf(f,",\"prefix_programs_frozen_before_question_files_opened\":true,\"constructor_future_queries\":0,\"prefix_program_sha256\":[\"%s\",\"%s\"],\"model_live_gpu_bytes\":%zu,\"chains\":[",initial_hash[0],initial_hash[1],model_live);REQUIRE(!fflush(f));
    int *question[3],counts[3];for(int t=0;t<3;t++)question[t]=tokens(argv[4+t],&counts[t]);const int budgets[]={8,32,8};const char *names[]={"native","float","byte17","byte17_18"};
    for(int mode=0;mode<4;mode++){
        OwnedCache *c=mode?&cache:NULL;
        if(mode){REQUIRE(gpu_live_bytes()==model_live);cache_open(&cache,&s,mode);uint64_t mask=UINT64_C(1)<<17;if(mode!=2)mask|=UINT64_C(1)<<18;REQUIRE(!qwen_session_init_layout(&s,&m,512,mask,owned_step,owned_rewind,&cache));cache_bind(&cache);root_import(&s,&cache,&meta);for(int i=0;i<cache.count;i++)REQUIRE(s.cache_slots[cache.layers[i].layer]==-1);
            for(int i=0;i<cache.count;i++)if(cache.layers[i].byte){size_t bytes;unsigned char *raw=cache_serialize(cache.layers+i,93,&bytes);digest(raw,bytes,hash);REQUIRE(!strcmp(hash,initial_hash[cache.layers[i].layer-17]));free(raw);}
        }
        if(mode)fputc(',',f);size_t steady=gpu_live_bytes(),state_bytes=s.allocated_bytes+(c?c->arena_bytes+c->workspace_bytes+(c->invalid?4:0):0);REQUIRE(steady==model_live+state_bytes);History history=corrected;
        fprintf(f,"{\"policy\":\"%s\",\"mode\":%d,\"dense_layers\":%d,\"session_gpu_bytes\":%zu,\"cache_arena_bytes\":%zu,\"decode_workspace_bytes\":%zu,\"status_bytes\":%d,\"steady_state_gpu_bytes\":%zu,\"tracked_live_gpu_bytes\":%zu,\"native_session_already_destroyed\":%s,\"mechanics\":",names[mode],mode,s.dense_layers,s.allocated_bytes,c?c->arena_bytes:0,c?c->workspace_bytes:0,c&&c->invalid?4:0,state_bytes,steady,mode?"true":"false");
        int skip=corrected.ids[corrected.count-1]==question[0][0];REQUIRE(counts[0]-skip>=3);mechanics(f,&s,c,names[mode],question[0][skip],question[0][skip+1],question[0][skip+2]);fputs(",\"turns\":[",f);
        for(int t=0;t<3;t++){if(t)fputc(',',f);compact_turn(f,&s,c,names[mode],t,question[t],counts[t],budgets[t],&history);}fprintf(f,"],\"owned_step_calls_including_mechanics\":%lu,\"published_rows_including_mechanics\":%lu,\"cursor_rewinds\":%lu}",c?c->calls:0,c?c->published:0,c?c->rewinds:0);qwen_session_free(&s);if(c)cache_close(c);REQUIRE(gpu_live_bytes()==model_live);
    }
    fprintf(f,"],\"peak_tracked_gpu_bytes\":%zu,\"model_weight_bytes\":%zu,\"seconds\":%.9g}\n",gpu_peak_bytes(),m.weight_bytes,clock_seconds()-started);REQUIRE(!fclose(f));qwen_model_free(&m);REQUIRE(gpu_live_bytes()==0);free(prompt);free(source_own);free(correction_ids);for(int i=0;i<3;i++)free(question[i]);return 0;
}
