/* The staged helper is the authenticated prior driver prefix, excluding main. */
#include "span_helpers.h"
#include "byte_bank.h"
extern void span_byte_arm(const signed char *,const signed char *,const float *,float *);
extern void span_poison_guard(const float *);

typedef struct {
    QwenSession *session;SpanProgram *program;ByteProgram *byte;Capture point;
    const char *query_name;int mode,sparse;unsigned ports;
} LoopCapture;
static void byte_gpu_digest(const ByteProgram *p,char out[65]){
    unsigned char host[17952];gpu_read(host,p->gpu_scales,544);gpu_read(host+544,p->gpu_keys,8704);gpu_read(host+9248,p->gpu_values,8704);digest(host,sizeof(host),out);
}
static void compile_byte(FILE *f,ByteProgram *p,const float *k,const float *v,const char *directory){
    REQUIRE(!byte_construct(p,k,v));p->gpu_keys=gpu_alloc(8704);p->gpu_values=gpu_alloc(8704);p->gpu_scales=gpu_alloc(544);
    gpu_write(p->gpu_keys,p->keys,8704);gpu_write(p->gpu_values,p->values,8704);gpu_write(p->gpu_scales,p->scales,544);byte_gpu_digest(p,p->gpu_before);
    unsigned char raw[17984];memcpy(raw,"TCMBYTE1",8);uint32_t header[]={1,2,128,34,8,0};memcpy(raw+8,header,24);memcpy(raw+32,p->scales,544);memcpy(raw+576,p->keys,8704);memcpy(raw+9280,p->values,8704);
    save(directory,"bank-byte.bin",raw,sizeof(raw));digest(raw,sizeof(raw),p->file_sha);
    fprintf(f,"{\"name\":\"byte\",\"file\":\"bank-byte.bin\",\"bytes\":17984,\"gpu_bytes\":17952,\"file_sha256\":\"%s\",\"gpu_before_sha256\":\"%s\",\"rounding\":\"FP32 division then nearest, ties away from zero; clip [-127,127]\"}",p->file_sha,p->gpu_before);
}
static void loop_observer(void *context,int position,int layer,QwenPort port,const float *data,size_t n){
    LoopCapture *c=context;REQUIRE(layer==17&&position==c->session->position&&!(c->ports&(1u<<port)));c->ports|=1u<<port;
    if(port==QWEN_QUERY){
        float *q=read_floats(data,n);save(c->point.directory,c->query_name,q,n*4);
        if(c->sparse&&(c->program||c->byte)){
            double response[12*129];for(int h=0;h<12;h++){
                int result=c->byte?byte_predict(c->byte,q+h*128,h/6,1.0f/sqrtf(128.0f),response+h*129,response+h*129+1):span_predict(c->program,q+h*128,h/6,1.0f/sqrtf(128.0f),response+h*129,response+h*129+1);REQUIRE(!result);
            }for(size_t i=0;i<12*129;i++)REQUIRE(isfinite(response[i]));save_array(&c->point,"response.f64",response,sizeof(response));
        }free(q);
    }
    if(c->sparse){char label[96];snprintf(label,sizeof(label),"%s.f32",qwen_port_name(port));gpu_array(&c->point,label,data,n);}
}
/* Explicitly named complement digest; never reported as an unmodified raw hash. */
static void complement_digest(const QwenSession *s,char out[65]){
    size_t rows=(size_t)s->position*128,count=(size_t)2*28*2*rows,offset=0;float *host=malloc(count*4);REQUIRE(host);
    for(int kind=0;kind<2;kind++)for(int layer=0;layer<28;layer++)for(int h=0;h<2;h++){
        gpu_read(host+offset,(kind?s->values:s->keys)+((size_t)layer*2+h)*512*128,rows*4);
        if(layer==17)memset(host+offset+59*128,0,34*128*4);offset+=rows;
    }for(size_t i=0;i<count;i++)REQUIRE(isfinite(host[i]));digest(host,count*4,out);free(host);
}
static void poison_digest(const QwenSession *s,char out[65]){
    uint32_t host[2*2*34*128];size_t offset=0;
    for(int kind=0;kind<2;kind++)for(int h=0;h<2;h++){
        gpu_read(host+offset,(kind?s->values:s->keys)+((size_t)17*2+h)*512*128+59*128,34*128*4);offset+=34*128;
    }for(size_t i=0;i<offset;i++)REQUIRE(host[i]==UINT32_C(0x7fc00000));digest(host,sizeof(host),out);
}
static void loop_boundary(FILE *f,const QwenSession *s,const Capture *c,const char *label){
    boundary(f,s,c,label);char hash[65];complement_digest(s,hash);fprintf(f,",\"%s_complement_cache_sha256\":\"%s\"",label,hash);
}
static void loop_step(FILE *f,QwenSession *s,SpanProgram *program,ByteProgram *byte,int mode,int token,History *history,const char *stem,const char *spot,int index,const char *role,int comma){
    int position=s->position;char query[160],point[160],hh[65];REQUIRE(snprintf(query,sizeof(query),"%s-s%03d-query.f32",stem,index)<(int)sizeof(query));REQUIRE(snprintf(point,sizeof(point),"%s-%s",stem,spot?spot:"uncaptured")<(int)sizeof(point));
    LoopCapture c={s,program,byte,{NULL,point},query,mode,spot!=NULL,0};extern const char *loop_directory;c.point.directory=loop_directory;
    s->observer=loop_observer;s->observer_context=&c;s->observe_position=position;s->observe_layer=17;REQUIRE(!qwen_begin_token(s,token));int applications=0,packed=0;
    while(s->layer<s->model->layers){int selected=s->layer==17;if(selected){if(byte)span_byte_arm(byte->gpu_keys,byte->gpu_values,byte->gpu_scales,NULL);else span_read_arm(mode,program?program->gpu_keys:NULL,program?program->gpu_values:NULL,program?program->gpu_logweights:NULL,NULL);}REQUIRE(!qwen_advance_layer(s));if(selected){REQUIRE(span_read_finish(&packed)==1);applications++;}}
    REQUIRE(!qwen_finish_token(s)&&applications==1&&c.ports==((1u<<QWEN_PORT_COUNT)-1));s->observer=NULL;s->observer_context=NULL;
    if(spot)current_kv(s,position,&c.point);REQUIRE(history->count<512);history->ids[history->count++]=token;history_digest(history,hh);
    if(comma)fputc(',',f);fprintf(f,"{\"step\":%d,\"position\":%d,\"input_id\":%d,\"role\":\"%s\",\"history_sha256\":\"%s\",\"top\":%d,\"query_file\":\"%s\",\"packed_length\":%d,\"selected_read_applications\":1,\"original_span_rows_read_by_selected_attention\":%d,\"span_dot_and_value_macs\":%d,\"capture\":",index,position,token,role,hh,s->pending,query,packed,mode<0?34:0,2*12*(mode<0?34:mode)*128);
    if(spot)fprintf(f,"\"%s\"",point);else fputs("null",f);fputc('}',f);REQUIRE(!fflush(f));
}
const char *loop_directory;
static void loop_turn(FILE *f,QwenSession *s,SpanProgram *program,ByteProgram *byte,int mode,const char *label,int turn,const int *template_ids,int count,int budget,History *history){
    int skip=history->ids[history->count-1]==template_ids[0],start=s->position,prefill=count-skip;char stem[96];snprintf(stem,sizeof(stem),"%s-t%d",label,turn);Capture capture={loop_directory,stem};REQUIRE(count>1&&template_ids[0]==151645&&budget<=32);
    fprintf(f,"{\"name\":\"%s\",\"turn\":%d,\"budget\":%d,\"parent_position\":%d,\"parent_history_ids\":",stem,turn,budget,start);ints(f,history->ids,history->count);fprintf(f,",\"template_ids\":");ints(f,template_ids,count);fprintf(f,",\"template_skip\":%d,\"steps\":[",skip);
    for(int j=0;j<prefill;j++)loop_step(f,s,program,byte,mode,template_ids[j+skip],history,stem,j==prefill-1?"prefill":NULL,j,"template-user",j>0);
    fputc(']',f);loop_boundary(f,s,&capture,"prefill");fprintf(f,",\"generation_steps\":[");int answers[32],n=0,stopped=0;
    for(int j=0;j<budget;j++){int token=s->pending;answers[n++]=token;stopped=token==s->model->eos||(s->model->eos_extra>0&&token==s->model->eos_extra);loop_step(f,s,program,byte,mode,token,history,stem,stopped||j==budget-1?"final":NULL,prefill+j,"own-generated",j>0);if(stopped)break;}
    fprintf(f,"],\"answer_ids\":");ints(f,answers,n);fprintf(f,",\"stopped\":%s,\"truncated\":%s,\"all_delivered_consumed\":true,\"final_position\":%d",stopped?"true":"false",stopped?"false":"true",s->position);loop_boundary(f,s,&capture,"final");fputc('}',f);REQUIRE(!fflush(f));fprintf(stderr,"completed %s: %d ingested,%d own consumed,stopped=%d\n",stem,prefill,n,stopped);
}
int main(int argc,char **argv){
    REQUIRE(argc==12);int n,ns,nc;int *prompt=tokens(argv[2],&n),*source_own=tokens(argv[8],&ns),*correction_ids=tokens(argv[3],&nc);loop_directory=argv[7];
    TcmModel bundle;QwenModel m;QwenSession s;char error[256],path[4096],hash[65],source_hash[65];History source={0};double started=clock_seconds();
    REQUIRE(!tcm_load(&bundle,argv[1],error,sizeof(error))&&!qwen_model_init(&m,&bundle,error,sizeof(error)));tcm_unload(&bundle);REQUIRE(m.hidden==1536&&m.qheads==12&&m.kvheads==2&&m.head==128&&m.layers==28&&!qwen_session_init(&s,&m,512));
    REQUIRE(n+ns<512&&ns>0);for(int j=0;j<n;j++){REQUIRE(!qwen_push(&s,prompt[j]));source.ids[source.count++]=prompt[j];}for(int j=0;j<ns;j++){REQUIRE(s.pending==source_own[j]&&!qwen_push(&s,source_own[j]));source.ids[source.count++]=source_own[j];}
    REQUIRE(source.count==59&&source_own[ns-1]==151645);prefix_digest(&s,s.position,source_hash);REQUIRE(!strcmp(source_hash,argv[9]));
    REQUIRE(snprintf(path,sizeof(path),"%s/driver.json",loop_directory)<(int)sizeof(path));FILE *f=fopen(path,"wb");REQUIRE(f);Capture capture={loop_directory,NULL};
    fprintf(f,"{\"schema\":\"tcm-span-recurrent-driver-v1\",\"model_digest\":\"%s\",\"layer\":17,\"capacity\":512,\"source_kv_sha256\":\"%s\",\"source_history_ids\":",m.digest,source_hash);ints(f,source.ids,source.count);fprintf(f,",\"correction\":");Turn correction;History corrected=source;run_turn(f,&s,NULL,0,"correction",correction_ids,nc,&corrected,&capture,&correction,NULL);
    REQUIRE(s.position==93);prefix_digest(&s,s.position,hash);REQUIRE(!strcmp(hash,argv[10]));history_digest(&corrected,hash);REQUIRE(!strcmp(hash,argv[11]));
    double compile_started=clock_seconds();float keys[2*34*128],values[2*34*128];for(int h=0;h<2;h++){size_t index=((size_t)17*2+h)*512*128+59*128;gpu_read(keys+(size_t)h*34*128,s.keys+index,34*128*4);gpu_read(values+(size_t)h*34*128,s.values+index,34*128*4);}
    save(loop_directory,"audit-span-keys.f32",keys,sizeof(keys));save(loop_directory,"audit-span-values.f32",values,sizeof(values));SpanProgram banks[2];ByteProgram byte;fprintf(f,",\"programs\":[");compile_program(f,&banks[0],8,keys,values,loop_directory,"eight");fputc(',',f);compile_program(f,&banks[1],34,keys,values,loop_directory,"exact");fputc(',',f);compile_byte(f,&byte,keys,values,loop_directory);
    fprintf(f,"],\"programs_frozen_before_question_files_opened\":true,\"constructor_future_queries\":0,\"constructor_position\":93,\"compile_seconds\":%.9g,\"workspace_bytes\":1048576,\"chains\":[",clock_seconds()-compile_started);REQUIRE(!fflush(f));memset(keys,0,sizeof(keys));memset(values,0,sizeof(values));span_workspace_open();
    int *question[3],counts[3];for(int t=0;t<3;t++)question[t]=tokens(argv[4+t],&counts[t]);const int budgets[]={8,32,8};const char *names[]={"native","exact","eight","byte","delete"};const int modes[]={-1,34,8,34,0};char poisoned_root[65]={0},poison_hash[65]={0};uint64_t native_epoch=s.epoch;
    for(int arm=0;arm<5;arm++){
        if(arm==1){
            /* No live unpoisoned frame/checkpoint may promise an immutable ancestor. */
            REQUIRE(s.depth==0&&(s.phase==0||s.phase==1)&&s.position==93&&s.checkpoint_position==0&&s.epoch<UINT64_MAX);uint32_t poison[34*128];for(size_t i=0;i<34*128;i++)poison[i]=UINT32_C(0x7fc00000);
            for(int kind=0;kind<2;kind++)for(int h=0;h<2;h++)gpu_write((kind?s.values:s.keys)+((size_t)17*2+h)*512*128+59*128,poison,sizeof(poison));
            s.epoch++;s.checkpoint_serial++;s.checkpoint_position=0;memset(s.frames,0,sizeof(s.frames));span_poison_guard(s.keys+(size_t)17*2*512*128);poison_digest(&s,poison_hash);prefix_digest(&s,s.position,poisoned_root);
        }
        if(arm)fputc(',',f);uint64_t branch;REQUIRE(s.depth==0&&!qwen_stack_push(&s,&branch));char parent[65],complement[65],after[65];prefix_digest(&s,s.position,parent);complement_digest(&s,complement);History history=corrected;
        fprintf(f,"{\"policy\":\"%s\",\"mode\":%d,\"poisoned\":%s,\"root_epoch\":%llu,\"parent_raw_cache_sha256\":\"%s\",\"parent_complement_cache_sha256\":\"%s\",\"turns\":[",names[arm],modes[arm],arm?"true":"false",(unsigned long long)s.epoch,parent,complement);
        SpanProgram *p=arm==1?&banks[1]:arm==2?&banks[0]:NULL;for(int t=0;t<3;t++){if(t)fputc(',',f);loop_turn(f,&s,p,arm==3?&byte:NULL,modes[arm],names[arm],t,question[t],counts[t],budgets[t],&history);}
        if(arm){poison_digest(&s,after);REQUIRE(!strcmp(after,poison_hash));}REQUIRE(!qwen_stack_rollback(&s,branch));prefix_digest(&s,s.position,after);REQUIRE(!strcmp(after,parent));fprintf(f,"],\"parent_intact\":true,\"poison_intact\":%s}",arm?"true":"false");
    }
    fprintf(f,"],\"poisoned_root\":{\"position\":93,\"epoch\":%llu,\"native_epoch\":%llu,\"raw_cache_sha256\":\"%s\",\"span_bytes_sha256\":\"%s\",\"pattern_u32\":2143289344,\"bytes\":69632,\"old_frames_discarded\":true,\"dense_L17_fallback_refused\":true,\"physical_cache_allocation_released\":false},\"programs_after\":[",(unsigned long long)s.epoch,(unsigned long long)native_epoch,poisoned_root,poison_hash);
    for(int j=0;j<2;j++){if(j)fputc(',',f);program_gpu_digest(&banks[j],hash);REQUIRE(!strcmp(hash,banks[j].gpu_before));fprintf(f,"{\"name\":\"%s\",\"gpu_after_sha256\":\"%s\"}",j?"exact":"eight",hash);gpu_free(banks[j].gpu_keys);gpu_free(banks[j].gpu_values);gpu_free(banks[j].gpu_logweights);span_host_free(&banks[j]);}
    byte_gpu_digest(&byte,hash);REQUIRE(!strcmp(hash,byte.gpu_before));fprintf(f,",{\"name\":\"byte\",\"gpu_after_sha256\":\"%s\"}],\"seconds\":%.9g}\n",hash,clock_seconds()-started);REQUIRE(!fclose(f));gpu_free(byte.gpu_keys);gpu_free(byte.gpu_values);gpu_free(byte.gpu_scales);span_workspace_close();qwen_session_free(&s);qwen_model_free(&m);free(prompt);free(source_own);free(correction_ids);for(int t=0;t<3;t++)free(question[t]);return 0;
}
