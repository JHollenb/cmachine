#include "turn_helpers.h"

extern void attention_trace_arm(float *host,int capacity);
extern int attention_trace_finish(float *scale);
typedef struct { Capture capture;QwenSession *session;unsigned ports; } AttentionCapture;

static void save_gpu(const Capture *c,const char *label,const float *pointer,size_t n){
    char file[256];REQUIRE(snprintf(file,sizeof(file),"%s-%s.f32",c->name,label)<(int)sizeof(file));float *host=read_floats(pointer,n);save(c->directory,file,host,n*4);free(host);
}
static void attention_observer(void *context,int position,int layer,QwenPort port,const float *data,size_t n){
    AttentionCapture *c=context;QwenSession *s=c->session;const QwenModel *m=s->model;
    REQUIRE(layer==17&&position==s->position&&!(c->ports&(1u<<port)));c->ports|=1u<<port;
    save_gpu(&c->capture,qwen_port_name(port),data,n);
    if(port==QWEN_ATTENTION){
        int length=position+1;save_gpu(&c->capture,"probabilities",s->scores,(size_t)m->qheads*length);
        size_t count=(size_t)m->kvheads*length*m->head;float *host=malloc(count*4);REQUIRE(host);
        for(int kind=0;kind<2;kind++){
            const float *cache=kind?s->values:s->keys;
            for(int h=0;h<m->kvheads;h++){
                size_t offset=((size_t)layer*m->kvheads+h)*s->capacity*m->head;
                gpu_read(host+(size_t)h*length*m->head,cache+offset,(size_t)length*m->head*4);
            }
            for(size_t j=0;j<count;j++)REQUIRE(isfinite(host[j]));char file[256];REQUIRE(snprintf(file,sizeof(file),"%s-%s.f32",c->capture.name,kind?"values":"keys")<(int)sizeof(file));save(c->capture.directory,file,host,count*4);
        }free(host);
    }
}
static void advance_observed(FILE *f,QwenSession *s,QwenSubspace *p,int policy,int index,int token,History *history,const Capture *capture,int observing,const char *role,int comma){
    int position=s->position,selected=observing&&(position==116||position==118);char stem[192],history_hash[65];
    REQUIRE(snprintf(stem,sizeof(stem),"%s-p%d",capture->name,position)<(int)sizeof(stem));
    AttentionCapture c={{capture->directory,stem},s,0};float scores[12*512],scale=0;int hits=0;
    s->observer=selected?attention_observer:NULL;s->observer_context=selected?&c:NULL;s->observe_position=position;s->observe_layer=17;
    REQUIRE(!qwen_begin_token(s,token));QwenSubspaceStats stats={0};
    while(s->layer<s->model->layers){
        int attention_selected=selected&&s->layer==17;
        if(attention_selected)attention_trace_arm(scores,12*512);
        if(s->layer==16&&policy==2)REQUIRE(!qwen_advance_layer_subspace(s,p,QWEN_SUBSPACE_PROJECT,&stats));else REQUIRE(!qwen_advance_layer(s));
        if(attention_selected)hits=attention_trace_finish(&scale);
    }
    REQUIRE(!qwen_finish_token(s));s->observer=NULL;s->observer_context=NULL;
    if(selected){
        REQUIRE(hits==1&&c.ports==((1u<<QWEN_PORT_COUNT)-1)&&isfinite(scale));size_t n=(size_t)s->model->qheads*(position+1);for(size_t j=0;j<n;j++)REQUIRE(isfinite(scores[j]));
        char file[256];REQUIRE(snprintf(file,sizeof(file),"%s-unscaled-scores.f32",stem)<(int)sizeof(file));save(capture->directory,file,scores,n*4);
        REQUIRE(snprintf(file,sizeof(file),"%s-scale.f32",stem)<(int)sizeof(file));save(capture->directory,file,&scale,4);
    }
    REQUIRE(history->count<512);history->ids[history->count++]=token;history_digest(history,history_hash);
    if(policy==2)REQUIRE(stats.work.gate_up_matvecs==0&&stats.work.down_matvecs==1&&!stats.work.compile_matvecs);
    if(comma)fputc(',',f);fprintf(f,"{\"step\":%d,\"position\":%d,\"input_id\":%d,\"role\":\"%s\",\"history_sha256\":\"%s\",\"top\":%d,\"substitution\":%s,\"gate_up_calls\":%u,\"down_calls\":1,\"compile_calls\":0,\"attention_trace\":%s}",index,position,token,role,history_hash,s->pending,policy==2?"true":"false",policy==2?0u:2u,selected?"true":"false");REQUIRE(!fflush(f));
}
static void episode(FILE *f,QwenSession *s,QwenSubspace *p,int policy,const char *name,const int *template_ids,int template_count,History *history,Capture *capture,Turn *result,int observing){
    Capture c={capture->directory,name};char hash[65];prefix_digest(s,s->position,hash);int parent=s->position,skip=history->ids[history->count-1]==template_ids[0];
    REQUIRE(template_count>1&&template_ids[0]==151645);memset(result,0,sizeof(*result));
    fprintf(f,"{\"name\":\"%s\",\"policy\":%d,\"autonomous\":true,\"parent_position\":%d,\"parent_kv_sha256\":\"%s\",\"parent_history_ids\":",name,policy,parent,hash);ints(f,history->ids,history->count);
    fprintf(f,",\"template_ids\":");ints(f,template_ids,template_count);fprintf(f,",\"template_skip\":%d,\"host_supplied_closing_token\":%s,\"budget\":8,\"steps\":[",skip,skip?"false":"true");
    for(int j=skip;j<template_count;j++){
        int index=j-skip;result->inputs[result->count++]=template_ids[j];advance_observed(f,s,p,policy,index,template_ids[j],history,&c,observing,!skip&&!j?"host-template-close":"template-user",index>0);
    }
    result->prefill=result->count;fputc(']',f);boundary(f,s,&c,"prefill");fprintf(f,",\"generation_steps\":[");
    for(int j=0;j<8;j++){
        int token=s->pending;result->answer[result->answers++]=token;result->stopped=token==s->model->eos||(s->model->eos_extra>0&&token==s->model->eos_extra);
        REQUIRE(result->count<512);result->inputs[result->count++]=token;advance_observed(f,s,p,policy,result->count-1,token,history,&c,observing,"own-generated",j>0);if(result->stopped)break;
    }
    fprintf(f,"],\"prefill_count\":%d,\"answer_ids\":",result->prefill);ints(f,result->answer,result->answers);fprintf(f,",\"stopped\":%s,\"truncated\":%s,\"all_delivered_consumed\":true,\"final_position\":%d",result->stopped?"true":"false",result->stopped?"false":"true",s->position);boundary(f,s,&c,"final");fputc('}',f);REQUIRE(!fflush(f));
    fprintf(stderr,"completed %s: %d ingested,%d own tokens,stop=%d\n",name,result->prefill,result->answers,result->stopped);
}
int main(int argc,char **argv){
    REQUIRE(argc==15);int n,ns,nt[3];int *prompt=tokens(argv[2],&n),*own=tokens(argv[8],&ns),*templates[]={tokens(argv[3],&nt[0]),tokens(argv[4],&nt[1]),tokens(argv[5],&nt[2])};
    TcmModel bundle;QwenModel m;QwenSession s;QwenSubspace p;char error[256],path[4096],source_hash[65],after[65],bank_before[65],bank_after[65];double started=clock_seconds();History source={0};
    REQUIRE(!tcm_load(&bundle,argv[1],error,sizeof(error))&&!qwen_model_init(&m,&bundle,error,sizeof(error)));tcm_unload(&bundle);REQUIRE(m.hidden==1536&&m.qheads==12&&m.kvheads==2&&m.head==128&&m.layers==28);
    REQUIRE(!qwen_session_init(&s,&m,512)&&!qwen_subspace_init(&p,&m));import_bank(&p,argv[7]);bank_fingerprint(&p,bank_before);REQUIRE(!strcmp(bank_before,argv[10]));
    REQUIRE(n+ns<=512&&ns>0);for(int j=0;j<n;j++){REQUIRE(!qwen_push(&s,prompt[j]));source.ids[source.count++]=prompt[j];}
    for(int j=0;j<ns;j++){REQUIRE(s.pending==own[j]&&!qwen_push(&s,own[j]));source.ids[source.count++]=own[j];}
    REQUIRE(own[ns-1]==151645&&source.count==s.position);prefix_digest(&s,s.position,source_hash);REQUIRE(!strcmp(source_hash,argv[9]));
    REQUIRE(snprintf(path,sizeof(path),"%s/driver.json",argv[6])<(int)sizeof(path));FILE *f=fopen(path,"wb");REQUIRE(f);Capture capture={argv[6],NULL};
    Capture op={argv[6],"operator"};save_gpu(&op,"o17",m.layer[17].o,(size_t)m.hidden*m.hidden);
    fprintf(f,"{\"schema\":\"tcm-attention-read-driver-v1\",\"model_digest\":\"%s\",\"layer\":17,\"replacement_layer\":16,\"capacity\":512,\"source_replay_checked_before_candidates\":true,\"source_position\":%d,\"source_kv_sha256\":\"%s\",\"source_history_ids\":",m.digest,s.position,source_hash);ints(f,source.ids,source.count);fprintf(f,",\"qheads\":12,\"kvheads\":2,\"head_dimension\":128,\"selected_positions\":[116,118],\"formations\":[");
    for(int formation=0;formation<2;formation++){
        if(formation)fputc(',',f);uint64_t root;REQUIRE(!qwen_stack_push(&s,&root));History h=source;Turn result;char name[128];
        fprintf(f,"{\"policy\":%d,\"correction\":",formation?2:0);snprintf(name,sizeof(name),"%c-correction",formation?'P':'N');episode(f,&s,&p,formation?2:0,name,templates[0],nt[0],&h,&capture,&result,0);History corrected=h;
        prefix_digest(&s,s.position,after);char history_hash[65];history_digest(&corrected,history_hash);REQUIRE(!strcmp(after,argv[formation?12:11])&&!strcmp(history_hash,argv[formation?14:13]));
        fprintf(f,",\"formation_checked_before_readers\":true,\"readers\":[");
        for(int reader=0;reader<2;reader++)for(int task=1;task<3;task++){
            if(reader||task>1)fputc(',',f);uint64_t child;h=corrected;REQUIRE(!qwen_stack_push(&s,&child));snprintf(name,sizeof(name),"%c%c-%s",formation?'P':'N',reader?'P':'N',tasks[task]);episode(f,&s,&p,reader?2:0,name,templates[task],nt[task],&h,&capture,&result,1);REQUIRE(!qwen_stack_rollback(&s,child));prefix_digest(&s,s.position,after);REQUIRE(!strcmp(after,argv[formation?12:11]));
        }
        REQUIRE(!qwen_stack_rollback(&s,root));prefix_digest(&s,s.position,after);REQUIRE(!strcmp(after,source_hash));fprintf(f,"],\"source_prefix_intact\":true}");
    }
    bank_fingerprint(&p,bank_after);REQUIRE(!strcmp(bank_before,bank_after));fprintf(f,"],\"frozen_bank_before_sha256\":\"%s\",\"frozen_bank_after_sha256\":\"%s\",\"seconds\":%.9g}\n",bank_before,bank_after,clock_seconds()-started);REQUIRE(!fclose(f));
    qwen_subspace_free(&p);qwen_session_free(&s);qwen_model_free(&m);free(prompt);free(own);for(int j=0;j<3;j++)free(templates[j]);return 0;
}
