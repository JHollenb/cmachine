#include "turn_helpers.h"
#include "span_bank.h"

extern void span_workspace_open(void);
extern void span_workspace_close(void);
extern void span_read_arm(int m,const float *k,const float *v,const float *logw,float *scores);
extern int span_read_finish(int *length);
extern int span_read_length(void);

typedef struct {Capture capture;SpanProgram *program;QwenSession *session;unsigned ports;int prediction_written,mode;char prediction_sha[65];} ReadCapture;
static void save_array(const Capture *c,const char *label,const void *host,size_t bytes){char file[256];REQUIRE(snprintf(file,sizeof(file),"%s-%s",c->name,label)<(int)sizeof(file));save(c->directory,file,(void*)host,bytes);}
static void gpu_array(const Capture *c,const char *label,const float *pointer,size_t n){float *host=read_floats(pointer,n);save_array(c,label,host,n*4);free(host);}
static void read_observer(void *context,int position,int layer,QwenPort port,const float *data,size_t n){
    ReadCapture *c=context;REQUIRE(layer==17&&position==c->session->position&&!(c->ports&(1u<<port)));c->ports|=1u<<port;char label[96];snprintf(label,sizeof(label),"%s.f32",qwen_port_name(port));gpu_array(&c->capture,label,data,n);
    if(port==QWEN_QUERY&&c->program){
        float *query=read_floats(data,n);double response[12*129];
        for(int h=0;h<12;h++)REQUIRE(!span_predict(c->program,query+h*128,h/6,1.0f/sqrtf(128.0f),response+h*129,response+h*129+1));
        for(size_t i=0;i<12*129;i++)REQUIRE(isfinite(response[i]));save_array(&c->capture,"response.f64",response,sizeof(response));digest(response,sizeof(response),c->prediction_sha);c->prediction_written=1;free(query);
    }
    if(port==QWEN_ATTENTION){
        int length=span_read_length();REQUIRE(length>0&&length<=512);gpu_array(&c->capture,"probabilities.f32",c->session->scores,(size_t)12*length);
        if(c->mode<0){
            int ncomp=length-34;float *host=malloc((size_t)2*ncomp*128*4);REQUIRE(host);
            for(int kind=0;kind<2;kind++){
                const float *cache=kind?c->session->values:c->session->keys;
                for(int h=0;h<2;h++){const float *source=cache+((size_t)17*2+h)*512*128;float *target=host+(size_t)h*ncomp*128;gpu_read(target,source,(size_t)59*128*4);gpu_read(target+(size_t)59*128,source+(size_t)93*128,(size_t)(length-93)*128*4);}
                save_array(&c->capture,kind?"complement-values.f32":"complement-keys.f32",host,(size_t)2*ncomp*128*4);
            }free(host);
        }
    }
}
static void program_gpu_digest(const SpanProgram *p,char hash[65]){
    size_t n=(size_t)2*p->clusters,bytes=(2*n*128+n)*4;float *host=malloc(bytes);REQUIRE(host);gpu_read(host,p->gpu_keys,n*128*4);gpu_read(host+n*128,p->gpu_values,n*128*4);gpu_read(host+2*n*128,p->gpu_logweights,n*4);digest(host,bytes,hash);free(host);
}
static void compile_program(FILE *f,SpanProgram *p,int clusters,const float *keys,const float *values,const char *directory,const char *label){
    REQUIRE(!span_construct(p,clusters,keys,values));size_t n=(size_t)2*clusters;p->gpu_keys=gpu_alloc(n*128*4);p->gpu_values=gpu_alloc(n*128*4);p->gpu_logweights=gpu_alloc(n*4);
    gpu_write(p->gpu_keys,p->keys,n*128*4);gpu_write(p->gpu_values,p->values,n*128*4);gpu_write(p->gpu_logweights,p->logweights,n*4);program_gpu_digest(p,p->gpu_before);
    size_t bytes=32+(2*n*128+n)*4;unsigned char *raw=malloc(bytes);REQUIRE(raw);memcpy(raw,"TCMSPAN1",8);uint32_t header[]={1,2,128,34,(uint32_t)clusters,0};memcpy(raw+8,header,sizeof(header));size_t offset=32;
    memcpy(raw+offset,p->counts,n*4);offset+=n*4;memcpy(raw+offset,p->keys,n*128*4);offset+=n*128*4;memcpy(raw+offset,p->values,n*128*4);offset+=n*128*4;REQUIRE(offset==bytes);
    char file[128];snprintf(file,sizeof(file),"bank-%s.bin",label);save(directory,file,raw,bytes);digest(raw,bytes,p->file_sha);free(raw);snprintf(file,sizeof(file),"bank-%s-logweights.f32",label);save(directory,file,p->logweights,n*4);
    fprintf(f,"{\"name\":\"%s\",\"clusters\":%d,\"file\":\"bank-%s.bin\",\"bytes\":%zu,\"float_payload_bytes\":%zu,\"gpu_bytes\":%zu,\"file_sha256\":\"%s\",\"gpu_before_sha256\":\"%s\",\"membership\":[",label,clusters,label,bytes,bytes-32,bytes-32,p->file_sha,p->gpu_before);
    for(size_t i=0;i<n;i++){if(i)fputc(',',f);fprintf(f,"%llu",(unsigned long long)p->members[i]);}fprintf(f,"]}");free(p->members);p->members=NULL;
}
static void current_kv(const QwenSession *s,int position,const Capture *c){
    float data[1536];size_t offset=0;const int layers[]={17,18,27};
    for(int kind=0;kind<2;kind++)for(int j=0;j<3;j++)for(int h=0;h<2;h++){size_t index=(((size_t)layers[j]*2+h)*s->capacity+position)*128;gpu_read(data+offset,(kind?s->values:s->keys)+index,128*4);offset+=128;}
    REQUIRE(offset==1536);save_array(c,"kv17-18-27.f32",data,sizeof(data));
}
static void own_step(FILE *f,QwenSession *s,int token,History *history,int index){
    int position=s->position;REQUIRE(history->count<512&&!qwen_push(s,token));history->ids[history->count++]=token;char hash[65];history_digest(history,hash);
    if(index)fputc(',',f);fprintf(f,"{\"step\":%d,\"position\":%d,\"input_id\":%d,\"top\":%d,\"history_sha256\":\"%s\"}",index,position,token,s->pending,hash);
}
static void reader(FILE *f,QwenSession *s,SpanProgram *program,int mode,const char *name,int final_id,int budget,History history,Capture *capture){
    uint64_t branch;REQUIRE(budget>0&&budget<=32&&!qwen_stack_push(s,&branch));char parent[65],after[65];prefix_digest(s,s->position,parent);int position=s->position;ReadCapture c={{capture->directory,name},program,s,0,0,mode,{0}};
    fprintf(f,"{\"name\":\"%s\",\"mode\":%d,\"budget\":%d,\"parent_position\":%d,\"parent_kv_sha256\":\"%s\",\"parent_history_ids\":",name,mode,budget,position,parent);ints(f,history.ids,history.count);
    s->observer=read_observer;s->observer_context=&c;s->observe_position=position;s->observe_layer=17;float scores[12*512];int length=0;
    REQUIRE(!qwen_begin_token(s,final_id));while(s->layer<s->model->layers){int selected=s->layer==17;if(selected)span_read_arm(mode,program?program->gpu_keys:NULL,program?program->gpu_values:NULL,program?program->gpu_logweights:NULL,scores);REQUIRE(!qwen_advance_layer(s));if(selected)REQUIRE(span_read_finish(&length)==1);}
    REQUIRE(!qwen_finish_token(s)&&c.ports==((1u<<QWEN_PORT_COUNT)-1));s->observer=NULL;s->observer_context=NULL;REQUIRE(history.count<512);history.ids[history.count++]=final_id;current_kv(s,position,&c.capture);
    if(mode>=0)save_array(&c.capture,"unscaled-scores.f32",scores,(size_t)12*length*4);
    REQUIRE(c.prediction_written==(program!=NULL));char history_hash[65];history_digest(&history,history_hash);fprintf(f,",\"input_id\":%d,\"history_sha256\":\"%s\",\"packed_length\":%d,\"original_span_rows_read_by_selected_attention\":%d,\"complement_copy_bytes\":%zu,\"virtual_copy_bytes\":%zu,\"span_dot_and_value_macs\":%d,\"prediction_sha256\":",final_id,history_hash,length,mode<0?34:0,mode<0?0:((size_t)position+1-34)*2048,mode<=0?0:(size_t)mode*2048,2*12*(mode<0?34:mode)*128);
    if(program)fprintf(f,"\"%s\"",c.prediction_sha);else fprintf(f,"null");boundary(f,s,&c.capture,"prefill");fprintf(f,",\"generation_steps\":[");
    int answers[32],count=0,stopped=0;for(int j=0;j<budget;j++){int token=s->pending;answers[count++]=token;stopped=token==s->model->eos||(s->model->eos_extra>0&&token==s->model->eos_extra);own_step(f,s,token,&history,j);if(stopped)break;}
    fprintf(f,"],\"answer_ids\":");ints(f,answers,count);fprintf(f,",\"stopped\":%s,\"truncated\":%s,\"all_delivered_consumed\":true,\"selected_read_applications\":1,\"later_attention_policy\":\"native\",\"final_position\":%d",stopped?"true":"false",stopped?"false":"true",s->position);boundary(f,s,&c.capture,"final");
    REQUIRE(!qwen_stack_rollback(s,branch));prefix_digest(s,s->position,after);REQUIRE(!strcmp(parent,after));fprintf(f,",\"parent_intact\":true}");REQUIRE(!fflush(f));fprintf(stderr,"completed %s: %d own tokens,stopped=%d\n",name,count,stopped);
}
int main(int argc,char **argv){
    REQUIRE(argc==11);int n,ns,nc;int *prompt=tokens(argv[2],&n),*source_own=tokens(argv[7],&ns),*correction_ids=tokens(argv[3],&nc);
    TcmModel bundle;QwenModel m;QwenSession s;char error[256],path[4096],hash[65],source_hash[65];History source={0};double started=clock_seconds();
    REQUIRE(!tcm_load(&bundle,argv[1],error,sizeof(error))&&!qwen_model_init(&m,&bundle,error,sizeof(error)));tcm_unload(&bundle);REQUIRE(m.hidden==1536&&m.qheads==12&&m.kvheads==2&&m.head==128&&m.layers==28&&!qwen_session_init(&s,&m,512));
    REQUIRE(n+ns<512&&ns>0);for(int j=0;j<n;j++){REQUIRE(!qwen_push(&s,prompt[j]));source.ids[source.count++]=prompt[j];}for(int j=0;j<ns;j++){REQUIRE(s.pending==source_own[j]&&!qwen_push(&s,source_own[j]));source.ids[source.count++]=source_own[j];}
    REQUIRE(source.count==59&&source_own[ns-1]==151645);prefix_digest(&s,s.position,source_hash);REQUIRE(!strcmp(source_hash,argv[8]));
    REQUIRE(snprintf(path,sizeof(path),"%s/driver.json",argv[6])<(int)sizeof(path));FILE *f=fopen(path,"wb");REQUIRE(f);Capture capture={argv[6],NULL};
    fprintf(f,"{\"schema\":\"tcm-span-response-driver-v1\",\"model_digest\":\"%s\",\"layer\":17,\"capacity\":512,\"source_kv_sha256\":\"%s\",\"source_history_ids\":",m.digest,source_hash);ints(f,source.ids,source.count);fprintf(f,",\"correction\":");Turn correction;History corrected=source;run_turn(f,&s,NULL,0,"correction",correction_ids,nc,&corrected,&capture,&correction,NULL);
    REQUIRE(s.position==93);prefix_digest(&s,s.position,hash);REQUIRE(!strcmp(hash,argv[9]));history_digest(&corrected,hash);REQUIRE(!strcmp(hash,argv[10]));
    double compile_started=clock_seconds();float keys[2*34*128],values[2*34*128];
    for(int h=0;h<2;h++){size_t index=((size_t)17*2+h)*s.capacity*128+(size_t)59*128;gpu_read(keys+(size_t)h*34*128,s.keys+index,(size_t)34*128*4);gpu_read(values+(size_t)h*34*128,s.values+index,(size_t)34*128*4);}
    save(capture.directory,"audit-span-keys.f32",keys,sizeof(keys));save(capture.directory,"audit-span-values.f32",values,sizeof(values));SpanProgram banks[3];const int sizes[]={1,8,34};const char *labels[]={"one","eight","exact"};fprintf(f,",\"programs\":[");
    for(int j=0;j<3;j++){if(j)fputc(',',f);compile_program(f,&banks[j],sizes[j],keys,values,capture.directory,labels[j]);}
    /* Construction is complete before question token files are opened. */
    fprintf(f,"],\"programs_frozen_before_question_files_opened\":true,\"constructor_future_queries\":0,\"constructor_position\":%d,\"compile_seconds\":%.9g,\"workspace_bytes\":1048576,\"questions\":[",s.position,clock_seconds()-compile_started);REQUIRE(!fflush(f));memset(keys,0,sizeof(keys));memset(values,0,sizeof(values));span_workspace_open();
    const char *tasks[]={"recall","prose"};const int budgets[]={8,32};const char *arms[]={"native","exact","one","eight","delete"};const int modes[]={-1,34,1,8,0};
    for(int task=0;task<2;task++){
        if(task)fputc(',',f);uint64_t parent;REQUIRE(!qwen_stack_push(&s,&parent));History history=corrected;int count;int *template_ids=tokens(argv[4+task],&count);REQUIRE(count>1&&template_ids[0]==151645);int skip=history.ids[history.count-1]==151645;
        for(int j=skip;j<count-1;j++){REQUIRE(history.count<512&&!qwen_push(&s,template_ids[j]));history.ids[history.count++]=template_ids[j];}
        fprintf(f,"{\"task\":\"%s\",\"template_ids\":",tasks[task]);ints(f,template_ids,count);fprintf(f,",\"template_skip\":%d,\"arms\":[",skip);
        for(int arm=0;arm<5;arm++){if(arm)fputc(',',f);char name[96];snprintf(name,sizeof(name),"%s-%s",tasks[task],arms[arm]);SpanProgram *program=arm==1?&banks[2]:arm==2?&banks[0]:arm==3?&banks[1]:NULL;reader(f,&s,program,modes[arm],name,template_ids[count-1],budgets[task],history,&capture);}
        REQUIRE(!qwen_stack_rollback(&s,parent));prefix_digest(&s,s.position,hash);REQUIRE(!strcmp(hash,argv[9]));fprintf(f,"],\"correction_parent_intact\":true}");free(template_ids);
    }
    fprintf(f,"],\"programs_after\":[");for(int j=0;j<3;j++){if(j)fputc(',',f);program_gpu_digest(&banks[j],hash);REQUIRE(!strcmp(hash,banks[j].gpu_before));fprintf(f,"{\"name\":\"%s\",\"gpu_after_sha256\":\"%s\"}",labels[j],hash);gpu_free(banks[j].gpu_keys);gpu_free(banks[j].gpu_values);gpu_free(banks[j].gpu_logweights);span_host_free(&banks[j]);}
    fprintf(f,"],\"seconds\":%.9g}\n",clock_seconds()-started);REQUIRE(!fclose(f));span_workspace_close();qwen_session_free(&s);qwen_model_free(&m);free(prompt);free(source_own);free(correction_ids);return 0;
}
