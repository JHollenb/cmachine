/* Reuse only authenticated import, port and evidence helpers from the prior assay. */
#define main legacy_composition_main
#include "legacy_driver.c"
#undef main
static float *zero_write;

typedef struct { int ids[512],count; } History;
typedef struct { int inputs[512],count,prefill,answer[8],answers,stopped; } Turn;
static const int policies[]={0,2,4,3};
static const char *labels[]={"native","project","zero","corrected"};
static const char *tasks[]={"correction","target","collateral"};

static void ints(FILE *f,const int *v,int n){fputc('[',f);for(int j=0;j<n;j++){if(j)fputc(',',f);fprintf(f,"%d",v[j]);}fputc(']',f);}
static void history_digest(const History *h,char out[65]){REQUIRE(sizeof(int)==4);digest(h->ids,(size_t)h->count*4,out);}
static void boundary(FILE *f,const QwenSession *s,const Capture *c,const char *which){
    char file[192],hash[65];REQUIRE(snprintf(file,sizeof(file),"%s-%s.f32",c->name,which)<(int)sizeof(file));prefix_digest(s,s->position,hash);
    fprintf(f,",\"%s_kv_sha256\":\"%s\",\"%s\":",which,hash,which);readout(f,s,1151,12203,c->directory,file);
}
static void turn_advance(FILE *f,QwenSession *s,QwenSubspace *p,int policy,int index,int token,History *history,Capture *capture,const char *role,int comma){
    int active=policy!=0,position=s->position;char stem[192],history_hash[65];REQUIRE(snprintf(stem,sizeof(stem),"%s-s%03d",capture->name,index)<(int)sizeof(stem));Capture c={capture->directory,stem};
    s->observer=trace;s->observer_context=&c;s->observe_position=position;s->observe_layer=16;REQUIRE(!qwen_begin_token(s,token));QwenSubspaceStats stats={0};
    while(s->layer<s->model->layers){
        if(s->layer==16&&active){QwenSubspace held=*p;if(policy==4)held.baseline_write=zero_write;REQUIRE(!qwen_advance_layer_subspace(s,policy==4?&held:p,policy==4?QWEN_SUBSPACE_HELD:policy==3?QWEN_SUBSPACE_CORRECT:QWEN_SUBSPACE_PROJECT,&stats));}
        else REQUIRE(!qwen_advance_layer(s));
    }
    REQUIRE(!qwen_finish_token(s));selected_kv(s,position,&c);REQUIRE(history->count<512);history->ids[history->count++]=token;history_digest(history,history_hash);
    if(active){REQUIRE(stats.work.gate_up_matvecs==(policy==3?2u:0u)&&stats.work.down_matvecs==(policy==4?0u:1u)&&!stats.work.compile_matvecs);
        const float *v[]={p->reconstruction,p->omitted};const char *names[]={"reconstruction","omitted"};for(int j=0;j<2;j++){char file[224];REQUIRE(snprintf(file,sizeof(file),"%s-%s.f32",stem,names[j])<(int)sizeof(file));float *a=read_floats(v[j],(size_t)s->model->hidden);save(c.directory,file,a,(size_t)s->model->hidden*4);free(a);}}
    if(comma)fputc(',',f);fprintf(f,"{\"step\":%d,\"position\":%d,\"input_id\":%d,\"role\":\"%s\",\"history_sha256\":\"%s\",\"top\":%d,\"substitution\":%s,\"gate_up_calls\":%u,\"down_calls\":%u,\"compile_calls\":0",index,position,token,role,history_hash,s->pending,active?"true":"false",active?stats.work.gate_up_matvecs:2,policy==4?0u:1u);
    if(active)fprintf(f,",\"rho\":%.9g,\"coefficients\":[%.17g,%.17g,%.17g],\"executed\":[%.9g,%.9g,%.9g],\"omitted_norm\":%.17g",stats.rho,stats.coefficients[0],stats.coefficients[1],stats.coefficients[2],stats.executed[0],stats.executed[1],stats.executed[2],stats.omitted_norm);
    fputc('}',f);REQUIRE(!fflush(f));s->observer_context=NULL;
}
static void run_turn(FILE *f,QwenSession *s,QwenSubspace *p,int policy,const char *name,const int *template_ids,int template_count,History *history,Capture *capture,Turn *result,const Turn *forced){
    char parent_hash[65],stem[160];REQUIRE(snprintf(stem,sizeof(stem),"%s",name)<(int)sizeof(stem));Capture c={capture->directory,stem};prefix_digest(s,s->position,parent_hash);
    int parent_position=s->position,skip=history->ids[history->count-1]==template_ids[0];REQUIRE(template_count>1&&template_ids[0]==151645);memset(result,0,sizeof(*result));
    fprintf(f,"{\"name\":\"%s\",\"policy\":%d,\"autonomous\":%s,\"parent_position\":%d,\"parent_kv_sha256\":\"%s\",\"parent_history_ids\":",name,policy,forced?"false":"true",parent_position,parent_hash);ints(f,history->ids,history->count);
    fprintf(f,",\"template_ids\":");ints(f,template_ids,template_count);fprintf(f,",\"template_skip\":%d,\"host_supplied_closing_token\":%s,\"budget\":8,\"steps\":[",skip,skip?"false":"true");
    int prefill=forced?forced->prefill:template_count-skip;REQUIRE(prefill==template_count-skip&&prefill>0);
    for(int j=0;j<prefill;j++){int token=forced?forced->inputs[j]:template_ids[j+skip];REQUIRE(token==template_ids[j+skip]);result->inputs[result->count++]=token;turn_advance(f,s,p,policy,j,token,history,&c,!skip&&!j?"host-template-close":"template-user",j>0);}
    result->prefill=result->count;fputc(']',f);boundary(f,s,&c,"prefill");fprintf(f,",\"generation_steps\":[");
    for(int j=0;j<(forced?forced->answers:8);j++){
        int token=forced?forced->answer[j]:s->pending;result->answer[result->answers++]=token;result->stopped=token==s->model->eos||(s->model->eos_extra>0&&token==s->model->eos_extra);
        REQUIRE(result->count<512);result->inputs[result->count++]=token;
        turn_advance(f,s,p,policy,result->count-1,token,history,&c,forced?"sealed-replay-answer":"own-generated",j>0);
        if(result->stopped)break;
    }
    s->observe_position=-1;s->observer=NULL;fprintf(f,"],\"prefill_count\":%d,\"answer_ids\":",result->prefill);ints(f,result->answer,result->answers);
    fprintf(f,",\"stopped\":%s,\"truncated\":%s,\"all_delivered_consumed\":true,\"final_position\":%d",result->stopped?"true":"false",result->stopped?"false":"true",s->position);boundary(f,s,&c,"final");fputc('}',f);REQUIRE(!fflush(f));
    if(forced)REQUIRE(result->count==forced->count&&!memcmp(result->inputs,forced->inputs,(size_t)result->count*sizeof(int)));
    fprintf(stderr,"completed %s: %d ingested, %d consumed reply tokens, stopped=%d\n",name,result->prefill,result->answers,result->stopped);
}
int main(int argc,char **argv){
    REQUIRE(argc==11);int n,ns,nt[3];int *prompt=tokens(argv[2],&n),*source_own=tokens(argv[8],&ns),*templates[]={tokens(argv[3],&nt[0]),tokens(argv[4],&nt[1]),tokens(argv[5],&nt[2])};
    REQUIRE(n+ns<=512&&ns>0);TcmModel bundle;QwenModel m;QwenSession s;QwenSubspace p;char error[256],path[4096],source_hash[65],after[65],bank_before[65],bank_after[65];double started=clock_seconds();History source={0};Turn saved[4][3];
    REQUIRE(!tcm_load(&bundle,argv[1],error,sizeof(error))&&!qwen_model_init(&m,&bundle,error,sizeof(error)));tcm_unload(&bundle);REQUIRE(!qwen_session_init(&s,&m,512)&&!qwen_subspace_init(&p,&m));import_bank(&p,argv[7]);bank_fingerprint(&p,bank_before);REQUIRE(!strcmp(bank_before,argv[10]));
    float *zeros=calloc((size_t)m.hidden,sizeof(float));REQUIRE(zeros);zero_write=gpu_alloc((size_t)m.hidden*4);gpu_write(zero_write,zeros,(size_t)m.hidden*4);free(zeros);
    for(int j=0;j<n;j++){REQUIRE(!qwen_push(&s,prompt[j]));source.ids[source.count++]=prompt[j];}
    for(int j=0;j<ns;j++){REQUIRE(s.pending==source_own[j]&&!qwen_push(&s,source_own[j]));source.ids[source.count++]=source_own[j];}
    REQUIRE(source.count<512&&source_own[ns-1]==151645);prefix_digest(&s,s.position,source_hash);REQUIRE(!strcmp(source_hash,argv[9]));
    REQUIRE(snprintf(path,sizeof(path),"%s/driver.json",argv[6])<(int)sizeof(path));FILE *f=fopen(path,"w+b");REQUIRE(f);Capture capture={argv[6],NULL};
    fprintf(f,"{\"schema\":\"tcm-whole-turn-driver-v1\",\"model_digest\":\"%s\",\"layer\":16,\"capacity\":512,\"source_replay_checked_before_candidates\":true,\"source_position\":%d,\"source_kv_sha256\":\"%s\",\"source_history_ids\":",m.digest,s.position,source_hash);ints(f,source.ids,source.count);fprintf(f,",\"gpu_program_workspace_bytes\":%zu,\"host_array_bytes\":%zu,\"zero_gpu_bytes\":%zu,\"arms\":[",p.gpu_bytes,p.host_bytes,(size_t)m.hidden*4);
    for(int arm=0;arm<4;arm++){
        if(arm)fputc(',',f);uint64_t root,child;REQUIRE(!qwen_stack_push(&s,&root));History h=source;char name[160];fprintf(f,"{\"name\":\"%s\",\"policy\":%d,\"turns\":[",labels[arm],policies[arm]);
        REQUIRE(snprintf(name,sizeof(name),"%s-correction",labels[arm])<(int)sizeof(name));run_turn(f,&s,&p,policies[arm],name,templates[0],nt[0],&h,&capture,&saved[arm][0],NULL);History corrected=h;
        for(int t=1;t<3;t++){fputc(',',f);REQUIRE(!qwen_stack_push(&s,&child));h=corrected;REQUIRE(snprintf(name,sizeof(name),"%s-%s",labels[arm],tasks[t])<(int)sizeof(name));run_turn(f,&s,&p,policies[arm],name,templates[t],nt[t],&h,&capture,&saved[arm][t],NULL);REQUIRE(!qwen_stack_rollback(&s,child));}
        REQUIRE(!qwen_stack_rollback(&s,root));prefix_digest(&s,s.position,after);REQUIRE(!strcmp(source_hash,after));fprintf(f,"],\"source_prefix_intact\":true}");
    }
    fprintf(f,"]");REQUIRE(!fflush(f));long sealed=ftell(f);REQUIRE(sealed>0);unsigned char *bytes=malloc((size_t)sealed);REQUIRE(bytes&&!fseek(f,0,SEEK_SET)&&fread(bytes,1,(size_t)sealed,f)==(size_t)sealed);char seal[65];digest(bytes,(size_t)sealed,seal);free(bytes);REQUIRE(!fseek(f,0,SEEK_END));
    fprintf(f,",\"consumer_prefix_seal_bytes\":%ld,\"consumer_prefix_seal_sha256\":\"%s\",\"diagnostics_after_seal\":true,\"diagnostics\":[",sealed,seal);REQUIRE(!fflush(f));fprintf(stderr,"sealed all primary consumer paths: %s\n",seal);
    for(int arm=1;arm<4;arm++){
        if(arm>1)fputc(',',f);uint64_t root,child;REQUIRE(!qwen_stack_push(&s,&root));History h=source;Turn result;char name[160];fprintf(f,"{\"candidate\":\"%s\",\"turns\":[",labels[arm]);
        REQUIRE(snprintf(name,sizeof(name),"diag-%s-correction",labels[arm])<(int)sizeof(name));run_turn(f,&s,&p,0,name,templates[0],nt[0],&h,&capture,&result,&saved[arm][0]);History corrected=h;
        for(int t=1;t<3;t++){fputc(',',f);REQUIRE(!qwen_stack_push(&s,&child));h=corrected;REQUIRE(snprintf(name,sizeof(name),"diag-%s-%s",labels[arm],tasks[t])<(int)sizeof(name));run_turn(f,&s,&p,0,name,templates[t],nt[t],&h,&capture,&result,&saved[arm][t]);REQUIRE(!qwen_stack_rollback(&s,child));}
        REQUIRE(!qwen_stack_rollback(&s,root));prefix_digest(&s,s.position,after);REQUIRE(!strcmp(source_hash,after));fprintf(f,"],\"source_prefix_intact\":true}");
    }
    bank_fingerprint(&p,bank_after);REQUIRE(!strcmp(bank_before,bank_after));fprintf(f,"],\"frozen_bank_before_sha256\":\"%s\",\"frozen_bank_after_sha256\":\"%s\",\"seconds\":%.9g}\n",bank_before,bank_after,clock_seconds()-started);REQUIRE(!fclose(f));
    gpu_free(zero_write);qwen_subspace_free(&p);qwen_session_free(&s);qwen_model_free(&m);free(prompt);free(source_own);for(int j=0;j<3;j++)free(templates[j]);return 0;
}
