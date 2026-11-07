#define _POSIX_C_SOURCE 200809L
#define REQUIRE(x) do{if(!(x)){fprintf(stderr,"gate-law %s:%d: %s\n",__FILE__,__LINE__,#x);exit(2);}}while(0)
#include "assay_io.h"
#include <dirent.h>
#include <sys/stat.h>

#define MAX_STEPS 384
typedef struct { int token,role,full; } Event;
typedef struct { Event event[MAX_STEPS];int count,answer[2][16],answers[2],stopped[2],skip; } Trajectory;
typedef struct { const char *directory,*name;QwenGateLaw *compile;const QwenModel *model; } Observation;
static const char *policy_names[]={"native","once","recurring","zero"};
static const char *roles[]={"initial-prompt","assistant","followup-template","host-close"};
static const char *output_directory;
static unsigned source_steps,primary_steps,reference_steps,replay_steps,local_native_steps,saved_gate_diagnostics;

static void seal_primary_files(const char *prefix_hash,long prefix_bytes){
    char path[4096];REQUIRE(snprintf(path,sizeof(path),"%s/primary-seal.json",output_directory)<(int)sizeof(path));FILE *out=fopen(path,"w");REQUIRE(out);
    fprintf(out,"{\"driver_prefix_bytes\":%ld,\"driver_prefix_sha256\":\"%s\",\"files\":[",prefix_bytes,prefix_hash);DIR *dir=opendir(output_directory);REQUIRE(dir);struct dirent *entry;int count=0;
    while((entry=readdir(dir))){
        size_t n=strlen(entry->d_name);if(n<4||(strcmp(entry->d_name+n-4,".f32")&&strcmp(entry->d_name+n-4,".txt")))continue;
        REQUIRE(!strchr(entry->d_name,'"')&&snprintf(path,sizeof(path),"%s/%s",output_directory,entry->d_name)<(int)sizeof(path));struct stat st;REQUIRE(!stat(path,&st)&&S_ISREG(st.st_mode));
        FILE *input=fopen(path,"rb");REQUIRE(input);unsigned char *bytes=malloc((size_t)st.st_size);REQUIRE(bytes&&fread(bytes,1,(size_t)st.st_size,input)==(size_t)st.st_size&&!fclose(input));char hash[65];digest(bytes,(size_t)st.st_size,hash);free(bytes);
        if(count++)fputc(',',out);fprintf(out,"{\"name\":\"%s\",\"bytes\":%zu,\"sha256\":\"%s\"}",entry->d_name,(size_t)st.st_size,hash);
    }
    REQUIRE(!closedir(dir));fprintf(out,"]}\n");REQUIRE(!fclose(out));
}

static void ints(FILE *f,const int *v,int n){fputc('[',f);for(int j=0;j<n;j++){if(j)fputc(',',f);fprintf(f,"%d",v[j]);}fputc(']',f);}
static void array_file(const char *stem,const char *suffix,const float *data,int n){
    char file[256];REQUIRE(snprintf(file,sizeof(file),"%s-%s.f32",stem,suffix)<(int)sizeof(file));
    float *v=read_floats(data,(size_t)n);save(output_directory,file,v,(size_t)n*4);free(v);
}
static void trace_gate(void *context,int position,int layer,QwenPort port_id,const float *data,size_t count){
    Observation *o=context;REQUIRE(layer==16);
    if(o->compile&&port_id==QWEN_MLP_GATE){REQUIRE(!qwen_gate_law_init(o->compile,o->model,layer,position,data));return;}
    if(!o->name)return;
    int reference=!strncmp(o->name,"same-",5)||!strncmp(o->name,"native-",7);
    if(reference&&(port_id==QWEN_MLP_GATE||port_id==QWEN_MLP_UP))return;
    if(port_id!=QWEN_RAW_RECEIVER&&port_id!=QWEN_MLP_INPUT&&port_id!=QWEN_MLP_GATE&&port_id!=QWEN_MLP_UP&&port_id!=QWEN_MLP_RETURN&&port_id!=QWEN_OUTPUT)return;
    array_file(o->name,qwen_port_name(port_id),data,(int)count);
}
static void current_kv(const QwenSession *s,int position,const char *stem){
    const QwenModel *m=s->model;int count=2*(m->layers-16)*m->kvheads*m->head,n=0;float *v=malloc((size_t)count*4);REQUIRE(v);
    for(int kind=0;kind<2;kind++)for(int layer=16;layer<m->layers;layer++)for(int head=0;head<m->kvheads;head++){
        size_t offset=(((size_t)layer*m->kvheads+head)*s->capacity+position)*m->head;
        gpu_read(v+n,(kind?s->values:s->keys)+offset,(size_t)m->head*4);n+=m->head;
    }
    for(int j=0;j<n;j++)REQUIRE(isfinite(v[j]));char file[256];REQUIRE(snprintf(file,sizeof(file),"%s-kv16-27.f32",stem)<(int)sizeof(file));save(output_directory,file,v,(size_t)n*4);free(v);
}
static void law_hash(const QwenGateLaw *law,char hash[65],const char *stem){
    int h=law->model->intermediate;float *v=malloc((size_t)5*h*4);REQUIRE(v);const float *fields[]={law->source_gate,law->probability,law->first,law->second,law->source_silu};
    for(int i=0;i<5;i++)gpu_read(v+(size_t)i*h,fields[i],(size_t)h*4);
    digest(v,(size_t)5*h*4,hash);if(stem){char file[192];REQUIRE(snprintf(file,sizeof(file),"%s-law.f32",stem)<(int)sizeof(file));save(output_directory,file,v,(size_t)5*h*4);}free(v);
}
static void build_source(QwenSession *s,QwenGateLaw *law,const int *ids,int n){
    REQUIRE(n>3&&!qwen_reset(s));for(int j=0;j<n-2;j++){REQUIRE(!qwen_push(s,ids[j]));source_steps++;}
    Observation o={output_directory,NULL,law,s->model};s->observer=trace_gate;s->observer_context=&o;s->observe_position=s->position;s->observe_layer=16;
    REQUIRE(!qwen_push(s,ids[n-2]));source_steps++;s->observer=NULL;s->observer_context=NULL;REQUIRE(law->ready&&s->position==n-1);
}
static QwenGateMode selected_mode(int policy,int step){return policy==3?QWEN_GATE_ZERO:policy==2||(policy==1&&!step)?QWEN_GATE_POLYNOMIAL:QWEN_GATE_NATIVE;}
static void consume(FILE *f,QwenSession *s,const QwenGateLaw *law,int policy,int index,Event event,const char *name,int phase){
    char stem[192],file[256],hash[65];REQUIRE(snprintf(stem,sizeof(stem),"%s-s%03d",name,index)<(int)sizeof(stem));int position=s->position;uint64_t parent=s->lineage_id;QwenGateWork work;
    Observation o={output_directory,stem,NULL,s->model};s->observer=trace_gate;s->observer_context=&o;s->observe_position=position;s->observe_layer=16;
    REQUIRE(!qwen_begin_token(s,event.token));QwenGateMode mode=selected_mode(policy,index);
    while(s->layer<s->model->layers){
        if(s->layer==16){REQUIRE(!qwen_advance_layer_gate(s,law,mode,parent,&work));if(mode!=QWEN_GATE_ZERO&&strncmp(name,"same-",5)&&strncmp(name,"native-",7))array_file(stem,"product",s->gate,s->model->intermediate);}
        else REQUIRE(!qwen_advance_layer(s));
    }
    REQUIRE(!qwen_finish_token(s));s->observer=NULL;s->observer_context=NULL;current_kv(s,position,stem);
    if(phase==0)primary_steps++;else if(phase==1)reference_steps++;else if(phase==2)replay_steps++;else local_native_steps++;
    if(index)fputc(',',f);fprintf(f,"{\"step\":%d,\"position\":%d,\"input_id\":%d,\"role\":\"%s\",\"mode\":%d,\"parent_lineage\":%llu,\"result_lineage\":%llu,\"top\":%d,\"runner_up\":%d,\"margin\":%.9g,\"gate_up_calls\":%u,\"down_calls\":%u,\"native_swiglu\":%u,\"polynomial_swiglu\":%u",index,position,event.token,roles[event.role],mode,(unsigned long long)parent,(unsigned long long)s->lineage_id,s->pending,s->top.second,s->top.first_value-s->top.second_value,work.linear.gate_up_matvecs,work.linear.down_matvecs,work.native_swiglu,work.polynomial_swiglu);
    if(event.full){REQUIRE(snprintf(file,sizeof(file),"%s-logits.f32",stem)<(int)sizeof(file));prefix_digest(s,s->position,hash);fprintf(f,",\"valid_kv_sha256\":\"%s\",\"readout\":",hash);readout(f,s,1151,12203,output_directory,file);}
    fputc('}',f);REQUIRE(!fflush(f));
}
static void add_event(FILE *f,QwenSession *s,const QwenGateLaw *law,int policy,Trajectory *t,int token,int role,int full,const char *name,int phase){
    REQUIRE(t->count<MAX_STEPS);Event event={token,role,full};consume(f,s,law,policy,t->count,event,name,phase);t->event[t->count++]=event;
}
static void path_fields(FILE *f,const Trajectory *t,double seconds){
    fprintf(f,"],\"answers\":[");for(int turn=0;turn<2;turn++){if(turn)fputc(',',f);fprintf(f,"{\"ids\":");ints(f,t->answer[turn],t->answers[turn]);fprintf(f,",\"stopped\":%s,\"truncated\":%s}",t->stopped[turn]?"true":"false",t->stopped[turn]?"false":"true");}
    fprintf(f,"],\"consumed_ids\":[");for(int j=0;j<t->count;j++){if(j)fputc(',',f);fprintf(f,"%d",t->event[j].token);}fprintf(f,"],\"template_skip\":%d,\"all_delivered_consumed\":true,\"instrumented_seconds\":%.9g}",t->skip,seconds);
}
static void generate_path(FILE *f,QwenSession *s,const QwenGateLaw *law,int policy,int final_prompt,const int *followup,int nf,const char *name,Trajectory *t,int phase){
    double started=clock_seconds();
    memset(t,0,sizeof(*t));fprintf(f,"{\"name\":\"%s\",\"policy\":%d,\"autonomous\":true,\"steps\":[",name,policy);
    add_event(f,s,law,policy,t,final_prompt,0,1,name,phase);
    for(int turn=0;turn<2;turn++){
        for(int j=0;j<16;j++){
            int token=s->pending;int stopped=token==s->model->eos||(s->model->eos_extra>0&&token==s->model->eos_extra);
            t->answer[turn][t->answers[turn]++]=token;t->stopped[turn]=stopped;
            add_event(f,s,law,policy,t,token,1,stopped||j==15,name,phase);if(stopped)break;
        }
        if(!turn){t->skip=t->answer[0][t->answers[0]-1]==followup[0];REQUIRE(nf>1&&followup[0]==151645);
            for(int j=t->skip;j<nf;j++)add_event(f,s,law,policy,t,followup[j],!t->skip&&!j?3:2,j==nf-1,name,phase);}
    }
    path_fields(f,t,clock_seconds()-started);fprintf(stderr,"completed %s: %d consumed events, answers %d/%d, EOS %d/%d\n",name,t->count,t->answers[0],t->answers[1],t->stopped[0],t->stopped[1]);
}
static void replay_path(FILE *f,QwenSession *s,const QwenGateLaw *law,const Trajectory *t,int policy,const char *name,int probe){
    double started=clock_seconds();
    fprintf(f,"{\"name\":\"%s\",\"policy\":%d,\"autonomous\":false,\"steps\":[",name,policy);
    int full_seen=0;
    for(int j=0;j<t->count;j++){
        /* Probe only initial and follow-up prefill outputs, on this candidate's
         * actual retained parent. All these calls occur after primary sealing. */
        int answer_clock=t->event[j].full&&(j==0||t->event[j].role==2);
        if(probe&&answer_clock){
            uint64_t frame;REQUIRE(!qwen_stack_push(s,&frame));char probe_name[192];REQUIRE(snprintf(probe_name,sizeof(probe_name),"local-%s-p%d",name,full_seen++)<(int)sizeof(probe_name));
            char path[4096];REQUIRE(snprintf(path,sizeof(path),"%s/%s.json",output_directory,probe_name)<(int)sizeof(path));FILE *g=fopen(path,"w");REQUIRE(g);fprintf(g,"{\"candidate\":\"%s\",\"candidate_step\":%d,\"steps\":[",name,j);
            consume(g,s,law,0,0,t->event[j],probe_name,3);fprintf(g,"]}\n");REQUIRE(!fclose(g)&&!qwen_stack_rollback(s,frame));
        }
        consume(f,s,law,policy,j,t->event[j],name,2);
    }
    path_fields(f,t,clock_seconds()-started);fprintf(stderr,"completed replay %s\n",name);
}
static void saved_gate_diagnostic(QwenSession *s,const Trajectory *t,const char *name){
    int h=s->model->intermediate,d=s->model->hidden;float *host=malloc((size_t)h*4),*gate=gpu_alloc((size_t)h*4),*up=gpu_alloc((size_t)h*4),*out=gpu_alloc((size_t)d*4);REQUIRE(host);
    for(int j=0;j<t->count;j++){
        if(selected_mode(!strncmp(name,"once",4)?1:2,j)==QWEN_GATE_NATIVE)continue;
        char stem[192],path[4096];REQUIRE(snprintf(stem,sizeof(stem),"%s-s%03d",name,j)<(int)sizeof(stem));
        REQUIRE(snprintf(path,sizeof(path),"%s/%s-mlp_gate.f32",output_directory,stem)<(int)sizeof(path));FILE *f=fopen(path,"rb");REQUIRE(f&&fread(host,4,(size_t)h,f)==(size_t)h&&fgetc(f)==EOF&&!fclose(f));gpu_write(gate,host,(size_t)h*4);
        REQUIRE(snprintf(path,sizeof(path),"%s/%s-mlp_up.f32",output_directory,stem)<(int)sizeof(path));f=fopen(path,"rb");REQUIRE(f&&fread(host,4,(size_t)h,f)==(size_t)h&&fgetc(f)==EOF&&!fclose(f));gpu_write(up,host,(size_t)h*4);
        gpu_swiglu(gate,up,h);gpu_matvec(s->model->gpu,s->model->layer[16].down,h,d,gate,out);array_file(stem,"exact-product",gate,h);array_file(stem,"exact-write",out,d);saved_gate_diagnostics++;
    }
    free(host);gpu_free(gate);gpu_free(up);gpu_free(out);
}
int main(int argc,char **argv){
    REQUIRE(argc==7);output_directory=argv[6];int n[2],nf[2];int *ids[]={tokens(argv[2],n),tokens(argv[4],n+1)},*follow[]={tokens(argv[3],nf),tokens(argv[5],nf+1)};
    TcmModel bundle;QwenModel model;QwenSession session;char error[256],path[4096],hash[65],after[65],source_hash[2][65],law_digest[2][65];QwenGateLaw laws[2]={{0}};Trajectory primary[2][3];double started=clock_seconds();
    REQUIRE(!tcm_load(&bundle,argv[1],error,sizeof(error))&&!qwen_model_init(&model,&bundle,error,sizeof(error)));tcm_unload(&bundle);REQUIRE(!qwen_session_init(&session,&model,512));
    REQUIRE(snprintf(path,sizeof(path),"%s/driver.json",output_directory)<(int)sizeof(path));FILE *f=fopen(path,"w+b");REQUIRE(f);fprintf(f,"{\"schema\":\"tcm-gate-law-driver-v1\",\"model_digest\":\"%s\",\"layer\":16,\"capacity\":512,\"budget\":16,\"primary_cases\":[",model.digest);
    for(int c=0;c<2;c++){
        if(c)fputc(',',f);build_source(&session,laws+c,ids[c],n[c]);prefix_digest(&session,session.position,source_hash[c]);char case_name[32];snprintf(case_name,sizeof(case_name),"case%d",c);law_hash(laws+c,law_digest[c],case_name);
        fprintf(f,"{\"case\":%d,\"source_position\":%d,\"source_kv_sha256\":\"%s\",\"source_anchor_position\":%d,\"law_sha256\":\"%s\",\"law_gpu_bytes\":%zu,\"arms\":[",c,session.position,source_hash[c],laws[c].source_position,law_digest[c],laws[c].gpu_bytes);
        for(int p=1;p<=3;p++){
            if(p>1)fputc(',',f);uint64_t frame;REQUIRE(!qwen_stack_push(&session,&frame));char name[64];snprintf(name,sizeof(name),"%s-c%d",policy_names[p],c);
            generate_path(f,&session,laws+c,p,ids[c][n[c]-1],follow[c],nf[c],name,&primary[c][p-1],0);
            REQUIRE(!qwen_stack_rollback(&session,frame));prefix_digest(&session,session.position,after);REQUIRE(!strcmp(source_hash[c],after));
        }
        law_hash(laws+c,after,NULL);REQUIRE(!strcmp(after,law_digest[c]));fprintf(f,"],\"source_prefix_intact\":true,\"frozen_law_intact\":true}");
    }
    fprintf(f,"]");REQUIRE(!fflush(f));long sealed=ftell(f);REQUIRE(sealed>0);unsigned char *bytes=malloc((size_t)sealed);REQUIRE(bytes&&!fseek(f,0,SEEK_SET)&&fread(bytes,1,(size_t)sealed,f)==(size_t)sealed);digest(bytes,(size_t)sealed,hash);free(bytes);REQUIRE(!fseek(f,0,SEEK_END));
    seal_primary_files(hash,sealed);
    fprintf(f,",\"primary_seal_bytes\":%ld,\"primary_seal_sha256\":\"%s\",\"diagnostics_after_seal\":true,\"references\":[",sealed,hash);REQUIRE(!fflush(f));fprintf(stderr,"sealed all six primary paths: %s\n",hash);
    for(int c=0;c<2;c++){
        if(c)fputc(',',f);QwenGateLaw rebuilt={0};build_source(&session,&rebuilt,ids[c],n[c]);law_hash(&rebuilt,after,NULL);REQUIRE(!strcmp(after,law_digest[c]));qwen_gate_law_free(&rebuilt);prefix_digest(&session,session.position,after);REQUIRE(!strcmp(after,source_hash[c]));
        fprintf(f,"{\"case\":%d,\"native\":",c);uint64_t frame;REQUIRE(!qwen_stack_push(&session,&frame));char name[80];Trajectory native;snprintf(name,sizeof(name),"native-c%d",c);generate_path(f,&session,laws+c,0,ids[c][n[c]-1],follow[c],nf[c],name,&native,1);REQUIRE(!qwen_stack_rollback(&session,frame));
        fprintf(f,",\"same_ids\":[");for(int p=1;p<=3;p++){if(p>1)fputc(',',f);REQUIRE(!qwen_stack_push(&session,&frame));snprintf(name,sizeof(name),"same-%s-c%d",policy_names[p],c);replay_path(f,&session,laws+c,&primary[c][p-1],0,name,0);REQUIRE(!qwen_stack_rollback(&session,frame));}
        fprintf(f,"],\"candidate_replays\":[");for(int p=1;p<=3;p++){if(p>1)fputc(',',f);REQUIRE(!qwen_stack_push(&session,&frame));snprintf(name,sizeof(name),"replay-%s-c%d",policy_names[p],c);replay_path(f,&session,laws+c,&primary[c][p-1],p,name,1);REQUIRE(!qwen_stack_rollback(&session,frame));}
        fprintf(f,"]");
        for(int p=1;p<=2;p++){snprintf(name,sizeof(name),"%s-c%d",policy_names[p],c);saved_gate_diagnostic(&session,&primary[c][p-1],name);}
        law_hash(laws+c,after,NULL);REQUIRE(!strcmp(after,law_digest[c]));
        int *changed=malloc((size_t)n[c]*sizeof(int));REQUIRE(changed);memcpy(changed,ids[c],(size_t)n[c]*sizeof(int));changed[n[c]-2]=(changed[n[c]-2]+1)%model.vocab;
        QwenGateLaw alternate={0};build_source(&session,&alternate,changed,n[c]);snprintf(name,sizeof(name),"changed-source-c%d",c);law_hash(&alternate,after,name);
        fprintf(f,",\"source_token_control\":{\"position\":%d,\"original_id\":%d,\"changed_id\":%d,\"law_sha256\":\"%s\",\"law_changed\":%s}}",n[c]-2,ids[c][n[c]-2],changed[n[c]-2],after,strcmp(after,law_digest[c])?"true":"false");qwen_gate_law_free(&alternate);free(changed);
    }
    fprintf(f,"],\"work\":{\"source_native_tokens\":%u,\"primary_tokens\":%u,\"reference_tokens\":%u,\"replay_tokens\":%u,\"local_native_tokens\":%u,\"saved_exact_swiglu_down_diagnostics\":%u},\"seconds\":%.9g}\n",source_steps,primary_steps,reference_steps,replay_steps,local_native_steps,saved_gate_diagnostics,clock_seconds()-started);REQUIRE(!fclose(f));
    for(int c=0;c<2;c++){qwen_gate_law_free(laws+c);free(ids[c]);free(follow[c]);}qwen_session_free(&session);qwen_model_free(&model);return 0;
}
