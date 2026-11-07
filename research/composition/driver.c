#define _POSIX_C_SOURCE 200809L
#define REQUIRE(x) do{if(!(x)){fprintf(stderr,"composition %s:%d: %s\n",__FILE__,__LINE__,#x);exit(2);}}while(0)
#include "assay_io.h"
static float *zero_write;

static void blob(const char *dir,const char *name,void *data,size_t bytes){
    char path[4096];REQUIRE(snprintf(path,sizeof(path),"%s/%s",dir,name)<(int)sizeof(path));FILE *f=fopen(path,"rb");REQUIRE(f);
    REQUIRE(fread(data,1,bytes,f)==bytes&&fgetc(f)==EOF&&!ferror(f)&&!fclose(f));
}
static void import_bank(QwenSubspace *p,const char *dir){
    int d=p->model->hidden,h=p->model->intermediate;REQUIRE(d==1536&&h==8960);
    size_t n=(size_t)3*d+4*h,ne=(size_t)d+2*h;float *base=malloc(n*4),*extra=malloc(ne*4);double *qr=malloc(((size_t)2*d+4)*8),*qe=malloc(((size_t)d+3)*8);REQUIRE(base&&extra&&qr&&qe);
    blob(dir,"reuse-program.f32",base,n*4);blob(dir,"enrich-program.f32",extra,ne*4);blob(dir,"reuse-qr.f64",qr,((size_t)2*d+4)*8);blob(dir,"enrich-qr.f64",qe,((size_t)d+3)*8);
    for(size_t j=0;j<n;j++)REQUIRE(isfinite(base[j]));for(size_t j=0;j<ne;j++)REQUIRE(isfinite(extra[j]));
    for(size_t j=0;j<(size_t)2*d+4;j++)REQUIRE(isfinite(qr[j]));for(size_t j=0;j<(size_t)d+3;j++)REQUIRE(isfinite(qe[j]));
    for(int k=0;k<2;k++){gpu_write(p->basis[k],base+(size_t)k*d,(size_t)d*4);gpu_write(p->gate[k],base+2*d+(size_t)k*h,(size_t)h*4);gpu_write(p->up[k],base+2*d+2*h+(size_t)k*h,(size_t)h*4);}
    gpu_write(p->baseline_write,base+n-d,(size_t)d*4);gpu_write(p->basis[2],extra,(size_t)d*4);gpu_write(p->gate[2],extra+d,(size_t)h*4);gpu_write(p->up[2],extra+d+h,(size_t)h*4);
    memcpy(p->q0,qr,(size_t)d*8);memcpy(p->q1,qr+d,(size_t)d*8);memcpy(p->q2,qe,(size_t)d*8);
    p->r00=qr[2*d];p->r01=qr[2*d+1];p->r11=qr[2*d+2];p->condition=qr[2*d+3];p->r02=qe[d];p->r12=qe[d+1];p->r22=qe[d+2];p->layer=16;p->rank=3;p->ready=1;
    REQUIRE(p->r00>0&&p->r11>0&&p->r22>0);free(base);free(extra);free(qr);free(qe);
}
static void bank_fingerprint(const QwenSubspace *p,char hash[65]){
    int d=p->model->hidden,h=p->model->intermediate;size_t bytes=((size_t)4*d+6*h)*4+((size_t)3*d+7)*8,offset=0;unsigned char *raw=malloc(bytes);REQUIRE(raw);
#define GPU_PART(pointer,count) do{size_t size=(size_t)(count)*4;gpu_read(raw+offset,pointer,size);offset+=size;}while(0)
#define HOST_PART(pointer,count) do{size_t size=(size_t)(count)*8;memcpy(raw+offset,pointer,size);offset+=size;}while(0)
    for(int j=0;j<2;j++)GPU_PART(p->basis[j],d);for(int j=0;j<2;j++)GPU_PART(p->gate[j],h);for(int j=0;j<2;j++)GPU_PART(p->up[j],h);GPU_PART(p->baseline_write,d);
    HOST_PART(p->q0,d);HOST_PART(p->q1,d);double qr[]={p->r00,p->r01,p->r11,p->condition};HOST_PART(qr,4);
    GPU_PART(p->basis[2],d);GPU_PART(p->gate[2],h);GPU_PART(p->up[2],h);HOST_PART(p->q2,d);double qe[]={p->r02,p->r12,p->r22};HOST_PART(qe,3);
#undef GPU_PART
#undef HOST_PART
    REQUIRE(offset==bytes);digest(raw,bytes,hash);free(raw);
}
static void trace(void *context,int position,int layer,QwenPort id,const float *data,size_t count){
    (void)position;REQUIRE(layer==16);Capture *c=context;size_t length=strlen(c->name);int first=length>=4&&!strcmp(c->name+length-4,"s000");
    if(id!=QWEN_INPUT&&id!=QWEN_RAW_RECEIVER&&id!=QWEN_MLP_INPUT&&id!=QWEN_MLP_RETURN&&id!=QWEN_OUTPUT&&!(first&&(id==QWEN_MLP_GATE||id==QWEN_MLP_UP)))return;
    char file[160];REQUIRE(snprintf(file,sizeof(file),"%s-%s.f32",c->name,qwen_port_name(id))<(int)sizeof(file));float *v=read_floats(data,count);save(c->directory,file,v,count*4);free(v);
}
static void selected_kv(const QwenSession *s,int position,const Capture *c){
    const QwenModel *m=s->model;float v[1024];size_t n=0;REQUIRE(m->kvheads*m->head==256);
    for(int kind=0;kind<2;kind++)for(int layer=16;layer<=17;layer++)for(int k=0;k<m->kvheads;k++){
        size_t off=(((size_t)layer*m->kvheads+k)*s->capacity+position)*m->head;gpu_read(v+n,(kind?s->values:s->keys)+off,(size_t)m->head*4);n+=m->head;
    }char file[160];REQUIRE(n==1024);for(size_t j=0;j<n;j++)REQUIRE(isfinite(v[j]));REQUIRE(snprintf(file,sizeof(file),"%s-kv16-17.f32",c->name)<(int)sizeof(file));save(c->directory,file,v,n*4);
}
static void advance(FILE *f,QwenSession *s,QwenSubspace *p,int policy,int step,int token,Capture *c){
    int active=policy==2||policy==3||policy==4||(policy==1&&step==0);s->observer=trace;s->observer_context=c;s->observe_position=s->position;s->observe_layer=16;int position=s->position;
    REQUIRE(!qwen_begin_token(s,token));QwenSubspaceStats stats={0};
    while(s->layer<s->model->layers){
        if(s->layer==16&&active){QwenSubspace held=*p;if(policy==4){REQUIRE(zero_write);held.baseline_write=zero_write;}REQUIRE(!qwen_advance_layer_subspace(s,policy==4?&held:p,policy==4?QWEN_SUBSPACE_HELD:policy==3?QWEN_SUBSPACE_CORRECT:QWEN_SUBSPACE_PROJECT,&stats));}
        else REQUIRE(!qwen_advance_layer(s));
    }
    REQUIRE(!qwen_finish_token(s));selected_kv(s,position,c);
    if(active)REQUIRE(stats.work.gate_up_matvecs==(policy==3?2u:0u)&&stats.work.down_matvecs==(policy==4?0u:1u)&&!stats.work.compile_matvecs);
    if(active){const float *a[]={p->reconstruction,p->omitted,p->correction_gate,p->correction_up};const char *names[]={"reconstruction","omitted","correction-gate","correction-up"};int count=policy==3&&!step?4:2;for(int j=0;j<count;j++){int extent=j<2?s->model->hidden:s->model->intermediate;char file[160];REQUIRE(snprintf(file,sizeof(file),"%s-%s.f32",c->name,names[j])<(int)sizeof(file));float *v=read_floats(a[j],(size_t)extent);save(c->directory,file,v,(size_t)extent*4);free(v);}}
    if(step)fputc(',',f);fprintf(f,"{\"step\":%d,\"position\":%d,\"input_id\":%d,\"top\":%d,\"runner_up\":%d,\"margin\":%.9g,\"substitution\":%s,\"gate_up_calls\":%u,\"down_calls\":%u,\"compile_calls\":0",step,position,token,s->pending,s->top.second,s->top.first_value-s->top.second_value,active?"true":"false",active?stats.work.gate_up_matvecs:2,policy==4?0u:1u);
    if(active)fprintf(f,",\"rho\":%.9g,\"omitted_norm\":%.17g,\"receiver_norm\":%.17g,\"coefficients\":[%.17g,%.17g,%.17g],\"executed\":[%.9g,%.9g,%.9g]",stats.rho,stats.omitted_norm,stats.receiver_norm,stats.coefficients[0],stats.coefficients[1],stats.coefficients[2],stats.executed[0],stats.executed[1],stats.executed[2]);
    if(!step){char hash[65];prefix_digest(s,s->position,hash);fprintf(f,",\"valid_kv_sha256\":\"%s\",\"first\":",hash);char file[160];REQUIRE(snprintf(file,sizeof(file),"%s-first.f32",c->name)<(int)sizeof(file));readout(f,s,1151,12203,c->directory,file);}
    fputc('}',f);REQUIRE(!fflush(f));
}
static void run_arm(FILE *f,QwenSession *s,QwenSubspace *p,int policy,const char *task,int final_id,int budget,Capture *capture){
    const char *names[]={"native","once","recurring","corrected","zero"};char name[128],stem[160],hash[65],after[65],file[160];int ids[48],count=0,stopped=0;uint64_t branch;
    REQUIRE(budget>0&&budget<=48);REQUIRE(snprintf(name,sizeof(name),"%s-%s",task,names[policy])<(int)sizeof(name));REQUIRE(!qwen_stack_push(s,&branch));int parent_position=s->position;prefix_digest(s,parent_position,hash);
    fprintf(f,"{\"name\":\"%s\",\"task\":\"%s\",\"policy\":%d,\"parent_position\":%d,\"parent_kv_sha256\":\"%s\",\"budget\":%d,\"steps\":[",name,task,policy,parent_position,hash,budget);
    REQUIRE(snprintf(stem,sizeof(stem),"%s-s000",name)<(int)sizeof(stem));capture->name=stem;advance(f,s,p,policy,0,final_id,capture);
    for(int j=0;j<budget;j++){
        int token=s->pending;ids[count++]=token;stopped=token==s->model->eos||(s->model->eos_extra>0&&token==s->model->eos_extra);
        REQUIRE(snprintf(stem,sizeof(stem),"%s-s%03d",name,j+1)<(int)sizeof(stem));advance(f,s,p,policy,j+1,token,capture);if(stopped)break;
    }
    s->observe_position=-1;fprintf(f,"],\"own\":{\"ids\":[");for(int j=0;j<count;j++){if(j)fputc(',',f);fprintf(f,"%d",ids[j]);}fprintf(f,"],\"stopped\":%s,\"truncated\":%s,\"all_delivered_consumed\":true},\"final\":",stopped?"true":"false",stopped?"false":"true");
    REQUIRE(snprintf(file,sizeof(file),"%s-final.f32",name)<(int)sizeof(file));readout(f,s,1151,12203,capture->directory,file);prefix_digest(s,s->position,after);fprintf(f,",\"final_position\":%d,\"final_valid_kv_sha256\":\"%s\"",s->position,after);
    REQUIRE(!qwen_stack_rollback(s,branch));prefix_digest(s,parent_position,after);REQUIRE(!strcmp(hash,after));fprintf(f,",\"prefix_intact\":true}");REQUIRE(!fflush(f));
    fprintf(stderr,"completed %s: %d emitted/consumed tokens, stopped=%d\n",name,count,stopped);
}
int main(int argc,char **argv){
    int null_panel=argc==12&&!strcmp(argv[11],"zero");REQUIRE(argc==11||null_panel);int n,nt[3],nsource;int *expected_source=tokens(argv[8],&nsource);int *prompt=tokens(argv[2],&n),*targets[]={tokens(argv[3],&nt[0]),tokens(argv[4],&nt[1]),tokens(argv[5],&nt[2])};
    TcmModel bundle;QwenModel m;QwenSession s;QwenSubspace p;char error[256],path[4096];double started=clock_seconds();
    REQUIRE(!tcm_load(&bundle,argv[1],error,sizeof(error)));REQUIRE(!qwen_model_init(&m,&bundle,error,sizeof(error)));tcm_unload(&bundle);REQUIRE(!qwen_session_init(&s,&m,512));REQUIRE(!qwen_subspace_init(&p,&m));import_bank(&p,argv[7]);
    char bank_before[65],bank_after[65];bank_fingerprint(&p,bank_before);REQUIRE(!strcmp(bank_before,argv[10]));
    if(null_panel){float *zeros=calloc((size_t)m.hidden,sizeof(float));REQUIRE(zeros);zero_write=gpu_alloc((size_t)m.hidden*4);gpu_write(zero_write,zeros,(size_t)m.hidden*4);free(zeros);}
    for(int j=0;j<n;j++)REQUIRE(!qwen_push(&s,prompt[j]));REQUIRE(snprintf(path,sizeof(path),"%s/driver.json",argv[6])<(int)sizeof(path));FILE *f=fopen(path,"wb");REQUIRE(f);
    fprintf(f,"{\"schema\":\"tcm-composition-driver-v1\",\"model_digest\":\"%s\",\"layer\":16,\"capacity\":512,\"gpu_program_workspace_bytes\":%zu,\"host_array_bytes\":%zu,\"source_own\":",m.digest,p.gpu_bytes,p.host_bytes);
    int last=-1;fprintf(f,"{\"ids\":[");REQUIRE(nsource>0&&nsource<=8);
    for(int j=0;j<nsource;j++){last=s.pending;REQUIRE(last==expected_source[j]);if(j)fputc(',',f);fprintf(f,"%d",last);REQUIRE(!qwen_push(&s,last));if(j<nsource-1)REQUIRE(last!=m.eos&&last!=m.eos_extra);}
    REQUIRE(last==m.eos||(m.eos_extra>0&&last==m.eos_extra));fprintf(f,"],\"stopped\":true,\"truncated\":false,\"all_delivered_consumed\":true}");
    char source_hash[65],after[65];prefix_digest(&s,s.position,source_hash);REQUIRE(!strcmp(source_hash,argv[9])&&s.position==n+nsource);
    fprintf(f,",\"source_replay_checked_before_candidates\":true,\"source_position\":%d,\"source_kv_sha256\":\"%s\",\"arms\":[",s.position,source_hash);
    const char *tasks[]={"recall","math","prose"};const int budgets[]={8,8,48};Capture capture={argv[6],NULL};unsigned printed=0;
    for(int t=0;t<3;t++){
        uint64_t parent;REQUIRE(!qwen_stack_push(&s,&parent));int skip=last==m.eos?1:0;REQUIRE(!skip||targets[t][0]==m.eos);
        for(int j=skip;j<nt[t]-1;j++)REQUIRE(!qwen_push(&s,targets[t][j]));
        for(int policy=null_panel?4:0;policy<(null_panel?5:4);policy++){if(printed++)fputc(',',f);run_arm(f,&s,&p,policy,tasks[t],targets[t][nt[t]-1],budgets[t],&capture);}
        REQUIRE(!qwen_stack_rollback(&s,parent));prefix_digest(&s,s.position,after);REQUIRE(!strcmp(source_hash,after));
    }
    bank_fingerprint(&p,bank_after);REQUIRE(!strcmp(bank_before,bank_after));
    fprintf(f,"],\"frozen_bank_before_sha256\":\"%s\",\"frozen_bank_after_sha256\":\"%s\",\"seconds\":%.9g}\n",bank_before,bank_after,clock_seconds()-started);REQUIRE(!fclose(f));if(null_panel)gpu_free(zero_write);qwen_subspace_free(&p);qwen_session_free(&s);qwen_model_free(&m);free(expected_source);free(prompt);for(int j=0;j<3;j++)free(targets[j]);return 0;
}
