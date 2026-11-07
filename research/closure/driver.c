#define main inherited_composition_main
#include "composition_driver.c"
#undef main
#include "runtime/qwen.c"

/* This assay reuses the authenticated importer, IO and native runtime. It does
 * not add an installation hook or change ordinary model execution. */
typedef struct {float *r,*n,*g,*u,*out,*cg,*cu;} Work;
static const float *source_receiver_host,*source_write_host;
static float expected_source_code[4];
static char native_first_logits[65],native_first_kv[65];
static void code_hex(const float c[4],char out[33]){const unsigned char *p=(const unsigned char*)c;const char *h="0123456789abcdef";for(int i=0;i<16;i++){out[2*i]=h[p[i]>>4];out[2*i+1]=h[p[i]&15];}out[32]=0;}
static void encode(QwenSubspace *p,Work *w,const float *r,float c[4]){
    int d=p->model->hidden;double v[3]={0};const double *q[]={p->q0,p->q1,p->q2};
    for(int j=0;j<d;j++)for(int k=0;k<3;k++)v[k]+=q[k][j]*(double)r[j];
    double c2=v[2]/p->r22,c1=(v[1]-p->r12*c2)/p->r11,c0=(v[0]-p->r01*c1-p->r02*c2)/p->r00;
    c[0]=(float)c0;c[1]=(float)c1;c[2]=(float)c2;
    gpu_write(w->r,r,(size_t)d*4);gpu_rms_scale(w->n,w->r,p->model->layer[16].post_norm,d,p->model->epsilon,p->rho);gpu_read(c+3,p->rho,4);
    for(int k=0;k<4;k++)REQUIRE(isfinite(c[k]));REQUIRE(c[3]>0);
    /* Check the copied encoder against the unchanged runtime's actual static
     * implementation, included in this translation unit without source edits. */
    gpu_copy(p->receiver,w->r,(size_t)d*4);QwenSubspaceStats actual={0};REQUIRE(!subspace_coordinates(p,&actual,3));
    REQUIRE(!memcmp(c,actual.executed,12)&&!memcmp(c+3,&actual.rho,4));
}
static void saved(const char *dir,const char *stem,const char *port_name,const float *data,int n){char file[256];REQUIRE(snprintf(file,sizeof(file),"%s-%s.f32",stem,port_name)<(int)sizeof(file));float *a=read_floats(data,(size_t)n);save(dir,file,a,(size_t)n*4);free(a);}
static void calculate(QwenSubspace *p,Work *w,const float *r,const float c[4],int mode,const char *dir,const char *stem){
    const QwenModel *m=p->model;const QwenLayer *l=&m->layer[16];int d=m->hidden,h=m->intermediate;
    gpu_write(w->r,r,(size_t)d*4);gpu_rms_scale(w->n,w->r,l->post_norm,d,m->epsilon,p->rho);
    float actual_rho;gpu_read(&actual_rho,p->rho,4);REQUIRE(!memcmp(&actual_rho,c+3,4));
    saved(dir,stem,"receiver",w->r,d);saved(dir,stem,"normalized",w->n,d);
    if(!mode){gpu_matvec(m->gpu,l->gate,d,h,w->n,w->g);gpu_matvec(m->gpu,l->up,d,h,w->n,w->u);}
    else{
        gpu_linear2(w->g,p->gate[0],p->gate[1],h,c[0],c[1],NULL);gpu_linear2(w->g,w->g,p->gate[2],h,1,c[2],p->rho);
        gpu_linear2(w->u,p->up[0],p->up[1],h,c[0],c[1],NULL);gpu_linear2(w->u,w->u,p->up[2],h,1,c[2],p->rho);
        if(mode==2){
            gpu_linear2(p->reconstruction,p->basis[0],p->basis[1],d,c[0],c[1],NULL);gpu_linear2(p->reconstruction,p->reconstruction,p->basis[2],d,1,c[2],NULL);
            gpu_plane(p->omitted,w->r,p->reconstruction,p->reconstruction,d,-1,0);gpu_multiply(w->n,p->omitted,l->post_norm,d);
            gpu_matvec(m->gpu,l->gate,d,h,w->n,w->cg);gpu_matvec(m->gpu,l->up,d,h,w->n,w->cu);
            gpu_linear2(w->cg,w->cg,NULL,h,1,0,p->rho);gpu_linear2(w->cu,w->cu,NULL,h,1,0,p->rho);gpu_add(w->g,w->cg,h);gpu_add(w->u,w->cu,h);
            saved(dir,stem,"omitted",p->omitted,d);saved(dir,stem,"correction-gate",w->cg,h);saved(dir,stem,"correction-up",w->cu,h);
        }
    }
    saved(dir,stem,"gate",w->g,h);saved(dir,stem,"up",w->u,h);gpu_swiglu(w->g,w->u,h);gpu_matvec(m->gpu,l->down,h,d,w->g,w->out);saved(dir,stem,"write",w->out,d);
}
static void attention_calibration(FILE *f,const QwenModel *m,const char *dir){
    const int d=128,capacity=3;float *kh=calloc((size_t)capacity*d,4),*vh=calloc((size_t)capacity*d,4),qh[128]={0};REQUIRE(kh&&vh);
    float *k=gpu_alloc((size_t)capacity*d*4),*v=gpu_alloc((size_t)capacity*d*4),*q=gpu_alloc(d*4),*o=gpu_alloc(d*4),*scores=gpu_alloc(capacity*4);
    vh[0]=1;vh[d]=-1;vh[2*d]=1;gpu_write(v,vh,(size_t)capacity*d*4);fprintf(f,"[ ");
    for(int query=0;query<2;query++)for(int length=2;length<=3;length++)for(int world=0;world<2;world++){
        kh[0]=world?-1:1;kh[d]=world?1:-1;gpu_write(k,kh,(size_t)capacity*d*4);qh[0]=query?sqrtf((float)d):0;gpu_write(q,qh,d*4);
        gpu_attention(m->gpu,o,q,k,v,scores,1,1,d,capacity,length);
        char stem[64];snprintf(stem,sizeof(stem),"attention-q%d-n%d-w%d",query,length,world);saved(dir,stem,"output",o,d);saved(dir,stem,"weights",scores,length);
        if(query||length!=2||world)fputc(',',f);float value;gpu_read(&value,o,4);fprintf(f,"{\"query\":%d,\"length\":%d,\"world\":%d,\"output_first\":%.9g,\"prefix\":\"%s\"}",query,length,world,value,stem);
    }fprintf(f,"]");free(kh);free(vh);gpu_free(k);gpu_free(v);gpu_free(q);gpu_free(o);gpu_free(scores);
}
static void closure_step(FILE *f,QwenSession *s,QwenSubspace *p,int policy,int step,int token,Capture *c,float *held){
    s->observer=trace;s->observer_context=c;s->observe_position=s->position;s->observe_layer=16;int position=s->position;QwenSubspaceStats stats={0};
    int recurring=policy==5||policy==6||policy==7;int active=recurring||(step==0&&policy!=0);
    REQUIRE(!qwen_begin_token(s,token));
    while(s->layer<s->model->layers){
        if(s->layer==16&&active){
            if(policy==5||policy==6)REQUIRE(!qwen_advance_layer_subspace(s,p,policy==5?QWEN_SUBSPACE_PROJECT:QWEN_SUBSPACE_CORRECT,&stats));
            else{QwenSubspace view=*p;view.baseline_write=held;REQUIRE(held);REQUIRE(!qwen_advance_layer_subspace(s,&view,QWEN_SUBSPACE_HELD,&stats));}
        }else REQUIRE(!qwen_advance_layer(s));
    }REQUIRE(!qwen_finish_token(s));selected_kv(s,position,c);
    if(active&&!step){float actual[4]={stats.executed[0],stats.executed[1],stats.executed[2],stats.rho};REQUIRE(!memcmp(actual,expected_source_code,16));}
    char hash[65],file[256];prefix_digest(s,s->position,hash);float *logits=read_floats(s->logits,(size_t)s->model->vocab);char lhash[65];digest(logits,(size_t)s->model->vocab*4,lhash);
    if(!step){REQUIRE(snprintf(file,sizeof(file),"%s-first.f32",c->name)<(int)sizeof(file));save(c->directory,file,logits,(size_t)s->model->vocab*4);}free(logits);
    if(!step&&policy==0){
        float *record=malloc((size_t)s->model->hidden*4);REQUIRE(record&&source_receiver_host&&source_write_host);
        snprintf(file,sizeof(file),"%s-raw_receiver.f32",c->name);blob(c->directory,file,record,(size_t)s->model->hidden*4);REQUIRE(!memcmp(record,source_receiver_host,(size_t)s->model->hidden*4));
        snprintf(file,sizeof(file),"%s-mlp_return.f32",c->name);blob(c->directory,file,record,(size_t)s->model->hidden*4);REQUIRE(!memcmp(record,source_write_host,(size_t)s->model->hidden*4));free(record);
        memcpy(native_first_logits,lhash,65);memcpy(native_first_kv,hash,65);
    }
    if(!step&&policy==1)REQUIRE(!strcmp(native_first_logits,lhash)&&!strcmp(native_first_kv,hash));
    if(step)fputc(',',f);fprintf(f,"{\"step\":%d,\"position\":%d,\"input_id\":%d,\"top\":%d,\"runner_up\":%d,\"margin\":%.9g,\"logits_sha256\":\"%s\",\"valid_kv_sha256\":\"%s\",\"substitution\":%s,\"gate_up_calls\":%u,\"down_calls\":%u",step,position,token,s->pending,s->top.second,s->top.first_value-s->top.second_value,lhash,hash,active?"true":"false",active?stats.work.gate_up_matvecs:2u,active?stats.work.down_matvecs:1u);
    if(active)fprintf(f,",\"rho\":%.9g,\"executed\":[%.9g,%.9g,%.9g]",stats.rho,stats.executed[0],stats.executed[1],stats.executed[2]);fputc('}',f);REQUIRE(!fflush(f));
}
static void closure_arm(FILE *f,QwenSession *s,QwenSubspace *p,int policy,const char *task,int final_id,Capture *capture,float *held){
    const char *labels[]={"native","sham","fiber","fiber-repair","compact-once","compact-recurring","corrected-recurring","zero"};char name[128],stem[160],parent_hash[65],after[65];uint64_t branch;
    int ids[8],count=0,stopped=0;REQUIRE(!qwen_stack_push(s,&branch));int parent_position=s->position;prefix_digest(s,parent_position,parent_hash);snprintf(name,sizeof(name),"%s-%s",task,labels[policy]);
    fprintf(f,"{\"name\":\"%s\",\"task\":\"%s\",\"policy\":%d,\"parent_position\":%d,\"parent_kv_sha256\":\"%s\",\"budget\":8,\"steps\":[",name,task,policy,parent_position,parent_hash);
    snprintf(stem,sizeof(stem),"%s-s000",name);capture->name=stem;closure_step(f,s,p,policy,0,final_id,capture,held);
    for(int j=0;j<8;j++){int token=s->pending;ids[count++]=token;stopped=token==s->model->eos||(s->model->eos_extra>0&&token==s->model->eos_extra);snprintf(stem,sizeof(stem),"%s-s%03d",name,j+1);closure_step(f,s,p,policy,j+1,token,capture,held);if(stopped)break;}
    s->observe_position=-1;fprintf(f,"],\"own\":{\"ids\":[");for(int j=0;j<count;j++){if(j)fputc(',',f);fprintf(f,"%d",ids[j]);}fprintf(f,"],\"stopped\":%s,\"truncated\":%s,\"all_delivered_consumed\":true},\"final_position\":%d,\"final_valid_kv_sha256\":\"%s\"",stopped?"true":"false",stopped?"false":"true",s->position,s->phase==1?(prefix_digest(s,s->position,after),after):"invalid");
    REQUIRE(!qwen_stack_rollback(s,branch));prefix_digest(s,parent_position,after);REQUIRE(!strcmp(after,parent_hash));fprintf(f,",\"prefix_intact\":true}");REQUIRE(!fflush(f));fprintf(stderr,"closure completed %s: %d emitted/consumed\n",name,count);
}
int main(int argc,char **argv){
    REQUIRE(argc==12);int n,nt[3],nsource;int *expected=tokens(argv[8],&nsource),*prompt=tokens(argv[2],&n),*targets[]={tokens(argv[3],&nt[0]),tokens(argv[4],&nt[1]),tokens(argv[5],&nt[2])};
    TcmModel bundle;QwenModel m;QwenSession s;QwenSubspace p;char error[256],file[512],path[4096];double started=clock_seconds();REQUIRE(!tcm_load(&bundle,argv[1],error,sizeof(error)));REQUIRE(!qwen_model_init(&m,&bundle,error,sizeof(error)));tcm_unload(&bundle);REQUIRE(!qwen_session_init(&s,&m,512));REQUIRE(!qwen_subspace_init(&p,&m));import_bank(&p,argv[7]);
    char bank_before[65],bank_after[65];bank_fingerprint(&p,bank_before);REQUIRE(!strcmp(bank_before,argv[10]));
    Work w;w.r=gpu_alloc(m.hidden*4);w.n=gpu_alloc(m.hidden*4);w.g=gpu_alloc(m.intermediate*4);w.u=gpu_alloc(m.intermediate*4);w.out=gpu_alloc(m.hidden*4);w.cg=gpu_alloc(m.intermediate*4);w.cu=gpu_alloc(m.intermediate*4);
    for(int j=0;j<n;j++)REQUIRE(!qwen_push(&s,prompt[j]));int last=-1;for(int j=0;j<nsource;j++){last=s.pending;REQUIRE(last==expected[j]);REQUIRE(!qwen_push(&s,last));}REQUIRE(last==m.eos||last==m.eos_extra);
    char source_hash[65],after[65];prefix_digest(&s,s.position,source_hash);REQUIRE(!strcmp(source_hash,argv[9])&&s.position==n+nsource);
    REQUIRE(snprintf(path,sizeof(path),"%s/driver.json",argv[6])<(int)sizeof(path));FILE *f=fopen(path,"wb");REQUIRE(f);fprintf(f,"{\"schema\":\"tcm-state-closure-driver-v1\",\"model_digest\":\"%s\",\"layer\":16,\"capacity\":512,\"source_replay_checked\":true,\"source_position\":%d,\"source_kv_sha256\":\"%s\",\"attention\":",m.digest,s.position,source_hash);attention_calibration(f,&m,argv[6]);fprintf(f,",\"cases\":[");
    const char *tasks[]={"recall","math","prose"};Capture capture={argv[6],NULL};
    for(int t=0;t<3;t++){
        const int nc=64;float *r0=malloc(m.hidden*4),*candidates=malloc((size_t)nc*m.hidden*4);REQUIRE(r0&&candidates);snprintf(file,sizeof(file),"%s-source.f32",tasks[t]);blob(argv[11],file,r0,m.hidden*4);snprintf(file,sizeof(file),"%s-candidates.f32",tasks[t]);blob(argv[11],file,candidates,(size_t)nc*m.hidden*4);
        float source_code[4];encode(&p,&w,r0,source_code);memcpy(expected_source_code,source_code,16);char chex[33];code_hex(source_code,chex);if(t)fputc(',',f);fprintf(f,"{\"task\":\"%s\",\"source_code_hex\":\"%s\",\"candidates\":[",tasks[t],chex);int chosen=-1;
        for(int j=0;j<nc;j++){float code[4];encode(&p,&w,candidates+(size_t)j*m.hidden,code);char encoded[33];code_hex(code,encoded);int equal=!memcmp(code,source_code,16),different=memcmp(r0,candidates+(size_t)j*m.hidden,m.hidden*4)!=0;if(chosen<0&&equal&&different)chosen=j;if(j)fputc(',',f);fprintf(f,"{\"index\":%d,\"code_hex\":\"%s\",\"rho\":%.9g,\"coefficients\":[%.9g,%.9g,%.9g],\"same_code\":%s,\"different_receiver\":%s}",j,encoded,code[3],code[0],code[1],code[2],equal?"true":"false",different?"true":"false");}
        fprintf(f,"],\"selected_index\":%d,\"selection_before_any_candidate_mlp\":true,\"actual_runtime_encoder_verified\":true,\"writes\":{",chosen);
        float *held[8]={0};for(int j=1;j<8;j++)if(j!=5&&j!=6)held[j]=gpu_alloc(m.hidden*4);
        char stem[160];snprintf(stem,sizeof(stem),"%s-source",tasks[t]);calculate(&p,&w,r0,source_code,0,argv[6],stem);gpu_copy(held[1],w.out,m.hidden*4);source_receiver_host=r0;source_write_host=read_floats(w.out,m.hidden);fprintf(f,"\"source\":{\"prefix\":\"%s\"}",stem);
        snprintf(stem,sizeof(stem),"%s-project",tasks[t]);calculate(&p,&w,r0,source_code,1,argv[6],stem);gpu_copy(held[4],w.out,m.hidden*4);fprintf(f,",\"project\":{\"prefix\":\"%s\"}",stem);
        if(chosen>=0){float *alternate=candidates+(size_t)chosen*m.hidden;float code[4];encode(&p,&w,alternate,code);REQUIRE(!memcmp(code,source_code,16));
            snprintf(stem,sizeof(stem),"%s-fiber",tasks[t]);calculate(&p,&w,alternate,code,0,argv[6],stem);gpu_copy(held[2],w.out,m.hidden*4);fprintf(f,",\"fiber\":{\"prefix\":\"%s\"}",stem);
            snprintf(stem,sizeof(stem),"%s-fiber-project",tasks[t]);calculate(&p,&w,alternate,code,1,argv[6],stem);REQUIRE(gpu_identical_finite(m.gpu,w.out,held[4],m.hidden));fprintf(f,",\"fiber_project\":{\"prefix\":\"%s\"}",stem);
            snprintf(stem,sizeof(stem),"%s-repair",tasks[t]);calculate(&p,&w,alternate,code,2,argv[6],stem);gpu_copy(held[3],w.out,m.hidden*4);fprintf(f,",\"repair\":{\"prefix\":\"%s\"}",stem);
        }gpu_zero(held[7],m.hidden*4);fprintf(f,"},\"arms\":[");
        uint64_t parent;REQUIRE(!qwen_stack_push(&s,&parent));int skip=last==m.eos?1:0;REQUIRE(!skip||targets[t][0]==m.eos);for(int j=skip;j<nt[t]-1;j++)REQUIRE(!qwen_push(&s,targets[t][j]));int printed=0;
        for(int policy=0;policy<8;policy++){if(chosen<0&&(policy==2||policy==3))continue;if(printed++)fputc(',',f);closure_arm(f,&s,&p,policy,tasks[t],targets[t][nt[t]-1],&capture,held[policy]);}
        REQUIRE(!qwen_stack_rollback(&s,parent));prefix_digest(&s,s.position,after);REQUIRE(!strcmp(source_hash,after));fprintf(f,"]}");
        for(int j=1;j<8;j++)if(held[j])gpu_free(held[j]);free((void*)source_write_host);source_receiver_host=NULL;source_write_host=NULL;free(r0);free(candidates);
    }
    bank_fingerprint(&p,bank_after);REQUIRE(!strcmp(bank_before,bank_after));fprintf(f,"],\"frozen_bank_before_sha256\":\"%s\",\"frozen_bank_after_sha256\":\"%s\",\"seconds\":%.9g}\n",bank_before,bank_after,clock_seconds()-started);REQUIRE(!fclose(f));gpu_free(w.r);gpu_free(w.n);gpu_free(w.g);gpu_free(w.u);gpu_free(w.out);gpu_free(w.cg);gpu_free(w.cu);qwen_subspace_free(&p);qwen_session_free(&s);qwen_model_free(&m);free(expected);free(prompt);for(int t=0;t<3;t++)free(targets[t]);return 0;
}
