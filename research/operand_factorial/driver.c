/* The immutable predecessor supplies basis execution, file I/O and token replay.
 * Its main is renamed; no serving source is modified. */
#define main predecessor_main
#include "predecessor.c"
#undef main

typedef struct { float *raw,*gate,*up,*write; unsigned seen; } Live;
static void live_capture(void *ctx,int position,int layer,QwenPort port_id,const float *p,size_t n){
    (void)position;if(layer!=LAYER)return;Live *v=ctx;float *out=NULL;unsigned bit=0;
    if(port_id==QWEN_RAW_RECEIVER){out=v->raw;bit=1;}
    if(port_id==QWEN_MLP_GATE){out=v->gate;bit=2;}
    if(port_id==QWEN_MLP_UP){out=v->up;bit=4;}
    if(port_id==QWEN_MLP_RETURN){out=v->write;bit=8;}
    if(out){gpu_copy(out,p,n*4);v->seen|=bit;}
}

/* Paid native evaluation supplies both exact diagnostic operands. The branch
 * consumes its declared write by replacing the residual before layer17. */
static void intervention(QwenSession *s,Basis *b,Live *v,int token,int arm,const char *stem){
    const QwenModel *m=s->model;int d=m->hidden,h=m->intermediate;
    REQUIRE(!qwen_begin_token(s,token));while(s->layer<LAYER)REQUIRE(!qwen_advance_layer(s));
    v->seen=0;s->observer=live_capture;s->observer_context=v;s->observe_position=s->position;s->observe_layer=LAYER;
    REQUIRE(!qwen_advance_layer(s));s->observer=NULL;s->observer_context=NULL;REQUIRE(v->seen==15);
    gpu_rms_scale(s->norm,v->raw,m->layer[LAYER].post_norm,d,m->epsilon,b->rho);
    gpu_matvec(m->gpu,b->qt,d,RANK,v->raw,b->z);
    gpu_matvec(m->gpu,b->gq,RANK,h,b->z,b->gate);gpu_matvec(m->gpu,b->uq,RANK,h,b->z,b->up);
    gpu_linear2(b->gate,b->gate,NULL,h,1,0,b->rho);gpu_linear2(b->up,b->up,NULL,h,1,0,b->rho);
    char name[256];
#define KEEP(suffix,p,n) do{REQUIRE(snprintf(name,sizeof(name),"%s-%s.f32",stem,suffix)<(int)sizeof(name));save_device(output_directory,name,p,n);}while(0)
    KEEP("receiver",v->raw,d);KEEP("native_gate",v->gate,h);KEEP("native_up",v->up,h);
    KEEP("candidate_gate",b->gate,h);KEEP("candidate_up",b->up,h);KEEP("native_write",v->write,d);
    if(arm==5){gpu_copy(s->projected,v->write,(size_t)d*4);} /* ordinary native */
    else if(arm==4){gpu_zero(s->projected,(size_t)d*4);}
    else{
        if(arm==1||arm==3)gpu_copy(b->gate,v->gate,(size_t)h*4);
        if(arm==2||arm==3)gpu_copy(b->up,v->up,(size_t)h*4);
        gpu_swiglu(b->gate,b->up,h);gpu_matvec(m->gpu,m->layer[LAYER].down,h,d,b->gate,s->projected);
    }
    KEEP("write",s->projected,d);
    gpu_copy(s->x,v->raw,(size_t)d*4);gpu_add(s->x,s->projected,d);
    while(s->layer<m->layers)REQUIRE(!qwen_advance_layer(s));REQUIRE(!qwen_finish_token(s));
#undef KEEP
}

int main(int argc,char **argv){
    REQUIRE(argc==6);const char *model_path=argv[1],*results=argv[2],*compiled=argv[3],*clock_list=argv[4],*expected_digest=argv[5];
    output_directory=results;mkdir_one(results);char error[256],name[256],path[4096];
    TcmModel bundle;QwenModel model;QwenSession session;REQUIRE(!tcm_load(&bundle,model_path,error,sizeof(error)));
    REQUIRE(!strcmp(bundle.digest,expected_digest));REQUIRE(!qwen_model_init(&model,&bundle,error,sizeof(error)));tcm_unload(&bundle);
    REQUIRE(!qwen_session_init(&session,&model,512));wrapped_model=&model;wrapped_stage=0;
    Basis source={0},random={0};basis_load(&source,&model,compiled,"source");basis_load(&random,&model,compiled,"random");
    Live live={gpu_alloc((size_t)model.hidden*4),gpu_alloc((size_t)model.intermediate*4),gpu_alloc((size_t)model.intermediate*4),gpu_alloc((size_t)model.hidden*4),0};
    save_device(results,"down.f32",model.layer[LAYER].down,model.hidden*model.intermediate);
    REQUIRE(snprintf(path,sizeof(path),"%s/driver.json",results)<(int)sizeof(path));FILE *out=fopen(path,"w");REQUIRE(out);
    FILE *list=fopen(clock_list,"r");REQUIRE(list);int count=0,policy;char idfile[4096],key[128];double started=clock_seconds();
    fprintf(out,"{\"schema\":\"tcm-operand-factorial-driver-v1\",\"model_digest\":\"%s\",\"records\":[",model.digest);
    const char *names[]={"00","10","01","11","zero","native","repeat00"};
    while(fscanf(list,"%127s %d %4095s",key,&policy,idfile)==3){
        REQUIRE(policy==1||policy==2);Basis *b=policy==1?&source:&random;int n;int *ids=tokens(idfile,&n);REQUIRE(n>1);
        for(int parent=0;parent<2;parent++){
            REQUIRE(!qwen_reset(&session));
            for(int i=0;i<n-1;i++){
                REQUIRE(!qwen_begin_token(&session,ids[i]));
                while(session.layer<model.layers){
                    if(parent==0&&session.layer==LAYER)REQUIRE(!advance_basis_layer(&session,b,0,NULL));
                    else REQUIRE(!qwen_advance_layer(&session));
                }REQUIRE(!qwen_finish_token(&session));
            }
            QwenCheckpoint cut;REQUIRE(!qwen_checkpoint(&session,&cut));char parent_hash[65];prefix_digest(&session,session.position,parent_hash);
            for(int arm=0;arm<7;arm++){
                REQUIRE(!qwen_restore(&session,&cut));char restored_hash[65];prefix_digest(&session,session.position,restored_hash);REQUIRE(!strcmp(parent_hash,restored_hash));
                REQUIRE(snprintf(name,sizeof(name),"%s-%s-%s",key,parent==0?"carried":"native",names[arm])<(int)sizeof(name));
                intervention(&session,b,&live,ids[n-1],arm,name);ReadoutInfo readout={0};capture_readout(&session,results,name,&readout);
                if(count++)fputc(',',out);
                fprintf(out,"{\"clock\":\"%s\",\"stem\":\"%s\",\"parent\":\"%s\",\"arm\":\"%s\",\"position\":%d,\"prefix_kv_sha256\":\"%s\",\"logits\":",key,name,parent==0?"carried":"native",names[arm],n,parent_hash);readout_json(out,&readout);
                fprintf(out,",\"continuation\":");generated(out,&session,4);fputc('}',out);REQUIRE(!fflush(out));
                REQUIRE(!qwen_restore(&session,&cut));prefix_digest(&session,session.position,restored_hash);REQUIRE(!strcmp(parent_hash,restored_hash));
            }
        }free(ids);
    }REQUIRE(feof(list)&&!fclose(list));fprintf(out,"],\"record_count\":%d,\"seconds\":%.9g}\n",count,clock_seconds()-started);REQUIRE(!fclose(out));
    gpu_free(live.raw);gpu_free(live.gate);gpu_free(live.up);gpu_free(live.write);basis_free(&source);basis_free(&random);qwen_session_free(&session);qwen_model_free(&model);return 0;
}
