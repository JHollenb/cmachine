#define _POSIX_C_SOURCE 200809L
#define REQUIRE(x) do{if(!(x)){fprintf(stderr,"operand-basis %s:%d: %s\n",__FILE__,__LINE__,#x);exit(2);}}while(0)
#include "assay_io.h"
#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/wait.h>

/*
 * This driver deliberately owns only the new rank-64 consumer.  The runtime is
 * imported by the worker into runtime/ and remains otherwise unchanged.
 *
 * CLI (the worker supplies the paths, so no model or tokenizer discovery is
 * hidden here):
 *
 *   operand-chat MODEL RESULTS B3_BANK COMPILED_BANK PYTHON COMPILER
 *       CASE0_PROMPT CASE0_FOLLOW CASE1_PROMPT CASE1_FOLLOW
 *       CASE2_PROMPT CASE2_FOLLOW [smoke]
 *
 * The compiler receives row-major G/U (H x D) and gamma (D) in
 * COMPILED_BANK/compile-input and writes, for prefixes source and random,
 * little-endian FP32 QT (R x D), GQ (H x R), UQ (H x R), plus a manifest.
 * QT is Q^T.  The C consumer calls gpu_matvec with (in=D,out=R) for QT and
 * (in=R,out=H) for GQ/UQ, which is the same row-major convention as Qwen
 * weights.  The model stays resident while the CPU compiler runs.
 */

#define LAYER 16
#define RANK 64
#define MAX_CASES 3
#define MAX_BUDGET 16

typedef enum { ARM_NATIVE=0, ARM_SOURCE=1, ARM_RANDOM=2, ARM_B3=3, ARM_ZERO=4 } Arm;
static const char *arm_names[] = {"native","source64","random64","b3","zero"};

typedef struct {
    const QwenModel *model;
    int rank;
    float *qt,*gq,*uq,*z,*gate,*up,*rho;
    char qt_sha[65],gq_sha[65],uq_sha[65];
    unsigned qt_calls,gq_calls,uq_calls,down_calls;
} Basis;

typedef struct {
    unsigned basis_q_calls,basis_g_calls,basis_u_calls;
    unsigned selected_gate_up_calls,selected_down_calls;
    float rho;
} StepWork;

/* The shared assay header already owns Capture (directory + name).  This
 * extension is only observer policy: raw receiver/return are retained at every
 * transition, while the expensive full gate/up vectors are sampled at a few
 * clocks.  It never changes the numerical path. */
typedef struct {
    Capture base;
    int save_gate_up;
    int minimal;
} OperandCapture;

typedef struct {
    int top,runner;
    float top_value,margin,red_minus_blue;
    char sha[65],file[256];
} ReadoutInfo;

typedef struct {
    int ids[64],count,stopped,truncated;
} Answer;

static const char *output_directory;
static unsigned primary_steps,reference_steps,diagnostic_steps;
static const QwenModel *wrapped_model;
static unsigned wrapped_gate_up_calls,wrapped_down_calls;
static int wrapped_stage;

/* Link-time wrapping gives the primary prohibition an executable witness:
 * source/random/B3/zero may use their basis or zero consumer at layer 16, but
 * they must not call the original selected dense G/U projections.  The normal
 * selected down projection remains part of the candidate consumer. */
void __real_gpu_matvec(TcmGpu *,const float *,int,int,const float *,float *);
void __wrap_gpu_matvec(TcmGpu *g,const float *w,int in,int out,const float *x,float *y){
    if(wrapped_model&&wrapped_stage){
        const QwenLayer *l=&wrapped_model->layer[LAYER];
        if(w==l->gate||w==l->up)wrapped_gate_up_calls++;
        if(w==l->down)wrapped_down_calls++;
    }
    __real_gpu_matvec(g,w,in,out,x,y);
}

static void mkdir_one(const char *path){if(mkdir(path,0755)&&errno!=EEXIST){perror(path);exit(2);}}

static void blob(const char *path,void *data,size_t bytes){
    FILE *f=fopen(path,"rb");REQUIRE(f);REQUIRE(fread(data,1,bytes,f)==bytes&&fgetc(f)==EOF&&!ferror(f)&&!fclose(f));
}

static void save_device(const char *directory,const char *name,const float *data,int n){
    float *host=read_floats(data,(size_t)n);save(directory,name,host,(size_t)n*4);free(host);
}

static void file_sha(const char *path,char sha[65],size_t *bytes_out){
    FILE *f=fopen(path,"rb");REQUIRE(f);REQUIRE(!fseek(f,0,SEEK_END));long n=ftell(f);REQUIRE(n>=0&&!fseek(f,0,SEEK_SET));unsigned char *p=malloc((size_t)n);REQUIRE(p||(n==0));REQUIRE(fread(p,1,(size_t)n,f)==(size_t)n&&!fclose(f));digest(p,(size_t)n,sha);if(bytes_out)*bytes_out=(size_t)n;free(p);
}

static void save_manifest_blob(const char *directory,const char *name,const float *data,size_t count,char sha[65]){
    float *host=malloc(count*4);REQUIRE(host);gpu_read(host,data,count*4);for(size_t i=0;i<count;i++)REQUIRE(isfinite(host[i]));
    digest(host,count*4,sha);save(directory,name,host,count*4);free(host);
}

static void export_compiler_inputs(const QwenModel *m,const char *directory,const char *digest_model){
    const QwenLayer *l=&m->layer[LAYER];char path[4096],gsha[65],usha[65],ash[65];size_t d=(size_t)m->hidden,h=(size_t)m->intermediate;
    mkdir_one(directory);
    REQUIRE(snprintf(path,sizeof(path),"%s/G.f32",directory)<(int)sizeof(path));save_manifest_blob(directory,"G.f32",l->gate,h*d,gsha);
    REQUIRE(snprintf(path,sizeof(path),"%s/U.f32",directory)<(int)sizeof(path));save_manifest_blob(directory,"U.f32",l->up,h*d,usha);
    REQUIRE(snprintf(path,sizeof(path),"%s/gamma.f32",directory)<(int)sizeof(path));save_manifest_blob(directory,"gamma.f32",l->post_norm,d,ash);
    REQUIRE(snprintf(path,sizeof(path),"%s/input-manifest.json",directory)<(int)sizeof(path));FILE *f=fopen(path,"w");REQUIRE(f);
    fprintf(f,"{\"schema\":\"tcm-operand-basis-input-v1\",\"model_digest\":\"%s\",\"layer\":%d,\"hidden\":%d,\"intermediate\":%d,\"rank\":%d,\"G\":{\"file\":\"G.f32\",\"rows\":%d,\"cols\":%d,\"sha256\":\"%s\"},\"U\":{\"file\":\"U.f32\",\"rows\":%d,\"cols\":%d,\"sha256\":\"%s\"},\"gamma\":{\"file\":\"gamma.f32\",\"count\":%d,\"sha256\":\"%s\"}}\n",digest_model,LAYER,m->hidden,m->intermediate,RANK,m->intermediate,m->hidden,gsha,m->intermediate,m->hidden,usha,m->hidden,ash);
    REQUIRE(!fclose(f));
}

static void run_compiler(const char *python,const char *compiler,const char *input,const char *output){
    char command[8192];
    REQUIRE(snprintf(command,sizeof(command),"%s %s --input-dir %s --output-dir %s --rank %d --seed 20261106",python,compiler,input,output,RANK)<(int)sizeof(command));
    fprintf(stderr,"running operand compiler: %s\n",command);int status=system(command);REQUIRE(status!=-1&&WIFEXITED(status)&&WEXITSTATUS(status)==0);
}

static float *load_blob_gpu(const char *directory,const char *name,size_t count,char sha[65]){
    char path[4096];REQUIRE(snprintf(path,sizeof(path),"%s/%s",directory,name)<(int)sizeof(path));float *host=malloc(count*4);REQUIRE(host);blob(path,host,count*4);for(size_t i=0;i<count;i++)REQUIRE(isfinite(host[i]));digest(host,count*4,sha);float *device=gpu_alloc(count*4);REQUIRE(device);gpu_write(device,host,count*4);free(host);return device;
}

static void basis_load(Basis *b,const QwenModel *m,const char *directory,const char *prefix){
    char name[256];size_t d=(size_t)m->hidden,h=(size_t)m->intermediate;b->model=m;b->rank=RANK;
    REQUIRE(snprintf(name,sizeof(name),"%s-QT.f32",prefix)<(int)sizeof(name));b->qt=load_blob_gpu(directory,name,(size_t)RANK*d,b->qt_sha);
    REQUIRE(snprintf(name,sizeof(name),"%s-GQ.f32",prefix)<(int)sizeof(name));b->gq=load_blob_gpu(directory,name,h*RANK,b->gq_sha);
    REQUIRE(snprintf(name,sizeof(name),"%s-UQ.f32",prefix)<(int)sizeof(name));b->uq=load_blob_gpu(directory,name,h*RANK,b->uq_sha);
    b->z=gpu_alloc((size_t)RANK*4);b->gate=gpu_alloc(h*4);b->up=gpu_alloc(h*4);b->rho=gpu_alloc(4);b->qt_calls=b->gq_calls=b->uq_calls=b->down_calls=0;
}

static void basis_free(Basis *b){if(!b)return;gpu_free(b->qt);gpu_free(b->gq);gpu_free(b->uq);gpu_free(b->z);gpu_free(b->gate);gpu_free(b->up);gpu_free(b->rho);memset(b,0,sizeof(*b));}

static void import_b3(QwenSubspace *p,const char *directory){
    int d=p->model->hidden,h=p->model->intermediate;size_t n=(size_t)3*d+4*h,ne=(size_t)d+2*h;float *base=malloc(n*4),*extra=malloc(ne*4);double *qr=malloc(((size_t)2*d+4)*8),*qe=malloc(((size_t)d+3)*8);char path[4096];REQUIRE(base&&extra&&qr&&qe);
    const char *files[]={"reuse-program.f32","enrich-program.f32","reuse-qr.f64","enrich-qr.f64"};void *ptrs[]={base,extra,qr,qe};size_t sizes[]={n*4,ne*4,((size_t)2*d+4)*8,((size_t)d+3)*8};
    for(int i=0;i<4;i++){REQUIRE(snprintf(path,sizeof(path),"%s/%s",directory,files[i])<(int)sizeof(path));blob(path,ptrs[i],sizes[i]);}
    for(size_t j=0;j<n;j++)REQUIRE(isfinite(base[j]));for(size_t j=0;j<ne;j++)REQUIRE(isfinite(extra[j]));
    for(int k=0;k<2;k++){gpu_write(p->basis[k],base+(size_t)k*d,(size_t)d*4);gpu_write(p->gate[k],base+2*d+(size_t)k*h,(size_t)h*4);gpu_write(p->up[k],base+2*d+2*h+(size_t)k*h,(size_t)h*4);}gpu_write(p->baseline_write,base+n-d,(size_t)d*4);
    gpu_write(p->basis[2],extra,(size_t)d*4);gpu_write(p->gate[2],extra+d,(size_t)h*4);gpu_write(p->up[2],extra+d+h,(size_t)h*4);
    memcpy(p->q0,qr,(size_t)d*8);memcpy(p->q1,qr+d,(size_t)d*8);memcpy(p->q2,qe,(size_t)d*8);p->r00=qr[2*d];p->r01=qr[2*d+1];p->r11=qr[2*d+2];p->condition=qr[2*d+3];p->r02=qe[d];p->r12=qe[d+1];p->r22=qe[d+2];p->layer=LAYER;p->rank=3;p->ready=1;
    REQUIRE(p->r00>0&&p->r11>0&&p->r22>0);free(base);free(extra);free(qr);free(qe);
}

static void trace_port(void *context,int position,int layer,QwenPort id,const float *data,size_t count){
    (void)position;OperandCapture *oc=context;Capture *c=&oc->base;if(layer!=LAYER)return;
    if(id!=QWEN_RAW_RECEIVER&&id!=QWEN_MLP_INPUT&&id!=QWEN_MLP_GATE&&id!=QWEN_MLP_UP&&id!=QWEN_MLP_RETURN&&id!=QWEN_OUTPUT)return;
    if(oc->minimal&&id!=QWEN_RAW_RECEIVER&&id!=QWEN_MLP_INPUT)return;
    if((id==QWEN_MLP_GATE||id==QWEN_MLP_UP)&&!oc->save_gate_up)return;
    char name[256];REQUIRE(snprintf(name,sizeof(name),"%s-%s.f32",c->name,qwen_port_name(id))<(int)sizeof(name));save_device(c->directory,name,data,(int)count);
}

static void emit_port(QwenSession *s,QwenPort id,const float *data,size_t count){if(s->observer&&s->position==s->observe_position&&s->layer==s->observe_layer)s->observer(s->observer_context,s->position,s->layer,id,data,count);}

/* Same layer order and same GPU kernels as qwen.c, with only G/U replaced. */
static int advance_basis_layer(QwenSession *s,Basis *b,int zero,StepWork *work){
    if(!s||!b||s->phase!=2||s->layer!=LAYER)return -1;const QwenModel *m=s->model;const QwenLayer *l=&m->layer[LAYER];int d=m->hidden,k=m->kvheads*m->head,h=m->intermediate;size_t base=((size_t)LAYER*m->kvheads)*s->capacity*m->head;
    emit_port(s,QWEN_INPUT,s->x,(size_t)d);gpu_rms(s->norm,s->x,l->input_norm,d,m->epsilon);gpu_matvec(m->gpu,l->q,d,d,s->norm,s->q);gpu_bias(s->q,l->qb,d);gpu_matvec(m->gpu,l->k,d,k,s->norm,s->k);gpu_bias(s->k,l->kb,k);gpu_matvec(m->gpu,l->v,d,k,s->norm,s->v);gpu_bias(s->v,l->vb,k);gpu_rope(s->q,m->qheads,m->head,s->position,m->theta);gpu_rope(s->k,m->kvheads,m->head,s->position,m->theta);emit_port(s,QWEN_QUERY,s->q,(size_t)d);emit_port(s,QWEN_KEY,s->k,(size_t)k);emit_port(s,QWEN_VALUE,s->v,(size_t)k);
    gpu_cache(s->keys+base,s->k,m->kvheads,m->head,s->capacity,s->position);gpu_cache(s->values+base,s->v,m->kvheads,m->head,s->capacity,s->position);gpu_attention(m->gpu,s->attention,s->q,s->keys+base,s->values+base,s->scores,m->qheads,m->kvheads,m->head,s->capacity,s->position+1);emit_port(s,QWEN_ATTENTION,s->attention,(size_t)d);gpu_matvec(m->gpu,l->o,d,d,s->attention,s->projected);emit_port(s,QWEN_ATTENTION_WRITE,s->projected,(size_t)d);gpu_add(s->x,s->projected,d);emit_port(s,QWEN_RAW_RECEIVER,s->x,(size_t)d);
    gpu_rms_scale(s->norm,s->x,l->post_norm,d,m->epsilon,b->rho);emit_port(s,QWEN_MLP_INPUT,s->norm,(size_t)d);if(work)memset(work,0,sizeof(*work));
    if(zero){gpu_zero(s->projected,(size_t)d*4);if(work){work->selected_gate_up_calls=0;work->selected_down_calls=0;}}
    else {gpu_matvec(m->gpu,b->qt,d,RANK,s->x,b->z);gpu_matvec(m->gpu,b->gq,RANK,h,b->z,b->gate);gpu_matvec(m->gpu,b->uq,RANK,h,b->z,b->up);b->qt_calls++;b->gq_calls++;b->uq_calls++;if(work){work->basis_q_calls=1;work->basis_g_calls=1;work->basis_u_calls=1;work->selected_gate_up_calls=0;work->selected_down_calls=1;}gpu_linear2(b->gate,b->gate,NULL,h,1,0,b->rho);gpu_linear2(b->up,b->up,NULL,h,1,0,b->rho);emit_port(s,QWEN_MLP_GATE,b->gate,(size_t)h);emit_port(s,QWEN_MLP_UP,b->up,(size_t)h);gpu_swiglu(b->gate,b->up,h);gpu_matvec(m->gpu,l->down,h,d,b->gate,s->projected);b->down_calls++;}
    emit_port(s,QWEN_MLP_RETURN,s->projected,(size_t)d);gpu_add(s->x,s->projected,d);emit_port(s,QWEN_OUTPUT,s->x,(size_t)d);s->layer++;return 0;
}

/* Begin/advance/finish is kept separate so the caller controls the exact token ID. */
static StepWork step(QwenSession *s,int token,Arm arm,Basis *source,Basis *random,QwenSubspace *b3,OperandCapture *capture){
    StepWork work={0};int position=s->position;wrapped_gate_up_calls=wrapped_down_calls=0;wrapped_stage=1;REQUIRE(!qwen_begin_token(s,token));s->observer=trace_port;s->observer_context=capture;s->observe_position=position;s->observe_layer=LAYER;
    while(s->layer<s->model->layers){if(s->layer!=LAYER||arm==ARM_NATIVE)REQUIRE(!qwen_advance_layer(s));else if(arm==ARM_B3){QwenSubspaceStats stats={0};REQUIRE(!qwen_advance_layer_subspace(s,b3,QWEN_SUBSPACE_PROJECT,&stats));work.basis_q_calls=0;work.basis_g_calls=0;work.basis_u_calls=0;work.selected_gate_up_calls=stats.work.gate_up_matvecs;work.selected_down_calls=stats.work.down_matvecs;}else if(arm==ARM_ZERO)REQUIRE(!advance_basis_layer(s,source,1,&work));else REQUIRE(!advance_basis_layer(s,arm==ARM_SOURCE?source:random,0,&work));}
    s->observer=NULL;s->observer_context=NULL;s->observe_position=-1;REQUIRE(!qwen_finish_token(s));
    /* These counts come from the link-time wrapper, rather than hand-written
     * claims in the candidate branch.  The expected selected dense calls are
     * native=(2 G/U, 1 down), source/random/B3=(0 G/U, 1 down), zero=(0,0). */
    work.selected_gate_up_calls=wrapped_gate_up_calls;work.selected_down_calls=wrapped_down_calls;wrapped_stage=0;
    if(arm==ARM_NATIVE)REQUIRE(work.selected_gate_up_calls==2&&work.selected_down_calls==1);
    else if(arm==ARM_ZERO)REQUIRE(work.selected_gate_up_calls==0&&work.selected_down_calls==0);
    else REQUIRE(work.selected_gate_up_calls==0&&work.selected_down_calls==1);
    return work;
}

static void prefix_hash(const QwenSession *s,char hash[65]){prefix_digest(s,s->position,hash);}

static void capture_readout(QwenSession *s,const char *directory,const char *stem,ReadoutInfo *out){
    REQUIRE(s->phase==1);int vocab=s->model->vocab;float *h=read_floats(s->logits,(size_t)vocab);REQUIRE(snprintf(out->file,sizeof(out->file),"%s-logits.f32",stem)<(int)sizeof(out->file));
    digest(h,(size_t)vocab*4,out->sha);save(directory,out->file,h,(size_t)vocab*4);out->top=s->top.first;out->runner=s->top.second;out->top_value=s->top.first_value;out->margin=s->top.first_value-s->top.second_value;out->red_minus_blue=h[1151]-h[12203];free(h);
}

static void readout_json(FILE *f,const ReadoutInfo *r){fprintf(f,"{\"top\":%d,\"runner_up\":%d,\"top_value\":%.9g,\"top_margin\":%.9g,\"red_minus_blue\":%.9g,\"sha256\":\"%s\",\"file\":\"%s\"}",r->top,r->runner,r->top_value,r->margin,r->red_minus_blue,r->sha,r->file);}

static void step_json(FILE *f,int index,int token,int position,const StepWork *w,const char *stem,const char *native_write_file){
    fprintf(f,"{\"step\":%d,\"position\":%d,\"input_id\":%d,\"stem\":\"%s\",\"basis_q_calls\":%u,\"basis_g_calls\":%u,\"basis_u_calls\":%u,\"selected_gate_up_calls\":%u,\"selected_down_calls\":%u",index,position,token,stem,w->basis_q_calls,w->basis_g_calls,w->basis_u_calls,w->selected_gate_up_calls,w->selected_down_calls);if(native_write_file)fprintf(f,",\"native_write_file\":\"%s\"",native_write_file);fputc('}',f);
}

static void append_answer(Answer *a,int token,int eos){REQUIRE(a->count<(int)(sizeof(a->ids)/sizeof(a->ids[0])));a->ids[a->count++]=token;if(eos)a->stopped=1;}

typedef struct {
    int index,token,position;
    StepWork work;
    char stem[256];
    char native_write_file[256];
} Event;

#define MAX_TRAJECTORY 2048
typedef struct {
    char primary_name[128];
    int ids[MAX_TRAJECTORY],count;
    int prompt_count,answer0_count,followup_count,answer1_count;
    int first_event_count,answer0_end_event_count,followup_end_event_count;
} Trajectory;

static void remember_event(Event *events,int *count,int index,int token,int position,const StepWork *work,Arm arm,const char *stem){
    REQUIRE(*count<2048);Event *e=&events[(*count)++];e->index=index;e->token=token;e->position=position;e->work=*work;REQUIRE(snprintf(e->stem,sizeof(e->stem),"%s",stem)<(int)sizeof(e->stem));e->native_write_file[0]=0;
    if(arm!=ARM_NATIVE)REQUIRE(snprintf(e->native_write_file,sizeof(e->native_write_file),"%s-native_write.f32",stem)<(int)sizeof(e->native_write_file));
}

static void answer_json(FILE *f,const Answer *a){fprintf(f,"{\"ids\":[");for(int j=0;j<a->count;j++){if(j)fputc(',',f);fprintf(f,"%d",a->ids[j]);}fprintf(f,"],\"stopped\":%s,\"truncated\":%s}",a->stopped?"true":"false",a->stopped?"false":"true");}

static void run_arm(FILE *f,QwenSession *s,Basis *source,Basis *random,QwenSubspace *b3,Arm arm,int case_id,const int *prompt,int np,const int *follow,int nf,int budget,Trajectory *trajectory){
    char name[128],stem[256],first_hash[65],final_hash[65];
    REQUIRE(snprintf(name,sizeof(name),"case%d-%s",case_id,arm_names[arm])<(int)sizeof(name));
    if(trajectory){memset(trajectory,0,sizeof(*trajectory));strcpy(trajectory->primary_name,name);}
    qwen_reset(s);Event events[2048];int event_count=0,step_index=0;OperandCapture oc={{output_directory,NULL},0,0};
    for(int j=0;j<np;j++){
        REQUIRE(snprintf(stem,sizeof(stem),"%s-s%03d",name,step_index)<(int)sizeof(stem));oc.base.name=stem;oc.save_gate_up=(j==np-1);
        int pos=s->position;StepWork w=step(s,prompt[j],arm,source,random,b3,&oc);remember_event(events,&event_count,step_index,prompt[j],pos,&w,arm,stem);if(trajectory){REQUIRE(trajectory->count<MAX_TRAJECTORY);trajectory->ids[trajectory->count++]=prompt[j];}step_index++;
    }
    if(trajectory){trajectory->prompt_count=np;trajectory->first_event_count=event_count;}
    prefix_hash(s,first_hash);ReadoutInfo first={0},after_answer0={0},after_followup={0},final={0};char answer0_hash[65],followup_hash[65];int followup_position=-1;REQUIRE(np>0);capture_readout(s,output_directory,stem,&first);
    Answer answers[2]={{0},{0}};int follow_skip=0,follow_steps=0;
    for(int turn=0;turn<2;turn++){
        for(int j=0;j<budget;j++){
            int token=s->pending;int eos=token==s->model->eos||(s->model->eos_extra>0&&token==s->model->eos_extra);append_answer(&answers[turn],token,eos);
            REQUIRE(snprintf(stem,sizeof(stem),"%s-s%03d",name,step_index)<(int)sizeof(stem));oc.base.name=stem;oc.save_gate_up=(j==budget-1||eos);
            int pos=s->position;StepWork w=step(s,token,arm,source,random,b3,&oc);remember_event(events,&event_count,step_index,token,pos,&w,arm,stem);if(trajectory){REQUIRE(trajectory->count<MAX_TRAJECTORY);trajectory->ids[trajectory->count++]=token;}step_index++;if(eos)break;
        }
        if(trajectory&&!turn){trajectory->answer0_count=answers[0].count;trajectory->answer0_end_event_count=event_count;}
        if(!turn){
            prefix_hash(s,answer0_hash);REQUIRE(snprintf(stem,sizeof(stem),"%s-after-answer0",name)<(int)sizeof(stem));capture_readout(s,output_directory,stem,&after_answer0);
            follow_skip=answers[0].count&&nf>0&&answers[0].ids[answers[0].count-1]==follow[0];
            for(int j=follow_skip;j<nf;j++){
                REQUIRE(snprintf(stem,sizeof(stem),"%s-s%03d",name,step_index)<(int)sizeof(stem));oc.base.name=stem;oc.save_gate_up=(j==nf-1);
                int pos=s->position;StepWork w=step(s,follow[j],arm,source,random,b3,&oc);remember_event(events,&event_count,step_index,follow[j],pos,&w,arm,stem);if(trajectory){REQUIRE(trajectory->count<MAX_TRAJECTORY);trajectory->ids[trajectory->count++]=follow[j];trajectory->followup_count++;}step_index++;follow_steps++;
            }
            prefix_hash(s,followup_hash);followup_position=s->position;REQUIRE(snprintf(stem,sizeof(stem),"%s-after-followup",name)<(int)sizeof(stem));capture_readout(s,output_directory,stem,&after_followup);if(trajectory)trajectory->followup_end_event_count=event_count;
        }
    }
    prefix_hash(s,final_hash);capture_readout(s,output_directory,name,&final);if(trajectory)trajectory->answer1_count=answers[1].count;
    fprintf(f,"{\"name\":\"%s\",\"case\":%d,\"arm\":\"%s\",\"autonomous\":true,\"steps\":[",name,case_id,arm_names[arm]);
    for(int i=0;i<event_count;i++){if(i)fputc(',',f);step_json(f,events[i].index,events[i].token,events[i].position,&events[i].work,events[i].stem,events[i].native_write_file[0]?events[i].native_write_file:NULL);}
    fprintf(f,"],\"first\":{\"valid_kv_sha256\":\"%s\",\"logits\":",first_hash);readout_json(f,&first);fprintf(f,"},\"answer0\":");answer_json(f,&answers[0]);fprintf(f,",\"after_answer0\":{\"valid_kv_sha256\":\"%s\",\"logits\":",answer0_hash);readout_json(f,&after_answer0);fprintf(f,"},\"followup_skip\":%d,\"followup_ids\":[",follow_skip);for(int j=0;j<nf;j++){if(j)fputc(',',f);fprintf(f,"%d",follow[j]);}fprintf(f,"],\"followup_steps\":%d,\"after_followup\":{\"valid_kv_sha256\":\"%s\",\"logits\":",follow_steps,followup_hash);readout_json(f,&after_followup);fprintf(f,"},\"followup_position\":%d,\"answer1\":",followup_position);answer_json(f,&answers[1]);fprintf(f,",\"final\":{\"valid_kv_sha256\":\"%s\",\"logits\":",final_hash);readout_json(f,&final);fprintf(f,"},\"final_position\":%d,\"all_delivered_consumed\":true}",s->position);
    REQUIRE(!fflush(f));if(arm!=ARM_NATIVE)primary_steps+=(unsigned)event_count;else reference_steps+=(unsigned)event_count;
}

/* Replay the candidate's exact consumed token history through the ordinary
 * native layer path after the primary seal.  This is deliberately a separate
 * control from a native arm that generates its own text. */
static void run_native_same_id(FILE *f,QwenSession *s,const Trajectory *t,int case_id){
    Event events[MAX_TRAJECTORY];int event_count=0;OperandCapture oc={{output_directory,NULL},0,0};char stem[256],readout_stem[256],first_hash[65],answer0_hash[65],followup_hash[65],final_hash[65];ReadoutInfo first={0},after_answer0={0},after_followup={0},final={0};
    qwen_reset(s);for(int i=0;i<t->count;i++){
        REQUIRE(snprintf(stem,sizeof(stem),"%s-sameid-s%03d",t->primary_name,i)<(int)sizeof(stem));oc.base.name=stem;oc.save_gate_up=0;oc.minimal=1;
        int pos=s->position;StepWork w=step(s,t->ids[i],ARM_NATIVE,NULL,NULL,NULL,&oc);remember_event(events,&event_count,i,t->ids[i],pos,&w,ARM_NATIVE,stem);
        if(i+1==t->first_event_count){prefix_hash(s,first_hash);capture_readout(s,output_directory,stem,&first);}
        if(i+1==t->answer0_end_event_count){prefix_hash(s,answer0_hash);REQUIRE(snprintf(readout_stem,sizeof(readout_stem),"%s-sameid-after-answer0",t->primary_name)<(int)sizeof(readout_stem));capture_readout(s,output_directory,readout_stem,&after_answer0);}
        if(i+1==t->followup_end_event_count){prefix_hash(s,followup_hash);REQUIRE(snprintf(readout_stem,sizeof(readout_stem),"%s-sameid-after-followup",t->primary_name)<(int)sizeof(readout_stem));capture_readout(s,output_directory,readout_stem,&after_followup);}
    }
    prefix_hash(s,final_hash);REQUIRE(snprintf(readout_stem,sizeof(readout_stem),"%s-sameid-final",t->primary_name)<(int)sizeof(readout_stem));capture_readout(s,output_directory,readout_stem,&final);
    fprintf(f,"{\"name\":\"%s-sameid-native\",\"primary_name\":\"%s\",\"source_primary\":\"%s\",\"case\":%d,\"same_id\":true,\"consumed_ids\":[",t->primary_name,t->primary_name,t->primary_name,case_id);for(int i=0;i<t->count;i++){if(i)fputc(',',f);fprintf(f,"%d",t->ids[i]);}
    fprintf(f,"],\"steps\":[");for(int i=0;i<event_count;i++){if(i)fputc(',',f);step_json(f,events[i].index,events[i].token,events[i].position,&events[i].work,events[i].stem,NULL);}fprintf(f,"],\"first\":{\"valid_kv_sha256\":\"%s\",\"logits\":",first_hash);readout_json(f,&first);fprintf(f,"},\"after_answer0\":{\"valid_kv_sha256\":\"%s\",\"logits\":",answer0_hash);readout_json(f,&after_answer0);fprintf(f,"},\"after_followup\":{\"valid_kv_sha256\":\"%s\",\"logits\":",followup_hash);readout_json(f,&after_followup);fprintf(f,"},\"final\":{\"valid_kv_sha256\":\"%s\",\"logits\":",final_hash);readout_json(f,&final);fprintf(f,"},\"final_position\":%d}",s->position);REQUIRE(!fflush(f));
}

static void dense_write_from_file(const QwenModel *m,const char *directory,const char *stem){
    int d=m->hidden,h=m->intermediate;char path[4096],outname[256],gate_name[256],up_name[256];REQUIRE(snprintf(path,sizeof(path),"%s/%s-raw_receiver.f32",directory,stem)<(int)sizeof(path));float *host=malloc((size_t)d*4);REQUIRE(host);blob(path,host,(size_t)d*4);float *r=gpu_alloc((size_t)d*4),*norm=gpu_alloc((size_t)d*4),*gate=gpu_alloc((size_t)h*4),*up=gpu_alloc((size_t)h*4),*out=gpu_alloc((size_t)d*4),*rho=gpu_alloc(4);gpu_write(r,host,(size_t)d*4);gpu_rms_scale(norm,r,m->layer[LAYER].post_norm,d,m->epsilon,rho);gpu_matvec(m->gpu,m->layer[LAYER].gate,d,h,norm,gate);gpu_matvec(m->gpu,m->layer[LAYER].up,d,h,norm,up);
    REQUIRE(snprintf(gate_name,sizeof(gate_name),"%s/%s-mlp_gate.f32",directory,stem)<(int)sizeof(gate_name));REQUIRE(snprintf(up_name,sizeof(up_name),"%s/%s-mlp_up.f32",directory,stem)<(int)sizeof(up_name));FILE *probe=fopen(gate_name,"rb");int selected=probe!=NULL;if(probe)REQUIRE(!fclose(probe));if(selected){REQUIRE(snprintf(outname,sizeof(outname),"%s-native_gate.f32",stem)<(int)sizeof(outname));save_device(directory,outname,gate,h);REQUIRE(snprintf(outname,sizeof(outname),"%s-native_up.f32",stem)<(int)sizeof(outname));save_device(directory,outname,up,h);}
    gpu_swiglu(gate,up,h);gpu_matvec(m->gpu,m->layer[LAYER].down,h,d,gate,out);REQUIRE(snprintf(outname,sizeof(outname),"%s-native_write.f32",stem)<(int)sizeof(outname));save_device(directory,outname,out,d);gpu_free(r);gpu_free(norm);gpu_free(gate);gpu_free(up);gpu_free(out);gpu_free(rho);free(host);diagnostic_steps++;
}

static void seal_primary_files(const char *directory,long prefix_bytes,const char *prefix_hash){char path[4096];REQUIRE(snprintf(path,sizeof(path),"%s/primary-seal.json",directory)<(int)sizeof(path));FILE *out=fopen(path,"w");REQUIRE(out);fprintf(out,"{\"driver_prefix_bytes\":%ld,\"driver_prefix_sha256\":\"%s\",\"files\":[",prefix_bytes,prefix_hash);DIR *dir=opendir(directory);REQUIRE(dir);struct dirent *entry;int count=0;while((entry=readdir(dir))){size_t n=strlen(entry->d_name);if(n<4||(strcmp(entry->d_name+n-4,".f32")&&strcmp(entry->d_name+n-4,".txt")))continue;REQUIRE(snprintf(path,sizeof(path),"%s/%s",directory,entry->d_name)<(int)sizeof(path));struct stat st;REQUIRE(!stat(path,&st)&&S_ISREG(st.st_mode));FILE *in=fopen(path,"rb");REQUIRE(in);unsigned char *bytes=malloc((size_t)st.st_size);REQUIRE(bytes&&fread(bytes,1,(size_t)st.st_size,in)==(size_t)st.st_size&&!fclose(in));char hash[65];digest(bytes,(size_t)st.st_size,hash);free(bytes);if(count++)fputc(',',out);fprintf(out,"{\"name\":\"%s\",\"bytes\":%zu,\"sha256\":\"%s\"}",entry->d_name,(size_t)st.st_size,hash);}REQUIRE(!closedir(dir));fprintf(out,"]}\n");REQUIRE(!fclose(out));
}

static void emit_primary_diagnostics(const QwenModel *m,const char *directory){const char *suffix="-raw_receiver.f32";size_t suffix_len=strlen(suffix);DIR *dir=opendir(directory);REQUIRE(dir);struct dirent *e;while((e=readdir(dir))){size_t n=strlen(e->d_name);if(n<suffix_len||strcmp(e->d_name+n-suffix_len,suffix))continue;char stem[256];REQUIRE(n-suffix_len<sizeof(stem));memcpy(stem,e->d_name,n-suffix_len);stem[n-suffix_len]=0;dense_write_from_file(m,directory,stem);}REQUIRE(!closedir(dir));
}

static void ints(FILE *f,const int *ids,int n){fputc('[',f);for(int i=0;i<n;i++){if(i)fputc(',',f);fprintf(f,"%d",ids[i]);}fputc(']',f);}

int main(int argc,char **argv){
    REQUIRE(argc>=13&&argc<=15);const char *model_path=argv[1],*results=argv[2],*b3dir=argv[3],*compiled=argv[4],*python=argv[5],*compiler=argv[6];int budget=MAX_BUDGET;if(argc>=14){budget=atoi(argv[13]);REQUIRE(budget>0&&budget<=MAX_BUDGET);}if(argc==15)REQUIRE(!strcmp(argv[14],"smoke"));output_directory=results;mkdir_one(results);mkdir_one(compiled);
    int np[MAX_CASES],nf[MAX_CASES];int *prompts[MAX_CASES],*follows[MAX_CASES];const char *pp[]={argv[7],argv[9],argv[11]},*ff[]={argv[8],argv[10],argv[12]};for(int c=0;c<MAX_CASES;c++){prompts[c]=tokens(pp[c],&np[c]);follows[c]=tokens(ff[c],&nf[c]);}
    TcmModel bundle;QwenModel model;QwenSession session;QwenSubspace b3={0};char error[256],path[4096],model_digest[65];double started=clock_seconds();REQUIRE(!tcm_load(&bundle,model_path,error,sizeof(error))&&!qwen_model_init(&model,&bundle,error,sizeof(error)));tcm_unload(&bundle);wrapped_model=&model;strcpy(model_digest,model.digest);REQUIRE(!qwen_session_init(&session,&model,512)&&!qwen_subspace_init(&b3,&model));import_b3(&b3,b3dir);
    char input_dir[4096];REQUIRE(snprintf(input_dir,sizeof(input_dir),"%s/compile-input",compiled)<(int)sizeof(input_dir));export_compiler_inputs(&model,input_dir,model_digest);run_compiler(python,compiler,input_dir,compiled);
    Basis source={0},random={0};basis_load(&source,&model,compiled,"source");basis_load(&random,&model,compiled,"random");
    char input_manifest_sha[65],bank_manifest_sha[65],source_qt_file_sha[65],source_gq_file_sha[65],source_uq_file_sha[65],random_qt_file_sha[65],random_gq_file_sha[65],random_uq_file_sha[65];size_t ignored_bytes;
    REQUIRE(snprintf(path,sizeof(path),"%s/input-manifest.json",input_dir)<(int)sizeof(path));file_sha(path,input_manifest_sha,&ignored_bytes);REQUIRE(snprintf(path,sizeof(path),"%s/bank_manifest.json",compiled)<(int)sizeof(path));file_sha(path,bank_manifest_sha,&ignored_bytes);
    REQUIRE(snprintf(path,sizeof(path),"%s/source-QT.f32",compiled)<(int)sizeof(path));file_sha(path,source_qt_file_sha,&ignored_bytes);REQUIRE(snprintf(path,sizeof(path),"%s/source-GQ.f32",compiled)<(int)sizeof(path));file_sha(path,source_gq_file_sha,&ignored_bytes);REQUIRE(snprintf(path,sizeof(path),"%s/source-UQ.f32",compiled)<(int)sizeof(path));file_sha(path,source_uq_file_sha,&ignored_bytes);REQUIRE(snprintf(path,sizeof(path),"%s/random-QT.f32",compiled)<(int)sizeof(path));file_sha(path,random_qt_file_sha,&ignored_bytes);REQUIRE(snprintf(path,sizeof(path),"%s/random-GQ.f32",compiled)<(int)sizeof(path));file_sha(path,random_gq_file_sha,&ignored_bytes);REQUIRE(snprintf(path,sizeof(path),"%s/random-UQ.f32",compiled)<(int)sizeof(path));file_sha(path,random_uq_file_sha,&ignored_bytes);
    REQUIRE(snprintf(path,sizeof(path),"%s/driver.json",results)<(int)sizeof(path));FILE *f=fopen(path,"w+b");REQUIRE(f);fprintf(f,"{\"schema\":\"tcm-operand-basis-driver-v1\",\"model_digest\":\"%s\",\"layer\":%d,\"rank\":%d,\"capacity\":512,\"budget\":%d,\"mode\":\"%s\",\"arms\":[\"source64\",\"random64\",\"b3\",\"zero\",\"native\"],\"compiler_input_dir\":\"%s\",\"compiled_dir\":\"%s\",\"input_manifest_sha256\":\"%s\",\"bank_manifest_sha256\":\"%s\",\"basis_file_sha256\":{\"source_qt\":\"%s\",\"source_gq\":\"%s\",\"source_uq\":\"%s\",\"random_qt\":\"%s\",\"random_gq\":\"%s\",\"random_uq\":\"%s\"},\"primary_cases\":[",model_digest,LAYER,RANK,budget,argc==15?"smoke":"full",input_dir,compiled,input_manifest_sha,bank_manifest_sha,source_qt_file_sha,source_gq_file_sha,source_uq_file_sha,random_qt_file_sha,random_gq_file_sha,random_uq_file_sha);
    Arm primary_arms[]={ARM_SOURCE,ARM_RANDOM,ARM_B3,ARM_ZERO};Trajectory primary_trajectory[MAX_CASES][4]={{0}};
    for(int c=0;c<MAX_CASES;c++){if(c)fputc(',',f);fprintf(f,"{\"case\":%d,\"prompt_tokens\":",c);ints(f,prompts[c],np[c]);fprintf(f,",\"followup_tokens\":");ints(f,follows[c],nf[c]);fprintf(f,",\"arms\":[");for(unsigned a=0;a<sizeof(primary_arms)/sizeof(primary_arms[0]);a++){if(a)fputc(',',f);run_arm(f,&session,&source,&random,&b3,primary_arms[a],c,prompts[c],np[c],follows[c],nf[c],budget,&primary_trajectory[c][a]);}fprintf(f,"]}");}
    REQUIRE(!fflush(f));long sealed=ftell(f);REQUIRE(sealed>0);unsigned char *raw=malloc((size_t)sealed);REQUIRE(raw&&!fseek(f,0,SEEK_SET)&&fread(raw,1,(size_t)sealed,f)==(size_t)sealed);char primary_hash[65];digest(raw,(size_t)sealed,primary_hash);free(raw);REQUIRE(!fseek(f,0,SEEK_END));seal_primary_files(results,sealed,primary_hash);emit_primary_diagnostics(&model,results);
    fprintf(f,"],\"primary_seal_bytes\":%ld,\"primary_seal_sha256\":\"%s\",\"diagnostics_after_seal\":true,\"references\":[",sealed,primary_hash);for(int c=0;c<MAX_CASES;c++){if(c)fputc(',',f);fprintf(f,"{\"case\":%d,\"native\":",c);run_arm(f,&session,&source,&random,&b3,ARM_NATIVE,c,prompts[c],np[c],follows[c],nf[c],budget,NULL);fprintf(f,",\"same_ids\":[");for(unsigned a=0;a<sizeof(primary_arms)/sizeof(primary_arms[0]);a++){if(a)fputc(',',f);run_native_same_id(f,&session,&primary_trajectory[c][a],c);}fprintf(f,"]}");}fprintf(f,"],\"work\":{\"primary_steps\":%u,\"reference_steps\":%u,\"same_id_replays\":%u,\"postseal_dense_diagnostics\":%u,\"source_basis_q_calls\":%u,\"source_basis_g_calls\":%u,\"source_basis_u_calls\":%u,\"random_basis_q_calls\":%u,\"random_basis_g_calls\":%u,\"random_basis_u_calls\":%u},\"basis_sha256\":{\"source_qt\":\"%s\",\"source_gq\":\"%s\",\"source_uq\":\"%s\",\"random_qt\":\"%s\",\"random_gq\":\"%s\",\"random_uq\":\"%s\"},\"seconds\":%.9g}\n",primary_steps,reference_steps,MAX_CASES*4,diagnostic_steps,source.qt_calls,source.gq_calls,source.uq_calls,random.qt_calls,random.gq_calls,random.uq_calls,source.qt_sha,source.gq_sha,source.uq_sha,random.qt_sha,random.gq_sha,random.uq_sha,clock_seconds()-started);REQUIRE(!fclose(f));basis_free(&source);basis_free(&random);qwen_subspace_free(&b3);qwen_session_free(&session);qwen_model_free(&model);for(int c=0;c<MAX_CASES;c++){free(prompts[c]);free(follows[c]);}return 0;
}
