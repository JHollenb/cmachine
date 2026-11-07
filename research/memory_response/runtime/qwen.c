#define _POSIX_C_SOURCE 200809L
#include "qwen.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdatomic.h>

static _Atomic uint64_t next_session_id=1;
static _Atomic uint64_t next_frame_id=1;

static float *weight(QwenModel *m,const TcmModel *b,const char *name,int rank,int a,int c){
    const TcmTensor *t=tcm_tensor(b,name);if(!t||t->rank!=(unsigned)rank||t->dim[0]!=(uint64_t)a||(rank==2&&t->dim[1]!=(uint64_t)c)||(t->dtype!=TCM_F32&&t->dtype!=TCM_BF16))return NULL;
    size_t count=t->bytes/(t->dtype==TCM_F32?4:2);float *p=gpu_weight(t->data,t->dtype,count);m->allocations[m->allocation_count++]=p;m->weight_bytes+=count*4;return p;
}
int qwen_model_init(QwenModel *m,const TcmModel *b,char *error,size_t cap){
    memset(m,0,sizeof(*m));
    if(b->architecture!=TCM_QWEN||b->count!=338||b->config[0]!=1536||b->config[1]!=28||b->config[2]!=12||b->config[3]!=2||b->config[4]!=128||b->config[5]!=8960||b->config[6]!=151936||b->config[10]!=32768){snprintf(error,cap,"unsupported Qwen2.5-1.5B profile");return -1;}
    m->hidden=(int)b->config[0];m->layers=(int)b->config[1];m->qheads=(int)b->config[2];m->kvheads=(int)b->config[3];m->head=(int)b->config[4];m->intermediate=(int)b->config[5];m->vocab=(int)b->config[6];m->eos=(int)b->config[9];m->max_position=(int)b->config[10];
    m->eos_extra=(int)b->config[12];
    memcpy(&m->epsilon,&b->config[7],4);memcpy(&m->theta,&b->config[8],4);memcpy(m->identity,b->identity,65);memcpy(m->digest,b->digest,65);
    if(m->epsilon!=1e-6f||m->theta!=1e6f||b->config[11]!=1){snprintf(error,cap,"Qwen numerical configuration differs");return -1;}
    m->allocations=calloc(b->count,sizeof(float*));m->layer=calloc((size_t)m->layers,sizeof(QwenLayer));if(!m->allocations||!m->layer){snprintf(error,cap,"Qwen descriptor allocation failed");qwen_model_free(m);return -1;}m->gpu=gpu_open();
    int bad=0,d=m->hidden,k=m->kvheads*m->head,h=m->intermediate;
    m->embedding=weight(m,b,"model.embed_tokens.weight",2,m->vocab,d);m->norm=weight(m,b,"model.norm.weight",1,d,0);bad|=!m->embedding||!m->norm;
    for(int i=0;i<m->layers;i++){
        QwenLayer *l=&m->layer[i];char name[64];
#define W(field,suffix,rank,a,c) snprintf(name,sizeof(name),"model.layers.%d." suffix,i);l->field=weight(m,b,name,rank,a,c);bad|=!l->field
        W(input_norm,"input_layernorm.weight",1,d,0);W(post_norm,"post_attention_layernorm.weight",1,d,0);
        W(q,"self_attn.q_proj.weight",2,d,d);W(k,"self_attn.k_proj.weight",2,k,d);W(v,"self_attn.v_proj.weight",2,k,d);
        W(qb,"self_attn.q_proj.bias",1,d,0);W(kb,"self_attn.k_proj.bias",1,k,0);W(vb,"self_attn.v_proj.bias",1,k,0);W(o,"self_attn.o_proj.weight",2,d,d);
        W(gate,"mlp.gate_proj.weight",2,h,d);W(up,"mlp.up_proj.weight",2,h,d);W(down,"mlp.down_proj.weight",2,d,h);
#undef W
    }
    if(bad||m->allocation_count!=338){snprintf(error,cap,"Qwen tensor names/shapes differ");qwen_model_free(m);return -1;}return 0;
}
void qwen_model_free(QwenModel *m){for(unsigned i=0;i<m->allocation_count;i++)gpu_free(m->allocations[i]);free(m->allocations);free(m->layer);gpu_close(m->gpu);memset(m,0,sizeof(*m));}
static void snapshot(const QwenSession *,QwenCheckpoint *);
static void restore(QwenSession *,const QwenCheckpoint *);
int qwen_session_init(QwenSession *s,const QwenModel *m,int capacity){return qwen_session_init_layout(s,m,capacity,0,NULL,NULL,NULL);}
int qwen_session_init_layout(QwenSession *s,const QwenModel *m,int capacity,uint64_t mask,QwenCacheStep step,QwenCacheRewind rewind,void *context){
    memset(s,0,sizeof(*s));if(capacity<1||capacity>m->max_position)return -1;s->model=m;s->capacity=capacity;s->pending=-1;s->session_id=atomic_fetch_add(&next_session_id,1);s->epoch=1;s->observe_position=s->observe_layer=-1;
    if(m->layers<1||m->layers>64||(m->layers<64&&(mask>>m->layers))||(mask&&(!step||!rewind)))return -1;
    s->cache_mask=mask;s->cache_step=step;s->cache_rewind=rewind;s->cache_context=context;
    for(int i=0;i<m->layers;i++)s->cache_slots[i]=(mask&(UINT64_C(1)<<i))?-1:s->dense_layers++;
    size_t kv=(size_t)s->dense_layers*m->kvheads*capacity*m->head;
#define ALLOC(field,n) do { size_t alloc_count=(size_t)(n); if(alloc_count){s->field=gpu_alloc(alloc_count*sizeof(float));s->allocated_bytes+=alloc_count*sizeof(float);} else s->field=NULL; } while(0)
    ALLOC(keys,kv);ALLOC(values,kv);ALLOC(x,m->hidden);ALLOC(norm,m->hidden);ALLOC(q,m->hidden);ALLOC(k,m->kvheads*m->head);ALLOC(v,m->kvheads*m->head);ALLOC(attention,m->hidden);ALLOC(projected,m->hidden);ALLOC(gate,m->intermediate);ALLOC(up,m->intermediate);ALLOC(logits,m->vocab);ALLOC(scores,(size_t)m->qheads*capacity);
#undef ALLOC
    return 0;
}
void qwen_session_free(QwenSession *s){
#define FREE(field) gpu_free(s->field)
    FREE(keys);FREE(values);FREE(x);FREE(norm);FREE(q);FREE(k);FREE(v);FREE(attention);FREE(projected);FREE(gate);FREE(up);FREE(logits);FREE(scores);
#undef FREE
    memset(s,0,sizeof(*s));
}
const char *qwen_port_name(QwenPort port){
    static const char *names[]={"input","query","key","value","attention","mlp_input","mlp_gate","mlp_up","mlp_return","output","attention_write","raw_receiver"};
    return port>=0&&port<QWEN_PORT_COUNT?names[port]:NULL;
}
static void observe(QwenSession *s,QwenPort port,const float *data,size_t count){
    if(s->observer&&s->position==s->observe_position&&s->layer==s->observe_layer)
        s->observer(s->observer_context,s->position,s->layer,port,data,count);
}
int qwen_begin_token(QwenSession *s,int token){
    const QwenModel *m=s->model;if(token<0||token>=m->vocab||s->position>=s->capacity||s->phase==2)return -1;
    snapshot(s,&s->token_parent);s->token_parent_valid=1;
    s->phase=2;s->layer=0;gpu_copy(s->x,m->embedding+(size_t)token*m->hidden,(size_t)m->hidden*4);return 0;
}
int qwen_plane_init(QwenPlane *p,const QwenModel *m){
    if(!p||!m||m->hidden<1||m->intermediate<1)return -1;
    memset(p,0,sizeof(*p));p->model=m;
    for(int j=0;j<3;j++){p->basis[j]=gpu_alloc((size_t)m->hidden*4);p->gate[j]=gpu_alloc((size_t)m->intermediate*4);p->up[j]=gpu_alloc((size_t)m->intermediate*4);}
    p->baseline_write=gpu_alloc((size_t)m->hidden*4);p->rho=gpu_alloc(4);
    p->allocated_bytes=((size_t)4*m->hidden+(size_t)6*m->intermediate+1)*4;return 0;
}
void qwen_plane_free(QwenPlane *p){
    for(int j=0;j<3;j++){gpu_free(p->basis[j]);gpu_free(p->gate[j]);gpu_free(p->up[j]);}
    gpu_free(p->baseline_write);gpu_free(p->rho);memset(p,0,sizeof(*p));
}
static int retry_layer(QwenSession *s,const float *entry){
    /* Scratch and the current KV row may have changed. Restore the layer input
     * so an explicit ordinary retry overwrites that row and recomputes safely. */
    gpu_copy(s->x,entry,(size_t)s->model->hidden*4);return -2;
}
int qwen_subspace_init(QwenSubspace *p,const QwenModel *m){
    if(!p||!m||m->hidden<2||m->intermediate<1)return -1;
    memset(p,0,sizeof(*p));p->model=m;p->layer=-1;int d=m->hidden,h=m->intermediate;
    for(int j=0;j<3;j++){p->basis[j]=gpu_alloc((size_t)d*4);p->gate[j]=gpu_alloc((size_t)h*4);p->up[j]=gpu_alloc((size_t)h*4);}
#define SUB_ALLOC(field,n) p->field=gpu_alloc((size_t)(n)*4)
    SUB_ALLOC(baseline_write,d);SUB_ALLOC(entry,d);SUB_ALLOC(attention_write,d);SUB_ALLOC(receiver,d);SUB_ALLOC(reconstruction,d);SUB_ALLOC(omitted,d);SUB_ALLOC(carry_omitted,d);SUB_ALLOC(read_complement,d);SUB_ALLOC(correction_operand,d);SUB_ALLOC(rho,1);SUB_ALLOC(correction_gate,h);SUB_ALLOC(correction_up,h);
#undef SUB_ALLOC
    p->q0=malloc((size_t)d*sizeof(double));p->q1=malloc((size_t)d*sizeof(double));p->q2=malloc((size_t)d*sizeof(double));p->host=malloc((size_t)d*4);
    if(!p->q0||!p->q1||!p->q2||!p->host){qwen_subspace_free(p);return -1;}
    p->gpu_bytes=((size_t)12*d+(size_t)8*h+1)*4;p->host_bytes=(size_t)d*(3*sizeof(double)+sizeof(float));return 0;
}
void qwen_subspace_free(QwenSubspace *p){
    for(int j=0;j<3;j++){gpu_free(p->basis[j]);gpu_free(p->gate[j]);gpu_free(p->up[j]);}
    gpu_free(p->baseline_write);gpu_free(p->entry);gpu_free(p->attention_write);gpu_free(p->receiver);gpu_free(p->reconstruction);gpu_free(p->omitted);gpu_free(p->carry_omitted);gpu_free(p->read_complement);gpu_free(p->correction_operand);gpu_free(p->rho);gpu_free(p->correction_gate);gpu_free(p->correction_up);
    free(p->q0);free(p->q1);free(p->q2);free(p->host);memset(p,0,sizeof(*p));
}
int qwen_subspace_extend(QwenSubspace *p,const float *direction,QwenPlaneWork *work){
    if(work)memset(work,0,sizeof(*work));
    if(!p||!p->ready||p->rank!=2||!direction||!work)return -1;
    const QwenModel *m=p->model;int d=m->hidden,h=m->intermediate;const QwenLayer *l=&m->layer[p->layer];
    gpu_read(p->host,direction,(size_t)d*4);
    for(int j=0;j<d;j++){if(!isfinite(p->host[j]))return -2;p->q2[j]=p->host[j];}
    /* Remove B2 twice, then round the actual stored operand to FP32. */
    for(int pass=0;pass<2;pass++)for(int k=0;k<2;k++){
        const double *q=k?p->q1:p->q0;double dot=0;for(int j=0;j<d;j++)dot+=q[j]*p->q2[j];
        for(int j=0;j<d;j++)p->q2[j]-=dot*q[j];
    }
    for(int j=0;j<d;j++){p->host[j]=(float)p->q2[j];if(!isfinite(p->host[j]))return -2;p->q2[j]=p->host[j];}
    /* QR describes the rounded operand, including its small B2 components. */
    p->r02=p->r12=0;
    for(int pass=0;pass<2;pass++)for(int k=0;k<2;k++){
        const double *q=k?p->q1:p->q0;double dot=0;for(int j=0;j<d;j++)dot+=q[j]*p->q2[j];
        if(k)p->r12+=dot;else p->r02+=dot;for(int j=0;j<d;j++)p->q2[j]-=dot*q[j];
    }
    double norm=0;for(int j=0;j<d;j++)norm+=p->q2[j]*p->q2[j];p->r22=sqrt(norm);
    if(!(p->r22>0)||!isfinite(p->r22))return -2;
    for(int j=0;j<d;j++)p->q2[j]/=p->r22;
    float *b=gpu_weight(p->host,TCM_F32,(size_t)d);gpu_free(p->basis[2]);p->basis[2]=b;
    gpu_multiply(p->reconstruction,b,l->post_norm,d);
    gpu_matvec(m->gpu,l->gate,d,h,p->reconstruction,p->gate[2]);work->compile_matvecs++;
    gpu_matvec(m->gpu,l->up,d,h,p->reconstruction,p->up[2]);work->compile_matvecs++;
    if(!gpu_identical_finite(m->gpu,p->gate[2],p->gate[2],h)||!gpu_identical_finite(m->gpu,p->up[2],p->up[2],h))return -2;
    p->rank=3;return 0;
}
static int subspace_qr(QwenSubspace *p){
    int d=p->model->hidden;double norm=0;
    gpu_read(p->host,p->basis[0],(size_t)d*4);
    for(int j=0;j<d;j++){if(!isfinite(p->host[j]))return -1;p->q0[j]=p->host[j];norm+=p->q0[j]*p->q0[j];}
    p->r00=sqrt(norm);if(!(p->r00>0)||!isfinite(p->r00))return -1;
    for(int j=0;j<d;j++)p->q0[j]/=p->r00;
    gpu_read(p->host,p->basis[1],(size_t)d*4);p->r01=0;
    for(int j=0;j<d;j++){if(!isfinite(p->host[j]))return -1;p->q1[j]=p->host[j];p->r01+=p->q0[j]*p->q1[j];}
    for(int j=0;j<d;j++)p->q1[j]-=p->r01*p->q0[j];
    double correction=0;for(int j=0;j<d;j++)correction+=p->q0[j]*p->q1[j];
    norm=0;for(int j=0;j<d;j++){p->q1[j]-=correction*p->q0[j];norm+=p->q1[j]*p->q1[j];}p->r01+=correction;
    p->r11=sqrt(norm);if(!(p->r11>0)||!isfinite(p->r11))return -1;
    for(int j=0;j<d;j++)p->q1[j]/=p->r11;
    double aa=p->r00*p->r00,bb=p->r00*p->r01,cc=p->r01*p->r01+p->r11*p->r11;
    double largest=(aa+cc+hypot(aa-cc,2*bb))/2,product=p->r00*p->r11;
    p->condition=largest/fabs(product);return isfinite(p->condition)?0:-1;
}
static int subspace_coordinates(QwenSubspace *p,QwenSubspaceStats *stats,int rank){
    int d=p->model->hidden;double v0=0,v1=0,v2=0,norm=0;gpu_read(p->host,p->receiver,(size_t)d*4);
    for(int j=0;j<d;j++){double v=p->host[j];if(!isfinite(v))return -1;v0+=p->q0[j]*v;v1+=p->q1[j]*v;if(rank==3)v2+=p->q2[j]*v;norm+=v*v;}
    stats->receiver_norm=sqrt(norm);stats->coefficients[2]=rank==3?v2/p->r22:0;
    if(rank==3){stats->coefficients[1]=(v1-p->r12*stats->coefficients[2])/p->r11;stats->coefficients[0]=(v0-p->r01*stats->coefficients[1]-p->r02*stats->coefficients[2])/p->r00;}
    else{stats->coefficients[1]=v1/p->r11;stats->coefficients[0]=(v0-p->r01*stats->coefficients[1])/p->r00;}
    for(int j=0;j<3;j++){stats->executed[j]=(float)stats->coefficients[j];if(!isfinite(stats->executed[j]))return -1;}
    gpu_linear2(p->reconstruction,p->basis[0],p->basis[1],d,stats->executed[0],stats->executed[1],NULL);
    if(rank==3)gpu_linear2(p->reconstruction,p->reconstruction,p->basis[2],d,1,stats->executed[2],NULL);
    gpu_plane(p->omitted,p->receiver,p->reconstruction,p->reconstruction,d,-1,0);
    gpu_read(&stats->rho,p->rho,4);if(!isfinite(stats->rho)||!(stats->rho>0))return -1;
    gpu_read(p->host,p->reconstruction,(size_t)d*4);norm=0;for(int j=0;j<d;j++){double v=p->host[j];if(!isfinite(v))return -1;norm+=v*v;}stats->reconstruction_norm=sqrt(norm);
    gpu_read(p->host,p->omitted,(size_t)d*4);norm=0;for(int j=0;j<d;j++){double v=p->host[j];if(!isfinite(v))return -1;norm+=v*v;}stats->omitted_norm=sqrt(norm);return 0;
}
static int subspace_split(QwenSubspace *p){
    int d=p->model->hidden;double v[3]={0,0,0};const double *q[]={p->q0,p->q1,p->q2};
    gpu_read(p->host,p->entry,(size_t)d*4);
    for(int j=0;j<d;j++)for(int k=0;k<3;k++)v[k]+=q[k][j]*(double)p->host[j];
    for(int j=0;j<d;j++){
        double e=p->host[j];for(int k=0;k<3;k++)e-=q[k][j]*v[k];
        p->host[j]=(float)e;if(!isfinite(p->host[j]))return -1;
    }
    gpu_write(p->carry_omitted,p->host,(size_t)d*4);
    gpu_plane(p->read_complement,p->omitted,p->carry_omitted,p->carry_omitted,d,-1,0);
    return gpu_identical_finite(p->model->gpu,p->read_complement,p->read_complement,d)?0:-1;
}
static int advance_layer(QwenSession *s,QwenPlane *compile,const QwenPlane *p,QwenPlaneMode mode,float a,float b,QwenPlaneWork *work,QwenSubspace *sub,QwenSubspaceMode submode,QwenSubspaceStats *stats,const float *operand){
    if(stats)memset(stats,0,sizeof(*stats));
    if(work)memset(work,0,sizeof(*work));
    const QwenModel *m=s->model;if(s->phase!=2||s->layer>=m->layers)return -1;
    int i=s->layer,d=m->hidden,k=m->kvheads*m->head;const QwenLayer *l=&m->layer[i];
        if(sub){
            if(sub->model!=m||submode<QWEN_SUBSPACE_COMPILE||submode>QWEN_SUBSPACE_OPERAND_CORRECT||!stats)return -1;
            if(submode==QWEN_SUBSPACE_COMPILE?sub->ready:(!sub->ready||sub->layer!=i))return -1;
            if(submode>=QWEN_SUBSPACE_CARRY_CORRECT&&sub->rank!=3)return -1;
            if(submode==QWEN_SUBSPACE_OPERAND_CORRECT){
                if(!operand)return -1;
                if(!gpu_identical_finite(m->gpu,operand,operand,d))return -2;
                gpu_copy(sub->correction_operand,operand,(size_t)d*4);
            }
            if(!gpu_identical_finite(m->gpu,s->x,s->x,d))return -2;
            gpu_copy(sub->entry,s->x,(size_t)d*4);
            if(submode==QWEN_SUBSPACE_COMPILE)gpu_copy(sub->basis[0],s->x,(size_t)d*4);
        }
        if(compile){
            if(compile->model!=m||compile->ready)return -1;
            if(!gpu_identical_finite(m->gpu,s->x,s->x,d))return -2;
            gpu_copy(compile->basis[2],s->x,(size_t)d*4);
        }
        if(p){
            if(!p->ready||p->model!=m||p->session_id!=s->session_id||p->epoch!=s->epoch||p->position!=s->position||p->layer!=i||!isfinite(a)||!isfinite(b)||mode<QWEN_PLANE_DENSE||mode>QWEN_PLANE_HELD_WRITE)return -1;
            if(!gpu_identical_finite(m->gpu,s->x,p->basis[2],d))return -1;
        }
        observe(s,QWEN_INPUT,s->x,(size_t)d);
        gpu_rms(s->norm,s->x,l->input_norm,d,m->epsilon);
        gpu_matvec(m->gpu,l->q,d,d,s->norm,s->q);gpu_bias(s->q,l->qb,d);
        gpu_matvec(m->gpu,l->k,d,k,s->norm,s->k);gpu_bias(s->k,l->kb,k);
        gpu_matvec(m->gpu,l->v,d,k,s->norm,s->v);gpu_bias(s->v,l->vb,k);
        gpu_rope(s->q,m->qheads,m->head,s->position,m->theta);gpu_rope(s->k,m->kvheads,m->head,s->position,m->theta);
        observe(s,QWEN_QUERY,s->q,(size_t)d);observe(s,QWEN_KEY,s->k,(size_t)k);observe(s,QWEN_VALUE,s->v,(size_t)k);
        if(s->cache_slots[i]<0){
            if(!s->cache_step)return -1;
            int error=s->cache_step(s->cache_context,i,s->position,s->q,s->k,s->v,s->attention,s->scores);if(error)return error;
        }else{
            size_t base=(size_t)s->cache_slots[i]*m->kvheads*s->capacity*m->head;
            gpu_cache(s->keys+base,s->k,m->kvheads,m->head,s->capacity,s->position);gpu_cache(s->values+base,s->v,m->kvheads,m->head,s->capacity,s->position);
            gpu_attention(m->gpu,s->attention,s->q,s->keys+base,s->values+base,s->scores,m->qheads,m->kvheads,m->head,s->capacity,s->position+1);
        }
        observe(s,QWEN_ATTENTION,s->attention,(size_t)d);
        gpu_matvec(m->gpu,l->o,d,d,s->attention,s->projected);
        /* A read-guard refusal can have written the current token's KV row, but
         * never retained history. The token is still partial; reset or recompute
         * this ordinary layer before continuing. There is no fallback claim. */
        if(p&&!gpu_identical_finite(m->gpu,s->projected,p->basis[1],d))return -2;
        if(compile){
            if(!gpu_identical_finite(m->gpu,s->projected,s->projected,d))return -2;
            gpu_copy(compile->basis[1],s->projected,(size_t)d*4);
        }
        if(sub){gpu_copy(sub->attention_write,s->projected,(size_t)d*4);if(submode==QWEN_SUBSPACE_COMPILE)gpu_copy(sub->basis[1],s->projected,(size_t)d*4);}
        observe(s,QWEN_ATTENTION_WRITE,s->projected,(size_t)d);
        gpu_add(s->x,s->projected,d);
        /* Actual post-attention add, before any experimental plane replacement. */
        observe(s,QWEN_RAW_RECEIVER,s->x,(size_t)d);
        if(compile)gpu_copy(compile->basis[0],s->x,(size_t)d*4);
        if(sub){
            gpu_copy(sub->receiver,s->x,(size_t)d*4);
            if(!gpu_identical_finite(m->gpu,s->x,s->x,d))return retry_layer(s,sub->entry);
            gpu_rms_scale(s->norm,s->x,l->post_norm,d,m->epsilon,sub->rho);
            if(submode!=QWEN_SUBSPACE_COMPILE&&subspace_coordinates(sub,stats,submode==QWEN_SUBSPACE_PROJECT2?2:sub->rank))return retry_layer(s,sub->entry);
            if(submode>=QWEN_SUBSPACE_CARRY_CORRECT&&subspace_split(sub))return retry_layer(s,sub->entry);
        }else if(p){
            gpu_plane(s->x,p->basis[0],p->basis[1],p->basis[2],d,a,b);
            if(!gpu_identical_finite(m->gpu,s->x,s->x,d))return retry_layer(s,p->basis[2]);
            gpu_rms_scale(s->norm,s->x,l->post_norm,d,m->epsilon,p->rho);
        }else gpu_rms(s->norm,s->x,l->post_norm,d,m->epsilon);
        observe(s,QWEN_MLP_INPUT,s->norm,(size_t)d);
        if((p&&mode==QWEN_PLANE_HELD_WRITE)||(sub&&submode==QWEN_SUBSPACE_HELD)){
            /* No gate/up ports are emitted: these computations did not run. */
            gpu_copy(s->projected,sub?sub->baseline_write:p->baseline_write,(size_t)d*4);
        }else{
            if(sub&&(submode==QWEN_SUBSPACE_PROJECT||submode==QWEN_SUBSPACE_CORRECT||submode==QWEN_SUBSPACE_PROJECT2||submode>=QWEN_SUBSPACE_CARRY_CORRECT)){
                int rank=submode==QWEN_SUBSPACE_PROJECT2?2:sub->rank;
                gpu_linear2(s->gate,sub->gate[0],sub->gate[1],m->intermediate,stats->executed[0],stats->executed[1],rank==2?sub->rho:NULL);
                gpu_linear2(s->up,sub->up[0],sub->up[1],m->intermediate,stats->executed[0],stats->executed[1],rank==2?sub->rho:NULL);
                if(rank==3){
                    gpu_linear2(s->gate,s->gate,sub->gate[2],m->intermediate,1,stats->executed[2],sub->rho);
                    gpu_linear2(s->up,s->up,sub->up[2],m->intermediate,1,stats->executed[2],sub->rho);
                }
                if(submode==QWEN_SUBSPACE_CORRECT||submode>=QWEN_SUBSPACE_CARRY_CORRECT){
                    const float *correction=submode==QWEN_SUBSPACE_OPERAND_CORRECT?sub->correction_operand:submode==QWEN_SUBSPACE_CARRY_CORRECT?sub->carry_omitted:submode==QWEN_SUBSPACE_READ_CORRECT?sub->read_complement:sub->omitted;
                    gpu_multiply(s->norm,correction,l->post_norm,d);
                    gpu_matvec(m->gpu,l->gate,d,m->intermediate,s->norm,sub->correction_gate);gpu_matvec(m->gpu,l->up,d,m->intermediate,s->norm,sub->correction_up);
                    gpu_linear2(sub->correction_gate,sub->correction_gate,NULL,m->intermediate,1,0,sub->rho);gpu_linear2(sub->correction_up,sub->correction_up,NULL,m->intermediate,1,0,sub->rho);
                    gpu_add(s->gate,sub->correction_gate,m->intermediate);gpu_add(s->up,sub->correction_up,m->intermediate);if(work)work->gate_up_matvecs+=2;
                }
            }else if(sub&&submode==QWEN_SUBSPACE_REASSOC){
                gpu_multiply(s->norm,sub->receiver,l->post_norm,d);
                gpu_matvec(m->gpu,l->gate,d,m->intermediate,s->norm,s->gate);gpu_matvec(m->gpu,l->up,d,m->intermediate,s->norm,s->up);
                gpu_linear2(s->gate,s->gate,NULL,m->intermediate,1,0,sub->rho);gpu_linear2(s->up,s->up,NULL,m->intermediate,1,0,sub->rho);if(work)work->gate_up_matvecs+=2;
            }else if(p&&mode==QWEN_PLANE_COMPILED){
                gpu_plane_scaled(s->gate,p->gate[0],p->gate[1],p->gate[2],m->intermediate,a,b,p->rho);
                gpu_plane_scaled(s->up,p->up[0],p->up[1],p->up[2],m->intermediate,a,b,p->rho);
            }else{
                gpu_matvec(m->gpu,l->gate,d,m->intermediate,s->norm,s->gate);gpu_matvec(m->gpu,l->up,d,m->intermediate,s->norm,s->up);
                if(work)work->gate_up_matvecs+=2;
            }
            observe(s,QWEN_MLP_GATE,s->gate,(size_t)m->intermediate);observe(s,QWEN_MLP_UP,s->up,(size_t)m->intermediate);
            gpu_swiglu(s->gate,s->up,m->intermediate);
            gpu_matvec(m->gpu,l->down,m->intermediate,d,s->gate,s->projected);
            if(work)work->down_matvecs++;
        }
        if(compile)gpu_copy(compile->baseline_write,s->projected,(size_t)d*4);
        if(sub&&submode==QWEN_SUBSPACE_COMPILE)gpu_copy(sub->baseline_write,s->projected,(size_t)d*4);
        gpu_add(s->x,s->projected,d);
        observe(s,QWEN_MLP_RETURN,s->projected,(size_t)d);observe(s,QWEN_OUTPUT,s->x,(size_t)d);
        if(compile){
            for(int j=0;j<3;j++){
                gpu_multiply(s->norm,compile->basis[j],l->post_norm,d);
                gpu_matvec(m->gpu,l->gate,d,m->intermediate,s->norm,compile->gate[j]);
                gpu_matvec(m->gpu,l->up,d,m->intermediate,s->norm,compile->up[j]);
                if(work)work->compile_matvecs+=2;
                if(!gpu_identical_finite(m->gpu,compile->gate[j],compile->gate[j],m->intermediate)||!gpu_identical_finite(m->gpu,compile->up[j],compile->up[j],m->intermediate))return retry_layer(s,compile->basis[2]);
            }
            if(!gpu_identical_finite(m->gpu,compile->baseline_write,compile->baseline_write,d))return retry_layer(s,compile->basis[2]);
            compile->session_id=s->session_id;compile->epoch=s->epoch;compile->position=s->position;compile->layer=i;compile->ready=1;
        }
        if(sub&&submode==QWEN_SUBSPACE_COMPILE){
            if(subspace_qr(sub))return retry_layer(s,sub->entry);
            for(int j=0;j<2;j++){
                gpu_multiply(s->norm,sub->basis[j],l->post_norm,d);
                gpu_matvec(m->gpu,l->gate,d,m->intermediate,s->norm,sub->gate[j]);gpu_matvec(m->gpu,l->up,d,m->intermediate,s->norm,sub->up[j]);if(work)work->compile_matvecs+=2;
                if(!gpu_identical_finite(m->gpu,sub->gate[j],sub->gate[j],m->intermediate)||!gpu_identical_finite(m->gpu,sub->up[j],sub->up[j],m->intermediate))return retry_layer(s,sub->entry);
            }
            if(!gpu_identical_finite(m->gpu,sub->baseline_write,sub->baseline_write,d))return retry_layer(s,sub->entry);
            sub->layer=i;sub->source_position=s->position;
            sub->rank=2;if(subspace_coordinates(sub,stats,2))return retry_layer(s,sub->entry);
            sub->ready=1;
        }
    s->layer++;return 0;
}
int qwen_advance_layer(QwenSession *s){return advance_layer(s,NULL,NULL,QWEN_PLANE_DENSE,0,0,NULL,NULL,0,NULL,NULL);}
int qwen_advance_layer_compile(QwenSession *s,QwenPlane *p,QwenPlaneWork *work){if(!p)return -1;return advance_layer(s,p,NULL,QWEN_PLANE_DENSE,0,0,work,NULL,0,NULL,NULL);}
int qwen_advance_layer_plane(QwenSession *s,const QwenPlane *p,QwenPlaneMode mode,float a,float b,QwenPlaneWork *work){if(!p)return -1;return advance_layer(s,NULL,p,mode,a,b,work,NULL,0,NULL,NULL);}
int qwen_advance_layer_subspace(QwenSession *s,QwenSubspace *p,QwenSubspaceMode mode,QwenSubspaceStats *stats){if(!p||!stats)return -1;return advance_layer(s,NULL,NULL,QWEN_PLANE_DENSE,0,0,&stats->work,p,mode,stats,NULL);}
int qwen_advance_layer_subspace_operand(QwenSession *s,QwenSubspace *p,const float *operand,size_t count,QwenSubspaceStats *stats){
    if(!s||!p||!stats||count!=(size_t)s->model->hidden)return -1;
    return advance_layer(s,NULL,NULL,QWEN_PLANE_DENSE,0,0,&stats->work,p,QWEN_SUBSPACE_OPERAND_CORRECT,stats,operand);
}
int qwen_finish_token(QwenSession *s){
    const QwenModel *m=s->model;if(s->phase!=2||s->layer!=m->layers)return -1;int d=m->hidden;
    gpu_rms(s->norm,s->x,m->norm,d,m->epsilon);gpu_matvec(m->gpu,m->embedding,d,m->vocab,s->norm,s->logits);s->top=gpu_top2(m->gpu,s->logits,m->vocab);
    s->pending=s->top.first;s->delivered=0;s->position++;s->layer=m->layers;s->phase=1;s->token_parent_valid=0;return 0;
}
int qwen_push(QwenSession *s,int token){
    if(qwen_begin_token(s,token))return -1;
    while(s->layer<s->model->layers)if(qwen_advance_layer(s))return -1;
    return qwen_finish_token(s);
}
int qwen_advance(QwenSession *s){return qwen_push(s,s->pending);}
static int at_boundary(const QwenSession *s){return (s->phase==0||s->phase==1)&&s->position>0&&s->pending>=0;}
static void snapshot(const QwenSession *s,QwenCheckpoint *c){
    memset(c,0,sizeof(*c));memcpy(c->model_digest,s->model->digest,65);c->position=s->position;c->pending=s->pending;c->capacity=s->capacity;c->delivered=s->delivered;c->top=s->top;c->session_id=s->session_id;c->epoch=s->epoch;
}
int qwen_abort_token(QwenSession *s){
    if(!s||s->phase!=2||!s->token_parent_valid||s->token_parent.session_id!=s->session_id||s->token_parent.epoch!=s->epoch)return -1;
    restore(s,&s->token_parent);return 0;
}
static void restore(QwenSession *s,const QwenCheckpoint *c){
    if(s->cache_rewind)s->cache_rewind(s->cache_context,c->position,s->session_id,s->epoch);
    s->token_parent_valid=0;
    s->position=c->position;s->pending=c->pending;s->delivered=c->delivered;s->top=c->top;s->phase=0;s->layer=0;
}
int qwen_checkpoint(QwenSession *s,QwenCheckpoint *c){
    if(!at_boundary(s))return -1;snapshot(s,c);c->serial=++s->checkpoint_serial;s->checkpoint_position=s->position;return 0;
}
int qwen_restore(QwenSession *s,const QwenCheckpoint *c){
    if(s->phase==2||s->depth||strcmp(c->model_digest,s->model->digest)||c->session_id!=s->session_id||c->epoch!=s->epoch||c->capacity!=s->capacity||c->serial!=s->checkpoint_serial||c->position!=s->checkpoint_position||c->position<1||c->position>s->position||c->pending<0||c->pending>=s->model->vocab)return -1;
    restore(s,c);return 0;
}
int qwen_stack_push(QwenSession *s,uint64_t *id){
    if(!id||!at_boundary(s)||s->depth>=QWEN_STACK_LIMIT)return -1;
    uint64_t unique=atomic_load(&next_frame_id);
    do{if(unique==UINT64_MAX)return -1;}while(!atomic_compare_exchange_weak(&next_frame_id,&unique,unique+1));
    QwenFrame *f=&s->frames[s->depth++];snapshot(s,&f->state);f->id=unique;*id=f->id;return 0;
}
static int valid_top(const QwenSession *s,uint64_t id){
    return s->phase!=2&&s->depth&&s->frames[s->depth-1].id==id&&s->frames[s->depth-1].state.epoch==s->epoch&&s->frames[s->depth-1].state.position<=s->position;
}
int qwen_stack_rollback(QwenSession *s,uint64_t id){
    if(!valid_top(s,id))return -1;
    restore(s,&s->frames[--s->depth].state);s->checkpoint_serial++;s->checkpoint_position=0;return 0;
}
int qwen_stack_commit(QwenSession *s,uint64_t id){
    if(!valid_top(s,id))return -1;s->depth--;return 0;
}
int qwen_reset(QwenSession *s){
    if(s->epoch==UINT64_MAX)return -1;
    s->position=0;s->pending=-1;s->delivered=0;s->phase=0;s->layer=0;s->depth=0;s->epoch++;s->checkpoint_serial++;s->checkpoint_position=0;s->token_parent_valid=0;if(s->cache_rewind)s->cache_rewind(s->cache_context,0,s->session_id,s->epoch);return 0;
}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static int integer(const char *text,int *out){char *end;errno=0;long x=strtol(text,&end,10);if(errno||end==text||*end||x<0||x>1000000)return -1;*out=(int)x;return 0;}
static int dump_logits(const QwenSession *s,const char *path){if(s->phase!=1)return -1;float *host=malloc((size_t)s->model->vocab*4);if(!host)return -1;gpu_read(host,s->logits,(size_t)s->model->vocab*4);FILE *f=fopen(path,"wb");int result=!f||fwrite(host,4,(size_t)s->model->vocab,f)!=(size_t)s->model->vocab;if(f&&fclose(f))result=1;free(host);return result?-1:0;}
typedef struct {const char *prefix,*digest;FILE *ledger;unsigned count;int failed;} Trace;
static void trace_port(void *context,int position,int layer,QwenPort port,const float *data,size_t count){
    Trace *t=context;if(t->failed)return;char path[4096];
    int n=snprintf(path,sizeof(path),"%s-%04u-%s.f32",t->prefix,t->count,qwen_port_name(port));
    if(n<0||(size_t)n>=sizeof(path)){t->failed=1;return;}
    float *host=malloc(count*4);if(!host){t->failed=1;return;}gpu_read(host,data,count*4);
    FILE *f=fopen(path,"wb");if(!f){free(host);t->failed=1;return;}
    int bad=fwrite(host,4,count,f)!=count;if(fclose(f))bad=1;free(host);if(bad){t->failed=1;return;}
    /* Paths are caller-supplied and omitted from JSON. The occurrence index and
     * fixed port name determine the raw filename, without JSON escaping hazards. */
    if(fprintf(t->ledger,"{\"model_digest\":\"%s\",\"position\":%d,\"layer\":%d,\"port\":\"%s\",\"occurrence\":%u,\"count\":%zu,\"dtype\":\"float32\"}\n",t->digest,position,layer,qwen_port_name(port),t->count,count)<0)t->failed=1;
    t->count++;
}
int qwen_main(int argc,char **argv){
    if(argc<3)return 2;const char *tokens_path=NULL,*output_path=NULL,*logits_path=NULL,*trace_prefix=NULL;int budget=32,capacity=4096,checkpoint_test=0,trace_layer=-1,trace_position=-1;
    for(int i=3;i<argc;i++){
        if(!strcmp(argv[i],"--checkpoint-test")){checkpoint_test=1;continue;}
        if(i+1>=argc)return 2;const char *value=argv[++i];
        if(!strcmp(argv[i-1],"--tokens-file"))tokens_path=value;
        else if(!strcmp(argv[i-1],"--output"))output_path=value;
        else if(!strcmp(argv[i-1],"--first-logits"))logits_path=value;
        else if(!strcmp(argv[i-1],"--new-tokens")){if(integer(value,&budget))return 2;}
        else if(!strcmp(argv[i-1],"--context")){if(integer(value,&capacity))return 2;}
        else if(!strcmp(argv[i-1],"--trace-prefix"))trace_prefix=value;
        else if(!strcmp(argv[i-1],"--trace-layer")){if(integer(value,&trace_layer))return 2;}
        else if(!strcmp(argv[i-1],"--trace-position")){if(integer(value,&trace_position))return 2;}
        else return 2;
    }
    if(!tokens_path||budget<1||budget>4096||capacity<1||capacity>32768)return 2;
    FILE *input=fopen(tokens_path,"r");if(!input){perror("tokens");return 2;}int *ids=malloc((size_t)capacity*sizeof(int));if(!ids)return 2;int count=0,id;
    while(fscanf(input,"%d",&id)==1){if(count>=capacity){fprintf(stderr,"prompt exceeds capacity\n");return 2;}ids[count++]=id;}if(!feof(input)||fclose(input)||!count||count+budget-1>capacity){fprintf(stderr,"invalid token input or insufficient context\n");return 2;}
    TcmModel bundle;QwenModel model;QwenSession session;char error[256];double start=now();
    if(tcm_load(&bundle,argv[2],error,sizeof(error))){fprintf(stderr,"%s\n",error);return 2;}
    if(qwen_model_init(&model,&bundle,error,sizeof(error))){fprintf(stderr,"%s\n",error);tcm_unload(&bundle);return 2;}
    size_t source_bytes=bundle.mapping_bytes;tcm_unload(&bundle);if(qwen_session_init(&session,&model,capacity))return 2;gpu_sync();double load=now()-start;
    Trace trace={0};
    if(trace_prefix){
        if(trace_layer<0||trace_layer>=model.layers||trace_position>=count+budget-1)return 2;
        char path[4096];int n=snprintf(path,sizeof(path),"%s.jsonl",trace_prefix);if(n<0||(size_t)n>=sizeof(path))return 2;
        trace.prefix=trace_prefix;trace.digest=model.digest;trace.ledger=fopen(path,"wb");if(!trace.ledger)return 2;
        session.observer=trace_port;session.observer_context=&trace;session.observe_layer=trace_layer;session.observe_position=trace_position<0?count-1:trace_position;
    }else if(trace_layer>=0||trace_position>=0)return 2;
    start=now();for(int i=0;i<count;i++)if(qwen_push(&session,ids[i])){fprintf(stderr,"invalid token or failed transition\n");return 2;}gpu_sync();double prefill=now()-start;
    if(logits_path&&dump_logits(&session,logits_path))return 2;
    int checkpoint_exact=1,checkpoint_tested=0;
    if(checkpoint_test&&session.position<capacity){
        QwenCheckpoint checkpoint;if(qwen_checkpoint(&session,&checkpoint))return 2;checkpoint_tested=1;if(qwen_advance(&session))return 2;int expected=session.pending;
        size_t bytes=(size_t)model.vocab*4;float *reference=malloc(bytes),*actual=malloc(bytes);if(!reference||!actual)return 2;gpu_read(reference,session.logits,bytes);
        if(qwen_restore(&session,&checkpoint)||qwen_advance(&session))return 2;gpu_read(actual,session.logits,bytes);checkpoint_exact=expected==session.pending&&!memcmp(reference,actual,bytes);free(reference);free(actual);if(qwen_restore(&session,&checkpoint))return 2;
    }
    FILE *output=output_path?fopen(output_path,"w"):stdout;if(!output)return 2;
    fprintf(output,"{\"schema\":\"tcm-qwen-run-v1\",\"model_identity\":\"%s\",\"model_digest\":\"%s\",\"arithmetic\":\"FP32; BF16 weights widened once; cuBLAS pedantic; CUDA contraction disabled\",\"prompt_tokens\":%d,\"capacity\":%d,\"source_bundle_bytes\":%zu,\"gpu_weight_bytes\":%zu,\"gpu_session_bytes\":%zu,\"load_seconds\":%.9g,\"prefill_seconds\":%.9g,\"tokens\":[",model.identity,model.digest,count,capacity,source_bytes,model.weight_bytes,session.allocated_bytes,load,prefill);
    int generated=0;start=now();
    for(int i=0;i<budget;i++){
        if(i)fputc(',',output);fprintf(output,"{\"id\":%d,\"position\":%d,\"logit\":%.9g,\"runner_up\":%d,\"margin\":%.9g}",session.pending,session.position,session.top.first_value,session.top.second,session.top.first_value-session.top.second_value);generated++;session.delivered=1;
        if(session.pending==model.eos||(model.eos_extra>0&&session.pending==model.eos_extra)||i+1==budget)break;if(qwen_advance(&session))return 2;
    }
    gpu_sync();double generation=now()-start;
    fprintf(output,"],\"generated_tokens\":%d,\"generation_forward_passes\":%d,\"kv_rows\":%d,\"pending_token\":%d,\"generation_seconds\":%.9g,\"checkpoint_tested\":%s,\"checkpoint_exact\":%s,\"trace_ports\":%u}\n",generated,generated-1,session.position,session.pending,generation,checkpoint_tested?"true":"false",checkpoint_tested?(checkpoint_exact?"true":"false"):"null",trace.count);
    if(output_path&&fclose(output))return 2;
    if(trace.ledger&&fclose(trace.ledger))trace.failed=1;
    free(ids);qwen_session_free(&session);qwen_model_free(&model);return trace.failed?2:checkpoint_exact?0:1;
}
