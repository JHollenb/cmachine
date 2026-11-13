#include "qwen_circuit.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

static int native_profile(void){
    return gpu_weight_profile()==0;
}
static int vector_read(const float *src,double *out,size_t n){
    if(!src||!out||!n||n>SIZE_MAX/sizeof(float))return -1;
    float *buf=malloc(n*sizeof(float));if(!buf)return -1;
    gpu_read(buf,src,n*sizeof(float));int error=0;
    for(size_t i=0;i<n;i++){out[i]=buf[i];if(!isfinite(out[i]))error=-1;}free(buf);return error;
}
static int live(const CmQwenCapture *c){
    const QwenSession *s=c?c->session:NULL;
    return c&&c->ready&&s&&s->model==c->model&&s->session_id==c->session_id&&s->epoch==c->epoch&&s->lineage_id==c->result_lineage_id&&s->position==c->position+1&&s->phase==1&&native_profile();
}
static int compatible(const QwenSession *s){
    if(!s||!s->model||s->observer||s->cache_mask||s->lexical_program||(s->phase!=0&&s->phase!=1)||!native_profile())return 0;
    const QwenModel *m=s->model;
    if(m->layers<1||m->layers>64||m->hidden<1||m->intermediate<1||m->head<2||m->head%2||m->qheads<1||m->kvheads<1||m->qheads%m->kvheads||m->qheads>INT_MAX/m->head||m->qheads*m->head!=m->hidden||m->vocab<1)return 0;
    for(int i=0;i<m->layers;i++)if(s->qkv_programs[i]||s->cache_slots[i]<0)return 0;
    return 1;
}
void cm_qwen_capture_free(CmQwenCapture *c){
    if(!c)return;
    if(c->layer)for(int i=0;i<c->model->layers;i++){
        CmQwenLayer *l=c->layer+i;
        free(l->input);free(l->receiver);free(l->normalized);free(l->gate);free(l->up);free(l->product);free(l->attention);free(l->output);free(l->query);free(l->key);free(l->value);free(l->probability);free(l->write);
    }free(c->layer);free(c->transfer);memset(c,0,sizeof(*c));
}
static double **port_slot(CmQwenLayer *l,QwenPort p){
    switch(p){
    case QWEN_INPUT:return &l->input;
    case QWEN_RAW_RECEIVER:return &l->receiver;
    case QWEN_MLP_INPUT:return &l->normalized;
    case QWEN_MLP_GATE:return &l->gate;
    case QWEN_MLP_UP:return &l->up;
    case QWEN_MLP_PRODUCT:return &l->product;
    case QWEN_ATTENTION:return &l->attention;
    case QWEN_OUTPUT:return &l->output;
    case QWEN_QUERY:return &l->query;
    case QWEN_KEY:return &l->key;
    case QWEN_VALUE:return &l->value;
    case QWEN_ATTENTION_PROBABILITIES:return &l->probability;
    case QWEN_CONSUMED_MLP_WRITE:return &l->write;
    default:return NULL;
    }
}
static void capture_port(void *ctx,int position,int layer,QwenPort port,const float *data,size_t n){
    CmQwenCapture *c=ctx;
    if(c->error)return;
    if(position!=c->position||layer<0||layer>=c->model->layers){c->error=-1;return;}
    CmQwenLayer *l=c->layer+layer;double **dst=port_slot(l,port);if(!dst)return;
    if(n>c->transfer_count||*dst){c->error=-1;return;}
    *dst=malloc(n*sizeof(double));if(!*dst){c->error=-1;return;}
    c->host_bytes+=n*sizeof(double);gpu_read(c->transfer,data,n*sizeof(float));
    for(size_t i=0;i<n;i++){(*dst)[i]=c->transfer[i];if(!isfinite((*dst)[i]))c->error=-1;}
    l->seen|=1u<<port;
}
int cm_qwen_capture_token(CmQwenCapture *c,QwenSession *s,int token){
    if(!c||c->layer||c->transfer||!compatible(s)||token<0||token>=s->model->vocab||s->position>=s->capacity)return -1;
    const QwenModel *m=s->model;
    memset(c,0,sizeof(*c));c->model=m;c->session=s;c->session_id=s->session_id;c->epoch=s->epoch;c->parent_lineage_id=s->lineage_id;c->position=s->position;c->token=token;
    c->transfer_count=(size_t)m->qheads*(s->position+1);
    if(c->transfer_count<(size_t)m->hidden)c->transfer_count=(size_t)m->hidden;
    if(c->transfer_count<(size_t)m->intermediate)c->transfer_count=(size_t)m->intermediate;
    c->layer=calloc((size_t)m->layers,sizeof(*c->layer));c->transfer=malloc(c->transfer_count*sizeof(float));
    if(!c->layer||!c->transfer){cm_qwen_capture_free(c);return -1;}
    c->host_bytes=(size_t)m->layers*sizeof(*c->layer)+c->transfer_count*sizeof(float);
    int old_position=s->observe_position,old_layer=s->observe_layer;void *old_context=s->observer_context;
    s->observer=capture_port;s->observer_context=c;s->observe_position=s->position;s->observe_layer=-1;
    int error=qwen_begin_token(s,token);
    while(!error&&!c->error&&s->layer<m->layers)error=qwen_advance_layer(s);
    if(!error&&!c->error)error=qwen_finish_token(s);
    s->observer=NULL;s->observer_context=old_context;s->observe_position=old_position;s->observe_layer=old_layer;
    if(error||c->error){if(s->phase==2)(void)qwen_abort_token(s);cm_qwen_capture_free(c);return -2;}
    unsigned required=(1u<<QWEN_INPUT)|(1u<<QWEN_RAW_RECEIVER)|(1u<<QWEN_MLP_INPUT)|(1u<<QWEN_MLP_GATE)|(1u<<QWEN_MLP_UP)|(1u<<QWEN_MLP_PRODUCT)|(1u<<QWEN_ATTENTION)|(1u<<QWEN_OUTPUT)|(1u<<QWEN_QUERY)|(1u<<QWEN_KEY)|(1u<<QWEN_VALUE)|(1u<<QWEN_ATTENTION_PROBABILITIES)|(1u<<QWEN_CONSUMED_MLP_WRITE);
    for(int i=0;i<m->layers;i++)if(c->layer[i].seen!=required){cm_qwen_capture_free(c);return -2;}
    c->result_lineage_id=s->lineage_id;c->ready=1;return 0;
}
static const float *parameter(const QwenModel *m,int layer,const char *name,size_t *rows,size_t *cols){
    if(!m||!name||layer<0||layer>=m->layers)return NULL;
    const QwenLayer *l=m->layer+layer;*rows=(size_t)m->hidden;*cols=(size_t)m->hidden;
    if(!strcmp(name,"gate")){*rows=m->intermediate;return l->gate;}
    if(!strcmp(name,"up")){*rows=m->intermediate;return l->up;}
    if(!strcmp(name,"down")){*cols=m->intermediate;return l->down;}
    if(!strcmp(name,"q"))return l->q;
    if(!strcmp(name,"k")){*rows=(size_t)m->kvheads*m->head;return l->k;}
    if(!strcmp(name,"v")){*rows=(size_t)m->kvheads*m->head;return l->v;}
    if(!strcmp(name,"o"))return l->o;
    if(!strcmp(name,"input_norm")){*rows=1;return l->input_norm;}
    if(!strcmp(name,"post_norm")){*rows=1;return l->post_norm;}
    return NULL;
}
int cm_qwen_parameter(const QwenModel *m,int layer,const char *name,size_t index,int column,double *out,size_t count){
    size_t rows,cols;const float *p=parameter(m,layer,name,&rows,&cols);
    if(!p||!native_profile()||!out||(column!=0&&column!=1))return -1;
    if(!column){if(index>=rows||count!=cols)return -1;return vector_read(p+index*cols,out,cols);}
    if(index>=cols||count!=rows)return -1;
    for(size_t i=0;i<rows;i++)if(vector_read(p+i*cols+index,out+i,1))return -1;
    return 0;
}
int cm_qwen_writer_reader(const QwenModel *m,int writer_layer,size_t neuron,int reader_layer,const char *reader,size_t row,const double *r,double *fixed,double *radial,double *total){
    if(!m||!r||!reader)return -1;
    size_t d=(size_t)m->hidden;double *buf=malloc(3*d*sizeof(double));if(!buf)return -1;
    double *w=buf,*v=buf+d,*g=buf+2*d;
    int error=cm_qwen_parameter(m,writer_layer,"down",neuron,1,v,d)||cm_qwen_parameter(m,reader_layer,reader,row,0,w,d);
    const char *norm=(!strcmp(reader,"gate")||!strcmp(reader,"up"))?"post_norm":"input_norm";
    if(strcmp(reader,"gate")&&strcmp(reader,"up")&&strcmp(reader,"q")&&strcmp(reader,"k")&&strcmp(reader,"v"))error=1;
    if(!error)error=cm_qwen_parameter(m,reader_layer,norm,0,0,g,d)||cm_writer_reader(d,w,v,r,g,m->epsilon,fixed,radial,total);
    free(buf);return error?-1:0;
}
/* Stream original parameter rows; no duplicate matrix bank is retained. */
int cm_qwen_static_link(const QwenModel *m,int writer_layer,size_t neuron,int reader_layer,const char *reader,size_t row,double *out){
    if(!m||!reader||!out||m->hidden<1)return -1;
    const char *norm=(!strcmp(reader,"gate")||!strcmp(reader,"up"))?"post_norm":"input_norm";
    if(strcmp(reader,"gate")&&strcmp(reader,"up")&&strcmp(reader,"q")&&strcmp(reader,"k")&&strcmp(reader,"v"))return -1;
    size_t d=m->hidden;double *buf=malloc(3*d*sizeof(double));if(!buf)return -1;
    int error=cm_qwen_parameter(m,writer_layer,"down",neuron,1,buf,d)||cm_qwen_parameter(m,reader_layer,reader,row,0,buf+d,d)||cm_qwen_parameter(m,reader_layer,norm,0,0,buf+2*d,d);
    double value=0;if(!error)for(size_t i=0;i<d;i++)value+=buf[i]*buf[d+i]*buf[2*d+i];
    free(buf);if(error||!isfinite(value))return -1;
    *out=value;return 0;
}
int cm_qwen_read_geometry(const QwenModel *m,int layer,size_t neuron,CmReadGeometry *out){
    if(!m||!out||m->hidden<1)return -1;
    size_t d=m->hidden;double *buf=malloc(2*d*sizeof(double));if(!buf)return -1;
    int error=cm_qwen_parameter(m,layer,"gate",neuron,0,buf,d)||cm_qwen_parameter(m,layer,"up",neuron,0,buf+d,d)||cm_read_geometry(d,buf,buf+d,out);
    free(buf);return error?-1:0;
}
static int transpose(const float *w,size_t rows,size_t cols,const double *in,double *out){
    double *row=malloc(cols*sizeof(double));if(!row)return -1;
    memset(out,0,cols*sizeof(double));int error=0;
    for(size_t j=0;j<rows&&!error;j++){
        error=vector_read(w+j*cols,row,cols);
        if(!error)for(size_t i=0;i<cols;i++)out[i]+=row[i]*in[j];
    }free(row);return error;
}
static void unrotate(double *v,int heads,int dim,int position,float theta){
    for(int h=0;h<heads;h++)for(int i=0;i<dim/2;i++){
        float inverse=1.0f/powf(theta,(2.0f*i)/dim),angle=(float)position*inverse;
        double c=cosf(angle),s=sinf(angle),a=v[h*dim+i],b=v[h*dim+i+dim/2];
        v[h*dim+i]=a*c+b*s;v[h*dim+i+dim/2]=b*c-a*s;
    }
}
void cm_qwen_adjoint_free(CmQwenAdjoint *a){
    if(!a)return;
    if(a->layer)for(int i=0;i<a->capture->model->layers;i++){
        CmQwenLayerAdjoint *l=a->layer+i;free(l->input);free(l->receiver);free(l->attention);free(l->product);free(l->gate);free(l->up);free(l->routing);free(l->output);
    }free(a->layer);memset(a,0,sizeof(*a));
}
int cm_qwen_consumer(CmQwenAdjoint *a,const CmQwenCapture *c,int token,int contrast){
    if(!a||a->layer||!live(c)||token<0||token>=c->model->vocab||contrast< -1||contrast>=c->model->vocab)return -1;
    const QwenModel *m=c->model;const QwenSession *s=c->session;
    size_t d=m->hidden,h=m->intermediate,n=(size_t)c->position+1,hd=m->head,k=(size_t)m->kvheads*hd;
    a->capture=c;a->token=token;a->contrast=contrast;a->layer=calloc((size_t)m->layers,sizeof(*a->layer));
    /* scratch: full residual vectors, positioned current Q/K/V gradients, one
     * head's historical K/V and their temporary adjoints, streamed norm/head. */
    size_t words=10*d+3*k+4*n*hd+n;
    double *work=calloc(words,sizeof(double));if(!a->layer||!work)goto fail;
    a->host_bytes=(size_t)m->layers*sizeof(*a->layer);
    double *lambda=work,*gamma=lambda+d,*tmp=gamma+d,*tmp2=tmp+d,*carry=tmp2+d,*qgrad=carry+d,*kgrad=qgrad+d,*vgrad=kgrad+k,*normgrad=vgrad+k,*xgrad=normgrad+d,*headrow=xgrad+d,*other=headrow+d;
    double *keys=other+d,*values=keys+n*hd,*dk=values+n*hd,*dv=dk+n*hd,*dl=dv+n*hd;
    if(vector_read(m->embedding+(size_t)token*d,headrow,d))goto fail;
    if(contrast>=0){if(vector_read(m->embedding+(size_t)contrast*d,other,d))goto fail;for(size_t i=0;i<d;i++)headrow[i]-=other[i];}
    if(vector_read(m->norm,gamma,d)||cm_rms_vjp(d,c->layer[m->layers-1].output,gamma,m->epsilon,headrow,lambda))goto fail;
    float logits[2];gpu_read(logits,s->logits+token,sizeof(float));a->native_margin=logits[0];
    if(contrast>=0){gpu_read(logits+1,s->logits+contrast,sizeof(float));a->native_margin-=logits[1];}
    if(!isfinite(a->native_margin))goto fail;
    for(int layer=m->layers-1;layer>=0;layer--){
        const QwenLayer *w=m->layer+layer;const CmQwenLayer *p=c->layer+layer;CmQwenLayerAdjoint *r=a->layer+layer;
#define ALLOC(field,count) r->field=calloc((count),sizeof(double));if(!r->field)goto fail;a->host_bytes+=(count)*sizeof(double)
        ALLOC(input,d);ALLOC(receiver,d);ALLOC(attention,d);ALLOC(product,h);ALLOC(gate,h);ALLOC(up,h);ALLOC(routing,(size_t)m->qheads*n);ALLOC(output,d);
#undef ALLOC
        memcpy(r->output,lambda,d*sizeof(double));
        if(transpose(w->down,d,h,lambda,r->product))goto fail;
        for(size_t i=0;i<h;i++){r->gate[i]=r->product[i]*p->up[i]*cm_silu_first(p->gate[i]);r->up[i]=r->product[i]*cm_silu(p->gate[i]);}
        if(transpose(w->gate,h,d,r->gate,tmp)||transpose(w->up,h,d,r->up,tmp2))goto fail;
        for(size_t i=0;i<d;i++)tmp[i]+=tmp2[i];
        if(vector_read(w->post_norm,gamma,d)||cm_rms_vjp(d,p->receiver,gamma,m->epsilon,tmp,carry))goto fail;
        for(size_t i=0;i<d;i++)r->receiver[i]=lambda[i]+carry[i];
        if(transpose(w->o,d,d,r->receiver,r->attention))goto fail;
        memset(qgrad,0,d*sizeof(double));memset(kgrad,0,k*sizeof(double));memset(vgrad,0,k*sizeof(double));
        size_t base=(size_t)s->cache_slots[layer]*m->kvheads*s->capacity*hd;
        for(int head=0;head<m->qheads;head++){
            int group=head/(m->qheads/m->kvheads);size_t offset=base+(size_t)group*s->capacity*hd;
            if(vector_read(s->keys+offset,keys,n*hd)||vector_read(s->values+offset,values,n*hd))goto fail;
            if(cm_attention_vjp(n,hd,p->probability+(size_t)head*n,p->query+(size_t)head*hd,keys,values,p->attention+(size_t)head*hd,r->attention+(size_t)head*hd,qgrad+(size_t)head*hd,dk,dv,dl))goto fail;
            memcpy(r->routing+(size_t)head*n,dl,n*sizeof(double));
            for(size_t i=0;i<hd;i++){kgrad[(size_t)group*hd+i]+=dk[(n-1)*hd+i];vgrad[(size_t)group*hd+i]+=dv[(n-1)*hd+i];}
        }
        unrotate(qgrad,m->qheads,m->head,c->position,m->theta);unrotate(kgrad,m->kvheads,m->head,c->position,m->theta);
        if(transpose(w->q,d,d,qgrad,normgrad)||transpose(w->k,k,d,kgrad,tmp)||transpose(w->v,k,d,vgrad,tmp2))goto fail;
        for(size_t i=0;i<d;i++)normgrad[i]+=tmp[i]+tmp2[i];
        if(vector_read(w->input_norm,gamma,d)||cm_rms_vjp(d,p->input,gamma,m->epsilon,normgrad,xgrad))goto fail;
        for(size_t i=0;i<d;i++)lambda[i]=r->input[i]=r->receiver[i]+xgrad[i];
    }
    for(int layer=0;layer<m->layers;layer++)for(size_t i=0;i<d;i++)if(!isfinite(a->layer[layer].input[i]))goto fail;
    free(work);a->ready=1;return 0;
fail:
    free(work);cm_qwen_adjoint_free(a);return -2;
}
static int paired(const CmQwenCapture *a,const CmQwenCapture *b,int layer){
    return live(a)&&live(b)&&a->model==b->model&&a->position==b->position&&layer>=0&&layer<a->model->layers;
}
int cm_qwen_mlp_finite(const CmQwenCapture *c0,const CmQwenCapture *c1,int layer,const double *lambda,double *out,CmQwenFiniteSummary *summary){
    if(!out||!summary||!paired(c0,c1,layer))return -1;
    const QwenModel *m=c0->model;size_t d=m->hidden,h=m->intermediate;
    const CmQwenLayer *a=c0->layer+layer,*b=c1->layer+layer;
    double *buf=calloc(4*h+3*d,sizeof(double));if(!buf)return -1;
    double *gate=buf,*up=gate+h,*interaction=up+h,*weight=interaction+h,*gamma=weight+h,*dn=gamma+d,*row=dn+d;
    CmQwenFiniteSummary s={0};
    for(size_t j=0;j<h;j++){
        CmNeuronDelta delta;if(cm_neuron_delta(a->gate[j],a->up[j],a->product[j],b->gate[j],b->up[j],b->product[j],&delta))goto fail;
        gate[j]=delta.gate;up[j]=delta.up;interaction[j]=delta.interaction;
    }
    if(vector_read(m->layer[layer].post_norm,gamma,d)||cm_rms_finite(d,a->receiver,b->receiver,gamma,m->epsilon,dn))goto fail;
    for(size_t i=0;i<d;i++)s.normalization_remainder=fmax(s.normalization_remainder,fabs((b->normalized[i]-a->normalized[i])-dn[i]));
    for(size_t j=0;j<h;j++){
        if(cm_qwen_parameter(m,layer,"gate",j,0,row,d))goto fail;
        double dg=0,du=0;for(size_t i=0;i<d;i++)dg+=row[i]*(b->normalized[i]-a->normalized[i]);
        if(cm_qwen_parameter(m,layer,"up",j,0,row,d))goto fail;
        for(size_t i=0;i<d;i++)du+=row[i]*(b->normalized[i]-a->normalized[i]);
        s.projection_remainder=fmax(s.projection_remainder,fabs(dg-(b->gate[j]-a->gate[j])));
        s.projection_remainder=fmax(s.projection_remainder,fabs(du-(b->up[j]-a->up[j])));
        double secant=.5*(a->up[j]+b->up[j])*cm_silu_secant(a->gate[j],b->gate[j])*dg+.5*(cm_silu(a->gate[j])+cm_silu(b->gate[j]))*du;
        s.secant_remainder=fmax(s.secant_remainder,fabs((b->product[j]-a->product[j])-secant));
    }
    memset(out,0,4*d*sizeof(double));
    for(size_t i=0;i<d;i++){
        if(vector_read(m->layer[layer].down+i*h,weight,h))goto fail;
        for(size_t j=0;j<h;j++){out[i]+=weight[j]*gate[j];out[d+i]+=weight[j]*up[j];out[2*d+i]+=weight[j]*interaction[j];}
        /* The remainder includes changed native product/down arithmetic.
         * Residual-add rounding is outside this declared write consumer. */
        double total=b->write[i]-a->write[i];out[3*d+i]=total-out[i]-out[d+i]-out[2*d+i];
        double l=lambda?lambda[i]:1;
        if(!isfinite(l)||!isfinite(total))goto fail;
        s.gate+=l*out[i];s.up+=l*out[d+i];s.interaction+=l*out[2*d+i];s.remainder+=l*out[3*d+i];s.total+=l*total;
    }
    free(buf);*summary=s;return 0;
fail:free(buf);return -2;
}
int cm_qwen_mlp_secant_action(const CmQwenCapture *c0,const CmQwenCapture *c1,int layer,const double *direction,double *out){
    if(!direction||!out||!paired(c0,c1,layer))return -1;
    const QwenModel *m=c0->model;size_t d=m->hidden,h=m->intermediate;
    double *buf=malloc((2*h+d)*sizeof(double));if(!buf)return -1;
    double *product=buf,*weight=product+h,*row=weight+h;
    const CmQwenLayer *a=c0->layer+layer,*b=c1->layer+layer;
    for(size_t i=0;i<d;i++)if(!isfinite(direction[i]))goto fail;
    for(size_t j=0;j<h;j++){
        double dg=0,du=0;
        if(cm_qwen_parameter(m,layer,"gate",j,0,row,d))goto fail;
        for(size_t i=0;i<d;i++)dg+=row[i]*direction[i];
        if(cm_qwen_parameter(m,layer,"up",j,0,row,d))goto fail;
        for(size_t i=0;i<d;i++)du+=row[i]*direction[i];
        product[j]=.5*(a->up[j]+b->up[j])*cm_silu_secant(a->gate[j],b->gate[j])*dg+.5*(cm_silu(a->gate[j])+cm_silu(b->gate[j]))*du;
    }
    for(size_t i=0;i<d;i++){
        if(vector_read(m->layer[layer].down+i*h,weight,h))goto fail;
        out[i]=0;for(size_t j=0;j<h;j++)out[i]+=weight[j]*product[j];if(!isfinite(out[i]))goto fail;
    }free(buf);return 0;
fail:free(buf);return -2;
}
int cm_qwen_attention_finite(const CmQwenCapture *c0,const CmQwenCapture *c1,int layer,int head,double *out){
    if(!out||!paired(c0,c1,layer)||head<0||head>=c0->model->qheads)return -1;
    const QwenModel *m=c0->model;size_t d=m->head,n=(size_t)c0->position+1;
    double *values=malloc(2*n*d*sizeof(double));if(!values)return -1;
    int group=head/(m->qheads/m->kvheads);const CmQwenCapture *cs[]={c0,c1};
    for(int arm=0;arm<2;arm++){
        const QwenSession *s=cs[arm]->session;size_t offset=((size_t)s->cache_slots[layer]*m->kvheads+group)*s->capacity*d;
        if(vector_read(s->values+offset,values+(size_t)arm*n*d,n*d)){free(values);return -2;}
    }
    const double *p0=c0->layer[layer].probability+(size_t)head*n,*p1=c1->layer[layer].probability+(size_t)head*n;
    memset(out,0,4*d*sizeof(double));
    for(size_t j=0;j<n;j++)for(size_t i=0;i<d;i++){
        double dp=p1[j]-p0[j],dv=values[n*d+j*d+i]-values[j*d+i];
        out[i]+=dp*values[j*d+i];out[d+i]+=p0[j]*dv;out[2*d+i]+=dp*dv;
    }
    for(size_t i=0;i<d;i++)out[3*d+i]=c1->layer[layer].attention[(size_t)head*d+i]-c0->layer[layer].attention[(size_t)head*d+i]-out[i]-out[d+i]-out[2*d+i];
    free(values);return 0;
}
int cm_qwen_source_frequency(const CmQwenCapture *c,const CmQwenAdjoint *a,int layer,int head,CmComplex *out,size_t count,double *native,double *ideal,double *remainder){
    if(!live(c)||!a||!a->ready||a->capture!=c||!out||!native||!ideal||!remainder||layer<0||layer>=c->model->layers||head<0||head>=c->model->qheads||count!=(size_t)c->position+1||count>4096)return -1;
    size_t n=count,d=c->model->head;CmComplex *buf=calloc(4*n,sizeof(*buf));double *values=malloc(n*d*sizeof(double));
    if(!buf||!values){free(buf);free(values);return -2;}
    CmComplex *p=buf,*v=p+n,*pf=v+n,*vf=pf+n;
    int group=head/(c->model->qheads/c->model->kvheads);const QwenSession *s=c->session;
    size_t offset=((size_t)s->cache_slots[layer]*c->model->kvheads+group)*s->capacity*d;
    if(vector_read(s->values+offset,values,n*d))goto fail;
    *native=0;*ideal=0;
    for(size_t i=0;i<d;i++)*native+=a->layer[layer].attention[(size_t)head*d+i]*c->layer[layer].attention[(size_t)head*d+i];
    for(size_t j=0;j<n;j++){
        p[j].re=c->layer[layer].probability[(size_t)head*n+j];
        for(size_t i=0;i<d;i++)v[j].re+=a->layer[layer].attention[(size_t)head*d+i]*values[j*d+i];
        *ideal+=p[j].re*v[j].re;
    }
    if(cm_dft(n,p,pf,0)||cm_dft(n,v,vf,0))goto fail;
    for(size_t k=0;k<n;k++)out[k]=(CmComplex){(pf[k].re*vf[k].re+pf[k].im*vf[k].im)/n,(pf[k].re*vf[k].im-pf[k].im*vf[k].re)/n};
    *remainder=*native-*ideal;free(buf);free(values);return 0;
fail:free(buf);free(values);return -2;
}
int cm_qwen_report(FILE *f,const CmQwenCapture *c,const CmQwenAdjoint *a,int layer,int head,int source,int neuron){
    if(!f||!live(c)||!a||!a->ready||a->capture!=c||layer<0||layer>=c->model->layers||head<0||head>=c->model->qheads||source<0||source>c->position||neuron<0||neuron>=c->model->intermediate)return -1;
    const QwenModel *m=c->model;const CmQwenLayer *p=c->layer+layer;const CmQwenLayerAdjoint *r=a->layer+layer;
    /* Only validated hex identity enters JSON unescaped; arbitrary free-form
     * names are not accepted as JSON fragments. */
    for(int i=0;i<64;i++)if(!((m->digest[i]>='0'&&m->digest[i]<='9')||(m->digest[i]>='a'&&m->digest[i]<='f')))return -1;
    size_t index=(size_t)head*(c->position+1)+source;
    int group=head/(m->qheads/m->kvheads),d=m->head;
    double *value=malloc((size_t)d*sizeof(double));if(!value)return -2;
    size_t offset=(((size_t)c->session->cache_slots[layer]*m->kvheads+group)*c->session->capacity+source)*d;
    if(vector_read(c->session->values+offset,value,(size_t)d)){free(value);return -2;}
    double source_dot=0,return_dot=0,payload_norm=0,row_sum=0,gauge_residual=0;
    for(int i=0;i<d;i++){
        double lambda=r->attention[head*d+i];source_dot+=lambda*value[i];return_dot+=lambda*p->attention[head*d+i];payload_norm=hypot(payload_norm,p->probability[index]*lambda);
    }free(value);
    for(int j=0;j<=c->position;j++){row_sum+=p->probability[(size_t)head*(c->position+1)+j];gauge_residual+=r->routing[(size_t)head*(c->position+1)+j];}
    fprintf(f,"{\"schema\":\"tcm-circuit-inspection-v1\",\"model_digest\":\"%.64s\",\"session_id\":%llu,\"epoch\":%llu,\"parent_lineage\":%llu,\"result_lineage\":%llu,\"position\":%d,\"consumed_token\":%d,",m->digest,(unsigned long long)c->session_id,(unsigned long long)c->epoch,(unsigned long long)c->parent_lineage_id,(unsigned long long)c->result_lineage_id,c->position,c->token);
    fprintf(f,"\"consumer\":{\"token\":%d,\"contrast\":%d,\"native_margin\":%.17g},\"layer\":%d,\"head\":%d,\"kv_group\":%d,\"source\":%d,",a->token,a->contrast,a->native_margin,layer,head,group,source);
    fprintf(f,"\"probability\":%.17g,\"routing_partial\":%.17g,\"source_value_dot\":%.17g,\"native_return_dot\":%.17g,\"centered_value_dot\":%.17g,\"payload_gradient_norm\":%.17g,\"probability_sum\":%.17g,\"routing_gauge_rounding_residual\":%.17g,",p->probability[index],r->routing[index],source_dot,return_dot,source_dot-return_dot,payload_norm,row_sum,gauge_residual);
    fprintf(f,"\"neuron\":%d,\"gate\":%.17g,\"up\":%.17g,\"native_product\":%.17g,\"product_partial\":%.17g,\"parameter_refs\":[\"model.layers.%d.mlp.gate_proj.weight[%d,:]\",\"model.layers.%d.mlp.up_proj.weight[%d,:]\",\"model.layers.%d.mlp.down_proj.weight[:,%d]\"],",neuron,p->gate[neuron],p->up[neuron],p->product[neuron],r->product[neuron],layer,neuron,layer,neuron,layer,neuron);
    fprintf(f,"\"attention_parameter_refs\":[\"model.layers.%d.self_attn.q_proj.weight[%d:%d,:]\",\"model.layers.%d.self_attn.k_proj.weight[%d:%d,:]\",\"model.layers.%d.self_attn.v_proj.weight[%d:%d,:]\",\"model.layers.%d.self_attn.o_proj.weight[:,%d:%d]\"],",layer,head*d,(head+1)*d,layer,group*d,(group+1)*d,layer,group*d,(group+1)*d,layer,head*d,(head+1)*d);
    fputs("\"numerical_profile\":\"native-FP32-ports/FP64-smooth-linearization\",\"derivative_scope\":\"fixed-prior-KV-first-readout\",\"mechanics_status\":\"captured\",\"terminal_status\":\"no-semantic-certification\"}\n",f);
    return ferror(f)?-1:0;
}
