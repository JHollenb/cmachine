/* Model-free ledger mechanics (src/qwen_ledger.h) on the CPU stub.
 * (a) ledger on/off bit identity, (b) every ledger field recomputed in double
 * from observer captures, (c) margin closure, (d) lifecycle/refusal cases.
 * This qualifies the lean-path wiring and the stub reductions, not CUDA. */
#include "qwen.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef NDEBUG
#error "The ledger assay requires assertions"
#endif

/* Same generator and shapes as the tests/state_mechanics.c fixture. gate_gain
 * scales layer 2's gate matrix so the census also sees |g| >= 5. */
static float *parameters(int n,int norm){
    static int index=0;float *p=malloc((size_t)n*4);assert(p);
    for(int i=0;i<n;i++)p[i]=norm?.6f+(i%7)*.13f:(float)((++index*17)%29-14)/70;return p;
}
static void fixture(QwenModel *m,float gate_gain,int tied){
    memset(m,0,sizeof(*m));m->gpu=gpu_open();m->hidden=8;m->layers=3;m->qheads=4;m->kvheads=2;m->head=2;m->intermediate=12;m->vocab=16;m->max_position=32;m->epsilon=1e-6f;m->theta=1e6f;memset(m->digest,'a',64);
    m->embedding=parameters(16*8,0);m->norm=parameters(8,1);m->layer=calloc(3,sizeof(QwenLayer));assert(m->layer);
    for(int i=0;i<3;i++){QwenLayer *l=m->layer+i;
        l->input_norm=parameters(8,1);l->post_norm=parameters(8,1);l->q=parameters(64,0);l->k=parameters(32,0);l->v=parameters(32,0);l->qb=parameters(8,0);l->kb=parameters(4,0);l->vb=parameters(4,0);l->o=parameters(64,0);l->gate=parameters(96,0);l->up=parameters(96,0);l->down=parameters(96,0);
    }
    for(int j=0;j<96;j++)m->layer[2].gate[j]*=gate_gain;
    /* tied: zero layer-0 queries, so every layer-0 score is exactly 0 and all
     * native probabilities tie (exercises the smallest-index argmax rule). */
    if(tied){memset(m->layer[0].q,0,64*4);memset(m->layer[0].qb,0,8*4);}
}
static void fixture_free(QwenModel *m){
    for(int i=0;i<3;i++){QwenLayer *l=m->layer+i;free(l->input_norm);free(l->post_norm);free(l->q);free(l->k);free(l->v);free(l->qb);free(l->kb);free(l->vb);free(l->o);free(l->gate);free(l->up);free(l->down);}free(m->layer);free(m->embedding);free(m->norm);gpu_close(m->gpu);
}

typedef struct { int layers; float *port[8][QWEN_PORT_COUNT]; size_t count[8][QWEN_PORT_COUNT]; } Capture;
static void capture(void *context,int position,int layer,QwenPort port,const float *data,size_t count){
    Capture *c=context;(void)position;assert(layer>=0&&layer<c->layers&&layer<8&&port>=0&&port<QWEN_PORT_COUNT);
    free(c->port[layer][port]);c->port[layer][port]=malloc(count*4);assert(c->port[layer][port]);gpu_read(c->port[layer][port],data,count*4);c->count[layer][port]=count;
}
static void capture_free(Capture *c){for(int l=0;l<8;l++)for(int p=0;p<QWEN_PORT_COUNT;p++){free(c->port[l][p]);c->port[l][p]=NULL;}}
static const float *port(const Capture *c,int layer,QwenPort p,size_t n){assert(c->port[layer][p]&&c->count[layer][p]==n);return c->port[layer][p];}

/* Measured maxima, reported in the summary line. */
typedef struct {
    double rho_rel,logz_abs,logz_scores_abs,entropy_rel,entropy_scores_abs,dot_rel,norm2_rel,gate_rel,prod2_rel,even_odd_vs_prod_rel;
    double margin_term_rel,c_norm_rel,margin_sum_rel,closure_abs,closure_rel,closure_double_abs,mean_square_identity_rel;
    int rows,gate_off,gate_on,gate_small;
} Stats;
static double fmaxd(double a,double b){return a>b?a:b;}
static double dot(const float *a,const float *b,int n){double s=0;for(int i=0;i<n;i++)s+=(double)a[i]*(double)b[i];return s;}
static double rel(double got,double ref,double scale){return fabs(got-ref)/fmaxd(scale,1e-300);}

/* Tolerances (stated): float storage of a double reduction is <= 2^-24 relative;
 * every bound below leaves >10x headroom over that, or is exact. */
#define TOL_REL 1e-6
#define TOL_LOGZ_ABS 1e-5     /* stub scores are float dots; the reference is double */

static void check_row(const QwenSession *s,const Capture *c,const float *row,int token,int position,Stats *st){
    const QwenModel *m=s->model;int d=m->hidden,H=m->qheads,hd=m->head,L=m->layers,h=m->intermediate,n=position+1,group=H/m->kvheads;
    assert(row[QL_TOKEN]==(float)token&&row[QL_POSITION]==(float)position&&row[QL_SOURCES]==(float)n&&row[QL_HAS_READOUT]==1);
    assert(row[QL_TOP1]==(float)s->top.first&&row[QL_TOP2]==(float)s->top.second);
    float *logits=malloc((size_t)m->vocab*4);assert(logits);gpu_read(logits,s->logits,(size_t)m->vocab*4);
    int a=s->top.first,b=s->top.second;assert(row[QL_MARGIN]==(float)((double)logits[a]-(double)logits[b]));free(logits);
    for(int j=QL_MARGIN_SUM+1;j<QL_HEAD;j++)assert(row[j]==0);
    double *cvec=malloc((size_t)d*sizeof(double));assert(cvec);double cnorm=0;
    for(int i=0;i<d;i++){cvec[i]=(double)m->norm[i]*((double)m->embedding[(size_t)a*d+i]-(double)m->embedding[(size_t)b*d+i]);cnorm+=cvec[i]*cvec[i];}
    cnorm=sqrt(cnorm);st->c_norm_rel=fmaxd(st->c_norm_rel,rel(row[QL_C_NORM],cnorm,cnorm));assert(rel(row[QL_C_NORM],cnorm,cnorm)<=TOL_REL);
    double rhoF=row[QL_RHO_FINAL],term_sum=0,term_abs=0;
#define TERM(vec,got) do{ double cw=0,wn=0;for(int i=0;i<d;i++){cw+=cvec[i]*(double)(vec)[i];wn+=(double)(vec)[i]*(vec)[i];} double ref=rhoF*cw,scale=fabs(rhoF)*cnorm*sqrt(wn)+1e-30; \
        double e=rel((got),ref,scale);st->margin_term_rel=fmaxd(st->margin_term_rel,e);assert(e<=TOL_REL);term_sum+=ref;term_abs+=fabs(ref);}while(0)
    TERM(port(c,0,QWEN_INPUT,(size_t)d),row[QL_MARGIN_EMBED]);
    float *keys=malloc((size_t)m->kvheads*s->capacity*hd*4);assert(keys);
    for(int l=0;l<L;l++){
        const float *blk=row+ql_layer_offset(l,H);
        const float *x0=port(c,l,QWEN_INPUT,(size_t)d),*r=port(c,l,QWEN_RAW_RECEIVER,(size_t)d),*wa=port(c,l,QWEN_CONSUMED_ATTENTION_WRITE,(size_t)d),*wm=port(c,l,QWEN_CONSUMED_MLP_WRITE,(size_t)d);
        const float *normed=port(c,l,QWEN_MLP_INPUT,(size_t)d),*q=port(c,l,QWEN_QUERY,(size_t)d),*P=port(c,l,QWEN_ATTENTION_PROBABILITIES,(size_t)H*n);
        const float *g=port(c,l,QWEN_MLP_GATE,(size_t)h),*u=port(c,l,QWEN_MLP_UP,(size_t)h),*prod=port(c,l,QWEN_MLP_PRODUCT,(size_t)h);
        /* RMS scales: double reference; RHO_MID also bit-exactly reproduces the native normalized vector. */
        double q0=dot(x0,x0,d)/d,q1=dot(r,r,d)/d,ref_in=1/sqrt(q0+(double)m->epsilon),ref_mid=1/sqrt(q1+(double)m->epsilon);
        double e=fmaxd(rel(blk[QL_RHO_IN],ref_in,ref_in),rel(blk[QL_RHO_MID],ref_mid,ref_mid));st->rho_rel=fmaxd(st->rho_rel,e);assert(e<=TOL_REL);
        for(int i=0;i<d;i++){float v=(r[i]*blk[QL_RHO_MID])*m->layer[l].post_norm[i];assert(!memcmp(&v,normed+i,4));}
        /* Softmax fields. */
        for(int k=0;k<m->kvheads;k++)gpu_read(keys+(size_t)k*s->capacity*hd,s->keys+(size_t)l*m->kvheads*s->capacity*hd+(size_t)k*s->capacity*hd,(size_t)n*hd*4);
        for(int head=0;head<H;head++){
            const float *p=P+(size_t)head*n;double *sc=malloc((size_t)n*sizeof(double));assert(sc);double mx=-INFINITY;
            for(int t=0;t<n;t++){double v=0;for(int i=0;i<hd;i++)v+=(double)q[head*hd+i]*(double)keys[((size_t)(head/group)*s->capacity+t)*hd+i];sc[t]=v/sqrt((double)hd);mx=fmaxd(mx,sc[t]);}
            double z=0;for(int t=0;t<n;t++)z+=exp(sc[t]-mx);double logz=mx+log(z),hs=0;for(int t=0;t<n;t++){double pp=exp(sc[t]-logz);if(pp>0)hs-=pp*log(pp);}
            double entropy=0;float best=-1;int arg=0;for(int t=0;t<n;t++){double pp=p[t];if(pp>0)entropy-=pp*log(pp);if(p[t]>best){best=p[t];arg=t;}}
            float lz=blk[QL_LAYER+QL_H_LOGZ*H+head],pf=blk[QL_LAYER+QL_H_PFIRST*H+head],en=blk[QL_LAYER+QL_H_ENTROPY*H+head],pm=blk[QL_LAYER+QL_H_PMAX*H+head],am=blk[QL_LAYER+QL_H_ARGMAX*H+head];
            assert(&blk[QL_LAYER+QL_H_ENTROPY*H+head]==row+ql_head_offset(l,H,QL_H_ENTROPY,head));
            st->logz_abs=fmaxd(st->logz_abs,fabs(lz-logz));assert(fabs(lz-logz)<=TOL_LOGZ_ABS);
            st->entropy_rel=fmaxd(st->entropy_rel,rel(en,entropy,fmaxd(entropy,1e-12)));assert(rel(en,entropy,fmaxd(entropy,1e-12))<=TOL_REL);
            st->entropy_scores_abs=fmaxd(st->entropy_scores_abs,fabs(en-hs));assert(fabs(en-hs)<=TOL_LOGZ_ABS);
            assert(pf==p[0]&&pm==best&&am==(float)arg);free(sc);
        }
        /* Write dots against the pre-add residual and the consumed write. */
        double da=dot(x0,wa,d),na=dot(wa,wa,d),dm=dot(r,wm,d),nm=dot(wm,wm,d);
        e=fmaxd(rel(blk[QL_ATTN_DOT],da,sqrt(dot(x0,x0,d)*na)),rel(blk[QL_MLP_DOT],dm,sqrt(dot(r,r,d)*nm)));st->dot_rel=fmaxd(st->dot_rel,e);assert(e<=TOL_REL);
        e=fmaxd(rel(blk[QL_ATTN_NORM2],na,na),rel(blk[QL_MLP_NORM2],nm,nm));st->norm2_rel=fmaxd(st->norm2_rel,e);assert(e<=TOL_REL);
        /* d*(q1-q0) = 2 dot + norm2, up to the FP32 rounding of the residual add. */
        e=rel(d*(q1-q0),2*(double)blk[QL_ATTN_DOT]+(double)blk[QL_ATTN_NORM2],d*q0+na);st->mean_square_identity_rel=fmaxd(st->mean_square_identity_rel,e);assert(e<=1e-5);
        /* Gate census. */
        int small=0,off=0,on=0;double e2=0,o2=0,eo=0,eoa=0,p2=0,split=0;
        for(int i=0;i<h;i++){
            if(fabsf(g[i])<1)small++;if(g[i]<=-5)off++;if(g[i]>=5)on++;
            double ev=(double)g[i]*u[i]/2,od=(double)g[i]*u[i]*tanh((double)g[i]/2)/2;e2+=ev*ev;o2+=od*od;eo+=ev*od;eoa+=fabs(ev*od);p2+=(double)prod[i]*prod[i];split+=(ev+od-prod[i])*(ev+od-prod[i]);
        }
        assert(blk[QL_GATE_SMALL]==(float)small&&blk[QL_GATE_OFF]==(float)off&&blk[QL_GATE_ON]==(float)on);st->gate_small+=small;st->gate_off+=off;st->gate_on+=on;
        e=fmaxd(fmaxd(rel(blk[QL_EVEN2],e2,e2),rel(blk[QL_ODD2],o2,o2)),rel(blk[QL_EVEN_ODD],eo,eoa));st->gate_rel=fmaxd(st->gate_rel,e);assert(e<=TOL_REL);
        st->prod2_rel=fmaxd(st->prod2_rel,rel(blk[QL_PROD2],p2,p2));assert(rel(blk[QL_PROD2],p2,p2)<=TOL_REL);
        st->even_odd_vs_prod_rel=fmaxd(st->even_odd_vs_prod_rel,sqrt(split/fmaxd(p2,1e-300)));assert(sqrt(split/fmaxd(p2,1e-300))<=1e-5);
        TERM(wa,blk[QL_MARGIN_ATTN]);TERM(wm,blk[QL_MARGIN_MLP]);
    }
    free(keys);
    /* Final scale: double reference and bit-exact reproduction of the native final norm. */
    const float *xf=port(c,L-1,QWEN_OUTPUT,(size_t)d);double ref=1/sqrt(dot(xf,xf,d)/d+(double)m->epsilon);
    st->rho_rel=fmaxd(st->rho_rel,rel(rhoF,ref,ref));assert(rel(rhoF,ref,ref)<=TOL_REL);
    float *fn=malloc((size_t)d*4);assert(fn);gpu_read(fn,s->norm,(size_t)d*4);for(int i=0;i<d;i++){float v=(xf[i]*row[QL_RHO_FINAL])*m->norm[i];assert(!memcmp(&v,fn+i,4));}free(fn);
    st->margin_sum_rel=fmaxd(st->margin_sum_rel,rel(row[QL_MARGIN_SUM],term_sum,term_abs));assert(rel(row[QL_MARGIN_SUM],term_sum,term_abs)<=TOL_REL);
    /* Closure: native FP32 logit difference vs the double decomposition. */
    double closure=fabs((double)row[QL_MARGIN]-(double)row[QL_MARGIN_SUM]);st->closure_abs=fmaxd(st->closure_abs,closure);st->closure_rel=fmaxd(st->closure_rel,closure/term_abs);
    assert(closure<=1e-5*(term_abs+fabs(row[QL_MARGIN])));
    /* Same decomposition of the final residual in double (isolates the FP32 residual adds). */
    double whole=0;for(int i=0;i<d;i++)whole+=cvec[i]*(double)xf[i];st->closure_double_abs=fmaxd(st->closure_double_abs,fabs(rhoF*whole-term_sum));
    free(cvec);st->rows++;
#undef TERM
}

static void equal_state(const QwenSession *a,const QwenSession *b){
    const QwenModel *m=a->model;assert(a->position==b->position&&a->pending==b->pending&&a->phase==b->phase&&a->layer==b->layer);
    assert(!memcmp(&a->top,&b->top,sizeof(a->top)));
    /* Live packed probability rows [qheads, length] of the last attention; bytes
     * beyond them are stale scratch from earlier, longer histories. */
    size_t live=(size_t)m->qheads*(size_t)(a->phase==2&&a->layer>0?a->position+1:a->position);
    struct { float *x,*y; size_t n; } v[]={{a->x,b->x,(size_t)m->hidden},{a->norm,b->norm,(size_t)m->hidden},{a->q,b->q,(size_t)m->hidden},{a->k,b->k,(size_t)m->kvheads*m->head},{a->v,b->v,(size_t)m->kvheads*m->head},
        {a->attention,b->attention,(size_t)m->hidden},{a->projected,b->projected,(size_t)m->hidden},{a->gate,b->gate,(size_t)m->intermediate},{a->up,b->up,(size_t)m->intermediate},{a->logits,b->logits,(size_t)m->vocab},
        {a->scores,b->scores,live}};
    for(unsigned j=0;j<sizeof(v)/sizeof(v[0]);j++){float *l=malloc(v[j].n*4),*r=malloc(v[j].n*4);assert(l&&r);gpu_read(l,v[j].x,v[j].n*4);gpu_read(r,v[j].y,v[j].n*4);assert(!memcmp(l,r,v[j].n*4));free(l);free(r);}
    /* Valid KV rows only: [0,position), plus the current row on layers already run. */
    size_t width=(size_t)m->head;float *l=malloc((size_t)a->capacity*width*4),*r=malloc((size_t)a->capacity*width*4);assert(l&&r);
    for(int layer=0;layer<m->layers;layer++)for(int k=0;k<m->kvheads;k++){
        size_t rows=(size_t)a->position+(a->phase==2&&layer<a->layer),off=((size_t)layer*m->kvheads+k)*a->capacity*width;
        gpu_read(l,a->keys+off,rows*width*4);gpu_read(r,b->keys+off,rows*width*4);assert(!memcmp(l,r,rows*width*4));
        gpu_read(l,a->values+off,rows*width*4);gpu_read(r,b->values+off,rows*width*4);assert(!memcmp(l,r,rows*width*4));
    }
    free(l);free(r);
}
/* (a)+(b)+(c): lockstep layer-by-layer, ledger off (a) vs on (b), every field checked. */
static void parity_and_fields(QwenModel *m,Stats *st){
    QwenSession off,on;assert(!qwen_session_init(&off,m,12)&&!qwen_session_init(&on,m,12));assert(!qwen_ledger_enable(&on,1));
    const int tokens[]={2,7,4,11,9,0,15,3,5,1};float *row=malloc((size_t)ql_row_floats(m->layers,m->qheads)*4);assert(row);
    Capture c={0};c.layers=m->layers;on.observer=capture;on.observer_context=&c;on.observe_layer=-1;
    for(unsigned t=0;t<sizeof(tokens)/sizeof(tokens[0]);t++){
        on.observe_position=on.position;int position=on.position;
        assert(!qwen_begin_token(&off,tokens[t])&&!qwen_begin_token(&on,tokens[t]));equal_state(&off,&on);
        for(int l=0;l<m->layers;l++){assert(!qwen_advance_layer(&off)&&!qwen_advance_layer(&on));equal_state(&off,&on);}
        assert(qwen_ledger_read(&on,row)==-1); /* mid-token: no finished row */
        assert(!qwen_finish_token(&off)&&!qwen_finish_token(&on));equal_state(&off,&on);
        assert(!qwen_ledger_read(&on,row));check_row(&on,&c,row,tokens[t],position,st);
        assert(qwen_ledger_read(&off,row)==-1);
    }
    /* Generation via qwen_advance, the ordinary pending-token path. */
    for(int t=0;t<2;t++){on.observe_position=on.position;int position=on.position,token=on.pending;assert(!qwen_advance(&off)&&!qwen_advance(&on));equal_state(&off,&on);assert(!qwen_ledger_read(&on,row));check_row(&on,&c,row,token,position,st);}
    capture_free(&c);free(row);qwen_session_free(&off);qwen_session_free(&on);
}

static int refuse_step(void *c,int l,int p,const float *q,const float *k,const float *v,float *o,float *sc){(void)c;(void)l;(void)p;(void)q;(void)k;(void)v;(void)o;(void)sc;return -1;}
static void refuse_rewind(void *c,int p,uint64_t a,uint64_t b){(void)c;(void)p;(void)a;(void)b;}
static void complete(QwenSession *s){while(s->layer<s->model->layers)assert(!qwen_advance_layer(s));assert(!qwen_finish_token(s));}
/* (d) lifecycle, rollback, abort and refusal. Returns the number of cases. */
static int lifecycle(QwenModel *m){
    int cases=0,n=ql_row_floats(m->layers,m->qheads);float *row=malloc((size_t)n*4),*first=malloc((size_t)n*4),*again=malloc((size_t)n*4);assert(row&&first&&again);
    QwenSession s,ref;assert(!qwen_session_init(&s,m,12)&&!qwen_session_init(&ref,m,12));
    assert(qwen_ledger_enable(NULL,1)==-1&&qwen_ledger_read(NULL,row)==-1);cases++;
    assert(qwen_ledger_read(&s,row)==-1);cases++;                                    /* off */
    assert(!qwen_ledger_enable(&s,1)&&!qwen_ledger_enable(&s,1));assert(qwen_ledger_read(&s,row)==-1);cases++; /* on, nothing finished */
    assert(qwen_ledger_read(&s,NULL)==-1);cases++;
    assert(!qwen_push(&s,2));assert(!qwen_ledger_read(&s,row));assert(row[QL_TOKEN]==2&&row[QL_POSITION]==0);cases++;
    /* Abort: begun token invalidates the old row; abort leaves none. */
    assert(!qwen_begin_token(&s,3));assert(qwen_ledger_read(&s,row)==-1);assert(!qwen_advance_layer(&s));assert(!qwen_abort_token(&s));assert(qwen_ledger_read(&s,row)==-1);cases++;
    assert(!qwen_push(&s,3));assert(!qwen_ledger_read(&s,first));assert(first[QL_POSITION]==1);cases++;
    /* Checkpoint restore: the rewound row is not presented; replay reproduces it bit-exactly. */
    QwenCheckpoint cp;assert(!qwen_checkpoint(&s,&cp));assert(!qwen_push(&s,4));assert(!qwen_ledger_read(&s,first));assert(!qwen_push(&s,5));
    assert(!qwen_restore(&s,&cp));assert(qwen_ledger_read(&s,row)==-1);assert(!qwen_push(&s,4));assert(!qwen_ledger_read(&s,again));assert(!memcmp(first,again,(size_t)n*4));cases++;
    /* Stack rollback invalidates; commit keeps the row. */
    uint64_t id;assert(!qwen_stack_push(&s,&id));assert(!qwen_push(&s,6));assert(!qwen_ledger_read(&s,row));assert(!qwen_stack_rollback(&s,id));assert(qwen_ledger_read(&s,row)==-1);cases++;
    assert(!qwen_stack_push(&s,&id));assert(!qwen_push(&s,6));assert(!qwen_stack_commit(&s,id));assert(!qwen_ledger_read(&s,row)&&row[QL_TOKEN]==6);cases++;
    /* Enabling mid-token is refused; disabling is always allowed and invalidates. */
    assert(!qwen_ledger_enable(&s,0));assert(!qwen_begin_token(&s,7));assert(qwen_ledger_enable(&s,1)==-1&&!s.ledger_on);complete(&s);assert(qwen_ledger_read(&s,row)==-1);cases++;
    assert(!qwen_ledger_enable(&s,1));assert(!qwen_push(&s,8));assert(!qwen_ledger_read(&s,row));assert(!qwen_ledger_enable(&s,0));assert(qwen_ledger_read(&s,row)==-1);cases++;
    assert(!qwen_ledger_enable(&s,1));assert(!qwen_begin_token(&s,9));assert(!qwen_advance_layer(&s));assert(!qwen_ledger_enable(&s,0));complete(&s);assert(!qwen_ledger_enable(&s,1));assert(qwen_ledger_read(&s,row)==-1);cases++;
    assert(!qwen_reset(&s));assert(qwen_ledger_read(&s,row)==-1);cases++;
    /* Replacement paths record nothing for the token; the next native token records. */
    for(int t=2;t<=4;t++)assert(!qwen_push(&s,t));
    QwenPlane plane;QwenPlaneWork work;assert(!qwen_plane_init(&plane,m));
    assert(!qwen_begin_token(&s,9));assert(!qwen_advance_layer(&s));assert(!qwen_advance_layer_compile(&s,&plane,&work));complete(&s);assert(qwen_ledger_read(&s,row)==-1);cases++;
    assert(!qwen_push(&s,10));assert(!qwen_ledger_read(&s,row)&&row[QL_TOKEN]==10);cases++;
    QwenSubspace sub;QwenSubspaceStats stats;assert(!qwen_subspace_init(&sub,m));
    assert(!qwen_begin_token(&s,11));assert(!qwen_advance_layer(&s));assert(!qwen_advance_layer_subspace(&s,&sub,QWEN_SUBSPACE_COMPILE,&stats));complete(&s);assert(qwen_ledger_read(&s,row)==-1);cases++;
    float *source=gpu_alloc((size_t)m->intermediate*4);float *host=malloc((size_t)m->intermediate*4);assert(host);for(int i=0;i<m->intermediate;i++)host[i]=.1f*(float)(i-5);gpu_write(source,host,(size_t)m->intermediate*4);free(host);
    QwenGateLaw law;QwenGateWork gwork;assert(!qwen_gate_law_init(&law,m,1,s.position,source));
    assert(!qwen_begin_token(&s,12));assert(!qwen_advance_layer(&s));assert(!qwen_advance_layer_gate(&s,&law,QWEN_GATE_NATIVE,s.lineage_id,&gwork));complete(&s);assert(qwen_ledger_read(&s,row)==-1);cases++;
    /* A refused replacement call leaves recording intact for the native retry. */
    assert(!qwen_begin_token(&s,13));assert(!qwen_advance_layer(&s));assert(qwen_advance_layer_gate(&s,&law,QWEN_GATE_NATIVE,s.lineage_id+1,&gwork)==-1);complete(&s);assert(!qwen_ledger_read(&s,row)&&row[QL_TOKEN]==13);cases++;
    qwen_gate_law_free(&law);gpu_free(source);qwen_subspace_free(&sub);qwen_plane_free(&plane);
    /* Bound programs: refuse to enable; a program bound after enabling records nothing. */
    QwenLexicalProgram lex={0};int ids[]={2,3,4};assert(!qwen_lexical_compile(&lex,m,ids,3));
    assert(!qwen_reset(&s));assert(!qwen_ledger_enable(&s,0));assert(!qwen_lexical_bind(&s,&lex));assert(qwen_ledger_enable(&s,1)==-1&&!s.ledger_on);cases++;
    assert(!qwen_lexical_bind(&s,NULL));assert(!qwen_ledger_enable(&s,1));assert(!qwen_lexical_bind(&s,&lex));assert(!qwen_push(&s,2));assert(qwen_ledger_read(&s,row)==-1);cases++;
    assert(!qwen_reset(&s));assert(!qwen_lexical_bind(&s,NULL));assert(!qwen_push(&s,2));assert(!qwen_ledger_read(&s,row));cases++;
    float basis[8]={1,0,0,0,0,0,0,0};QwenQkvProgram qkv={0};assert(!qwen_qkv_program_init(&qkv,m,1,1,1,basis,m->layer[1].q,NULL,NULL));
    assert(!qwen_reset(&s));assert(!qwen_ledger_enable(&s,0));assert(!qwen_qkv_bind(&s,1,&qkv));assert(qwen_ledger_enable(&s,1)==-1);cases++;
    assert(!qwen_qkv_bind(&s,1,NULL));assert(!qwen_ledger_enable(&s,1));assert(!qwen_qkv_bind(&s,1,&qkv));assert(!qwen_push(&s,2));assert(qwen_ledger_read(&s,row)==-1);cases++;
    assert(!qwen_reset(&s));assert(!qwen_qkv_bind(&s,1,NULL));
    QwenSession compact;assert(!qwen_session_init_layout(&compact,m,12,UINT64_C(2),refuse_step,refuse_rewind,NULL));assert(qwen_ledger_enable(&compact,1)==-1&&!compact.ledger_on);cases++;
    qwen_session_free(&compact);qwen_qkv_program_free(&qkv);qwen_lexical_free(&lex);
    /* After all of the above, the ledger session still matches a never-enabled one. */
    assert(!qwen_ledger_enable(&s,1));for(int t=2;t<=6;t++){assert(!qwen_push(&s,t));assert(!qwen_push(&ref,t));}equal_state(&s,&ref);assert(!qwen_ledger_read(&s,row));cases++;
    free(row);free(first);free(again);qwen_session_free(&s);qwen_session_free(&ref);return cases;
}
static void summary(const char *name,const Stats *s){
    printf("{\"fixture\":\"%s\",\"rows_checked\":%d,\"gate_small\":%d,\"gate_off\":%d,\"gate_on\":%d,\"max_rho_rel\":%.3g,\"max_logz_abs\":%.3g,\"max_entropy_rel\":%.3g,\"max_entropy_vs_double_scores_abs\":%.3g,"
           "\"max_dot_rel\":%.3g,\"max_norm2_rel\":%.3g,\"max_gate_sum_rel\":%.3g,\"max_prod2_rel\":%.3g,\"max_even_plus_odd_vs_product_rel\":%.3g,\"max_mean_square_identity_rel\":%.3g,"
           "\"max_margin_term_rel\":%.3g,\"max_c_norm_rel\":%.3g,\"max_margin_sum_rel\":%.3g,\"max_closure_abs\":%.3g,\"max_closure_rel_to_sum_abs_terms\":%.3g,\"max_double_residual_closure_abs\":%.3g}\n",
           name,s->rows,s->gate_small,s->gate_off,s->gate_on,s->rho_rel,s->logz_abs,s->entropy_rel,s->entropy_scores_abs,s->dot_rel,s->norm2_rel,s->gate_rel,s->prod2_rel,s->even_odd_vs_prod_rel,s->mean_square_identity_rel,
           s->margin_term_rel,s->c_norm_rel,s->margin_sum_rel,s->closure_abs,s->closure_rel,s->closure_double_abs);
}
int main(void){
    QwenModel m;Stats plain={0},scaled={0};
    fixture(&m,1,0);parity_and_fields(&m,&plain);int cases=lifecycle(&m);fixture_free(&m);summary("state-mechanics fixture",&plain);
    fixture(&m,40,0);parity_and_fields(&m,&scaled);fixture_free(&m);summary("same fixture, layer-2 gate x40",&scaled);
    Stats tied={0};fixture(&m,1,1);parity_and_fields(&m,&tied);fixture_free(&m);summary("same fixture, layer-0 queries zero (exact ties)",&tied);
    assert(scaled.gate_off>0&&scaled.gate_on>0&&scaled.gate_small>0);
    printf("ledger mechanics: passed (bit-identical on/off, all fields recomputed, %d lifecycle/refusal cases)\n",cases);return 0;
}
