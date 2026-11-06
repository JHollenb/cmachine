#include "qwen.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef NDEBUG
#error "The lifecycle assay requires assertions; transitions must not be compiled out"
#endif

#ifndef TCM_REAL_MODEL
static float *parameters(int n,int norm){
    static int index=0;float *p=malloc((size_t)n*4);assert(p);
    for(int i=0;i<n;i++)p[i]=norm?.6f+(i%7)*.13f:(float)((++index*17)%29-14)/70;return p;
}
static void fixture(QwenModel *m){
    memset(m,0,sizeof(*m));m->gpu=gpu_open();m->hidden=8;m->layers=3;m->qheads=4;m->kvheads=2;m->head=2;m->intermediate=12;m->vocab=16;m->max_position=32;m->epsilon=1e-6f;m->theta=1e6f;memset(m->digest,'a',64);
    m->embedding=parameters(16*8,0);m->norm=parameters(8,1);m->layer=calloc(3,sizeof(QwenLayer));
    for(int i=0;i<3;i++){QwenLayer *l=m->layer+i;
        l->input_norm=parameters(8,1);l->post_norm=parameters(8,1);l->q=parameters(64,0);l->k=parameters(32,0);l->v=parameters(32,0);l->qb=parameters(8,0);l->kb=parameters(4,0);l->vb=parameters(4,0);l->o=parameters(64,0);l->gate=parameters(96,0);l->up=parameters(96,0);l->down=parameters(96,0);
    }
}
static void fixture_free(QwenModel *m){
    for(int i=0;i<3;i++){QwenLayer *l=m->layer+i;free(l->input_norm);free(l->post_norm);free(l->q);free(l->k);free(l->v);free(l->qb);free(l->kb);free(l->vb);free(l->o);free(l->gate);free(l->up);free(l->down);}free(m->layer);free(m->embedding);free(m->norm);gpu_close(m->gpu);
}
#endif
static void prefix(QwenSession *s){assert(!qwen_reset(s));for(int i=2;i<=4;i++)assert(!qwen_push(s,i));}
static void equal(const QwenSession *a,const QwenSession *b){
    assert(a->position==b->position&&a->pending==b->pending&&a->delivered==b->delivered);
    assert(a->phase==1&&b->phase==1);
    size_t bytes=(size_t)a->model->vocab*4;float *left=malloc(bytes),*right=malloc(bytes);assert(left&&right);
    gpu_read(left,a->logits,bytes);gpu_read(right,b->logits,bytes);assert(!memcmp(left,right,bytes));free(left);free(right);
    int rows=a->model->layers*a->model->kvheads,d=a->model->head;
    bytes=(size_t)a->position*d*4;left=malloc(bytes);right=malloc(bytes);assert(left&&right);
    for(int i=0;i<rows;i++){size_t offset=(size_t)i*a->capacity*d;gpu_read(left,a->keys+offset,bytes);gpu_read(right,b->keys+offset,bytes);assert(!memcmp(left,right,bytes));gpu_read(left,a->values+offset,bytes);gpu_read(right,b->values+offset,bytes);assert(!memcmp(left,right,bytes));}free(left);free(right);
}
static unsigned seen;
static void observer(void *context,int position,int layer,QwenPort port,const float *data,size_t count){
    (void)context;assert(position==3&&layer==1&&count&&data);assert(!(seen&(1u<<port)));seen|=1u<<port;
}
static void to_plane(QwenSession *s){assert(!qwen_begin_token(s,9));assert(!qwen_advance_layer(s));}
static void complete(QwenSession *s){while(s->layer<s->model->layers)assert(!qwen_advance_layer(s));assert(!qwen_finish_token(s));}
static double plane_mechanics(QwenSession *a,QwenSession *b){
    QwenPlane p;QwenPlaneWork work;QwenCheckpoint parent;assert(!qwen_plane_init(&p,a->model));
    prefix(a);prefix(b);assert(!qwen_checkpoint(a,&parent));to_plane(a);
    assert(!qwen_advance_layer_compile(a,&p,&work));assert(p.ready&&work.gate_up_matvecs==2&&work.down_matvecs==1&&work.compile_matvecs==6);complete(a);
    assert(!qwen_push(b,9));equal(a,b);
    assert(!qwen_restore(a,&parent));to_plane(a);assert(!qwen_advance_layer_plane(a,&p,QWEN_PLANE_DENSE,0,0,&work));
    assert(work.gate_up_matvecs==2&&work.down_matvecs==1&&!work.compile_matvecs);complete(a);equal(a,b);
    const float doses[][2]={{0,0},{.25f,0},{-.25f,0},{-1,0},{0,.25f},{.5f,.5f}};double largest=0;
    size_t bytes=(size_t)a->model->hidden*4;float *dense=malloc(bytes),*compiled=malloc(bytes);assert(dense&&compiled);
    for(unsigned j=0;j<sizeof(doses)/sizeof(doses[0]);j++){
        assert(!qwen_restore(a,&parent));to_plane(a);
        QwenSession before=*a;assert(qwen_advance_layer_plane(a,&p,QWEN_PLANE_COMPILED,NAN,0,&work)==-1);assert(!memcmp(&before,a,sizeof(*a)));
        assert(!qwen_advance_layer_plane(a,&p,QWEN_PLANE_DENSE,doses[j][0],doses[j][1],&work));gpu_read(dense,a->projected,bytes);complete(a);
        assert(!qwen_restore(a,&parent));to_plane(a);assert(!qwen_advance_layer_plane(a,&p,QWEN_PLANE_COMPILED,doses[j][0],doses[j][1],&work));
        assert(!work.gate_up_matvecs&&work.down_matvecs==1&&!work.compile_matvecs);gpu_read(compiled,a->projected,bytes);
        double delta=0,norm=0;for(int i=0;i<a->model->hidden;i++){assert(isfinite(dense[i])&&isfinite(compiled[i]));double e=(double)dense[i]-compiled[i];delta+=e*e;norm+=(double)dense[i]*dense[i];}double relative=sqrt(delta/fmax(norm,1e-30));assert(isfinite(relative));if(relative>largest)largest=relative;
        complete(a);
    }
#ifndef TCM_REAL_MODEL
    /* Tiny fixture: rounding bound for this actual arithmetic/geometry only. */
    assert(largest<1e-5);
#endif
    assert(!qwen_restore(a,&parent));to_plane(a);assert(!qwen_advance_layer_plane(a,&p,QWEN_PLANE_HELD_WRITE,.25f,0,&work));assert(!work.gate_up_matvecs&&!work.down_matvecs);assert(gpu_identical_finite(a->model->gpu,a->projected,p.baseline_write,a->model->hidden));complete(a);
    /* Session identity and the live incoming operand are independently guarded. */
    prefix(b);to_plane(b);QwenSession before=*b;assert(qwen_advance_layer_plane(b,&p,QWEN_PLANE_DENSE,0,0,&work)==-1);assert(!memcmp(&before,b,sizeof(*b)));assert(!qwen_reset(b));
    assert(!qwen_restore(a,&parent));to_plane(a);gpu_add(a->x,p.basis[2],a->model->hidden);assert(qwen_advance_layer_plane(a,&p,QWEN_PLANE_DENSE,0,0,&work)==-1);assert(!qwen_reset(a));
#ifndef TCM_REAL_MODEL
    /* Hostile instrument failures: changed read, non-finite materialization,
     * and failed coefficient compilation all leave a safely retryable layer. */
    prefix(a);assert(!qwen_checkpoint(a,&parent));to_plane(a);
    QwenPlane retry;assert(!qwen_plane_init(&retry,a->model));assert(!qwen_advance_layer_compile(a,&retry,&work));complete(a);
    prefix(b);assert(!qwen_push(b,9));equal(a,b);
    assert(!qwen_restore(a,&parent));to_plane(a);float saved=retry.basis[1][0];retry.basis[1][0]+=1;
    assert(qwen_advance_layer_plane(a,&retry,QWEN_PLANE_DENSE,0,0,&work)==-2);retry.basis[1][0]=saved;assert(!qwen_advance_layer(a));complete(a);equal(a,b);
    assert(!qwen_restore(a,&parent));to_plane(a);saved=retry.basis[0][0];retry.basis[0][0]=NAN;
    assert(qwen_advance_layer_plane(a,&retry,QWEN_PLANE_COMPILED,0,0,&work)==-2);retry.basis[0][0]=saved;assert(!qwen_advance_layer(a));complete(a);equal(a,b);qwen_plane_free(&retry);
    assert(!qwen_restore(a,&parent));to_plane(a);assert(!qwen_plane_init(&retry,a->model));float *post=a->model->layer[1].post_norm;saved=post[0];post[0]=NAN;
    assert(qwen_advance_layer_compile(a,&retry,&work)==-2);assert(!retry.ready);post[0]=saved;assert(!qwen_advance_layer(a));complete(a);equal(a,b);qwen_plane_free(&retry);
#endif
    for(int i=2;i<=4;i++)assert(!qwen_push(a,i));to_plane(a);assert(qwen_advance_layer_plane(a,&p,QWEN_PLANE_DENSE,0,0,&work)==-1);assert(!qwen_reset(a));
    free(dense);free(compiled);qwen_plane_free(&p);prefix(a);prefix(b);return largest;
}
static double subspace_mechanics(QwenSession *a,QwenSession *b){
    QwenSubspace p;QwenSubspaceStats stats;QwenCheckpoint parent;assert(!qwen_subspace_init(&p,a->model));
    prefix(a);prefix(b);to_plane(a);assert(!qwen_advance_layer_subspace(a,&p,QWEN_SUBSPACE_COMPILE,&stats));
    assert(p.ready&&p.layer==1&&stats.work.compile_matvecs==4&&stats.work.gate_up_matvecs==2&&stats.work.down_matvecs==1);assert(isfinite(p.condition));complete(a);assert(!qwen_push(b,9));equal(a,b);
    assert(!qwen_checkpoint(a,&parent));size_t bytes=(size_t)a->model->hidden*4;float *native=malloc(bytes),*actual=malloc(bytes),*receiver=malloc(bytes),*reference_receiver=malloc(bytes);assert(native&&actual&&receiver&&reference_receiver);double largest=0;
    const QwenSubspaceMode modes[]={QWEN_SUBSPACE_DENSE,QWEN_SUBSPACE_PROJECT,QWEN_SUBSPACE_CORRECT,QWEN_SUBSPACE_REASSOC,QWEN_SUBSPACE_HELD};
    for(unsigned mode=0;mode<sizeof(modes)/sizeof(modes[0]);mode++){
        assert(!qwen_restore(a,&parent));assert(!qwen_begin_token(a,10));
        QwenSession before=*a;assert(qwen_advance_layer_subspace(a,&p,modes[mode],&stats)==-1);assert(!memcmp(&before,a,sizeof(*a)));assert(!qwen_advance_layer(a));
        assert(!qwen_advance_layer_subspace(a,&p,modes[mode],&stats));assert(!stats.work.compile_matvecs);
        assert(stats.work.gate_up_matvecs==(modes[mode]==QWEN_SUBSPACE_PROJECT||modes[mode]==QWEN_SUBSPACE_HELD?0u:2u));assert(stats.work.down_matvecs==(modes[mode]==QWEN_SUBSPACE_HELD?0u:1u));
        gpu_read(actual,a->projected,bytes);gpu_read(receiver,p.receiver,bytes);
        if(!mode){memcpy(native,actual,bytes);memcpy(reference_receiver,receiver,bytes);}else assert(!memcmp(reference_receiver,receiver,bytes));
        assert(isfinite(stats.rho)&&isfinite(stats.omitted_norm)&&isfinite(stats.executed[0])&&isfinite(stats.executed[1]));
        double delta=0,norm=0;for(int i=0;i<a->model->hidden;i++){assert(isfinite(actual[i])&&isfinite(native[i]));double e=(double)actual[i]-native[i];delta+=e*e;norm+=(double)native[i]*native[i];}double relative=sqrt(delta/fmax(norm,1e-30));assert(isfinite(relative));
        if(modes[mode]==QWEN_SUBSPACE_CORRECT||modes[mode]==QWEN_SUBSPACE_REASSOC)if(relative>largest)largest=relative;
        if(modes[mode]==QWEN_SUBSPACE_HELD)assert(gpu_identical_finite(a->model->gpu,a->projected,p.baseline_write,a->model->hidden));
        complete(a);
    }
#ifndef TCM_REAL_MODEL
    assert(largest<1e-5);
    assert(!qwen_restore(a,&parent));assert(!qwen_begin_token(a,10));assert(!qwen_advance_layer(a));double saved=p.q0[0];p.q0[0]=NAN;
    assert(qwen_advance_layer_subspace(a,&p,QWEN_SUBSPACE_PROJECT,&stats)==-2);p.q0[0]=saved;assert(!qwen_advance_layer(a));complete(a);assert(!qwen_push(b,10));equal(a,b);
    /* Failed compilation restores the incoming layer state and remains unready. */
    prefix(a);prefix(b);to_plane(a);QwenSubspace failed;assert(!qwen_subspace_init(&failed,a->model));float *post=a->model->layer[1].post_norm;float old=post[0];post[0]=NAN;
    assert(qwen_advance_layer_subspace(a,&failed,QWEN_SUBSPACE_COMPILE,&stats)==-2);assert(!failed.ready);post[0]=old;assert(!qwen_advance_layer(a));complete(a);assert(!qwen_push(b,9));equal(a,b);qwen_subspace_free(&failed);
#endif
    /* One explicit operand enriches the program without altering B2 execution. */
    prefix(a);assert(!qwen_push(a,9));assert(!qwen_checkpoint(a,&parent));
    assert(!qwen_begin_token(a,10));assert(!qwen_advance_layer(a));assert(!qwen_advance_layer_subspace(a,&p,QWEN_SUBSPACE_PROJECT2,&stats));gpu_read(actual,a->projected,bytes);QwenSubspaceStats original=stats;complete(a);
    assert(!qwen_restore(a,&parent));assert(!qwen_begin_token(a,10));assert(!qwen_advance_layer(a));QwenSession split_before=*a;assert(qwen_advance_layer_subspace(a,&p,QWEN_SUBSPACE_CARRY_CORRECT,&stats)==-1);assert(!memcmp(&split_before,a,sizeof(*a)));complete(a);
    QwenPlaneWork compile_work;
#ifndef TCM_REAL_MODEL
    float valid=p.omitted[0];p.omitted[0]=NAN;assert(qwen_subspace_extend(&p,p.omitted,&compile_work)==-2&&p.rank==2&&!compile_work.compile_matvecs);p.omitted[0]=valid;
    float *gate_weight=a->model->layer[1].gate;valid=gate_weight[0];gate_weight[0]=NAN;assert(qwen_subspace_extend(&p,p.omitted,&compile_work)==-2&&p.rank==2&&compile_work.compile_matvecs==2);gate_weight[0]=valid;
    assert(!qwen_restore(a,&parent));assert(!qwen_begin_token(a,10));assert(!qwen_advance_layer(a));assert(!qwen_advance_layer_subspace(a,&p,QWEN_SUBSPACE_PROJECT2,&stats));gpu_read(receiver,a->projected,bytes);assert(!memcmp(actual,receiver,bytes));complete(a);
#endif
    assert(!qwen_subspace_extend(&p,p.omitted,&compile_work));assert(p.rank==3&&compile_work.compile_matvecs==2&&!compile_work.gate_up_matvecs&&!compile_work.down_matvecs);
    assert(qwen_subspace_extend(&p,p.omitted,&compile_work)==-1&&!compile_work.compile_matvecs);
    assert(!qwen_restore(a,&parent));assert(!qwen_begin_token(a,10));assert(!qwen_advance_layer(a));assert(!qwen_advance_layer_subspace(a,&p,QWEN_SUBSPACE_PROJECT2,&stats));gpu_read(receiver,a->projected,bytes);assert(!memcmp(actual,receiver,bytes));assert(!memcmp(original.executed,stats.executed,sizeof(stats.executed)));complete(a);
    assert(!qwen_restore(a,&parent));assert(!qwen_begin_token(a,10));assert(!qwen_advance_layer(a));assert(!qwen_advance_layer_subspace(a,&p,QWEN_SUBSPACE_PROJECT,&stats));assert(stats.work.gate_up_matvecs==0&&stats.work.down_matvecs==1);assert(stats.omitted_norm<1e-5*stats.receiver_norm);complete(a);
    for(int mode=QWEN_SUBSPACE_CARRY_CORRECT;mode<=QWEN_SUBSPACE_READ_CORRECT;mode++){
        assert(!qwen_restore(a,&parent));assert(!qwen_begin_token(a,11));assert(!qwen_advance_layer(a));assert(!qwen_advance_layer_subspace(a,&p,(QwenSubspaceMode)mode,&stats));
        assert(stats.work.gate_up_matvecs==2&&stats.work.down_matvecs==1&&!stats.work.compile_matvecs);
        gpu_read(receiver,p.entry,bytes);double v[3]={0,0,0};const double *q[]={p.q0,p.q1,p.q2};for(int j=0;j<a->model->hidden;j++)for(int k=0;k<3;k++)v[k]+=q[k][j]*(double)receiver[j];
        for(int j=0;j<a->model->hidden;j++){double e=receiver[j];for(int k=0;k<3;k++)e-=q[k][j]*v[k];receiver[j]=(float)e;}
        gpu_read(reference_receiver,p.carry_omitted,bytes);assert(!memcmp(receiver,reference_receiver,bytes));gpu_read(native,p.omitted,bytes);gpu_read(actual,p.read_complement,bytes);
        for(int j=0;j<a->model->hidden;j++){float e=(native[j]+(-1.0f*receiver[j]))+(0.0f*receiver[j]);assert(!memcmp(&e,&actual[j],4));assert(isfinite(e));}
        complete(a);
    }
    /* Explicit operands are staged before aliases can be overwritten. Shape,
     * missing operand and nonfinite refusals precede session mutation. */
    assert(!qwen_restore(a,&parent));assert(!qwen_begin_token(a,11));assert(!qwen_advance_layer(a));
    QwenSession explicit_before=*a;gpu_read(native,a->x,bytes);
    assert(qwen_advance_layer_subspace_operand(a,&p,a->x,(size_t)a->model->hidden-1,&stats)==-1);assert(!memcmp(&explicit_before,a,sizeof(*a)));
    assert(qwen_advance_layer_subspace(a,&p,QWEN_SUBSPACE_OPERAND_CORRECT,&stats)==-1);assert(!memcmp(&explicit_before,a,sizeof(*a)));
    for(int j=0;j<a->model->hidden;j++)actual[j]=NAN;gpu_write(p.correction_operand,actual,bytes);
    assert(qwen_advance_layer_subspace_operand(a,&p,p.correction_operand,(size_t)a->model->hidden,&stats)==-2);assert(!memcmp(&explicit_before,a,sizeof(*a)));
    assert(!qwen_advance_layer_subspace_operand(a,&p,a->x,(size_t)a->model->hidden,&stats));
    gpu_read(actual,p.correction_operand,bytes);assert(!memcmp(native,actual,bytes));
    assert(stats.work.gate_up_matvecs==2&&stats.work.down_matvecs==1&&!stats.work.compile_matvecs);complete(a);
    /* A pure model/layer operator can execute in another session and epoch. */
    prefix(b);to_plane(b);assert(!qwen_advance_layer_subspace(b,&p,QWEN_SUBSPACE_PROJECT,&stats));complete(b);
    free(native);free(actual);free(receiver);free(reference_receiver);qwen_subspace_free(&p);prefix(a);prefix(b);return largest;
}
#include "writer_mechanics.h"
#include "gate_mechanics.h"
int main(int argc,char **argv){
    QwenModel m;
#ifdef TCM_REAL_MODEL
    assert(argc==2);TcmModel bundle;char error[256];assert(!tcm_load(&bundle,argv[1],error,sizeof(error)));assert(!qwen_model_init(&m,&bundle,error,sizeof(error)));tcm_unload(&bundle);
#else
    (void)argc;(void)argv;fixture(&m);
#endif
    QwenSession a,b;assert(!qwen_session_init(&a,&m,12));assert(!qwen_session_init(&b,&m,12));
    QwenCheckpoint legacy;uint64_t outer,inner,deep,foreign,replacement;
    assert(qwen_stack_push(&a,&outer)==-1);prefix(&a);prefix(&b);equal(&a,&b);
    assert(!qwen_checkpoint(&a,&legacy));a.delivered=1;assert(!qwen_stack_push(&a,&outer));
    assert(!qwen_stack_push(&b,&foreign));assert(qwen_stack_commit(&a,foreign)==-1);assert(!qwen_stack_commit(&b,foreign));
    assert(qwen_restore(&a,&legacy)==-1);assert(!qwen_push(&a,5));assert(!qwen_stack_push(&a,&inner));
    assert(!qwen_push(&a,6));assert(!qwen_stack_push(&a,&deep));assert(!qwen_push(&a,7));
    assert(qwen_stack_rollback(&a,outer)==-1);assert(!qwen_stack_rollback(&a,deep));
    assert(a.position==5&&a.phase==0);assert(!qwen_push(&b,5));assert(!qwen_push(&b,6));
    assert(!qwen_advance(&a));assert(!qwen_advance(&b));equal(&a,&b);
    assert(!qwen_stack_rollback(&a,inner));assert(!qwen_stack_push(&a,&replacement));assert(replacement!=deep);assert(qwen_stack_rollback(&a,deep)==-1);
    assert(!qwen_push(&a,8));assert(!qwen_stack_commit(&a,replacement));assert(!qwen_stack_rollback(&a,outer));
    assert(a.position==3&&a.delivered==1&&a.depth==0);assert(qwen_restore(&a,&legacy)==-1);
    prefix(&b);b.delivered=1;assert(!qwen_advance(&a));assert(!qwen_advance(&b));equal(&a,&b);
    /* Observing and explicitly stepping preserve the complete toy continuation. */
    prefix(&a);prefix(&b);a.observer=observer;a.observe_position=3;a.observe_layer=1;
    assert(!qwen_begin_token(&a,9));assert(qwen_begin_token(&a,9)==-1);assert(qwen_finish_token(&a)==-1);assert(qwen_checkpoint(&a,&legacy)==-1);assert(qwen_stack_push(&a,&outer)==-1);
    for(int i=0;i<m.layers;i++)assert(!qwen_advance_layer(&a));assert(qwen_advance_layer(&a)==-1);assert(!qwen_finish_token(&a));assert(qwen_finish_token(&a)==-1);assert(!qwen_push(&b,9));equal(&a,&b);assert(seen==(1u<<QWEN_PORT_COUNT)-1);a.observer=NULL;
    double plane_error=plane_mechanics(&a,&b);
    double subspace_error=subspace_mechanics(&a,&b);
    writer_mechanics(&a,&b);
    gate_mechanics(&a,&b);
    /* Same-position markers, overflow refusal and epoch invalidation. */
    uint64_t ids[QWEN_STACK_LIMIT];for(int i=0;i<QWEN_STACK_LIMIT;i++)assert(!qwen_stack_push(&a,ids+i));
    QwenSession before=a;assert(qwen_stack_push(&a,&outer)==-1);assert(!memcmp(&before,&a,sizeof(a)));
    for(int i=QWEN_STACK_LIMIT-1;i>=0;i--)assert(!qwen_stack_rollback(&a,ids[i]));
    assert(!qwen_checkpoint(&a,&legacy));assert(!qwen_stack_push(&a,&outer));assert(!qwen_begin_token(&a,10));assert(!qwen_advance_layer(&a));
    before=a;assert(qwen_stack_commit(&a,outer)==-1);assert(qwen_stack_rollback(&a,outer)==-1);assert(qwen_restore(&a,&legacy)==-1);assert(!memcmp(&before,&a,sizeof(a)));
    assert(!qwen_reset(&a));assert(a.depth==0&&a.pending==-1);assert(qwen_stack_rollback(&a,outer)==-1);
    prefix(&a);prefix(&b);assert(!qwen_push(&a,10));assert(!qwen_push(&b,10));equal(&a,&b);
    while(a.position<a.capacity)assert(!qwen_push(&a,1));before=a;assert(qwen_push(&a,1)==-1);assert(!memcmp(&before,&a,sizeof(a)));
    qwen_session_free(&a);qwen_session_free(&b);
#ifdef TCM_REAL_MODEL
    qwen_model_free(&m);const char *scope="pretrained Qwen lifecycle; synthetic token inputs";
#else
    fixture_free(&m);const char *scope="model-free toy lifecycle";
#endif
    printf("{\"scope\":\"%s\",\"nested_replay\":true,\"pending_policy\":true,\"foreign_stale_handles\":true,\"overflow_atomic\":true,\"step_and_observation_parity\":true,\"partial_reset\":true,\"valid_kv_and_full_logits_exact\":true,\"plane_guards_and_work_counts\":true,\"plane_max_write_relative_error\":%.9g,\"subspace_lifecycle_and_work_counts\":true,\"subspace_arithmetic_control_max_relative_error\":%.9g,\"writer_checks\":true,\"writer_parent_lineage_checks\":true,\"stack_limit\":%d,\"host_marker_bytes\":%zu}\n",scope,plane_error,subspace_error,QWEN_STACK_LIMIT,sizeof(a.frames));return 0;
}
