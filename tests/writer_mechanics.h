#ifndef TCM_WRITER_MECHANICS_H
#define TCM_WRITER_MECHANICS_H
typedef struct { float *v[6]; int d; } WriterProbe;
static void writer_probe(void *context,int position,int layer,QwenPort port_id,const float *data,size_t count){
    (void)position;(void)layer;WriterProbe *p=context;int slot=-1;
    if(port_id==QWEN_ATTENTION_WRITE)slot=0;if(port_id==QWEN_CONSUMED_ATTENTION_WRITE)slot=1;
    if(port_id==QWEN_RAW_RECEIVER)slot=2;if(port_id==QWEN_MLP_RETURN)slot=3;
    if(port_id==QWEN_CONSUMED_MLP_WRITE)slot=4;if(port_id==QWEN_OUTPUT)slot=5;
    if(slot>=0){assert(count==(size_t)p->d);gpu_read(p->v[slot],data,count*4);}
}
static void writer_current_kv(const QwenSession *s,float *host){
    const QwenModel *m=s->model;size_t offset=0;
    for(int kind=0;kind<2;kind++)for(int l=0;l<m->layers;l++)for(int k=0;k<m->kvheads;k++){
        size_t index=(((size_t)l*m->kvheads+k)*s->capacity+s->position)*m->head;
        gpu_read(host+offset,(kind?s->values:s->keys)+index,(size_t)m->head*4);offset+=m->head;
    }
}
static void writer_lineage_mechanics(QwenSession *s,const QwenSubspace *supplied){
    assert(!qwen_reset(s));assert(!qwen_push(s,2));assert(!qwen_push(s,3));uint64_t earlier,parent;
    assert(!qwen_stack_push(s,&earlier));assert(!qwen_push(s,4));uint64_t lineage=s->lineage_id,epoch=s->epoch;
    int position=s->position;QwenWriter w;assert(!qwen_writer_init(&w,s,supplied,1));assert(w.parent_lineage_id==lineage);
    assert(!qwen_stack_push(s,&parent));to_plane(s);assert(s->lineage_id==lineage);assert(!qwen_advance_layer_writer(s,&w,0));complete(s);
    assert(s->lineage_id!=lineage);assert(!qwen_stack_rollback(s,parent));assert(s->lineage_id==lineage);
    /* A retained parent can be replayed even after bookmark serial changes. */
    to_plane(s);assert(!qwen_advance_layer_writer(s,&w,0));complete(s);
    assert(!qwen_stack_rollback(s,earlier));assert(!qwen_push(s,5));
    assert(s->position==position&&s->epoch==epoch&&s->lineage_id!=lineage);
    to_plane(s);QwenSession before=*s;size_t bytes=(size_t)s->model->hidden*4;
    size_t kv_bytes=(size_t)2*s->model->layers*s->model->kvheads*s->model->head*4;
    float *x=malloc(bytes),*after=malloc(bytes),*kv=malloc(kv_bytes),*kv_after=malloc(kv_bytes);assert(x&&after&&kv&&kv_after);
    gpu_read(x,s->x,bytes);writer_current_kv(s,kv);
    assert(qwen_advance_layer_writer(s,&w,0)==-1);assert(!memcmp(&before,s,sizeof(*s)));
    gpu_read(after,s->x,bytes);writer_current_kv(s,kv_after);assert(!memcmp(x,after,bytes)&&!memcmp(kv,kv_after,kv_bytes));
    free(x);free(after);free(kv);free(kv_after);qwen_writer_free(&w);assert(!qwen_reset(s));
}
static void writer_mechanics(QwenSession *a,QwenSession *b){
    int d=a->model->hidden,h=a->model->intermediate;size_t bytes=(size_t)d*4;
    double *columns=calloc((size_t)3*d,sizeof(double));assert(columns);for(int k=0;k<3;k++)columns[k*d+k]=1;
    QwenSubspace supplied={0};supplied.model=a->model;supplied.ready=1;supplied.rank=3;supplied.layer=0;supplied.q0=columns;supplied.q1=columns+d;supplied.q2=columns+2*d;
    prefix(a);prefix(b);QwenWriter w;assert(!qwen_writer_init(&w,a,&supplied,1));
    assert(w.compile_layer==0&&w.layer==1);QwenCheckpoint parent;assert(!qwen_checkpoint(a,&parent));
    to_plane(a);QwenSession before=*a;
    assert(qwen_advance_layer_writer(a,&w,4)==-1);assert(!memcmp(&before,a,sizeof(*a)));
    w.layer=0;assert(qwen_advance_layer_writer(a,&w,0)==-1);assert(!memcmp(&before,a,sizeof(*a)));w.layer=1;
    uint64_t id=w.session_id;w.session_id=b->session_id;assert(qwen_advance_layer_writer(a,&w,0)==-1);assert(!memcmp(&before,a,sizeof(*a)));w.session_id=id;
    w.position++;assert(qwen_advance_layer_writer(a,&w,0)==-1);assert(!memcmp(&before,a,sizeof(*a)));w.position--;
    w.epoch++;assert(qwen_advance_layer_writer(a,&w,0)==-1);assert(!memcmp(&before,a,sizeof(*a)));w.epoch--;
    assert(!qwen_advance_layer_writer(a,&w,0));complete(a);assert(!qwen_push(b,9));equal(a,b);
    WriterProbe p={.d=d};for(int k=0;k<6;k++){p.v[k]=malloc(bytes);assert(p.v[k]);}
    float *entry=malloc(bytes),*reference=malloc(bytes);assert(entry&&reference);
    for(unsigned policy=0;policy<4;policy++){
        assert(!qwen_restore(a,&parent));to_plane(a);gpu_read(entry,a->x,bytes);a->observer=writer_probe;a->observer_context=&p;a->observe_position=3;a->observe_layer=1;
        assert(!qwen_advance_layer_writer(a,&w,policy));a->observer=NULL;
        for(int j=0;j<d;j++){
            assert(p.v[1][j]==((policy&1)&&j>=3?0:p.v[0][j]));
            assert(p.v[4][j]==((policy&2)&&j>=3?0:p.v[3][j]));
            float receiver=entry[j]+p.v[1][j],out=p.v[2][j]+p.v[4][j];assert(receiver==p.v[2][j]&&out==p.v[5][j]);
        }
        /* Independent wiring check: recompute M on the observed edited receiver.
         * This catches projecting the stale pre-edit M in the both policy. */
        const QwenLayer *l=a->model->layer+1;gpu_write(b->x,p.v[2],bytes);gpu_rms(b->norm,b->x,l->post_norm,d,a->model->epsilon);
        gpu_matvec(a->model->gpu,l->gate,d,h,b->norm,b->gate);gpu_matvec(a->model->gpu,l->up,d,h,b->norm,b->up);gpu_swiglu(b->gate,b->up,h);
        gpu_matvec(a->model->gpu,l->down,h,d,b->gate,b->projected);gpu_read(reference,b->projected,bytes);assert(!memcmp(reference,p.v[3],bytes));complete(a);
    }
#ifndef TCM_REAL_MODEL
    for(unsigned policy=1;policy<=2;policy++){
        assert(!qwen_restore(a,&parent));to_plane(a);gpu_read(entry,a->x,bytes);w.basis[0]=NAN;
        assert(qwen_advance_layer_writer(a,&w,policy)==-2);gpu_read(reference,a->x,bytes);assert(!memcmp(entry,reference,bytes));assert(a->phase==2&&a->layer==1);
        w.basis[0]=1;assert(!qwen_advance_layer(a));complete(a);
    }
#endif
    for(int k=0;k<6;k++)free(p.v[k]);free(entry);free(reference);qwen_writer_free(&w);
    writer_lineage_mechanics(a,&supplied);free(columns);prefix(a);prefix(b);
}
#endif
