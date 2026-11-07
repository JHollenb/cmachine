/* Actual source capture, native no-op, finite anchor identity and parent guard. */
typedef struct { QwenGateLaw *law; const QwenModel *model; int position; } GateCapture;
static void gate_capture(void *context,int position,int layer,QwenPort port,const float *data,size_t count){
    GateCapture *capture=context;if(port!=QWEN_MLP_GATE)return;
    assert(position==capture->position&&layer==1&&count==(size_t)capture->model->intermediate);
    assert(!qwen_gate_law_init(capture->law,capture->model,layer,position,data));
}
static void gate_mechanics(QwenSession *a,QwenSession *b){
    QwenGateLaw law={0};QwenGateWork work;QwenCheckpoint parent;prefix(a);prefix(b);assert(!qwen_checkpoint(a,&parent));
    GateCapture capture={&law,a->model,a->position};a->observer=gate_capture;a->observer_context=&capture;a->observe_position=a->position;a->observe_layer=1;
    assert(!qwen_push(a,9));a->observer=NULL;a->observer_context=NULL;assert(law.ready&&law.gpu_bytes==(size_t)5*a->model->intermediate*4);
    assert(!qwen_push(b,9));equal(a,b);
    for(int mode=QWEN_GATE_NATIVE;mode<=QWEN_GATE_POLYNOMIAL;mode++){
        assert(!qwen_restore(a,&parent));to_plane(a);QwenSession before=*a;
        size_t bytes=(size_t)a->model->hidden*4;float *x=malloc(bytes),*after=malloc(bytes);assert(x&&after);gpu_read(x,a->x,bytes);
        assert(qwen_advance_layer_gate(a,&law,(QwenGateMode)mode,a->lineage_id+1,&work)==-1);assert(!memcmp(&before,a,sizeof(*a)));gpu_read(after,a->x,bytes);assert(!memcmp(x,after,bytes));free(x);free(after);
        assert(!qwen_advance_layer_gate(a,&law,(QwenGateMode)mode,a->lineage_id,&work));
        assert(work.linear.gate_up_matvecs==2&&work.linear.down_matvecs==1&&!work.linear.compile_matvecs);
        assert(work.native_swiglu==(mode==QWEN_GATE_NATIVE)&&work.polynomial_swiglu==(mode==QWEN_GATE_POLYNOMIAL));complete(a);equal(a,b);
    }
    assert(!qwen_restore(a,&parent));to_plane(a);assert(!qwen_advance_layer_gate(a,&law,QWEN_GATE_ZERO,a->lineage_id,&work));
    assert(!work.linear.gate_up_matvecs&&!work.linear.down_matvecs&&!work.native_swiglu&&!work.polynomial_swiglu);
    float *write=malloc((size_t)a->model->hidden*4);assert(write);gpu_read(write,a->projected,(size_t)a->model->hidden*4);for(int i=0;i<a->model->hidden;i++)assert(write[i]==0);free(write);complete(a);
    qwen_gate_law_free(&law);assert(!law.ready&&!law.model);prefix(a);prefix(b);
}
