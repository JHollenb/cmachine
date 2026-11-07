/* Real decoder overwrite/abort/replay; executed only through a CUDA lease. */
#include "qwen.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define REQUIRE(x) assert(x)
#include "bounded_cache.h"
static void fingerprint(Ring *c,char text[65]){
    size_t n=c->arena_bytes+(size_t)c->capacity*4+(size_t)c->session->model->vocab*4;
    unsigned char *p=malloc(n),raw[32];assert(p);gpu_read(p,c->arena,c->arena_bytes);memcpy(p+c->arena_bytes,c->positions,(size_t)c->capacity*4);gpu_read(p+c->arena_bytes+(size_t)c->capacity*4,c->session->logits,(size_t)c->session->model->vocab*4);tcm_sha256(p,n,raw);tcm_hex(raw,32,text);free(p);
}
int main(int argc,char **argv){
    assert(argc==2);gpu_weight_configure(2);TcmModel bundle;QwenModel m;QwenSession s;Ring c;char error[256];assert(!tcm_load(&bundle,argv[1],error,sizeof(error)));assert(!qwen_model_init(&m,&bundle,error,sizeof(error)));tcm_unload(&bundle);ring_open(&c,&s,&m,768,16,64,32,0);
    for(int i=0;i<300;i++)assert(!ring_push(&c,1+i%13));int tested=0;
    for(int attempt=0;attempt<128&&!tested;attempt++){
        uint64_t parent;assert(!ring_mark(&c,&parent));char before[65],after[65],child[65];fingerprint(&c,before);RingView view=c.view;int position=s.position,pending=s.pending,delivered=s.delivered;uint64_t lineage=s.lineage_id;
        assert(!ring_begin(&c,5));int occupied=c.sample_slot>=0&&c.cuts[0].positions[c.sample_slot]>=0;
        if(!occupied){assert(!ring_abort(&c)&&!ring_commit(&c,parent)&&!ring_push(&c,5));continue;}
        c.fail_layer=27;c.fail_position=position;int result=0;while(s.layer<28&&!result)result=qwen_advance_layer(&s);assert(result==-77&&!ring_abort(&c));fingerprint(&c,after);
        assert(!strcmp(before,after)&&s.position==position&&s.pending==pending&&s.delivered==delivered&&s.lineage_id==lineage&&c.view.cursor==view.cursor&&c.view.population==view.population&&c.view.rng==view.rng);
        assert(!ring_push(&c,5));fingerprint(&c,child);assert(!ring_rollback(&c,parent));assert(s.lineage_id==lineage);assert(!ring_mark(&c,&parent)&&!ring_push(&c,5));fingerprint(&c,after);assert(!strcmp(child,after)&&!ring_rollback(&c,parent));
        /* Refuse AFTER the selected cache read, then retry that same layer. */
        assert(!ring_mark(&c,&parent)&&!ring_begin(&c,5));while(s.layer<17)assert(!qwen_advance_layer(&s));QwenPlane plane;QwenPlaneWork work;assert(!qwen_plane_init(&plane,&m));plane.ready=1;plane.session_id=s.session_id;plane.epoch=s.epoch;plane.position=s.position;plane.layer=17;gpu_copy(plane.basis[2],s.x,(size_t)m.hidden*4);gpu_zero(plane.basis[1],(size_t)m.hidden*4);float bad=NAN;gpu_write(plane.basis[1],&bad,4);
        assert(qwen_advance_layer_plane(&s,&plane,QWEN_PLANE_DENSE,0,0,&work)==-2&&s.layer==17);assert(!qwen_advance_layer(&s));while(s.layer<28)assert(!qwen_advance_layer(&s));assert(!ring_finish(&c));fingerprint(&c,after);assert(!strcmp(child,after));qwen_plane_free(&plane);assert(!ring_rollback(&c,parent));tested=1;
    }
    assert(tested);printf("{\"occupied_sample_abort\":true,\"cache_logits_rng_lineage_restored\":true,\"exact_child_replay\":true,\"same_layer_retry_exact\":true,\"position\":%d,\"state_gpu_bytes\":%zu}\n",s.position,s.allocated_bytes+c.gpu_bytes);ring_close(&c);qwen_session_free(&s);qwen_model_free(&m);gpu_weight_release();assert(!gpu_live_bytes());return 0;
}
