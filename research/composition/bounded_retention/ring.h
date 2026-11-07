#ifndef TCM_BOUNDED_RING_H
#define TCM_BOUNDED_RING_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
extern int gpu_encode_current(signed char*,signed char*,float*,const float*,const float*,int,int,int*);
extern void gpu_ring_attention(TcmGpu*,float*,const float*,const signed char*,const signed char*,const float*,float*,float*,const int*,const float*,int,int,int);
extern size_t gpu_live_bytes(void);
extern size_t gpu_peak_bytes(void);
#define RING_LAYERS 28
#define RING_ROW_BYTES 528
#define RING_CUTS 3
typedef struct {int cursor;uint64_t population,rng;} RingView;
typedef struct {RingView view;int *positions;void *arena;float *logits;uint64_t id,owner,epoch;int valid;} RingCut;
typedef struct {
    QwenSession *session;int anchors,recent,sample,capacity,full,transaction,restore_slot,write_slot,sample_slot,active,fail_layer,fail_position;
    uint64_t owner,epoch;RingView view;int *positions,*order;float *weights;
    void *arena;float *workspace;int *invalid,*device_order;float *device_weights;
    RingCut cuts[RING_CUTS];size_t arena_bytes,gpu_bytes,host_bytes;
    unsigned long long calls,published,restores,main_tokens;
    FILE *rows;
} Ring;
static uint64_t ring_random(Ring *c){uint64_t z=(c->view.rng+=UINT64_C(0x9e3779b97f4a7c15));z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);return z^(z>>31);}
static uint64_t ring_draw(Ring *c,uint64_t n){uint64_t x,t=(uint64_t)(-n)%n;do{x=ring_random(c);}while(x<t);return x%n;}
static signed char *ring_keys(Ring *c,int l){return (signed char*)c->arena+(size_t)l*c->capacity*RING_ROW_BYTES;}
static signed char *ring_values(Ring *c,int l){return ring_keys(c,l)+(size_t)c->capacity*256;}
static float *ring_scales(Ring *c,int l){return (float*)(ring_keys(c,l)+(size_t)c->capacity*512);}
static void ring_save_cut(Ring *c,int slot,uint64_t id){
    RingCut *p=&c->cuts[slot];p->view=c->view;memcpy(p->positions,c->positions,(size_t)c->capacity*4);gpu_copy(p->arena,c->arena,c->arena_bytes);gpu_copy(p->logits,c->session->logits,(size_t)c->session->model->vocab*4);p->id=id;p->owner=c->owner;p->epoch=c->epoch;p->valid=1;
}
static int ring_guard(void *context,int operation,int position,uint64_t owner,uint64_t epoch){
    Ring *c=context;if(owner!=c->owner||epoch!=c->epoch)return -1;
    if(!operation)return !(c->transaction&&c->cuts[0].valid&&c->cuts[0].view.cursor==position);
    int slot=c->restore_slot;if(slot<0||slot>=RING_CUTS)return -1;RingCut *p=&c->cuts[slot];return !(p->valid&&p->owner==owner&&p->epoch==epoch&&p->view.cursor==position);
}
static void ring_rewind(void *context,int position,uint64_t owner,uint64_t epoch){
    Ring *c=context;REQUIRE(owner==c->owner);
    if(epoch!=c->epoch){REQUIRE(epoch==c->epoch+1&&position==0);c->epoch=epoch;c->view=(RingView){.rng=7};for(int i=0;i<c->capacity;i++)c->positions[i]=-1;gpu_zero(c->arena,c->arena_bytes);for(int j=0;j<RING_CUTS;j++)c->cuts[j].valid=0;c->transaction=0;c->restore_slot=-1;return;}
    REQUIRE(!ring_guard(c,1,position,owner,epoch));RingCut *p=&c->cuts[c->restore_slot];gpu_copy(c->arena,p->arena,c->arena_bytes);gpu_copy(c->session->logits,p->logits,(size_t)c->session->model->vocab*4);memcpy(c->positions,p->positions,(size_t)c->capacity*4);c->view=p->view;c->restores++;c->transaction=0;
}
static void ring_copy_row(Ring *c,int layer,int to,int from){
    for(int kind=0;kind<2;kind++)for(int h=0;h<2;h++){
        signed char *p=kind?ring_values(c,layer):ring_keys(c,layer);gpu_copy(p+((size_t)h*c->capacity+to)*128,p+((size_t)h*c->capacity+from)*128,128);
        float *s=ring_scales(c,layer)+(kind*2+h)*c->capacity;gpu_copy(s+to,s+from,4);
    }
}
static void ring_order(Ring *c){
    c->active=0;for(int i=0;i<c->capacity;i++)if(c->positions[i]>=0)c->order[c->active++]=i;
    for(int i=1;i<c->active;i++){int x=c->order[i],j=i;while(j&&c->positions[c->order[j-1]]>c->positions[x]){c->order[j]=c->order[j-1];j--;}c->order[j]=x;}
    int m=c->view.population<(uint64_t)c->sample?(int)c->view.population:c->sample;
    for(int i=0;i<c->active;i++)c->weights[i]=(c->order[i]>=c->anchors+c->recent&&m)?logf((float)c->view.population/(float)m):0;
    gpu_write(c->device_order,c->order,(size_t)c->active*4);gpu_write(c->device_weights,c->weights,(size_t)c->active*4);
}
static int ring_step(void *context,int layer,int position,const float *q,const float *k,const float *v,float *out,float *scores){
    Ring *c=context;QwenSession *s=c->session;if(!c->transaction||layer<0||layer>=28||position!=c->cuts[0].view.cursor||s->position!=position||s->phase!=2||c->view.cursor!=position+1)return -80;
    c->calls++;if(c->sample_slot>=0)ring_copy_row(c,layer,c->sample_slot,c->write_slot);
    if(c->rows&&(layer==0||layer==17||layer==27)){float row[2][256];gpu_read(row[0],k,1024);gpu_read(row[1],v,1024);REQUIRE(fwrite(row,1,sizeof(row),c->rows)==sizeof(row));}
    if(gpu_encode_current(ring_keys(c,layer),ring_values(c,layer),ring_scales(c,layer),k,v,c->capacity,c->write_slot,c->invalid))return -81;
    if(c->fail_layer==layer&&c->fail_position==position){c->fail_layer=-1;return -77;}
    c->published++;gpu_ring_attention(s->model->gpu,out,q,ring_keys(c,layer),ring_values(c,layer),ring_scales(c,layer),scores,c->workspace,c->device_order,c->device_weights,c->capacity,c->active,c->sample&&c->view.population>0);return 0;
}
static void ring_open(Ring *c,QwenSession *s,const QwenModel *m,int logical,int anchors,int recent,int sample,int full){
    memset(c,0,sizeof(*c));c->session=s;c->anchors=anchors;c->recent=recent;c->sample=sample;c->full=full;c->capacity=anchors+recent+sample;c->restore_slot=-1;c->fail_layer=-1;c->view.rng=7;REQUIRE(recent>0&&anchors>=0&&sample>=0&&c->capacity<=logical&&logical<=m->max_position&&m->layers==28&&m->qheads==12&&m->kvheads==2&&m->head==128);
    REQUIRE(!qwen_session_init_layout(s,m,c->capacity,(UINT64_C(1)<<28)-1,ring_step,ring_rewind,c));s->capacity=logical;s->cache_guard=ring_guard;c->owner=s->session_id;c->epoch=s->epoch;gpu_zero(s->logits,(size_t)m->vocab*4);
    c->arena_bytes=(size_t)28*c->capacity*528;c->arena=gpu_alloc(c->arena_bytes);gpu_zero(c->arena,c->arena_bytes);
    c->workspace=gpu_alloc((size_t)2*c->capacity*128*4);c->invalid=gpu_alloc(4);c->device_order=gpu_alloc((size_t)c->capacity*4);c->device_weights=gpu_alloc((size_t)c->capacity*4);
    c->positions=malloc((size_t)c->capacity*4);c->order=malloc((size_t)c->capacity*4);c->weights=malloc((size_t)c->capacity*4);REQUIRE(c->positions&&c->order&&c->weights);for(int i=0;i<c->capacity;i++)c->positions[i]=-1;
    c->gpu_bytes=c->arena_bytes+(size_t)c->capacity*(1024+8)+4;c->host_bytes=sizeof(*c)+(size_t)c->capacity*12;
    for(int j=0;j<RING_CUTS;j++){c->cuts[j].arena=gpu_alloc(c->arena_bytes);c->cuts[j].logits=gpu_alloc((size_t)m->vocab*4);c->cuts[j].positions=malloc((size_t)c->capacity*4);REQUIRE(c->cuts[j].positions);c->gpu_bytes+=c->arena_bytes+(size_t)m->vocab*4;c->host_bytes+=(size_t)c->capacity*4;}
}
static int ring_begin(Ring *c,int token){
    QwenSession *s=c->session;if(c->transaction||s->phase==2||s->position>=s->capacity||token<0||token>=s->model->vocab)return -1;
    ring_save_cut(c,0,0);c->transaction=1;int p=s->position;c->sample_slot=-1;c->write_slot=p<c->anchors?p:c->anchors+(p-c->anchors)%c->recent;
    if(c->positions[c->write_slot]>=0){c->view.population++;if(c->sample){uint64_t n=c->view.population,j=n<=(uint64_t)c->sample?n-1:ring_draw(c,n);if(j<(uint64_t)c->sample){c->sample_slot=c->anchors+c->recent+(int)j;c->positions[c->sample_slot]=c->positions[c->write_slot];}}}
    c->positions[c->write_slot]=p;c->view.cursor=p+1;ring_order(c);
    int e=qwen_begin_token(s,token);REQUIRE(!e);return 0;
}
static int ring_abort(Ring *c){c->restore_slot=0;int e=qwen_abort_token(c->session);c->restore_slot=-1;if(!e)c->cuts[0].valid=0;return e;}
static int ring_finish(Ring *c){int e=qwen_finish_token(c->session);if(!e){c->transaction=0;c->cuts[0].valid=0;c->main_tokens++;}return e;}
static int ring_push(Ring *c,int token){if(ring_begin(c,token))return -1;int e=0;while(c->session->layer<28&&!e)e=qwen_advance_layer(c->session);if(e){REQUIRE(!ring_abort(c));return e;}return ring_finish(c);}
static int ring_mark(Ring *c,uint64_t *id){if(c->transaction||c->session->depth>=2)return -1;int slot=(int)c->session->depth+1;if(qwen_stack_push(c->session,id))return -1;ring_save_cut(c,slot,*id);return 0;}
static int ring_rollback(Ring *c,uint64_t id){
    QwenSession *s=c->session;if(c->transaction||!s->depth||s->depth>2)return -1;int slot=(int)s->depth;RingCut *p=&c->cuts[slot];
    if(!p->valid||p->id!=id||p->owner!=s->session_id||p->epoch!=s->epoch||s->frames[s->depth-1].id!=id)return -1;
    c->restore_slot=slot;int e=qwen_stack_rollback(s,id);c->restore_slot=-1;if(!e)p->valid=0;return e;
}
static int ring_commit(Ring *c,uint64_t id){QwenSession *s=c->session;if(c->transaction||!s->depth||s->depth>2)return -1;RingCut *p=&c->cuts[s->depth];if(!p->valid||p->id!=id||qwen_stack_commit(s,id))return -1;p->valid=0;return 0;}
static void ring_close(Ring *c){gpu_free(c->arena);gpu_free(c->workspace);gpu_free(c->invalid);gpu_free(c->device_order);gpu_free(c->device_weights);free(c->positions);free(c->order);free(c->weights);for(int j=0;j<RING_CUTS;j++){gpu_free(c->cuts[j].arena);gpu_free(c->cuts[j].logits);free(c->cuts[j].positions);}memset(c,0,sizeof(*c));}
#endif
