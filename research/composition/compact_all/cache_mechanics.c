#include "toy_fixture.h"
typedef struct {QwenSession *s;int cursor[3],fail;uint64_t owner,epoch;float keys[3][2*32*2],values[3][2*32*2];} ToyAll;
static int toy_step(void *context,int layer,int position,const float *q,const float *k,const float *v,float *out,float *scores){
    ToyAll *c=context;assert(layer>=0&&layer<3&&c->s->session_id==c->owner&&c->s->epoch==c->epoch&&c->cursor[layer]==position);gpu_cache(c->keys[layer],k,2,2,32,position);gpu_cache(c->values[layer],v,2,2,32,position);
    if(c->fail&&layer==2){c->fail=0;return -77;}c->cursor[layer]=position+1;gpu_attention(c->s->model->gpu,out,q,c->keys[layer],c->values[layer],scores,4,2,2,32,c->cursor[layer]);return 0;
}
static void toy_rewind(void *context,int position,uint64_t owner,uint64_t epoch){ToyAll *c=context;assert(owner==c->owner);if(epoch!=c->epoch){assert(position==0&&epoch==c->epoch+1);c->epoch=epoch;}for(int l=0;l<3;l++){assert(position<=c->cursor[l]);c->cursor[l]=position;}}
int main(void){
    QwenModel m;fixture(&m);QwenSession a,b;ToyAll c={.s=&b};assert(!qwen_session_init(&a,&m,32));assert(!qwen_session_init_layout(&b,&m,32,7,toy_step,toy_rewind,&c));c.owner=b.session_id;c.epoch=b.epoch;
    assert(a.dense_layers==3&&!b.dense_layers&&!b.keys&&!b.values);for(int l=0;l<3;l++)assert(b.cache_slots[l]==-1);assert(a.allocated_bytes-b.allocated_bytes==3*2*2*32*2*4);
    for(int t=2;t<=4;t++){assert(!qwen_push(&a,t)&&!qwen_push(&b,t));for(int l=0;l<3;l++)assert(c.cursor[l]==b.position);assert(a.pending==b.pending&&!memcmp(a.logits,b.logits,16*4));}
    uint64_t marker;assert(!qwen_stack_push(&b,&marker));int p=b.position,pending=b.pending;assert(!qwen_push(&b,7));float y[16];memcpy(y,b.logits,sizeof(y));assert(!qwen_stack_rollback(&b,marker));for(int l=0;l<3;l++)assert(c.cursor[l]==p);
    assert(!qwen_stack_push(&b,&marker));c.fail=1;assert(!qwen_begin_token(&b,7)&&!qwen_advance_layer(&b)&&!qwen_advance_layer(&b));assert(qwen_advance_layer(&b)==-77&&c.cursor[0]==p+1&&c.cursor[1]==p+1&&c.cursor[2]==p);assert(!qwen_abort_token(&b)&&b.position==p&&b.pending==pending&&b.phase==0);for(int l=0;l<3;l++)assert(c.cursor[l]==p);assert(!qwen_stack_rollback(&b,marker));assert(!qwen_push(&b,7)&&!memcmp(y,b.logits,sizeof(y)));
    assert(!qwen_reset(&b));for(int l=0;l<3;l++)assert(!c.cursor[l]);assert(c.epoch==b.epoch);qwen_session_free(&a);qwen_session_free(&b);fixture_free(&m);puts("all-layer zero dense allocation, current rows, rollback, abort, replay and reset pass");return 0;
}
