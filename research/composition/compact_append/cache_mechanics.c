/* Exercise the actual patched runtime against the established tiny CPU fixture. */
#include "toy_fixture.h"
typedef struct {QwenSession *s;int cursor,fail;uint64_t owner,epoch;float keys[2*32*2],values[2*32*2];} ToyCache;
static int toy_step(void *context,int layer,int position,const float *q,const float *k,const float *v,float *out,float *scores){
    ToyCache *c=context;assert(layer==1&&c->s->session_id==c->owner&&c->s->epoch==c->epoch&&c->cursor==position);gpu_cache(c->keys,k,2,2,32,position);gpu_cache(c->values,v,2,2,32,position);
    if(c->fail){c->fail=0;return -77;}c->cursor=position+1;gpu_attention(c->s->model->gpu,out,q,c->keys,c->values,scores,4,2,2,32,c->cursor);return 0;
}
static void toy_rewind(void *context,int position,uint64_t owner,uint64_t epoch){ToyCache *c=context;assert(owner==c->owner&&position<=c->cursor);if(epoch!=c->epoch){assert(position==0&&epoch==c->epoch+1);c->epoch=epoch;}c->cursor=position;}
int main(void){
    QwenModel m;fixture(&m);QwenSession a,b;ToyCache c={0};c.s=&b;assert(!qwen_session_init(&a,&m,32));assert(!qwen_session_init_layout(&b,&m,32,UINT64_C(1)<<1,toy_step,toy_rewind,&c));c.owner=b.session_id;c.epoch=b.epoch;
    assert(a.dense_layers==3&&b.dense_layers==2&&b.cache_slots[0]==0&&b.cache_slots[1]==-1&&b.cache_slots[2]==1);assert(a.allocated_bytes-b.allocated_bytes==2*2*32*2*4);
    for(int t=2;t<=4;t++){assert(!qwen_push(&a,t)&&!qwen_push(&b,t));assert(c.cursor==b.position);assert(a.pending==b.pending&&!memcmp(a.logits,b.logits,16*4));}
    uint64_t marker;assert(!qwen_stack_push(&b,&marker));QwenCheckpoint old={0};assert(!qwen_checkpoint(&b,&old));assert(qwen_restore(&b,&old)==-1);int parent=b.position,pending=b.pending;assert(!qwen_push(&b,7)&&c.cursor==parent+1);float y[16];memcpy(y,b.logits,sizeof(y));assert(!qwen_stack_rollback(&b,marker)&&c.cursor==parent&&b.phase==0);
    assert(!qwen_stack_push(&b,&marker));c.fail=1;assert(!qwen_begin_token(&b,7));assert(!qwen_advance_layer(&b));assert(qwen_advance_layer(&b)==-77&&b.phase==2&&c.cursor==parent);assert(qwen_stack_rollback(&b,marker)==-1);assert(!qwen_abort_token(&b)&&b.position==parent&&b.pending==pending&&b.phase==0&&c.cursor==parent);assert(qwen_abort_token(&b)==-1);assert(!qwen_stack_rollback(&b,marker));
    assert(!qwen_push(&b,7)&&!memcmp(y,b.logits,sizeof(y)));QwenCheckpoint checkpoint;assert(!qwen_checkpoint(&b,&checkpoint));assert(!qwen_push(&b,8)&&!qwen_restore(&b,&checkpoint)&&c.cursor==checkpoint.position);assert(!qwen_reset(&b)&&!c.cursor&&b.epoch==c.epoch);
    float x[128]={0},scale;int8_t code[128];x[0]=127;x[1]=.5f;x[2]=-.5f;x[3]=1.5f;x[4]=-1.5f;assert(!codec_row(x,code,&scale)&&scale==1&&code[1]==1&&code[2]==-1&&code[3]==2&&code[4]==-2);memset(x,0,sizeof(x));assert(!codec_row(x,code,&scale)&&scale==1);for(int d=0;d<128;d++)assert(!code[d]);x[0]=NAN;assert(codec_row(x,code,&scale)==-1);
    qwen_session_free(&a);qwen_session_free(&b);fixture_free(&m);puts("actual layout, immediate current row, rollback, abort, replay, latest checkpoint, reset and codec pass");return 0;
}
