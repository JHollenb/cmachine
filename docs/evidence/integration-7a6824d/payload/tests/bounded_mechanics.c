#include <assert.h>
#include <stdio.h>
#define REQUIRE(x) assert(x)
#include "qwen.h"
#include "byte_codec.h"
#include "bounded_cache.h"
int gpu_encode_current(signed char *k,signed char *v,float *s,const float *kr,const float *vr,int cap,int p,int *invalid){
    *invalid=0;for(int kind=0;kind<2;kind++)for(int h=0;h<2;h++)if(codec_row((kind?vr:kr)+h*128,(kind?v:k)+((size_t)h*cap+p)*128,s+(kind*2+h)*cap+p))return -1;return 0;
}
void gpu_ring_attention(TcmGpu *g,float *out,const float *q,const signed char *k,const signed char *v,const float *s,float *scores,float *workspace,const int *order,const float *weights,int cap,int n,int weighted){
    (void)g;(void)q;(void)k;(void)scores;(void)workspace;(void)weighted;float sum=0,mass=0;for(int i=0;i<n;i++){float w=expf(weights[i]);mass+=w;sum+=w*(float)v[order[i]*128]*s[2*cap+order[i]];}for(int i=0;i<1536;i++)out[i]=sum/mass;
}
static void fingerprint(Ring *c,unsigned char *out){memcpy(out,c->arena,c->arena_bytes);memcpy(out+c->arena_bytes,c->positions,(size_t)c->capacity*4);}
static int advance(Ring *c,int token,int fail){
    assert(!ring_begin(c,token));float k[256],v[256],q[1536]={0},out[1536];for(int i=0;i<256;i++){k[i]=(float)((c->session->position+i)%31)-15;v[i]=(float)((token+i)%19)-9;}
    for(int l=0;l<28;l++){c->session->layer=l;if(fail&&l==27){c->fail_layer=27;c->fail_position=c->session->position;}int e=ring_step(c,l,c->session->position,q,k,v,out,c->session->scores);if(e){assert(e==-77);assert(!ring_abort(c));return e;}
        if(l==17&&c->sample_slot>=0){size_t n=c->arena_bytes+(size_t)c->capacity*4;unsigned char *before=malloc(n),*after=malloc(n);assert(before&&after);fingerprint(c,before);assert(!ring_step(c,l,c->session->position,q,k,v,out,c->session->scores));fingerprint(c,after);assert(!memcmp(before,after,n));free(before);free(after);}
        c->session->x[0]+=out[0]*.001f;}
    c->session->layer=28;return ring_finish(c);
}
int main(void){
    QwenModel m={.hidden=4,.layers=28,.qheads=12,.kvheads=2,.head=128,.intermediate=8,.vocab=16,.max_position=4096,.epsilon=1e-6f};m.gpu=gpu_open();float embedding[64],norm[4]={1,1,1,1};for(int i=0;i<64;i++)embedding[i]=((i%7)-3)*.1f;m.embedding=embedding;m.norm=norm;
    QwenSession s,t;Ring c,d;ring_open(&c,&s,&m,1024,2,4,3,0);ring_open(&d,&t,&m,2048,2,4,3,0);assert(c.gpu_bytes==d.gpu_bytes&&s.allocated_bytes==t.allocated_bytes&&c.host_bytes==d.host_bytes);ring_close(&d);qwen_session_free(&t);
    for(int i=0;i<60;i++)assert(!advance(&c,i%16,0));assert(c.view.population==54&&c.active==9);size_t bytes=c.arena_bytes+(size_t)c.capacity*4;unsigned char *before=malloc(bytes),*after=malloc(bytes);fingerprint(&c,before);uint64_t parent,child;assert(!ring_mark(&c,&parent));assert(!advance(&c,7,0)&&!ring_mark(&c,&child)&&!advance(&c,8,0));float y[16];memcpy(y,s.logits,sizeof(y));assert(!ring_rollback(&c,child)&&!advance(&c,9,0)&&!ring_rollback(&c,parent));fingerprint(&c,after);assert(!memcmp(before,after,bytes));
    assert(!ring_mark(&c,&parent)&&!advance(&c,7,0)&&!advance(&c,8,0)&&!memcmp(y,s.logits,sizeof(y))&&!ring_rollback(&c,parent));assert(ring_rollback(&c,parent));
    assert(!ring_mark(&c,&parent));uint64_t lineage=s.frames[s.depth-1].state.lineage_id;s.frames[s.depth-1].state.lineage_id++;assert(ring_rollback(&c,parent));s.frames[s.depth-1].state.lineage_id=lineage;assert(!ring_rollback(&c,parent));
    int tested=0;for(int attempt=0;attempt<128&&!tested;attempt++){
        fingerprint(&c,before);RingView view=c.view;float logits[16];memcpy(logits,s.logits,sizeof(logits));int pending=s.pending;TcmTop2 top=s.top;assert(!ring_mark(&c,&parent));assert(!ring_begin(&c,5));int replacement=c.sample_slot>=0&&c.cuts[0].positions[c.sample_slot]>=0;assert(!ring_abort(&c));
        if(replacement){assert(advance(&c,5,1)==-77);fingerprint(&c,after);assert(!memcmp(before,after,bytes)&&view.cursor==c.view.cursor&&view.population==c.view.population&&view.rng==c.view.rng&&!memcmp(logits,s.logits,sizeof(logits))&&pending==s.pending&&!memcmp(&top,&s.top,sizeof(top)));tested=1;assert(!advance(&c,5,0));float replay[16];memcpy(replay,s.logits,sizeof(replay));assert(!ring_rollback(&c,parent)&&!ring_mark(&c,&parent)&&!advance(&c,5,0)&&!memcmp(replay,s.logits,sizeof(replay))&&!ring_rollback(&c,parent));}
        else{assert(!ring_commit(&c,parent)&&!advance(&c,5,0));}
    }assert(tested);
    QwenCheckpoint cp;assert(!qwen_checkpoint(&s,&cp));QwenSession old=s;assert(qwen_restore(&s,&cp)&&!memcmp(&old,&s,sizeof(s)));assert(qwen_begin_token(&s,3)&&!memcmp(&old,&s,sizeof(s)));assert(!ring_mark(&c,&parent)&&!ring_mark(&c,&child));old=s;uint64_t extra;assert(ring_mark(&c,&extra)&&ring_rollback(&c,parent)&&!memcmp(&old,&s,sizeof(s)));assert(!ring_commit(&c,child)&&!ring_commit(&c,parent));assert(!qwen_reset(&s)&&!c.view.cursor&&!c.view.population);for(int i=0;i<c.capacity;i++)assert(c.positions[i]==-1);assert(ring_rollback(&c,parent));ring_close(&c);qwen_session_free(&s);gpu_close(m.gpu);free(before);free(after);puts("bounded allocation, own current rows, nested overwrite/replay, accepted-sample abort/RNG/logit restoration, refusals and reset pass");return 0;
}
