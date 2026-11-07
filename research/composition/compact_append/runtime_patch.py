"""Small explicit transform of the authenticated C runtime for cache ownership."""
from pathlib import Path

def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError("Pinned runtime patch context differs: "+old[:80])
    return text.replace(old,new,1)

def transform(name,text):
    if name=="qwen.h":
        text=replace_once(text,"typedef struct {\n    const QwenModel *model;\n    int position,capacity,pending,layer,phase,delivered;",'''typedef int (*QwenCacheStep)(void *,int,int,const float *,const float *,const float *,float *,float *);
typedef void (*QwenCacheRewind)(void *,int,uint64_t,uint64_t);
typedef struct {
    const QwenModel *model;
    int position,capacity,pending,layer,phase,delivered;''')
        text=replace_once(text,"    int observe_position,observe_layer;\n} QwenSession;",'''    int observe_position,observe_layer;
    uint64_t cache_mask;
    int cache_slots[64],dense_layers;
    QwenCacheStep cache_step;
    QwenCacheRewind cache_rewind;
    void *cache_context;
    QwenCheckpoint token_parent;
    int token_parent_valid;
} QwenSession;''')
        text=replace_once(text,"int qwen_session_init(QwenSession *,const QwenModel *,int);",'''int qwen_session_init(QwenSession *,const QwenModel *,int);
int qwen_session_init_layout(QwenSession *,const QwenModel *,int,uint64_t,QwenCacheStep,QwenCacheRewind,void *);
int qwen_abort_token(QwenSession *);''')
    elif name=="qwen.c":
        text=replace_once(text,"int qwen_session_init(QwenSession *s,const QwenModel *m,int capacity){",'''static void snapshot(const QwenSession *,QwenCheckpoint *);
static void restore(QwenSession *,const QwenCheckpoint *);
int qwen_session_init(QwenSession *s,const QwenModel *m,int capacity){return qwen_session_init_layout(s,m,capacity,0,NULL,NULL,NULL);}
int qwen_session_init_layout(QwenSession *s,const QwenModel *m,int capacity,uint64_t mask,QwenCacheStep step,QwenCacheRewind rewind,void *context){''')
        text=replace_once(text,"    size_t kv=(size_t)m->layers*m->kvheads*capacity*m->head;",'''    if(m->layers<1||m->layers>64||(m->layers<64&&(mask>>m->layers))||(mask&&(!step||!rewind)))return -1;
    s->cache_mask=mask;s->cache_step=step;s->cache_rewind=rewind;s->cache_context=context;
    for(int i=0;i<m->layers;i++)s->cache_slots[i]=(mask&(UINT64_C(1)<<i))?-1:s->dense_layers++;
    size_t kv=(size_t)s->dense_layers*m->kvheads*capacity*m->head;''')
        text=replace_once(text,"    s->phase=2;s->layer=0;gpu_copy", "    snapshot(s,&s->token_parent);s->token_parent_valid=1;\n    s->phase=2;s->layer=0;gpu_copy")
        old='''        size_t base=(size_t)i*m->kvheads*s->capacity*m->head;
        gpu_cache(s->keys+base,s->k,m->kvheads,m->head,s->capacity,s->position);gpu_cache(s->values+base,s->v,m->kvheads,m->head,s->capacity,s->position);
        gpu_attention(m->gpu,s->attention,s->q,s->keys+base,s->values+base,s->scores,m->qheads,m->kvheads,m->head,s->capacity,s->position+1);'''
        new='''        if(s->cache_slots[i]<0){
            if(!s->cache_step)return -1;
            int error=s->cache_step(s->cache_context,i,s->position,s->q,s->k,s->v,s->attention,s->scores);if(error)return error;
        }else{
            size_t base=(size_t)s->cache_slots[i]*m->kvheads*s->capacity*m->head;
            gpu_cache(s->keys+base,s->k,m->kvheads,m->head,s->capacity,s->position);gpu_cache(s->values+base,s->v,m->kvheads,m->head,s->capacity,s->position);
            gpu_attention(m->gpu,s->attention,s->q,s->keys+base,s->values+base,s->scores,m->qheads,m->kvheads,m->head,s->capacity,s->position+1);
        }'''
        text=replace_once(text,old,new)
        text=replace_once(text,"s->position++;s->layer=m->layers;s->phase=1;return 0;","s->position++;s->layer=m->layers;s->phase=1;s->token_parent_valid=0;return 0;")
        text=replace_once(text,"static void restore(QwenSession *s,const QwenCheckpoint *c){\n",'''int qwen_abort_token(QwenSession *s){
    if(!s||s->phase!=2||!s->token_parent_valid||s->token_parent.session_id!=s->session_id||s->token_parent.epoch!=s->epoch)return -1;
    restore(s,&s->token_parent);return 0;
}
static void restore(QwenSession *s,const QwenCheckpoint *c){
    if(s->cache_rewind)s->cache_rewind(s->cache_context,c->position,s->session_id,s->epoch);
    s->token_parent_valid=0;
''')
        text=replace_once(text,"s->epoch++;s->checkpoint_serial++;s->checkpoint_position=0;return 0;", "s->epoch++;s->checkpoint_serial++;s->checkpoint_position=0;s->token_parent_valid=0;if(s->cache_rewind)s->cache_rewind(s->cache_context,0,s->session_id,s->epoch);return 0;")
    elif name=="qwen_cuda.cu":
        text=replace_once(text,'extern "C" void *gpu_alloc(size_t n){void *p=NULL;CUDA(cudaMalloc(&p,n));return p;}\nextern "C" void gpu_free(void *p){if(p)CUDA(cudaFree(p));}', '''static void *allocation_pointer[512];
static size_t allocation_extent[512],live_allocation_bytes,peak_allocation_bytes;
extern "C" size_t gpu_live_bytes(void){return live_allocation_bytes;}
extern "C" size_t gpu_peak_bytes(void){return peak_allocation_bytes;}
extern "C" void *gpu_alloc(size_t n){void *p=NULL;CUDA(cudaMalloc(&p,n));int slot=-1;for(int i=0;i<512;i++)if(!allocation_pointer[i]){slot=i;break;}if(slot<0)exit(2);allocation_pointer[slot]=p;allocation_extent[slot]=n;live_allocation_bytes+=n;if(live_allocation_bytes>peak_allocation_bytes)peak_allocation_bytes=live_allocation_bytes;return p;}
extern "C" void gpu_free(void *p){if(p){int slot=-1;for(int i=0;i<512;i++)if(allocation_pointer[i]==p){slot=i;break;}if(slot<0)exit(2);CUDA(cudaFree(p));live_allocation_bytes-=allocation_extent[slot];allocation_pointer[slot]=NULL;allocation_extent[slot]=0;}}''')
    return text

def apply(runtime):
    runtime=Path(runtime)
    for name in ("qwen.h","qwen.c","qwen_cuda.cu"):
        p=runtime/name;p.write_text(transform(name,p.read_text()))
