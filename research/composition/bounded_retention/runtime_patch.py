"""Guard physical cache recovery before the original session mutates metadata."""
from pathlib import Path
def once(text,before,after):
    if text.count(before)!=1:raise ValueError('Recovery transform context differs: '+before[:100])
    return text.replace(before,after)
def transform(name,text):
    if name=='qwen.h':
        text=once(text,'typedef void (*QwenCacheRewind)', 'typedef int (*QwenCacheGuard)(void *,int,int,uint64_t,uint64_t);\ntypedef void (*QwenCacheRewind)')
        return once(text,'    QwenCacheRewind cache_rewind;','    QwenCacheRewind cache_rewind;\n    QwenCacheGuard cache_guard;')
    if name!='qwen.c':return text
    text=once(text,'    snapshot(s,&s->token_parent);s->token_parent_valid=1;', '    if(s->cache_guard&&s->cache_guard(s->cache_context,0,s->position,s->session_id,s->epoch))return -1;\n    snapshot(s,&s->token_parent);s->token_parent_valid=1;')
    text=once(text,'    restore(s,&s->token_parent);return 0;', '    if(s->cache_guard&&s->cache_guard(s->cache_context,1,s->token_parent.position,s->session_id,s->epoch))return -1;\n    restore(s,&s->token_parent);return 0;')
    text=once(text,'    restore(s,c);return 0;', '    if(s->cache_guard&&s->cache_guard(s->cache_context,1,c->position,s->session_id,s->epoch))return -1;\n    restore(s,c);return 0;')
    text=once(text,'    restore(s,&s->frames[--s->depth].state);', '    if(s->cache_guard&&s->cache_guard(s->cache_context,1,s->frames[s->depth-1].state.position,s->session_id,s->epoch))return -1;\n    restore(s,&s->frames[--s->depth].state);')
    return text
def apply(root):
    root=Path(root)
    for name in ('qwen.h','qwen.c'):
        p=root/'runtime'/name;p.write_text(transform(name,p.read_text()))
