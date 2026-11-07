"""Small deterministic extension of the authenticated compact-append runtime."""
from pathlib import Path
def transform(name,text):
    if name!="qwen.c":return text
    before='#define ALLOC(field,n) s->field=gpu_alloc((size_t)(n)*sizeof(float));s->allocated_bytes+=(size_t)(n)*sizeof(float)'
    after='#define ALLOC(field,n) s->field=(size_t)(n)?gpu_alloc((size_t)(n)*sizeof(float)):NULL;s->allocated_bytes+=(size_t)(n)*sizeof(float)'
    if text.count(before)!=1:raise ValueError('Allocation transform source differs')
    return text.replace(before,after)
def apply(root):
    root=Path(root);p=root/'qwen.c';p.write_text(transform(p.name,p.read_text()))
