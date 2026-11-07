"""Reproducible owned-weight API and embedding dispatch over the pinned all-cache runtime."""
from pathlib import Path
def once(text,before,after):
    if text.count(before)!=1:raise ValueError('Weight transform context differs: '+before[:90])
    return text.replace(before,after)
def transform(name,text):
    if name!='qwen.c':return text
    text=once(text,'static float *weight(QwenModel *m,', 'extern float *gpu_owned_weight(TcmGpu *,const void *,uint32_t,size_t,int,int,int);\nextern size_t gpu_owned_weight_bytes(const float *,size_t);\nextern void gpu_owned_embedding(float *,const float *,int,int);\nstatic float *weight(QwenModel *m,')
    text=once(text,'float *p=gpu_weight(t->data,t->dtype,count);m->allocations[m->allocation_count++]=p;m->weight_bytes+=count*4;','float *p=gpu_owned_weight(m->gpu,t->data,t->dtype,count,rank,a,c);m->allocations[m->allocation_count++]=p;m->weight_bytes+=gpu_owned_weight_bytes(p,count);')
    return once(text,'gpu_copy(s->x,m->embedding+(size_t)token*m->hidden,(size_t)m->hidden*4);','gpu_owned_embedding(s->x,m->embedding,token,m->hidden);')
def driver_transform(text):
    text=once(text,'static const char *directory;','extern void gpu_weight_configure(int);\nextern void gpu_weight_release(void);\nextern void gpu_weight_info(size_t*,size_t*,size_t*,int*,unsigned long long*,unsigned long long*,unsigned long long*);\nstatic const char *directory;')
    text=once(text,'REQUIRE(argc==8);directory=argv[7];','REQUIRE(argc==9);int weight_profile=atoi(argv[8]);REQUIRE(weight_profile>=0&&weight_profile<=2);gpu_weight_configure(weight_profile);directory=argv[7];')
    text=once(text,'for(int mode=0;mode<3;mode++){','for(int mode=2;mode<3;mode++){')
    # The list's sole entry is still mode2, so suppress the native-mode comma convention.
    text=once(text,"if(mode)fputc(',',f);fprintf(f,",'fprintf(f,')
    before='fprintf(f,"],\\\"model_weight_bytes\\\":%zu,\\\"peak_tracked_gpu_bytes\\\":%zu,\\\"seconds\\\":%.9g}\\n",m.weight_bytes,gpu_peak_bytes(),clock_seconds()-started);'
    # Build a direct exact literal to avoid source-code escaping ambiguity.
    before='fprintf(f,\"],\\\"model_weight_bytes\\\":%zu,\\\"peak_tracked_gpu_bytes\\\":%zu,\\\"seconds\\\":%.9g}\\n\",m.weight_bytes,gpu_peak_bytes(),clock_seconds()-started);'
    after='size_t matrix_bytes,vector_bytes,scratch_bytes;int matrix_count;unsigned long long matvecs,tiles,embeddings;gpu_weight_info(&matrix_bytes,&vector_bytes,&scratch_bytes,&matrix_count,&matvecs,&tiles,&embeddings);fprintf(f,\"],\\\"weight_profile\\\":%d,\\\"matrix_count\\\":%d,\\\"permanent_matrix_bytes\\\":%zu,\\\"rank1_bytes\\\":%zu,\\\"weight_tile_bytes\\\":%zu,\\\"matvec_calls\\\":%llu,\\\"tile_calls\\\":%llu,\\\"embedding_calls\\\":%llu,\\\"model_weight_bytes\\\":%zu,\\\"peak_tracked_gpu_bytes\\\":%zu,\\\"seconds\\\":%.9g}\\n\",weight_profile,matrix_count,matrix_bytes,vector_bytes,scratch_bytes,matvecs,tiles,embeddings,m.weight_bytes,gpu_peak_bytes(),clock_seconds()-started);'
    text=once(text,before,after)
    return once(text,'qwen_model_free(&m);REQUIRE(!gpu_live_bytes());','qwen_model_free(&m);gpu_weight_release();REQUIRE(!gpu_live_bytes());')
def apply(root):
    root=Path(root);p=root/'runtime/qwen.c';p.write_text(transform(p.name,p.read_text()));p=root/'driver.c';p.write_text(driver_transform(p.read_text()))
