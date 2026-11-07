/* Included by qwen.c: source-qualified operand programs, no global routing. */
static int operand_finite(const float *a,size_t n){if(!a)return 0;for(size_t i=0;i<n;i++)if(!isfinite(a[i]))return 0;return 1;}
static void operand_hash(const void *p,size_t n,char out[65]){unsigned char h[32];tcm_sha256(p,n,h);tcm_hex(h,32,out);}
static int operand_empty(const QwenSession *s){return s&&s->model&&s->position==0&&s->phase!=2&&!s->depth;}
int qwen_qkv_program_init(QwenQkvProgram *p,const QwenModel *m,int layer,int rank,int mask,const float *basis,const float *q,const float *k,const float *v){
    if(!p||!m||layer<0||layer>=m->layers||rank<1||rank>m->hidden||mask<1||mask>7)return -1;
    int widths[]={m->hidden,m->kvheads*m->head,m->kvheads*m->head};const float *images[]={q,k,v};
    if(!operand_finite(basis,(size_t)rank*m->hidden))return -2;
    for(int j=0;j<3;j++)if((mask&(1<<j))&&!operand_finite(images[j],(size_t)rank*widths[j]))return -2;
    QwenQkvProgram out={0};out.model=m;out.layer=layer;out.rank=rank;out.mask=mask;
    out.basis=gpu_weight(basis,TCM_F32,(size_t)rank*m->hidden);out.gpu_bytes=(size_t)rank*m->hidden*4;
    char hashes[4][65]={{0}},binding[512];operand_hash(basis,out.gpu_bytes,hashes[0]);
    for(int j=0;j<3;j++)if(mask&(1<<j)){size_t n=(size_t)rank*widths[j];out.image[j]=gpu_weight(images[j],TCM_F32,n);out.gpu_bytes+=n*4;operand_hash(images[j],n*4,hashes[j+1]);}
    int n=snprintf(binding,sizeof(binding),"qkv-f32-v1/%s/%d/%d/%d/%s/%s/%s/%s",m->digest,layer,rank,mask,hashes[0],hashes[1],hashes[2],hashes[3]);
    if(n<0||(size_t)n>=sizeof(binding)){qwen_qkv_program_free(&out);return -1;}
    operand_hash(binding,(size_t)n,out.fingerprint);out.ready=1;*p=out;return 0;
}
void qwen_qkv_program_free(QwenQkvProgram *p){if(!p)return;gpu_free(p->basis);for(int j=0;j<3;j++)gpu_free(p->image[j]);memset(p,0,sizeof(*p));}
int qwen_qkv_bind(QwenSession *s,int layer,const QwenQkvProgram *p){
    if(!operand_empty(s)||layer<0||layer>=s->model->layers||layer>=64)return -1;
    if(p&&(!p->ready||p->model!=s->model||p->layer!=layer||(!layer&&s->lexical_program)))return -1;
    if(p&&(size_t)p->rank>s->qkv_code_capacity){float *code=gpu_alloc((size_t)p->rank*4);gpu_free(s->qkv_code);s->allocated_bytes+=((size_t)p->rank-s->qkv_code_capacity)*4;s->qkv_code=code;s->qkv_code_capacity=(size_t)p->rank;}
    s->qkv_programs[layer]=p;return 0;
}
static int lexical_index(const QwenLexicalProgram *p,int token){
    size_t lo=0,hi=p->rows;while(lo<hi){size_t mid=lo+(hi-lo)/2;if(p->ids[mid]<token)lo=mid+1;else hi=mid;}return lo<p->rows&&p->ids[lo]==token?(int)lo:-1;
}
static void lexical_fingerprint(QwenLexicalProgram *p){
    char a[65],b[65],binding[256];size_t width=(size_t)p->model->hidden+2*(size_t)p->model->kvheads*p->model->head;
    operand_hash(p->ids,p->rows*4,a);operand_hash(p->values,p->rows*width*4,b);int n=snprintf(binding,sizeof(binding),"lexical-f32-v1/%s/%zu/%s/%s",p->model->digest,p->rows,a,b);operand_hash(binding,(size_t)n,p->fingerprint);
}
int qwen_lexical_compile(QwenLexicalProgram *p,const QwenModel *m,const int *ids,size_t rows){
    if(!p||!m||!rows||rows>(size_t)m->vocab||(!ids&&rows!=(size_t)m->vocab))return -1;
    for(size_t i=0;i<rows;i++)if(ids&&(ids[i]<0||ids[i]>=m->vocab||(i&&ids[i]<=ids[i-1])))return -1;
    QwenLexicalProgram out={0};out.model=m;out.rows=rows;int d=m->hidden,k=m->kvheads*m->head;size_t width=(size_t)d+2*(size_t)k;
    if(width>SIZE_MAX/4/rows)return -1;
    out.ids=malloc(rows*4);out.values=malloc(rows*width*4);if(!out.ids||!out.values){qwen_lexical_free(&out);return -1;}
    float *x=gpu_alloc((size_t)d*4),*norm=gpu_alloc((size_t)d*4),*row=gpu_alloc(width*4);const QwenLayer *l=m->layer;
    for(size_t i=0;i<rows;i++){
        int token=ids?ids[i]:(int)i;out.ids[i]=token;gpu_owned_embedding(x,m->embedding,token,d);gpu_rms(norm,x,l->input_norm,d,m->epsilon);
        gpu_matvec(m->gpu,l->q,d,d,norm,row);gpu_bias(row,l->qb,d);gpu_matvec(m->gpu,l->k,d,k,norm,row+d);gpu_bias(row+d,l->kb,k);gpu_matvec(m->gpu,l->v,d,k,norm,row+d+k);gpu_bias(row+d+k,l->vb,k);
        gpu_read(out.values+i*width,row,width*4);
    }
    gpu_free(x);gpu_free(norm);gpu_free(row);
    if(!operand_finite(out.values,rows*width)){qwen_lexical_free(&out);return -2;}
    out.compile_linear_macs=rows*(uint64_t)d*(d+2*(uint64_t)k);out.host_bytes=rows*(4+width*4);lexical_fingerprint(&out);out.ready=1;*p=out;return 0;
}
void qwen_lexical_free(QwenLexicalProgram *p){if(!p)return;free(p->ids);free(p->values);memset(p,0,sizeof(*p));}
int qwen_lexical_bind(QwenSession *s,const QwenLexicalProgram *p){
    if(!operand_empty(s)||(p&&(!p->ready||p->model!=s->model||s->qkv_programs[0])))return -1;s->lexical_program=p;return 0;
}
int qwen_lexical_save(const QwenLexicalProgram *p,const char *path){
    if(!p||!p->ready||!path)return -1;FILE *f=fopen(path,"wb");if(!f)return -1;const QwenModel *m=p->model;uint32_t meta[]={(uint32_t)m->hidden,(uint32_t)(m->kvheads*m->head),(uint32_t)m->vocab,(uint32_t)p->rows};size_t width=(size_t)meta[0]+2*(size_t)meta[1];
    int bad=fwrite("TQATOM1\0",1,8,f)!=8||fwrite(m->digest,1,65,f)!=65||fwrite(p->fingerprint,1,65,f)!=65||fwrite(meta,4,4,f)!=4||fwrite(p->ids,4,p->rows,f)!=p->rows||fwrite(p->values,4,p->rows*width,f)!=p->rows*width;return fclose(f)||bad?-1:0;
}
int qwen_lexical_load(QwenLexicalProgram *p,const QwenModel *m,const char *path){
    if(!p||!m||!path)return -1;FILE *f=fopen(path,"rb");if(!f)return -1;char magic[8],model[65],fingerprint[65];uint32_t meta[4];int bad=fread(magic,1,8,f)!=8||fread(model,1,65,f)!=65||fread(fingerprint,1,65,f)!=65||fread(meta,4,4,f)!=4;
    if(bad||memcmp(magic,"TQATOM1\0",8)||memcmp(model,m->digest,65)||meta[0]!=(uint32_t)m->hidden||meta[1]!=(uint32_t)(m->kvheads*m->head)||meta[2]!=(uint32_t)m->vocab||!meta[3]||meta[3]>(uint32_t)m->vocab){fclose(f);return -1;}
    QwenLexicalProgram out={0};out.model=m;out.rows=meta[3];size_t width=(size_t)meta[0]+2*(size_t)meta[1];if(width>SIZE_MAX/4/out.rows){fclose(f);return -1;}
    out.ids=malloc(out.rows*4);out.values=malloc(out.rows*width*4);if(!out.ids||!out.values){fclose(f);qwen_lexical_free(&out);return -1;}
    bad=fread(out.ids,4,out.rows,f)!=out.rows||fread(out.values,4,out.rows*width,f)!=out.rows*width||fgetc(f)!=EOF;if(fclose(f))bad=1;
    for(size_t i=0;i<out.rows;i++)if(out.ids[i]<0||out.ids[i]>=m->vocab||(i&&out.ids[i]<=out.ids[i-1]))bad=1;
    if(!operand_finite(out.values,out.rows*width))bad=1;
    if(bad){qwen_lexical_free(&out);return -1;}out.host_bytes=out.rows*(4+width*4);lexical_fingerprint(&out);
    if(memcmp(out.fingerprint,fingerprint,65)){qwen_lexical_free(&out);return -1;}out.ready=1;*p=out;return 0;
}
static int qkv_valid(const QwenSession *s,int layer){
    if(!layer&&s->lexical_program&&(!s->lexical_program->ready||s->lexical_program->model!=s->model||lexical_index(s->lexical_program,s->current_token)<0))return 0;
    const QwenQkvProgram *p=s->qkv_programs[layer];return !p||(p->ready&&p->model==s->model&&p->layer==layer&&p->rank>0&&(size_t)p->rank<=s->qkv_code_capacity&&p->mask>0&&p->mask<8);
}
static void qkv_apply(QwenSession *s,int layer){
    const QwenModel *m=s->model;const QwenLayer *l=m->layer+layer;int d=m->hidden,k=m->kvheads*m->head;
    if(!layer&&s->lexical_program){
        const QwenLexicalProgram *p=s->lexical_program;int row=lexical_index(p,s->current_token);size_t width=(size_t)d+2*(size_t)k;const float *a=p->values+(size_t)row*width;
        gpu_write(s->q,a,(size_t)d*4);gpu_write(s->k,a+d,(size_t)k*4);gpu_write(s->v,a+d+k,(size_t)k*4);s->operand_work.lexical_hits++;s->operand_work.lexical_transfer_bytes+=width*4;return;
    }
    const QwenQkvProgram *p=s->qkv_programs[layer];float *outputs[]={s->q,s->k,s->v};const float *weights[]={l->q,l->k,l->v},*biases[]={l->qb,l->kb,l->vb};int widths[]={d,k,k};
    if(p){gpu_matvec(m->gpu,p->basis,d,p->rank,s->norm,s->qkv_code);s->operand_work.code_calls++;s->operand_work.linear_macs+=(uint64_t)d*p->rank;s->operand_work.applications++;}
    for(int j=0;j<3;j++){
        if(p&&(p->mask&(1<<j))){gpu_matvec(m->gpu,p->image[j],p->rank,widths[j],s->qkv_code,outputs[j]);s->operand_work.compiled_calls++;s->operand_work.linear_macs+=(uint64_t)p->rank*widths[j];}
        else{gpu_matvec(m->gpu,weights[j],d,widths[j],s->norm,outputs[j]);if(p){s->operand_work.native_calls++;s->operand_work.linear_macs+=(uint64_t)d*widths[j];}}
        gpu_bias(outputs[j],biases[j],widths[j]);
    }
}
