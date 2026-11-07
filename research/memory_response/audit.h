#ifndef TCM_MEMORY_RESPONSE_AUDIT_H
#define TCM_MEMORY_RESPONSE_AUDIT_H

/* Read-only provenance for two declared layers. These host buffers are never
 * passed to a cache callback. The serving policies cannot read their erased
 * history from this audit bank. Response calculations happen after sealing. */
typedef struct { char file[256], sha[65]; int shape[3], ndim; } AuditArray;
typedef struct {
    int layer, position, length, start, codec_exact, codec_mismatches, codec_zero_mismatches;
    float codec_max_abs;
    AuditArray query, original_keys, original_values, byte_keys, byte_values,
               actual_context, actual_weights;
} AuditRow;
typedef struct {
    QwenSession *session;
    MemoryCache *cache;
    float *keys, *values;
    size_t host_bytes;
    int capture_position, case_id, query_id, count;
    const char *policy;
    const char *directory;
    AuditRow rows[2];
} MemoryAudit;

static void audit_open(MemoryAudit *a, QwenSession *s, MemoryCache *cache,
                       const char *directory, int case_id, const char *policy) {
    memset(a,0,sizeof(*a)); a->session=s; a->cache=cache; a->directory=directory;
    a->case_id=case_id; a->policy=policy; a->capture_position=-1;
    size_t elements=(size_t)2*s->model->kvheads*CAPACITY*s->model->head;
    a->keys=calloc(elements,sizeof(float)); a->values=calloc(elements,sizeof(float));
    REQUIRE(a->keys&&a->values); a->host_bytes=2*elements*sizeof(float);
}
static void audit_close(MemoryAudit *a) { free(a->keys); free(a->values); memset(a,0,sizeof(*a)); }

static void audit_save(MemoryAudit *a, AuditRow *r, AuditArray *array,
                       const char *name, const float *data, int ndim, int x, int y, int z) {
    array->ndim=ndim; array->shape[0]=x; array->shape[1]=y; array->shape[2]=z;
    size_t count=(size_t)x*(ndim>1?y:1)*(ndim>2?z:1);
    REQUIRE(snprintf(array->file,sizeof(array->file),"case%d-%s-query%d-l%d-%s.f32",
                     a->case_id,a->policy,a->query_id,r->layer,name)<(int)sizeof(array->file));
    for(size_t i=0;i<count;i++) REQUIRE(isfinite(data[i]));
    digest(data,count*sizeof(float),array->sha); save(a->directory,array->file,data,count*sizeof(float));
}

static void audit_codec(const float *source,float *decoded,int groups,int length,int dim) {
    for(int h=0;h<groups;h++) for(int p=0;p<length;p++) {
        size_t base=((size_t)h*length+p)*dim; float maximum=0;
        for(int d=0;d<dim;d++) maximum=fmaxf(maximum,fabsf(source[base+d]));
        float scale=maximum>0?maximum/127.0f:1.0f;
        REQUIRE(scale>0&&isfinite(scale));
        for(int d=0;d<dim;d++) {
            float code=roundf(source[base+d]/scale); code=fminf(127,fmaxf(-127,code));
            /* Serving stores a signed integer byte, so rounded -0 must pass
             * through that integer representation before float decoding. */
            decoded[base+d]=(float)(signed char)code*scale;
        }
    }
}

static int audit_codec_matches(MemoryAudit *a,AuditRow *r,int layer,int length,const float *keys,const float *values) {
    if(!a->cache||!memory_is_byte(a->cache->policy)) return 1;
    const QwenModel *m=a->session->model; MemoryLayer *bank=&a->cache->layers[layer];
    int storage=bank->storage_capacity,first=a->cache->policy==MEMORY_BYTE_RECENT4&&length>4?length-4:0;
    size_t n=(size_t)m->kvheads*storage*m->head;
    signed char *kc=malloc(n),*vc=malloc(n); float *sc=malloc((size_t)2*m->kvheads*storage*sizeof(float));
    REQUIRE(kc&&vc&&sc); gpu_read(kc,bank->key_codes,n); gpu_read(vc,bank->value_codes,n);
    gpu_read(sc,bank->scales,(size_t)2*m->kvheads*storage*sizeof(float));
    int same=1;
    for(int h=0;h<m->kvheads;h++) for(int p=first;p<length;p++) for(int d=0;d<m->head;d++) {
        int slot=a->cache->policy==MEMORY_BYTE_RECENT4?p%4:p;
        size_t source=((size_t)h*storage+slot)*m->head+d,target=((size_t)h*length+p)*m->head+d;
        float k=(float)kc[source]*sc[(size_t)h*storage+slot],v=(float)vc[source]*sc[(size_t)(m->kvheads+h)*storage+slot];
        float expected[2]={keys[target],values[target]},actual[2]={k,v};
        for(int kind=0;kind<2;kind++)if(memcmp(actual+kind,expected+kind,sizeof(float))) {
            same=0;r->codec_mismatches++;if(actual[kind]==expected[kind])r->codec_zero_mismatches++;
            r->codec_max_abs=fmaxf(r->codec_max_abs,fabsf(actual[kind]-expected[kind]));
        }
    }
    free(kc);free(vc);free(sc);return same;
}

static void audit_observe(void *context,int position,int layer,QwenPort port_id,const float *device,size_t count) {
    MemoryAudit *a=context; if(layer!=17&&layer!=18) return;
    const QwenModel *m=a->session->model; int li=layer-17,kv=m->kvheads,dim=m->head;
    if(port_id==QWEN_KEY||port_id==QWEN_VALUE) {
        REQUIRE(count==(size_t)kv*dim&&position<CAPACITY); float *row=read_floats(device,count);
        float *bank=port_id==QWEN_KEY?a->keys:a->values;
        for(int h=0;h<kv;h++) memcpy(bank+(((size_t)li*kv+h)*CAPACITY+position)*dim,row+(size_t)h*dim,(size_t)dim*sizeof(float));
        free(row); return;
    }
    if(port_id!=QWEN_ATTENTION||position!=a->capture_position) return;
    REQUIRE(a->count<2); AuditRow *r=&a->rows[a->count++]; memset(r,0,sizeof(*r));
    r->layer=layer;r->position=position;r->length=position+1;r->start=0;
    if(a->cache&&a->cache->policy==MEMORY_BYTE_RECENT4) r->start=r->length>4?r->length-4:0;
    if(a->cache&&a->cache->policy==MEMORY_FP32_ALL&&a->cache->floor>=0) r->start=a->cache->floor;
    int n=r->length; size_t elements=(size_t)kv*n*dim;
    float *keys=malloc(elements*sizeof(float)),*values=malloc(elements*sizeof(float));
    float *bk=malloc(elements*sizeof(float)),*bv=malloc(elements*sizeof(float));REQUIRE(keys&&values&&bk&&bv);
    for(int h=0;h<kv;h++) {
        memcpy(keys+(size_t)h*n*dim,a->keys+((size_t)li*kv+h)*CAPACITY*dim,(size_t)n*dim*sizeof(float));
        memcpy(values+(size_t)h*n*dim,a->values+((size_t)li*kv+h)*CAPACITY*dim,(size_t)n*dim*sizeof(float));
    }
    audit_codec(keys,bk,kv,n,dim);audit_codec(values,bv,kv,n,dim);
    r->codec_exact=audit_codec_matches(a,r,layer,n,bk,bv);
    if(!r->codec_exact)fprintf(stderr,"memory_response audit mismatch case%d %s query%d layer%d bits=%d zero=%d max_abs=%.9g\n",a->case_id,a->policy,a->query_id,layer,r->codec_mismatches,r->codec_zero_mismatches,r->codec_max_abs);
    float *query=read_floats(a->session->q,(size_t)m->qheads*dim);
    float *actual=read_floats(device,count),*weights=read_floats(a->session->scores,(size_t)m->qheads*(n-r->start));
    audit_save(a,r,&r->query,"query",query,2,m->qheads,dim,0);
    audit_save(a,r,&r->original_keys,"original-keys",keys,3,kv,n,dim);
    audit_save(a,r,&r->original_values,"original-values",values,3,kv,n,dim);
    audit_save(a,r,&r->byte_keys,"byte-keys",bk,3,kv,n,dim);
    audit_save(a,r,&r->byte_values,"byte-values",bv,3,kv,n,dim);
    audit_save(a,r,&r->actual_context,"actual-context",actual,2,m->qheads,dim,0);
    audit_save(a,r,&r->actual_weights,"actual-weights",weights,2,m->qheads,n-r->start,0);
    free(keys);free(values);free(bk);free(bv);free(query);free(actual);free(weights);
}
static void audit_descriptor(FILE *f,const AuditArray *a) {
    fprintf(f,"{\"file\":\"%s\",\"sha256\":\"%s\",\"dtype\":\"<f4\",\"shape\":[",a->file,a->sha);
    for(int i=0;i<a->ndim;i++){if(i)fputc(',',f);fprintf(f,"%d",a->shape[i]);}fputs("]}",f);
}
static void audit_write(FILE *f,const MemoryAudit *a) {
    fputc('[',f);
    for(int i=0;i<a->count;i++) {
        if(i)fputc(',',f);const AuditRow *r=&a->rows[i];
        fprintf(f,"{\"layer\":%d,\"position\":%d,\"key_count\":%d,\"active_start\":%d,\"policy\":\"%s\",\"codec_audit_exact\":%s,\"codec_bit_mismatches\":%d,\"codec_zero_mismatches\":%d,\"codec_max_abs_difference\":%.9g,\"audit_only\":true",r->layer,r->position,r->length,r->start,a->policy,r->codec_exact?"true":"false",r->codec_mismatches,r->codec_zero_mismatches,r->codec_max_abs);
#define AUDIT_FIELD(name) fputs(",\"" #name "\":",f);audit_descriptor(f,&r->name)
        AUDIT_FIELD(query); AUDIT_FIELD(original_keys); AUDIT_FIELD(original_values); AUDIT_FIELD(byte_keys);
        AUDIT_FIELD(byte_values); AUDIT_FIELD(actual_context); AUDIT_FIELD(actual_weights);
#undef AUDIT_FIELD
        fputc('}',f);
    }
    fputc(']',f);
}
#endif
