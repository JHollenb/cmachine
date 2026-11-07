#ifndef TCM_APPEND_CACHE_H
#define TCM_APPEND_CACHE_H
#include "codec.h"
extern int gpu_encode_current(signed char *,signed char *,float *,const float *,const float *,int,int,int *);
extern void gpu_byte_attention(TcmGpu *,float *,const float *,const signed char *,const signed char *,const float *,float *,float *,int,int);
extern size_t gpu_live_bytes(void);
extern size_t gpu_peak_bytes(void);
typedef struct {int layer,byte,cursor;float *keys,*values,*scales;int8_t *key_codes,*value_codes;} CacheLayer;
typedef struct {
    QwenSession *session;uint64_t owner,epoch;int count,fail_layer,fail_position;
    CacheLayer layers[2];float *workspace;int *invalid;
    size_t arena_bytes,workspace_bytes;unsigned long calls,published,rewinds;
} OwnedCache;
static CacheLayer *cache_layer(OwnedCache *c,int layer){for(int i=0;i<c->count;i++)if(c->layers[i].layer==layer)return &c->layers[i];return NULL;}
static int owned_step(void *context,int layer,int position,const float *q,const float *k,const float *v,float *out,float *scores){
    OwnedCache *c=context;QwenSession *s=c->session;CacheLayer *a=cache_layer(c,layer);
    if(!a||s->session_id!=c->owner||s->epoch!=c->epoch||s->phase!=2||s->position!=position||position<0||position>=s->capacity||(a->cursor!=position&&a->cursor!=position+1))return -70;
    c->calls++;
    if(a->byte){
        if(gpu_encode_current(a->key_codes,a->value_codes,a->scales,k,v,s->capacity,position,c->invalid))return -71;
        if(c->fail_layer==layer&&c->fail_position==position){c->fail_layer=-1;return -77;}
        a->cursor=position+1;c->published++;
        gpu_byte_attention(s->model->gpu,out,q,a->key_codes,a->value_codes,a->scales,scores,c->workspace,s->capacity,a->cursor);
    }else{
        gpu_cache(a->keys,k,2,128,s->capacity,position);gpu_cache(a->values,v,2,128,s->capacity,position);a->cursor=position+1;c->published++;
        gpu_attention(s->model->gpu,out,q,a->keys,a->values,scores,12,2,128,s->capacity,a->cursor);
    }return 0;
}
static void owned_rewind(void *context,int position,uint64_t owner,uint64_t epoch){
    OwnedCache *c=context;REQUIRE(owner==c->owner);
    if(epoch!=c->epoch){REQUIRE(position==0&&epoch==c->epoch+1);c->epoch=epoch;}
    for(int i=0;i<c->count;i++){REQUIRE(position>=0&&position<=c->layers[i].cursor);c->layers[i].cursor=position;}c->rewinds++;
}
static void cache_open(OwnedCache *c,QwenSession *s,int mode){
    memset(c,0,sizeof(*c));c->session=s;c->fail_layer=-1;c->count=mode==2?1:2;int any_byte=mode!=1;
    for(int i=0;i<c->count;i++){
        CacheLayer *a=&c->layers[i];a->layer=17+i;a->byte=any_byte;
        if(a->byte){a->key_codes=gpu_alloc(2*512*128);a->value_codes=gpu_alloc(2*512*128);a->scales=gpu_alloc(4*512*4);c->arena_bytes+=270336;}
        else{a->keys=gpu_alloc(2*512*128*4);a->values=gpu_alloc(2*512*128*4);c->arena_bytes+=1048576;}
    }
    if(any_byte){c->workspace_bytes=2*512*128*4;c->workspace=gpu_alloc(c->workspace_bytes);c->invalid=gpu_alloc(4);}
}
static void cache_bind(OwnedCache *c){c->owner=c->session->session_id;c->epoch=c->session->epoch;}
static void cache_close(OwnedCache *c){
    for(int i=0;i<c->count;i++){CacheLayer *a=&c->layers[i];gpu_free(a->keys);gpu_free(a->values);gpu_free(a->key_codes);gpu_free(a->value_codes);gpu_free(a->scales);}gpu_free(c->workspace);gpu_free(c->invalid);
}
/* Hydration uses source operands once, then only arena ownership survives. */
static void cache_import(OwnedCache *c,CacheLayer *a,int length,const float *keys,const float *values){
    REQUIRE(length>=0&&length<=512);size_t count=(size_t)2*512*128;
    if(a->byte){
        int8_t *k=calloc(count,1),*v=calloc(count,1);float *scales=calloc(4*512,4);REQUIRE(k&&v&&scales);
        for(int h=0;h<2;h++)for(int p=0;p<length;p++){
            REQUIRE(!codec_row(keys+((size_t)h*length+p)*128,k+((size_t)h*512+p)*128,scales+h*512+p));
            REQUIRE(!codec_row(values+((size_t)h*length+p)*128,v+((size_t)h*512+p)*128,scales+(2+h)*512+p));
        }gpu_write(a->key_codes,k,count);gpu_write(a->value_codes,v,count);gpu_write(a->scales,scales,4*512*4);memset(k,0,count);memset(v,0,count);memset(scales,0,4*512*4);free(k);free(v);free(scales);
    }else for(int h=0;h<2;h++){gpu_write(a->keys+(size_t)h*512*128,keys+(size_t)h*length*128,(size_t)length*128*4);gpu_write(a->values+(size_t)h*512*128,values+(size_t)h*length*128,(size_t)length*128*4);}
    a->cursor=length;(void)c;
}
/* Export is an audit operation, never part of attention or a native repair. */
static void cache_read_plane(const QwenSession *s,OwnedCache *c,int kind,int layer,int length,float *out){
    CacheLayer *a=c?cache_layer(c,layer):NULL;
    if(a&&a->byte){
        REQUIRE(length<=a->cursor);int8_t codes[2*512*128];float scales[4*512];gpu_read(codes,kind?a->value_codes:a->key_codes,sizeof(codes));gpu_read(scales,a->scales,sizeof(scales));
        for(int h=0;h<2;h++)for(int p=0;p<length;p++)for(int d=0;d<128;d++)out[((size_t)h*length+p)*128+d]=(float)codes[((size_t)h*512+p)*128+d]*scales[(kind*2+h)*512+p];
    }else{
        REQUIRE(a||s->cache_slots[layer]>=0);
        const float *base=a?(kind?a->values:a->keys):(kind?s->values:s->keys)+(size_t)s->cache_slots[layer]*2*512*128;
        for(int h=0;h<2;h++)gpu_read(out+(size_t)h*length*128,base+(size_t)h*512*128,(size_t)length*128*4);
    }
}
static unsigned char *cache_serialize(CacheLayer *a,int length,size_t *bytes){
    REQUIRE(a->byte&&length>=0&&length<=a->cursor);*bytes=32+(size_t)length*528;unsigned char *raw=malloc(*bytes);REQUIRE(raw);memcpy(raw,"TCMAPND1",8);uint32_t header[]={1,(uint32_t)a->layer,2,128,(uint32_t)length,8};memcpy(raw+8,header,24);size_t off=32;
    for(int kind=0;kind<2;kind++)for(int h=0;h<2;h++){gpu_read(raw+off,a->scales+(kind*2+h)*512,(size_t)length*4);off+=(size_t)length*4;}
    for(int kind=0;kind<2;kind++)for(int h=0;h<2;h++){gpu_read(raw+off,(kind?a->value_codes:a->key_codes)+(size_t)h*512*128,(size_t)length*128);off+=(size_t)length*128;}
    REQUIRE(off==*bytes);return raw;
}
static void cache_import_program(CacheLayer *a,const unsigned char *raw,size_t bytes){
    uint32_t header[6];REQUIRE(bytes>=32&&!memcmp(raw,"TCMAPND1",8));memcpy(header,raw+8,24);int length=(int)header[4];REQUIRE(a->byte&&header[0]==1&&header[1]==(uint32_t)a->layer&&header[2]==2&&header[3]==128&&header[5]==8&&length==93&&bytes==32+(size_t)length*528);size_t off=32;
    for(int kind=0;kind<2;kind++)for(int h=0;h<2;h++){gpu_write(a->scales+(kind*2+h)*512,raw+off,(size_t)length*4);off+=(size_t)length*4;}
    for(int kind=0;kind<2;kind++)for(int h=0;h<2;h++){gpu_write((kind?a->value_codes:a->key_codes)+(size_t)h*512*128,raw+off,(size_t)length*128);off+=(size_t)length*128;}
    REQUIRE(off==bytes);a->cursor=length;
}
#endif
