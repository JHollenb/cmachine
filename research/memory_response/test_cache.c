/* Model-free host-memory substitute for the exact ring snapshot copy extents.
 * Run under AddressSanitizer. This is not an attention or model parity test. */
#include <assert.h>
#include <stdio.h>
#define REQUIRE(x) assert(x)
#include "cache.h"
void *gpu_alloc(size_t n){void *p=calloc(1,n);assert(p);return p;}
void gpu_free(void *p){free(p);}
void gpu_read(void *to,const void *from,size_t n){memcpy(to,from,n);}
void gpu_write(void *to,const void *from,size_t n){memcpy(to,from,n);}

static void check(MemoryPolicy policy) {
    QwenModel model={0};model.layers=28;model.kvheads=2;model.head=128;model.max_position=512;
    QwenSession session={0};session.model=&model;session.capacity=512;session.session_id=1;session.epoch=1;
    MemoryCache cache;assert(!memory_cache_open(&cache,&session,policy,512));cache.count=7;
    for(int l=0;l<28;l++) {
        MemoryLayer *a=&cache.layers[l];a->cursor=7;
        size_t cb=(size_t)2*a->storage_capacity*128,sc=(size_t)4*a->storage_capacity;
        memset(a->key_codes,l+1,cb);memset(a->value_codes,l+29,cb);
        for(size_t i=0;i<sc;i++)a->scales[i]=(float)(l*10000+i+1);
    }
    MemoryMark mark={0};assert(!memory_mark_capture(&cache,&mark));
    assert(mark.count==7&&mark.valid);
    if(policy==MEMORY_BYTE_RECENT4) {
        assert(mark.code_bytes==28672&&mark.scale_bytes==1792);
        for(int l=0;l<28;l++) {
            MemoryLayer *a=&cache.layers[l];size_t cb=(size_t)2*4*128,sb=(size_t)4*4*sizeof(float);
            memset(a->key_codes,0,cb);memset(a->value_codes,0,cb);memset(a->scales,0,sb);a->cursor=8;
        }
    } else assert(!mark.key_codes&&!mark.value_codes&&!mark.scales);
    cache.count=8;assert(!memory_mark_restore(&cache,&mark));assert(cache.count==7);
    for(int l=0;l<28;l++) {
        MemoryLayer *a=&cache.layers[l];assert(a->cursor==7);
        size_t cb=(size_t)2*a->storage_capacity*128,sc=(size_t)4*a->storage_capacity;
        for(size_t i=0;i<cb;i++){assert(a->key_codes[i]==l+1);assert(a->value_codes[i]==l+29);}
        for(size_t i=0;i<sc;i++)assert(a->scales[i]==(float)(l*10000+i+1));
    }
    memory_mark_free(&mark);memory_cache_close(&cache);
}
int main(void){check(MEMORY_BYTE_RECENT4);check(MEMORY_BYTE_ALL);puts("ring bytes/scales and metadata-only full-bank snapshots exact");return 0;}
