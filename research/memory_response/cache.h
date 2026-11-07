#ifndef TCM_MEMORY_RESPONSE_CACHE_H
#define TCM_MEMORY_RESPONSE_CACHE_H

/*
 * All-layer memory policies for the memory-response organism.
 *
 * The Qwen runtime calls memory_step once for every layer and token after Q/K/V
 * and RoPE have been formed.  The callback therefore owns the K/V write and the
 * attention read, while the ordinary residual, RMSNorm, MLP, and final readout
 * remain in the frozen runtime.  This file deliberately keeps policy state
 * separate from QwenSession: qwen_stack_rollback restores the logical cursor,
 * and memory_mark/memory_restore additionally restore ring bytes that a cursor
 * alone cannot recover.
 */

#include "runtime/qwen.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MEMORY_RESPONSE_LAYERS 28
#define MEMORY_RESPONSE_RECENT 4

typedef enum {
    MEMORY_NATIVE = 0,
    MEMORY_BYTE_ALL = 1,
    MEMORY_BYTE_TOP4 = 2,
    MEMORY_BYTE_RECENT4 = 3,
    MEMORY_FP32_ALL = 4,
    MEMORY_ZERO = 5
} MemoryPolicy;

typedef struct {
    int layer;
    int cursor;
    int storage_capacity;
    signed char *key_codes;
    signed char *value_codes;
    float *scales;                 /* [2 * kvheads][storage_capacity]. */
    float *keys;                   /* FP32 storage for the history diagnostic. */
    float *values;
} MemoryLayer;

typedef struct {
    QwenSession *session;
    MemoryPolicy policy;
    uint64_t owner, epoch;
    int capacity, count, floor, fail_layer, fail_position;
    MemoryLayer layers[MEMORY_RESPONSE_LAYERS];
    float *workspace;
    int *invalid;
    size_t arena_bytes, workspace_bytes;
    unsigned long calls, published, rewinds;
    uint64_t score_macs, value_macs, decoded_scalars, value_probability_scans;
} MemoryCache;

typedef struct {
    int valid;
    MemoryPolicy policy;
    int capacity, count, floor;
    int cursors[MEMORY_RESPONSE_LAYERS];
    size_t code_bytes, scale_bytes, fp_bytes;
    signed char *key_codes, *value_codes;
    float *scales, *keys, *values;
} MemoryMark;

static inline const char *memory_policy_name(MemoryPolicy p) {
    switch (p) {
        case MEMORY_NATIVE: return "native";
        case MEMORY_BYTE_ALL: return "byte_full";
        case MEMORY_BYTE_TOP4: return "byte_top4";
        case MEMORY_BYTE_RECENT4: return "byte_recent4";
        case MEMORY_FP32_ALL: return "history_deleted";
        case MEMORY_ZERO: return "zero";
        default: return "unknown";
    }
}

static inline int memory_is_byte(MemoryPolicy p) {
    return p == MEMORY_BYTE_ALL || p == MEMORY_BYTE_TOP4 || p == MEMORY_BYTE_RECENT4;
}

static inline MemoryLayer *memory_layer(MemoryCache *c, int layer) {
    if (!c || layer < 0 || layer >= MEMORY_RESPONSE_LAYERS) return NULL;
    return &c->layers[layer];
}

static inline const MemoryLayer *memory_const_layer(const MemoryCache *c, int layer) {
    if (!c || layer < 0 || layer >= MEMORY_RESPONSE_LAYERS) return NULL;
    return &c->layers[layer];
}

static inline void memory_free_layer(MemoryLayer *a) {
    if (!a) return;
    gpu_free(a->key_codes); gpu_free(a->value_codes); gpu_free(a->scales);
    gpu_free(a->keys); gpu_free(a->values);
    memset(a, 0, sizeof(*a));
}

static inline int memory_cache_open(MemoryCache *c, QwenSession *s, MemoryPolicy policy, int capacity) {
    if (!c || !s || !s->model || capacity < 1 || capacity > s->model->max_position ||
        s->model->layers != MEMORY_RESPONSE_LAYERS || s->model->head != 128) return -1;
    memset(c, 0, sizeof(*c));
    c->session = s; c->policy = policy; c->capacity = capacity; c->floor = -1; c->fail_layer = -1;
    const int kv = s->model->kvheads, dim = s->model->head;
    const int storage = policy == MEMORY_BYTE_RECENT4 ? MEMORY_RESPONSE_RECENT : capacity;
    c->workspace_bytes = (size_t)2 * kv * storage * dim * sizeof(float);
    if (policy != MEMORY_ZERO) {
        c->workspace = (float *)gpu_alloc(c->workspace_bytes);
        c->invalid = (int *)gpu_alloc(sizeof(int));
    }
    for (int layer = 0; layer < s->model->layers; ++layer) {
        MemoryLayer *a = &c->layers[layer];
        a->layer = layer; a->storage_capacity = storage; a->cursor = 0;
        if (memory_is_byte(policy)) {
            size_t codes = (size_t)kv * storage * dim;
            size_t scales = (size_t)2 * kv * storage;
            a->key_codes = (signed char *)gpu_alloc(codes);
            a->value_codes = (signed char *)gpu_alloc(codes);
            a->scales = (float *)gpu_alloc(scales * sizeof(float));
            c->arena_bytes += 2 * codes + scales * sizeof(float);
        } else if (policy == MEMORY_FP32_ALL) {
            size_t rows = (size_t)kv * storage * dim;
            a->keys = (float *)gpu_alloc(rows * sizeof(float));
            a->values = (float *)gpu_alloc(rows * sizeof(float));
            c->arena_bytes += 2 * rows * sizeof(float);
        }
    }
    c->owner = s->session_id; c->epoch = s->epoch;
    return 0;
}

static inline void memory_cache_close(MemoryCache *c) {
    if (!c) return;
    for (int i = 0; i < MEMORY_RESPONSE_LAYERS; ++i) memory_free_layer(&c->layers[i]);
    gpu_free(c->workspace); gpu_free(c->invalid);
    memset(c, 0, sizeof(*c));
}

static inline void memory_cache_bind(MemoryCache *c) {
    REQUIRE(c && c->session);
    c->owner = c->session->session_id; c->epoch = c->session->epoch;
}

static inline int memory_cache_set_floor(MemoryCache *c, int floor) {
    if (!c || c->policy != MEMORY_FP32_ALL || floor < 0 || floor > c->count) return -1;
    c->floor = floor;
    return 0;
}

/* The callback publishes one layer at a time, but the logical token boundary
 * advances only after qwen_push has completed every layer and the final logits.
 * The driver calls this once per successful token; partial aborts therefore
 * leave count at the last committed boundary. */
static inline int memory_cache_commit(MemoryCache *c, int position) {
    if (!c || !c->session || position < 0 || position > c->capacity) return -1;
    for (int i = 0; i < c->session->model->layers; ++i)
        if (c->layers[i].cursor != position) return -1;
    c->count = position;
    return 0;
}

static inline void memory_cache_rewind(void *context, int position, uint64_t owner, uint64_t epoch) {
    MemoryCache *c = (MemoryCache *)context;
    REQUIRE(c && c->session && owner == c->owner);
    if (epoch != c->epoch) {
        REQUIRE(position == 0 && epoch == c->epoch + 1);
        c->epoch = epoch; c->floor = -1;
    }
    REQUIRE(position >= 0 && position <= c->count);
    for (int i = 0; i < c->session->model->layers; ++i) {
        REQUIRE(position <= c->layers[i].cursor);
        c->layers[i].cursor = position;
    }
    c->count = position;
    c->rewinds++;
}

static inline int memory_cache_step(void *context, int layer, int position,
                                    const float *q, const float *k, const float *v,
                                    float *out, float *scores) {
    MemoryCache *c = (MemoryCache *)context;
    if (!c || !c->session || layer < 0 || layer >= c->session->model->layers ||
        c->session->session_id != c->owner || c->session->epoch != c->epoch ||
        c->session->phase != 2 || c->session->position != position ||
        position < 0 || position >= c->capacity) return -70;
    MemoryLayer *a = &c->layers[layer];
    if (a->cursor != position || !q || !k || !v || !out || !scores) return -70;
    const QwenModel *m = c->session->model;
    const int kv = m->kvheads, dim = m->head;
    c->calls++;
    int storage_position = position;
    if (c->policy == MEMORY_BYTE_RECENT4) storage_position = position % MEMORY_RESPONSE_RECENT;
    if (memory_is_byte(c->policy)) {
        if (gpu_encode_current(a->key_codes, a->value_codes, a->scales, k, v,
                               a->storage_capacity, storage_position, c->invalid, kv, dim)) return -71;
    } else if (c->policy == MEMORY_FP32_ALL) {
        gpu_cache(a->keys, k, kv, dim, a->storage_capacity, storage_position);
        gpu_cache(a->values, v, kv, dim, a->storage_capacity, storage_position);
    } else if (c->policy == MEMORY_ZERO) {
        gpu_zero(out, (size_t)m->hidden * sizeof(float));
    } else return -72;
    if (c->fail_layer == layer && c->fail_position == position) { c->fail_layer = -1; return -77; }
    a->cursor = position + 1; c->published++;
    if (c->policy == MEMORY_ZERO) return 0;
    int length = position + 1;
    int start = 0;
    if (c->policy == MEMORY_BYTE_RECENT4)
        start = length > MEMORY_RESPONSE_RECENT ? length - MEMORY_RESPONSE_RECENT : 0;
    else if (c->policy == MEMORY_FP32_ALL && c->floor >= 0) start = c->floor;
    if (start < 0 || start > position) return -73;
    int read_length = length - start;
    c->score_macs += (uint64_t)m->qheads * dim * read_length;
    c->value_macs += (uint64_t)m->qheads * dim * (c->policy == MEMORY_BYTE_TOP4 && read_length > 4 ? 4 : read_length);
    if (memory_is_byte(c->policy)) c->decoded_scalars += (uint64_t)2 * kv * dim * read_length;
    if (c->policy == MEMORY_BYTE_TOP4) c->value_probability_scans += (uint64_t)m->qheads * dim * read_length;
    if (c->policy == MEMORY_BYTE_ALL)
        gpu_byte_attention(m->gpu, out, q, a->key_codes, a->value_codes, a->scales,
                           scores, c->workspace, a->storage_capacity, length,
                           m->qheads, kv, dim);
    else if (c->policy == MEMORY_BYTE_TOP4)
        gpu_byte_top4_attention(m->gpu, out, q, a->key_codes, a->value_codes, a->scales,
                                scores, c->workspace, a->storage_capacity, length,
                                m->qheads, kv, dim);
    else if (c->policy == MEMORY_BYTE_RECENT4)
        gpu_recent_attention(m->gpu, out, q, a->key_codes, a->value_codes, a->scales,
                             scores, c->workspace, a->storage_capacity, start, length,
                             m->qheads, kv, dim);
    else if (c->policy == MEMORY_FP32_ALL)
        gpu_history_attention(m->gpu, out, q, a->keys, a->values, scores, c->workspace,
                              a->storage_capacity, start, length, m->qheads, kv, dim);
    else return -72;
    return 0;
}

/* Recent-ring rows are the only primary policy whose physical bytes can be
 * overwritten while a sibling branch is live.  Save a complete ring mark
 * before qwen_stack_push and restore it after qwen_stack_rollback. */
static inline void memory_mark_free(MemoryMark *m) {
    if (!m) return;
    free(m->key_codes); free(m->value_codes); free(m->scales);
    free(m->keys); free(m->values); memset(m, 0, sizeof(*m));
}

static inline int memory_mark_capture(MemoryCache *c, MemoryMark *m) {
    if (!c || !m) return -1;
    memory_mark_free(m); memset(m, 0, sizeof(*m));
    m->policy = c->policy; m->capacity = c->capacity; m->count = c->count; m->floor = c->floor;
    const int kv = c->session->model->kvheads, dim = c->session->model->head;
    int storage = c->layers[0].storage_capacity;
    for (int i = 0; i < c->session->model->layers; ++i) m->cursors[i] = c->layers[i].cursor;
    /* Full byte and FP32 banks are append-only logical prefixes: after a
     * rollback, rows at and beyond the restored cursor are outside the read
     * domain and need no physical snapshot.  Only the recent ring can overwrite
     * a still-live row, so only that arm copies bytes. */
    if (c->policy != MEMORY_BYTE_RECENT4) { m->valid = 1; return 0; }
    if (c->policy == MEMORY_BYTE_RECENT4) {
        m->code_bytes = (size_t)c->session->model->layers * kv * storage * dim;
        m->scale_bytes = (size_t)c->session->model->layers * 2 * kv * storage * sizeof(float);
        m->key_codes = (signed char *)malloc(m->code_bytes);
        m->value_codes = (signed char *)malloc(m->code_bytes);
        m->scales = (float *)malloc(m->scale_bytes);
        REQUIRE(m->key_codes && m->value_codes && m->scales);
        size_t co = 0, so = 0;
        for (int i = 0; i < c->session->model->layers; ++i) {
            MemoryLayer *a = &c->layers[i]; size_t cb = (size_t)kv * storage * dim, sb = (size_t)2 * kv * storage * sizeof(float);
            gpu_read(m->key_codes + co, a->key_codes, cb); gpu_read(m->value_codes + co, a->value_codes, cb); gpu_read(m->scales + so, a->scales, sb); co += cb; so += sb / sizeof(float);
        }
    }
    m->valid = 1; return 0;
}

static inline int memory_mark_restore(MemoryCache *c, const MemoryMark *m) {
    if (!c || !m || !m->valid || m->policy != c->policy) return -1;
    const int kv = c->session->model->kvheads, dim = c->session->model->head;
    int storage = c->layers[0].storage_capacity;
    for (int i = 0; i < c->session->model->layers; ++i) c->layers[i].cursor = m->cursors[i];
    c->count = m->count; c->floor = m->floor;
    if (c->policy == MEMORY_BYTE_RECENT4) {
        size_t co = 0, so = 0;
        for (int i = 0; i < c->session->model->layers; ++i) {
            size_t cb = (size_t)kv * storage * dim, sb = (size_t)2 * kv * storage * sizeof(float);
            gpu_write(c->layers[i].key_codes, m->key_codes + co, cb); gpu_write(c->layers[i].value_codes, m->value_codes + co, cb); gpu_write(c->layers[i].scales, m->scales + so, sb); co += cb; so += sb / sizeof(float);
        }
    }
    return 0;
}

typedef struct { uint32_t policy, length, floor, layer, position; unsigned char key[32], value[32]; } MemoryDigestRow;
typedef struct { char magic[8]; uint32_t policy, length, floor, layers, kvheads, dim; } MemoryDigestHeader;

static inline void memory_hash_row(const void *data, size_t bytes, unsigned char out[32]) { tcm_sha256(data, bytes, out); }

static inline void memory_cache_digest(const MemoryCache *c, int length, char out[65]) {
    REQUIRE(c && c->session && length >= 0 && length <= c->count);
    const QwenModel *m = c->session->model; const int kv = m->kvheads, dim = m->head;
    int first = 0;
    if (c->policy == MEMORY_BYTE_RECENT4) first = length > MEMORY_RESPONSE_RECENT ? length - MEMORY_RESPONSE_RECENT : 0;
    size_t rows = (size_t)m->layers * (length - first), bytes = sizeof(MemoryDigestHeader) + rows * sizeof(MemoryDigestRow);
    unsigned char *buffer = (unsigned char *)calloc(1, bytes); REQUIRE(buffer);
    MemoryDigestHeader head = {{'M','R','E','S','P','1',0,0},(uint32_t)c->policy,(uint32_t)length,(uint32_t)(c->floor < 0 ? 0xffffffffu : c->floor),(uint32_t)m->layers,(uint32_t)kv,(uint32_t)dim};
    memcpy(buffer, &head, sizeof(head)); size_t off = sizeof(head);
    size_t row_elems = (size_t)kv * dim;
    float *fp_k = (float *)malloc(row_elems * sizeof(float)), *fp_v = (float *)malloc(row_elems * sizeof(float));
    unsigned char *raw_k = (unsigned char *)malloc(row_elems + (size_t)kv * sizeof(float));
    unsigned char *raw_v = (unsigned char *)malloc(row_elems + (size_t)kv * sizeof(float));
    REQUIRE(fp_k && fp_v && raw_k && raw_v);
    for (int layer = 0; layer < m->layers; ++layer) {
        const MemoryLayer *a = &c->layers[layer];
        size_t storage = (size_t)a->storage_capacity, full = (size_t)kv * a->storage_capacity * dim;
        if (memory_is_byte(c->policy)) {
            signed char *kc = (signed char *)malloc(full), *vc = (signed char *)malloc(full); float *sc = (float *)malloc((size_t)2 * kv * a->storage_capacity * sizeof(float)); REQUIRE(kc && vc && sc);
            gpu_read(kc, a->key_codes, full); gpu_read(vc, a->value_codes, full); gpu_read(sc, a->scales, (size_t)2 * kv * a->storage_capacity * sizeof(float));
            for (int p = first; p < length; ++p) {
                int slot = c->policy == MEMORY_BYTE_RECENT4 ? p % MEMORY_RESPONSE_RECENT : p;
                size_t idx = 0; for (int h = 0; h < kv; ++h) memcpy(raw_k + idx, kc + ((size_t)h * storage + slot) * dim, (size_t)dim), idx += dim;
                for (int h = 0; h < kv; ++h) memcpy(raw_k + idx, sc + (size_t)h * storage + slot, sizeof(float)), idx += sizeof(float);
                idx = 0; for (int h = 0; h < kv; ++h) memcpy(raw_v + idx, vc + ((size_t)h * storage + slot) * dim, (size_t)dim), idx += dim;
                for (int h = 0; h < kv; ++h) memcpy(raw_v + idx, sc + (size_t)(kv + h) * storage + slot, sizeof(float)), idx += sizeof(float);
                MemoryDigestRow r; memset(&r, 0, sizeof(r)); r.policy=(uint32_t)c->policy; r.length=(uint32_t)length; r.floor=(uint32_t)(c->floor < 0 ? 0xffffffffu : c->floor); r.layer=(uint32_t)layer; r.position=(uint32_t)p;
                memory_hash_row(raw_k, row_elems + (size_t)kv * sizeof(float), r.key); memory_hash_row(raw_v, row_elems + (size_t)kv * sizeof(float), r.value); memcpy(buffer + off, &r, sizeof(r)); off += sizeof(r);
            }
            free(kc); free(vc); free(sc);
        } else if (c->policy == MEMORY_FP32_ALL) {
            float *kh = (float *)malloc(full * sizeof(float)), *vh = (float *)malloc(full * sizeof(float)); REQUIRE(kh && vh); gpu_read(kh, a->keys, full * sizeof(float)); gpu_read(vh, a->values, full * sizeof(float));
            for (int p = first; p < length; ++p) {
                for (int h = 0; h < kv; ++h) { memcpy(fp_k + (size_t)h * dim, kh + ((size_t)h * storage + p) * dim, (size_t)dim * sizeof(float)); memcpy(fp_v + (size_t)h * dim, vh + ((size_t)h * storage + p) * dim, (size_t)dim * sizeof(float)); }
                MemoryDigestRow r; memset(&r, 0, sizeof(r)); r.policy=(uint32_t)c->policy; r.length=(uint32_t)length; r.floor=(uint32_t)(c->floor < 0 ? 0xffffffffu : c->floor); r.layer=(uint32_t)layer; r.position=(uint32_t)p; memory_hash_row(fp_k, row_elems*sizeof(float), r.key); memory_hash_row(fp_v, row_elems*sizeof(float), r.value); memcpy(buffer + off, &r, sizeof(r)); off += sizeof(r);
            }
            free(kh); free(vh);
        }
    }
    REQUIRE(off == bytes); memory_hash_row(buffer, bytes, (unsigned char *)fp_k); tcm_hex((unsigned char *)fp_k, 32, out);
    free(fp_k); free(fp_v); free(raw_k); free(raw_v); free(buffer);
}

/* Dense native and FP32 callback digests share this row-hash representation so
 * the history-deleted prefix can be checked against the ordinary consumer. */
static inline void memory_dense_digest(const QwenSession *s, int length, char out[65]) {
    REQUIRE(s && s->model && length >= 0 && length <= s->capacity);
    const QwenModel *m=s->model; const int kv=m->kvheads, dim=m->head; size_t rows=(size_t)m->layers*length, bytes=sizeof(MemoryDigestHeader)+rows*sizeof(MemoryDigestRow); unsigned char *buffer=(unsigned char*)calloc(1,bytes); REQUIRE(buffer);
    MemoryDigestHeader head={{'M','R','E','S','P','1',0,0},(uint32_t)MEMORY_FP32_ALL,(uint32_t)length,0xffffffffu,(uint32_t)m->layers,(uint32_t)kv,(uint32_t)dim}; memcpy(buffer,&head,sizeof(head)); size_t off=sizeof(head); size_t row=(size_t)kv*dim; float *k=(float*)malloc(row*sizeof(float)),*v=(float*)malloc(row*sizeof(float)); REQUIRE(k&&v);
    for(int layer=0;layer<m->layers;layer++){int slot=s->cache_slots[layer];REQUIRE(slot>=0);size_t storage=(size_t)s->capacity;float *kh=(float*)malloc((size_t)kv*storage*dim*sizeof(float)),*vh=(float*)malloc((size_t)kv*storage*dim*sizeof(float));REQUIRE(kh&&vh);gpu_read(kh,s->keys+(size_t)slot*kv*storage*dim,(size_t)kv*storage*dim*sizeof(float));gpu_read(vh,s->values+(size_t)slot*kv*storage*dim,(size_t)kv*storage*dim*sizeof(float));for(int p=0;p<length;p++){for(int h=0;h<kv;h++){memcpy(k+(size_t)h*dim,kh+((size_t)h*storage+p)*dim,(size_t)dim*sizeof(float));memcpy(v+(size_t)h*dim,vh+((size_t)h*storage+p)*dim,(size_t)dim*sizeof(float));}MemoryDigestRow r;memset(&r,0,sizeof(r));r.policy=(uint32_t)MEMORY_FP32_ALL;r.length=(uint32_t)length;r.floor=0xffffffffu;r.layer=(uint32_t)layer;r.position=(uint32_t)p;memory_hash_row(k,row*sizeof(float),r.key);memory_hash_row(v,row*sizeof(float),r.value);memcpy(buffer+off,&r,sizeof(r));off+=sizeof(r);}free(kh);free(vh);}REQUIRE(off==bytes);memory_hash_row(buffer,bytes,(unsigned char*)k);tcm_hex((unsigned char*)k,32,out);free(k);free(v);free(buffer);
}

#endif
