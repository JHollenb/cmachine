#define _POSIX_C_SOURCE 200809L

/*
 * Persistent whole-token driver for the transition-laws experiment.
 *
 * The executable is deliberately a small protocol server rather than a
 * one-shot benchmark.  The model is loaded once, COMPILE creates an
 * authenticated layer-indexed factor program, and RUN advances independent
 * sessions through the exact native Qwen runtime.  Only gate/up construction
 * is intercepted.  RMSNorm, SiLU, down projection, residual addition, Q/K/V,
 * attention, the full FP32 KV cache, and the vocabulary head remain native.
 *
 * Protocol (stdin/stdout):
 *   READY
 *   COMPILE p0|p1
 *   DONE COMPILE p0|p1
 *   RUN <label> <none|p0|p1> <mask:uint32> <capture:0|1> <ncases>
 *   <case-id> <budget>                 (one line per case)
 *   DONE <label>
 *   QUIT
 *
 * RESULTS/<program>/E-<layer>.f32 is row-major [256,1536].  COMPILE forms
 * Lg = G E^T and Lu = U E^T, stored row-major [8960,256], by 256 native
 * GPU matvecs per native image.  The source E image remains resident for
 * serving; the factor images are retained on the device and are also saved
 * with hashes and a JSON manifest.
 */

#include "qwen.h"
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define REQUIRE(x) do { \
    if (!(x)) { \
        fprintf(stderr, "transition-laws %s:%d %s\n", __FILE__, __LINE__, #x); \
        exit(2); \
    } \
} while (0)

enum {
    LAYERS = 28,
    D = 1536,
    H = 8960,
    V = 151936,
    RANK = 256,
    CAPACITY = 512,
    MAX_BUDGET = 64,
    MAX_CASES = 32,
    MAX_LABEL = 96,
    MAX_CASE_ID = 96
};

static const char *results_dir;
static QwenModel *model_active;
static int compiling_program;
static uint64_t compile_linear_calls;
static uint64_t compile_linear_macs;
static uint64_t total_linear_calls;
static uint64_t total_linear_macs;
static int diagnostic_pass;
static uint64_t diagnostic_linear_calls;
static uint64_t diagnostic_linear_macs;
static int shadow_pass;
static uint64_t shadow_linear_calls;
static uint64_t shadow_linear_macs;
static int sham_pass;
static uint64_t sham_linear_calls;
static uint64_t sham_linear_macs;

static double clock_seconds(void) {
    struct timespec t;
    REQUIRE(!clock_gettime(CLOCK_MONOTONIC, &t));
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static void digest_bytes(const void *data, size_t n, char out[65]) {
    unsigned char raw[32];
    tcm_sha256(data, n, raw);
    tcm_hex(raw, 32, out);
}

static void finite_read(float *host, const float *device, size_t n) {
    gpu_read(host, device, n * sizeof(float));
    for (size_t i = 0; i < n; i++) REQUIRE(isfinite(host[i]));
}

static void save_file(const char *relative, const void *data, size_t n) {
    char path[4096];
    REQUIRE(snprintf(path, sizeof(path), "%s/%s", results_dir, relative) < (int)sizeof(path));
    FILE *f = fopen(path, "wb");
    REQUIRE(f && fwrite(data, 1, n, f) == n && !fclose(f));
}

static void save_file_path(const char *path, const void *data, size_t n) {
    FILE *f = fopen(path, "wb");
    REQUIRE(f && fwrite(data, 1, n, f) == n && !fclose(f));
}

static void make_directory(const char *relative) {
    char path[4096];
    REQUIRE(snprintf(path, sizeof(path), "%s/%s", results_dir, relative) < (int)sizeof(path));
    if (mkdir(path, 0775) && errno != EEXIST) {
        fprintf(stderr, "transition-laws cannot create %s: %s\n", path, strerror(errno));
        exit(2);
    }
}

static int safe_component(const char *s) {
    if (!s || !*s) return 0;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++)
        if (!isalnum(*p) && *p != '-' && *p != '_' && *p != '.') return 0;
    return 1;
}

static void write_ints(FILE *f, const int *v, int n) {
    fputc('[', f);
    for (int i = 0; i < n; i++) fprintf(f, "%s%d", i ? "," : "", v[i]);
    fputc(']', f);
}

static int *read_tokens(const char *case_id, const char *suffix, int *count) {
    char path[4096], stem[256];
    REQUIRE(safe_component(case_id) && safe_component(suffix));
    REQUIRE(snprintf(stem, sizeof(stem), "%s-%s.txt", case_id, suffix) < (int)sizeof(stem));
    REQUIRE(snprintf(path, sizeof(path), "%s/%s", results_dir, stem) < (int)sizeof(path));
    FILE *f = fopen(path, "r");
    int *ids = calloc(CAPACITY, sizeof(*ids));
    REQUIRE(f && ids);
    *count = 0;
    int value;
    while (fscanf(f, "%d", &value) == 1) {
        REQUIRE(*count < CAPACITY && value >= 0 && value < V);
        ids[(*count)++] = value;
    }
    REQUIRE(feof(f) && *count > 0 && !fclose(f));
    return ids;
}

typedef struct {
    int ready;
    char name[8];
    float *e[LAYERS];
    float *gate[LAYERS];
    float *up[LAYERS];
    char e_hash[LAYERS][65];
    char gate_hash[LAYERS][65];
    char up_hash[LAYERS][65];
    uint64_t compile_calls;
    uint64_t compile_macs;
    size_t gpu_bytes;
} FactorProgram;

typedef struct {
    int start;
    int ingested;
    int skipped;
    int prefill;
    int final;
    int budget;
    int answers[MAX_BUDGET];
    int answer_count;
    int stopped;
    int truncated;
    int decision_offset;
    int decision_count;
    char prefill_logits[256];
    char final_logits[256];
    char prefill_x[256];
    char final_x[256];
    char prefill_write[256];
    char final_write[256];
    char prefill_native_write[256];
    char final_native_write[256];
} TurnInfo;

typedef struct {
    char case_id[MAX_CASE_ID];
    char label[MAX_LABEL];
    char program_name[8];
    uint32_t mask;
    int capture;
    int budget;
    int ids[CAPACITY];
    int tops[CAPACITY];
    int count;
    TurnInfo turns[2];
    float *decision_logits;
    int decision_count;
    int decision_capacity;
    float *stream_x;
    float *boundary_x;
    float *boundary_write;
    int boundary_pending;
    int boundary_kind;
    int boundary_turn;
    uint8_t *coord_ready;
    float *coords;
    QwenSession *session;
    FactorProgram *program;
    uint64_t linear_calls;
    uint64_t linear_macs;
    uint64_t native_linear_calls;
    uint64_t native_linear_macs;
    uint64_t compiled_linear_calls;
    uint64_t compiled_linear_macs;
    uint64_t mlp_coordinate_calls;
    uint64_t compiled_gate_up_calls;
    uint64_t native_gate_up_calls;
    uint64_t native_swiglu_calls;
    uint64_t native_qkv_calls;
    uint64_t native_attention_calls;
    size_t session_gpu_bytes;
    size_t factor_gpu_bytes;
    size_t stream_bytes;
    double seconds;
    char candidate_decision_file[256];
    char native_shadow_file[256];
    char consumed_file[256];
    char tops_file[256];
    char x_file[256];
} PathInfo;

static PathInfo *active_run;

/* The pre-macro helper calls the real CUDA matvec.  Every runtime call is
 * counted here, while transition_matvec below attributes the chosen MLP
 * operation to native or compiled arithmetic. */
static void native_matvec(TcmGpu *gpu, const float *w, int in, int out,
                          const float *x, float *y) {
    gpu_matvec(gpu, w, in, out, x, y);
    total_linear_calls++;
    total_linear_macs += (uint64_t)in * (uint64_t)out;
    if (compiling_program) {
        compile_linear_calls++;
        compile_linear_macs += (uint64_t)in * (uint64_t)out;
    }
    if (diagnostic_pass) {
        diagnostic_linear_calls++;
        diagnostic_linear_macs += (uint64_t)in * (uint64_t)out;
    }
    if (shadow_pass) {
        shadow_linear_calls++;
        shadow_linear_macs += (uint64_t)in * (uint64_t)out;
    }
    if (sham_pass) {
        sham_linear_calls++;
        sham_linear_macs += (uint64_t)in * (uint64_t)out;
    }
    if (active_run) {
        active_run->linear_calls++;
        active_run->linear_macs += (uint64_t)in * (uint64_t)out;
    }
}

static void free_program(FactorProgram *p) {
    if (!p) return;
    for (int layer = 0; layer < LAYERS; layer++) {
        gpu_free(p->e[layer]);
        gpu_free(p->gate[layer]);
        gpu_free(p->up[layer]);
    }
    memset(p, 0, sizeof(*p));
}

static float *load_f32_file(const char *path, size_t n, char hash[65]) {
    FILE *f = fopen(path, "rb");
    float *host = malloc(n * sizeof(*host));
    REQUIRE(f && host && fread(host, sizeof(*host), n, f) == n && fgetc(f) == EOF && !fclose(f));
    for (size_t i = 0; i < n; i++) REQUIRE(isfinite(host[i]));
    digest_bytes(host, n * sizeof(*host), hash);
    return host;
}

static void compile_factor_columns(FactorProgram *p, int layer, const float *e_host,
                                   const char *program_dir, const QwenLayer *native) {
    const size_t e_count = (size_t)RANK * D;
    const size_t factor_count = (size_t)H * RANK;
    float *gate_host = calloc(factor_count, sizeof(*gate_host));
    float *up_host = calloc(factor_count, sizeof(*up_host));
    float *column_host = malloc((size_t)H * sizeof(*column_host));
    float *e_device = gpu_weight(e_host, TCM_F32, e_count);
    float *column_device = gpu_alloc((size_t)H * sizeof(*column_device));
    REQUIRE(gate_host && up_host && column_host && e_device && column_device);
    for (int col = 0; col < RANK; col++) {
        native_matvec(model_active->gpu, native->gate, D, H,
                      e_device + (size_t)col * D, column_device);
        finite_read(column_host, column_device, H);
        for (int row = 0; row < H; row++) gate_host[(size_t)row * RANK + col] = column_host[row];
        native_matvec(model_active->gpu, native->up, D, H,
                      e_device + (size_t)col * D, column_device);
        finite_read(column_host, column_device, H);
        for (int row = 0; row < H; row++) up_host[(size_t)row * RANK + col] = column_host[row];
    }
    digest_bytes(gate_host, factor_count * sizeof(*gate_host), p->gate_hash[layer]);
    digest_bytes(up_host, factor_count * sizeof(*up_host), p->up_hash[layer]);

    char path[4096], name[256];
    REQUIRE(snprintf(name, sizeof(name), "%s/Lg-%d.f32", program_dir, layer) < (int)sizeof(name));
    REQUIRE(snprintf(path, sizeof(path), "%s/%s", results_dir, name) < (int)sizeof(path));
    save_file_path(path, gate_host, factor_count * sizeof(*gate_host));
    REQUIRE(snprintf(name, sizeof(name), "%s/Lu-%d.f32", program_dir, layer) < (int)sizeof(name));
    REQUIRE(snprintf(path, sizeof(path), "%s/%s", results_dir, name) < (int)sizeof(path));
    save_file_path(path, up_host, factor_count * sizeof(*up_host));

    p->gate[layer] = gpu_weight(gate_host, TCM_F32, factor_count);
    p->up[layer] = gpu_weight(up_host, TCM_F32, factor_count);
    REQUIRE(p->gate[layer] && p->up[layer]);
    p->gpu_bytes += factor_count * sizeof(float) * 2;
    gpu_free(e_device);
    free(gate_host);
    free(up_host);
    free(column_host);
    gpu_free(column_device);
}

static void compile_program(FactorProgram *p, const char *name) {
    REQUIRE(!p->ready && (!strcmp(name, "p0") || !strcmp(name, "p1")));
    memset(p, 0, sizeof(*p));
    strncpy(p->name, name, sizeof(p->name) - 1);
    char program_dir[64], path[4096];
    REQUIRE(snprintf(program_dir, sizeof(program_dir), "%s", name) < (int)sizeof(program_dir));
    make_directory(program_dir);
    compile_linear_calls = compile_linear_macs = 0;
    compiling_program = 1;
    for (int layer = 0; layer < LAYERS; layer++) {
        char relative[256];
        REQUIRE(snprintf(relative, sizeof(relative), "%s/E-%d.f32", name, layer) < (int)sizeof(relative));
        REQUIRE(snprintf(path, sizeof(path), "%s/%s", results_dir, relative) < (int)sizeof(path));
        float *host = load_f32_file(path, (size_t)RANK * D, p->e_hash[layer]);
        p->e[layer] = gpu_weight(host, TCM_F32, (size_t)RANK * D);
        REQUIRE(p->e[layer]);
        p->gpu_bytes += (size_t)RANK * D * sizeof(float);
        compile_factor_columns(p, layer, host, program_dir, &model_active->layer[layer]);
        free(host);
    }
    compiling_program = 0;
    p->compile_calls = compile_linear_calls;
    p->compile_macs = compile_linear_macs;
    char manifest[4096];
    REQUIRE(snprintf(manifest, sizeof(manifest), "%s/%s/compile.json", results_dir, name) < (int)sizeof(manifest));
    FILE *f = fopen(manifest, "w");
    REQUIRE(f);
    fprintf(f, "{\"schema\":\"tcm-transition-laws-program-v1\",\"program\":\"%s\",\"rank\":%d,\"layers\":%d,\"hidden\":%d,\"intermediate\":%d,\"model_identity\":\"%s\",\"model_digest\":\"%s\",\"compile_linear_calls\":%llu,\"compile_linear_macs\":%llu,\"gpu_bytes\":%zu,\"files\":[",
            name, RANK, LAYERS, D, H, model_active->identity, model_active->digest,
            (unsigned long long)p->compile_calls, (unsigned long long)p->compile_macs, p->gpu_bytes);
    for (int layer = 0; layer < LAYERS; layer++) {
        fprintf(f, "%s{\"layer\":%d,\"e\":\"%s/E-%d.f32\",\"e_sha256\":\"%s\",\"gate\":\"%s/Lg-%d.f32\",\"gate_sha256\":\"%s\",\"up\":\"%s/Lu-%d.f32\",\"up_sha256\":\"%s\"}",
                layer ? "," : "", layer, name, layer, p->e_hash[layer], name, layer,
                p->gate_hash[layer], name, layer, p->up_hash[layer]);
    }
    fprintf(f, "]}\n");
    REQUIRE(!fclose(f));
    p->ready = 1;
    fprintf(stderr, "compiled %s rank=%d layers=%d calls=%llu macs=%llu bytes=%zu\n", name, RANK, LAYERS,
            (unsigned long long)p->compile_calls, (unsigned long long)p->compile_macs, p->gpu_bytes);
}

static FactorProgram *find_program(const char *name, FactorProgram programs[2]) {
    if (!strcmp(name, "p0")) return &programs[0];
    if (!strcmp(name, "p1")) return &programs[1];
    return NULL;
}

/* This is the sole serving interception.  Pointer identity ties the branch to
 * the model's actual layer weights; mask and program are both checked before
 * any native gate/up call can be bypassed. */
static void transition_matvec(TcmGpu *gpu, const float *w, int in, int out,
                              const float *x, float *y) {
    PathInfo *r = active_run;
    if (!r) {
        native_matvec(gpu, w, in, out, x, y);
        return;
    }
    if (!r->program || !r->program->ready || r->session->layer < 0 || r->session->layer >= LAYERS) {
        const QwenLayer *native = (r->session->layer >= 0 && r->session->layer < LAYERS)
            ? &r->session->model->layer[r->session->layer] : NULL;
        r->native_linear_calls++;
        r->native_linear_macs += (uint64_t)in * (uint64_t)out;
        if (native && (w == native->gate || w == native->up)) r->native_gate_up_calls++;
        if (native && (w == native->q || w == native->k || w == native->v)) r->native_qkv_calls++;
        native_matvec(gpu, w, in, out, x, y);
        return;
    }
    int layer = r->session->layer;
    const QwenLayer *native = &r->session->model->layer[layer];
    int gate = w == native->gate;
    int up = w == native->up;
    if (!(r->mask & (1u << layer)) || (!gate && !up)) {
        r->native_linear_calls++;
        r->native_linear_macs += (uint64_t)in * (uint64_t)out;
        native_matvec(gpu, w, in, out, x, y);
        if (gate || up) {
            r->native_gate_up_calls++;
        }
        if (w == native->q || w == native->k || w == native->v) r->native_qkv_calls++;
        return;
    }
    REQUIRE(in == D && out == H && r->program->e[layer] && r->program->gate[layer] && r->program->up[layer]);
    if (!r->coord_ready[layer]) {
        native_matvec(gpu, r->program->e[layer], D, RANK, x,
                      r->coords + (size_t)layer * RANK);
        r->coord_ready[layer] = 1;
        r->mlp_coordinate_calls++;
        r->compiled_linear_calls++;
        r->compiled_linear_macs += (uint64_t)D * RANK;
    }
    const float *factor = gate ? r->program->gate[layer] : r->program->up[layer];
    native_matvec(gpu, factor, RANK, H,
                  r->coords + (size_t)layer * RANK, y);
    r->compiled_gate_up_calls++;
    r->compiled_linear_calls++;
    r->compiled_linear_macs += (uint64_t)RANK * H;
}

static void native_attention(TcmGpu *gpu, float *out, const float *q,
                             const float *keys, const float *values, float *scores,
                             int qheads, int kvheads, int dimension, int capacity, int length) {
    if (active_run) active_run->native_attention_calls++;
    gpu_attention(gpu, out, q, keys, values, scores, qheads, kvheads,
                  dimension, capacity, length);
}

#define gpu_matvec transition_matvec
#define gpu_attention native_attention
#include "runtime/qwen.c"
#undef gpu_matvec
#undef gpu_attention

static void observer(void *context, int position, int layer, QwenPort port,
                     const float *data, size_t n) {
    PathInfo *r = context;
    REQUIRE(r && r->session && position == r->count && layer >= 0 && layer < LAYERS);
    if (port == QWEN_MLP_INPUT) {
        REQUIRE(n == D);
        if (r->capture && r->stream_x)
            finite_read(r->stream_x + ((size_t)position * LAYERS + (size_t)layer) * D, data, D);
        if (r->boundary_pending)
            finite_read(r->boundary_x + (size_t)layer * D, data, D);
    } else if (port == QWEN_MLP_RETURN) {
        REQUIRE(n == D);
        if (r->boundary_pending)
            finite_read(r->boundary_write + (size_t)layer * D, data, D);
    }
}

static void cache_digest(const QwenSession *s, char out[65]) {
    size_t row = (size_t)s->position * s->model->head;
    size_t n = (size_t)s->model->layers * s->model->kvheads * row;
    float *host = malloc(2 * n * sizeof(*host));
    REQUIRE(host);
    size_t offset = 0;
    for (int kind = 0; kind < 2; kind++) {
        for (int layer = 0; layer < s->model->layers; layer++) {
            for (int group = 0; group < s->model->kvheads; group++) {
                const float *base = (kind ? s->values : s->keys) +
                    ((size_t)layer * s->model->kvheads + (size_t)group) * s->capacity * s->model->head;
                finite_read(host + offset, base, row);
                offset += row;
            }
        }
    }
    digest_bytes(host, offset * sizeof(*host), out);
    free(host);
}

static void save_boundary(PathInfo *r, TurnInfo *turn, const char *stage,
                          int turn_number, int position) {
    char stem[256], path[256], hash[65];
    REQUIRE(snprintf(stem, sizeof(stem), "%s-%s-t%d-%s", r->case_id, r->label,
                     turn_number, stage) < (int)sizeof(stem));
    float *logits = malloc((size_t)V * sizeof(*logits));
    REQUIRE(logits);
    finite_read(logits, r->session->logits, V);
    REQUIRE(snprintf(path, sizeof(path), "%s-logits.f32", stem) < (int)sizeof(path));
    save_file(path, logits, (size_t)V * sizeof(*logits));
    digest_bytes(logits, (size_t)V * sizeof(*logits), hash);
    free(logits);
    if (stage[0] == 'p') {
        strncpy(turn->prefill_logits, path, sizeof(turn->prefill_logits) - 1);
        strncpy(turn->prefill_x, "", sizeof(turn->prefill_x));
        strncpy(turn->prefill_write, "", sizeof(turn->prefill_write));
    } else {
        strncpy(turn->final_logits, path, sizeof(turn->final_logits) - 1);
        strncpy(turn->final_x, "", sizeof(turn->final_x));
        strncpy(turn->final_write, "", sizeof(turn->final_write));
    }
    REQUIRE(snprintf(path, sizeof(path), "%s-x.f32", stem) < (int)sizeof(path));
    save_file(path, r->boundary_x, (size_t)LAYERS * D * sizeof(float));
    if (stage[0] == 'p') strncpy(turn->prefill_x, path, sizeof(turn->prefill_x) - 1);
    else strncpy(turn->final_x, path, sizeof(turn->final_x) - 1);
    REQUIRE(snprintf(path, sizeof(path), "%s-write.f32", stem) < (int)sizeof(path));
    save_file(path, r->boundary_write, (size_t)LAYERS * D * sizeof(float));
    if (stage[0] == 'p') strncpy(turn->prefill_write, path, sizeof(turn->prefill_write) - 1);
    else strncpy(turn->final_write, path, sizeof(turn->final_write) - 1);
    cache_digest(r->session, hash);
    fprintf(stderr, "%s %s boundary=%s position=%d top=%d kv=%s\n", r->label, r->case_id,
            stage, position, r->session->pending, hash);
}

static void advance_one(PathInfo *r, int token, int boundary_pending) {
    REQUIRE(r->count < CAPACITY && r->session->position == r->count);
    memset(r->coord_ready, 0, LAYERS);
    r->boundary_pending = boundary_pending;
    active_run = r;
    REQUIRE(!qwen_begin_token(r->session, token));
    while (r->session->layer < LAYERS) {
        r->session->observer = observer;
        r->session->observer_context = r;
        r->session->observe_position = r->count;
        r->session->observe_layer = r->session->layer;
        REQUIRE(!qwen_advance_layer(r->session));
    }
    REQUIRE(!qwen_finish_token(r->session));
    r->ids[r->count] = token;
    r->tops[r->count] = r->session->pending;
    r->count++;
    r->boundary_pending = 0;
    active_run = NULL;
}

static void fresh_session(QwenSession *s) {
    REQUIRE(!qwen_session_init(s, model_active, CAPACITY));
    size_t n = (size_t)model_active->layers * model_active->kvheads * CAPACITY * model_active->head;
    gpu_zero(s->keys, n * sizeof(float));
    gpu_zero(s->values, n * sizeof(float));
}

static void allocate_run_buffers(PathInfo *r, int token_capacity, int capture) {
    r->boundary_x = calloc((size_t)LAYERS * D, sizeof(float));
    r->boundary_write = calloc((size_t)LAYERS * D, sizeof(float));
    r->coord_ready = calloc(LAYERS, sizeof(uint8_t));
    r->coords = gpu_alloc((size_t)LAYERS * RANK * sizeof(float));
    REQUIRE(r->boundary_x && r->boundary_write && r->coord_ready && r->coords);
    if (capture) {
        (void)token_capacity;
        r->stream_x = calloc((size_t)CAPACITY * LAYERS * D, sizeof(float));
        REQUIRE(r->stream_x);
    }
    r->decision_capacity = token_capacity;
    r->decision_logits = calloc((size_t)token_capacity * V, sizeof(float));
    REQUIRE(r->decision_logits);
}

static void free_run_buffers(PathInfo *r) {
    free(r->stream_x);
    free(r->boundary_x);
    free(r->boundary_write);
    free(r->decision_logits);
    free(r->coord_ready);
    gpu_free(r->coords);
    r->stream_x = r->boundary_x = r->boundary_write = r->decision_logits = NULL;
    r->coord_ready = NULL;
    r->coords = NULL;
}

static void record_decision(PathInfo *r, TurnInfo *turn) {
    REQUIRE(r->decision_count < r->decision_capacity);
    finite_read(r->decision_logits + (size_t)r->decision_count * V, r->session->logits, V);
    if (!turn->decision_count) turn->decision_offset = r->decision_count;
    turn->decision_count++;
    r->decision_count++;
}

static void save_primary_arrays(PathInfo *r) {
    char stem[256], path[256];
    REQUIRE(snprintf(stem, sizeof(stem), "%s-%s", r->case_id, r->label) < (int)sizeof(stem));
    REQUIRE(snprintf(path, sizeof(path), "%s-consumed.i32", stem) < (int)sizeof(path));
    save_file(path, r->ids, (size_t)r->count * sizeof(int));
    strncpy(r->consumed_file, path, sizeof(r->consumed_file) - 1);
    REQUIRE(snprintf(path, sizeof(path), "%s-tops.i32", stem) < (int)sizeof(path));
    save_file(path, r->tops, (size_t)r->count * sizeof(int));
    strncpy(r->tops_file, path, sizeof(r->tops_file) - 1);
    REQUIRE(snprintf(path, sizeof(path), "%s-candidate-decision-logits.f32", stem) < (int)sizeof(path));
    save_file(path, r->decision_logits, (size_t)r->decision_count * V * sizeof(float));
    strncpy(r->candidate_decision_file, path, sizeof(r->candidate_decision_file) - 1);
    if (r->capture) {
        REQUIRE(snprintf(path, sizeof(path), "%s-x.f32", stem) < (int)sizeof(path));
        save_file(path, r->stream_x, (size_t)r->count * LAYERS * D * sizeof(float));
        strncpy(r->x_file, path, sizeof(r->x_file) - 1);
        r->stream_bytes = (size_t)r->count * LAYERS * D * sizeof(float);
    }
}

/* Re-evaluate each saved candidate post-norm input with the original layer
 * G/U/D weights, after the candidate path is closed.  This is a local
 * producer diagnostic: it has no attention, residual suffix, or cache write.
 * It therefore separates an error in the factor response at the same input
 * from the later autoregressive state/cache effect measured by the shadow. */
static void native_write_diagnostic(PathInfo *r, const TurnInfo *turn, const char *stage,
                                   char out_file[256]) {
    char path[256];
    const char *input_file = stage[0] == 'p' ? turn->prefill_x : turn->final_x;
    REQUIRE(input_file[0]);
    size_t count = (size_t)LAYERS * D;
    REQUIRE(snprintf(path, sizeof(path), "%s/%s", results_dir, input_file) < (int)sizeof(path));
    FILE *f = fopen(path, "rb");
    float *host = malloc(count * sizeof(*host));
    float *write_host = malloc(count * sizeof(*write_host));
    float *input_device = gpu_alloc((size_t)D * sizeof(float));
    float *gate_device = gpu_alloc((size_t)H * sizeof(float));
    float *up_device = gpu_alloc((size_t)H * sizeof(float));
    float *write_device = gpu_alloc((size_t)D * sizeof(float));
    REQUIRE(f && host && write_host && input_device && gate_device && up_device && write_device);
    REQUIRE(fread(host, sizeof(*host), count, f) == count && fgetc(f) == EOF && !fclose(f));
    for (size_t i = 0; i < count; i++) REQUIRE(isfinite(host[i]));
    diagnostic_pass = 1;
    for (int layer = 0; layer < LAYERS; layer++) {
        const QwenLayer *l = &model_active->layer[layer];
        gpu_write(input_device, host + (size_t)layer * D, (size_t)D * sizeof(float));
        native_matvec(model_active->gpu, l->gate, D, H, input_device, gate_device);
        native_matvec(model_active->gpu, l->up, D, H, input_device, up_device);
        gpu_swiglu(gate_device, up_device, H);
        native_matvec(model_active->gpu, l->down, H, D, gate_device, write_device);
        finite_read(write_host + (size_t)layer * D, write_device, D);
    }
    diagnostic_pass = 0;
    char stem[256];
    REQUIRE(snprintf(stem, sizeof(stem), "%s-%s-t%d-%s-native-write.f32", r->case_id, r->label,
                     (int)(turn - r->turns), stage) < (int)sizeof(stem));
    save_file(stem, write_host, count * sizeof(*write_host));
    strncpy(out_file, stem, 255);
    out_file[255] = 0;
    free(host);
    free(write_host);
    gpu_free(input_device);
    gpu_free(gate_device);
    gpu_free(up_device);
    gpu_free(write_device);
}

static void write_primary_json(const PathInfo *r) {
    char path[4096];
    REQUIRE(snprintf(path, sizeof(path), "%s/%s-%s-primary.json", results_dir, r->case_id, r->label) < (int)sizeof(path));
    FILE *f = fopen(path, "w");
    REQUIRE(f);
    fprintf(f, "{\"schema\":\"tcm-transition-laws-primary-v1\",\"case_id\":\"%s\",\"label\":\"%s\",\"program\":\"%s\",\"mask\":%u,\"model_identity\":\"%s\",\"model_digest\":\"%s\",\"own_generation\":true,\"count\":%d,\"consumed_ids_file\":\"%s\",\"tops_file\":\"%s\",\"consumed_ids\":",
            r->case_id, r->label, r->program_name, r->mask, model_active->identity, model_active->digest,
            r->count, r->consumed_file, r->tops_file);
    write_ints(f, r->ids, r->count);
    fprintf(f, ",\"tops\":");
    write_ints(f, r->tops, r->count);
    fprintf(f, ",\"candidate_decision_logits_file\":\"%s\",\"candidate_logits_file\":\"%s\",\"candidate_decision_logits_shape\":[%d,%d],\"capture\":%s,\"x_file\":\"%s\",\"receiver_file\":\"%s\",\"turns\":[",
            r->candidate_decision_file, r->candidate_decision_file, r->decision_count, V,
            r->capture ? "true" : "false", r->capture ? r->x_file : "", r->capture ? r->x_file : "");
    for (int t = 0; t < 2; t++) {
        const TurnInfo *turn = &r->turns[t];
        fprintf(f, "%s{\"turn\":%d,\"start\":%d,\"ingested\":%d,\"template_skip\":%d,\"prefill\":%d,\"final\":%d,\"budget\":%d,\"answer_ids\":",
                t ? "," : "", t, turn->start, turn->ingested, turn->skipped, turn->prefill,
                turn->final, turn->budget);
        write_ints(f, turn->answers, turn->answer_count);
        fprintf(f, ",\"stopped\":%s,\"truncated\":%s,\"decision_offset\":%d,\"decision_count\":%d,\"decision_positions\":[",
                turn->stopped ? "true" : "false", turn->truncated ? "true" : "false",
                turn->decision_offset, turn->decision_count);
        for (int j = 0; j < turn->decision_count; j++) fprintf(f, "%s%d", j ? "," : "", turn->prefill - 1 + j);
        fprintf(f, "],\"boundaries\":{\"prefill\":{\"logits_file\":\"%s\",\"x_file\":\"%s\",\"write_file\":\"%s\",\"native_write_file\":\"%s\"},\"final\":{\"logits_file\":\"%s\",\"x_file\":\"%s\",\"write_file\":\"%s\",\"native_write_file\":\"%s\"}}}",
                turn->prefill_logits, turn->prefill_x, turn->prefill_write,
                turn->prefill_native_write, turn->final_logits, turn->final_x,
                turn->final_write, turn->final_native_write);
    }
    fprintf(f, "],\"native_shadow_logits_file\":\"%s\",\"native_shadow_status\":\"pending\",\"linear_calls\":%llu,\"linear_macs\":%llu,\"native_linear_calls\":%llu,\"native_linear_macs\":%llu,\"compiled_linear_calls\":%llu,\"compiled_linear_macs\":%llu,\"mlp_coordinate_calls\":%llu,\"compiled_gate_up_calls\":%llu,\"native_gate_up_calls\":%llu,\"session_gpu_bytes\":%zu,\"factor_gpu_bytes\":%zu,\"stream_bytes\":%zu}\n",
            "", (unsigned long long)r->linear_calls, (unsigned long long)r->linear_macs,
            (unsigned long long)r->native_linear_calls, (unsigned long long)r->native_linear_macs,
            (unsigned long long)r->compiled_linear_calls, (unsigned long long)r->compiled_linear_macs,
            (unsigned long long)r->mlp_coordinate_calls, (unsigned long long)r->compiled_gate_up_calls,
            (unsigned long long)r->native_gate_up_calls, r->session->allocated_bytes,
            r->program ? r->program->gpu_bytes : 0, r->stream_bytes);
    REQUIRE(!fclose(f));
}

static void native_shadow(PathInfo *r) {
    QwenSession shadow;
    fresh_session(&shadow);
    float *logits = calloc((size_t)r->decision_count * V, sizeof(*logits));
    REQUIRE(logits);
    int cursor = 0;
    shadow_pass = 1;
    for (int t = 0; t < 2; t++) {
        const TurnInfo *turn = &r->turns[t];
        REQUIRE(cursor == turn->start);
        while (cursor < turn->prefill) {
            shadow.observer = NULL;
            shadow.observe_layer = shadow.observe_position = -1;
            REQUIRE(!qwen_push(&shadow, r->ids[cursor++]));
        }
        for (int j = 0; j < turn->answer_count; j++) {
            finite_read(logits + (size_t)(turn->decision_offset + j) * V, shadow.logits, V);
            REQUIRE(cursor == turn->prefill + j);
            REQUIRE(!qwen_push(&shadow, r->ids[cursor++]));
        }
        REQUIRE(cursor == turn->final);
    }
    shadow_pass = 0;
    REQUIRE(cursor == r->count);
    char stem[256], path[256];
    REQUIRE(snprintf(stem, sizeof(stem), "%s-%s", r->case_id, r->label) < (int)sizeof(stem));
    REQUIRE(snprintf(path, sizeof(path), "%s-native-shadow-decision-logits.f32", stem) < (int)sizeof(path));
    save_file(path, logits, (size_t)r->decision_count * V * sizeof(float));
    strncpy(r->native_shadow_file, path, sizeof(r->native_shadow_file) - 1);
    free(logits);
    qwen_session_free(&shadow);
}

static void update_shadow_json(const PathInfo *r) {
    char path[4096];
    REQUIRE(snprintf(path, sizeof(path), "%s/%s-%s-shadow.json", results_dir, r->case_id, r->label) < (int)sizeof(path));
    FILE *f = fopen(path, "w");
    REQUIRE(f);
    fprintf(f, "{\"schema\":\"tcm-transition-laws-native-shadow-v1\",\"case_id\":\"%s\",\"label\":\"%s\",\"own_history\":true,\"native_shadow_logits_file\":\"%s\",\"shape\":[%d,%d],\"decision_positions\":[",
            r->case_id, r->label, r->native_shadow_file, r->decision_count, V);
    int first = 1;
    for (int t = 0; t < 2; t++) {
        for (int j = 0; j < r->turns[t].decision_count; j++) {
            fprintf(f, "%s%d", first ? "" : ",", r->turns[t].prefill - 1 + j);
            first = 0;
        }
    }
    fprintf(f, "]}\n");
    REQUIRE(!fclose(f));
}

static void write_path_json(FILE *f, const PathInfo *r, int comma) {
    fprintf(f, "%s{\"case_id\":\"%s\",\"label\":\"%s\",\"program\":\"%s\",\"mask\":%u,\"own_generation\":true,\"count\":%d,\"consumed_ids_file\":\"%s\",\"tops_file\":\"%s\",\"consumed_ids\":",
            comma ? "," : "", r->case_id, r->label, r->program_name, r->mask, r->count,
            r->consumed_file, r->tops_file);
    write_ints(f, r->ids, r->count);
    fprintf(f, ",\"tops\":");
    write_ints(f, r->tops, r->count);
    fprintf(f, ",\"candidate_decision_logits_file\":\"%s\",\"candidate_logits_file\":\"%s\",\"native_shadow_logits_file\":\"%s\",\"shadow_logits_file\":\"%s\",\"decision_count\":%d,\"capture\":%s,\"x_file\":\"%s\",\"receiver_file\":\"%s\",\"turns\":[",
            r->candidate_decision_file, r->candidate_decision_file, r->native_shadow_file,
            r->native_shadow_file, r->decision_count, r->capture ? "true" : "false",
            r->capture ? r->x_file : "", r->capture ? r->x_file : "");
    for (int t = 0; t < 2; t++) {
        const TurnInfo *turn = &r->turns[t];
        fprintf(f, "%s{\"turn\":%d,\"start\":%d,\"ingested\":%d,\"template_skip\":%d,\"prefill\":%d,\"final\":%d,\"budget\":%d,\"answer_ids\":",
                t ? "," : "", t, turn->start, turn->ingested, turn->skipped, turn->prefill,
                turn->final, turn->budget);
        write_ints(f, turn->answers, turn->answer_count);
        fprintf(f, ",\"stopped\":%s,\"truncated\":%s,\"decision_offset\":%d,\"decision_count\":%d,\"boundaries\":{\"prefill\":{\"logits_file\":\"%s\",\"x_file\":\"%s\",\"write_file\":\"%s\",\"native_write_file\":\"%s\"},\"final\":{\"logits_file\":\"%s\",\"x_file\":\"%s\",\"write_file\":\"%s\",\"native_write_file\":\"%s\"}}}",
                turn->stopped ? "true" : "false", turn->truncated ? "true" : "false",
                turn->decision_offset, turn->decision_count,
                turn->prefill_logits, turn->prefill_x, turn->prefill_write, turn->prefill_native_write,
                turn->final_logits, turn->final_x, turn->final_write, turn->final_native_write);
    }
    fprintf(f, "],\"linear_calls\":%llu,\"linear_macs\":%llu,\"native_linear_calls\":%llu,\"native_linear_macs\":%llu,\"compiled_linear_calls\":%llu,\"compiled_linear_macs\":%llu,\"mlp_coordinate_calls\":%llu,\"compiled_gate_up_calls\":%llu,\"native_gate_up_calls\":%llu,\"native_qkv_calls\":%llu,\"native_attention_calls\":%llu,\"session_gpu_bytes\":%zu,\"factor_gpu_bytes\":%zu,\"stream_bytes\":%zu}",
            (unsigned long long)r->linear_calls, (unsigned long long)r->linear_macs,
            (unsigned long long)r->native_linear_calls, (unsigned long long)r->native_linear_macs,
            (unsigned long long)r->compiled_linear_calls, (unsigned long long)r->compiled_linear_macs,
            (unsigned long long)r->mlp_coordinate_calls, (unsigned long long)r->compiled_gate_up_calls,
            (unsigned long long)r->native_gate_up_calls, (unsigned long long)r->native_qkv_calls,
            (unsigned long long)r->native_attention_calls, r->session_gpu_bytes,
            r->factor_gpu_bytes, r->stream_bytes);
}

static void run_case(PathInfo *r) {
    int *prompt = read_tokens(r->case_id, "prompt", &r->turns[0].ingested);
    int prompt_count = r->turns[0].ingested;
    int *follow = read_tokens(r->case_id, "followup", &r->turns[1].ingested);
    int follow_count = r->turns[1].ingested;
    int max_tokens = prompt_count + follow_count + 2 * r->budget;
    REQUIRE(max_tokens > 0 && max_tokens <= CAPACITY);
    allocate_run_buffers(r, 2 * r->budget + 2, r->capture);
    QwenSession session;
    fresh_session(&session);
    r->session = &session;
    double started = clock_seconds();

    TurnInfo *t0 = &r->turns[0];
    t0->start = 0;
    t0->skipped = 0;
    t0->ingested = prompt_count;
    for (int i = 0; i < prompt_count; i++)
        advance_one(r, prompt[i], i == prompt_count - 1);
    t0->prefill = r->count;
    save_boundary(r, t0, "prefill", 0, r->count);
    t0->decision_offset = r->decision_count;
    for (int j = 0; j < r->budget; j++) {
        record_decision(r, t0);
        int token = r->session->pending;
        REQUIRE(t0->answer_count < MAX_BUDGET);
        t0->answers[t0->answer_count++] = token;
        t0->stopped = token == model_active->eos || token == model_active->eos_extra;
        advance_one(r, token, t0->stopped || j == r->budget - 1);
        if (t0->stopped) break;
    }
    t0->final = r->count;
    t0->truncated = !t0->stopped;
    save_boundary(r, t0, "final", 0, r->count);

    TurnInfo *t1 = &r->turns[1];
    t1->start = r->count;
    t1->skipped = r->ids[r->count - 1] == follow[0];
    t1->ingested = follow_count - t1->skipped;
    REQUIRE(follow_count > 0);
    for (int i = t1->skipped; i < follow_count; i++)
        advance_one(r, follow[i], i == follow_count - 1);
    t1->prefill = r->count;
    save_boundary(r, t1, "prefill", 1, r->count);
    t1->decision_offset = r->decision_count;
    for (int j = 0; j < r->budget; j++) {
        record_decision(r, t1);
        int token = r->session->pending;
        REQUIRE(t1->answer_count < MAX_BUDGET);
        t1->answers[t1->answer_count++] = token;
        t1->stopped = token == model_active->eos || token == model_active->eos_extra;
        advance_one(r, token, t1->stopped || j == r->budget - 1);
        if (t1->stopped) break;
    }
    t1->final = r->count;
    t1->truncated = !t1->stopped;
    save_boundary(r, t1, "final", 1, r->count);

    REQUIRE(r->count <= CAPACITY);
    save_primary_arrays(r);
    write_primary_json(r);
    native_write_diagnostic(r, &r->turns[0], "prefill", r->turns[0].prefill_native_write);
    native_write_diagnostic(r, &r->turns[0], "final", r->turns[0].final_native_write);
    native_write_diagnostic(r, &r->turns[1], "prefill", r->turns[1].prefill_native_write);
    native_write_diagnostic(r, &r->turns[1], "final", r->turns[1].final_native_write);
    native_shadow(r);
    update_shadow_json(r);
    r->seconds = clock_seconds() - started;
    r->session_gpu_bytes = session.allocated_bytes;
    r->factor_gpu_bytes = r->program ? r->program->gpu_bytes : 0;
    fprintf(stderr, "completed %s/%s program=%s mask=%u tokens=%d decisions=%d seconds=%.3f\n",
            r->label, r->case_id, r->program_name, r->mask, r->count, r->decision_count, r->seconds);
    qwen_session_free(&session);
    free(prompt);
    free(follow);
    free_run_buffers(r);
    r->session = NULL;
}

static void native_sham(const char *case_id) {
    int n;
    int *ids = read_tokens(case_id, "prompt", &n);
    REQUIRE(n >= 32);
    QwenSession a, b;
    fresh_session(&a);
    fresh_session(&b);
    float *x = malloc((size_t)V * sizeof(*x));
    float *y = malloc((size_t)V * sizeof(*y));
    REQUIRE(x && y);
    sham_pass = 1;
    for (int i = 0; i < 32; i++) {
        REQUIRE(!qwen_push(&a, ids[i]));
        REQUIRE(!qwen_push(&b, ids[i]));
        finite_read(x, a.logits, V);
        finite_read(y, b.logits, V);
        REQUIRE(a.pending == b.pending && !memcmp(x, y, (size_t)V * sizeof(float)));
    }
    sham_pass = 0;
    char path[256], hash_a[65], hash_b[65];
    cache_digest(&a, hash_a);
    cache_digest(&b, hash_b);
    REQUIRE(!strcmp(hash_a, hash_b));
    REQUIRE(snprintf(path, sizeof(path), "native-sham-%s-last-logits.f32", case_id) < (int)sizeof(path));
    save_file(path, x, (size_t)V * sizeof(float));
    char meta[4096];
    REQUIRE(snprintf(meta, sizeof(meta), "%s/native-sham.json", results_dir) < (int)sizeof(meta));
    FILE *f = fopen(meta, "w");
    REQUIRE(f);
    fprintf(f, "{\"schema\":\"tcm-transition-laws-native-sham-v1\",\"case_id\":\"%s\",\"steps\":32,\"logits_file\":\"%s\",\"cache_sha256\":\"%s\",\"byte_exact\":true}\n", case_id, path, hash_a);
    REQUIRE(!fclose(f));
    free(x);
    free(y);
    free(ids);
    qwen_session_free(&a);
    qwen_session_free(&b);
}

static void run_group(const char *label, const char *program_name, uint32_t mask,
                      int capture, int ncases, char case_ids[MAX_CASES][MAX_CASE_ID],
                      int budgets[MAX_CASES], FactorProgram programs[2]) {
    REQUIRE(safe_component(label) && ncases > 0 && ncases <= MAX_CASES);
    REQUIRE(!strcmp(program_name, "none") || !strcmp(program_name, "p0") || !strcmp(program_name, "p1"));
    if (!strcmp(program_name, "none")) REQUIRE(mask == 0);
    FactorProgram *program = find_program(program_name, programs);
    if (program_name[0] == 'p') REQUIRE(program && program->ready && mask);
    for (int i = 0; i < ncases; i++) {
        REQUIRE(safe_component(case_ids[i]) && budgets[i] > 0 && budgets[i] <= MAX_BUDGET);
        if (i == 0) {
            static int sham_done;
            if (!sham_done) { native_sham(case_ids[i]); sham_done = 1; }
        }
    }
    PathInfo *paths = calloc((size_t)ncases, sizeof(*paths));
    REQUIRE(paths);
    for (int i = 0; i < ncases; i++) {
        PathInfo *r = &paths[i];
        strncpy(r->case_id, case_ids[i], sizeof(r->case_id) - 1);
        strncpy(r->label, label, sizeof(r->label) - 1);
        strncpy(r->program_name, program_name, sizeof(r->program_name) - 1);
        r->mask = mask;
        r->capture = capture != 0;
        r->budget = budgets[i];
        r->program = program;
        run_case(r);
    }
    char path[4096];
    REQUIRE(snprintf(path, sizeof(path), "%s/%s.json", results_dir, label) < (int)sizeof(path));
    FILE *f = fopen(path, "w");
    REQUIRE(f);
    fprintf(f, "{\"schema\":\"tcm-transition-laws-driver-v1\",\"label\":\"%s\",\"program\":\"%s\",\"mask\":%u,\"capture\":%s,\"model_identity\":\"%s\",\"model_digest\":\"%s\",\"layers\":%d,\"hidden\":%d,\"intermediate\":%d,\"vocab\":%d,\"rank\":%d,\"memory_policy\":\"full-fp32-all-layers\",\"native_shadow_conditional\":true,\"native_sham_steps\":32,\"paths\":[",
            label, program_name, mask, capture ? "true" : "false", model_active->identity,
            model_active->digest, LAYERS, D, H, V, RANK);
    for (int i = 0; i < ncases; i++) write_path_json(f, &paths[i], i != 0);
    uint64_t linear_calls = 0, linear_macs = 0, compiled_calls = 0, compiled_macs = 0, coordinate_calls = 0;
    for (int i = 0; i < ncases; i++) {
        linear_calls += paths[i].linear_calls;
        linear_macs += paths[i].linear_macs;
        compiled_calls += paths[i].compiled_linear_calls;
        compiled_macs += paths[i].compiled_linear_macs;
        coordinate_calls += paths[i].mlp_coordinate_calls;
    }
    fprintf(f, "],\"path_count\":%d,\"linear_calls\":%llu,\"linear_macs\":%llu,\"compiled_linear_calls\":%llu,\"compiled_linear_macs\":%llu,\"mlp_coordinate_calls\":%llu,\"native_write_diagnostic_linear_calls\":%llu,\"native_write_diagnostic_linear_macs\":%llu,\"program_gpu_bytes\":%zu,\"model_weight_gpu_bytes\":%zu}\n",
            ncases, (unsigned long long)linear_calls, (unsigned long long)linear_macs,
            (unsigned long long)compiled_calls, (unsigned long long)compiled_macs,
            (unsigned long long)coordinate_calls, (unsigned long long)diagnostic_linear_calls,
            (unsigned long long)diagnostic_linear_macs, program ? program->gpu_bytes : 0,
            model_active->weight_bytes);
    REQUIRE(!fclose(f));
    free(paths);
    printf("DONE %s\n", label);
    fflush(stdout);
}

static void write_execution_totals(const FactorProgram programs[2]) {
    char path[4096];
    REQUIRE(snprintf(path, sizeof(path), "%s/execution-totals.json", results_dir) < (int)sizeof(path));
    FILE *f = fopen(path, "w");
    REQUIRE(f);
    fprintf(f, "{\"schema\":\"tcm-transition-laws-execution-totals-v1\",\"model_identity\":\"%s\",\"model_digest\":\"%s\",\"total_linear_calls\":%llu,\"total_linear_macs\":%llu,\"local_native_write_diagnostic\":{\"calls\":%llu,\"macs\":%llu},\"native_shadow\":{\"calls\":%llu,\"macs\":%llu},\"native_sham\":{\"calls\":%llu,\"macs\":%llu},\"compile\":{",
            model_active->identity, model_active->digest,
            (unsigned long long)total_linear_calls, (unsigned long long)total_linear_macs,
            (unsigned long long)diagnostic_linear_calls, (unsigned long long)diagnostic_linear_macs,
            (unsigned long long)shadow_linear_calls, (unsigned long long)shadow_linear_macs,
            (unsigned long long)sham_linear_calls, (unsigned long long)sham_linear_macs);
    for (int i = 0; i < 2; i++) {
        fprintf(f, "%s\"%s\":{\"calls\":%llu,\"macs\":%llu,\"gpu_bytes\":%zu,\"ready\":%s}",
                i ? "," : "", programs[i].name[0] ? programs[i].name : (i == 0 ? "p0" : "p1"),
                (unsigned long long)programs[i].compile_calls,
                (unsigned long long)programs[i].compile_macs, programs[i].gpu_bytes,
                programs[i].ready ? "true" : "false");
    }
    fprintf(f, "},\"formula_native_linear_calls_per_advance\":197}\n");
    REQUIRE(!fclose(f));
}

int main(int argc, char **argv) {
    REQUIRE(argc == 3); /* RESULTS may be absolute or relative; case/label components are checked separately. */
    results_dir = argv[2];
    if (mkdir(results_dir, 0775) && errno != EEXIST) {
        fprintf(stderr, "transition-laws cannot create RESULTS: %s\n", strerror(errno));
        return 2;
    }
    TcmModel bundle;
    QwenModel model;
    char error[256];
    REQUIRE(!tcm_load(&bundle, argv[1], error, sizeof(error)));
    REQUIRE(!qwen_model_init(&model, &bundle, error, sizeof(error)));
    tcm_unload(&bundle);
    REQUIRE(model.layers == LAYERS && model.hidden == D && model.intermediate == H && model.vocab == V);
    model_active = &model;
    FactorProgram programs[2] = {{0}, {0}};
    printf("READY\n");
    fflush(stdout);
    char line[1024];
    while (fgets(line, sizeof(line), stdin)) {
        if (!strncmp(line, "QUIT", 4)) break;
        if (!strncmp(line, "COMPILE ", 8)) {
            char name[8];
            REQUIRE(sscanf(line + 8, "%7s", name) == 1 && (!strcmp(name, "p0") || !strcmp(name, "p1")));
            FactorProgram *p = find_program(name, programs);
            REQUIRE(p && !p->ready);
            compile_program(p, name);
            printf("DONE COMPILE %s\n", name);
            fflush(stdout);
            continue;
        }
        if (!strncmp(line, "RUN ", 4)) {
            char label[MAX_LABEL], program_name[8];
            unsigned mask;
            int capture, ncases;
            REQUIRE(sscanf(line + 4, "%95s %7s %u %d %d", label, program_name, &mask, &capture, &ncases) == 5);
            REQUIRE(safe_component(label) && ncases > 0 && ncases <= MAX_CASES && mask <= UINT32_MAX);
            char case_ids[MAX_CASES][MAX_CASE_ID];
            int budgets[MAX_CASES];
            for (int i = 0; i < ncases; i++) {
                REQUIRE(fgets(line, sizeof(line), stdin));
                REQUIRE(sscanf(line, "%95s %d", case_ids[i], &budgets[i]) == 2);
            }
            run_group(label, program_name, (uint32_t)mask, capture, ncases, case_ids, budgets, programs);
            continue;
        }
        fprintf(stderr, "transition-laws unknown command: %s", line);
        return 2;
    }
    write_execution_totals(programs);
    free_program(&programs[0]);
    free_program(&programs[1]);
    qwen_model_free(&model);
    return 0;
}
