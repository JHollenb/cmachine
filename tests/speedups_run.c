#define _POSIX_C_SOURCE 200809L
/* Speedup arms for the C/CUDA Qwen executor (research/speedups, docs/SPEEDUPS.md).
 *
 *   speedups-run MODEL PROMPTS OUTDIR [--steps N] [--warmup-steps N] [--smoke-steps N]
 *                [--hash-prompts N] [--phase-file PATH]
 *
 * One model load. PROMPTS lines: "n t0 ... t(n-1)". Writes OUTDIR/runs.jsonl and
 * OUTDIR/logits-<name>.f32 ([prompts][vocab] float32, first = after prefill,
 * final = after the last decode step). Arms:
 *   C0      lean qwen_push prefill + qwen_advance decode (reference, unchanged path)
 *   F1s     fast engine, one-row (sequential) prefill, B=1 decode
 *   F1      fast engine, batched prefill, B=1 decode
 *   G1      F1 with the decode step replayed as a CUDA graph
 *   B<n>/G<n>  batched decode of n sequences (prompts cycled), non-graph / graph
 *   FORCE_*  teacher-forced on C0's ids (final logits on identical contexts)
 *   TF32_*   the batched arms with TF32 tensor-op sgemm
 *   H0/HFULL/HCPU/HASYNC/HGPU  lean decode with no hash / full-cache SHA (as
 *            overhead C3) / incremental chain on CPU / on a worker thread / GPU leaves
 *   BF16_*, INT8_*  reduced-precision weight lanes (FP32 matrices freed first)
 * Smoke passes (4 tokens, prompt 0) gate each phase; exit 3 aborts before timing. */
#include "qwen.h"
#include "qwen_fast.h"
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double mono(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static double wall(void) { struct timespec t; clock_gettime(CLOCK_REALTIME, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void die(const char *what, int code) { fprintf(stderr, "speedups-run: %s (%d)\n", what, code); exit(2); }
static const char *phase_path;
static void phase(const char *fmt, ...) {
    if (!phase_path) return;
    char label[128], tmp[1100]; va_list ap; va_start(ap, fmt); vsnprintf(label, sizeof(label), fmt, ap); va_end(ap);
    snprintf(tmp, sizeof(tmp), "%s.tmp", phase_path); FILE *f = fopen(tmp, "w"); if (!f) return; fputs(label, f); fclose(f); rename(tmp, phase_path);
}

static int NP, *LENS, **PROMPTS, STEPS = 64, V;
static QwenModel M; static QwenSession S; static QwenFast *F; static FILE *OUT; static const char *DIR_;

/* ---------------- output ---------------- */
typedef struct {
    const char *arm, *lane, *prefill_mode; int B, graph, tf32, forced, steps, nseq;
    int prompt[128]; double prefill_ms[128]; int *ids[128]; double *ms; double t_start, t_end; double extra_ms;
    const char *hash_mode; char chain[65]; int verify_ok;
} Run;
static void emit(const char *type, const Run *r) {
    fprintf(OUT, "{\"type\":\"%s\",\"arm\":\"%s\",\"lane\":\"%s\",\"prefill_mode\":\"%s\",\"B\":%d,\"graph\":%d,\"tf32\":%d,\"forced\":%d,\"t_start\":%.6f,\"t_end\":%.6f,\"prompts\":[",
            type, r->arm, r->lane, r->prefill_mode, r->B, r->graph, r->tf32, r->forced, r->t_start, r->t_end);
    for (int b = 0; b < r->nseq; b++) fprintf(OUT, "%s%d", b ? "," : "", r->prompt[b]);
    fprintf(OUT, "],\"prompt_tokens\":["); for (int b = 0; b < r->nseq; b++) fprintf(OUT, "%s%d", b ? "," : "", LENS[r->prompt[b]]);
    fprintf(OUT, "],\"prefill_ms\":["); for (int b = 0; b < r->nseq; b++) fprintf(OUT, "%s%.4f", b ? "," : "", r->prefill_ms[b]);
    fprintf(OUT, "],\"ms\":["); for (int i = 0; i < r->steps; i++) fprintf(OUT, "%s%.4f", i ? "," : "", r->ms[i]);
    fprintf(OUT, "],\"ids\":[");
    for (int b = 0; b < r->nseq; b++) { fprintf(OUT, "%s[", b ? "," : ""); for (int i = 0; i <= r->steps; i++) fprintf(OUT, "%s%d", i ? "," : "", r->ids[b][i]); fprintf(OUT, "]"); }
    fprintf(OUT, "],\"extra_ms\":%.4f", r->extra_ms);
    if (r->hash_mode) fprintf(OUT, ",\"hash_mode\":\"%s\",\"chain\":\"%s\",\"verify_ok\":%d", r->hash_mode, r->chain, r->verify_ok);
    fprintf(OUT, "}\n"); fflush(OUT);
}
static Run *run_new(const char *arm, const char *lane, int B, int nseq) {
    Run *r = calloc(1, sizeof(Run)); r->arm = arm; r->lane = lane; r->B = B; r->nseq = nseq; r->steps = STEPS; r->prefill_mode = "none";
    r->ms = calloc((size_t)STEPS + 1, sizeof(double)); for (int b = 0; b < nseq; b++) r->ids[b] = calloc((size_t)STEPS + 1, sizeof(int));
    return r;
}
static void run_free(Run *r) { for (int b = 0; b < r->nseq; b++) free(r->ids[b]); free(r->ms); free(r); }
/* Logit banks: [NP][V]; filled[p] marks rows already written. */
typedef struct { const char *name; float *data; int *filled; } Bank;
#define MAXBANK 48
static Bank BANKS[MAXBANK]; static int NBANK;
static Bank *bank(const char *name) {
    for (int i = 0; i < NBANK; i++) if (!strcmp(BANKS[i].name, name)) return &BANKS[i];
    if (NBANK == MAXBANK) die("too many logit banks", 0);
    Bank *b = &BANKS[NBANK++]; b->name = name; b->data = calloc((size_t)NP * V, 4); b->filled = calloc(NP, sizeof(int)); if (!b->data || !b->filled) die("bank alloc", 0); return b;
}
static float *bank_row(const char *name, int p) { Bank *b = bank(name); if (b->filled[p]) return NULL; b->filled[p] = 1; return b->data + (size_t)p * V; }
static void banks_save(void) {
    for (int i = 0; i < NBANK; i++) {
        char path[4096]; snprintf(path, sizeof(path), "%s/logits-%s.f32", DIR_, BANKS[i].name); FILE *f = fopen(path, "wb"); if (!f) die("bank save", i);
        fwrite(BANKS[i].data, 4, (size_t)NP * V, f); fclose(f);
        fprintf(OUT, "{\"type\":\"bank\",\"name\":\"%s\",\"filled\":[", BANKS[i].name); for (int p = 0; p < NP; p++) fprintf(OUT, "%s%d", p ? "," : "", BANKS[i].filled[p]); fprintf(OUT, "]}\n");
    }
    fflush(OUT);
}

/* ---------------- lean arm (C0) ---------------- */
static void lean_prefill(int p, double *ms) {
    qwen_reset(&S); gpu_sync(); double a = mono();
    for (int i = 0; i < LENS[p]; i++) if (qwen_push(&S, PROMPTS[p][i])) die("lean prefill", i);
    gpu_sync(); *ms = (mono() - a) * 1e3;
}
static Run *lean_run(int p, int steps, const char *first_bank, const char *final_bank) {
    Run *r = run_new("C0", "fp32", 1, 1); r->steps = steps; r->prefill_mode = "lean_sequential"; r->prompt[0] = p;
    lean_prefill(p, &r->prefill_ms[0]);
    float *row = first_bank ? bank_row(first_bank, p) : NULL; if (row) gpu_read(row, S.logits, (size_t)V * 4);
    r->t_start = wall();
    for (int i = 0; i < steps; i++) {
        r->ids[0][i] = S.pending; gpu_sync(); double a = mono();
        if (qwen_advance(&S)) die("lean advance", i);
        gpu_sync(); r->ms[i] = (mono() - a) * 1e3;
    }
    r->t_end = wall(); r->ids[0][steps] = S.pending;
    row = final_bank ? bank_row(final_bank, p) : NULL; if (row) gpu_read(row, S.logits, (size_t)V * 4);
    return r;
}

/* ---------------- fast arms ---------------- */
static int **LEAN_IDS; /* C0 ids per prompt, for teacher forcing */
static Run *fast_run(const char *arm, const char *lane, int B, int base, int graph, int batched_prefill, int forced, int steps, const char *first_bank, const char *final_bank) {
    Run *r = run_new(arm, lane, B, B); r->steps = steps; r->graph = graph; r->forced = forced; r->prefill_mode = batched_prefill ? "batched" : "fast_sequential";
    qf_reset(F);
    for (int b = 0; b < B; b++) {
        int p = (base + b) % NP; r->prompt[b] = p;
        qf_sync(F); double a = mono();
        if (qf_prefill(F, b, PROMPTS[p], LENS[p], batched_prefill)) die("fast prefill", b);
        qf_sync(F); r->prefill_ms[b] = (mono() - a) * 1e3;
        float *row = first_bank ? bank_row(first_bank, p) : NULL; if (row) qf_read_logits(F, 0, row);
    }
    if (graph) { double a = mono(); if (qf_graph_capture(F, B)) die("graph capture", B); qf_sync(F); r->extra_ms = (mono() - a) * 1e3; }
    int *forced_ids = malloc((size_t)B * sizeof(int));
    r->t_start = wall();
    for (int i = 0; i < steps; i++) {
        if (forced) for (int b = 0; b < B; b++) forced_ids[b] = LEAN_IDS[r->prompt[b]][i];
        qf_sync(F); double a = mono();
        if (graph ? qf_graph_step(F) : qf_decode(F, B, forced ? forced_ids : NULL)) die("fast decode", i);
        qf_sync(F); r->ms[i] = (mono() - a) * 1e3;
    }
    r->t_end = wall(); free(forced_ids);
    for (int b = 0; b < B; b++) {
        if (qf_read_ids(F, b, r->ids[b], steps + 1)) die("read ids", b);
        float *row = final_bank ? bank_row(final_bank, r->prompt[b]) : NULL; if (row) qf_read_logits(F, b, row);
    }
    return r;
}

/* ---------------- hashing ---------------- */
#define LEAF 1024
static void state_hashes(const QwenSession *s, char cache[65], char logits[65]) { /* copy of qwen_serve.c S / overhead C3 */
    const QwenModel *m = s->model; size_t row = (size_t)s->position * m->head, n = (size_t)2 * m->layers * m->kvheads * row, offset = 0;
    float *host = malloc((n ? n : 1) * 4); if (!host) exit(2);
    for (int kind = 0; kind < 2; kind++) for (int l = 0; l < m->layers; l++) for (int h = 0; h < m->kvheads; h++) {
        size_t address = ((size_t)l * m->kvheads + h) * s->capacity * m->head;
        if (row) gpu_read(host + offset, (kind ? s->values : s->keys) + address, row * 4); offset += row;
    }
    unsigned char digest[32]; tcm_sha256(host, n * 4, digest); tcm_hex(digest, 32, cache); free(host);
    host = malloc((size_t)m->vocab * 4); if (!host) exit(2);
    gpu_read(host, s->logits, (size_t)m->vocab * 4); tcm_sha256(host, (size_t)m->vocab * 4, digest); tcm_hex(digest, 32, logits); free(host);
}
/* Incremental digest inc-merkle1k-v1: leaf_i = SHA256(bytes[1024 i, 1024 i + 1024)), root = SHA256(leaf_0 || ...),
 * chain_t = SHA256(chain_{t-1} || kv_root(row written at t) || logits_root(t)); chain_{-1} = 32 zero bytes, and
 * prompt rows enter with a zero logits root. */
static void merkle_cpu(const void *data, size_t n, unsigned char root[32]) {
    size_t leaves = (n + LEAF - 1) / LEAF; unsigned char *d = malloc(leaves * 32); if (!d) exit(2);
    for (size_t i = 0; i < leaves; i++) tcm_sha256((const unsigned char *)data + i * LEAF, n - i * LEAF < LEAF ? n - i * LEAF : LEAF, d + 32 * i);
    tcm_sha256(d, leaves * 32, root); free(d);
}
static void chain_step(unsigned char chain[32], const unsigned char kv[32], const unsigned char lg[32]) {
    unsigned char buf[96]; memcpy(buf, chain, 32); memcpy(buf + 32, kv, 32); memcpy(buf + 64, lg, 32); tcm_sha256(buf, 96, chain);
}
typedef struct {
    size_t rbytes, lbytes, rleaves, lleaves; float *dev_rows; unsigned char *dev_dig, *host_dig; float *rows[2], *lg[2];
    unsigned char chain[32], (*kv)[32], (*lgr)[32];
    pthread_t th; pthread_mutex_t mu; pthread_cond_t cv; int full[2], step[2], stop;
} Hasher;
static void read_row(Hasher *h, int position, float *host) { qf_gather_rows(h->dev_rows, S.keys, S.values, M.layers, M.kvheads, S.capacity, M.head, position); gpu_read(host, h->dev_rows, h->rbytes); }
static void hasher_init(Hasher *h) {
    memset(h, 0, sizeof(*h)); h->rbytes = (size_t)2 * M.layers * M.kvheads * M.head * 4; h->lbytes = (size_t)V * 4;
    h->rleaves = (h->rbytes + LEAF - 1) / LEAF; h->lleaves = (h->lbytes + LEAF - 1) / LEAF;
    h->dev_rows = gpu_alloc(h->rbytes); h->dev_dig = gpu_alloc((h->rleaves + h->lleaves) * 32); h->host_dig = malloc((h->rleaves + h->lleaves) * 32);
    for (int i = 0; i < 2; i++) { h->rows[i] = malloc(h->rbytes); h->lg[i] = malloc(h->lbytes); }
    h->kv = calloc((size_t)STEPS + 8, 32); h->lgr = calloc((size_t)STEPS + 8, 32);
    pthread_mutex_init(&h->mu, NULL); pthread_cond_init(&h->cv, NULL);
}
static void hasher_free(Hasher *h) {
    gpu_free(h->dev_rows); gpu_free(h->dev_dig); free(h->host_dig); for (int i = 0; i < 2; i++) { free(h->rows[i]); free(h->lg[i]); }
    free(h->kv); free(h->lgr); pthread_mutex_destroy(&h->mu); pthread_cond_destroy(&h->cv);
}
static void seed_chain(Hasher *h, int upto) {
    static const unsigned char zero[32]; memset(h->chain, 0, 32);
    for (int p = 0; p < upto; p++) { unsigned char kv[32]; read_row(h, p, h->rows[0]); merkle_cpu(h->rows[0], h->rbytes, kv); chain_step(h->chain, kv, zero); }
}
static void hash_cpu_slot(Hasher *h, int slot, int t) {
    merkle_cpu(h->rows[slot], h->rbytes, h->kv[t]); merkle_cpu(h->lg[slot], h->lbytes, h->lgr[t]); chain_step(h->chain, h->kv[t], h->lgr[t]);
}
static void *hasher_thread(void *arg) {
    Hasher *h = arg; int cur = 0;
    for (;;) {
        pthread_mutex_lock(&h->mu); while (!h->full[cur] && !h->stop) pthread_cond_wait(&h->cv, &h->mu);
        if (!h->full[cur]) { pthread_mutex_unlock(&h->mu); break; }
        int t = h->step[cur]; pthread_mutex_unlock(&h->mu);
        hash_cpu_slot(h, cur, t);
        pthread_mutex_lock(&h->mu); h->full[cur] = 0; pthread_cond_broadcast(&h->cv); pthread_mutex_unlock(&h->mu); cur ^= 1;
    }
    return NULL;
}
enum { H_NONE, H_FULL, H_CPU, H_ASYNC, H_GPU };
static const char *HASH_NAMES[] = {"H0", "HFULL", "HCPU", "HASYNC", "HGPU"};
static void hash_token(Hasher *h, int mode, int t, char cache[65], char lsha[65]) {
    if (mode == H_FULL) { state_hashes(&S, cache, lsha); return; }
    int position = S.position - 1;
    if (mode == H_CPU) { read_row(h, position, h->rows[0]); gpu_read(h->lg[0], S.logits, h->lbytes); hash_cpu_slot(h, 0, t); return; }
    if (mode == H_ASYNC) {
        int slot = t & 1;
        pthread_mutex_lock(&h->mu); while (h->full[slot]) pthread_cond_wait(&h->cv, &h->mu); pthread_mutex_unlock(&h->mu);
        read_row(h, position, h->rows[slot]); gpu_read(h->lg[slot], S.logits, h->lbytes);
        pthread_mutex_lock(&h->mu); h->full[slot] = 1; h->step[slot] = t; pthread_cond_broadcast(&h->cv); pthread_mutex_unlock(&h->mu);
        return;
    }
    if (mode == H_GPU) {
        qf_gather_rows(h->dev_rows, S.keys, S.values, M.layers, M.kvheads, S.capacity, M.head, position);
        qf_sha_leaves(h->dev_rows, h->rbytes, LEAF, h->dev_dig); qf_sha_leaves(S.logits, h->lbytes, LEAF, h->dev_dig + h->rleaves * 32);
        gpu_read(h->host_dig, h->dev_dig, (h->rleaves + h->lleaves) * 32);
        tcm_sha256(h->host_dig, h->rleaves * 32, h->kv[t]); tcm_sha256(h->host_dig + h->rleaves * 32, h->lleaves * 32, h->lgr[t]); chain_step(h->chain, h->kv[t], h->lgr[t]);
    }
}
static Run *hash_run(Hasher *h, int p, int mode, int steps) {
    Run *r = run_new(HASH_NAMES[mode], "fp32", 1, 1); r->steps = steps; r->prefill_mode = "lean_sequential"; r->prompt[0] = p; r->hash_mode = HASH_NAMES[mode]; r->chain[0] = 0; r->verify_ok = -1;
    lean_prefill(p, &r->prefill_ms[0]);
    int p0 = S.position; char cache[65], lsha[65];
    if (mode >= H_CPU) seed_chain(h, p0);
    if (mode == H_ASYNC) { h->stop = 0; h->full[0] = h->full[1] = 0; if (pthread_create(&h->th, NULL, hasher_thread, h)) die("pthread", 0); }
    r->t_start = wall();
    for (int i = 0; i < steps; i++) {
        r->ids[0][i] = S.pending; gpu_sync(); double a = mono();
        if (qwen_advance(&S)) die("hash advance", i);
        if (mode != H_NONE) hash_token(h, mode, i, cache, lsha);
        gpu_sync(); r->ms[i] = (mono() - a) * 1e3;
    }
    r->t_end = wall(); r->ids[0][steps] = S.pending;
    if (mode == H_ASYNC) { double a = mono(); pthread_mutex_lock(&h->mu); h->stop = 1; pthread_cond_broadcast(&h->cv); pthread_mutex_unlock(&h->mu); pthread_join(h->th, NULL); r->extra_ms = (mono() - a) * 1e3; }
    if (mode == H_FULL) snprintf(r->chain, sizeof(r->chain), "%s", cache);
    if (mode >= H_CPU) {
        tcm_hex(h->chain, 32, r->chain);
        /* Verify: re-derive each kv root from the final cache and replay the chain with the recorded logits roots. */
        unsigned char keep[32]; memcpy(keep, h->chain, 32); seed_chain(h, p0); int ok = 1;
        for (int i = 0; i < steps; i++) { unsigned char kv[32]; read_row(h, p0 + i, h->rows[0]); merkle_cpu(h->rows[0], h->rbytes, kv); if (memcmp(kv, h->kv[i], 32)) ok = 0; chain_step(h->chain, kv, h->lgr[i]); }
        r->verify_ok = ok && !memcmp(keep, h->chain, 32);
    }
    return r;
}
/* GPU leaf digests must equal CPU SHA-256 of the same bytes. */
static int gpu_sha_selftest(Hasher *h) {
    qf_sha_leaves(S.logits, h->lbytes, LEAF, h->dev_dig); gpu_read(h->host_dig, h->dev_dig, h->lleaves * 32); gpu_read(h->lg[0], S.logits, h->lbytes);
    for (size_t i = 0; i < h->lleaves; i++) { unsigned char d[32]; size_t n = h->lbytes - i * LEAF < LEAF ? h->lbytes - i * LEAF : LEAF; tcm_sha256((unsigned char *)h->lg[0] + i * LEAF, n, d); if (memcmp(d, h->host_dig + 32 * i, 32)) return 0; }
    return 1;
}

/* ---------------- helpers ---------------- */
static void free_matrix(void *ctx, float *p) {
    QwenModel *m = ctx; for (unsigned i = 0; i < m->allocation_count; i++) if (m->allocations[i] == p) { gpu_free(p); m->allocations[i] = NULL; return; }
    die("free_matrix: unknown pointer", 0);
}
static int **read_prompts(const char *path, int *count, int **lengths, int *longest) {
    FILE *f = fopen(path, "r"); if (!f) die("cannot open prompts", 0);
    int cap = 8, n = 0; int **rows = malloc(cap * sizeof(int *)); int *len = malloc(cap * sizeof(int)); *longest = 0; int k;
    while (fscanf(f, "%d", &k) == 1) {
        if (n == cap) { cap *= 2; rows = realloc(rows, cap * sizeof(int *)); len = realloc(len, cap * sizeof(int)); }
        rows[n] = malloc((size_t)k * sizeof(int)); for (int i = 0; i < k; i++) if (fscanf(f, "%d", &rows[n][i]) != 1) die("short prompt line", n);
        len[n] = k; if (k > *longest) *longest = k; n++;
    }
    fclose(f); *count = n; *lengths = len; return rows;
}
static double max_abs_diff(const float *a, const float *b, int n) { double m = 0; for (int i = 0; i < n; i++) { double d = fabs((double)a[i] - b[i]); if (!(d <= m)) m = d; } return m; }
static void smoke_result(const char *name, int ok, const char *why) { fprintf(OUT, "{\"type\":\"smoke_result\",\"phase\":\"%s\",\"ok\":%s,\"why\":\"%s\"}\n", name, ok ? "true" : "false", why); fflush(OUT); fprintf(stderr, "[smoke %s] ok=%d %s\n", name, ok, why); }
static void emit_free(const char *type, Run *r) { emit(type, r); run_free(r); }

/* Smoke for one lane of the fast engine: B=1 graph == non-graph bitwise, B=8 graph == non-graph bitwise,
 * every first-step logit row within tol of C0's (bug detector, not the exactness gate), all ids finite/legal. */
static int smoke_fast(const char *lane, int smoke_steps, const float *c0_first, const int *c0_ids, double tol, char *why, size_t whyn) {
    int saved = STEPS; STEPS = smoke_steps; int ok = 1;
    float *first = malloc((size_t)V * 4), *fa = malloc((size_t)V * 4), *fb = malloc((size_t)V * 4);
    Run *a = fast_run("smoke_F1", lane, 1, 0, 0, 1, 0, smoke_steps, NULL, NULL); qf_read_logits(F, 0, fa);
    Run *b = fast_run("smoke_G1", lane, 1, 0, 1, 1, 0, smoke_steps, NULL, NULL); qf_read_logits(F, 0, fb);
    if (memcmp(a->ids[0], b->ids[0], (size_t)(smoke_steps + 1) * 4) || memcmp(fa, fb, (size_t)V * 4)) { ok = 0; snprintf(why, whyn, "%s: graph B=1 not bit-identical to non-graph", lane); }
    if (ok && strcmp(lane, "fp32") == 0 && memcmp(a->ids[0], c0_ids, (size_t)(smoke_steps + 1) * 4)) snprintf(why, whyn, "%s: F1 ids differ from C0 over smoke tokens (warning only)", lane);
    /* first-step logits after batched prefill */
    qf_reset(F); if (qf_prefill(F, 0, PROMPTS[0], LENS[0], 1)) die("smoke prefill", 0); qf_read_logits(F, 0, first);
    double d = max_abs_diff(first, c0_first, V); if (!(d <= tol)) { ok = 0; snprintf(why, whyn, "%s: first-step max|dlogit| %.4g > %.3g vs C0", lane, d, tol); }
    Run *c = fast_run("smoke_B8", lane, 8, 0, 0, 1, 0, smoke_steps, NULL, NULL);
    Run *g = fast_run("smoke_G8", lane, 8, 0, 1, 1, 0, smoke_steps, NULL, NULL);
    for (int s = 0; s < 8; s++) if (memcmp(c->ids[s], g->ids[s], (size_t)(smoke_steps + 1) * 4)) { ok = 0; snprintf(why, whyn, "%s: graph B=8 ids differ from non-graph", lane); }
    emit_free("smoke", a); emit_free("smoke", b); emit_free("smoke", c); emit_free("smoke", g);
    free(first); free(fa); free(fb); STEPS = saved; return ok;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: speedups-run MODEL PROMPTS OUTDIR [--steps N] [--warmup-steps N] [--smoke-steps N] [--hash-prompts N] [--phase-file P]\n"); return 2; }
    int warmup = 8, smoke_steps = 4, hash_prompts = 8;
    for (int i = 4; i < argc; i += 2) {
        if (i + 1 >= argc) return 2;
        if (!strcmp(argv[i], "--steps")) STEPS = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--warmup-steps")) warmup = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--smoke-steps")) smoke_steps = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--hash-prompts")) hash_prompts = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--phase-file")) phase_path = argv[i + 1];
        else return 2;
    }
    DIR_ = argv[3]; double started = mono();
    int longest; PROMPTS = read_prompts(argv[2], &NP, &LENS, &longest); if (NP < 1) die("no prompts", 0);
    if (hash_prompts > NP) hash_prompts = NP;
    int capacity = longest + STEPS + 8;
    char path[4096]; snprintf(path, sizeof(path), "%s/runs.jsonl", DIR_); OUT = fopen(path, "w"); if (!OUT) die("output", 0);
    phase("c_load");
    TcmModel bundle; char error[512];
    if (tcm_load(&bundle, argv[1], error, sizeof(error))) { fprintf(stderr, "%s\n", error); return 2; }
    if (qwen_model_init(&M, &bundle, error, sizeof(error))) { fprintf(stderr, "%s\n", error); return 2; }
    tcm_unload(&bundle); gpu_sync(); double load_s = mono() - started; V = M.vocab;
    if (qwen_session_init(&S, &M, capacity)) die("session", 0);
    QfWeights w; memset(&w, 0, sizeof(w));
    w.hidden = M.hidden; w.layers = M.layers; w.qheads = M.qheads; w.kvheads = M.kvheads; w.head = M.head; w.intermediate = M.intermediate; w.vocab = M.vocab; w.epsilon = M.epsilon; w.theta = M.theta;
    w.embedding = M.embedding; w.norm = M.norm;
    for (int l = 0; l < M.layers; l++) {
        const QwenLayer *L = &M.layer[l];
        w.input_norm[l] = L->input_norm; w.post_norm[l] = L->post_norm; w.q[l] = L->q; w.k[l] = L->k; w.v[l] = L->v; w.qb[l] = L->qb; w.kb[l] = L->kb; w.vb[l] = L->vb;
        w.o[l] = L->o; w.gate[l] = L->gate; w.up[l] = L->up; w.down[l] = L->down;
    }
    int BMAX = 128; F = qf_open(&w, BMAX, capacity, longest > BMAX ? longest : BMAX, STEPS + 8); if (!F) die("qf_open", 0);
    fprintf(OUT, "{\"type\":\"header\",\"model_identity\":\"%s\",\"model_digest\":\"%s\",\"vocab\":%d,\"layers\":%d,\"capacity\":%d,\"steps\":%d,\"prompts\":%d,\"hash_prompts\":%d,\"load_s\":%.3f,"
                 "\"gpu_weight_bytes\":%zu,\"lean_session_bytes\":%zu,\"fast_device_bytes\":%zu,\"lane_weight_bytes\":{\"fp32\":%zu,\"bf16\":%zu,\"int8\":%zu},\"kv_bytes_per_token\":%zu,\"leaf_bytes\":%d}\n",
            M.identity, M.digest, V, M.layers, capacity, STEPS, NP, hash_prompts, load_s, M.weight_bytes, S.allocated_bytes, qf_device_bytes(F),
            qf_lane_weight_bytes(F, QF_FP32), qf_lane_weight_bytes(F, QF_BF16), qf_lane_weight_bytes(F, QF_INT8), (size_t)2 * M.layers * M.kvheads * M.head * 4, LEAF);
    fflush(OUT);
    char why[256];
    Hasher H; hasher_init(&H);

    /* ================= smoke, FP32 ================= */
    phase("smoke_fp32");
    float *c0_first = malloc((size_t)V * 4); int *c0_ids = calloc((size_t)smoke_steps + 1, sizeof(int));
    {
        Run *r = lean_run(0, smoke_steps, NULL, NULL); memcpy(c0_ids, r->ids[0], (size_t)(smoke_steps + 1) * 4);
        qwen_reset(&S); for (int i = 0; i < LENS[0]; i++) qwen_push(&S, PROMPTS[0][i]); gpu_read(c0_first, S.logits, (size_t)V * 4);
        emit_free("smoke", r);
        why[0] = 0; int ok = smoke_fast("fp32", smoke_steps, c0_first, c0_ids, 0.05, why, sizeof(why));
        /* hashing: GPU leaves == CPU, all incremental modes agree and verify */
        int saved = STEPS; STEPS = smoke_steps;
        if (!gpu_sha_selftest(&H)) { ok = 0; snprintf(why, sizeof(why), "GPU SHA-256 leaves differ from CPU"); }
        Run *hc = hash_run(&H, 0, H_CPU, smoke_steps), *ha = hash_run(&H, 0, H_ASYNC, smoke_steps), *hg = hash_run(&H, 0, H_GPU, smoke_steps), *hf = hash_run(&H, 0, H_FULL, smoke_steps);
        if (strcmp(hc->chain, ha->chain) || strcmp(hc->chain, hg->chain) || hc->verify_ok != 1 || ha->verify_ok != 1 || hg->verify_ok != 1 || !hf->chain[0]) { ok = 0; snprintf(why, sizeof(why), "incremental chains disagree or failed verification"); }
        emit_free("smoke", hc); emit_free("smoke", ha); emit_free("smoke", hg); emit_free("smoke", hf);
        STEPS = saved;
        /* TF32 must run and stay near FP32 (bug detector only) */
        qf_set_tf32(F, 1); float *t = malloc((size_t)V * 4); qf_reset(F); qf_prefill(F, 0, PROMPTS[0], LENS[0], 1); qf_read_logits(F, 0, t); qf_set_tf32(F, 0);
        double d = max_abs_diff(t, c0_first, V); if (!(d <= 2.0)) { ok = 0; snprintf(why, sizeof(why), "tf32 first-step max|dlogit| %.4g", d); }
        free(t);
        smoke_result("fp32", ok, why);
        if (!ok) { banks_save(); fclose(OUT); return 3; }
    }

    /* ================= warmup ================= */
    phase("warmup");
    emit_free("warmup", lean_run(0, warmup, NULL, NULL));
    emit_free("warmup", fast_run("F1", "fp32", 1, 0, 0, 1, 0, warmup, NULL, NULL));
    emit_free("warmup", fast_run("G1", "fp32", 1, 0, 1, 1, 0, warmup, NULL, NULL));
    emit_free("warmup", fast_run("B64", "fp32", 64, 0, 0, 1, 0, warmup, NULL, NULL));

    /* ================= FP32 B=1 arms, rotated per prompt ================= */
    LEAN_IDS = calloc(NP, sizeof(int *));
    for (int p = 0; p < NP; p++) {
        for (int k = 0; k < 4; k++) {
            int a = (k + p) % 4; Run *r;
            if (a == 0) { phase("C0"); r = lean_run(p, STEPS, "first_C0", "final_C0"); LEAN_IDS[p] = malloc((size_t)(STEPS + 1) * 4); memcpy(LEAN_IDS[p], r->ids[0], (size_t)(STEPS + 1) * 4); }
            else if (a == 1) { phase("F1s"); r = fast_run("F1s", "fp32", 1, p, 0, 0, 0, STEPS, "first_F1s", "final_F1s"); }
            else if (a == 2) { phase("F1"); r = fast_run("F1", "fp32", 1, p, 0, 1, 0, STEPS, "first_F1", "final_F1"); }
            else { phase("G1"); r = fast_run("G1", "fp32", 1, p, 1, 1, 0, STEPS, NULL, "final_G1"); }
            emit_free("run", r);
        }
    }
    /* ================= FP32 batched decode ================= */
    static const int BS[] = {8, 16, 32, 64, 128};
    for (int bi = 0; bi < 5; bi++) {
        int B = BS[bi], reps = B < NP ? (NP + B - 1) / B : 2;
        char *an = malloc(16), *gn = malloc(16), *fan = malloc(32), *fgn = malloc(32);
        snprintf(an, 16, "B%d", B); snprintf(gn, 16, "G%d", B); snprintf(fan, 32, "final_B%d", B); snprintf(fgn, 32, "final_G%d", B);
        for (int rep = 0; rep < reps; rep++) {
            int base = B < NP ? rep * B : 0;
            for (int k = 0; k < 2; k++) {
                int graph = (k + rep) % 2; phase("%s", graph ? gn : an);
                emit_free("run", fast_run(graph ? gn : an, "fp32", B, base, graph, 1, 0, STEPS, NULL, graph ? fgn : fan));
            }
        }
    }
    phase("FORCE_fp32");
    { int B = NP < 64 ? NP : 64; emit_free("run", fast_run("FORCE_B16", "fp32", B, 0, 0, 1, 1, STEPS, NULL, "final_FORCE_fp32")); }
    emit_free("run", fast_run("FORCE_B1", "fp32", 1, 0, 0, 1, 1, STEPS, NULL, NULL));
    /* ================= TF32 ================= */
    qf_set_tf32(F, 1);
    phase("TF32");
    { int B = NP < 64 ? NP : 64;
      emit_free("run", fast_run("TF32_B16", "fp32", B, 0, 0, 1, 0, STEPS, "first_TF32", "final_TF32_B16"));
      emit_free("run", fast_run("TF32_FORCE_B16", "fp32", B, 0, 0, 1, 1, STEPS, NULL, "final_FORCE_tf32"));
      for (int rep = 0; rep < 2; rep++) for (int k = 0; k < 2; k++) {
          int graph = (k + rep) % 2;
          emit_free("run", fast_run(graph ? "TF32_G64" : "TF32_B64", "fp32", 64, 0, graph, 1, 0, STEPS, NULL, graph ? "final_TF32_G64" : "final_TF32_B64"));
      }
      emit_free("run", fast_run("TF32_B32", "fp32", 32, 0, 0, 1, 0, STEPS, NULL, NULL));
      emit_free("run", fast_run("TF32_B128", "fp32", 128, 0, 0, 1, 0, STEPS, NULL, NULL));
    }
    qf_set_tf32(F, 0);
    /* ================= hashing (lean path), rotated ================= */
    for (int p = 0; p < hash_prompts; p++) for (int k = 0; k < 5; k++) {
        int mode = (k + p) % 5; phase("%s", HASH_NAMES[mode]); emit_free("run", hash_run(&H, p, mode, STEPS));
    }
    hasher_free(&H);

    /* ================= reduced precision ================= */
    phase("convert");
    double a = mono(); qf_build_low_precision(F, free_matrix, &M); gpu_sync(); double convert_s = mono() - a;
    unsigned long long inexact, total; qf_bf16_exactness(F, &inexact, &total);
    fprintf(OUT, "{\"type\":\"convert\",\"seconds\":%.3f,\"bf16_inexact_elements\":%llu,\"bf16_total_elements\":%llu,\"live_gpu_bytes\":%zu,\"peak_gpu_bytes\":%zu}\n", convert_s, inexact, total, gpu_live_bytes(), gpu_peak_bytes()); fflush(OUT);
    static const char *LANES[] = {"bf16", "int8"}; static const int LANE_ID[] = {QF_BF16, QF_INT8};
    int lp_ok = 1;
    for (int li = 0; li < 2; li++) {
        const char *lane = LANES[li]; if (qf_set_lane(F, LANE_ID[li])) die("set lane", li);
        phase("smoke_%s", lane); why[0] = 0;
        int ok = smoke_fast(lane, smoke_steps, c0_first, c0_ids, 5.0, why, sizeof(why));
        char name[32]; snprintf(name, sizeof(name), "%s", lane); smoke_result(name, ok, why);
        if (!ok) { lp_ok = 0; continue; }
        char *n1 = malloc(32), *g1 = malloc(32), *b8 = malloc(32), *g8 = malloc(32), *fr = malloc(32), *ff = malloc(32), *fi = malloc(32), *fg = malloc(32), *bF = malloc(32), *b64 = malloc(32);
        snprintf(n1, 32, "%s_F1", lane); snprintf(g1, 32, "%s_G1", lane); snprintf(b8, 32, "%s_B8", lane); snprintf(g8, 32, "%s_G8", lane); snprintf(fr, 32, "%s_FORCE_B16", lane);
        snprintf(ff, 32, "final_FORCE_%s", lane); snprintf(fi, 32, "first_%s", lane); snprintf(fg, 32, "final_%s_F1", lane); snprintf(bF, 32, "final_%s_G1", lane); snprintf(b64, 32, "%s_B64", lane);
        emit_free("warmup", fast_run(g1, lane, 1, 0, 1, 1, 0, warmup, NULL, NULL));
        for (int p = 0; p < NP; p++) for (int k = 0; k < 2; k++) {
            int graph = (k + p) % 2; phase("%s", graph ? g1 : n1);
            emit_free("run", fast_run(graph ? g1 : n1, lane, 1, p, graph, 1, 0, STEPS, graph ? NULL : fi, graph ? bF : fg));
        }
        for (int rep = 0; rep < (NP + 7) / 8; rep++) for (int k = 0; k < 2; k++) {
            int graph = (k + rep) % 2; phase("%s", graph ? g8 : b8);
            emit_free("run", fast_run(graph ? g8 : b8, lane, 8, rep * 8, graph, 1, 0, STEPS, NULL, NULL));
        }
        emit_free("run", fast_run(b64, lane, 64, 0, 0, 1, 0, STEPS, NULL, NULL));
        phase("%s", fr); { int B = NP < 64 ? NP : 64; emit_free("run", fast_run(fr, lane, B, 0, 0, 1, 1, STEPS, NULL, ff)); }
    }
    phase("c_done");
    banks_save();
    fprintf(OUT, "{\"type\":\"footer\",\"lp_ok\":%s,\"wall_s\":%.3f,\"peak_gpu_bytes\":%zu,\"live_gpu_bytes\":%zu}\n", lp_ok ? "true" : "false", mono() - started, gpu_peak_bytes(), gpu_live_bytes());
    fclose(OUT); qf_close(F); qwen_session_free(&S); qwen_model_free(&M);
    return 0;
}
