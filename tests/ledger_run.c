#define _POSIX_C_SOURCE 200809L
/* Per-token ledger arms and gates (research/ledger, docs/LEDGER.md).
 *
 *   ledger-run MODEL PROMPTS OUTDIR [--steps N] [--warmup-steps N] [--smoke-steps N]
 *              [--g3-prompts N] [--g3-tokens N] [--batches 8,64] [--bf16 0|1] [--phase-file PATH]
 *
 * One model load. PROMPTS lines: "n t0 ... t(n-1)". Ledger mode per arm: off, on
 * (rows read inside the timed region: per token for lean, per pass for the fast
 * engine), onnr (fast graph arms: computed on the device, read once after the
 * timed loop). Arms:
 *   C0       lean qwen_push prefill + qwen_advance decode, off/on
 *   F1       fast FP32, batched prefill, B=1 non-graph decode, off/on
 *   G1/G<B>  fast FP32 batched prefill, graph decode at B=1 and each --batches B, off/on/onnr
 *   FORCE_F1 fast FP32 B=1 teacher-forced on C0's ids, ledger on (gate G2 rows)
 *   G3       lean, ledger on, observer ports of every layer dumped for recomputation
 *   bf16_G1  BF16 lane (FP32 matrices freed first), graph B=1, off/on/onnr
 * Outputs in OUTDIR: runs.jsonl (runs, row indexes, gates' raw data), logits-<bank>.f32
 * ([prompts][vocab]), ledger-<arm>-<mode>.f32 (rows, float32), states.f32 (final-norm
 * output of every lean decode row), g3.f32 + g3_static.f32 (G3 records).
 * Smoke passes (prompt 0) gate the scored arms; exit 3 aborts before timing. */
#include "qwen.h"
#include "qwen_fast.h"
#include "qwen_ledger.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double mono(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static double wall(void) { struct timespec t; clock_gettime(CLOCK_REALTIME, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void die(const char *what, int code) { fprintf(stderr, "ledger-run: %s (%d)\n", what, code); exit(2); }
static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) die("out of memory", 0); return p; }
static void *xcalloc(size_t n, size_t s) { void *p = calloc(n ? n : 1, s); if (!p) die("out of memory", 0); return p; }
static const char *phase_path;
static void phase(const char *fmt, ...) {
    if (!phase_path) return;
    char label[128], tmp[1100]; va_list ap; va_start(ap, fmt); vsnprintf(label, sizeof(label), fmt, ap); va_end(ap);
    snprintf(tmp, sizeof(tmp), "%s.tmp", phase_path); FILE *f = fopen(tmp, "w"); if (!f) return; fputs(label, f); fclose(f); rename(tmp, phase_path);
}

static int NP, *LENS, **PROMPTS, STEPS = 64, V, RF, D, LONGEST;
static QwenModel M; static QwenSession S; static QwenFast *F; static FILE *OUT; static const char *DIR_;
static int **LEAN_IDS; /* C0 ids per prompt (ids[0..STEPS]), for teacher forcing */
#define LS ((struct QwenSession *)(void *)&S)
enum { OFF = 0, ON = 1, ONNR = 2 };
static const char *MODE[] = {"off", "on", "onnr"};

/* ---------------- runs and ledger row storage ---------------- */
typedef struct {
    const char *arm, *lane, *ledger; int mode, B, graph, forced, steps, nseq;
    int prompt[128]; double prefill_ms[128]; int *ids[128]; double *ms; double t_start, t_end, extra_ms;
    float *rows; int nrows, cap_rows; int *rprompt, *rstep, *rseq;
} Run;
static Run *run_new(const char *arm, const char *lane, int mode, int B, int steps) {
    if (B < 1 || B > 128) die("batch out of range", B);
    Run *r = xcalloc(1, sizeof(Run)); r->arm = arm; r->lane = lane; r->mode = mode; r->ledger = MODE[mode]; r->B = B; r->nseq = B; r->steps = steps;
    r->ms = xcalloc((size_t)steps + 1, sizeof(double)); for (int b = 0; b < B; b++) r->ids[b] = xcalloc((size_t)steps + 1, sizeof(int));
    if (mode != OFF) {
        r->cap_rows = B * (LONGEST + steps + 1);
        r->rows = xmalloc((size_t)r->cap_rows * RF * 4); r->rprompt = xmalloc((size_t)r->cap_rows * sizeof(int));
        r->rstep = xmalloc((size_t)r->cap_rows * sizeof(int)); r->rseq = xmalloc((size_t)r->cap_rows * sizeof(int));
    }
    return r;
}
static void run_free(Run *r) { for (int b = 0; b < r->nseq; b++) free(r->ids[b]); free(r->ms); free(r->rows); free(r->rprompt); free(r->rstep); free(r->rseq); free(r); }
/* Space for the next `count` rows; mark() then records their (prompt, step, seq). step -1 = prefill row. */
static float *take(Run *r, int count) { if (!r->rows || r->nrows + count > r->cap_rows) die("ledger row buffer", count); return r->rows + (size_t)r->nrows * RF; }
static void mark(Run *r, int count, int prompt, int step, int seq) { for (int i = 0; i < count; i++) { r->rprompt[r->nrows] = prompt; r->rstep[r->nrows] = step; r->rseq[r->nrows] = seq; r->nrows++; } }

/* Ledger dump files: one per (arm, mode), rows appended; index lines in runs.jsonl. */
typedef struct { char name[96]; FILE *f; long rows; } Dump;
static Dump DUMPS[64]; static int NDUMP;
static Dump *dump_for(const char *arm, const char *mode) {
    char name[96]; snprintf(name, sizeof(name), "ledger-%s-%s.f32", arm, mode);
    for (int i = 0; i < NDUMP; i++) if (!strcmp(DUMPS[i].name, name)) return &DUMPS[i];
    if (NDUMP == 64) die("too many ledger dumps", 0);
    Dump *d = &DUMPS[NDUMP++]; snprintf(d->name, sizeof(d->name), "%s", name);
    char path[4096]; snprintf(path, sizeof(path), "%s/%s", DIR_, name); d->f = fopen(path, "wb"); if (!d->f) die("ledger dump open", 0); d->rows = 0; return d;
}
static void ints(const char *key, const int *v, int n) { fprintf(OUT, ",\"%s\":[", key); for (int i = 0; i < n; i++) fprintf(OUT, "%s%d", i ? "," : "", v[i]); fprintf(OUT, "]"); }
static void emit(const char *type, const Run *r) {
    fprintf(OUT, "{\"type\":\"%s\",\"arm\":\"%s\",\"lane\":\"%s\",\"ledger\":\"%s\",\"B\":%d,\"graph\":%d,\"forced\":%d,\"steps\":%d,\"t_start\":%.6f,\"t_end\":%.6f,\"extra_ms\":%.4f",
            type, r->arm, r->lane, r->ledger, r->B, r->graph, r->forced, r->steps, r->t_start, r->t_end, r->extra_ms);
    ints("prompts", r->prompt, r->nseq);
    fprintf(OUT, ",\"prompt_tokens\":["); for (int b = 0; b < r->nseq; b++) fprintf(OUT, "%s%d", b ? "," : "", LENS[r->prompt[b]]);
    fprintf(OUT, "],\"prefill_ms\":["); for (int b = 0; b < r->nseq; b++) fprintf(OUT, "%s%.4f", b ? "," : "", r->prefill_ms[b]);
    fprintf(OUT, "],\"ms\":["); for (int i = 0; i < r->steps; i++) fprintf(OUT, "%s%.4f", i ? "," : "", r->ms[i]);
    fprintf(OUT, "],\"ids\":[");
    for (int b = 0; b < r->nseq; b++) { fprintf(OUT, "%s[", b ? "," : ""); for (int i = 0; i <= r->steps; i++) fprintf(OUT, "%s%d", i ? "," : "", r->ids[b][i]); fprintf(OUT, "]"); }
    fprintf(OUT, "],\"ledger_rows\":%d}\n", r->nrows);
    if (r->nrows) {
        Dump *d = dump_for(r->arm, r->ledger);
        if (fwrite(r->rows, (size_t)RF * 4, (size_t)r->nrows, d->f) != (size_t)r->nrows) die("ledger dump write", r->nrows);
        fprintf(OUT, "{\"type\":\"ledger_rows\",\"run_type\":\"%s\",\"arm\":\"%s\",\"ledger\":\"%s\",\"lane\":\"%s\",\"B\":%d,\"graph\":%d,\"forced\":%d,\"file\":\"%s\",\"row_start\":%ld,\"count\":%d",
                type, r->arm, r->ledger, r->lane, r->B, r->graph, r->forced, d->name, d->rows, r->nrows);
        ints("prompt", r->rprompt, r->nrows); ints("step", r->rstep, r->nrows); ints("seq", r->rseq, r->nrows);
        fprintf(OUT, "}\n");
        d->rows += r->nrows;
    }
    fflush(OUT);
}
static void emit_free(const char *type, Run *r) { emit(type, r); run_free(r); }

/* Logit banks: [NP][V]; filled[p] marks rows already written. */
typedef struct { char name[64]; float *data; int *filled; } Bank;
#define MAXBANK 96
static Bank BANKS[MAXBANK]; static int NBANK;
static Bank *bank(const char *name) {
    for (int i = 0; i < NBANK; i++) if (!strcmp(BANKS[i].name, name)) return &BANKS[i];
    if (NBANK == MAXBANK) die("too many logit banks", 0);
    Bank *b = &BANKS[NBANK++]; snprintf(b->name, sizeof(b->name), "%s", name); b->data = xcalloc((size_t)NP * V, 4); b->filled = xcalloc(NP, sizeof(int)); return b;
}
static float *bank_row(const char *name, int p) { if (!name) return NULL; Bank *b = bank(name); if (b->filled[p]) return NULL; b->filled[p] = 1; return b->data + (size_t)p * V; }
static void banks_save(void) {
    for (int i = 0; i < NBANK; i++) {
        char path[4096]; snprintf(path, sizeof(path), "%s/logits-%s.f32", DIR_, BANKS[i].name); FILE *f = fopen(path, "wb"); if (!f) die("bank save", i);
        fwrite(BANKS[i].data, 4, (size_t)NP * V, f); fclose(f);
        fprintf(OUT, "{\"type\":\"bank\",\"name\":\"%s\",\"filled\":[", BANKS[i].name); for (int p = 0; p < NP; p++) fprintf(OUT, "%s%d", p ? "," : "", BANKS[i].filled[p]); fprintf(OUT, "]}\n");
    }
    fflush(OUT);
}
static const char *bname(const char *kind, const char *arm, int mode) { char *s = xmalloc(80); snprintf(s, 80, "%s_%s_%s", kind, arm, MODE[mode]); return s; }

/* ---------------- Part 4 states: lean final-norm output of every scored decode row ---------------- */
static float *ST; static int *ST_P, *ST_S, *ST_POS, *ST_T1, *ST_T2; static float *ST_M; static int NST, CAPST;
static void state_add(int p, int step) {
    if (NST >= CAPST) return;
    gpu_read(ST + (size_t)NST * D, S.norm, (size_t)D * 4);
    ST_P[NST] = p; ST_S[NST] = step; ST_POS[NST] = S.position - 1; ST_T1[NST] = S.top.first; ST_T2[NST] = S.top.second;
    ST_M[NST] = (float)((double)S.top.first_value - (double)S.top.second_value); NST++;
}
static void states_save(void) {
    char path[4096]; snprintf(path, sizeof(path), "%s/states.f32", DIR_); FILE *f = fopen(path, "wb"); if (!f) die("states open", 0);
    if (NST && fwrite(ST, (size_t)D * 4, (size_t)NST, f) != (size_t)NST) die("states write", NST);
    fclose(f);
    fprintf(OUT, "{\"type\":\"states\",\"file\":\"states.f32\",\"count\":%d,\"width\":%d,\"what\":\"lean final-norm output (gain applied), one per C0-on decode row\"", NST, D);
    ints("prompt", ST_P, NST); ints("step", ST_S, NST); ints("position", ST_POS, NST); ints("top1", ST_T1, NST); ints("top2", ST_T2, NST);
    fprintf(OUT, ",\"margin\":["); for (int i = 0; i < NST; i++) fprintf(OUT, "%s%.9g", i ? "," : "", ST_M[i]); fprintf(OUT, "]}\n"); fflush(OUT);
}

/* ---------------- lean arm (C0) ---------------- */
static void lean_ledger(int on) { int rc = qwen_ledger_enable(LS, on); if (on && rc) die("qwen_ledger_enable", rc); }
static Run *lean_run(int p, int steps, int mode, const char *first_bank, const char *final_bank, float *first_out, float *final_out, int states) {
    Run *r = run_new("C0", "fp32", mode, 1, steps); r->prompt[0] = p;
    qwen_reset(&S); lean_ledger(mode != OFF);
    gpu_sync(); double a = mono();
    for (int i = 0; i < LENS[p]; i++) {
        if (qwen_push(&S, PROMPTS[p][i])) die("lean prefill", i);
        if (mode != OFF) { if (qwen_ledger_read(LS, take(r, 1))) die("qwen_ledger_read prefill", i); mark(r, 1, p, -1, 0); }
    }
    gpu_sync(); r->prefill_ms[0] = (mono() - a) * 1e3;
    float *row = bank_row(first_bank, p); if (row) gpu_read(row, S.logits, (size_t)V * 4);
    if (first_out) gpu_read(first_out, S.logits, (size_t)V * 4);
    r->t_start = wall();
    for (int i = 0; i < steps; i++) {
        r->ids[0][i] = S.pending; gpu_sync(); a = mono();
        if (qwen_advance(&S)) die("lean advance", i);
        if (mode != OFF) { if (qwen_ledger_read(LS, take(r, 1))) die("qwen_ledger_read", i); mark(r, 1, p, i, 0); }
        gpu_sync(); r->ms[i] = (mono() - a) * 1e3;
        if (states) state_add(p, i); /* outside the timed region */
    }
    r->t_end = wall(); r->ids[0][steps] = S.pending;
    row = bank_row(final_bank, p); if (row) gpu_read(row, S.logits, (size_t)V * 4);
    if (final_out) gpu_read(final_out, S.logits, (size_t)V * 4);
    lean_ledger(0);
    return r;
}

/* ---------------- fast arms ---------------- */
/* Batched prefill of B sequences (prompts base, base+1, ... cycled), then `steps` decode steps.
 * mode ON reads all prefill rows inside the prefill timing and the B rows of every step
 * inside the step timing; ONNR reads the last step's rows once after the timed loop. */
static Run *fast_run(const char *arm, const char *lane, int B, int base, int graph, int forced, int steps, int mode,
                     const char *first_bank, const char *final_bank, float *first_out, float *final_all) {
    Run *r = run_new(arm, lane, mode, B, steps); r->graph = graph; r->forced = forced;
    qf_reset(F); if (qf_ledger_enable(F, mode != OFF)) die("qf_ledger_enable", mode);
    for (int b = 0; b < B; b++) {
        int p = (base + b) % NP; r->prompt[b] = p;
        qf_sync(F); double a = mono();
        if (qf_prefill(F, b, PROMPTS[p], LENS[p], 1)) die("fast prefill", b);
        if (mode == ON) { if (qf_ledger_read_rows(F, 0, LENS[p], take(r, LENS[p]))) die("qf_ledger_read_rows prefill", b); mark(r, LENS[p], p, -1, b); }
        qf_sync(F); r->prefill_ms[b] = (mono() - a) * 1e3;
        float *row = bank_row(first_bank, p); if (row) qf_read_logits(F, 0, row);
        if (first_out && b == 0) qf_read_logits(F, 0, first_out);
    }
    if (graph) { double a = mono(); if (qf_graph_capture(F, B)) die("graph capture", B); qf_sync(F); r->extra_ms = (mono() - a) * 1e3; }
    int *fids = xmalloc((size_t)B * sizeof(int));
    r->t_start = wall();
    for (int i = 0; i < steps; i++) {
        if (forced) for (int b = 0; b < B; b++) fids[b] = LEAN_IDS[r->prompt[b]][i];
        qf_sync(F); double a = mono();
        if (graph ? qf_graph_step(F) : qf_decode(F, B, forced ? fids : NULL)) die("fast decode", i);
        if (mode == ON) { if (qf_ledger_read_rows(F, 0, B, take(r, B))) die("qf_ledger_read_rows", i); for (int b = 0; b < B; b++) mark(r, 1, r->prompt[b], i, b); }
        qf_sync(F); r->ms[i] = (mono() - a) * 1e3;
    }
    r->t_end = wall(); free(fids);
    if (mode == ONNR && steps > 0) { if (qf_ledger_read_rows(F, 0, B, take(r, B))) die("qf_ledger_read_rows onnr", 0); for (int b = 0; b < B; b++) mark(r, 1, r->prompt[b], steps - 1, b); }
    for (int b = 0; b < B; b++) {
        if (qf_read_ids(F, b, r->ids[b], steps + 1)) die("read ids", b);
        float *row = bank_row(final_bank, r->prompt[b]); if (row) qf_read_logits(F, b, row);
        if (final_all) qf_read_logits(F, b, final_all + (size_t)b * V);
    }
    return r;
}

/* ---------------- row checks (smoke; bug detectors, not the scored gates) ---------------- */
static int fin(float x) { return isfinite(x); }
/* Structural sanity of one row; returns NULL or a reason. */
static const char *row_bad(const float *row, int expect_readout) {
    static char why[160];
    int L = M.layers, H = M.qheads;
    for (int i = 0; i < RF; i++) if (!fin(row[i])) { snprintf(why, sizeof(why), "non-finite field %d", i); return why; }
    int pos = (int)row[QL_POSITION];
    if (row[QL_SOURCES] != (float)(pos + 1)) return "sources != position+1";
    if (expect_readout >= 0 && row[QL_HAS_READOUT] != (float)expect_readout) { snprintf(why, sizeof(why), "has_readout %g, expected %d", row[QL_HAS_READOUT], expect_readout); return why; }
    if (row[QL_HAS_READOUT] == 1.0f) {
        if (row[QL_TOP1] == row[QL_TOP2] || row[QL_MARGIN] < 0 || !(row[QL_C_NORM] > 0) || !(row[QL_RHO_FINAL] > 0)) return "bad readout fields";
        double rem = fabs((double)row[QL_MARGIN] - row[QL_MARGIN_SUM]);
        if (rem > 0.5 + 0.1 * fabs(row[QL_MARGIN])) { snprintf(why, sizeof(why), "margin closure remainder %.4g (margin %.4g)", rem, row[QL_MARGIN]); return why; }
    } else {
        const int f[] = {QL_TOP1, QL_TOP2, QL_MARGIN, QL_RHO_FINAL, QL_C_NORM, QL_MARGIN_EMBED, QL_MARGIN_SUM};
        for (unsigned i = 0; i < sizeof(f) / sizeof(f[0]); i++) if (row[f[i]] != 0.0f) return "readout field nonzero without readout";
    }
    for (int l = 0; l < L; l++) {
        const float *b = row + ql_layer_offset(l, H);
        if (!(b[QL_RHO_IN] > 0) || !(b[QL_RHO_MID] > 0) || b[QL_ATTN_NORM2] < 0 || b[QL_MLP_NORM2] < 0 || !(b[QL_PROD2] > 0)) { snprintf(why, sizeof(why), "layer %d scalars", l); return why; }
        if (b[QL_GATE_SMALL] < 0 || b[QL_GATE_SMALL] > M.intermediate || b[QL_GATE_OFF] < 0 || b[QL_GATE_ON] < 0) { snprintf(why, sizeof(why), "layer %d gate counts", l); return why; }
        if (row[QL_HAS_READOUT] != 1.0f && (b[QL_MARGIN_ATTN] != 0 || b[QL_MARGIN_MLP] != 0)) return "layer margin without readout";
        for (int h = 0; h < H; h++) {
            float pm = row[ql_head_offset(l, H, QL_H_PMAX, h)], pf = row[ql_head_offset(l, H, QL_H_PFIRST, h)], en = row[ql_head_offset(l, H, QL_H_ENTROPY, h)], am = row[ql_head_offset(l, H, QL_H_ARGMAX, h)];
            if (!(pm > 0 && pm <= 1.0f) || pf < 0 || pf > 1.0f || en < -1e-4f || en > logf((float)(pos + 1)) + 1e-3f || am < 0 || am > pos) { snprintf(why, sizeof(why), "layer %d head %d softmax fields", l, h); return why; }
        }
    }
    return NULL;
}
static int close_rel(double a, double b, double tol) { return fabs(a - b) <= tol * fmax(1.0, fmax(fabs(a), fabs(b))); }
/* Lean vs fast rows of the same (prompt, position): continuous fields to 1e-2 (bug detector). */
static const char *rows_disagree(const float *a, const float *b) {
    static char why[160];
    int L = M.layers, H = M.qheads;
    if (a[QL_POSITION] != b[QL_POSITION] || a[QL_TOKEN] != b[QL_TOKEN]) return "token/position differ";
    if (a[QL_HAS_READOUT] == 1.0f && b[QL_HAS_READOUT] == 1.0f) {
        if (a[QL_TOP1] != b[QL_TOP1]) return "top1 differs";
        const int f[] = {QL_MARGIN, QL_RHO_FINAL, QL_C_NORM, QL_MARGIN_SUM};
        for (unsigned i = 0; i < 4; i++) if (!close_rel(a[f[i]], b[f[i]], 1e-2)) { snprintf(why, sizeof(why), "head field %d: %.6g vs %.6g", f[i], a[f[i]], b[f[i]]); return why; }
    }
    for (int l = 0; l < L; l++) {
        const float *x = a + ql_layer_offset(l, H), *y = b + ql_layer_offset(l, H);
        const int f[] = {QL_RHO_IN, QL_RHO_MID, QL_ATTN_NORM2, QL_MLP_NORM2, QL_EVEN2, QL_ODD2, QL_PROD2};
        for (unsigned i = 0; i < 7; i++) if (!close_rel(x[f[i]], y[f[i]], 1e-2)) { snprintf(why, sizeof(why), "layer %d field %d: %.6g vs %.6g", l, f[i], x[f[i]], y[f[i]]); return why; }
        for (int h = 0; h < H; h++) {
            int o = ql_head_offset(l, H, QL_H_LOGZ, h) , q = ql_head_offset(l, H, QL_H_PMAX, h);
            if (!close_rel(a[o], b[o], 1e-2) || !close_rel(a[q], b[q], 1e-2)) { snprintf(why, sizeof(why), "layer %d head %d logZ/pmax", l, h); return why; }
        }
    }
    return NULL;
}
static void smoke_result(const char *name, int ok, const char *why) {
    fprintf(OUT, "{\"type\":\"smoke_result\",\"phase\":\"%s\",\"ok\":%s,\"why\":\"%s\"}\n", name, ok ? "true" : "false", why); fflush(OUT);
    fprintf(stderr, "[smoke %s] ok=%d %s\n", name, ok, why);
}
#define FAIL(...) do { if (ok) snprintf(why, whyn, __VA_ARGS__); ok = 0; } while (0)
static int same_bits(const float *a, const float *b, size_t n) { return !memcmp(a, b, n * 4); }
static int same_ids(const Run *a, const Run *b) { for (int s = 0; s < a->nseq; s++) if (memcmp(a->ids[s], b->ids[s], (size_t)(a->steps + 1) * 4)) return 0; return 1; }

/* FP32 smoke on prompt 0: lean off/on and fast F1/G1/G<B> off/on/onnr bit-identical in
 * logits and ids; every ledger row structurally sane; fast forced rows close to lean rows. */
static int smoke_fp32(int ss, int bsmoke, char *why, size_t whyn) {
    int ok = 1; why[0] = 0;
    float *lf[2], *ll[2]; for (int i = 0; i < 2; i++) { lf[i] = xmalloc((size_t)V * 4); ll[i] = xmalloc((size_t)V * 4); }
    Run *l0 = lean_run(0, ss, OFF, NULL, NULL, lf[0], ll[0], 0), *l1 = lean_run(0, ss, ON, NULL, NULL, lf[1], ll[1], 0);
    memcpy(LEAN_IDS[0], l0->ids[0], (size_t)(ss + 1) * 4);
    if (!same_ids(l0, l1) || !same_bits(lf[0], lf[1], V) || !same_bits(ll[0], ll[1], V)) FAIL("lean: ledger on changes logits or ids");
    if (l1->nrows != LENS[0] + ss) FAIL("lean: %d ledger rows, expected %d", l1->nrows, LENS[0] + ss);
    for (int i = 0; i < l1->nrows; i++) { const char *b = row_bad(l1->rows + (size_t)i * RF, 1); if (b) { FAIL("lean row %d: %s", i, b); break; } }
    /* fast B=1: F1 off/on, G1 off/on/onnr */
    float *ff[5], *fl[5]; Run *fr[5];
    static const int FG[5] = {0, 0, 1, 1, 1}, FM[5] = {OFF, ON, OFF, ON, ONNR};
    for (int k = 0; k < 5; k++) {
        ff[k] = xmalloc((size_t)V * 4); fl[k] = xmalloc((size_t)V * 4);
        fr[k] = fast_run(FG[k] ? "smoke_G1" : "smoke_F1", "fp32", 1, 0, FG[k], 0, ss, FM[k], NULL, NULL, ff[k], fl[k]);
        if (k && (!same_ids(fr[0], fr[k]) || !same_bits(ff[0], ff[k], V) || !same_bits(fl[0], fl[k], V))) FAIL("fast fp32 %s/%s not bit-identical to F1 off", FG[k] ? "G1" : "F1", MODE[FM[k]]);
    }
    {   /* prefill rows: only the last has a readout */
        Run *r = fr[1];
        if (r->nrows != LENS[0] + ss) FAIL("fast: %d ledger rows, expected %d", r->nrows, LENS[0] + ss);
        for (int i = 0; i < r->nrows; i++) { const char *b = row_bad(r->rows + (size_t)i * RF, i < LENS[0] - 1 ? 0 : 1); if (b) { FAIL("fast row %d: %s", i, b); break; } }
        /* graph on rows == non-graph on rows (same kernels, replayed) */
        if (fr[3]->nrows == r->nrows && memcmp(fr[3]->rows, r->rows, (size_t)r->nrows * RF * 4)) FAIL("fast G1 ledger rows differ from F1 ledger rows");
    }
    /* fast forced on C0 ids vs lean rows, decode positions */
    Run *fo = fast_run("smoke_FORCE_F1", "fp32", 1, 0, 0, 1, ss, ON, NULL, NULL, NULL, NULL);
    /* every row of prompt 0: all prefill rows (layer and head fields; readout fields
     * only on the last, where both have one) and every decode row */
    for (int i = 0; i < LENS[0] + ss && ok; i++) {
        const char *b = rows_disagree(l1->rows + (size_t)i * RF, fo->rows + (size_t)i * RF);
        if (b) FAIL("lean vs fast row %d (%s): %s", i, i < LENS[0] ? "prefill" : "decode", b);
    }
    /* graph at B=bsmoke off/on/onnr: ids and final logits of every sequence */
    if (bsmoke > 1) {
        float *ga[3]; Run *gr[3];
        for (int k = 0; k < 3; k++) {
            ga[k] = xmalloc((size_t)bsmoke * V * 4);
            gr[k] = fast_run("smoke_GB", "fp32", bsmoke, 0, 1, 0, ss, k, NULL, NULL, NULL, ga[k]);
            if (k && (!same_ids(gr[0], gr[k]) || !same_bits(ga[0], ga[k], (size_t)bsmoke * V))) FAIL("fast graph B=%d ledger %s not bit-identical to off", bsmoke, MODE[k]);
        }
        for (int i = 0; i < gr[1]->nrows && ok; i++) { const char *b = row_bad(gr[1]->rows + (size_t)i * RF, -1); if (b) FAIL("fast B=%d row %d: %s", bsmoke, i, b); }
        for (int k = 0; k < 3; k++) { emit_free("smoke", gr[k]); free(ga[k]); }
    }
    emit_free("smoke", l0); emit_free("smoke", l1); emit_free("smoke", fo);
    for (int k = 0; k < 5; k++) { emit_free("smoke", fr[k]); free(ff[k]); free(fl[k]); }
    for (int i = 0; i < 2; i++) { free(lf[i]); free(ll[i]); }
    return ok;
}
static int smoke_lane(const char *lane, int ss, char *why, size_t whyn) {
    int ok = 1; why[0] = 0;
    float *ff[5], *fl[5]; Run *fr[5];
    static const int FG[5] = {0, 0, 1, 1, 1}, FM[5] = {OFF, ON, OFF, ON, ONNR};
    for (int k = 0; k < 5; k++) {
        ff[k] = xmalloc((size_t)V * 4); fl[k] = xmalloc((size_t)V * 4);
        fr[k] = fast_run(FG[k] ? "smoke_lane_G1" : "smoke_lane_F1", lane, 1, 0, FG[k], 0, ss, FM[k], NULL, NULL, ff[k], fl[k]);
        if (k && (!same_ids(fr[0], fr[k]) || !same_bits(ff[0], ff[k], V) || !same_bits(fl[0], fl[k], V))) FAIL("%s %s/%s not bit-identical to F1 off", lane, FG[k] ? "G1" : "F1", MODE[FM[k]]);
    }
    for (int i = 0; i < fr[1]->nrows && ok; i++) { const char *b = row_bad(fr[1]->rows + (size_t)i * RF, i < LENS[0] - 1 ? 0 : 1); if (b) FAIL("%s row %d: %s", lane, i, b); }
    for (int k = 0; k < 5; k++) { emit_free("smoke", fr[k]); free(ff[k]); free(fl[k]); }
    return ok;
}

/* ---------------- G3: lean observer ports for independent recomputation ----------------
 * Record layout (float32), pos = position of the consumed token, P = pos+1:
 *   for each layer: input[d] query[d] attention_write[d] raw_receiver[d] gate[h] up[h]
 *                   product[h] mlp_write[d] output[d] probabilities[qheads*P]
 *   keys: [layer][kvhead][P][head]   (post-RoPE cache rows 0..pos)
 *   W_a[d] W_b[d] logit_a logit_b    (a, b = lean top-2 of this token)
 *   ledger row [row_floats]          (qwen_ledger_read of this token)
 * g3_static.f32 = final-norm gain [d]. */
enum { G_INPUT, G_QUERY, G_AW, G_RAW, G_GATE, G_UP, G_PROD, G_MW, G_OUT, G_PROBS, G_N };
static const QwenPort G_PORTS[G_N] = {QWEN_INPUT, QWEN_QUERY, QWEN_CONSUMED_ATTENTION_WRITE, QWEN_RAW_RECEIVER, QWEN_MLP_GATE, QWEN_MLP_UP, QWEN_MLP_PRODUCT, QWEN_CONSUMED_MLP_WRITE, QWEN_OUTPUT, QWEN_ATTENTION_PROBABILITIES};
static const char *G_NAMES[G_N] = {"input", "query", "attention_write", "raw_receiver", "gate", "up", "product", "mlp_write", "output", "probabilities"};
typedef struct { float *buf[QF_MAX_LAYERS][G_N]; size_t count[QF_MAX_LAYERS][G_N]; int seen[QF_MAX_LAYERS][G_N]; int bad; } Obs;
static size_t g_cap(int k) { return k == G_PROBS ? (size_t)M.qheads * S.capacity : (k == G_GATE || k == G_UP || k == G_PROD) ? (size_t)M.intermediate : (size_t)D; }
static void g3_observer(void *ctx, int position, int layer, QwenPort port, const float *data, size_t count) {
    Obs *o = ctx; (void)position;
    if (layer < 0 || layer >= M.layers) { o->bad = 1; return; }
    for (int k = 0; k < G_N; k++) if (G_PORTS[k] == port) {
        if (count > g_cap(k)) { o->bad = 2; return; }
        gpu_read(o->buf[layer][k], data, count * 4); o->count[layer][k] = count; o->seen[layer][k]++; return;
    }
}
static long G3_FLOATS;
static void fw(FILE *f, const float *p, size_t n) { if (n && fwrite(p, 4, n, f) != n) die("g3 write", (int)n); G3_FLOATS += (long)n; }
static void g3_run(FILE *g, Obs *o, int p, int tokens, float *tmp) {
    qwen_reset(&S); lean_ledger(1);
    for (int i = 0; i < LENS[p]; i++) if (qwen_push(&S, PROMPTS[p][i])) die("g3 prefill", i);
    float *lrow = xmalloc((size_t)RF * 4);
    for (int i = 0; i < tokens; i++) {
        int pos = S.position, tok = S.pending;
        memset(o->count, 0, sizeof(o->count)); memset(o->seen, 0, sizeof(o->seen)); o->bad = 0;
        S.observer = g3_observer; S.observer_context = o; S.observe_position = pos; S.observe_layer = -1;
        int rc = qwen_advance(&S);
        S.observer = NULL; S.observer_context = NULL;
        if (rc) die("g3 advance", i);
        int lrc = qwen_ledger_read(LS, lrow);
        const char *err = NULL;
        if (o->bad) err = "observer refused a port (size or layer)";
        if (lrc) err = "qwen_ledger_read failed";
        for (int l = 0; l < M.layers && !err; l++) for (int k = 0; k < G_N && !err; k++) {
            size_t want = k == G_PROBS ? (size_t)M.qheads * (pos + 1) : (k == G_GATE || k == G_UP || k == G_PROD) ? (size_t)M.intermediate : (size_t)D;
            if (o->seen[l][k] != 1 || o->count[l][k] != want) err = "port missing, repeated or wrong size";
        }
        if (err) { fprintf(OUT, "{\"type\":\"g3_error\",\"prompt\":%d,\"step\":%d,\"position\":%d,\"why\":\"%s\"}\n", p, i, pos, err); fflush(OUT); continue; }
        long start = G3_FLOATS;
        for (int l = 0; l < M.layers; l++) for (int k = 0; k < G_N; k++) fw(g, o->buf[l][k], o->count[l][k]);
        for (int l = 0; l < M.layers; l++) {
            size_t base = (size_t)S.cache_slots[l] * M.kvheads * S.capacity * M.head;
            for (int h = 0; h < M.kvheads; h++) { gpu_read(tmp, S.keys + base + (size_t)h * S.capacity * M.head, (size_t)(pos + 1) * M.head * 4); fw(g, tmp, (size_t)(pos + 1) * M.head); }
        }
        int a = S.top.first, b = S.top.second; float lab[2];
        gpu_read(tmp, M.embedding + (size_t)a * D, (size_t)D * 4); fw(g, tmp, D);
        gpu_read(tmp, M.embedding + (size_t)b * D, (size_t)D * 4); fw(g, tmp, D);
        gpu_read(&lab[0], S.logits + a, 4); gpu_read(&lab[1], S.logits + b, 4); fw(g, lab, 2);
        fw(g, lrow, RF);
        fprintf(OUT, "{\"type\":\"g3\",\"prompt\":%d,\"step\":%d,\"position\":%d,\"token\":%d,\"top1\":%d,\"top2\":%d,\"offset\":%ld,\"floats\":%ld}\n", p, i, pos, tok, a, b, start, G3_FLOATS - start);
        fflush(OUT);
    }
    free(lrow); lean_ledger(0);
}

/* ---------------- helpers ---------------- */
static void free_matrix(void *ctx, float *p) {
    QwenModel *m = ctx; for (unsigned i = 0; i < m->allocation_count; i++) if (m->allocations[i] == p) { gpu_free(p); m->allocations[i] = NULL; return; }
    die("free_matrix: unknown pointer", 0);
}
static int **read_prompts(const char *path, int *count, int **lengths, int *longest) {
    FILE *f = fopen(path, "r"); if (!f) die("cannot open prompts", 0);
    int cap = 8, n = 0; int **rows = xmalloc(cap * sizeof(int *)); int *len = xmalloc(cap * sizeof(int)); *longest = 0; int k;
    while (fscanf(f, "%d", &k) == 1) {
        if (k < 1) die("empty prompt line", n);
        if (n == cap) { cap *= 2; rows = realloc(rows, cap * sizeof(int *)); len = realloc(len, cap * sizeof(int)); if (!rows || !len) die("realloc", 0); }
        rows[n] = xmalloc((size_t)k * sizeof(int)); for (int i = 0; i < k; i++) if (fscanf(f, "%d", &rows[n][i]) != 1) die("short prompt line", n);
        len[n] = k; if (k > *longest) *longest = k; n++;
    }
    fclose(f); *count = n; *lengths = len; return rows;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ledger-run MODEL PROMPTS OUTDIR [--steps N] [--warmup-steps N] [--smoke-steps N] [--g3-prompts N] [--g3-tokens N] [--batches 8,64] [--bf16 0|1] [--phase-file P]\n"); return 2; }
    int warmup = 8, smoke_steps = 4, g3_prompts = 4, g3_tokens = 16, do_bf16 = 1, NB = 0, BS[8], profile = 0;
    for (int i = 4; i < argc; i += 2) {
        if (i + 1 >= argc) return 2;
        if (!strcmp(argv[i], "--steps")) STEPS = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--warmup-steps")) warmup = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--smoke-steps")) smoke_steps = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--g3-prompts")) g3_prompts = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--g3-tokens")) g3_tokens = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--bf16")) do_bf16 = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--profile")) profile = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--phase-file")) phase_path = argv[i + 1];
        else if (!strcmp(argv[i], "--batches")) {
            char *s = argv[i + 1];
            while (*s && NB < 8) { char *end; long b = strtol(s, &end, 10); if (end == s || b < 2 || b > 128) return 2; BS[NB++] = (int)b; s = *end == ',' ? end + 1 : end; if (*end && *end != ',') return 2; }
        } else return 2;
    }
    if (STEPS < 1 || smoke_steps < 1 || warmup < 1 || g3_tokens < 0 || g3_prompts < 0) return 2;
    if (smoke_steps > STEPS) smoke_steps = STEPS; /* LEAN_IDS rows hold STEPS+1 ids; capacity holds STEPS decode rows */
    if (warmup > STEPS) warmup = STEPS;
    DIR_ = argv[3]; double started = mono();
    PROMPTS = read_prompts(argv[2], &NP, &LENS, &LONGEST); if (NP < 1) die("no prompts", 0);
    if (g3_prompts > NP) g3_prompts = NP;
    if (g3_tokens > STEPS) g3_tokens = STEPS;
    int capacity = LONGEST + STEPS + 8;
    char path[4096]; snprintf(path, sizeof(path), "%s/runs.jsonl", DIR_); OUT = fopen(path, "w"); if (!OUT) die("output", 0);
    phase("c_load");
    TcmModel bundle; char error[512];
    if (tcm_load(&bundle, argv[1], error, sizeof(error))) { fprintf(stderr, "%s\n", error); return 2; }
    if (qwen_model_init(&M, &bundle, error, sizeof(error))) { fprintf(stderr, "%s\n", error); return 2; }
    tcm_unload(&bundle); gpu_sync(); double load_s = mono() - started; V = M.vocab; D = M.hidden;
    if (qwen_session_init(&S, &M, capacity)) die("session", 0);
    QfWeights w; memset(&w, 0, sizeof(w));
    w.hidden = M.hidden; w.layers = M.layers; w.qheads = M.qheads; w.kvheads = M.kvheads; w.head = M.head; w.intermediate = M.intermediate; w.vocab = M.vocab; w.epsilon = M.epsilon; w.theta = M.theta;
    w.embedding = M.embedding; w.norm = M.norm;
    for (int l = 0; l < M.layers; l++) {
        const QwenLayer *L = &M.layer[l];
        w.input_norm[l] = L->input_norm; w.post_norm[l] = L->post_norm; w.q[l] = L->q; w.k[l] = L->k; w.v[l] = L->v; w.qb[l] = L->qb; w.kb[l] = L->kb; w.vb[l] = L->vb;
        w.o[l] = L->o; w.gate[l] = L->gate; w.up[l] = L->up; w.down[l] = L->down;
    }
    int BMAX = 1; for (int i = 0; i < NB; i++) if (BS[i] > BMAX) BMAX = BS[i];
    F = qf_open(&w, BMAX, capacity, LONGEST > BMAX ? LONGEST : BMAX, STEPS + 8); if (!F) die("qf_open", 0);
    RF = qf_ledger_row_floats(F); if (RF != ql_row_floats(M.layers, M.qheads)) die("row floats", RF);
    size_t fast_bytes_off = qf_device_bytes(F);
    fprintf(OUT, "{\"type\":\"header\",\"model_identity\":\"%s\",\"model_digest\":\"%s\",\"vocab\":%d,\"layers\":%d,\"qheads\":%d,\"kvheads\":%d,\"head\":%d,\"hidden\":%d,\"intermediate\":%d,"
                 "\"epsilon\":%.9g,\"capacity\":%d,\"steps\":%d,\"prompts\":%d,\"smoke_steps\":%d,\"g3_prompts\":%d,\"g3_tokens\":%d,\"row_floats\":%d,\"ql_head\":%d,\"ql_layer\":%d,\"ql_per_head\":%d,"
                 "\"load_s\":%.3f,\"gpu_weight_bytes\":%zu,\"lean_session_bytes\":%zu,\"fast_device_bytes\":%zu,\"lane_weight_bytes\":{\"fp32\":%zu,\"bf16\":%zu,\"int8\":%zu}",
            M.identity, M.digest, V, M.layers, M.qheads, M.kvheads, M.head, M.hidden, M.intermediate, (double)M.epsilon, capacity, STEPS, NP, smoke_steps, g3_prompts, g3_tokens, RF, QL_HEAD, QL_LAYER, QL_PER_HEAD,
            load_s, M.weight_bytes, S.allocated_bytes, fast_bytes_off, qf_lane_weight_bytes(F, QF_FP32), qf_lane_weight_bytes(F, QF_BF16), qf_lane_weight_bytes(F, QF_INT8));
    ints("batches", BS, NB); fprintf(OUT, "}\n"); fflush(OUT);
    LEAN_IDS = xcalloc(NP, sizeof(int *)); for (int p = 0; p < NP; p++) LEAN_IDS[p] = xcalloc((size_t)STEPS + 1, sizeof(int));
    CAPST = NP * STEPS; ST = xmalloc((size_t)CAPST * D * 4); ST_P = xmalloc(CAPST * sizeof(int)); ST_S = xmalloc(CAPST * sizeof(int)); ST_POS = xmalloc(CAPST * sizeof(int));
    ST_T1 = xmalloc(CAPST * sizeof(int)); ST_T2 = xmalloc(CAPST * sizeof(int)); ST_M = xmalloc(CAPST * sizeof(float));
    char why[256];

    /* ================= smoke, FP32 (gates everything below) ================= */
    phase("smoke_fp32");
    {
        int ok = smoke_fp32(smoke_steps, NB ? BS[0] : 0, why, sizeof(why));
        smoke_result("fp32", ok, why);
        if (!ok) { banks_save(); fclose(OUT); return 3; }
    }
    size_t fast_bytes_on = qf_device_bytes(F);

    /* ================= warmup ================= */
    phase("warmup");
    emit_free("warmup", lean_run(0, warmup, OFF, NULL, NULL, NULL, NULL, 0));
    emit_free("warmup", lean_run(0, warmup, ON, NULL, NULL, NULL, NULL, 0));
    emit_free("warmup", fast_run("G1", "fp32", 1, 0, 1, 0, warmup, OFF, NULL, NULL, NULL, NULL));
    emit_free("warmup", fast_run("G1", "fp32", 1, 0, 1, 0, warmup, ON, NULL, NULL, NULL, NULL));
    if (NB) emit_free("warmup", fast_run("GB", "fp32", BMAX, 0, 1, 0, warmup, ON, NULL, NULL, NULL, NULL));

    /* ================= profile mode: cost of each ledger piece (graph decode, rows read once) =================
     * Arm P<B>_s<mask>: ledger on with the pieces in mask running their native kernel instead
     * (1 RMS dots, 2 attention softmax fields, 4 SwiGLU census, 8 readout: top-2 + final norm + margin).
     * P<B>_off: ledger off. Order rotated per prompt / rep. Nothing else runs. */
    if (profile) {
        static const int MASKS[] = {-1, 0, QF_SKIP_RMS, QF_SKIP_ATTENTION, QF_SKIP_SWIGLU, QF_SKIP_READOUT, 15};
        enum { NM = 7 };
        int PB[9] = {1}, npb = 1; for (int i = 0; i < NB; i++) PB[npb++] = BS[i];
        for (int bi = 0; bi < npb; bi++) {
            int B = PB[bi], reps = B == 1 ? NP : 2;
            for (int rep = 0; rep < reps; rep++) for (int k = 0; k < NM; k++) {
                int m = MASKS[(k + rep) % NM]; char *arm = xmalloc(24);
                if (m < 0) snprintf(arm, 24, "P%d_off", B); else snprintf(arm, 24, "P%d_s%d", B, m);
                phase("%s", arm);
                qf_ledger_profile_skip(F, m < 0 ? 0 : m);
                emit_free("run", fast_run(arm, "fp32", B, B == 1 ? rep : 0, 1, 0, STEPS, m < 0 ? OFF : ONNR, NULL, NULL, NULL, NULL));
            }
        }
        qf_ledger_profile_skip(F, 0);
        phase("c_done");
        for (int i = 0; i < NDUMP; i++) fclose(DUMPS[i].f);
        fprintf(OUT, "{\"type\":\"footer\",\"profile\":1,\"wall_s\":%.3f,\"peak_gpu_bytes\":%zu}\n", mono() - started, gpu_peak_bytes());
        fclose(OUT); qf_close(F); qwen_session_free(&S); qwen_model_free(&M);
        return 0;
    }

    /* ================= A. B=1 arms, order rotated per prompt ================= */
    enum { A_C0_OFF, A_C0_ON, A_F1_OFF, A_F1_ON, A_G1_OFF, A_G1_ON, A_G1_ONNR, A_N };
    for (int p = 0; p < NP; p++) {
        for (int k = 0; k < A_N; k++) {
            int a = (k + p) % A_N; Run *r;
            switch (a) {
            case A_C0_OFF: phase("C0_off"); r = lean_run(p, STEPS, OFF, "first_C0_off", "final_C0_off", NULL, NULL, 0); memcpy(LEAN_IDS[p], r->ids[0], (size_t)(STEPS + 1) * 4); break;
            case A_C0_ON: phase("C0_on"); r = lean_run(p, STEPS, ON, "first_C0_on", "final_C0_on", NULL, NULL, 1); break;
            case A_F1_OFF: phase("F1_off"); r = fast_run("F1", "fp32", 1, p, 0, 0, STEPS, OFF, "first_F1_off", "final_F1_off", NULL, NULL); break;
            case A_F1_ON: phase("F1_on"); r = fast_run("F1", "fp32", 1, p, 0, 0, STEPS, ON, "first_F1_on", "final_F1_on", NULL, NULL); break;
            case A_G1_OFF: phase("G1_off"); r = fast_run("G1", "fp32", 1, p, 1, 0, STEPS, OFF, "first_G1_off", "final_G1_off", NULL, NULL); break;
            case A_G1_ON: phase("G1_on"); r = fast_run("G1", "fp32", 1, p, 1, 0, STEPS, ON, "first_G1_on", "final_G1_on", NULL, NULL); break;
            default: phase("G1_onnr"); r = fast_run("G1", "fp32", 1, p, 1, 0, STEPS, ONNR, "first_G1_onnr", "final_G1_onnr", NULL, NULL); break;
            }
            emit_free("run", r);
        }
    }
    /* ================= B. graph batched decode ================= */
    for (int bi = 0; bi < NB; bi++) {
        int B = BS[bi], reps = B < NP ? (NP + B - 1) / B : 2;
        char *arm = xmalloc(16); snprintf(arm, 16, "G%d", B);
        for (int rep = 0; rep < reps; rep++) {
            int base = B < NP ? rep * B : 0;
            for (int k = 0; k < 3; k++) {
                int mode = (k + rep) % 3; phase("%s_%s", arm, MODE[mode]);
                emit_free("run", fast_run(arm, "fp32", B, base, 1, 0, STEPS, mode, bname("first", arm, mode), bname("final", arm, mode), NULL, NULL));
            }
        }
    }
    /* ================= C. G2 rows: fast FP32 teacher-forced on C0 ids ================= */
    phase("FORCE_F1");
    for (int p = 0; p < NP; p++) emit_free("run", fast_run("FORCE_F1", "fp32", 1, p, 0, 1, STEPS, ON, NULL, "final_FORCE_F1_on", NULL, NULL));
    qf_ledger_enable(F, 0);
    /* ================= D. G3 observer dumps ================= */
    phase("G3");
    {
        snprintf(path, sizeof(path), "%s/g3_static.f32", DIR_); FILE *gs = fopen(path, "wb"); if (!gs) die("g3 static", 0);
        float *gam = xmalloc((size_t)D * 4); gpu_read(gam, M.norm, (size_t)D * 4); fwrite(gam, 4, D, gs); fclose(gs); free(gam);
        snprintf(path, sizeof(path), "%s/g3.f32", DIR_); FILE *g = fopen(path, "wb"); if (!g) die("g3 open", 0);
        Obs *o = xcalloc(1, sizeof(Obs));
        for (int l = 0; l < M.layers; l++) for (int k = 0; k < G_N; k++) o->buf[l][k] = xmalloc(g_cap(k) * 4);
        float *tmp = xmalloc((size_t)S.capacity * M.head * 4 > (size_t)D * 4 ? (size_t)S.capacity * M.head * 4 : (size_t)D * 4);
        fprintf(OUT, "{\"type\":\"g3_layout\",\"ports\":["); for (int k = 0; k < G_N; k++) fprintf(OUT, "%s\"%s\"", k ? "," : "", G_NAMES[k]);
        fprintf(OUT, "],\"then\":[\"keys[layer][kvhead][P][head]\",\"W_a\",\"W_b\",\"logit_a\",\"logit_b\",\"ledger_row\"],\"static\":\"g3_static.f32 = final-norm gain\"}\n");
        for (int p = 0; p < g3_prompts; p++) g3_run(g, o, p, g3_tokens, tmp);
        fclose(g); free(tmp);
        for (int l = 0; l < M.layers; l++) for (int k = 0; k < G_N; k++) free(o->buf[l][k]);
        free(o);
    }
    states_save();

    /* ================= E. BF16 lane (frees the FP32 matrices: lean path retired) ================= */
    int lp_ok = -1;
    if (do_bf16) {
        phase("convert");
        double a = mono(); qf_build_low_precision(F, free_matrix, &M); gpu_sync(); double convert_s = mono() - a;
        unsigned long long inexact, total; qf_bf16_exactness(F, &inexact, &total);
        fprintf(OUT, "{\"type\":\"convert\",\"seconds\":%.3f,\"bf16_inexact_elements\":%llu,\"bf16_total_elements\":%llu,\"live_gpu_bytes\":%zu,\"peak_gpu_bytes\":%zu}\n", convert_s, inexact, total, gpu_live_bytes(), gpu_peak_bytes()); fflush(OUT);
        if (qf_set_lane(F, QF_BF16)) die("set lane bf16", 0);
        phase("smoke_bf16");
        lp_ok = smoke_lane("bf16", smoke_steps, why, sizeof(why)); smoke_result("bf16", lp_ok, why);
        {   /* INT8 must be refused while the ledger is on (contract), and enable must refuse on INT8 */
            int refused = 1;
            if (qf_ledger_enable(F, 1)) refused = 0; else if (qf_set_lane(F, QF_INT8) == 0) refused = 0;
            qf_ledger_enable(F, 0);
            if (qf_set_lane(F, QF_INT8) == 0) { if (qf_ledger_enable(F, 1) == 0) refused = 0; qf_ledger_enable(F, 0); }
            if (qf_set_lane(F, QF_BF16)) die("set lane bf16 again", 0);
            fprintf(OUT, "{\"type\":\"int8_refusal\",\"ok\":%s}\n", refused ? "true" : "false"); fflush(OUT);
        }
        if (lp_ok) {
            emit_free("warmup", fast_run("bf16_G1", "bf16", 1, 0, 1, 0, warmup, ON, NULL, NULL, NULL, NULL));
            for (int p = 0; p < NP; p++) for (int k = 0; k < 3; k++) {
                int mode = (k + p) % 3; phase("bf16_G1_%s", MODE[mode]);
                emit_free("run", fast_run("bf16_G1", "bf16", 1, p, 1, 0, STEPS, mode, bname("first", "bf16_G1", mode), bname("final", "bf16_G1", mode), NULL, NULL));
            }
        }
    }
    phase("c_done");
    banks_save();
    for (int i = 0; i < NDUMP; i++) fclose(DUMPS[i].f);
    fprintf(OUT, "{\"type\":\"footer\",\"lp_ok\":%d,\"wall_s\":%.3f,\"peak_gpu_bytes\":%zu,\"live_gpu_bytes\":%zu,\"fast_device_bytes_ledger_off\":%zu,\"fast_device_bytes_ledger_on\":%zu,\"g3_floats\":%ld,\"states\":%d}\n",
            lp_ok, mono() - started, gpu_peak_bytes(), gpu_live_bytes(), fast_bytes_off, fast_bytes_on, G3_FLOATS, NST);
    fclose(OUT); qf_close(F); qwen_session_free(&S); qwen_model_free(&M);
    return 0;
}
