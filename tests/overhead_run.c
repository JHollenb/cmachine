#define _POSIX_C_SOURCE 200809L
/* Analysis-overhead timing for the C/CUDA Qwen executor (research/overhead_timing).
 *
 *   overhead-run MODEL PROMPTS OUT_JSONL [--steps N] [--warmup-steps N] [--smoke-steps N]
 *                [--capacity N] [--deadline SECONDS] [--phase-file PATH]
 *
 * One model load, one session. PROMPTS lines: "n t0 t1 ... t(n-1)". Per prompt: reset, lean prefill
 * (timed once), then every arm decodes the same N greedy tokens from the same prompt boundary inside a
 * stack frame that is rolled back afterwards. Arm order rotates per prompt. Every timed region and every
 * token is bracketed by gpu_sync(). Arms (bits): 1 full logits download, 2 per-layer residual capture
 * through the session observer (QWEN_INPUT on all layers), 4 observer capture of every port on every
 * layer, 8 cache+logits SHA-256 (copy of qwen_serve.c state_hashes).
 * A smoke pass (first prompt, --smoke-steps tokens, all arms) runs first and aborts (exit 3) when token
 * ids differ across arms, the observer captured nothing, or logits/hashes disagree. */
#include "qwen.h"
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { A_LOGITS = 1, A_OBS = 2, A_OBSALL = 4, A_HASH = 8 };
typedef struct { const char *name; int mask; } Arm;
static const Arm ARMS[] = {{"C0", 0}, {"C1", A_LOGITS}, {"C2", A_OBS}, {"C2b", A_OBSALL}, {"C3", A_HASH}, {"C4", A_LOGITS | A_OBS | A_HASH}};
#define NARMS ((int)(sizeof(ARMS) / sizeof(ARMS[0])))

static double mono(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static double wall(void) { struct timespec t; clock_gettime(CLOCK_REALTIME, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void die(const char *what, int code) { fprintf(stderr, "overhead-run: %s (%d)\n", what, code); exit(2); }

static const char *phase_path;
static void phase(const char *fmt, ...) {
    if (!phase_path) return;
    char label[128], tmp[1100];
    va_list ap; va_start(ap, fmt); vsnprintf(label, sizeof(label), fmt, ap); va_end(ap);
    snprintf(tmp, sizeof(tmp), "%s.tmp", phase_path);
    FILE *f = fopen(tmp, "w"); if (!f) return;
    fputs(label, f); fclose(f); rename(tmp, phase_path);
}

/* Copy of qwen_serve.c state_hashes (S command): SHA-256 of the live K/V rows of every layer/head, then of the logits. */
static void state_hashes(const QwenSession *s, char cache[65], char logits[65]) {
    const QwenModel *m = s->model; size_t row = (size_t)s->position * m->head, n = (size_t)2 * m->layers * m->kvheads * row, offset = 0;
    float *host = malloc((n ? n : 1) * 4); if (!host) exit(2);
    for (int kind = 0; kind < 2; kind++) for (int l = 0; l < m->layers; l++) for (int h = 0; h < m->kvheads; h++) {
        size_t address = ((size_t)l * m->kvheads + h) * s->capacity * m->head;
        if (row) gpu_read(host + offset, (kind ? s->values : s->keys) + address, row * 4); offset += row;
    }
    unsigned char digest[32]; tcm_sha256(host, n * 4, digest); tcm_hex(digest, 32, cache); free(host);
    if (s->phase != 1) { logits[0] = 0; return; }
    host = malloc((size_t)m->vocab * 4); if (!host) exit(2);
    gpu_read(host, s->logits, (size_t)m->vocab * 4); tcm_sha256(host, (size_t)m->vocab * 4, digest); tcm_hex(digest, 32, logits); free(host);
}

typedef struct { int mode; int hidden; float *host; size_t capacity, offset, bytes, calls; } Observed;
/* Observers read the device pointer during the call only; this one copies it to host (the capture cost). */
static void observer(void *context, int position, int layer, QwenPort port, const float *data, size_t count) {
    (void)position; Observed *o = context;
    if (o->mode == 1) {
        if (port != QWEN_INPUT) return;
        size_t at = (size_t)layer * o->hidden; if (at + count > o->capacity) die("observer buffer (residual)", 0);
        gpu_read(o->host + at, data, count * 4);
    } else {
        if (o->offset + count > o->capacity) die("observer buffer (all ports)", 0);
        gpu_read(o->host + o->offset, data, count * 4); o->offset += count;
    }
    o->bytes += count * 4; o->calls++;
}

typedef struct {
    int ids_n; int *ids; double *ms; double t_start, t_end;
    size_t obs_bytes_total, obs_calls_total; /* over all steps */
    char cache_sha[65], logits_sha[65];
    int logits_match_pending; /* downloaded logits' maximum equals the logit of the token the session chose */
} ArmRun;

static void run_arm(QwenSession *s, int mask, int steps, Observed *obs, float *host_logits, ArmRun *r) {
    uint64_t frame; if (qwen_stack_push(s, &frame)) die("frame push", 0);
    int V = s->model->vocab;
    r->cache_sha[0] = r->logits_sha[0] = 0; r->obs_bytes_total = r->obs_calls_total = 0; r->logits_match_pending = -1;
    s->observer = (mask & (A_OBS | A_OBSALL)) ? observer : NULL;
    s->observer_context = obs; obs->mode = (mask & A_OBSALL) ? 2 : (mask & A_OBS) ? 1 : 0;
    gpu_sync(); r->t_start = wall();
    for (int i = 0; i < steps; i++) {
        r->ids[i] = s->pending;
        obs->offset = 0; obs->bytes = 0; obs->calls = 0;
        if (obs->mode) { s->observe_position = s->position; s->observe_layer = -1; }
        gpu_sync(); double a = mono();
        if (qwen_advance(s)) die("advance", i);
        if (mask & A_LOGITS) gpu_read(host_logits, s->logits, (size_t)V * 4);
        if (mask & A_HASH) state_hashes(s, r->cache_sha, r->logits_sha);
        gpu_sync(); r->ms[i] = (mono() - a) * 1e3;
        r->obs_bytes_total += obs->bytes; r->obs_calls_total += obs->calls;
    }
    r->t_end = wall(); r->ids[steps] = s->pending; r->ids_n = steps + 1;
    if (mask & A_LOGITS) { float best = host_logits[0]; for (int j = 1; j < V; j++) if (host_logits[j] > best) best = host_logits[j]; r->logits_match_pending = host_logits[s->pending] == best; }
    s->observer = NULL; s->observe_position = s->observe_layer = -1; obs->mode = 0;
    if (qwen_stack_rollback(s, frame)) die("rollback", 0);
}

static int **read_prompts(const char *path, int *count, int **lengths, int *longest) {
    FILE *f = fopen(path, "r"); if (!f) die("cannot open prompts", 0);
    int cap = 8, n = 0; int **rows = malloc(cap * sizeof(int *)); int *len = malloc(cap * sizeof(int)); *longest = 0;
    int k; while (fscanf(f, "%d", &k) == 1) {
        if (n == cap) { cap *= 2; rows = realloc(rows, cap * sizeof(int *)); len = realloc(len, cap * sizeof(int)); }
        rows[n] = malloc((size_t)k * sizeof(int)); for (int i = 0; i < k; i++) if (fscanf(f, "%d", &rows[n][i]) != 1) die("short prompt line", n);
        len[n] = k; if (k > *longest) *longest = k; n++;
    }
    fclose(f); *count = n; *lengths = len; return rows;
}

static void emit_run(FILE *out, const char *kind, int prompt, int order_pos, const Arm *arm, int prompt_len, const ArmRun *r, int steps, int obs_mode, double prefill_ms) {
    fprintf(out, "{\"type\":\"%s\",\"prompt\":%d,\"arm\":\"%s\",\"order_pos\":%d,\"mask\":%d,\"prompt_tokens\":%d,\"prefill_ms\":%.4f,\"t_start\":%.6f,\"t_end\":%.6f,\"ids\":[", kind, prompt, arm->name, order_pos, arm->mask, prompt_len, prefill_ms, r->t_start, r->t_end);
    for (int i = 0; i < r->ids_n; i++) fprintf(out, "%s%d", i ? "," : "", r->ids[i]);
    fprintf(out, "],\"ms\":[");
    for (int i = 0; i < steps; i++) fprintf(out, "%s%.4f", i ? "," : "", r->ms[i]);
    fprintf(out, "],\"obs_mode\":%d,\"obs_bytes_per_token\":%.1f,\"obs_calls_per_token\":%.2f,\"cache_sha256\":\"%s\",\"logits_sha256\":\"%s\",\"logits_match_pending\":%d}\n",
        obs_mode, steps ? (double)r->obs_bytes_total / steps : 0.0, steps ? (double)r->obs_calls_total / steps : 0.0, r->cache_sha, r->logits_sha, r->logits_match_pending);
    fflush(out);
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: overhead-run MODEL PROMPTS OUT_JSONL [--steps N] [--warmup-steps N] [--smoke-steps N] [--capacity N] [--deadline S] [--phase-file P]\n"); return 2; }
    int steps = 64, warmup_steps = 8, smoke_steps = 4, capacity = 0; double deadline = 1e9;
    for (int i = 4; i < argc; i += 2) {
        if (i + 1 >= argc) return 2;
        if (!strcmp(argv[i], "--steps")) steps = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--warmup-steps")) warmup_steps = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--smoke-steps")) smoke_steps = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--capacity")) capacity = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--deadline")) deadline = atof(argv[i + 1]);
        else if (!strcmp(argv[i], "--phase-file")) phase_path = argv[i + 1];
        else return 2;
    }
    double started = mono();
    int np, longest, *lens; int **prompts = read_prompts(argv[2], &np, &lens, &longest);
    if (np < 1) die("no prompts", 0);
    int need = longest + (steps > warmup_steps ? steps : warmup_steps) + 4; if (capacity < need) capacity = need;
    phase("c_load");
    TcmModel bundle; QwenModel model; char error[512];
    if (tcm_load(&bundle, argv[1], error, sizeof(error))) { fprintf(stderr, "%s\n", error); return 2; }
    if (qwen_model_init(&model, &bundle, error, sizeof(error))) { fprintf(stderr, "%s\n", error); return 2; }
    tcm_unload(&bundle); gpu_sync(); double load_s = mono() - started;
    QwenSession s; if (qwen_session_init(&s, &model, capacity)) die("session", 0);
    int V = model.vocab;
    FILE *out = fopen(argv[3], "w"); if (!out) die("output", 0);
    Observed obs; memset(&obs, 0, sizeof(obs)); obs.hidden = model.hidden; obs.capacity = (size_t)model.layers * ((size_t)model.hidden * 12 + (size_t)model.intermediate * 4 + (size_t)model.qheads * capacity + 4096);
    obs.host = malloc(obs.capacity * 4); if (!obs.host) die("observer alloc", 0);
    float *host_logits = malloc((size_t)V * 4); if (!host_logits) die("logits alloc", 0);
    int maxsteps = steps > smoke_steps ? steps : smoke_steps; if (warmup_steps > maxsteps) maxsteps = warmup_steps;
    ArmRun runs[NARMS]; for (int a = 0; a < NARMS; a++) { runs[a].ids = malloc((size_t)(maxsteps + 1) * sizeof(int)); runs[a].ms = malloc((size_t)maxsteps * sizeof(double)); }
    fprintf(out, "{\"type\":\"header\",\"model_identity\":\"%s\",\"model_digest\":\"%s\",\"hidden\":%d,\"layers\":%d,\"vocab\":%d,\"kvheads\":%d,\"head\":%d,\"intermediate\":%d,\"gpu_weight_bytes\":%zu,\"gpu_session_bytes\":%zu,\"capacity\":%d,\"steps\":%d,\"prompts_in_file\":%d,\"load_s\":%.3f,\"arms\":[",
        model.identity, model.digest, model.hidden, model.layers, V, model.kvheads, model.head, model.intermediate, model.weight_bytes, s.allocated_bytes, capacity, steps, np, load_s);
    for (int a = 0; a < NARMS; a++) fprintf(out, "%s\"%s\"", a ? "," : "", ARMS[a].name);
    fprintf(out, "]}\n"); fflush(out);

    /* ---- smoke: first prompt, every arm, a few tokens; abort on any disagreement ---- */
    phase("c_smoke");
    qwen_reset(&s); for (int i = 0; i < lens[0]; i++) if (qwen_push(&s, prompts[0][i])) die("smoke prefill", i);
    for (int a = 0; a < NARMS; a++) {
        run_arm(&s, ARMS[a].mask, smoke_steps, &obs, host_logits, &runs[a]);
        emit_run(out, "smoke", 0, a, &ARMS[a], lens[0], &runs[a], smoke_steps, (ARMS[a].mask & A_OBSALL) ? 2 : (ARMS[a].mask & A_OBS) ? 1 : 0, 0.0);
    }
    int smoke_ok = 1; char why[256] = "";
    for (int a = 1; a < NARMS; a++) if (memcmp(runs[a].ids, runs[0].ids, (size_t)(smoke_steps + 1) * sizeof(int))) { smoke_ok = 0; snprintf(why, sizeof(why), "token ids of %s differ from C0", ARMS[a].name); }
    size_t want = (size_t)model.layers * model.hidden * 4 * smoke_steps;
    if (runs[2].obs_bytes_total != want) { smoke_ok = 0; snprintf(why, sizeof(why), "C2 captured %zu bytes, want %zu", runs[2].obs_bytes_total, want); }
    if (runs[3].obs_bytes_total <= want) { smoke_ok = 0; snprintf(why, sizeof(why), "C2b captured %zu bytes, not more than C2", runs[3].obs_bytes_total); }
    if (!runs[4].cache_sha[0] || !runs[4].logits_sha[0] || strcmp(runs[4].cache_sha, runs[5].cache_sha) || strcmp(runs[4].logits_sha, runs[5].logits_sha)) { smoke_ok = 0; snprintf(why, sizeof(why), "state hashes missing or C3 != C4"); }
    if (runs[1].logits_match_pending != 1 || runs[5].logits_match_pending != 1) { smoke_ok = 0; snprintf(why, sizeof(why), "downloaded logits do not match the chosen token"); }
    fprintf(out, "{\"type\":\"smoke_result\",\"ok\":%s,\"why\":\"%s\"}\n", smoke_ok ? "true" : "false", why); fflush(out);
    fprintf(stderr, "[smoke] ok=%d %s\n", smoke_ok, why);
    if (!smoke_ok) { fclose(out); return 3; }

    /* ---- warmup: every arm over the first prompt, untimed ---- */
    phase("c_warmup");
    for (int a = 0; a < NARMS; a++) run_arm(&s, ARMS[a].mask, warmup_steps, &obs, host_logits, &runs[a]);

    /* ---- main ---- */
    double main_started = mono(); int done = 0, truncated = 0;
    for (int p = 0; p < np; p++) {
        if (mono() - main_started > deadline) { truncated = 1; break; }
        phase("c_prefill");
        qwen_reset(&s); gpu_sync(); double a0 = mono();
        for (int i = 0; i < lens[p]; i++) if (qwen_push(&s, prompts[p][i])) die("prefill", i);
        gpu_sync(); double prefill_ms = (mono() - a0) * 1e3;
        for (int k = 0; k < NARMS; k++) {
            int a = (k + p) % NARMS; phase("%s", ARMS[a].name);
            run_arm(&s, ARMS[a].mask, steps, &obs, host_logits, &runs[a]);
            emit_run(out, "run", p, k, &ARMS[a], lens[p], &runs[a], steps, (ARMS[a].mask & A_OBSALL) ? 2 : (ARMS[a].mask & A_OBS) ? 1 : 0, prefill_ms);
        }
        done++;
    }
    phase("c_done");
    fprintf(out, "{\"type\":\"footer\",\"prompts_done\":%d,\"truncated\":%s,\"main_s\":%.3f,\"wall_s\":%.3f,\"peak_gpu_bytes\":%zu,\"live_gpu_bytes\":%zu}\n",
        done, truncated ? "true" : "false", mono() - main_started, mono() - started, gpu_peak_bytes(), gpu_live_bytes());
    fclose(out); qwen_session_free(&s); qwen_model_free(&model);
    return 0;
}
