/* Opt-in batched executor; see qwen_fast.h and docs/SPEEDUPS.md. Built with the
 * same nvcc flags as qwen_cuda.cu (--fmad=false), so the elementwise kernels
 * copied from the lean path produce the same bits. */
#include "gpu.h"
#include "qwen_fast.h"
#include "qwen_ledger.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(call) do { cudaError_t e_=(call); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)
#define BK(call) do { cublasStatus_t e_=(call); if (e_ != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cuBLAS %s:%d: status %d\n", __FILE__, __LINE__, (int)e_); exit(2); } } while (0)
#define LAUNCHED() CK(cudaGetLastError())

struct Mat { int in, out, n1, n2; const float *f; const __nv_bfloat16 *b; const int8_t *q; const float *s; };
enum { M_QKV, M_O, M_GATEUP, M_DOWN, M_COUNT };

struct QwenFast {
    QfWeights w;
    int max_seqs, cap, max_rows, max_steps, ids_stride, lane, tf32, lp_ready, fp32_freed;
    cudaStream_t st; cublasHandle_t blas; void *workspace;
    int *tok, *pos, *row_tok, *row_seq, *row_pos, *ids, *idsn, *invalid;
    float *X, *N, *Q, *K, *V, *A, *P, *G, *U, *logits, *keys, *values;
    int *hpos;
    size_t bytes;
    void *lp_alloc[8 * QF_MAX_LAYERS + 8]; int lp_count;
    Mat mb[QF_MAX_LAYERS][M_COUNT], mq[QF_MAX_LAYERS][M_COUNT], eb, eq;
    size_t lane_bytes[3];
    cudaGraph_t graph; cudaGraphExec_t exec; int graph_B;
    unsigned long long *inexact; unsigned long long bf16_inexact, bf16_total;
    /* Ledger (qwen_ledger.h). Device rows lg[max_rows][stride]; retained writes
     * lg_keep[max_rows][slots][hidden] (slot 0 = h_0, 1+2l = attention write of
     * layer l, 2+2l = MLP write of layer l); SwiGLU chunk partials and counters.
     * lg_rows = rows written by the last pass (host bookkeeping only). */
    int lg_on, lg_rows, lg_stride, lg_slots, lg_chunks, graph_lg, lg_skip;
    float *lg, *lg_keep, *lg_host; double *lg_part; unsigned *lg_count;
};

static void *fast_alloc(QwenFast *f, size_t n) { void *p = gpu_alloc(n); f->bytes += n; return p; }

/* ---------------- elementwise kernels (lean-path copies, per row) ---------------- */
__global__ static void embed_rows(float *X, const float *ef, const __nv_bfloat16 *eb, const int8_t *eq, const float *es, const int *row_tok, int d) {
    int r = blockIdx.x, t = row_tok[r];
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        size_t at = (size_t)t * d + i; float v;
        if (ef) v = ef[at]; else if (eb) v = __bfloat162float(eb[at]); else v = (float)eq[at] * es[t];
        X[(size_t)r * d + i] = v;
    }
}
/* rms_kernel of qwen_cuda.cu, one block per row, with the lean residual add
 * (x[i]+=y[i]) fused in front; each thread adds and then reads only its own elements. */
__global__ static void rms_rows(float *out, float *x0, const float *add0, const float *w, int n, float epsilon) {
    float *x = x0 + (size_t)blockIdx.x * n; float *o = out + (size_t)blockIdx.x * n;
    if (add0) { const float *y = add0 + (size_t)blockIdx.x * n; for (int i = threadIdx.x; i < n; i += blockDim.x) x[i] += y[i]; }
    __shared__ float sums[256]; float sum = 0; for (int i = threadIdx.x; i < n; i += blockDim.x) sum += x[i] * x[i]; sums[threadIdx.x] = sum; __syncthreads();
    for (int stride = 128; stride; stride /= 2) { if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride]; __syncthreads(); }
    float scale = rsqrtf(sums[0] / n + epsilon); for (int i = threadIdx.x; i < n; i += blockDim.x) o[i] = (x[i] * scale) * w[i];
}
__device__ static void rope_pair(float *x, int h, int dd, int dimension, int position, float theta) {
    int half = dimension / 2;
    float inverse = 1.0f / powf(theta, (2.0f * dd) / dimension); float angle = (float)position * inverse;
    float cosine = cosf(angle), sine = sinf(angle), a = x[h * dimension + dd], b = x[h * dimension + dd + half];
    x[h * dimension + dd] = a * cosine + (-b) * sine; x[h * dimension + dd + half] = b * cosine + a * sine;
}
/* Bias (gpu_bias), RoPE (rope_kernel) and cache write (cache_kernel) for one row. */
__global__ static void qkv_post(float *Q, float *K, float *V, const float *qb, const float *kb, const float *vb, float *keys, float *values,
                                const int *row_seq, const int *row_pos, int qheads, int kvheads, int head, int cap, float theta) {
    int r = blockIdx.y, t = blockIdx.x * blockDim.x + threadIdx.x, half = head / 2, d = qheads * head, k = kvheads * head;
    int nq = qheads * half, nk = kvheads * half;
    float *q = Q + (size_t)r * d, *kk = K + (size_t)r * k, *v = V + (size_t)r * k;
    int seq = row_seq[r], p = row_pos[r];
    if (t < nq) {
        int h = t / half, dd = t % half, a = h * head + dd;
        q[a] += qb[a]; q[a + half] += qb[a + half];
        rope_pair(q, h, dd, head, p, theta);
    } else if (t < nq + nk) {
        int u = t - nq, h = u / half, dd = u % half, a = h * head + dd;
        kk[a] += kb[a]; kk[a + half] += kb[a + half];
        rope_pair(kk, h, dd, head, p, theta);
        float *c = keys + (((size_t)seq * kvheads + h) * cap + p) * head;
        c[dd] = kk[a]; c[dd + half] = kk[a + half];
    } else if (t < nq + nk + k) {
        int e = t - nq - nk, h = e / head, dd = e % head;
        v[e] += vb[e];
        values[(((size_t)seq * kvheads + h) * cap + p) * head + dd] = v[e];
    }
}
/* Causal attention for one (row, query head): scores over positions 0..pos,
 * softmax with the lean expression order, then probability-weighted values.
 * blockDim = head = 128. Summation order differs from the lean cuBLAS path. */
__global__ static void attention_rows(float *out, const float *Q, const float *keys, const float *values, const int *row_seq, const int *row_pos,
                                      int qheads, int kvheads, int head, int cap, float scale) {
    extern __shared__ float sc[];
    __shared__ float qs[128], red[128];
    int r = blockIdx.x / qheads, h = blockIdx.x % qheads, kh = h / (qheads / kvheads), tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, warps = blockDim.x >> 5;
    int seq = row_seq[r], L = row_pos[r] + 1;
    const float *Kc = keys + ((size_t)seq * kvheads + kh) * cap * head, *Vc = values + ((size_t)seq * kvheads + kh) * cap * head;
    qs[tid] = Q[(size_t)r * qheads * head + h * head + tid]; __syncthreads();
    for (int j = warp; j < L; j += warps) {
        float dot = 0; for (int i = lane; i < head; i += 32) dot += qs[i] * Kc[(size_t)j * head + i];
        for (int off = 16; off; off >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, off);
        if (!lane) sc[j] = dot * scale;
    }
    __syncthreads();
    float m = -INFINITY; for (int j = tid; j < L; j += blockDim.x) m = fmaxf(m, sc[j]); red[tid] = m; __syncthreads();
    for (int stride = 64; stride; stride /= 2) { if (tid < stride) red[tid] = fmaxf(red[tid], red[tid + stride]); __syncthreads(); }
    m = red[0]; __syncthreads();
    float s = 0; for (int j = tid; j < L; j += blockDim.x) { float e = expf(sc[j] - m); sc[j] = e; s += e; } red[tid] = s; __syncthreads();
    for (int stride = 64; stride; stride /= 2) { if (tid < stride) red[tid] += red[tid + stride]; __syncthreads(); }
    s = red[0];
    for (int j = tid; j < L; j += blockDim.x) sc[j] /= s;
    __syncthreads();
    float acc = 0; for (int j = 0; j < L; j++) acc += sc[j] * Vc[(size_t)j * head + tid];
    out[(size_t)r * qheads * head + h * head + tid] = acc;
}
__global__ static void swiglu_rows(float *gate, const float *up, size_t n) { size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) gate[i] = (gate[i] / (1.0f + expf(-gate[i]))) * up[i]; }
__device__ static bool better(float a, int ai, float b, int bi) { return a > b || (a == b && ai < bi); }
/* Greedy id per logits row (ties -> smaller id, as gpu_top2's first), then
 * advance that row's sequence: pending token, position, recorded id. */
__global__ static void argmax_update(const float *logits, int V, const int *row_seq, const int *row_pos, int *tok, int *pos, int *ids, int *idsn, int stride, int *invalid) {
    __shared__ float bv[1024]; __shared__ int bi[1024];
    const float *x = logits + (size_t)blockIdx.x * V; float best = -INFINITY; int id = 2147483647, bad = 0;
    for (int i = threadIdx.x; i < V; i += blockDim.x) { float v = x[i]; if (!isfinite(v)) bad = 1; if (better(v, i, best, id)) { best = v; id = i; } }
    if (bad) atomicExch(invalid, 1);
    bv[threadIdx.x] = best; bi[threadIdx.x] = id; __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride /= 2) { if (threadIdx.x < stride && better(bv[threadIdx.x + stride], bi[threadIdx.x + stride], bv[threadIdx.x], bi[threadIdx.x])) { bv[threadIdx.x] = bv[threadIdx.x + stride]; bi[threadIdx.x] = bi[threadIdx.x + stride]; } __syncthreads(); }
    if (!threadIdx.x) {
        int seq = row_seq[blockIdx.x], n = idsn[seq];
        tok[seq] = bi[0]; pos[seq] = row_pos[blockIdx.x] + 1;
        if (n < stride) ids[(size_t)seq * stride + n] = bi[0];
        idsn[seq] = n + 1;
    }
}
__global__ static void prepare_decode(const int *tok, const int *pos, int *row_tok, int *row_seq, int *row_pos, int B) {
    int b = blockIdx.x * blockDim.x + threadIdx.x; if (b < B) { row_tok[b] = tok[b]; row_seq[b] = b; row_pos[b] = pos[b]; }
}

/* ---------------- ledger kernels (qwen_ledger.h); launched only while the ledger is on ----------------
 * The *_lg kernels repeat the native statements of rms_rows, attention_rows,
 * swiglu_rows and argmax_update in the same order with the same operands (the
 * file is built with --fmad=false), so every native output is bit-identical;
 * the ledger part only reads native values and accumulates in double. With the
 * ledger off, forward()/finish() launch the original kernels unchanged. */
#define LG_CHUNK 1024 /* SwiGLU neurons per ledger block */
__device__ static int lg_layer(int l, int qheads) { return QL_HEAD + l * (QL_LAYER + QL_PER_HEAD * qheads); }
/* Zero row r, then set token, position, sources. Readout fields stay 0 unless a readout runs. */
__global__ static void lg_row_init(float *lg, int stride, const int *row_tok, const int *row_pos) {
    float *row = lg + (size_t)blockIdx.x * stride;
    for (int i = threadIdx.x; i < stride; i += blockDim.x) row[i] = 0.0f;
    __syncthreads();
    if (!threadIdx.x) { int p = row_pos[blockIdx.x]; row[QL_TOKEN] = (float)row_tok[blockIdx.x]; row[QL_POSITION] = (float)p; row[QL_SOURCES] = (float)(p + 1); }
}
/* Ledger arguments of one rms_rows_lg launch. Offsets are within a row; -1 = not written.
 * keep: retained-write base (row r at keep + r*slots*n); keep_x retains x itself
 * (used for h_0 when there is no add); keep_add retains the added write; rows < keep_from are not retained. */
struct LgRms { float *row; int stride, rho, dot; float *keep; int slots, keep_x, keep_add, keep_from; };
__global__ static void rms_rows_lg(float *out, float *x0, const float *add0, const float *w, int n, float epsilon, LgRms g) {
    int r = blockIdx.x, t = threadIdx.x;
    float *x = x0 + (size_t)r * n; float *o = out + (size_t)r * n;
    bool keep = g.keep && r >= g.keep_from;
    float *kx = keep && g.keep_x >= 0 ? g.keep + ((size_t)r * g.slots + g.keep_x) * n : NULL;
    float *ka = keep && g.keep_add >= 0 ? g.keep + ((size_t)r * g.slots + g.keep_add) * n : NULL;
    double dot = 0, nn = 0;
    if (add0) {
        const float *y = add0 + (size_t)r * n;
        for (int i = t; i < n; i += blockDim.x) {
            float xv = x[i], yv = y[i];                          /* x = residual BEFORE the write is added */
            dot += (double)xv * (double)yv; nn += (double)yv * (double)yv;
            if (ka) ka[i] = yv;
            x[i] = xv + yv;                                      /* native: x[i] += y[i] */
        }
    } else if (kx) for (int i = t; i < n; i += blockDim.x) kx[i] = x[i];
    __shared__ float sums[256]; __shared__ double d1[256], d2[256];
    float sum = 0; for (int i = t; i < n; i += blockDim.x) sum += x[i] * x[i]; sums[t] = sum; d1[t] = dot; d2[t] = nn; __syncthreads();
    for (int stride = 128; stride; stride /= 2) { if (t < stride) { sums[t] += sums[t + stride]; d1[t] += d1[t + stride]; d2[t] += d2[t + stride]; } __syncthreads(); }
    float scale = rsqrtf(sums[0] / n + epsilon); for (int i = t; i < n; i += blockDim.x) o[i] = (x[i] * scale) * w[i];
    if (!t) {
        float *row = g.row + (size_t)r * g.stride;
        if (g.rho >= 0) row[g.rho] = scale;
        if (add0 && g.dot >= 0) { row[g.dot] = (float)d1[0]; row[g.dot + 1] = (float)d2[0]; }
    }
}
/* Last-layer MLP dot/norm2 for rows that get no final norm (non-final prefill rows):
 * same per-thread order and tree as rms_rows_lg, so a row gives the same values either way. */
__global__ static void lg_tail(const float *X, const float *P, int n, float *lg, int stride, int dot) {
    const float *x = X + (size_t)blockIdx.x * n, *y = P + (size_t)blockIdx.x * n; int t = threadIdx.x;
    double a = 0, b = 0;
    for (int i = t; i < n; i += blockDim.x) { float xv = x[i], yv = y[i]; a += (double)xv * (double)yv; b += (double)yv * (double)yv; }
    __shared__ double d1[256], d2[256]; d1[t] = a; d2[t] = b; __syncthreads();
    for (int s = 128; s; s /= 2) { if (t < s) { d1[t] += d1[t + s]; d2[t] += d2[t + s]; } __syncthreads(); }
    if (!t) { float *row = lg + (size_t)blockIdx.x * stride; row[dot] = (float)d1[0]; row[dot + 1] = (float)d2[0]; }
}
/* attention_rows plus the per-head softmax fields, taken inside the softmax pass:
 * logZ = max + log(sum of the native exp terms, in double); probabilities are the native ones. */
__global__ static void attention_rows_lg(float *out, const float *Q, const float *keys, const float *values, const int *row_seq, const int *row_pos,
                                         int qheads, int kvheads, int head, int cap, float scale, float *lg, int lstride, int hoff) {
    extern __shared__ float sc[];
    __shared__ float qs[128], red[128];
    __shared__ double rd[128]; __shared__ int ri[128];
    int r = blockIdx.x / qheads, h = blockIdx.x % qheads, kh = h / (qheads / kvheads), tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, warps = blockDim.x >> 5;
    int seq = row_seq[r], L = row_pos[r] + 1;
    const float *Kc = keys + ((size_t)seq * kvheads + kh) * cap * head, *Vc = values + ((size_t)seq * kvheads + kh) * cap * head;
    qs[tid] = Q[(size_t)r * qheads * head + h * head + tid]; __syncthreads();
    for (int j = warp; j < L; j += warps) {
        float dot = 0; for (int i = lane; i < head; i += 32) dot += qs[i] * Kc[(size_t)j * head + i];
        for (int off = 16; off; off >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, off);
        if (!lane) sc[j] = dot * scale;
    }
    __syncthreads();
    float m = -INFINITY; for (int j = tid; j < L; j += blockDim.x) m = fmaxf(m, sc[j]); red[tid] = m; __syncthreads();
    for (int stride = 64; stride; stride /= 2) { if (tid < stride) red[tid] = fmaxf(red[tid], red[tid + stride]); __syncthreads(); }
    m = red[0]; __syncthreads();
    float s = 0; double sd = 0;
    for (int j = tid; j < L; j += blockDim.x) { float e = expf(sc[j] - m); sc[j] = e; s += e; sd += (double)e; }
    red[tid] = s; rd[tid] = sd; __syncthreads();
    for (int stride = 64; stride; stride /= 2) { if (tid < stride) { red[tid] += red[tid + stride]; rd[tid] += rd[tid + stride]; } __syncthreads(); }
    s = red[0]; double z = rd[0];
    for (int j = tid; j < L; j += blockDim.x) sc[j] /= s;
    __syncthreads();
    /* every thread has read red[0]/rd[0] before the barrier above; reuse them */
    double ent = 0; float pm = -1.0f; int pi = 2147483647; /* entropy terms in FP32 (logf), summed in FP64 */
    for (int j = tid; j < L; j += blockDim.x) { float p = sc[j]; if (p > 0.0f) ent -= (double)(p * logf(p)); if (p > pm) { pm = p; pi = j; } }
    rd[tid] = ent; red[tid] = pm; ri[tid] = pi; __syncthreads();
    for (int stride = 64; stride; stride /= 2) {
        if (tid < stride) {
            rd[tid] += rd[tid + stride];
            float b = red[tid + stride]; int bj = ri[tid + stride];
            if (b > red[tid] || (b == red[tid] && bj < ri[tid])) { red[tid] = b; ri[tid] = bj; }
        }
        __syncthreads();
    }
    if (!tid) {
        float *o = lg + (size_t)r * lstride + hoff + h;
        o[QL_H_LOGZ * qheads] = (float)((double)m + log(z)); o[QL_H_PFIRST * qheads] = sc[0]; o[QL_H_ENTROPY * qheads] = (float)rd[0];
        o[QL_H_PMAX * qheads] = red[0]; o[QL_H_ARGMAX * qheads] = (float)ri[0];
    }
    float acc = 0; for (int j = 0; j < L; j++) acc += sc[j] * Vc[(size_t)j * head + tid];
    out[(size_t)r * qheads * head + h * head + tid] = acc;
}
/* swiglu_rows plus the gate census, grid (chunks, rows). Per-element terms are FP32
 * (tanhf; FP64 tanh per neuron cost 0.43 ms/token at B=1 and 2.8 ms/step at B=64,
 * job-8d3e1f797eac); each thread sums its <= LG_CHUNK/256 elements in FP32, then the
 * block tree and the cross-block sum are FP64. The last block of a row (threadfence +
 * counter) sums the chunk partials in chunk order, writes the fields, resets the counter. */
__global__ static void swiglu_rows_lg(float *gate, const float *up, int h, int chunks, double *part, unsigned *count, float *lg, int lstride, int loff) {
    int r = blockIdx.y, c = blockIdx.x, t = threadIdx.x;
    float *g0 = gate + (size_t)r * h; const float *u0 = up + (size_t)r * h;
    int lo = c * LG_CHUNK, hi = lo + LG_CHUNK < h ? lo + LG_CHUNK : h;
    float v[7] = {0, 0, 0, 0, 0, 0, 0}; /* small, off, on, even2, odd2, even_odd, prod2 */
    for (int i = lo + t; i < hi; i += blockDim.x) {
        float g = g0[i], u = u0[i];
        float prod = (g / (1.0f + expf(-g))) * u;                /* native: gate[i] = (gate[i] / (1.0f + expf(-gate[i]))) * up[i] */
        g0[i] = prod;
        if (fabsf(g) < 1.0f) v[0] += 1.0f;
        if (g <= -5.0f) v[1] += 1.0f;
        if (g >= 5.0f) v[2] += 1.0f;
        float even = (g * u) * 0.5f, odd = even * tanhf(g * 0.5f);
        v[3] += even * even; v[4] += odd * odd; v[5] += even * odd; v[6] += prod * prod;
    }
    __shared__ double red[7][256]; __shared__ int last;
    for (int k = 0; k < 7; k++) red[k][t] = (double)v[k];
    __syncthreads();
    for (int s = 128; s; s /= 2) { if (t < s) for (int k = 0; k < 7; k++) red[k][t] += red[k][t + s]; __syncthreads(); }
    if (!t) {
        double *p = part + ((size_t)r * chunks + c) * 8;
        for (int k = 0; k < 7; k++) p[k] = red[k][0];
        __threadfence();
        last = atomicAdd(count + r, 1u) == (unsigned)(chunks - 1);
    }
    __syncthreads();
    if (last && !t) {
        __threadfence();
        volatile const double *p = part + (size_t)r * chunks * 8; double sum[7] = {0, 0, 0, 0, 0, 0, 0};
        for (int cc = 0; cc < chunks; cc++) for (int k = 0; k < 7; k++) sum[k] += p[(size_t)cc * 8 + k];
        float *o = lg + (size_t)r * lstride + loff;
        o[QL_GATE_SMALL] = (float)sum[0]; o[QL_GATE_OFF] = (float)sum[1]; o[QL_GATE_ON] = (float)sum[2];
        o[QL_EVEN2] = (float)sum[3]; o[QL_ODD2] = (float)sum[4]; o[QL_EVEN_ODD] = (float)sum[5]; o[QL_PROD2] = (float)sum[6];
        count[r] = 0;
    }
}
/* gpu_top2's insert rule (ties -> smaller id; the runner-up is a different id). */
__device__ static void lg_insert(float v, int i, float &a, int &ai, float &b, int &bi) {
    if (better(v, i, a, ai)) { b = a; bi = ai; a = v; ai = i; } else if (i != ai && better(v, i, b, bi)) { b = v; bi = i; }
}
/* argmax_update with the runner-up tracked alongside. The winner is the unique
 * maximum of the same total order (value, then smaller id), so it equals
 * argmax_update's id; the sequence update statements are the same. */
__global__ static void argmax_update_lg(const float *logits, int V, const int *row_seq, const int *row_pos, int *tok, int *pos, int *ids, int *idsn, int stride, int *invalid,
                                        float *lg, int lstride) {
    __shared__ float bv[1024], sv[1024]; __shared__ int bi[1024], si[1024];
    const float *x = logits + (size_t)blockIdx.x * V; float best = -INFINITY, second = -INFINITY; int id = 2147483647, id2 = 2147483647, bad = 0, t = threadIdx.x;
    for (int i = t; i < V; i += blockDim.x) { float v = x[i]; if (!isfinite(v)) bad = 1; lg_insert(v, i, best, id, second, id2); }
    if (bad) atomicExch(invalid, 1);
    bv[t] = best; bi[t] = id; sv[t] = second; si[t] = id2; __syncthreads();
    for (int s = blockDim.x / 2; s; s /= 2) {
        if (t < s) {
            float a = bv[t], b = sv[t]; int ai = bi[t], bj = si[t];
            lg_insert(bv[t + s], bi[t + s], a, ai, b, bj); lg_insert(sv[t + s], si[t + s], a, ai, b, bj);
            bv[t] = a; bi[t] = ai; sv[t] = b; si[t] = bj;
        }
        __syncthreads();
    }
    if (!t) {
        int seq = row_seq[blockIdx.x], n = idsn[seq];
        tok[seq] = bi[0]; pos[seq] = row_pos[blockIdx.x] + 1;
        if (n < stride) ids[(size_t)seq * stride + n] = bi[0];
        idsn[seq] = n + 1;
        float *row = lg + (size_t)blockIdx.x * lstride;
        row[QL_HAS_READOUT] = 1.0f; row[QL_TOP1] = (float)bi[0]; row[QL_TOP2] = (float)si[0];
        row[QL_MARGIN] = (float)((double)bv[0] - (double)sv[0]);
    }
}
/* Margin split for one readout row: c = gamma (.) (W_a - W_b) in double from the
 * resident tied rows (FP32, or BF16 widened exactly); one warp per retained write
 * (plus ||c||^2), thread 0 sums in fixed order: h_0, then attention and MLP of each layer. */
__global__ static void lg_margin(float *lg, int lstride, const float *keep, int slots, int d, int layers, int qheads, int V,
                                 const float *gamma, const float *Wf, const __nv_bfloat16 *Wb) {
    extern __shared__ double c[];
    __shared__ double res[2 * QF_MAX_LAYERS + 2];
    int r = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5, warps = blockDim.x >> 5;
    float *row = lg + (size_t)r * lstride;
    int a = (int)row[QL_TOP1], b = (int)row[QL_TOP2];
    if (a < 0 || a >= V || b < 0 || b >= V) return; /* same for every thread of the block */
    for (int i = t; i < d; i += blockDim.x) {
        double wa = Wf ? (double)Wf[(size_t)a * d + i] : (double)__bfloat162float(Wb[(size_t)a * d + i]);
        double wb = Wf ? (double)Wf[(size_t)b * d + i] : (double)__bfloat162float(Wb[(size_t)b * d + i]);
        c[i] = (double)gamma[i] * (wa - wb);
    }
    __syncthreads();
    const float *k0 = keep + (size_t)r * slots * d;
    for (int task = warp; task <= slots; task += warps) {
        double acc = 0;
        if (task < slots) { const float *wv = k0 + (size_t)task * d; for (int i = lane; i < d; i += 32) acc += c[i] * (double)wv[i]; }
        else for (int i = lane; i < d; i += 32) acc += c[i] * c[i];
        for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (!lane) res[task] = acc;
    }
    __syncthreads();
    if (!t) {
        double rho = (double)row[QL_RHO_FINAL], emb = rho * res[0], sum = emb;
        row[QL_C_NORM] = (float)sqrt(res[slots]); row[QL_MARGIN_EMBED] = (float)emb;
        for (int l = 0; l < layers; l++) {
            double ma = rho * res[1 + 2 * l], mm = rho * res[2 + 2 * l];
            float *blk = row + lg_layer(l, qheads);
            blk[QL_MARGIN_ATTN] = (float)ma; blk[QL_MARGIN_MLP] = (float)mm;
            sum += ma; sum += mm;
        }
        row[QL_MARGIN_SUM] = (float)sum;
    }
}

/* ---------------- low-precision matvec: one warp per output row ---------------- */
template <int KIND>
__global__ static void wmv(const void *W, const float *S, const float *X, int in, int out, int nb, int n1, int n2, float *Y0, float *Y1, float *Y2) {
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= out) return;
    float acc[8];
#pragma unroll
    for (int b = 0; b < 8; b++) acc[b] = 0;
    if (KIND == QF_BF16) {
        const uint4 *w4 = (const uint4 *)((const __nv_bfloat16 *)W + (size_t)row * in);
        for (int k = lane; k < in / 8; k += 32) {
            uint4 u = w4[k]; unsigned words[4] = {u.x, u.y, u.z, u.w}; float wf[8];
#pragma unroll
            for (int j = 0; j < 4; j++) { wf[2 * j] = __uint_as_float(words[j] << 16); wf[2 * j + 1] = __uint_as_float(words[j] & 0xffff0000u); }
#pragma unroll
            for (int b = 0; b < 8; b++) if (b < nb) {
                const float4 *x4 = (const float4 *)(X + (size_t)b * in) + 2 * k; float4 a = x4[0], c = x4[1];
                acc[b] += ((wf[0] * a.x + wf[1] * a.y) + (wf[2] * a.z + wf[3] * a.w)) + ((wf[4] * c.x + wf[5] * c.y) + (wf[6] * c.z + wf[7] * c.w));
            }
        }
    } else {
        const uint4 *w4 = (const uint4 *)((const int8_t *)W + (size_t)row * in);
        for (int k = lane; k < in / 16; k += 32) {
            uint4 u = w4[k]; unsigned words[4] = {u.x, u.y, u.z, u.w}; float wf[16];
#pragma unroll
            for (int j = 0; j < 4; j++) for (int z = 0; z < 4; z++) wf[4 * j + z] = (float)(signed char)((words[j] >> (8 * z)) & 0xffu);
#pragma unroll
            for (int b = 0; b < 8; b++) if (b < nb) {
                const float4 *x4 = (const float4 *)(X + (size_t)b * in) + 4 * k; float part = 0;
#pragma unroll
                for (int z = 0; z < 4; z++) { float4 a = x4[z]; part += (wf[4 * z] * a.x + wf[4 * z + 1] * a.y) + (wf[4 * z + 2] * a.z + wf[4 * z + 3] * a.w); }
                acc[b] += part;
            }
        }
    }
#pragma unroll
    for (int b = 0; b < 8; b++) for (int off = 16; off; off >>= 1) acc[b] += __shfl_down_sync(0xffffffffu, acc[b], off);
    if (lane) return;
    float sc = KIND == QF_INT8 ? S[row] : 1.0f;
#pragma unroll
    for (int b = 0; b < 8; b++) if (b < nb) {
        float v = acc[b] * sc;
        if (row < n1) Y0[(size_t)b * n1 + row] = v;
        else if (row < n2) Y1[(size_t)b * (n2 - n1) + row - n1] = v;
        else Y2[(size_t)b * (out - n2) + row - n2] = v;
    }
}
__global__ static void to_bf16(__nv_bfloat16 *dst, const float *src, size_t n, unsigned long long *inexact) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    __nv_bfloat16 b = __float2bfloat16_rn(src[i]); dst[i] = b;
    if (__float_as_uint(__bfloat162float(b)) != __float_as_uint(src[i])) atomicAdd(inexact, 1ull);
}
/* Symmetric per-row int8: scale = max|w|/127, code = rint(w/scale) clamped to +-127. */
__global__ static void quant_rows(int8_t *codes, float *scales, const float *src, int in) {
    __shared__ float red[256]; const float *x = src + (size_t)blockIdx.x * in; float m = 0;
    for (int i = threadIdx.x; i < in; i += blockDim.x) m = fmaxf(m, fabsf(x[i]));
    red[threadIdx.x] = m; __syncthreads();
    for (int s = 128; s; s /= 2) { if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]); __syncthreads(); }
    float scale = red[0] / 127.0f;
    for (int i = threadIdx.x; i < in; i += blockDim.x) { float c = scale > 0 ? rintf(x[i] / scale) : 0.0f; c = fminf(127.0f, fmaxf(-127.0f, c)); codes[(size_t)blockIdx.x * in + i] = (int8_t)c; }
    if (!threadIdx.x) scales[blockIdx.x] = scale;
}

/* ---------------- engine ---------------- */
extern "C" QwenFast *qf_open(const QfWeights *w, int max_seqs, int capacity, int max_rows, int max_steps) {
    if (w->layers > QF_MAX_LAYERS || w->head != 128 || capacity * 4 > 48 * 1024 || max_rows < 1 || max_seqs < 1) return NULL;
    QwenFast *f = (QwenFast *)calloc(1, sizeof(*f)); if (!f) exit(2);
    f->w = *w; f->max_seqs = max_seqs; f->cap = capacity; f->max_rows = max_rows < max_seqs ? max_seqs : max_rows; f->max_steps = max_steps; f->ids_stride = max_steps + 1;
    CK(cudaStreamCreateWithFlags(&f->st, cudaStreamNonBlocking));
    BK(cublasCreate(&f->blas)); BK(cublasSetStream(f->blas, f->st)); BK(cublasSetMathMode(f->blas, CUBLAS_PEDANTIC_MATH));
    size_t ws = (size_t)32 << 20; f->workspace = fast_alloc(f, ws); BK(cublasSetWorkspace(f->blas, f->workspace, ws));
    int d = w->hidden, k = w->kvheads * w->head, h = w->intermediate, R = f->max_rows, S = max_seqs;
    f->tok = (int *)fast_alloc(f, S * 4); f->pos = (int *)fast_alloc(f, S * 4); f->idsn = (int *)fast_alloc(f, S * 4);
    f->ids = (int *)fast_alloc(f, (size_t)S * f->ids_stride * 4); f->invalid = (int *)fast_alloc(f, 4);
    f->row_tok = (int *)fast_alloc(f, R * 4); f->row_seq = (int *)fast_alloc(f, R * 4); f->row_pos = (int *)fast_alloc(f, R * 4);
#define FA(field, n) f->field = (float *)fast_alloc(f, (size_t)(n) * 4)
    FA(X, (size_t)R * d); FA(N, (size_t)R * d); FA(Q, (size_t)R * d); FA(K, (size_t)R * k); FA(V, (size_t)R * k); FA(A, (size_t)R * d); FA(P, (size_t)R * d);
    FA(G, (size_t)R * h); FA(U, (size_t)R * h); FA(logits, (size_t)R * w->vocab);
    size_t kv = (size_t)w->layers * S * k * capacity; FA(keys, kv); FA(values, kv);
#undef FA
    f->hpos = (int *)calloc(S, sizeof(int)); if (!f->hpos) exit(2);
    size_t mat = 0, vec = 0;
    mat += (size_t)w->vocab * d; vec += d;
    for (int l = 0; l < w->layers; l++) { mat += (size_t)d * d * 2 + (size_t)k * d * 2 + (size_t)h * d * 3; vec += (size_t)d * 3 + 2 * k; }
    size_t rows = (size_t)w->vocab + (size_t)w->layers * (d + 2 * k + d + 2 * h + d);
    f->lane_bytes[QF_FP32] = (mat + vec) * 4; f->lane_bytes[QF_BF16] = mat * 2 + vec * 4; f->lane_bytes[QF_INT8] = mat + rows * 4 + vec * 4;
    qf_reset(f); CK(cudaStreamSynchronize(f->st));
    return f;
}
extern "C" size_t qf_device_bytes(const QwenFast *f) { return f->bytes; }
extern "C" size_t qf_lane_weight_bytes(const QwenFast *f, int lane) { return f->lane_bytes[lane]; }
extern "C" void qf_graph_free(QwenFast *f) { if (f->exec) CK(cudaGraphExecDestroy(f->exec)); if (f->graph) CK(cudaGraphDestroy(f->graph)); f->exec = NULL; f->graph = NULL; f->graph_B = 0; f->graph_lg = 0; }
extern "C" void qf_close(QwenFast *f) {
    if (!f) return; qf_graph_free(f);
    void *all[] = {f->workspace, f->tok, f->pos, f->idsn, f->ids, f->invalid, f->row_tok, f->row_seq, f->row_pos, f->X, f->N, f->Q, f->K, f->V, f->A, f->P, f->G, f->U, f->logits, f->keys, f->values,
                   f->lg, f->lg_keep, f->lg_part, f->lg_count};
    for (unsigned i = 0; i < sizeof(all) / sizeof(all[0]); i++) gpu_free(all[i]);
    if (f->lg_host) CK(cudaFreeHost(f->lg_host));
    for (int i = 0; i < f->lp_count; i++) gpu_free(f->lp_alloc[i]);
    BK(cublasDestroy(f->blas)); CK(cudaStreamDestroy(f->st)); free(f->hpos); free(f);
}
extern "C" void qf_sync(QwenFast *f) { CK(cudaStreamSynchronize(f->st)); }
extern "C" void qf_reset(QwenFast *f) {
    CK(cudaMemsetAsync(f->pos, 0, f->max_seqs * 4, f->st)); CK(cudaMemsetAsync(f->idsn, 0, f->max_seqs * 4, f->st));
    CK(cudaMemsetAsync(f->tok, 0, f->max_seqs * 4, f->st)); CK(cudaMemsetAsync(f->invalid, 0, 4, f->st));
    memset(f->hpos, 0, f->max_seqs * sizeof(int));
    f->lg_rows = 0;
}
extern "C" int qf_position(const QwenFast *f, int seq) { return f->hpos[seq]; }
extern "C" void qf_set_tf32(QwenFast *f, int on) { f->tf32 = on; BK(cublasSetMathMode(f->blas, on ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_PEDANTIC_MATH)); }
extern "C" int qf_set_lane(QwenFast *f, int lane) {
    if (lane < QF_FP32 || lane > QF_INT8) return -1;
    if (lane == QF_FP32 && f->fp32_freed) return -1;
    if (lane != QF_FP32 && !f->lp_ready) return -1;
    if (lane == QF_INT8 && f->lg_on) return -1; /* the ledger covers exact-in-weights lanes only */
    if (lane != f->lane) qf_graph_free(f);
    f->lane = lane; return 0;
}

static void mm(QwenFast *f, const Mat &m, const float *X, int R, float *Y0, float *Y1, float *Y2) {
    const float one = 1, zero = 0;
    if (f->lane == QF_FP32) {
        if (R == 1) BK(cublasSgemv(f->blas, CUBLAS_OP_T, m.in, m.out, &one, m.f, m.in, X, 1, &zero, Y0, 1));
        else BK(cublasSgemm(f->blas, CUBLAS_OP_T, CUBLAS_OP_N, m.out, R, m.in, &one, m.f, m.in, X, m.in, &zero, Y0, m.out));
        return;
    }
    int blocks = (m.out + 7) / 8;
    for (int c = 0; c < R; c += 8) {
        int nb = R - c < 8 ? R - c : 8; const float *x = X + (size_t)c * m.in;
        float *y0 = Y0 + (size_t)c * m.n1, *y1 = Y1 ? Y1 + (size_t)c * (m.n2 - m.n1) : NULL, *y2 = Y2 ? Y2 + (size_t)c * (m.out - m.n2) : NULL;
        if (f->lane == QF_BF16) wmv<QF_BF16><<<blocks, 256, 0, f->st>>>(m.b, NULL, x, m.in, m.out, nb, m.n1, m.n2, y0, y1, y2);
        else wmv<QF_INT8><<<blocks, 256, 0, f->st>>>(m.q, m.s, x, m.in, m.out, nb, m.n1, m.n2, y0, y1, y2);
        LAUNCHED();
    }
}
static Mat single(const float *p, int in, int out) { Mat m; memset(&m, 0, sizeof(m)); m.in = in; m.out = out; m.n1 = m.n2 = out; m.f = p; return m; }

/* keep_from: with the ledger on, rows >= keep_from retain their writes for the margin split
 * (decode: 0; batched prefill: the last row only). Ignored with the ledger off. */
static void forward(QwenFast *f, int R, int keep_from) {
    const QfWeights &w = f->w; int d = w.hidden, k = w.kvheads * w.head, h = w.intermediate;
    const int lg = f->lg_on, H = w.qheads, ls = f->lg_stride, slots = f->lg_slots;
    /* profiling only (qf_ledger_profile_skip): a skipped piece runs the native kernel instead */
    const int lg_rms = lg && !(f->lg_skip & QF_SKIP_RMS), lg_att = lg && !(f->lg_skip & QF_SKIP_ATTENTION), lg_glu = lg && !(f->lg_skip & QF_SKIP_SWIGLU);
    if (lg) { lg_row_init<<<R, 256, 0, f->st>>>(f->lg, ls, f->row_tok, f->row_pos); LAUNCHED(); f->lg_rows = R; }
    const Mat *emb = f->lane == QF_BF16 ? &f->eb : f->lane == QF_INT8 ? &f->eq : NULL;
    embed_rows<<<R, 256, 0, f->st>>>(f->X, emb ? NULL : w.embedding, f->lane == QF_BF16 ? f->eb.b : NULL, f->lane == QF_INT8 ? f->eq.q : NULL, f->eq.s, f->row_tok, d); LAUNCHED();
    float scale = 1.0f / sqrtf((float)w.head);
    size_t layer_kv = (size_t)f->max_seqs * k * f->cap;
    int post_threads = w.qheads * (w.head / 2) + w.kvheads * (w.head / 2) + k;
    for (int l = 0; l < w.layers; l++) {
        int lay = lg ? ql_layer_offset(l, H) : 0;
        if (lg_rms) {
            /* layer l input norm: adds the MLP write of layer l-1 (slot 2l); layer 0 retains h_0 (slot 0) */
            LgRms g = {f->lg, ls, lay + QL_RHO_IN, l ? ql_layer_offset(l - 1, H) + QL_MLP_DOT : -1, f->lg_keep, slots, l ? -1 : 0, l ? 2 * l : -1, keep_from};
            rms_rows_lg<<<R, 256, 0, f->st>>>(f->N, f->X, l ? f->P : NULL, w.input_norm[l], d, w.epsilon, g); LAUNCHED();
        } else { rms_rows<<<R, 256, 0, f->st>>>(f->N, f->X, l ? f->P : NULL, w.input_norm[l], d, w.epsilon); LAUNCHED(); }
        if (f->lane == QF_FP32) { mm(f, single(w.q[l], d, d), f->N, R, f->Q, NULL, NULL); mm(f, single(w.k[l], d, k), f->N, R, f->K, NULL, NULL); mm(f, single(w.v[l], d, k), f->N, R, f->V, NULL, NULL); }
        else mm(f, (f->lane == QF_BF16 ? f->mb : f->mq)[l][M_QKV], f->N, R, f->Q, f->K, f->V);
        float *keys = f->keys + l * layer_kv, *values = f->values + l * layer_kv;
        qkv_post<<<dim3((post_threads + 127) / 128, R), 128, 0, f->st>>>(f->Q, f->K, f->V, w.qb[l], w.kb[l], w.vb[l], keys, values, f->row_seq, f->row_pos, w.qheads, w.kvheads, w.head, f->cap, w.theta); LAUNCHED();
        if (lg_att) { attention_rows_lg<<<R * w.qheads, w.head, (size_t)f->cap * 4, f->st>>>(f->A, f->Q, keys, values, f->row_seq, f->row_pos, w.qheads, w.kvheads, w.head, f->cap, scale, f->lg, ls, lay + QL_LAYER); LAUNCHED(); }
        else { attention_rows<<<R * w.qheads, w.head, (size_t)f->cap * 4, f->st>>>(f->A, f->Q, keys, values, f->row_seq, f->row_pos, w.qheads, w.kvheads, w.head, f->cap, scale); LAUNCHED(); }
        if (f->lane == QF_FP32) mm(f, single(w.o[l], d, d), f->A, R, f->P, NULL, NULL); else mm(f, (f->lane == QF_BF16 ? f->mb : f->mq)[l][M_O], f->A, R, f->P, NULL, NULL);
        if (lg_rms) {
            /* post-attention norm: adds the attention write of layer l (slot 1+2l) */
            LgRms g = {f->lg, ls, lay + QL_RHO_MID, lay + QL_ATTN_DOT, f->lg_keep, slots, -1, 1 + 2 * l, keep_from};
            rms_rows_lg<<<R, 256, 0, f->st>>>(f->N, f->X, f->P, w.post_norm[l], d, w.epsilon, g); LAUNCHED();
        } else { rms_rows<<<R, 256, 0, f->st>>>(f->N, f->X, f->P, w.post_norm[l], d, w.epsilon); LAUNCHED(); }
        if (f->lane == QF_FP32) { mm(f, single(w.gate[l], d, h), f->N, R, f->G, NULL, NULL); mm(f, single(w.up[l], d, h), f->N, R, f->U, NULL, NULL); }
        else mm(f, (f->lane == QF_BF16 ? f->mb : f->mq)[l][M_GATEUP], f->N, R, f->G, f->U, NULL);
        if (lg_glu) { swiglu_rows_lg<<<dim3(f->lg_chunks, R), 256, 0, f->st>>>(f->G, f->U, h, f->lg_chunks, f->lg_part, f->lg_count, f->lg, ls, lay); LAUNCHED(); }
        else { size_t n = (size_t)R * h; swiglu_rows<<<(unsigned)((n + 255) / 256), 256, 0, f->st>>>(f->G, f->U, n); LAUNCHED(); }
        if (f->lane == QF_FP32) mm(f, single(w.down[l], h, d), f->G, R, f->P, NULL, NULL); else mm(f, (f->lane == QF_BF16 ? f->mb : f->mq)[l][M_DOWN], f->G, R, f->P, NULL, NULL);
    }
}
/* Ledger, rows 0..count-1 that get no final norm: the last layer's MLP write is never
 * added to X, so take its dot/norm2 directly. */
static void lg_no_readout(QwenFast *f, int count) {
    if (!f->lg_on || count < 1) return;
    lg_tail<<<count, 256, 0, f->st>>>(f->X, f->P, f->w.hidden, f->lg, f->lg_stride, ql_layer_offset(f->w.layers - 1, f->w.qheads) + QL_MLP_DOT); LAUNCHED();
}
/* Final norm (with the last layer's residual add) and logits for rows first..first+count-1. */
static void finish(QwenFast *f, int first, int count) {
    const QfWeights &w = f->w; int d = w.hidden;
    const int lg = f->lg_on && !(f->lg_skip & QF_SKIP_READOUT);
    if (lg) {
        lg_no_readout(f, first);
        int ls = f->lg_stride, slots = f->lg_slots;
        LgRms g = {f->lg + (size_t)first * ls, ls, QL_RHO_FINAL, ql_layer_offset(w.layers - 1, w.qheads) + QL_MLP_DOT, f->lg_keep + (size_t)first * slots * d, slots, -1, 2 * w.layers, 0};
        rms_rows_lg<<<count, 256, 0, f->st>>>(f->N, f->X + (size_t)first * d, f->P + (size_t)first * d, w.norm, d, w.epsilon, g); LAUNCHED();
    } else { rms_rows<<<count, 256, 0, f->st>>>(f->N, f->X + (size_t)first * d, f->P + (size_t)first * d, w.norm, d, w.epsilon); LAUNCHED(); }
    if (f->lane == QF_FP32) mm(f, single(w.embedding, d, w.vocab), f->N, count, f->logits, NULL, NULL);
    else mm(f, f->lane == QF_BF16 ? f->eb : f->eq, f->N, count, f->logits, NULL, NULL);
    if (lg) {
        int ls = f->lg_stride, slots = f->lg_slots;
        argmax_update_lg<<<count, 1024, 0, f->st>>>(f->logits, w.vocab, f->row_seq + first, f->row_pos + first, f->tok, f->pos, f->ids, f->idsn, f->ids_stride, f->invalid, f->lg + (size_t)first * ls, ls); LAUNCHED();
        lg_margin<<<count, 256, (size_t)d * 8, f->st>>>(f->lg + (size_t)first * ls, ls, f->lg_keep + (size_t)first * slots * d, slots, d, w.layers, w.qheads, w.vocab,
                                                       w.norm, f->lane == QF_FP32 ? w.embedding : NULL, f->lane == QF_BF16 ? f->eb.b : NULL); LAUNCHED();
    } else { argmax_update<<<count, 1024, 0, f->st>>>(f->logits, w.vocab, f->row_seq + first, f->row_pos + first, f->tok, f->pos, f->ids, f->idsn, f->ids_stride, f->invalid); LAUNCHED(); }
}
extern "C" int qf_prefill(QwenFast *f, int seq, const int *tokens, int n, int batched) {
    if (seq < 0 || seq >= f->max_seqs || n < 1 || f->hpos[seq] + n > f->cap || (batched && n > f->max_rows)) return -1;
    int p0 = f->hpos[seq];
    if (batched) {
        int *rt = (int *)malloc(n * 3 * sizeof(int)); if (!rt) exit(2); int *rs = rt + n, *rp = rt + 2 * n;
        for (int i = 0; i < n; i++) { rt[i] = tokens[i]; rs[i] = seq; rp[i] = p0 + i; }
        CK(cudaMemcpyAsync(f->row_tok, rt, n * 4, cudaMemcpyHostToDevice, f->st)); CK(cudaMemcpyAsync(f->row_seq, rs, n * 4, cudaMemcpyHostToDevice, f->st)); CK(cudaMemcpyAsync(f->row_pos, rp, n * 4, cudaMemcpyHostToDevice, f->st));
        forward(f, n, n - 1); finish(f, n - 1, 1);
        CK(cudaStreamSynchronize(f->st)); free(rt);
    } else {
        for (int i = 0; i < n; i++) {
            int row[3] = {tokens[i], seq, p0 + i};
            CK(cudaMemcpyAsync(f->row_tok, row, 4, cudaMemcpyHostToDevice, f->st)); CK(cudaMemcpyAsync(f->row_seq, row + 1, 4, cudaMemcpyHostToDevice, f->st)); CK(cudaMemcpyAsync(f->row_pos, row + 2, 4, cudaMemcpyHostToDevice, f->st));
            forward(f, 1, i == n - 1 ? 0 : 1);
            if (i == n - 1) finish(f, 0, 1); else lg_no_readout(f, 1);
            CK(cudaStreamSynchronize(f->st));
        }
    }
    f->hpos[seq] = p0 + n; return 0;
}
static int decode_check(QwenFast *f, int B) {
    if (B < 1 || B > f->max_seqs || B > f->max_rows) return -1;
    for (int b = 0; b < B; b++) if (f->hpos[b] < 1 || f->hpos[b] + 1 > f->cap) return -1;
    return 0;
}
static void decode_body(QwenFast *f, int B) {
    prepare_decode<<<(B + 63) / 64, 64, 0, f->st>>>(f->tok, f->pos, f->row_tok, f->row_seq, f->row_pos, B); LAUNCHED();
    forward(f, B, 0); finish(f, 0, B);
}
extern "C" int qf_decode(QwenFast *f, int B, const int *forced) {
    if (decode_check(f, B)) return -1;
    if (forced) CK(cudaMemcpyAsync(f->tok, forced, B * 4, cudaMemcpyHostToDevice, f->st));
    decode_body(f, B);
    for (int b = 0; b < B; b++) f->hpos[b]++;
    return 0;
}
extern "C" int qf_graph_capture(QwenFast *f, int B) {
    if (B < 1 || B > f->max_seqs || B > f->max_rows) return -1;
    qf_graph_free(f); CK(cudaStreamSynchronize(f->st));
    CK(cudaStreamBeginCapture(f->st, cudaStreamCaptureModeThreadLocal));
    decode_body(f, B);
    CK(cudaStreamEndCapture(f->st, &f->graph));
    CK(cudaGraphInstantiateWithFlags(&f->exec, f->graph, 0));
    f->graph_B = B; f->graph_lg = f->lg_on; return 0;
}
extern "C" int qf_graph_step(QwenFast *f) {
    if (!f->exec || decode_check(f, f->graph_B)) return -1;
    CK(cudaGraphLaunch(f->exec, f->st));
    for (int b = 0; b < f->graph_B; b++) f->hpos[b]++;
    if (f->graph_lg) f->lg_rows = f->graph_B;
    return 0;
}
extern "C" int qf_read_ids(QwenFast *f, int seq, int *out, int n) {
    int invalid = 0, have = 0;
    CK(cudaStreamSynchronize(f->st));
    CK(cudaMemcpy(&invalid, f->invalid, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&have, f->idsn + seq, 4, cudaMemcpyDeviceToHost));
    if (invalid) { fprintf(stderr, "qwen_fast: non-finite logits\n"); return -2; }
    if (n > have || n > f->ids_stride) return -1;
    CK(cudaMemcpy(out, f->ids + (size_t)seq * f->ids_stride, (size_t)n * 4, cudaMemcpyDeviceToHost)); return 0;
}
extern "C" void qf_read_logits(QwenFast *f, int row, float *host) { CK(cudaStreamSynchronize(f->st)); CK(cudaMemcpy(host, f->logits + (size_t)row * f->w.vocab, (size_t)f->w.vocab * 4, cudaMemcpyDeviceToHost)); }

/* ---------------- ledger API (qwen_ledger.h) ---------------- */
extern "C" int qf_ledger_row_floats(const QwenFast *f) { return ql_row_floats(f->w.layers, f->w.qheads); }
extern "C" int qf_ledger_enable(QwenFast *f, int on) {
    if (!f) return -1;
    on = on ? 1 : 0;
    if (on) {
        const QfWeights &w = f->w;
        if (f->lane == QF_INT8) return -1;                                   /* refused: approximate lane */
        if ((size_t)f->cap * 4 + 4096 > 48 * 1024 || (size_t)w.hidden * 8 > 40 * 1024 || w.head != 128) return -1; /* shared-memory limits of the _lg kernels */
        if (!f->lg) {
            int R = f->max_rows;
            f->lg_stride = ql_row_floats(w.layers, w.qheads); f->lg_slots = 2 * w.layers + 1; f->lg_chunks = (w.intermediate + LG_CHUNK - 1) / LG_CHUNK;
            f->lg = (float *)fast_alloc(f, (size_t)R * f->lg_stride * 4);
            f->lg_keep = (float *)fast_alloc(f, (size_t)R * f->lg_slots * w.hidden * 4);
            f->lg_part = (double *)fast_alloc(f, (size_t)R * f->lg_chunks * 8 * sizeof(double));
            f->lg_count = (unsigned *)fast_alloc(f, (size_t)R * sizeof(unsigned));
            CK(cudaMemsetAsync(f->lg, 0, (size_t)R * f->lg_stride * 4, f->st));
            CK(cudaMemsetAsync(f->lg_keep, 0, (size_t)R * f->lg_slots * w.hidden * 4, f->st));
            CK(cudaMemsetAsync(f->lg_count, 0, (size_t)R * sizeof(unsigned), f->st));
            CK(cudaMallocHost((void **)&f->lg_host, (size_t)R * f->lg_stride * 4));
            CK(cudaStreamSynchronize(f->st));
        }
    }
    if (on != f->lg_on) qf_graph_free(f); /* a captured graph holds the other kernel set: recapture */
    f->lg_on = on; f->lg_rows = 0; return 0;
}
/* Rows first..first+count-1 of the last pass, one device-to-host copy (pinned staging). */
extern "C" int qf_ledger_read_rows(QwenFast *f, int first, int count, float *host) {
    if (!f || !host || !f->lg_on || first < 0 || count < 1 || first + count > f->lg_rows) return -1;
    size_t n = (size_t)count * f->lg_stride * 4;
    CK(cudaMemcpyAsync(f->lg_host, f->lg + (size_t)first * f->lg_stride, n, cudaMemcpyDeviceToHost, f->st));
    CK(cudaStreamSynchronize(f->st));
    memcpy(host, f->lg_host, n); return 0;
}
extern "C" int qf_ledger_read(QwenFast *f, int row, float *host_row) { return qf_ledger_read_rows(f, row, 1, host_row); }
extern "C" void qf_ledger_profile_skip(QwenFast *f, int mask) { if (mask != f->lg_skip) qf_graph_free(f); f->lg_skip = mask; }

/* ---------------- low-precision weight build ---------------- */
static void *lp_alloc(QwenFast *f, size_t n) { void *p = fast_alloc(f, n); f->lp_alloc[f->lp_count++] = p; return p; }
static void convert_into(QwenFast *f, const float *src, int rows, int in, __nv_bfloat16 *b, int8_t *q, float *s) {
    size_t n = (size_t)rows * in;
    to_bf16<<<(unsigned)((n + 255) / 256), 256, 0, f->st>>>(b, src, n, f->inexact); LAUNCHED(); f->bf16_total += n;
    quant_rows<<<rows, 256, 0, f->st>>>(q, s, src, in); LAUNCHED();
}
static void build_mat(QwenFast *f, Mat *mb, Mat *mq, int in, const float *const *src, const int *rows, int parts) {
    int out = 0; for (int i = 0; i < parts; i++) out += rows[i];
    __nv_bfloat16 *b = (__nv_bfloat16 *)lp_alloc(f, (size_t)out * in * 2); int8_t *q = (int8_t *)lp_alloc(f, (size_t)out * in); float *s = (float *)lp_alloc(f, (size_t)out * 4);
    int at = 0; for (int i = 0; i < parts; i++) { convert_into(f, src[i], rows[i], in, b + (size_t)at * in, q + (size_t)at * in, s + at); at += rows[i]; }
    Mat m; memset(&m, 0, sizeof(m)); m.in = in; m.out = out; m.n1 = rows[0]; m.n2 = parts == 3 ? rows[0] + rows[1] : out;
    *mb = m; mb->b = b; *mq = m; mq->q = q; mq->s = s;
}
extern "C" int qf_build_low_precision(QwenFast *f, void (*free_matrix)(void *, float *), void *ctx) {
    if (f->lp_ready) return 0;
    f->inexact = (unsigned long long *)fast_alloc(f, 8); CK(cudaMemsetAsync(f->inexact, 0, 8, f->st));
    QfWeights &w = f->w; int d = w.hidden, k = w.kvheads * w.head, h = w.intermediate;
    for (int l = 0; l < w.layers; l++) {
        const float *qkv[] = {w.q[l], w.k[l], w.v[l]}; int qkv_rows[] = {d, k, k};
        build_mat(f, &f->mb[l][M_QKV], &f->mq[l][M_QKV], d, qkv, qkv_rows, 3);
        const float *o[] = {w.o[l]}; int o_rows[] = {d}; build_mat(f, &f->mb[l][M_O], &f->mq[l][M_O], d, o, o_rows, 1);
        const float *gu[] = {w.gate[l], w.up[l]}; int gu_rows[] = {h, h}; build_mat(f, &f->mb[l][M_GATEUP], &f->mq[l][M_GATEUP], d, gu, gu_rows, 2);
        const float *dn[] = {w.down[l]}; int dn_rows[] = {d}; build_mat(f, &f->mb[l][M_DOWN], &f->mq[l][M_DOWN], h, dn, dn_rows, 1);
        if (free_matrix) {
            CK(cudaStreamSynchronize(f->st));
            float **owned[] = {&w.q[l], &w.k[l], &w.v[l], &w.o[l], &w.gate[l], &w.up[l], &w.down[l]};
            for (int i = 0; i < 7; i++) { free_matrix(ctx, *owned[i]); *owned[i] = NULL; }
        }
    }
    const float *e[] = {w.embedding}; int e_rows[] = {w.vocab}; build_mat(f, &f->eb, &f->eq, d, e, e_rows, 1);
    CK(cudaStreamSynchronize(f->st));
    CK(cudaMemcpy(&f->bf16_inexact, f->inexact, 8, cudaMemcpyDeviceToHost)); gpu_free(f->inexact); f->bytes -= 8; f->inexact = NULL;
    if (free_matrix) { free_matrix(ctx, w.embedding); w.embedding = NULL; f->fp32_freed = 1; }
    f->lp_ready = 1; qf_graph_free(f); return 0;
}

/* ---------------- hashing helpers ---------------- */
__global__ static void gather_kernel(float *dst, const float *keys, const float *values, int layers, int kvheads, int cap, int head, int position) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x, per = (size_t)layers * kvheads * head; if (i >= 2 * per) return;
    int kind = (int)(i / per); size_t rem = i % per; int l = (int)(rem / ((size_t)kvheads * head)), hh = (int)((rem / head) % kvheads), e = (int)(rem % head);
    dst[i] = (kind ? values : keys)[(((size_t)l * kvheads + hh) * cap + position) * head + e];
}
extern "C" void qf_gather_rows(float *dst, const float *keys, const float *values, int layers, int kvheads, int capacity, int head, int position) {
    size_t n = (size_t)2 * layers * kvheads * head; gather_kernel<<<(unsigned)((n + 255) / 256), 256>>>(dst, keys, values, layers, kvheads, capacity, head, position); LAUNCHED();
}
__constant__ static uint32_t SHA_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
__device__ static uint32_t rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }
__device__ static void sha_block(uint32_t hs[8], const uint32_t in[16]) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) w[i] = in[i];
    for (int i = 16; i < 64; i++) { uint32_t x = w[i - 15], y = w[i - 2]; w[i] = w[i - 16] + (rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)) + w[i - 7] + (rotr(y, 17) ^ rotr(y, 19) ^ (y >> 10)); }
    uint32_t a = hs[0], b = hs[1], c = hs[2], d = hs[3], e = hs[4], f = hs[5], g = hs[6], v = hs[7];
    for (int i = 0; i < 64; i++) {
        uint32_t t = v + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + SHA_K[i] + w[i];
        uint32_t u = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        v = g; g = f; f = e; e = d + t; d = c; c = b; b = a; a = t + u;
    }
    hs[0] += a; hs[1] += b; hs[2] += c; hs[3] += d; hs[4] += e; hs[5] += f; hs[6] += g; hs[7] += v;
}
__global__ static void sha_leaves_kernel(const unsigned char *data, size_t bytes, size_t leaf, unsigned char *out, size_t leaves) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= leaves) return;
    size_t start = i * leaf, len = bytes - start < leaf ? bytes - start : leaf;
    const uint32_t *p = (const uint32_t *)(data + start);
    uint32_t hs[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19}, blk[16];
    for (size_t b = 0; b < len / 64; b++) { for (int j = 0; j < 16; j++) blk[j] = __byte_perm(p[b * 16 + j], 0, 0x0123); sha_block(hs, blk); }
    uint64_t bits = (uint64_t)len * 8; blk[0] = 0x80000000u; for (int j = 1; j < 14; j++) blk[j] = 0; blk[14] = (uint32_t)(bits >> 32); blk[15] = (uint32_t)bits; sha_block(hs, blk);
    for (int j = 0; j < 8; j++) for (int z = 0; z < 4; z++) out[i * 32 + 4 * j + z] = (unsigned char)(hs[j] >> (24 - 8 * z));
}
extern "C" void qf_sha_leaves(const void *data, size_t bytes, size_t leaf, unsigned char *out) {
    if (bytes % 64 || leaf % 64 || !leaf) { fprintf(stderr, "qf_sha_leaves: sizes must be multiples of 64\n"); exit(2); }
    size_t leaves = (bytes + leaf - 1) / leaf;
    sha_leaves_kernel<<<(unsigned)((leaves + 127) / 128), 128>>>((const unsigned char *)data, bytes, leaf, out, leaves); LAUNCHED();
}
extern "C" void qf_bf16_exactness(const QwenFast *f, unsigned long long *inexact, unsigned long long *total) { *inexact = f->bf16_inexact; *total = f->bf16_total; }
