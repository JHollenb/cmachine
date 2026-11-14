#ifndef TCM_QWEN_FAST_H
#define TCM_QWEN_FAST_H
/* Opt-in batched executor for Qwen2.5-1.5B (docs/SPEEDUPS.md). It reads the lean
 * model's device weights and never changes the lean qwen_push/qwen_advance path.
 *
 * Rows are (sequence, position, token) triples. One forward pass runs any set of
 * rows through all layers with one weight sweep: batched prefill = T rows of one
 * sequence; batched decode = one row per sequence. Each sequence owns its K/V.
 * Tokens, positions and greedy ids live on the device, so a decode step needs no
 * host round trip and can be captured once as a CUDA graph.
 *
 * Numerics: FP32 lane with one row uses cublasSgemv exactly like the lean path;
 * several rows use cublasSgemm (different summation order). Attention is a custom
 * kernel (different summation order from the lean cuBLAS attention). RMSNorm,
 * bias, RoPE, SwiGLU and residual adds reproduce the lean kernels bit for bit.
 * Opt-in TF32 tensor-op math applies to the sgemm (several-row) calls only.
 * BF16 and INT8 lanes are approximate: matrices are converted from the FP32
 * weights (BF16 round-to-nearest-even; INT8 symmetric per-row absmax/127 codes
 * with an FP32 row scale), activations stay FP32, accumulation is FP32. */
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define QF_MAX_LAYERS 64
typedef struct {
    int hidden, layers, qheads, kvheads, head, intermediate, vocab;
    float epsilon, theta;
    float *embedding, *norm;
    float *input_norm[QF_MAX_LAYERS], *post_norm[QF_MAX_LAYERS];
    float *q[QF_MAX_LAYERS], *k[QF_MAX_LAYERS], *v[QF_MAX_LAYERS], *qb[QF_MAX_LAYERS], *kb[QF_MAX_LAYERS], *vb[QF_MAX_LAYERS];
    float *o[QF_MAX_LAYERS], *gate[QF_MAX_LAYERS], *up[QF_MAX_LAYERS], *down[QF_MAX_LAYERS];
} QfWeights;
enum { QF_FP32 = 0, QF_BF16 = 1, QF_INT8 = 2 };
typedef struct QwenFast QwenFast;
/* max_rows bounds both the prefill block length and the decode batch. */
QwenFast *qf_open(const QfWeights *w, int max_seqs, int capacity, int max_rows, int max_steps);
void qf_close(QwenFast *);
size_t qf_device_bytes(const QwenFast *);
/* Weight bytes one decode step streams in a lane (matrices + FP32 vectors/scales). */
size_t qf_lane_weight_bytes(const QwenFast *, int lane);
/* Build BF16 and INT8 copies of every matrix (fused QKV and gate/up layouts).
 * free_fp32 != 0 frees each FP32 matrix after conversion through free_matrix;
 * afterwards only the BF16/INT8 lanes can run. */
int qf_build_low_precision(QwenFast *, void (*free_matrix)(void *ctx, float *p), void *ctx);
/* Matrix elements whose FP32 value is not exactly representable in BF16 (counted during the build). */
void qf_bf16_exactness(const QwenFast *, unsigned long long *inexact, unsigned long long *total);
int qf_set_lane(QwenFast *, int lane);
void qf_set_tf32(QwenFast *, int on);
void qf_reset(QwenFast *);
/* Prefill n tokens of sequence seq (starting at its current position).
 * batched=1: one pass of n rows; batched=0: n one-row passes. The greedy token
 * after the last row becomes the sequence's pending token and ids[0]. */
int qf_prefill(QwenFast *, int seq, const int *tokens, int n, int batched);
/* One decode step for sequences 0..B-1. forced (host, B ids) overrides the
 * input tokens first (teacher forcing); NULL feeds each sequence its own greedy id. */
int qf_decode(QwenFast *, int B, const int *forced);
int qf_graph_capture(QwenFast *, int B);
int qf_graph_step(QwenFast *);
void qf_graph_free(QwenFast *);
void qf_sync(QwenFast *);
/* ids[seq][0..n): the greedy id after prefill, then one per decode step. */
int qf_read_ids(QwenFast *, int seq, int *out, int n);
/* Logits rows of the last final stage: decode row b = sequence b; prefill row 0. */
void qf_read_logits(QwenFast *, int row, float *host);
int qf_position(const QwenFast *, int seq);

/* Hashing helpers (incremental state digest, docs/SPEEDUPS.md). */
/* Copy K then V row `position` of every layer/kv-head of a lean session cache
 * ([layer][kvhead][capacity][head]) into dst (2*layers*kvheads*head floats),
 * in the order of the full-cache SHA: kind, layer, head. */
void qf_gather_rows(float *dst, const float *keys, const float *values, int layers, int kvheads, int capacity, int head, int position);
/* SHA-256 of each `leaf`-byte leaf of a device buffer (bytes % 64 == 0,
 * leaf % 64 == 0); writes ceil(bytes/leaf) 32-byte digests to device out. */
void qf_sha_leaves(const void *data, size_t bytes, size_t leaf, unsigned char *out);
#ifdef __cplusplus
}
#endif
#endif
