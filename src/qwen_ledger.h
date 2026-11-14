#ifndef TCM_QWEN_LEDGER_H
#define TCM_QWEN_LEDGER_H
/* Per-token scalar ledger: quantities the forward pass already computes (or can
 * compute with a few short dot products) and normally throws away.
 *
 * CONTRACT
 *  1. Opt-in. With the ledger off, nothing about the executor changes.
 *  2. With the ledger on, every native value (K/V, residual, logits, ids) is
 *     bit-identical to the same path with the ledger off. The ledger only reads.
 *  3. No extra learned-matrix product. Extra work is scalar reductions inside
 *     existing kernels, short dot products of width `hidden`, and one readback
 *     per token (lean) or per pass (fast engine). No sync inside the layer loop.
 *  4. One ledger row per consumed (sequence, position) row. A row is a flat
 *     float32 array; integer fields are stored as floats (all are < 2^24).
 *  5. Reductions that are not native values are summed in double on the device
 *     (block trees and cross-block sums) and stored as float. Dots, norms and
 *     the margin split are double throughout; logZ sums its exp terms in double
 *     (lean: double exp; fast engine: the native FP32 exp values). The per-element terms of the
 *     SwiGLU census (even, odd with tanhf, their products, and prod^2 in the fast
 *     engine) and of the attention entropy (p*logf(p)) are FP32; the fast engine
 *     also sums at most 4 census elements per thread in FP32 before the double tree.
 *
 * ROW LAYOUT   (L = layers, H = query heads)
 *   row   = QL_HEAD floats, then L layer blocks
 *   block = QL_LAYER floats, then 5*H per-head floats
 *
 * Row head:
 *   QL_TOKEN        consumed token id
 *   QL_POSITION     position of this row (0-based)
 *   QL_SOURCES      visible source rows = position + 1
 *   QL_HAS_READOUT  1 if a vocabulary readout was executed for this row, else 0.
 *                   All readout fields below are 0 when this is 0.
 *   QL_TOP1,QL_TOP2 greedy winner a and runner-up b (native tie rule)
 *   QL_MARGIN       native logit[a] - logit[b]
 *   QL_RHO_FINAL    final RMS scale rho_F = rsqrt(mean(x^2)+eps), native FP32 value
 *   QL_C_NORM       ||c||, c = Gamma_F (.) (W_a - W_b)   (final-norm gain times row difference)
 *                   flip distance in normalized state = QL_MARGIN / QL_C_NORM
 *   QL_MARGIN_EMBED rho_F * c . h_0          (h_0 = token embedding)
 *   QL_MARGIN_SUM   QL_MARGIN_EMBED + sum over layers of margin_attn + margin_mlp.
 *                   Equals QL_MARGIN up to arithmetic; QL_MARGIN - QL_MARGIN_SUM is the
 *                   closure remainder and must be reported, not hidden.
 *
 * Layer block scalars:
 *   QL_RHO_IN       RMS scale of the attention-input norm, native FP32 value
 *   QL_RHO_MID      RMS scale of the MLP-input norm, native FP32 value
 *   QL_ATTN_DOT     x . w_attn   (x = residual BEFORE the attention write is added)
 *   QL_ATTN_NORM2   ||w_attn||^2   (w_attn = the write that is actually added)
 *   QL_MLP_DOT      r . w_mlp    (r = residual BEFORE the MLP write is added)
 *   QL_MLP_NORM2    ||w_mlp||^2
 *                   => exact change of the next norm's mean square:
 *                      d*(q1-q0) = 2*dot + norm2   (bible ch.13 cross term; cm_rms_finite)
 *   QL_MARGIN_ATTN  rho_F * c . w_attn       (0 without a readout)
 *   QL_MARGIN_MLP   rho_F * c . w_mlp        (0 without a readout)
 *   QL_GATE_SMALL   count of MLP neurons with |g| < 1        (g = gate pre-activation)
 *   QL_GATE_OFF     count with g <= -5
 *   QL_GATE_ON      count with g >=  5
 *   QL_EVEN2        sum_i (g_i u_i / 2)^2                     (bilinear half of SiLU(g)u)
 *   QL_ODD2         sum_i (g_i u_i tanh(g_i/2) / 2)^2         (the half that holds the nonlinearity)
 *   QL_EVEN_ODD     sum_i even_i * odd_i
 *   QL_PROD2        sum_i (SiLU(g_i) u_i)^2  using the native product values
 *                   (neuron space, before the down map; even+odd = product up to arithmetic)
 *
 * Per-head arrays (each H long, in this order), s_j = native scaled score of source j:
 *   QL_H_LOGZ       log sum_j exp(s_j) = max + log(sum exp(s - max))
 *   QL_H_PFIRST     native probability of source row 0
 *   QL_H_ENTROPY    -sum_j p_j log p_j  (nats; 0 log 0 = 0). exp() of it = effective rows
 *   QL_H_PMAX       largest native probability
 *   QL_H_ARGMAX     its source index (smallest index on ties)
 */
#ifdef __cplusplus
extern "C" {
#endif

enum { QL_TOKEN = 0, QL_POSITION, QL_SOURCES, QL_HAS_READOUT, QL_TOP1, QL_TOP2, QL_MARGIN, QL_RHO_FINAL,
       QL_C_NORM, QL_MARGIN_EMBED, QL_MARGIN_SUM, QL_HEAD = 16 };
enum { QL_RHO_IN = 0, QL_RHO_MID, QL_ATTN_DOT, QL_ATTN_NORM2, QL_MLP_DOT, QL_MLP_NORM2, QL_MARGIN_ATTN, QL_MARGIN_MLP,
       QL_GATE_SMALL, QL_GATE_OFF, QL_GATE_ON, QL_EVEN2, QL_ODD2, QL_EVEN_ODD, QL_PROD2, QL_LAYER = 16 };
enum { QL_H_LOGZ = 0, QL_H_PFIRST, QL_H_ENTROPY, QL_H_PMAX, QL_H_ARGMAX, QL_PER_HEAD = 5 };

static inline int ql_block_floats(int qheads) { return QL_LAYER + QL_PER_HEAD * qheads; }
static inline int ql_row_floats(int layers, int qheads) { return QL_HEAD + layers * ql_block_floats(qheads); }
static inline int ql_layer_offset(int layer, int qheads) { return QL_HEAD + layer * ql_block_floats(qheads); }
static inline int ql_head_offset(int layer, int qheads, int field, int head) { return ql_layer_offset(layer, qheads) + QL_LAYER + field * qheads + head; }

/* Lean path (src/qwen.c). Enable before qwen_begin_token/qwen_push of the token
 * to be recorded. qwen_ledger_read copies the row of the last finished token
 * (valid after qwen_finish_token, until the next token begins); returns 0 on
 * success, -1 if the ledger is off or no finished row is available.
 * Replacement paths (plane, subspace, writer, gate law, lexical/compiled
 * programs, compact cache backends) refuse to enable or record nothing: the
 * ledger describes the original equations only. */
struct QwenSession;
int qwen_ledger_enable(struct QwenSession *s, int on);
int qwen_ledger_read(struct QwenSession *s, float *host_row);

/* Fast engine (src/qwen_fast.cu). Enable before qf_graph_capture. Rows of the
 * last pass: decode row b = sequence b; batched prefill row r = r-th token of
 * the block (only the last one has a readout). Returns 0 on success. */
struct QwenFast;
int qf_ledger_enable(struct QwenFast *f, int on);
int qf_ledger_read(struct QwenFast *f, int row, float *host_row);

#ifdef __cplusplus
}
#endif
#endif
