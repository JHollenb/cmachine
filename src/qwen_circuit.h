#ifndef TCM_QWEN_CIRCUIT_H
#define TCM_QWEN_CIRCUIT_H
#include "qwen.h"
#include "circuit_math.h"
#include "circuit_expansions.h"
#include <stdio.h>
/* Temporary native operand cache. Original weights stay in QwenModel.
 * init/free ownership: zero initialize, do not shallow-copy, free before reuse.
 * Capture runs one ordinary token synchronously with exclusive session access. */
typedef struct {
    double *input,*receiver,*normalized,*gate,*up,*product,*attention,*output;
    double *query,*key,*value,*probability,*write;
    unsigned seen;
} CmQwenLayer;
typedef struct {
    const QwenSession *session;
    const QwenModel *model;
    uint64_t session_id,epoch,parent_lineage_id,result_lineage_id;
    int position,token,ready,error;
    CmQwenLayer *layer;
    float *transfer;
    size_t transfer_count,host_bytes;
} CmQwenCapture;
typedef struct {
    double *input,*receiver,*attention,*product,*gate,*up,*routing,*output;
} CmQwenLayerAdjoint;
typedef struct {
    int token,contrast,ready;
    double native_margin;
    const CmQwenCapture *capture;
    CmQwenLayerAdjoint *layer;
    size_t host_bytes;
} CmQwenAdjoint;
int cm_qwen_capture_token(CmQwenCapture *,QwenSession *,int);
void cm_qwen_capture_free(CmQwenCapture *);
/* Reverse smooth equations evaluated at actual native FP32 ports. Fixed prior
 * KV, all current-token Q/K/V paths, no derivative of rounding or argmax.
 * Original FP32 weight profile only; compact cache/programs refuse preflight. */
int cm_qwen_consumer(CmQwenAdjoint *,const CmQwenCapture *,int,int);
void cm_qwen_adjoint_free(CmQwenAdjoint *);
/* Original selected parameter row/column hydrated on demand. names:
 * gate,up,down,q,k,v,o,input_norm,post_norm. Norm vectors use index=0,row.
 * out_count must match full selected row/column; no truncation. */
int cm_qwen_parameter(const QwenModel *,int,const char *,size_t,int,double *,size_t);
int cm_qwen_writer_reader(const QwenModel *,int,size_t,int,const char *,size_t,const double *,double *,double *,double *);
/* Prompt-free pre-RoPE pairing and neuron read geometry from original rows. */
int cm_qwen_static_link(const QwenModel *,int,size_t,int,const char *,size_t,double *);
int cm_qwen_read_geometry(const QwenModel *,int,size_t,CmReadGeometry *);
typedef struct {
    double gate,up,interaction,remainder,total;
    double normalization_remainder,projection_remainder,secant_remainder;
} CmQwenFiniteSummary;
/* Full hidden-vector contributions, optional downstream scalar projection.
 * output [4*hidden]: gate,up,interaction,remainder. Captures must be from the
 * same model/clock, and remain live; differing parents are explicit. */
int cm_qwen_mlp_finite(const CmQwenCapture *,const CmQwenCapture *,int,const double *,double *,CmQwenFiniteSummary *);
int cm_qwen_mlp_secant_action(const CmQwenCapture *,const CmQwenCapture *,int,const double *,double *);
/* One selected head's actual return: [4*head] routing,payload,interaction,
 * numerical remainder. Same support/clock/model; other rows remain explicit. */
int cm_qwen_attention_finite(const CmQwenCapture *,const CmQwenCapture *,int,int,double *);
/* One receiver, source-axis correlation. Each complex coefficient is
 * conj(DFT(p))[k]*DFT(dot(lambda,V))[k]/N. Sum real coefficients equals the
 * ideal scalar read; remainder retains native return arithmetic. */
int cm_qwen_source_frequency(const CmQwenCapture *,const CmQwenAdjoint *,int,int,CmComplex *,size_t,double *,double *,double *);
/* Compact JSON metadata + bounded requested source/neuron observations.
 * No weight/activation array export. Consumer and occurrence links included. */
int cm_qwen_report(FILE *,const CmQwenCapture *,const CmQwenAdjoint *,int,int,int,int);
typedef struct { double A,B,C; } CmReadoutNorm;
/* Full vocabulary a,b for final state r+alpha*m: logits=(a+alpha*b)/sqrt(q).
 * Uses final-layer captured receiver/write, tied bias-free head and gain.
 * Caller owns arrays [vocab]; no persistent coefficient/weight bank. */
int cm_qwen_final_readout_lines(const CmQwenCapture *,double *,double *,CmReadoutNorm *);
int cm_qwen_write_readout_lines(FILE *,const CmQwenCapture *);
typedef struct { double signed_effect,squared_effect,omission_l2,adjoint_l2; } CmOmissionMetric;
/* Local predicted effect from a declared original port's full omission vector.
 * Supports input, output, receiver, attention, product, gate and up. */
int cm_qwen_omission(const CmQwenCapture *,const CmQwenAdjoint *,int,const char *,const double *,size_t,CmOmissionMetric *);
#endif
