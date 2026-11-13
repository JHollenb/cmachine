#ifndef TCM_QWEN_H
#define TCM_QWEN_H
#include "machine.h"
#include "gpu.h"

typedef struct {
    float *input_norm,*post_norm,*q,*k,*v,*qb,*kb,*vb,*o,*gate,*up,*down;
} QwenLayer;
typedef struct {
    TcmGpu *gpu;
    int hidden,layers,qheads,kvheads,head,intermediate,vocab,max_position,eos,eos_extra;
    float epsilon,theta;
    char identity[65],digest[65];
    float *embedding,*norm;
    QwenLayer *layer;
    float **allocations;
    unsigned allocation_count;
    size_t weight_bytes;
} QwenModel;
/* Explicit immutable, model/layer-qualified operand programs. Inputs are the
 * actual learned RMS-normalized receiver; original bias and RoPE follow once.
 * Callers retain program lifetime until every attached session is detached. */
enum { QWEN_QKV_Q=1,QWEN_QKV_K=2,QWEN_QKV_V=4 };
typedef struct {
    const QwenModel *model;
    int layer,rank,mask,ready;
    float *basis,*image[3];
    size_t gpu_bytes;
    char fingerprint[65];
} QwenQkvProgram;
typedef struct {
    uint64_t applications,code_calls,compiled_calls,native_calls,linear_macs;
    uint64_t lexical_hits,lexical_transfer_bytes;
} QwenOperandWork;
/* Finite token-local first-layer program. Values include original biases and
 * exclude RoPE. Missing IDs refuse before token mutation; no online fill. */
typedef struct {
    const QwenModel *model;
    size_t rows,host_bytes;
    int *ids;
    float *values;
    int ready;
    uint64_t compile_linear_macs;
    char fingerprint[65];
} QwenLexicalProgram;
typedef struct {
    char model_digest[65];
    int position,pending,capacity,delivered;
    TcmTop2 top;
    uint64_t serial,session_id,epoch,lineage_id;
} QwenCheckpoint;
typedef struct { QwenCheckpoint state; uint64_t id; } QwenFrame;
#define QWEN_STACK_LIMIT 64
typedef enum {
    QWEN_INPUT,QWEN_QUERY,QWEN_KEY,QWEN_VALUE,QWEN_ATTENTION,
    QWEN_MLP_INPUT,QWEN_MLP_GATE,QWEN_MLP_UP,QWEN_MLP_RETURN,QWEN_OUTPUT,
    QWEN_ATTENTION_WRITE,QWEN_RAW_RECEIVER,
    QWEN_CONSUMED_ATTENTION_WRITE,QWEN_CONSUMED_MLP_WRITE,
    QWEN_ATTENTION_PROBABILITIES,QWEN_MLP_PRODUCT,
    QWEN_PORT_COUNT
} QwenPort;
/* Observers may read the device pointer during this call only. They are not cuts
 * or mutation callbacks. No observer, transfer or synchronization runs by default.
 * An attached observer with observe_layer=-1 sees every layer at its selected
 * position. Probability rows are packed [qheads,position+1], native cache only. */
typedef void (*QwenObserver)(void *,int,int,QwenPort,const float *,size_t);
/* Explicit cache backend. A selected layer publishes its current K/V and reads
 * its own memory before returning attention. Guards run before begin/restore;
 * rewind restores backend-owned bytes. Callers own the context lifetime. */
typedef int (*QwenCacheStep)(void *,int,int,const float *,const float *,const float *,float *,float *);
typedef int (*QwenCacheGuard)(void *,int,int,uint64_t,uint64_t,uint64_t);
typedef void (*QwenCacheRewind)(void *,int,uint64_t,uint64_t);
typedef struct {
    const QwenModel *model;
    int position,capacity,pending,layer,phase,delivered;
    float *keys,*values,*x,*norm,*q,*k,*v,*attention,*projected,*gate,*up,*logits,*scores;
    size_t allocated_bytes;
    TcmTop2 top;
    uint64_t checkpoint_serial;
    uint64_t session_id;
    int checkpoint_position;
    uint64_t epoch;
    uint64_t lineage_id; /* Identity of the complete token parent, not its length. */
    unsigned depth;
    QwenFrame frames[QWEN_STACK_LIMIT];
    QwenObserver observer;
    void *observer_context;
    int observe_position,observe_layer;
    uint64_t cache_mask;
    int cache_slots[64],dense_layers;
    QwenCacheStep cache_step;
    QwenCacheRewind cache_rewind;
    QwenCacheGuard cache_guard;
    void *cache_context;
    QwenCheckpoint token_parent;
    int token_parent_valid;
    const QwenQkvProgram *qkv_programs[64];
    const QwenLexicalProgram *lexical_program;
    float *qkv_code;
    size_t qkv_code_capacity;
    int current_token;
    QwenOperandWork operand_work; /* Physical work: deliberately not rewound. */
} QwenSession;
int qwen_qkv_program_init(QwenQkvProgram *,const QwenModel *,int,int,int,const float *,const float *,const float *,const float *);
void qwen_qkv_program_free(QwenQkvProgram *);
int qwen_qkv_bind(QwenSession *,int,const QwenQkvProgram *);
int qwen_lexical_compile(QwenLexicalProgram *,const QwenModel *,const int *,size_t);
void qwen_lexical_free(QwenLexicalProgram *);
int qwen_lexical_bind(QwenSession *,const QwenLexicalProgram *);
int qwen_lexical_save(const QwenLexicalProgram *,const char *);
int qwen_lexical_load(QwenLexicalProgram *,const QwenModel *,const char *);
/* A frozen program for one model/session/epoch/token/layer and exactly guarded
 * live operands. The numerical basis is immutable after compilation. rho is
 * execution scratch, excluded from the frozen program. No policy enters cuts. */
typedef struct {
    const QwenModel *model;
    uint64_t session_id,epoch;
    int position,layer,ready;
    float *basis[3],*gate[3],*up[3],*baseline_write,*rho;
    size_t allocated_bytes;
} QwenPlane;
typedef enum { QWEN_PLANE_DENSE,QWEN_PLANE_COMPILED,QWEN_PLANE_HELD_WRITE } QwenPlaneMode;
typedef struct { unsigned gate_up_matvecs,down_matvecs,compile_matvecs; } QwenPlaneWork;
int qwen_plane_init(QwenPlane *,const QwenModel *);
void qwen_plane_free(QwenPlane *);
int qwen_advance_layer_compile(QwenSession *,QwenPlane *,QwenPlaneWork *);
int qwen_advance_layer_plane(QwenSession *,const QwenPlane *,QwenPlaneMode,float,float,QwenPlaneWork *);
/* Separate approximate live-receiver experiment. Frozen basis/QR/images bind
 * model+layer, not a token/session. Mutable workspace is excluded from hashes. */
typedef struct {
    const QwenModel *model;
    int ready,layer,source_position,rank;
    float *basis[3],*gate[3],*up[3],*baseline_write;
    double *q0,*q1,*q2,r00,r01,r11,condition,r02,r12,r22;
    float *entry,*attention_write,*receiver,*reconstruction,*omitted,*carry_omitted,*read_complement,*correction_operand,*rho,*correction_gate,*correction_up,*host;
    size_t gpu_bytes,host_bytes;
} QwenSubspace;
typedef enum {
    QWEN_SUBSPACE_COMPILE,QWEN_SUBSPACE_DENSE,QWEN_SUBSPACE_PROJECT,
    QWEN_SUBSPACE_CORRECT,QWEN_SUBSPACE_REASSOC,QWEN_SUBSPACE_HELD,
    QWEN_SUBSPACE_PROJECT2,QWEN_SUBSPACE_CARRY_CORRECT,QWEN_SUBSPACE_READ_CORRECT,
    QWEN_SUBSPACE_OPERAND_CORRECT
} QwenSubspaceMode;
typedef struct {
    QwenPlaneWork work;
    double coefficients[3],receiver_norm,reconstruction_norm,omitted_norm;
    float executed[3],rho;
} QwenSubspaceStats;
int qwen_subspace_init(QwenSubspace *,const QwenModel *);
void qwen_subspace_free(QwenSubspace *);
/* One explicit calibration operand; never called automatically by execution.
 * Orthogonalize and compile two images, then atomically admit rank three. */
int qwen_subspace_extend(QwenSubspace *,const float *,QwenPlaneWork *);
int qwen_advance_layer_subspace(QwenSession *,QwenSubspace *,QwenSubspaceMode,QwenSubspaceStats *);
/* Explicit experimental correction; copy a finite hidden-sized device operand
 * into owned storage before any layer scratch or caller alias can change it. */
int qwen_advance_layer_subspace_operand(QwenSession *,QwenSubspace *,const float *,size_t,QwenSubspaceStats *);
/* Supplied cross-layer projector, copied into owned immutable host storage.
 * One session/epoch/parent-lineage/position/edit-layer; token-parent replay may reuse it.
 * Original writer ports remain pre-edit; appended ports expose consumed writes. */
typedef struct {
    const QwenModel *model;
    uint64_t session_id,epoch,parent_lineage_id;
    int position,layer,compile_layer,ready;
    double *basis;
    float *host,*entry;
    size_t host_bytes,gpu_bytes;
} QwenWriter;
enum { QWEN_WRITER_NATIVE=0,QWEN_WRITER_ATTENTION=1,QWEN_WRITER_MLP=2,QWEN_WRITER_BOTH=3 };
int qwen_writer_init(QwenWriter *,const QwenSession *,const QwenSubspace *,int);
void qwen_writer_free(QwenWriter *);
int qwen_advance_layer_writer(QwenSession *,QwenWriter *,unsigned);
/* A reusable source-anchored gate law, not a receiver subspace. Its immutable
 * identity is model/layer/source/coefficient bytes; each invocation separately
 * binds the retained parent. Native projections and complete state remain. */
typedef struct {
    const QwenModel *model;
    int layer,source_position,ready;
    float *source_gate,*probability,*first,*second,*source_silu;
    size_t gpu_bytes;
} QwenGateLaw;
typedef enum { QWEN_GATE_NATIVE,QWEN_GATE_POLYNOMIAL,QWEN_GATE_ZERO } QwenGateMode;
typedef struct { QwenPlaneWork linear; unsigned native_swiglu,polynomial_swiglu; } QwenGateWork;
int qwen_gate_law_init(QwenGateLaw *,const QwenModel *,int,int,const float *);
void qwen_gate_law_free(QwenGateLaw *);
int qwen_advance_layer_gate(QwenSession *,const QwenGateLaw *,QwenGateMode,uint64_t,QwenGateWork *);
int qwen_model_init(QwenModel *,const TcmModel *,char *,size_t);
void qwen_model_free(QwenModel *);
int qwen_session_init(QwenSession *,const QwenModel *,int);
int qwen_session_init_layout(QwenSession *,const QwenModel *,int,uint64_t,QwenCacheStep,QwenCacheRewind,void *);
int qwen_abort_token(QwenSession *);
void qwen_session_free(QwenSession *);
int qwen_push(QwenSession *,int);
int qwen_begin_token(QwenSession *,int);
int qwen_advance_layer(QwenSession *);
int qwen_finish_token(QwenSession *);
int qwen_advance(QwenSession *);
int qwen_checkpoint(QwenSession *,QwenCheckpoint *);
int qwen_restore(QwenSession *,const QwenCheckpoint *);
int qwen_stack_push(QwenSession *,uint64_t *);
int qwen_stack_rollback(QwenSession *,uint64_t);
int qwen_stack_commit(QwenSession *,uint64_t);
int qwen_reset(QwenSession *);
const char *qwen_port_name(QwenPort);
int qwen_main(int,char **);
int qwen_serve_main(int,char **);
int qwen_bounded_serve_main(int,char **);
#endif
