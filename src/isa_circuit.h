#ifndef TCM_ISA_CIRCUIT_H
#define TCM_ISA_CIRCUIT_H
#include "machine.h"
#include "circuit_expansions.h"
#include <stdio.h>
typedef struct {
    double input[48],u[96],selection[9],dt[96],retention[384],injection[384],state[384],read[96],gate[96],gated[96];
    double post_input[48],normalized[48],pre[96],activation[96],write[48],output[48];
    unsigned seen;
} CmIsaRow;
typedef struct {
    const IsaSession *session;const IsaModel *model;
    IsaCheckpoint parent;
    IsaArch root_arch;
    unsigned char window[15];size_t window_length;
    unsigned layer;
    char result_digest[65];
    CmIsaRow *rows; /* 64 bit positions, transient */
    int ready,error;
} CmIsaCapture;
/* Zero initialize; exclusive synchronous capture from instruction boundary.
 * Executes ordinary begin + 64 row steps, leaves native heads awaiting commit.
 * Replacement policies/existing observers refuse; failures restore parent. */
int cm_isa_capture(CmIsaCapture *,IsaSession *,const unsigned char *,size_t,unsigned);
void cm_isa_capture_free(CmIsaCapture *);
int cm_isa_scan_kernel(const CmIsaCapture *,unsigned,double *);
int cm_isa_scan_cut(const CmIsaCapture *,unsigned,size_t,double *,double *);
int cm_isa_scan_path(const CmIsaCapture *,unsigned,size_t,size_t,double *);
/* GELU write split [3,48]: affine, nonlinear, native remainder. Biases and
 * actual normalized input retained. Optional fixed read row is length 48. */
int cm_isa_gelu_write(const CmIsaCapture *,size_t,const double *,double *,double *);
typedef struct {
    double copy_prior,head_bias,carry,gelu_affine,gelu_nonlinear;
    double mlp_remainder,readout_remainder,native_logit;
} CmIsaRegisterAccount;
/* Exact finite native accounting for a register logit after the final post
 * block only (layer 2). Earlier blocks have a nonlinear suffix. */
int cm_isa_register_account(const CmIsaCapture *,size_t,unsigned,CmIsaRegisterAccount *);
/* Point intervention on one recurrent coordinate after cut bit rows; complete
 * native remainder executes to all heads. Restores caller's exact state.
 * Returns register-head logit delta for declared register/bit, not a gradient
 * or a frozen-producer kernel. Commit is never called. */
int cm_isa_state_effect(const CmIsaCapture *,size_t,unsigned,unsigned,float,unsigned,unsigned,double *);
int cm_isa_report(FILE *,const CmIsaCapture *,unsigned,size_t,size_t);
#endif
