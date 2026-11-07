#ifndef TCM_PROGRAM_H
#define TCM_PROGRAM_H
/* Versioned vector SSA schedule. All indices/shapes are validated before use.
   Each invocation consumes one input row and commits convolution/scan slots.
   Numerical serving uses only C, libm and optional CPU BLAS. */
enum { P_LINEAR=1, P_ADD, P_MUL, P_REMAINDER, P_COS, P_AFFINE,
       P_CONCAT, P_SLICE, P_RMS, P_EMBED, P_CONV, P_SCAN, P_SILU, P_CONST,
       P_RELU, P_DIV, P_MATCH_ROWS };
int tcm_program_main(int, char **);
#endif
