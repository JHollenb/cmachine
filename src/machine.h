#ifndef TCM_MACHINE_H
#define TCM_MACHINE_H

#include <stddef.h>
#include <stdint.h>

enum { TCM_ISA = 1, TCM_QWEN = 2, TCM_ISA_LAW = 3, TCM_PROGRAM = 4 };
enum { TCM_F32 = 1, TCM_BF16 = 2, TCM_F64 = 3 };
typedef struct {
    char name[65];
    uint32_t rank, dtype;
    uint64_t dim[4], bytes;
    const void *data;
} TcmTensor;
typedef struct {
    uint32_t architecture, count, config[16];
    char identity[65], digest[65];
    TcmTensor *tensor;
    void *mapping;
    size_t mapping_bytes;
} TcmModel;

int tcm_load(TcmModel *, const char *, char *, size_t);
void tcm_unload(TcmModel *);
const TcmTensor *tcm_tensor(const TcmModel *, const char *);
void tcm_sha256(const void *, size_t, unsigned char[32]);
void tcm_hex(const unsigned char *, size_t, char *);

typedef struct { const float *w, *b; int in, out; } IsaLinear;
typedef struct {
    IsaLinear input, select, dt, output, post1, post2;
    const float *conv, *conv_bias, *a_log, *d, *gamma, *beta;
} IsaMixer;
typedef struct {
    const TcmModel *bundle;
    IsaLinear decoder1, decoder2, dest, src, query, offset, input;
    IsaLinear flag_query, reg, imm, store, flags1, flags2;
    IsaLinear width, length, syscall, store_flag;
    IsaMixer mixer[3];
    const float *pos, *ins_pos;
    float post_epsilon;
} IsaModel;
typedef struct {
    const TcmModel *bundle;
    const float *x0, *p0;
    const double *h0, *h1, *coeff;
    unsigned layer, row;
} IsaLaw;
typedef struct {
    uint64_t registers[8], rip, instruction;
    unsigned char flags[4];
} IsaArch;
typedef struct {
    float regs[8][64], flags[4], dest[8], src[8], width[3], length[15];
    float syscall, imm[64], store, store_addr[64], store_value[64];
} IsaOutput;
typedef struct {
    int enabled, quadratic;
    uint64_t instruction;
    double s, t;
} IsaLawPolicy;
/* Complete row-boundary mutable closure. No tensor pointers are serialized. */
typedef struct {
    IsaArch arch;
    unsigned row, phase;
    float window[120], before[8][64], before_flags[4];
    float z[256], dest[8], src[8], a[64], s[64], imm[64][2];
    float offsets[8], flag_point[4][64];
    float scan[3][96][4], conv[3][2][96];
    float flag_reads[4][48], hidden_sum[48];
    IsaOutput output;
    uint64_t dense_post_rows, law_rows;
    double law_coordinates[2], off_plane_norm;
    float law_input[48], law_branch[48], law_hidden[48];
} IsaState;
typedef struct {
    const IsaModel *model;
    const IsaLaw *law;
    IsaLawPolicy policy;
    IsaState state;
} IsaSession;
typedef struct {
    char model_digest[65], law_digest[65];
    IsaLawPolicy policy;
    IsaState state;
} IsaCheckpoint;

int isa_model_init(IsaModel *, const TcmModel *, char *, size_t);
int isa_law_init(IsaLaw *, const TcmModel *, const IsaModel *, char *, size_t);
void isa_session_init(IsaSession *, const IsaModel *, const IsaArch *);
int isa_begin(IsaSession *, const unsigned char *, size_t);
int isa_advance(IsaSession *);
int isa_commit(IsaSession *);
void isa_checkpoint(const IsaSession *, IsaCheckpoint *);
int isa_restore(IsaSession *, const IsaCheckpoint *);
unsigned isa_argmax(const float *, unsigned);
uint64_t isa_bits(const float *);

#endif
