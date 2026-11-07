/* The ordinary Qwen CUDA runtime is included in this translation unit so the
 * callback attention kernels use the same cublas handle, stream, and FP32
 * reduction policy as the native path. */
#define gpu_matvec memory_original_matvec
#include "runtime/qwen_cuda.cu"
#undef gpu_matvec
#include <stdint.h>
static uint64_t memory_linear_calls, memory_linear_macs;
extern "C" uint64_t gpu_linear_calls(void) { return memory_linear_calls; }
extern "C" uint64_t gpu_linear_macs(void) { return memory_linear_macs; }
extern "C" void gpu_matvec(TcmGpu *g,const float *w,int rows,int cols,const float *x,float *out) {
    memory_linear_calls++; memory_linear_macs += (uint64_t)rows * (uint64_t)cols;
    memory_original_matvec(g,w,rows,cols,x,out);
}

__global__ static void memory_encode_kernel(signed char *key, signed char *value,
                                             float *scales, const float *k, const float *v,
                                             int capacity, int position, int *invalid,
                                             int kvheads, int dim) {
    int kind = blockIdx.x / kvheads, head = blockIdx.x % kvheads;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float x = (kind ? v : k)[head * dim + d];
        if (!isfinite(x)) atomicExch(invalid, 1);
        /* Qwen's shipped profile has dim=128.  The reduction is written with a
         * bounded shared array so a future small head dimension remains safe. */
        __shared__ float maxima[256];
        maxima[threadIdx.x] = d < dim && isfinite(x) ? fabsf(x) : 0.0f;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride; stride /= 2) {
            if (threadIdx.x < stride) maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
            __syncthreads();
        }
        float scale = maxima[0] > 0.0f ? __fdiv_rn(maxima[0], 127.0f) : 1.0f;
        if (!(scale > 0.0f) || !isfinite(scale)) { if (!threadIdx.x) atomicExch(invalid, 1); return; }
        float q = isfinite(x) ? roundf(__fdiv_rn(x, scale)) : 0.0f;
        q = fminf(127.0f, fmaxf(-127.0f, q));
        (kind ? value : key)[((size_t)head * capacity + position) * dim + d] = (signed char)q;
        if (!threadIdx.x) scales[(kind * kvheads + head) * capacity + position] = scale;
        __syncthreads();
    }
}

extern "C" int gpu_encode_current(signed char *key, signed char *value, float *scales,
                                    const float *k, const float *v, int capacity, int position,
                                    int *invalid, int kvheads, int dim) {
    if (kvheads < 1 || kvheads > 32 || dim != 128 || position < 0 || position >= capacity) return -1;
    gpu_zero(invalid, sizeof(int));
    memory_encode_kernel<<<2 * kvheads, 128>>>(key, value, scales, k, v, capacity, position, invalid, kvheads, dim);
    CUDA(cudaGetLastError());
    int status = 0; gpu_read(&status, invalid, sizeof(status));
    return status ? -1 : 0;
}

__global__ static void memory_decode_rows(float *out, const signed char *codes, const float *scales,
                                          int source_capacity, int output_capacity, int start,
                                          int length, int kvheads, int dim, int kind, int recent) {
    int index = blockIdx.x, head = index / length, row = index % length;
    if (head >= kvheads) return;
    int source_row = recent ? (start + row) % source_capacity : start + row;
    float scale = scales[(kind * kvheads + head) * source_capacity + source_row];
    for (int d = threadIdx.x; d < dim; d += blockDim.x)
        out[((size_t)head * output_capacity + row) * dim + d] = (float)codes[((size_t)head * source_capacity + source_row) * dim + d] * scale;
}

static void memory_dense_attention(TcmGpu *g, float *out, const float *q, const float *keys,
                                   const float *values, float *scores, int qheads, int kvheads,
                                   int dim, int capacity, int length) {
    const float one = 1.0f, zero = 0.0f; int group = qheads / kvheads;
    for (int h = 0; h < kvheads; ++h)
        BLAS(cublasSgemm(g->blas, CUBLAS_OP_T, CUBLAS_OP_N, length, group, dim, &one,
                         keys + (size_t)h * capacity * dim, dim,
                         q + (size_t)h * group * dim, dim, &zero,
                         scores + (size_t)h * group * length, length));
    softmax_kernel<<<qheads, 256>>>(scores, length, 1.0f / sqrtf((float)dim));
    CUDA(cudaGetLastError());
    for (int h = 0; h < kvheads; ++h)
        BLAS(cublasSgemm(g->blas, CUBLAS_OP_N, CUBLAS_OP_N, dim, group, length, &one,
                         values + (size_t)h * capacity * dim, dim,
                         scores + (size_t)h * group * length, length, &zero,
                         out + (size_t)h * group * dim, dim));
}

extern "C" void gpu_byte_attention(TcmGpu *g, float *out, const float *q,
                                     const signed char *key, const signed char *value,
                                     const float *scales, float *scores, float *workspace,
                                     int capacity, int length, int qheads, int kvheads, int dim) {
    if (length < 1) { gpu_zero(out, (size_t)qheads * dim * sizeof(float)); return; }
    memory_decode_rows<<<kvheads * length, 128>>>(workspace, key, scales, capacity, capacity, 0, length, kvheads, dim, 0, 0);
    CUDA(cudaGetLastError());
    memory_decode_rows<<<kvheads * length, 128>>>(workspace + (size_t)kvheads * capacity * dim,
                                                  value, scales, capacity, capacity, 0, length, kvheads, dim, 1, 0);
    CUDA(cudaGetLastError());
    memory_dense_attention(g, out, q, workspace,
                           workspace + (size_t)kvheads * capacity * dim,
                           scores, qheads, kvheads, dim, capacity, length);
}

__global__ static void memory_top4_mask(float *scores, int length, int group) {
    if (threadIdx.x) return;
    int qhead = blockIdx.x, kvhead = qhead / group, local = qhead % group;
    float best[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    int ids[4] = {-1, -1, -1, -1};
    float *x = scores + ((size_t)kvhead * group + local) * length;
    for (int row = 0; row < length; ++row) {
        float v = x[row]; int slot = -1;
        for (int j = 0; j < 4; ++j) if (slot < 0 && (v > best[j] || (v == best[j] && (ids[j] < 0 || row < ids[j])))) slot = j;
        if (slot >= 0) { for (int j = 3; j > slot; --j) { best[j] = best[j-1]; ids[j] = ids[j-1]; } best[slot] = v; ids[slot] = row; }
    }
    for (int row = 0; row < length; ++row) {
        bool keep = false; for (int j = 0; j < 4; ++j) keep = keep || ids[j] == row;
        if (!keep) x[row] = -INFINITY;
    }
}

/* The top-four arm scans every key score, then performs a genuinely sparse
 * value read.  The output is one q-head per block; scores retain their full
 * length only as an audit trace of the selection work. */
__global__ static void memory_sparse_values(float *out, const float *values, const float *scores,
                                            int capacity, int length, int qheads, int kvheads, int dim) {
    int qhead = blockIdx.x, d = threadIdx.x, group = qheads / kvheads;
    if (qhead >= qheads || d >= dim) return;
    int kvhead = qhead / group, local = qhead % group;
    const float *p = scores + ((size_t)kvhead * group + local) * length;
    const float *v = values + (size_t)kvhead * capacity * dim;
    float sum = 0.0f; for (int row = 0; row < length; ++row) if (p[row] > 0.0f) sum += p[row] * v[(size_t)row * dim + d];
    out[(size_t)qhead * dim + d] = sum;
}

extern "C" void gpu_byte_top4_attention(TcmGpu *g, float *out, const float *q,
                                          const signed char *key, const signed char *value,
                                          const float *scales, float *scores, float *workspace,
                                          int capacity, int length, int qheads, int kvheads, int dim) {
    if (length < 1) { gpu_zero(out, (size_t)qheads * dim * sizeof(float)); return; }
    memory_decode_rows<<<kvheads * length, 128>>>(workspace, key, scales, capacity, capacity, 0, length, kvheads, dim, 0, 0);
    CUDA(cudaGetLastError());
    const float one = 1.0f, zero = 0.0f; int group = qheads / kvheads;
    for (int h = 0; h < kvheads; ++h)
        BLAS(cublasSgemm(g->blas, CUBLAS_OP_T, CUBLAS_OP_N, length, group, dim, &one,
                         workspace + (size_t)h * capacity * dim, dim,
                         q + (size_t)h * group * dim, dim, &zero,
                         scores + (size_t)h * group * length, length));
    memory_top4_mask<<<qheads, 1>>>(scores, length, group); CUDA(cudaGetLastError());
    softmax_kernel<<<qheads, 256>>>(scores, length, 1.0f / sqrtf((float)dim)); CUDA(cudaGetLastError());
    memory_decode_rows<<<kvheads * length, 128>>>(workspace, value, scales, capacity, capacity, 0, length, kvheads, dim, 1, 0); CUDA(cudaGetLastError());
    memory_sparse_values<<<qheads, 128>>>(out, workspace, scores, capacity, length, qheads, kvheads, dim); CUDA(cudaGetLastError());
}

extern "C" void gpu_recent_attention(TcmGpu *g, float *out, const float *q,
                                       const signed char *key, const signed char *value,
                                       const float *scales, float *scores, float *workspace,
                                       int ring_capacity, int start, int length,
                                       int qheads, int kvheads, int dim) {
    int n = length - start;
    if (n < 1) { gpu_zero(out, (size_t)qheads * dim * sizeof(float)); return; }
    memory_decode_rows<<<kvheads * n, 128>>>(workspace, key, scales, ring_capacity, n, start, n, kvheads, dim, 0, 1); CUDA(cudaGetLastError());
    memory_decode_rows<<<kvheads * n, 128>>>(workspace + (size_t)kvheads * n * dim,
                                             value, scales, ring_capacity, n, start, n, kvheads, dim, 1, 1); CUDA(cudaGetLastError());
    memory_dense_attention(g, out, q, workspace, workspace + (size_t)kvheads * n * dim,
                           scores, qheads, kvheads, dim, n, n);
}

__global__ static void memory_gather_fp32(float *out, const float *source, int source_capacity,
                                          int output_capacity, int start, int length, int kvheads, int dim) {
    int index = blockIdx.x, head = index / length, row = index % length;
    if (head >= kvheads) return;
    int source_row = start + row;
    for (int d = threadIdx.x; d < dim; d += blockDim.x)
        out[((size_t)head * output_capacity + row) * dim + d] = source[((size_t)head * source_capacity + source_row) * dim + d];
}

extern "C" void gpu_history_attention(TcmGpu *g, float *out, const float *q,
                                        const float *keys, const float *values, float *scores,
                                        float *workspace, int capacity, int start, int length,
                                        int qheads, int kvheads, int dim) {
    if (length < 1) { gpu_zero(out, (size_t)qheads * dim * sizeof(float)); return; }
    if (start == 0) { gpu_attention(g, out, q, keys, values, scores, qheads, kvheads, dim, capacity, length); return; }
    int n = length - start;
    memory_gather_fp32<<<kvheads * n, 128>>>(workspace, keys, capacity, n, start, n, kvheads, dim); CUDA(cudaGetLastError());
    memory_gather_fp32<<<kvheads * n, 128>>>(workspace + (size_t)kvheads * n * dim, values, capacity, n, start, n, kvheads, dim); CUDA(cudaGetLastError());
    memory_dense_attention(g, out, q, workspace, workspace + (size_t)kvheads * n * dim,
                           scores, qheads, kvheads, dim, n, n);
}
