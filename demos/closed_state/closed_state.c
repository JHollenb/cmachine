#define _POSIX_C_SOURCE 200809L

/*
 * A small, supplied nonlinear organism for the closed-state demonstration.
 *
 * This deliberately uses the canonical cmachine GPU API, but links against
 * cmachine/tests/gpu_stub.c.  It is therefore a CPU mechanics profile, not a
 * CUDA measurement and not a learned-model result.  The point is to make the
 * state-interface claim executable: a compact code closes when its admitted
 * reads and writes stay inside the retained space, and fails when one write
 * exposes an omitted direction.
 */
#include "machine.h"
#include "gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIM 8
#define WIDTH 4
#define HORIZON 16
#define EPSILON 1.0e-6f

#define REQUIRE(condition)                                                     \
    do {                                                                       \
        if (!(condition)) {                                                    \
            fprintf(stderr, "closed_state:%s:%d: %s\n", __FILE__, __LINE__, \
                    #condition);                                              \
            return 2;                                                          \
        }                                                                      \
    } while (0)

typedef struct {
    const char *name;
    const char *role;
    int rank;
    int escape;
    int sign;
    int compact;
    int steps;
    float *states;
    float *norms;
    float *gates;
    float *ups;
    float *products;
    float *writes;
    float *rho;
} Path;

static unsigned norm_calls;
static unsigned matvec_calls;
static unsigned swiglu_calls;

static void matrices(float g[WIDTH * DIM], float u[WIDTH * DIM],
                     float down[DIM * WIDTH], int escape) {
    memset(g, 0, sizeof(float) * WIDTH * DIM);
    memset(u, 0, sizeof(float) * WIDTH * DIM);
    memset(down, 0, sizeof(float) * DIM * WIDTH);

    /* G and U read only coordinates 0 and 1. */
    g[0] = 0.5f;
    g[DIM + 1] = -0.75f;
    g[2 * DIM] = 0.25f;
    g[3 * DIM + 1] = 0.5f;
    u[1] = 0.5f;
    u[DIM] = 0.25f;
    u[2 * DIM + 1] = -0.5f;
    u[3 * DIM] = 0.5f;

    /* D writes coordinates 0 and 1; escape adds the same write to coord 2. */
    down[0] = 0.125f;
    down[WIDTH + 1] = -0.125f;
    if (escape) down[2 * WIDTH] = 0.25f;
}

static void finite_vector(const float *x, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (!isfinite(x[i])) {
            fprintf(stderr, "closed_state: non-finite trajectory value at %zu\n", i);
            exit(2);
        }
    }
}

static float *zeroed(size_t count) {
    float *x = (float *)calloc(count, sizeof(float));
    if (!x) {
        fprintf(stderr, "closed_state: allocation failed\n");
        exit(2);
    }
    return x;
}

static void project_matrices(const float g[WIDTH * DIM],
                             const float u[WIDTH * DIM],
                             const float down[DIM * WIDTH], int rank,
                             float *cg, float *cu, float *cd) {
    memset(cg, 0, sizeof(float) * WIDTH * rank);
    memset(cu, 0, sizeof(float) * WIDTH * rank);
    memset(cd, 0, sizeof(float) * rank * WIDTH);
    for (int row = 0; row < WIDTH; ++row) {
        for (int col = 0; col < rank; ++col) {
            cg[row * rank + col] = g[row * DIM + col];
            cu[row * rank + col] = u[row * DIM + col];
        }
    }
    for (int row = 0; row < rank; ++row) {
        for (int col = 0; col < WIDTH; ++col) {
            cd[row * WIDTH + col] = down[row * WIDTH + col];
        }
    }
}

static int run_path(TcmGpu *gpu, Path *path) {
    float g[WIDTH * DIM], u[WIDTH * DIM], down[DIM * WIDTH];
    matrices(g, u, down, path->escape);

    float *cg = zeroed(WIDTH * (size_t)path->rank);
    float *cu = zeroed(WIDTH * (size_t)path->rank);
    float *cd = zeroed((size_t)path->rank * WIDTH);
    project_matrices(g, u, down, path->rank, cg, cu, cd);

    float *wg = gpu_weight(cg, TCM_F32, WIDTH * (size_t)path->rank);
    float *wu = gpu_weight(cu, TCM_F32, WIDTH * (size_t)path->rank);
    float *wd = gpu_weight(cd, TCM_F32, (size_t)path->rank * WIDTH);
    float gamma_host[DIM];
    for (int i = 0; i < DIM; ++i) gamma_host[i] = 1.0f;
    float *gamma = gpu_weight(gamma_host, TCM_F32, DIM);

    float *state = (float *)gpu_alloc((size_t)path->rank * sizeof(float));
    float *representative = (float *)gpu_alloc(DIM * sizeof(float));
    float *normal = (float *)gpu_alloc(DIM * sizeof(float));
    float *gate = (float *)gpu_alloc(WIDTH * sizeof(float));
    float *up = (float *)gpu_alloc(WIDTH * sizeof(float));
    float *write = (float *)gpu_alloc((size_t)path->rank * sizeof(float));
    float *rho = (float *)gpu_alloc(sizeof(float));
    if (!wg || !wu || !wd || !gamma || !state || !representative || !normal ||
        !gate || !up || !write || !rho) {
        fprintf(stderr, "closed_state: canonical GPU allocation failed\n");
        return 2;
    }

    float initial[DIM] = {0.75f, -0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    initial[2] = (float)path->sign;
    const float retained_energy = path->compact && path->rank == 2 ? 1.0f : 0.0f;
    gpu_write(state, initial, (size_t)path->rank * sizeof(float));
    gpu_read(path->states, state, (size_t)path->rank * sizeof(float));
    finite_vector(path->states, path->rank);

    for (int step = 0; step < path->steps; ++step) {
        if (path->compact) {
            float canonical[DIM] = {0.0f, 0.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f};
            gpu_read(canonical, state, (size_t)path->rank * sizeof(float));
            canonical[path->rank] = sqrtf(retained_energy);
            gpu_write(representative, canonical, sizeof(canonical));
        }

        gpu_rms_scale(normal, path->compact ? representative : state, gamma,
                      DIM, EPSILON, rho);
        ++norm_calls;
        gpu_read(path->norms + (size_t)step * DIM, normal,
                 DIM * sizeof(float));
        gpu_read(path->rho + step, rho, sizeof(float));

        gpu_matvec(gpu, wg, path->rank, WIDTH, normal, gate);
        gpu_matvec(gpu, wu, path->rank, WIDTH, normal, up);
        matvec_calls += 2;
        gpu_read(path->gates + (size_t)step * WIDTH, gate,
                 WIDTH * sizeof(float));
        gpu_read(path->ups + (size_t)step * WIDTH, up,
                 WIDTH * sizeof(float));

        gpu_swiglu(gate, up, WIDTH);
        ++swiglu_calls;
        gpu_read(path->products + (size_t)step * WIDTH, gate,
                 WIDTH * sizeof(float));

        gpu_matvec(gpu, wd, WIDTH, path->rank, gate, write);
        ++matvec_calls;
        gpu_read(path->writes + (size_t)step * path->rank, write,
                 (size_t)path->rank * sizeof(float));

        gpu_add(state, write, path->rank);
        gpu_read(path->states + (size_t)(step + 1) * path->rank, state,
                 (size_t)path->rank * sizeof(float));
        finite_vector(path->states + (size_t)(step + 1) * path->rank,
                      path->rank);
    }

    gpu_free(wg);
    gpu_free(wu);
    gpu_free(wd);
    gpu_free(gamma);
    gpu_free(state);
    gpu_free(representative);
    gpu_free(normal);
    gpu_free(gate);
    gpu_free(up);
    gpu_free(write);
    gpu_free(rho);
    free(cg);
    free(cu);
    free(cd);
    return 0;
}

static Path *find_path(Path *paths, int count, const char *name) {
    for (int i = 0; i < count; ++i) {
        if (!strcmp(paths[i].name, name)) return &paths[i];
    }
    return NULL;
}

static int bytes_equal(const float *a, const float *b, size_t n) {
    return memcmp(a, b, n * sizeof(float)) == 0;
}

static int prefix_trajectory_equal(const Path *candidate, const Path *reference) {
    if (candidate->steps != reference->steps) return 0;
    for (int step = 0; step <= candidate->steps; ++step) {
        if (!bytes_equal(candidate->states + (size_t)step * candidate->rank,
                         reference->states + (size_t)step * reference->rank,
                         (size_t)candidate->rank)) {
            return 0;
        }
    }
    for (int step = 0; step < candidate->steps; ++step) {
        if (!bytes_equal(candidate->norms + (size_t)step * DIM,
                         reference->norms + (size_t)step * DIM,
                         (size_t)candidate->rank)) {
            return 0;
        }
    }
    if (!bytes_equal(candidate->gates, reference->gates,
                     (size_t)candidate->steps * WIDTH)) return 0;
    if (!bytes_equal(candidate->ups, reference->ups,
                     (size_t)candidate->steps * WIDTH)) return 0;
    if (!bytes_equal(candidate->products, reference->products,
                     (size_t)candidate->steps * WIDTH)) return 0;
    if (!bytes_equal(candidate->rho, reference->rho,
                     (size_t)candidate->steps)) return 0;
    for (int step = 0; step < candidate->steps; ++step) {
        if (!bytes_equal(candidate->writes + (size_t)step * candidate->rank,
                         reference->writes + (size_t)step * reference->rank,
                         (size_t)candidate->rank)) {
            return 0;
        }
    }
    return 1;
}

static double max_abs_prefix(const float *a, const float *b, int rows,
                             int cols_a, int cols_b) {
    double maximum = 0.0;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols_a; ++col) {
            double delta = fabs((double)a[(size_t)row * cols_a + col] -
                                (double)b[(size_t)row * cols_b + col]);
            if (delta > maximum) maximum = delta;
        }
    }
    return maximum;
}

static int first_state_difference(const Path *candidate,
                                  const Path *reference) {
    for (int row = 0; row <= candidate->steps; ++row) {
        if (!bytes_equal(candidate->states + (size_t)row * candidate->rank,
                         reference->states + (size_t)row * reference->rank,
                         (size_t)candidate->rank)) {
            return row;
        }
    }
    return -1;
}

static void json_float(FILE *out, float value) {
    fprintf(out, "%.9g", (double)value);
}

static void json_vector(FILE *out, const float *values, int count) {
    fputc('[', out);
    for (int i = 0; i < count; ++i) {
        if (i) fputc(',', out);
        json_float(out, values[i]);
    }
    fputc(']', out);
}

static void json_matrix(FILE *out, const float *values, int rows, int cols) {
    fputc('[', out);
    for (int row = 0; row < rows; ++row) {
        if (row) fputc(',', out);
        json_vector(out, values + (size_t)row * cols, cols);
    }
    fputc(']', out);
}

static void json_path(FILE *out, const Path *path, int *first) {
    if (!*first) fputc(',', out);
    *first = 0;
    fprintf(out,
            "{\"name\":\"%s\",\"role\":\"%s\",\"rank\":%d,"
            "\"escape_write\":%s,\"sign\":%d,\"compact\":%s,"
            "\"steps\":%d,\"retained_energy\":%.9g,\"state_bytes\":%d,"
            "\"static_affine_bytes\":%d,\"affine_macs_per_step\":%d,"
            "\"normalization_width\":%d,\"states\":",
            path->name, path->role, path->rank, path->escape ? "true" : "false",
            path->sign, path->compact ? "true" : "false", path->steps,
            path->compact && path->rank == 2 ? 1.0 : 0.0,
            4 * (path->rank + (path->compact ? 1 : 0)),
            4 * 3 * path->rank * WIDTH, 3 * path->rank * WIDTH, DIM);
    json_matrix(out, path->states, path->steps + 1, path->rank);
    fputs(",\"norms\":", out);
    json_matrix(out, path->norms, path->steps, DIM);
    fputs(",\"gates\":", out);
    json_matrix(out, path->gates, path->steps, WIDTH);
    fputs(",\"ups\":", out);
    json_matrix(out, path->ups, path->steps, WIDTH);
    fputs(",\"products\":", out);
    json_matrix(out, path->products, path->steps, WIDTH);
    fputs(",\"writes\":", out);
    json_matrix(out, path->writes, path->steps, path->rank);
    fputs(",\"rho\":", out);
    json_vector(out, path->rho, path->steps);
    fputc('}', out);
}

static void free_path(Path *path) {
    free(path->states);
    free(path->norms);
    free(path->gates);
    free(path->ups);
    free(path->products);
    free(path->writes);
    free(path->rho);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s REPORT.json\n", argv[0]);
        return 2;
    }

    static Path paths[] = {
        {.name = "closed-compact", .role = "candidate", .rank = 2,
         .escape = 0, .sign = 1, .compact = 1, .steps = HORIZON},
        {.name = "escape-stale", .role = "candidate", .rank = 2,
         .escape = 1, .sign = 1, .compact = 1, .steps = HORIZON},
        {.name = "repair-plus", .role = "candidate", .rank = 3,
         .escape = 1, .sign = 1, .compact = 1, .steps = HORIZON},
        {.name = "repair-minus", .role = "candidate", .rank = 3,
         .escape = 1, .sign = -1, .compact = 1, .steps = HORIZON},
        {.name = "closed-plus", .role = "reference", .rank = DIM,
         .escape = 0, .sign = 1, .compact = 0, .steps = HORIZON},
        {.name = "closed-minus", .role = "reference", .rank = DIM,
         .escape = 0, .sign = -1, .compact = 0, .steps = HORIZON},
        {.name = "escape-plus", .role = "reference", .rank = DIM,
         .escape = 1, .sign = 1, .compact = 0, .steps = HORIZON},
        {.name = "escape-minus", .role = "reference", .rank = DIM,
         .escape = 1, .sign = -1, .compact = 0, .steps = HORIZON},
    };
    const int path_count = (int)(sizeof(paths) / sizeof(paths[0]));
    for (int i = 0; i < path_count; ++i) {
        Path *path = &paths[i];
        path->states = zeroed((size_t)(HORIZON + 1) * path->rank);
        path->norms = zeroed((size_t)HORIZON * DIM);
        path->gates = zeroed((size_t)HORIZON * WIDTH);
        path->ups = zeroed((size_t)HORIZON * WIDTH);
        path->products = zeroed((size_t)HORIZON * WIDTH);
        path->writes = zeroed((size_t)HORIZON * path->rank);
        path->rho = zeroed(HORIZON);
    }

    TcmGpu *gpu = gpu_open();
    if (!gpu) {
        fprintf(stderr, "closed_state: gpu_open failed\n");
        return 2;
    }
    for (int i = 0; i < path_count; ++i) {
        if (run_path(gpu, &paths[i])) {
            gpu_close(gpu);
            return 2;
        }
    }

    Path *closed = find_path(paths, path_count, "closed-compact");
    Path *closed_plus = find_path(paths, path_count, "closed-plus");
    Path *closed_minus = find_path(paths, path_count, "closed-minus");
    Path *stale = find_path(paths, path_count, "escape-stale");
    Path *escape_plus = find_path(paths, path_count, "escape-plus");
    Path *escape_minus = find_path(paths, path_count, "escape-minus");
    Path *repair_plus = find_path(paths, path_count, "repair-plus");
    Path *repair_minus = find_path(paths, path_count, "repair-minus");
    if (!closed || !closed_plus || !closed_minus || !stale || !escape_plus ||
        !escape_minus || !repair_plus || !repair_minus) {
        fprintf(stderr, "closed_state: path table incomplete\n");
        gpu_close(gpu);
        return 2;
    }

    const int closed_plus_exact = prefix_trajectory_equal(closed, closed_plus);
    const int closed_minus_exact = prefix_trajectory_equal(closed, closed_minus);
    const int repair_plus_exact = prefix_trajectory_equal(repair_plus, escape_plus);
    const int repair_minus_exact = prefix_trajectory_equal(repair_minus, escape_minus);
    const int stale_plus_first = first_state_difference(stale, escape_plus);
    const int stale_minus_first = first_state_difference(stale, escape_minus);
    const int reference_initial_code_equal =
        memcmp(escape_plus->states, escape_minus->states, 2 * sizeof(float)) == 0;
    const int reference_initial_energy_equal =
        escape_plus->states[0 * DIM + 2] * escape_plus->states[0 * DIM + 2] ==
        escape_minus->states[0 * DIM + 2] * escape_minus->states[0 * DIM + 2];

    FILE *out = fopen(argv[1], "w");
    if (!out) {
        fprintf(stderr, "closed_state: cannot open %s\n", argv[1]);
        gpu_close(gpu);
        return 2;
    }
    fprintf(out,
            "{\"schema\":\"tcm-closed-state-demo-v1\","
            "\"profile\":{\"backend\":\"cpu-gpu-stub\","
            "\"canonical_header\":\"src/gpu.h\","
            "\"canonical_primitives\":[\"gpu_rms_scale\",\"gpu_matvec\","
            "\"gpu_swiglu\",\"gpu_add\",\"gpu_weight\"],"
            "\"stub\":\"tests/gpu_stub.c\","
            "\"cuda_executed\":false,\"learned_model\":false},"
            "\"organism\":{\"dimension\":%d,\"width\":%d,"
            "\"epsilon\":%.9g,\"horizon\":%d,"
            "\"initial_states\":[[0.75,-0.5,1,0,0,0,0,0],"
            "[0.75,-0.5,-1,0,0,0,0,0]],"
            "\"reads\":[\"coordinate0\",\"coordinate1\"],"
            "\"closed_writes\":[\"coordinate0\",\"coordinate1\"],"
            "\"escaping_write\":\"coordinate2\"},",
            DIM, WIDTH, (double)EPSILON, HORIZON);
    fprintf(out,
            "\"nominal_costs\":{\"full_state_bytes\":32,"
            "\"closed_state_bytes\":12,\"repaired_state_bytes\":16,"
            "\"full_affine_macs_per_update\":96,"
            "\"closed_affine_macs_per_update\":24,"
            "\"repaired_affine_macs_per_update\":36,"
            "\"normalization_width_always\":8},"
            "\"mechanics\":{\"closed_vs_plus_byte_exact\":%s,"
            "\"closed_vs_minus_byte_exact\":%s,"
            "\"repair_vs_escape_plus_byte_exact\":%s,"
            "\"repair_vs_escape_minus_byte_exact\":%s,"
            "\"escape_initial_retained_code_equal\":%s,"
            "\"escape_initial_omitted_energy_equal\":%s,"
            "\"comparison_scope\":\"retained states and normalization coordinates, "
            "scales, gates, ups, products, and writes\","
            "\"stale_vs_escape_plus_first_state_difference\":%d,"
            "\"stale_vs_escape_minus_first_state_difference\":%d},",
            closed_plus_exact ? "true" : "false",
            closed_minus_exact ? "true" : "false",
            repair_plus_exact ? "true" : "false",
            repair_minus_exact ? "true" : "false",
            reference_initial_code_equal ? "true" : "false",
            reference_initial_energy_equal ? "true" : "false",
            stale_plus_first, stale_minus_first);
    fprintf(out, "\"paths\":[");
    int first = 1;
    for (int i = 0; i < path_count; ++i) json_path(out, &paths[i], &first);
    fprintf(out, "],\"calls\":{\"rms\":%u,\"matvec\":%u,\"swiglu\":%u},",
            norm_calls, matvec_calls, swiglu_calls);

    const double escape_plus_energy =
        (double)escape_plus->states[1 * escape_plus->rank + 2] *
        escape_plus->states[1 * escape_plus->rank + 2];
    const double escape_minus_energy =
        (double)escape_minus->states[1 * escape_minus->rank + 2] *
        escape_minus->states[1 * escape_minus->rank + 2];
    const double final_pair_distance =
        sqrt(pow((double)escape_plus->states[HORIZON * DIM] -
                     escape_minus->states[HORIZON * DIM],
                 2) +
             pow((double)escape_plus->states[HORIZON * DIM + 1] -
                     escape_minus->states[HORIZON * DIM + 1],
                 2));
    fprintf(out,
            "\"escape_observation\":{\"one_step_omitted_energy_plus\":%.9g,"
            "\"one_step_omitted_energy_minus\":%.9g,"
            "\"full_final_declared_output_l2\":%.9g,"
            "\"stale_max_state_error_vs_plus\":%.9g,"
            "\"stale_max_state_error_vs_minus\":%.9g},"
            "\"scientific_scope\":{\"status\":\"observation\","
            "\"claim\":\"a supplied nonlinear recurrence demonstrates a closed "
            "compact code, an escaping write, and a repaired code\","
            "\"not_claimed\":[\"learned_model_compression\","
            "\"CUDA_numerical_parity\",\"serving_speedup\","
            "\"general_chat_accuracy\"],"
            "\"finite_profile\":\"FP32 CPU stub arithmetic; exactness is only "
            "for this checked-in fixture\"}}\n",
            escape_plus_energy, escape_minus_energy, final_pair_distance,
            max_abs_prefix(stale->states, escape_plus->states, HORIZON + 1,
                           stale->rank, escape_plus->rank),
            max_abs_prefix(stale->states, escape_minus->states, HORIZON + 1,
                           stale->rank, escape_minus->rank));
    if (fclose(out)) {
        fprintf(stderr, "closed_state: close failed\n");
        gpu_close(gpu);
        return 2;
    }

    gpu_close(gpu);
    for (int i = 0; i < path_count; ++i) free_path(&paths[i]);
    return 0;
}
