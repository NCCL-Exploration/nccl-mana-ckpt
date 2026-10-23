/*************************************************************************
 *                                                                       *
 *       N  A  S     P A R A L L E L     B E N C H M A R K S  3.4        *
 *                                                                       *
 *                       N C C L     V E R S I O N                       *
 *                                                                       *
 *                                  M G                                  *
 *                                                                       *
 *************************************************************************
 *                                                                       *
 *   This benchmark is an NCCL port of the NAS Parallel Benchmarks 3.4   *
 *   MG (Multi-Grid) 3D Poisson V-cycle solver. All MPI collectives      *
 *   and 3D Cartesian nearest-neighbor halo exchanges have been          *
 *   replaced with their NCCL equivalents operating on pinned host       *
 *   memory mapped into the CUDA address space.                          *
 *                                                                       *
 *************************************************************************/

#include <cuda_runtime.h>
#include <math.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double mg_wtime(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#include "c_timers.h"
#include "npbparams.h"

/*****************************************************************/
/* Class-dependent parameters (NPB 3.4.4 MG)                     */
/*****************************************************************/
#if CLASS == 'S'
#define NX_DEFAULT   32
#define NY_DEFAULT   32
#define NZ_DEFAULT   32
#define NIT_DEFAULT  4
#define LT_DEFAULT   5
#define VERIFY_VALUE 0.5307707005734e-04
#elif CLASS == 'W'
#define NX_DEFAULT   128
#define NY_DEFAULT   128
#define NZ_DEFAULT   128
#define NIT_DEFAULT  4
#define LT_DEFAULT   7
#define VERIFY_VALUE 0.6467329375339e-05
#elif CLASS == 'A'
#define NX_DEFAULT   256
#define NY_DEFAULT   256
#define NZ_DEFAULT   256
#define NIT_DEFAULT  4
#define LT_DEFAULT   8
#define VERIFY_VALUE 0.2433365309069e-05
#elif CLASS == 'B'
#define NX_DEFAULT   256
#define NY_DEFAULT   256
#define NZ_DEFAULT   256
#define NIT_DEFAULT  20
#define LT_DEFAULT   8
#define VERIFY_VALUE 0.1800564401355e-05
#elif CLASS == 'C'
#define NX_DEFAULT   512
#define NY_DEFAULT   512
#define NZ_DEFAULT   512
#define NIT_DEFAULT  20
#define LT_DEFAULT   9
#define VERIFY_VALUE 0.5706732285740e-06
#elif CLASS == 'D'
#define NX_DEFAULT   1024
#define NY_DEFAULT   1024
#define NZ_DEFAULT   1024
#define NIT_DEFAULT  50
#define LT_DEFAULT   10
#define VERIFY_VALUE 0.1583275060440e-09
#else
#error "Unknown CLASS for MG benchmark"
#endif

#define MAXLEVEL  (LT_DEFAULT + 1)
#define MIN_PROCS 1
#define MAX_PROCS 131072

/* 1-based 3D column-major index matching Fortran u(i1, i2, i3) */
#define U3(arr, i1, i2, i3, n1, n2)                                                                \
    (arr)[((i1) - 1) + (size_t)(n1) * (((i2) - 1) + (size_t)(n2) * ((i3) - 1))]

/*****************************************************************/
/* Timing constants                                              */
/*****************************************************************/
#define T_BENCH   1
#define T_INIT    2
#define T_PSINV   3
#define T_RESID   4
#define T_RPRJ3   5
#define T_INTERP  6
#define T_NORM2U3 7
#define T_COMM3   8
#define T_RCOMM   9
#define T_LAST    9

/*****************************************************************/
/* Global variables                                              */
/*****************************************************************/
static int my_rank, comm_size, np_total, me;
static int nm, nv, nr, nm2, m_scratch;
static int lt, lb, nit;
static int is1, is2, is3, ie1, ie2, ie3;

static int nx[MAXLEVEL + 1], ny[MAXLEVEL + 1], nz[MAXLEVEL + 1];
static int ir[MAXLEVEL + 1], m1[MAXLEVEL + 1], m2[MAXLEVEL + 1], m3[MAXLEVEL + 1];
static int dead[MAXLEVEL + 1];
static int give_ex[4][MAXLEVEL + 1], take_ex[4][MAXLEVEL + 1];
/* dir index: 0 for dir=-1, 1 for dir=+1 */
static int nbr[4][2][MAXLEVEL + 1];
static int debug_vec[8];
static int timeron;

static double *u;
static double *v;
static double *r;
static double *buff[5]; /* buff[1..4], each of length nm2+4, pinned for NCCL */
static double *send_dbl;
static double *recv_dbl;
static int *send_int;
static int *recv_int;

static ncclComm_t nccl_comm;
static cudaStream_t cuda_stream;
static ncclUniqueId nccl_id;

extern "C" void c_print_results(char *name, char class_npb, int n1, int n2, int n3, int niter,
                                int nprocs_active, int nprocs_total, double t, double mops,
                                char *optype, int passed_verification, char *npbversion,
                                char *compiletime, char *mpicc, char *clink, char *cmpi_lib,
                                char *cmpi_inc, char *cflags, char *clinkflags);

static void comm3(double *u_arr, int n1, int n2, int n3, int kk);
static void comm3_ex(double *u_arr, int n1, int n2, int n3, int kk);
static void zero3(double *z_arr, int n1, int n2, int n3);

/*****************************************************************/
/* Random number generators (randlc and vranlc)                  */
/*****************************************************************/
static double randlc(double *x_seed, double a_mult) {
    const double r23 = 1.1920928955078125e-07;
    const double r46 = r23 * r23;
    const double t23 = 8388608.0;
    const double t46 = t23 * t23;

    double t1, t2, t3, t4, a1, a2, x1, x2, z_val;

    t1 = r23 * a_mult;
    a1 = (int)t1;
    a2 = a_mult - t23 * a1;

    t1 = r23 * (*x_seed);
    x1 = (int)t1;
    x2 = (*x_seed) - t23 * x1;
    t1 = a1 * x2 + a2 * x1;
    t2 = (int)(r23 * t1);
    z_val = t1 - t23 * t2;
    t3 = t23 * z_val + a2 * x2;
    t4 = (int)(r46 * t3);
    *x_seed = t3 - t46 * t4;

    return r46 * (*x_seed);
}

static void vranlc(int n, double *x_seed, double a_mult, double *y) {
    const double r23 = 1.1920928955078125e-07;
    const double r46 = r23 * r23;
    const double t23 = 8388608.0;
    const double t46 = t23 * t23;

    double t1, t2, t3, t4, a1, a2, x1, x2, z_val, xv;
    int i;

    t1 = r23 * a_mult;
    a1 = (int)t1;
    a2 = a_mult - t23 * a1;
    xv = *x_seed;

    for (i = 0; i < n; i++) {
        t1 = r23 * xv;
        x1 = (int)t1;
        x2 = xv - t23 * x1;
        t1 = a1 * x2 + a2 * x1;
        t2 = (int)(r23 * t1);
        z_val = t1 - t23 * t2;
        t3 = t23 * z_val + a2 * x2;
        t4 = (int)(r46 * t3);
        xv = t3 - t46 * t4;
        y[i] = r46 * xv;
    }
    *x_seed = xv;
}

/*****************************************************************/
/* Memory allocation and cleanup                                 */
/*****************************************************************/
static void alloc_space(void) {
    int log_p = 0;
    int tmp = comm_size;
    while (tmp > 1) {
        log_p++;
        tmp /= 2;
    }

    int log2_size = LT_DEFAULT;
    int lm = log2_size - log_p / 3;
    int ndim1 = lm;
    int ndim3 = log2_size - (log_p + 2) / 3;
    int ndim2 = log2_size - (log_p + 1) / 3;

    nm = 2 + (1 << lm);
    nv = (2 + (1 << ndim1)) * (2 + (1 << ndim2)) * (2 + (1 << ndim3));
    nm2 = 2 * nm * nm;
    nr = (8 * (nv + nm * nm + 5 * nm + 14 * LT_DEFAULT - 7 * lm)) / 7;
    m_scratch = nm + 2;

    u = (double *)calloc((size_t)(nr + 16), sizeof(double));
    v = (double *)calloc((size_t)(nv + 16), sizeof(double));
    r = (double *)calloc((size_t)(nr + 16), sizeof(double));

    for (int b = 1; b <= 4; b++) {
        cudaMallocHost((void **)&buff[b], (size_t)(nm2 + 16) * sizeof(double));
        memset(buff[b], 0, (size_t)(nm2 + 16) * sizeof(double));
    }
    cudaMallocHost((void **)&send_dbl, 16 * sizeof(double));
    cudaMallocHost((void **)&recv_dbl, 16 * sizeof(double));
    cudaMallocHost((void **)&send_int, 16 * sizeof(int));
    cudaMallocHost((void **)&recv_int, 16 * sizeof(int));

    if (!u || !v || !r || !buff[1] || !buff[2] || !buff[3] || !buff[4] || !send_dbl || !recv_dbl ||
        !send_int || !recv_int) {
        fprintf(stderr, "Rank %d: Failed to allocate memory for MG\n", my_rank);
        exit(1);
    }
}

static void free_space(void) {
    free(u);
    free(v);
    free(r);
    for (int b = 1; b <= 4; b++) {
        cudaFreeHost(buff[b]);
    }
    cudaFreeHost(send_dbl);
    cudaFreeHost(recv_dbl);
    cudaFreeHost(send_int);
    cudaFreeHost(recv_int);
}

/*****************************************************************/
/* Setup multigrid hierarchy and 3D processor topology           */
/*****************************************************************/
static void setup(int *n1, int *n2, int *n3, int *k_out) {
    int dx, dy, log_p = 0;
    int k, ax, j;
    int next[4], mi[4][MAXLEVEL + 1], mip[4][MAXLEVEL + 1];
    int ng[4][MAXLEVEL + 1];
    int idi[4], pi[4], idin[4][2];

    ng[1][lt] = nx[lt];
    ng[2][lt] = ny[lt];
    ng[3][lt] = nz[lt];
    for (ax = 1; ax <= 3; ax++) {
        next[ax] = 1;
        for (k = lt - 1; k >= 1; k--) {
            ng[ax][k] = ng[ax][k + 1] / 2;
        }
    }
    for (k = lt; k >= 1; k--) {
        nx[k] = ng[1][k];
        ny[k] = ng[2][k];
        nz[k] = ng[3][k];
    }

    int tmp = comm_size;
    while (tmp > 1) {
        log_p++;
        tmp /= 2;
    }
    dx = log_p / 3;
    pi[1] = 1 << dx;
    idi[1] = me % pi[1];

    dy = (log_p - dx) / 2;
    pi[2] = 1 << dy;
    idi[2] = (me / pi[1]) % pi[2];

    pi[3] = comm_size / (pi[1] * pi[2]);
    idi[3] = me / (pi[1] * pi[2]);

    for (k = lt; k >= 1; k--) {
        dead[k] = 0;
        for (ax = 1; ax <= 3; ax++) {
            take_ex[ax][k] = 0;
            give_ex[ax][k] = 0;

            mi[ax][k] =
                2 + ((idi[ax] + 1) * ng[ax][k]) / pi[ax] - ((idi[ax] + 0) * ng[ax][k]) / pi[ax];
            mip[ax][k] = 2 + ((next[ax] + idi[ax] + 1) * ng[ax][k]) / pi[ax] -
                         ((next[ax] + idi[ax] + 0) * ng[ax][k]) / pi[ax];

            if (mip[ax][k] == 2 || mi[ax][k] == 2) {
                next[ax] = 2 * next[ax];
            }

            if (k + 1 <= lt) {
                if (mip[ax][k] == 2 && mi[ax][k] == 3) {
                    give_ex[ax][k + 1] = 1;
                }
                if (mip[ax][k] == 3 && mi[ax][k] == 2) {
                    take_ex[ax][k + 1] = 1;
                }
            }
        }

        if (mi[1][k] == 2 || mi[2][k] == 2 || mi[3][k] == 2) {
            dead[k] = 1;
        }
        m1[k] = mi[1][k];
        m2[k] = mi[2][k];
        m3[k] = mi[3][k];

        for (ax = 1; ax <= 3; ax++) {
            /* dir = +1 -> index 1; dir = -1 -> index 0 */
            idin[ax][1] = (idi[ax] + next[ax] + pi[ax]) % pi[ax];
            idin[ax][0] = (idi[ax] - next[ax] + pi[ax]) % pi[ax];
        }
        for (int dir_idx = 0; dir_idx <= 1; dir_idx++) {
            nbr[1][dir_idx][k] = idin[1][dir_idx] + pi[1] * (idi[2] + pi[2] * idi[3]);
            nbr[2][dir_idx][k] = idi[1] + pi[1] * (idin[2][dir_idx] + pi[2] * idi[3]);
            nbr[3][dir_idx][k] = idi[1] + pi[1] * (idi[2] + pi[2] * idin[3][dir_idx]);
        }
    }

    k = lt;
    is1 = 2 + ng[1][k] - ((pi[1] - idi[1]) * ng[1][lt]) / pi[1];
    ie1 = 1 + ng[1][k] - ((pi[1] - 1 - idi[1]) * ng[1][lt]) / pi[1];
    *n1 = 3 + ie1 - is1;
    is2 = 2 + ng[2][k] - ((pi[2] - idi[2]) * ng[2][lt]) / pi[2];
    ie2 = 1 + ng[2][k] - ((pi[2] - 1 - idi[2]) * ng[2][lt]) / pi[2];
    *n2 = 3 + ie2 - is2;
    is3 = 2 + ng[3][k] - ((pi[3] - idi[3]) * ng[3][lt]) / pi[3];
    ie3 = 1 + ng[3][k] - ((pi[3] - 1 - idi[3]) * ng[3][lt]) / pi[3];
    *n3 = 3 + ie3 - is3;

    ir[lt] = 1;
    for (j = lt - 1; j >= 1; j--) {
        ir[j] = ir[j + 1] + m1[j + 1] * m2[j + 1] * m3[j + 1];
    }

    *k_out = lt;
}

/*****************************************************************/
/* Core stencil kernels (zero3, psinv, resid, rprj3, interp)     */
/*****************************************************************/
static void zero3(double *z_arr, int n1, int n2, int n3) {
    size_t total = (size_t)n1 * (size_t)n2 * (size_t)n3;
    for (size_t i = 0; i < total; i++) {
        z_arr[i] = 0.0;
    }
}

static void psinv(double *r_arr, double *u_arr, int n1, int n2, int n3, double c[4], int k) {
    int i3, i2, i1;
    double r1[1040], r2[1040];

    if (timeron)
        timer_start(T_PSINV);
    for (i3 = 2; i3 <= n3 - 1; i3++) {
        for (i2 = 2; i2 <= n2 - 1; i2++) {
            for (i1 = 1; i1 <= n1; i1++) {
                r1[i1] = U3(r_arr, i1, i2 - 1, i3, n1, n2) + U3(r_arr, i1, i2 + 1, i3, n1, n2) +
                         U3(r_arr, i1, i2, i3 - 1, n1, n2) + U3(r_arr, i1, i2, i3 + 1, n1, n2);
                r2[i1] =
                    U3(r_arr, i1, i2 - 1, i3 - 1, n1, n2) + U3(r_arr, i1, i2 + 1, i3 - 1, n1, n2) +
                    U3(r_arr, i1, i2 - 1, i3 + 1, n1, n2) + U3(r_arr, i1, i2 + 1, i3 + 1, n1, n2);
            }
            for (i1 = 2; i1 <= n1 - 1; i1++) {
                U3(u_arr, i1, i2, i3, n1, n2) =
                    U3(u_arr, i1, i2, i3, n1, n2) + c[0] * U3(r_arr, i1, i2, i3, n1, n2) +
                    c[1] * (U3(r_arr, i1 - 1, i2, i3, n1, n2) + U3(r_arr, i1 + 1, i2, i3, n1, n2) +
                            r1[i1]) +
                    c[2] * (r2[i1] + r1[i1 - 1] + r1[i1 + 1]);
            }
        }
    }
    if (timeron)
        timer_stop(T_PSINV);

    comm3(u_arr, n1, n2, n3, k);
}

static void resid(double *u_arr, double *v_arr, double *r_arr, int n1, int n2, int n3, double a[4],
                  int k) {
    int i3, i2, i1;
    double u1[1040], u2[1040];

    if (timeron)
        timer_start(T_RESID);
    for (i3 = 2; i3 <= n3 - 1; i3++) {
        for (i2 = 2; i2 <= n2 - 1; i2++) {
            for (i1 = 1; i1 <= n1; i1++) {
                u1[i1] = U3(u_arr, i1, i2 - 1, i3, n1, n2) + U3(u_arr, i1, i2 + 1, i3, n1, n2) +
                         U3(u_arr, i1, i2, i3 - 1, n1, n2) + U3(u_arr, i1, i2, i3 + 1, n1, n2);
                u2[i1] =
                    U3(u_arr, i1, i2 - 1, i3 - 1, n1, n2) + U3(u_arr, i1, i2 + 1, i3 - 1, n1, n2) +
                    U3(u_arr, i1, i2 - 1, i3 + 1, n1, n2) + U3(u_arr, i1, i2 + 1, i3 + 1, n1, n2);
            }
            for (i1 = 2; i1 <= n1 - 1; i1++) {
                U3(r_arr, i1, i2, i3, n1, n2) = U3(v_arr, i1, i2, i3, n1, n2) -
                                                a[0] * U3(u_arr, i1, i2, i3, n1, n2) -
                                                a[2] * (u2[i1] + u1[i1 - 1] + u1[i1 + 1]) -
                                                a[3] * (u2[i1 - 1] + u2[i1 + 1]);
            }
        }
    }
    if (timeron)
        timer_stop(T_RESID);

    comm3(r_arr, n1, n2, n3, k);
}

static void rprj3(double *r_arr, int m1k, int m2k, int m3k, double *s_arr, int m1j, int m2j,
                  int m3j, int k) {
    int j3, j2, j1, i3, i2, i1, d1, d2, d3;
    double x1[1040], y1[1040], x2, y2;

    if (timeron)
        timer_start(T_RPRJ3);
    d1 = (m1k == 3) ? 2 : 1;
    d2 = (m2k == 3) ? 2 : 1;
    d3 = (m3k == 3) ? 2 : 1;

    for (j3 = 2; j3 <= m3j - 1; j3++) {
        i3 = 2 * j3 - d3;
        for (j2 = 2; j2 <= m2j - 1; j2++) {
            i2 = 2 * j2 - d2;
            for (j1 = 2; j1 <= m1j; j1++) {
                i1 = 2 * j1 - d1;
                x1[i1 - 1] = U3(r_arr, i1 - 1, i2 - 1, i3, m1k, m2k) +
                             U3(r_arr, i1 - 1, i2 + 1, i3, m1k, m2k) +
                             U3(r_arr, i1 - 1, i2, i3 - 1, m1k, m2k) +
                             U3(r_arr, i1 - 1, i2, i3 + 1, m1k, m2k);
                y1[i1 - 1] = U3(r_arr, i1 - 1, i2 - 1, i3 - 1, m1k, m2k) +
                             U3(r_arr, i1 - 1, i2 - 1, i3 + 1, m1k, m2k) +
                             U3(r_arr, i1 - 1, i2 + 1, i3 - 1, m1k, m2k) +
                             U3(r_arr, i1 - 1, i2 + 1, i3 + 1, m1k, m2k);
            }
            for (j1 = 2; j1 <= m1j - 1; j1++) {
                i1 = 2 * j1 - d1;
                y2 = U3(r_arr, i1, i2 - 1, i3 - 1, m1k, m2k) +
                     U3(r_arr, i1, i2 - 1, i3 + 1, m1k, m2k) +
                     U3(r_arr, i1, i2 + 1, i3 - 1, m1k, m2k) +
                     U3(r_arr, i1, i2 + 1, i3 + 1, m1k, m2k);
                x2 = U3(r_arr, i1, i2 - 1, i3, m1k, m2k) + U3(r_arr, i1, i2 + 1, i3, m1k, m2k) +
                     U3(r_arr, i1, i2, i3 - 1, m1k, m2k) + U3(r_arr, i1, i2, i3 + 1, m1k, m2k);
                U3(s_arr, j1, j2, j3, m1j, m2j) =
                    0.5 * U3(r_arr, i1, i2, i3, m1k, m2k) +
                    0.25 * (U3(r_arr, i1 - 1, i2, i3, m1k, m2k) +
                            U3(r_arr, i1 + 1, i2, i3, m1k, m2k) + x2) +
                    0.125 * (x1[i1 - 1] + x1[i1 + 1] + y2) + 0.0625 * (y1[i1 - 1] + y1[i1 + 1]);
            }
        }
    }
    if (timeron)
        timer_stop(T_RPRJ3);

    comm3(s_arr, m1j, m2j, m3j, k - 1);
}

static void interp(double *z_arr, int mm1, int mm2, int mm3, double *u_arr, int n1, int n2, int n3,
                   int k) {
    int i3, i2, i1, d1, d2, d3, t1, t2, t3;
    double z1[1040], z2[1040], z3[1040];

    if (timeron)
        timer_start(T_INTERP);
    if (n1 != 3 && n2 != 3 && n3 != 3) {
        for (i3 = 1; i3 <= mm3 - 1; i3++) {
            for (i2 = 1; i2 <= mm2 - 1; i2++) {
                for (i1 = 1; i1 <= mm1; i1++) {
                    z1[i1] = U3(z_arr, i1, i2 + 1, i3, mm1, mm2) + U3(z_arr, i1, i2, i3, mm1, mm2);
                    z2[i1] = U3(z_arr, i1, i2, i3 + 1, mm1, mm2) + U3(z_arr, i1, i2, i3, mm1, mm2);
                    z3[i1] = U3(z_arr, i1, i2 + 1, i3 + 1, mm1, mm2) +
                             U3(z_arr, i1, i2, i3 + 1, mm1, mm2) + z1[i1];
                }
                for (i1 = 1; i1 <= mm1 - 1; i1++) {
                    U3(u_arr, 2 * i1 - 1, 2 * i2 - 1, 2 * i3 - 1, n1, n2) +=
                        U3(z_arr, i1, i2, i3, mm1, mm2);
                    U3(u_arr, 2 * i1, 2 * i2 - 1, 2 * i3 - 1, n1, n2) +=
                        0.5 *
                        (U3(z_arr, i1 + 1, i2, i3, mm1, mm2) + U3(z_arr, i1, i2, i3, mm1, mm2));
                }
                for (i1 = 1; i1 <= mm1 - 1; i1++) {
                    U3(u_arr, 2 * i1 - 1, 2 * i2, 2 * i3 - 1, n1, n2) += 0.5 * z1[i1];
                    U3(u_arr, 2 * i1, 2 * i2, 2 * i3 - 1, n1, n2) += 0.25 * (z1[i1] + z1[i1 + 1]);
                }
                for (i1 = 1; i1 <= mm1 - 1; i1++) {
                    U3(u_arr, 2 * i1 - 1, 2 * i2 - 1, 2 * i3, n1, n2) += 0.5 * z2[i1];
                    U3(u_arr, 2 * i1, 2 * i2 - 1, 2 * i3, n1, n2) += 0.25 * (z2[i1] + z2[i1 + 1]);
                }
                for (i1 = 1; i1 <= mm1 - 1; i1++) {
                    U3(u_arr, 2 * i1 - 1, 2 * i2, 2 * i3, n1, n2) += 0.25 * z3[i1];
                    U3(u_arr, 2 * i1, 2 * i2, 2 * i3, n1, n2) += 0.125 * (z3[i1] + z3[i1 + 1]);
                }
            }
        }
    } else {
        d1 = (n1 == 3) ? 2 : 1;
        t1 = (n1 == 3) ? 1 : 0;
        d2 = (n2 == 3) ? 2 : 1;
        t2 = (n2 == 3) ? 1 : 0;
        d3 = (n3 == 3) ? 2 : 1;
        t3 = (n3 == 3) ? 1 : 0;

        for (i3 = d3; i3 <= mm3 - 1; i3++) {
            for (i2 = d2; i2 <= mm2 - 1; i2++) {
                for (i1 = d1; i1 <= mm1 - 1; i1++) {
                    U3(u_arr, 2 * i1 - d1, 2 * i2 - d2, 2 * i3 - d3, n1, n2) +=
                        U3(z_arr, i1, i2, i3, mm1, mm2);
                }
                for (i1 = 1; i1 <= mm1 - 1; i1++) {
                    U3(u_arr, 2 * i1 - t1, 2 * i2 - d2, 2 * i3 - d3, n1, n2) +=
                        0.5 *
                        (U3(z_arr, i1 + 1, i2, i3, mm1, mm2) + U3(z_arr, i1, i2, i3, mm1, mm2));
                }
            }
            for (i2 = 1; i2 <= mm2 - 1; i2++) {
                for (i1 = d1; i1 <= mm1 - 1; i1++) {
                    U3(u_arr, 2 * i1 - d1, 2 * i2 - t2, 2 * i3 - d3, n1, n2) +=
                        0.5 *
                        (U3(z_arr, i1, i2 + 1, i3, mm1, mm2) + U3(z_arr, i1, i2, i3, mm1, mm2));
                }
                for (i1 = 1; i1 <= mm1 - 1; i1++) {
                    U3(u_arr, 2 * i1 - t1, 2 * i2 - t2, 2 * i3 - d3, n1, n2) +=
                        0.25 *
                        (U3(z_arr, i1 + 1, i2 + 1, i3, mm1, mm2) +
                         U3(z_arr, i1 + 1, i2, i3, mm1, mm2) + U3(z_arr, i1, i2 + 1, i3, mm1, mm2) +
                         U3(z_arr, i1, i2, i3, mm1, mm2));
                }
            }
        }

        for (i3 = 1; i3 <= mm3 - 1; i3++) {
            for (i2 = d2; i2 <= mm2 - 1; i2++) {
                for (i1 = d1; i1 <= mm1 - 1; i1++) {
                    U3(u_arr, 2 * i1 - d1, 2 * i2 - d2, 2 * i3 - t3, n1, n2) +=
                        0.5 *
                        (U3(z_arr, i1, i2, i3 + 1, mm1, mm2) + U3(z_arr, i1, i2, i3, mm1, mm2));
                }
                for (i1 = 1; i1 <= mm1 - 1; i1++) {
                    U3(u_arr, 2 * i1 - t1, 2 * i2 - d2, 2 * i3 - t3, n1, n2) +=
                        0.25 *
                        (U3(z_arr, i1 + 1, i2, i3 + 1, mm1, mm2) +
                         U3(z_arr, i1, i2, i3 + 1, mm1, mm2) + U3(z_arr, i1 + 1, i2, i3, mm1, mm2) +
                         U3(z_arr, i1, i2, i3, mm1, mm2));
                }
            }
            for (i2 = 1; i2 <= mm2 - 1; i2++) {
                for (i1 = d1; i1 <= mm1 - 1; i1++) {
                    U3(u_arr, 2 * i1 - d1, 2 * i2 - t2, 2 * i3 - t3, n1, n2) +=
                        0.25 *
                        (U3(z_arr, i1, i2 + 1, i3 + 1, mm1, mm2) +
                         U3(z_arr, i1, i2, i3 + 1, mm1, mm2) + U3(z_arr, i1, i2 + 1, i3, mm1, mm2) +
                         U3(z_arr, i1, i2, i3, mm1, mm2));
                }
                for (i1 = 1; i1 <= mm1 - 1; i1++) {
                    U3(u_arr, 2 * i1 - t1, 2 * i2 - t2, 2 * i3 - t3, n1, n2) +=
                        0.125 * (U3(z_arr, i1 + 1, i2 + 1, i3 + 1, mm1, mm2) +
                                 U3(z_arr, i1 + 1, i2, i3 + 1, mm1, mm2) +
                                 U3(z_arr, i1, i2 + 1, i3 + 1, mm1, mm2) +
                                 U3(z_arr, i1, i2, i3 + 1, mm1, mm2) +
                                 U3(z_arr, i1 + 1, i2 + 1, i3, mm1, mm2) +
                                 U3(z_arr, i1 + 1, i2, i3, mm1, mm2) +
                                 U3(z_arr, i1, i2 + 1, i3, mm1, mm2) +
                                 U3(z_arr, i1, i2, i3, mm1, mm2));
                }
            }
        }
    }
    if (timeron)
        timer_stop(T_INTERP);

    comm3_ex(u_arr, n1, n2, n3, k);
}

static void mg3P(double *u_arr, double *v_arr, double *r_arr, double a[4], double c[4], int n1,
                 int n2, int n3, int k_in) {
    int j, k;

    for (k = lt; k >= lb + 1; k--) {
        j = k - 1;
        rprj3(&r_arr[ir[k] - 1], m1[k], m2[k], m3[k], &r_arr[ir[j] - 1], m1[j], m2[j], m3[j], k);
    }

    k = lb;
    zero3(&u_arr[ir[k] - 1], m1[k], m2[k], m3[k]);
    psinv(&r_arr[ir[k] - 1], &u_arr[ir[k] - 1], m1[k], m2[k], m3[k], c, k);

    for (k = lb + 1; k <= lt - 1; k++) {
        j = k - 1;
        zero3(&u_arr[ir[k] - 1], m1[k], m2[k], m3[k]);
        interp(&u_arr[ir[j] - 1], m1[j], m2[j], m3[j], &u_arr[ir[k] - 1], m1[k], m2[k], m3[k], k);
        resid(&u_arr[ir[k] - 1], &r_arr[ir[k] - 1], &r_arr[ir[k] - 1], m1[k], m2[k], m3[k], a, k);
        psinv(&r_arr[ir[k] - 1], &u_arr[ir[k] - 1], m1[k], m2[k], m3[k], c, k);
    }

    j = lt - 1;
    k = lt;
    interp(&u_arr[ir[j] - 1], m1[j], m2[j], m3[j], u_arr, n1, n2, n3, k);
    resid(u_arr, v_arr, r_arr, n1, n2, n3, a, k);
    psinv(r_arr, u_arr, n1, n2, n3, c, k);
}

/*****************************************************************/
/* Norm evaluation using NCCL AllReduce                          */
/*****************************************************************/
static void norm2u3(double *r_arr, int n1, int n2, int n3, double *rnm2, double *rnmu, int nx0,
                    int ny0, int nz0) {
    double s = 0.0, a_val;
    int i3, i2, i1;
    double dn = 1.0 * nx0 * ny0 * nz0;

    if (timeron)
        timer_start(T_NORM2U3);
    *rnmu = 0.0;
    for (i3 = 2; i3 <= n3 - 1; i3++) {
        for (i2 = 2; i2 <= n2 - 1; i2++) {
            for (i1 = 2; i1 <= n1 - 1; i1++) {
                double val = U3(r_arr, i1, i2, i3, n1, n2);
                s += val * val;
                a_val = fabs(val);
                if (a_val > *rnmu)
                    *rnmu = a_val;
            }
        }
    }
    if (timeron)
        timer_stop(T_NORM2U3);

    if (timeron)
        timer_start(T_RCOMM);
    send_dbl[0] = *rnmu;
    send_dbl[1] = s;
    cudaDeviceSynchronize();
    ncclGroupStart();
    ncclAllReduce(&send_dbl[0], &recv_dbl[0], 1, ncclFloat64, ncclMax, nccl_comm, cuda_stream);
    ncclAllReduce(&send_dbl[1], &recv_dbl[1], 1, ncclFloat64, ncclSum, nccl_comm, cuda_stream);
    ncclGroupEnd();
    cudaStreamSynchronize(cuda_stream);
    *rnmu = recv_dbl[0];
    s = recv_dbl[1];
    if (timeron)
        timer_stop(T_RCOMM);

    *rnm2 = sqrt(s / dn);
}

/*****************************************************************/
/* 3D Halo exchange (comm3 and comm3_ex) via NCCL                */
/*****************************************************************/
static void comm1p(int axis, double *u_arr, int n1, int n2, int n3, int kk) {
    int i3, i2, i1;
    if (axis == 1) {
        for (i3 = 2; i3 <= n3 - 1; i3++) {
            for (i2 = 2; i2 <= n2 - 1; i2++) {
                U3(u_arr, 1, i2, i3, n1, n2) = U3(u_arr, n1 - 1, i2, i3, n1, n2);
                U3(u_arr, n1, i2, i3, n1, n2) = U3(u_arr, 2, i2, i3, n1, n2);
            }
        }
    } else if (axis == 2) {
        for (i3 = 2; i3 <= n3 - 1; i3++) {
            for (i1 = 1; i1 <= n1; i1++) {
                U3(u_arr, i1, 1, i3, n1, n2) = U3(u_arr, i1, n2 - 1, i3, n1, n2);
                U3(u_arr, i1, n2, i3, n1, n2) = U3(u_arr, i1, 2, i3, n1, n2);
            }
        }
    } else if (axis == 3) {
        for (i2 = 1; i2 <= n2; i2++) {
            for (i1 = 1; i1 <= n1; i1++) {
                U3(u_arr, i1, i2, 1, n1, n2) = U3(u_arr, i1, i2, n3 - 1, n1, n2);
                U3(u_arr, i1, i2, n3, n1, n2) = U3(u_arr, i1, i2, 2, n1, n2);
            }
        }
    }
}

static void comm1p_ex(int axis, double *u_arr, int n1, int n2, int n3, int kk) {
    int i3, i2, i1, indx;

    if (take_ex[axis][kk]) {
        for (int i = 1; i <= nm2; i++) {
            buff[2][i] = 0.0;
            buff[4][i] = 0.0;
        }
        indx = 0;
        if (axis == 1) {
            for (i3 = 1; i3 <= n3; i3++)
                for (i2 = 1; i2 <= n2; i2++)
                    U3(u_arr, n1, i2, i3, n1, n2) = buff[2][++indx];
            indx = 0;
            for (i3 = 1; i3 <= n3; i3++)
                for (i2 = 1; i2 <= n2; i2++)
                    for (i1 = 1; i1 <= 2; i1++)
                        U3(u_arr, i1, i2, i3, n1, n2) = buff[4][++indx];
        } else if (axis == 2) {
            for (i3 = 1; i3 <= n3; i3++)
                for (i1 = 1; i1 <= n1; i1++)
                    U3(u_arr, i1, n2, i3, n1, n2) = buff[2][++indx];
            indx = 0;
            for (i3 = 1; i3 <= n3; i3++)
                for (i2 = 1; i2 <= 2; i2++)
                    for (i1 = 1; i1 <= n1; i1++)
                        U3(u_arr, i1, i2, i3, n1, n2) = buff[4][++indx];
        } else if (axis == 3) {
            for (i2 = 1; i2 <= n2; i2++)
                for (i1 = 1; i1 <= n1; i1++)
                    U3(u_arr, i1, i2, n3, n1, n2) = buff[2][++indx];
            indx = 0;
            for (i3 = 1; i3 <= 2; i3++)
                for (i2 = 1; i2 <= n2; i2++)
                    for (i1 = 1; i1 <= n1; i1++)
                        U3(u_arr, i1, i2, i3, n1, n2) = buff[4][++indx];
        }
    }
    if (give_ex[axis][kk]) {
        int len_pos = 0, len_neg = 0;
        if (axis == 1) {
            for (i3 = 1; i3 <= n3; i3++)
                for (i2 = 1; i2 <= n2; i2++) {
                    for (i1 = n1 - 1; i1 <= n1; i1++)
                        buff[3][++len_pos] = U3(u_arr, i1, i2, i3, n1, n2);
                    buff[1][++len_neg] = U3(u_arr, 2, i2, i3, n1, n2);
                }
        } else if (axis == 2) {
            for (i3 = 1; i3 <= n3; i3++) {
                for (i2 = n2 - 1; i2 <= n2; i2++)
                    for (i1 = 1; i1 <= n1; i1++)
                        buff[3][++len_pos] = U3(u_arr, i1, i2, i3, n1, n2);
                for (i1 = 1; i1 <= n1; i1++)
                    buff[1][++len_neg] = U3(u_arr, i1, 2, i3, n1, n2);
            }
        } else if (axis == 3) {
            for (i3 = n3 - 1; i3 <= n3; i3++)
                for (i2 = 1; i2 <= n2; i2++)
                    for (i1 = 1; i1 <= n1; i1++)
                        buff[3][++len_pos] = U3(u_arr, i1, i2, i3, n1, n2);
            for (i2 = 1; i2 <= n2; i2++)
                for (i1 = 1; i1 <= n1; i1++)
                    buff[1][++len_neg] = U3(u_arr, i1, i2, 2, n1, n2);
        }
    }
    for (int i = 1; i <= nm2; i++) {
        buff[4][i] = buff[3][i];
        buff[2][i] = buff[1][i];
    }
}

static void comm3(double *u_arr, int n1, int n2, int n3, int kk) {
    int axis, i3, i2, i1;

    if (!dead[kk]) {
        for (axis = 1; axis <= 3; axis++) {
            if (comm_size != 1) {
                int len_pos = 0, len_neg = 0;

                if (axis == 1) {
                    for (i3 = 2; i3 <= n3 - 1; i3++) {
                        for (i2 = 2; i2 <= n2 - 1; i2++) {
                            buff[3][++len_pos] = U3(u_arr, n1 - 1, i2, i3, n1, n2);
                            buff[1][++len_neg] = U3(u_arr, 2, i2, i3, n1, n2);
                        }
                    }
                } else if (axis == 2) {
                    for (i3 = 2; i3 <= n3 - 1; i3++) {
                        for (i1 = 1; i1 <= n1; i1++) {
                            buff[3][++len_pos] = U3(u_arr, i1, n2 - 1, i3, n1, n2);
                            buff[1][++len_neg] = U3(u_arr, i1, 2, i3, n1, n2);
                        }
                    }
                } else if (axis == 3) {
                    for (i2 = 1; i2 <= n2; i2++) {
                        for (i1 = 1; i1 <= n1; i1++) {
                            buff[3][++len_pos] = U3(u_arr, i1, i2, n3 - 1, n1, n2);
                            buff[1][++len_neg] = U3(u_arr, i1, i2, 2, n1, n2);
                        }
                    }
                }

                if (timeron)
                    timer_start(T_COMM3);
                cudaDeviceSynchronize();
                ncclGroupStart();
                /* dir = +1: send buff[3] to nbr(+1), recv buff[4] from nbr(-1) */
                ncclSend(&buff[3][1], len_pos, ncclFloat64, nbr[axis][1][kk], nccl_comm,
                         cuda_stream);
                ncclRecv(&buff[4][1], len_pos, ncclFloat64, nbr[axis][0][kk], nccl_comm,
                         cuda_stream);
                /* dir = -1: send buff[1] to nbr(-1), recv buff[2] from nbr(+1) */
                ncclSend(&buff[1][1], len_neg, ncclFloat64, nbr[axis][0][kk], nccl_comm,
                         cuda_stream);
                ncclRecv(&buff[2][1], len_neg, ncclFloat64, nbr[axis][1][kk], nccl_comm,
                         cuda_stream);
                ncclGroupEnd();
                cudaStreamSynchronize(cuda_stream);
                if (timeron)
                    timer_stop(T_COMM3);

                int indx_neg = 0, indx_pos = 0;
                if (axis == 1) {
                    for (i3 = 2; i3 <= n3 - 1; i3++) {
                        for (i2 = 2; i2 <= n2 - 1; i2++) {
                            U3(u_arr, n1, i2, i3, n1, n2) = buff[2][++indx_neg];
                            U3(u_arr, 1, i2, i3, n1, n2) = buff[4][++indx_pos];
                        }
                    }
                } else if (axis == 2) {
                    for (i3 = 2; i3 <= n3 - 1; i3++) {
                        for (i1 = 1; i1 <= n1; i1++) {
                            U3(u_arr, i1, n2, i3, n1, n2) = buff[2][++indx_neg];
                            U3(u_arr, i1, 1, i3, n1, n2) = buff[4][++indx_pos];
                        }
                    }
                } else if (axis == 3) {
                    for (i2 = 1; i2 <= n2; i2++) {
                        for (i1 = 1; i1 <= n1; i1++) {
                            U3(u_arr, i1, i2, n3, n1, n2) = buff[2][++indx_neg];
                            U3(u_arr, i1, i2, 1, n1, n2) = buff[4][++indx_pos];
                        }
                    }
                }
            } else {
                comm1p(axis, u_arr, n1, n2, n3, kk);
            }
        }
    } else {
        zero3(u_arr, n1, n2, n3);
    }
}

static void comm3_ex(double *u_arr, int n1, int n2, int n3, int kk) {
    int axis, i3, i2, i1;

    for (axis = 1; axis <= 3; axis++) {
        if (comm_size != 1) {
            if (!take_ex[axis][kk] && !give_ex[axis][kk])
                continue;

            int len_pos = 0, len_neg = 0;
            if (axis == 1) {
                len_neg = n2 * n3;
                len_pos = 2 * n2 * n3;
            } else if (axis == 2) {
                len_neg = n1 * n3;
                len_pos = 2 * n1 * n3;
            } else if (axis == 3) {
                len_neg = n1 * n2;
                len_pos = 2 * n1 * n2;
            }

            if (give_ex[axis][kk]) {
                int p_pos = 0, p_neg = 0;
                if (axis == 1) {
                    for (i3 = 1; i3 <= n3; i3++) {
                        for (i2 = 1; i2 <= n2; i2++) {
                            for (i1 = n1 - 1; i1 <= n1; i1++) {
                                buff[3][++p_pos] = U3(u_arr, i1, i2, i3, n1, n2);
                            }
                            buff[1][++p_neg] = U3(u_arr, 2, i2, i3, n1, n2);
                        }
                    }
                } else if (axis == 2) {
                    for (i3 = 1; i3 <= n3; i3++) {
                        for (i2 = n2 - 1; i2 <= n2; i2++) {
                            for (i1 = 1; i1 <= n1; i1++) {
                                buff[3][++p_pos] = U3(u_arr, i1, i2, i3, n1, n2);
                            }
                        }
                        for (i1 = 1; i1 <= n1; i1++) {
                            buff[1][++p_neg] = U3(u_arr, i1, 2, i3, n1, n2);
                        }
                    }
                } else if (axis == 3) {
                    for (i3 = n3 - 1; i3 <= n3; i3++) {
                        for (i2 = 1; i2 <= n2; i2++) {
                            for (i1 = 1; i1 <= n1; i1++) {
                                buff[3][++p_pos] = U3(u_arr, i1, i2, i3, n1, n2);
                            }
                        }
                    }
                    for (i2 = 1; i2 <= n2; i2++) {
                        for (i1 = 1; i1 <= n1; i1++) {
                            buff[1][++p_neg] = U3(u_arr, i1, i2, 2, n1, n2);
                        }
                    }
                }
            }

            if (timeron)
                timer_start(T_COMM3);
            cudaDeviceSynchronize();
            ncclGroupStart();
            if (give_ex[axis][kk]) {
                ncclSend(&buff[3][1], len_pos, ncclFloat64, nbr[axis][1][kk], nccl_comm,
                         cuda_stream);
            }
            if (take_ex[axis][kk]) {
                ncclRecv(&buff[4][1], len_pos, ncclFloat64, nbr[axis][0][kk], nccl_comm,
                         cuda_stream);
            }
            if (give_ex[axis][kk]) {
                ncclSend(&buff[1][1], len_neg, ncclFloat64, nbr[axis][0][kk], nccl_comm,
                         cuda_stream);
            }
            if (take_ex[axis][kk]) {
                ncclRecv(&buff[2][1], len_neg, ncclFloat64, nbr[axis][1][kk], nccl_comm,
                         cuda_stream);
            }
            ncclGroupEnd();
            cudaStreamSynchronize(cuda_stream);
            if (timeron)
                timer_stop(T_COMM3);

            if (take_ex[axis][kk]) {
                int indx_neg = 0, indx_pos = 0;
                if (axis == 1) {
                    for (i3 = 1; i3 <= n3; i3++) {
                        for (i2 = 1; i2 <= n2; i2++) {
                            U3(u_arr, n1, i2, i3, n1, n2) = buff[2][++indx_neg];
                            for (i1 = 1; i1 <= 2; i1++) {
                                U3(u_arr, i1, i2, i3, n1, n2) = buff[4][++indx_pos];
                            }
                        }
                    }
                } else if (axis == 2) {
                    for (i3 = 1; i3 <= n3; i3++) {
                        for (i1 = 1; i1 <= n1; i1++) {
                            U3(u_arr, i1, n2, i3, n1, n2) = buff[2][++indx_neg];
                        }
                        for (i2 = 1; i2 <= 2; i2++) {
                            for (i1 = 1; i1 <= n1; i1++) {
                                U3(u_arr, i1, i2, i3, n1, n2) = buff[4][++indx_pos];
                            }
                        }
                    }
                } else if (axis == 3) {
                    for (i2 = 1; i2 <= n2; i2++) {
                        for (i1 = 1; i1 <= n1; i1++) {
                            U3(u_arr, i1, i2, n3, n1, n2) = buff[2][++indx_neg];
                        }
                    }
                    for (i3 = 1; i3 <= 2; i3++) {
                        for (i2 = 1; i2 <= n2; i2++) {
                            for (i1 = 1; i1 <= n1; i1++) {
                                U3(u_arr, i1, i2, i3, n1, n2) = buff[4][++indx_pos];
                            }
                        }
                    }
                }
            }
        } else {
            comm1p_ex(axis, u_arr, n1, n2, n3, kk);
        }
    }
}

/*****************************************************************/
/* Random right-hand side initialization (zran3, power, bubble)  */
/*****************************************************************/
static double power_mg(double a_val, int n1, int n2, int n3) {
    double p_val = 1.0;
    double aj = a_val;
    int nj = n3;
    int n1j = n1;
    int n2j = n2;

    for (;;) {
        if (n2j > 0) {
            if ((n2j % 2) == 1)
                nj += n1j;
            n2j /= 2;
        } else if (nj == 0) {
            break;
        }
        if ((nj % 2) == 1)
            randlc(&p_val, aj);
        randlc(&aj, aj);
        nj /= 2;
    }
    return p_val;
}

static void bubble(double ten[11][2], int j1[11][2], int j2[11][2], int j3[11][2], int m_len,
                   int ind) {
    double temp;
    int i, j_temp;

    if (ind == 1) {
        for (i = 1; i <= m_len - 1; i++) {
            if (ten[i][ind] > ten[i + 1][ind]) {
                temp = ten[i + 1][ind];
                ten[i + 1][ind] = ten[i][ind];
                ten[i][ind] = temp;

                j_temp = j1[i + 1][ind];
                j1[i + 1][ind] = j1[i][ind];
                j1[i][ind] = j_temp;

                j_temp = j2[i + 1][ind];
                j2[i + 1][ind] = j2[i][ind];
                j2[i][ind] = j_temp;

                j_temp = j3[i + 1][ind];
                j3[i + 1][ind] = j3[i][ind];
                j3[i][ind] = j_temp;
            } else {
                return;
            }
        }
    } else {
        for (i = 1; i <= m_len - 1; i++) {
            if (ten[i][ind] < ten[i + 1][ind]) {
                temp = ten[i + 1][ind];
                ten[i + 1][ind] = ten[i][ind];
                ten[i][ind] = temp;

                j_temp = j1[i + 1][ind];
                j1[i + 1][ind] = j1[i][ind];
                j1[i][ind] = j_temp;

                j_temp = j2[i + 1][ind];
                j2[i + 1][ind] = j2[i][ind];
                j2[i][ind] = j_temp;

                j_temp = j3[i + 1][ind];
                j3[i + 1][ind] = j3[i][ind];
                j3[i][ind] = j_temp;
            } else {
                return;
            }
        }
    }
}

static void zran3(double *z_arr, int n1, int n2, int n3, int nx_val, int ny_val, int k) {
    const int mm = 10;
    const double a_val = 1220703125.0; /* 5^13 */
    const double x_init = 314159265.0;
    double ten[11][2], best;
    int i, j1[11][2], j2[11][2], j3[11][2];
    int jg[4][11][2];
    int i1, i2, i3, d1, e2, e3, i0, m0, m1_idx;
    double xx, x0, x1, a1, a2, ai;

    a1 = power_mg(a_val, nx_val, 1, 0);
    a2 = power_mg(a_val, nx_val, ny_val, 0);

    zero3(z_arr, n1, n2, n3);

    ai = power_mg(a_val, nx_val, is2 - 2 + ny_val * (is3 - 2), is1 - 2);
    d1 = ie1 - is1 + 1;
    e2 = ie2 - is2 + 2;
    e3 = ie3 - is3 + 2;
    x0 = x_init;
    randlc(&x0, ai);

    for (i3 = 2; i3 <= e3; i3++) {
        x1 = x0;
        for (i2 = 2; i2 <= e2; i2++) {
            xx = x1;
            vranlc(d1, &xx, a_val, &U3(z_arr, 2, i2, i3, n1, n2));
            randlc(&x1, a1);
        }
        randlc(&x0, a2);
    }

    for (i = 1; i <= mm; i++) {
        ten[i][1] = 0.0;
        j1[i][1] = 0;
        j2[i][1] = 0;
        j3[i][1] = 0;
        ten[i][0] = 1.0;
        j1[i][0] = 0;
        j2[i][0] = 0;
        j3[i][0] = 0;
    }

    for (i3 = 2; i3 <= n3 - 1; i3++) {
        for (i2 = 2; i2 <= n2 - 1; i2++) {
            for (i1 = 2; i1 <= n1 - 1; i1++) {
                double zv = U3(z_arr, i1, i2, i3, n1, n2);
                if (zv > ten[1][1]) {
                    ten[1][1] = zv;
                    j1[1][1] = i1;
                    j2[1][1] = i2;
                    j3[1][1] = i3;
                    bubble(ten, j1, j2, j3, mm, 1);
                }
                if (zv < ten[1][0]) {
                    ten[1][0] = zv;
                    j1[1][0] = i1;
                    j2[1][0] = i2;
                    j3[1][0] = i3;
                    bubble(ten, j1, j2, j3, mm, 0);
                }
            }
        }
    }

    i1 = mm;
    i0 = mm;
    for (i = mm; i >= 1; i--) {
        double local_best1 =
            (j1[i1][1] > 0) ? U3(z_arr, j1[i1][1], j2[i1][1], j3[i1][1], n1, n2) : 0.0;
        send_dbl[0] = local_best1;
        cudaDeviceSynchronize();
        ncclAllReduce(send_dbl, recv_dbl, 1, ncclFloat64, ncclMax, nccl_comm, cuda_stream);
        cudaStreamSynchronize(cuda_stream);
        best = recv_dbl[0];

        if (best == local_best1) {
            jg[0][i][1] = me;
            jg[1][i][1] = is1 - 2 + j1[i1][1];
            jg[2][i][1] = is2 - 2 + j2[i1][1];
            jg[3][i][1] = is3 - 2 + j3[i1][1];
            i1--;
        } else {
            jg[0][i][1] = 0;
            jg[1][i][1] = 0;
            jg[2][i][1] = 0;
            jg[3][i][1] = 0;
        }
        ten[i][1] = best;
        for (int c = 0; c < 4; c++)
            send_int[c] = jg[c][i][1];
        cudaDeviceSynchronize();
        ncclAllReduce(send_int, recv_int, 4, ncclInt, ncclMax, nccl_comm, cuda_stream);
        cudaStreamSynchronize(cuda_stream);
        for (int c = 0; c < 4; c++)
            jg[c][i][1] = recv_int[c];

        double local_best0 =
            (j1[i0][0] > 0) ? U3(z_arr, j1[i0][0], j2[i0][0], j3[i0][0], n1, n2) : 1.0;
        send_dbl[0] = local_best0;
        cudaDeviceSynchronize();
        ncclAllReduce(send_dbl, recv_dbl, 1, ncclFloat64, ncclMin, nccl_comm, cuda_stream);
        cudaStreamSynchronize(cuda_stream);
        best = recv_dbl[0];

        if (best == local_best0) {
            jg[0][i][0] = me;
            jg[1][i][0] = is1 - 2 + j1[i0][0];
            jg[2][i][0] = is2 - 2 + j2[i0][0];
            jg[3][i][0] = is3 - 2 + j3[i0][0];
            i0--;
        } else {
            jg[0][i][0] = 0;
            jg[1][i][0] = 0;
            jg[2][i][0] = 0;
            jg[3][i][0] = 0;
        }
        ten[i][0] = best;
        for (int c = 0; c < 4; c++)
            send_int[c] = jg[c][i][0];
        cudaDeviceSynchronize();
        ncclAllReduce(send_int, recv_int, 4, ncclInt, ncclMax, nccl_comm, cuda_stream);
        cudaStreamSynchronize(cuda_stream);
        for (int c = 0; c < 4; c++)
            jg[c][i][0] = recv_int[c];
    }
    m1_idx = i1 + 1;
    m0 = i0 + 1;

    zero3(z_arr, n1, n2, n3);
    for (i = mm; i >= m0; i--) {
        U3(z_arr, j1[i][0], j2[i][0], j3[i][0], n1, n2) = -1.0;
    }
    for (i = mm; i >= m1_idx; i--) {
        U3(z_arr, j1[i][1], j2[i][1], j3[i][1], n1, n2) = +1.0;
    }
    comm3(z_arr, n1, n2, n3, k);
}

/*****************************************************************/
/* Main program                                                  */
/*****************************************************************/
int main(int argc, char **argv) {
    int k, it, i, n1, n2, n3, active;
    double rnm2 = 0.0, rnmu = 0.0, epsilon;
    double t, t0, mflops, err;
    double a_coeff[4], c_coeff[4];
    int verified = 0;

    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    /* Initialize ranks from SLURM or args */
    char *slurm_id = getenv("SLURM_PROCID");
    char *slurm_tasks = getenv("SLURM_NTASKS");
    if (slurm_id && slurm_tasks) {
        my_rank = atoi(slurm_id);
        np_total = atoi(slurm_tasks);
    } else if (argc >= 3) {
        my_rank = atoi(argv[1]);
        np_total = atoi(argv[2]);
    } else {
        my_rank = 0;
        np_total = 1;
    }

    printf("Rank %d/%d starting (SLURM_PROCID=%s, SLURM_NTASKS=%s, argc=%d)\n", my_rank, np_total,
           slurm_id ? slurm_id : "NULL", slurm_tasks ? slurm_tasks : "NULL", argc);

    if (np_total <= 0) {
        printf("ERROR: np_total is %d (must be > 0). Exiting.\n", np_total);
        exit(1);
    }

    /* Initialize CUDA device */
    int deviceCount = 0;
    cudaError_t cuda_err = cudaGetDeviceCount(&deviceCount);
    if (cuda_err != cudaSuccess || deviceCount <= 0) {
        printf("WARNING: cudaGetDeviceCount reported %d devices. Assuming 1 device.\n",
               deviceCount);
        deviceCount = 1;
    }

    /* By default, map rank directly to local device when invoked with args */
    int local_rank = my_rank;
    if (argc < 3) {
        const char *local_env = getenv("SLURM_LOCALID");
        if (local_env == NULL)
            local_env = getenv("LOCAL_RANK");
        if (local_env != NULL)
            local_rank = atoi(local_env);
    }
    printf("Rank %d using device %d of %d available devices\n", my_rank, local_rank % deviceCount,
           deviceCount);
    cudaSetDevice(local_rank % deviceCount);
    cudaStreamCreate(&cuda_stream);

    if (np_total < MIN_PROCS || np_total > MAX_PROCS) {
        if (my_rank == 0)
            printf("\n ERROR: number of processes %d not within range %d-%d\n Exiting program!\n\n",
                   np_total, MIN_PROCS, MAX_PROCS);
        exit(1);
    }

    for (comm_size = 1; comm_size < np_total; comm_size *= 2)
        ;
    if (comm_size > np_total)
        comm_size /= 2;

    active = 1;
    if (comm_size != np_total) {
        if (my_rank == 0) {
            char *ep = getenv("NPB_NPROCS_STRICT");
            if (ep && *ep) {
                if (strchr("nNfF-", *ep) || strcmp(ep, "0") == 0 || strcmp(ep, "off") == 0 ||
                    strcmp(ep, "OFF") == 0)
                    active = 0;
            }
        }
        if (active) {
            if (my_rank == 0)
                printf("\n ERROR: Number of processes (%d) is not a power of two (%d?)\n Exiting "
                       "program!\n\n",
                       np_total, comm_size);
            exit(1);
        }
        active = (my_rank >= comm_size) ? 0 : 1;
    }

    if (!active) {
        exit(0);
    }

    me = my_rank;

    /* Initialize NCCL for active ranks */
    const char *id_file = getenv("MG_NCCL_ID_FILE");
    if (id_file == NULL || id_file[0] == '\0')
        id_file = getenv("IS_NCCL_ID_FILE");
    if (id_file == NULL || id_file[0] == '\0')
        id_file = "/tmp/mg_nccl_id.bin";

    if (my_rank == 0) {
        ncclGetUniqueId(&nccl_id);
        FILE *f = fopen(id_file, "wb");
        if (f) {
            fwrite(&nccl_id, sizeof(nccl_id), 1, f);
            fclose(f);
        }
    } else {
        FILE *f = NULL;
        while ((f = fopen(id_file, "rb")) == NULL) {
            usleep(100000);
        }
        usleep(200000);
        fread(&nccl_id, sizeof(nccl_id), 1, f);
        fclose(f);
    }
    ncclCommInitRank(&nccl_comm, comm_size, nccl_id, my_rank);

    for (i = 1; i <= T_LAST; i++) {
        timer_clear(i);
    }
    timer_start(T_INIT);

    if (me == 0) {
        printf("\n\n NAS Parallel Benchmarks 3.4 -- MG Benchmark\n\n");
        timeron = check_timer_flag();
    }

    lt = LT_DEFAULT;
    nit = NIT_DEFAULT;
    nx[lt] = NX_DEFAULT;
    ny[lt] = NY_DEFAULT;
    nz[lt] = NZ_DEFAULT;
    for (i = 0; i <= 7; i++)
        debug_vec[i] = 0;

    alloc_space();

    a_coeff[0] = -8.0 / 3.0;
    a_coeff[1] = 0.0;
    a_coeff[2] = 1.0 / 6.0;
    a_coeff[3] = 1.0 / 12.0;

    if (CLASS == 'A' || CLASS == 'S' || CLASS == 'W') {
        c_coeff[0] = -3.0 / 8.0;
        c_coeff[1] = +1.0 / 32.0;
        c_coeff[2] = -1.0 / 64.0;
        c_coeff[3] = 0.0;
    } else {
        c_coeff[0] = -3.0 / 17.0;
        c_coeff[1] = +1.0 / 33.0;
        c_coeff[2] = -1.0 / 61.0;
        c_coeff[3] = 0.0;
    }
    lb = 1;
    k = lt;

    setup(&n1, &n2, &n3, &k);
    zero3(u, n1, n2, n3);
    zran3(v, n1, n2, n3, nx[lt], ny[lt], k);
    norm2u3(v, n1, n2, n3, &rnm2, &rnmu, nx[lt], ny[lt], nz[lt]);

    if (me == 0) {
        printf(" Size: %4dx%4dx%4d  (class %c)\n", nx[lt], ny[lt], nz[lt], CLASS);
        printf(" Iterations: %4d\n", nit);
        printf(" Total number of processes: %6d\n\n", np_total);
    }

    resid(u, v, r, n1, n2, n3, a_coeff, k);
    norm2u3(r, n1, n2, n3, &rnm2, &rnmu, nx[lt], ny[lt], nz[lt]);

    /* One untimed iteration for startup */
    mg3P(u, v, r, a_coeff, c_coeff, n1, n2, n3, k);
    resid(u, v, r, n1, n2, n3, a_coeff, k);
    setup(&n1, &n2, &n3, &k);
    zero3(u, n1, n2, n3);
    zran3(v, n1, n2, n3, nx[lt], ny[lt], k);

    timer_stop(T_INIT);
    if (me == 0) {
        printf("\n Initialization time: %15.3f seconds\n\n", timer_read(T_INIT));
    }

    for (i = 1; i <= T_LAST; i++) {
        if (i != T_INIT)
            timer_clear(i);
    }

    const char *sleep_env = getenv("MG_ITER_SLEEP");
    if (!sleep_env)
        sleep_env = getenv("IS_ITER_SLEEP");
    unsigned iter_sleep = sleep_env ? (unsigned)atoi(sleep_env) : 1;

    const char *repeat_env = getenv("MG_REPEAT");
    if (!repeat_env)
        repeat_env = getenv("IS_REPEAT");
    const char *target_env = getenv("MG_TARGET_SECONDS");
    if (!target_env)
        target_env = getenv("IS_TARGET_SECONDS");
    double target = target_env ? atof(target_env) : 0.0;
    int repeat = repeat_env ? atoi(repeat_env) : 1;
    if (repeat < 1)
        repeat = 1;

    if (me == 0) {
        printf("   iteration\n");
    }

    timer_start(T_BENCH);

    for (int pass = 0; pass < repeat; pass++) {
        double pass_start = mg_wtime();
        if (pass > 0) {
            zero3(u, n1, n2, n3);
        }
        resid(u, v, r, n1, n2, n3, a_coeff, k);
        norm2u3(r, n1, n2, n3, &rnm2, &rnmu, nx[lt], ny[lt], nz[lt]);

        for (it = 1; it <= nit; it++) {
            if (me == 0) {
                printf("  iter %4d\n", pass * nit + it);
            }
            mg3P(u, v, r, a_coeff, c_coeff, n1, n2, n3, k);
            resid(u, v, r, n1, n2, n3, a_coeff, k);
            if (iter_sleep > 0)
                sleep(iter_sleep);
        }

        if (pass == 0 && target > 0) {
            double pass_t = mg_wtime() - pass_start;
            send_dbl[0] = pass_t;
            cudaDeviceSynchronize();
            ncclAllReduce(send_dbl, recv_dbl, 1, ncclFloat64, ncclMax, nccl_comm, cuda_stream);
            cudaStreamSynchronize(cuda_stream);
            pass_t = recv_dbl[0];
            repeat = pass_t > 0 ? (int)ceil(target / pass_t) : 1;
            if (repeat < 1)
                repeat = 1;
            if (repeat > 1000000)
                repeat = 1000000;
            if (me == 0)
                printf(" MG_TARGET_SECONDS=%.1f: pass time %.4f s -> %d passes\n", target, pass_t,
                       repeat);
        }
    }

    norm2u3(r, n1, n2, n3, &rnm2, &rnmu, nx[lt], ny[lt], nz[lt]);
    timer_stop(T_BENCH);

    t0 = timer_read(T_BENCH);
    send_dbl[0] = t0;
    cudaDeviceSynchronize();
    ncclReduce(send_dbl, recv_dbl, 1, ncclFloat64, ncclMax, 0, nccl_comm, cuda_stream);
    cudaStreamSynchronize(cuda_stream);
    t = recv_dbl[0];

    if (me == 0) {
        printf("\n Benchmark completed\n");
        epsilon = 1.0e-8;
        err = fabs(rnm2 - VERIFY_VALUE) / VERIFY_VALUE;
        if (err <= epsilon) {
            verified = 1;
            printf(" VERIFICATION SUCCESSFUL\n");
            printf(" L2 Norm is %20.13e\n", rnm2);
            printf(" Error is   %20.13e\n", err);
        } else {
            verified = 0;
            printf(" VERIFICATION FAILED\n");
            printf(" L2 Norm is             %20.13e\n", rnm2);
            printf(" The correct L2 Norm is %20.13e\n", (double)VERIFY_VALUE);
        }

        double nn = 1.0 * nx[lt] * ny[lt] * nz[lt];
        if (t != 0.0) {
            mflops = 58.0 * 1.0e-6 * (double)(nit * repeat) * nn / t;
        } else {
            mflops = 0.0;
        }

        c_print_results((char *)"MG", CLASS, nx[lt], ny[lt], nz[lt], nit * repeat, comm_size,
                        np_total, t, mflops, (char *)"          floating point", verified,
                        (char *)NPBVERSION, (char *)COMPILETIME, (char *)MPICC, (char *)CLINK,
                        (char *)CMPI_LIB, (char *)CMPI_INC, (char *)CFLAGS, (char *)CLINKFLAGS);
    }

    free_space();

    ncclCommDestroy(nccl_comm);
    cudaStreamDestroy(cuda_stream);

    if (my_rank == 0)
        remove(id_file);

    return 0;
}
