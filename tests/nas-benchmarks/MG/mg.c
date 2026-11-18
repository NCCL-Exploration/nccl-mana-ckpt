/*************************************************************************
 *                                                                       *
 *       N  A  S     P A R A L L E L     B E N C H M A R K S  3.4        *
 *                                                                       *
 *                        M P I     V E R S I O N                        *
 *                                                                       *
 *                                  M G                                  *
 *                                                                       *
 *************************************************************************/

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "c_timers.h"
#include "npbparams.h"

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

#define U3(arr, i1, i2, i3, n1, n2)                                                                \
    (arr)[((i1) - 1) + (size_t)(n1) * (((i2) - 1) + (size_t)(n2) * ((i3) - 1))]

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

static int my_rank, comm_size, np_total, me;
static int nm, nv_size, nr, nm2;
static int lt, lb, nit;
static int is1, is2, is3, ie1, ie2, ie3;

static int nx[MAXLEVEL + 1], ny[MAXLEVEL + 1], nz[MAXLEVEL + 1];
static int ir[MAXLEVEL + 1], m1[MAXLEVEL + 1], m2[MAXLEVEL + 1], m3[MAXLEVEL + 1];
static int dead[MAXLEVEL + 1];
static int give_ex[4][MAXLEVEL + 1], take_ex[4][MAXLEVEL + 1];
static int nbr[4][2][MAXLEVEL + 1];
static int debug_vec[8];
static int timeron;
static MPI_Comm comm_work;

static double *u;
static double *v;
static double *r;
static double *buff[5];

void c_print_results(char *name, char class_npb, int n1, int n2, int n3, int niter,
                     int nprocs_active, int nprocs_total, double t, double mops, char *optype,
                     int passed_verification, char *npbversion, char *compiletime, char *mpicc,
                     char *clink, char *cmpi_lib, char *cmpi_inc, char *cflags, char *clinkflags);

static void comm3(double *u_arr, int n1, int n2, int n3, int kk);
static void comm3_ex(double *u_arr, int n1, int n2, int n3, int kk);
static void zero3(double *z_arr, int n1, int n2, int n3);

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
    nv_size = (2 + (1 << ndim1)) * (2 + (1 << ndim2)) * (2 + (1 << ndim3));
    nm2 = 2 * nm * nm;
    nr = (8 * (nv_size + nm * nm + 5 * nm + 14 * LT_DEFAULT - 7 * lm)) / 7;

    u = (double *)calloc((size_t)(nr + 16), sizeof(double));
    v = (double *)calloc((size_t)(nv_size + 16), sizeof(double));
    r = (double *)calloc((size_t)(nr + 16), sizeof(double));
    for (int b = 1; b <= 4; b++) {
        buff[b] = (double *)calloc((size_t)(nm2 + 16), sizeof(double));
    }
}

static void free_space(void) {
    free(u);
    free(v);
    free(r);
    for (int b = 1; b <= 4; b++) {
        free(buff[b]);
    }
}

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

static void norm2u3(double *r_arr, int n1, int n2, int n3, double *rnm2, double *rnmu, int nx0,
                    int ny0, int nz0) {
    double s = 0.0, a_val, ss;
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
    MPI_Allreduce(rnmu, &ss, 1, MPI_DOUBLE, MPI_MAX, comm_work);
    *rnmu = ss;
    MPI_Allreduce(&s, &ss, 1, MPI_DOUBLE, MPI_SUM, comm_work);
    s = ss;
    if (timeron)
        timer_stop(T_RCOMM);

    *rnm2 = sqrt(s / dn);
}

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
    MPI_Request req[2];
    MPI_Status statuses[2];

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
                MPI_Irecv(&buff[2][1], nm2, MPI_DOUBLE, nbr[axis][1][kk], axis * 10 + 1, comm_work,
                          &req[0]);
                MPI_Irecv(&buff[4][1], nm2, MPI_DOUBLE, nbr[axis][0][kk], axis * 10 + 2, comm_work,
                          &req[1]);
                MPI_Send(&buff[3][1], len_pos, MPI_DOUBLE, nbr[axis][1][kk], axis * 10 + 2,
                         comm_work);
                MPI_Send(&buff[1][1], len_neg, MPI_DOUBLE, nbr[axis][0][kk], axis * 10 + 1,
                         comm_work);
                MPI_Waitall(2, req, statuses);
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
    MPI_Request req[2];
    MPI_Status statuses[2];

    for (axis = 1; axis <= 3; axis++) {
        if (comm_size != 1) {
            if (take_ex[axis][kk]) {
                MPI_Irecv(&buff[2][1], nm2, MPI_DOUBLE, nbr[axis][1][kk], axis * 10 + 1, comm_work,
                          &req[0]);
                MPI_Irecv(&buff[4][1], nm2, MPI_DOUBLE, nbr[axis][0][kk], axis * 10 + 2, comm_work,
                          &req[1]);
                MPI_Waitall(2, req, statuses);

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
                MPI_Send(&buff[3][1], p_pos, MPI_DOUBLE, nbr[axis][1][kk], axis * 10 + 2,
                         comm_work);
                MPI_Send(&buff[1][1], p_neg, MPI_DOUBLE, nbr[axis][0][kk], axis * 10 + 1,
                         comm_work);
            }
        } else {
            comm1p_ex(axis, u_arr, n1, n2, n3, kk);
        }
    }
}

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
    const double a_val = 1220703125.0;
    const double x_init = 314159265.0;
    double ten[11][2], best, temp;
    int i, j1[11][2], j2[11][2], j3[11][2];
    int jg[4][11][2], jg_loc[4], jg_temp[4];
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
        MPI_Allreduce(&local_best1, &temp, 1, MPI_DOUBLE, MPI_MAX, comm_work);
        best = temp;
        if (best == local_best1) {
            jg_loc[0] = me;
            jg_loc[1] = is1 - 2 + j1[i1][1];
            jg_loc[2] = is2 - 2 + j2[i1][1];
            jg_loc[3] = is3 - 2 + j3[i1][1];
            i1--;
        } else {
            jg_loc[0] = 0;
            jg_loc[1] = 0;
            jg_loc[2] = 0;
            jg_loc[3] = 0;
        }
        ten[i][1] = best;
        MPI_Allreduce(jg_loc, jg_temp, 4, MPI_INT, MPI_MAX, comm_work);
        for (int c = 0; c < 4; c++)
            jg[c][i][1] = jg_temp[c];

        double local_best0 =
            (j1[i0][0] > 0) ? U3(z_arr, j1[i0][0], j2[i0][0], j3[i0][0], n1, n2) : 1.0;
        MPI_Allreduce(&local_best0, &temp, 1, MPI_DOUBLE, MPI_MIN, comm_work);
        best = temp;
        if (best == local_best0) {
            jg_loc[0] = me;
            jg_loc[1] = is1 - 2 + j1[i0][0];
            jg_loc[2] = is2 - 2 + j2[i0][0];
            jg_loc[3] = is3 - 2 + j3[i0][0];
            i0--;
        } else {
            jg_loc[0] = 0;
            jg_loc[1] = 0;
            jg_loc[2] = 0;
            jg_loc[3] = 0;
        }
        ten[i][0] = best;
        MPI_Allreduce(jg_loc, jg_temp, 4, MPI_INT, MPI_MAX, comm_work);
        for (int c = 0; c < 4; c++)
            jg[c][i][0] = jg_temp[c];
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

int main(int argc, char **argv) {
    int k, it, i, n1, n2, n3;
    double rnm2 = 0.0, rnmu = 0.0, epsilon;
    double t, t0, mflops, err;
    double a_coeff[4], c_coeff[4];
    int verified = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np_total);

    for (comm_size = 1; comm_size < np_total; comm_size *= 2)
        ;
    if (comm_size > np_total)
        comm_size /= 2;

    int color = (my_rank < comm_size) ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, color, my_rank, &comm_work);
    if (my_rank >= comm_size) {
        MPI_Finalize();
        return 0;
    }
    me = my_rank;

    for (i = 1; i <= T_LAST; i++) {
        timer_clear(i);
    }
    timer_start(T_INIT);

    if (me == 0) {
        printf("\n\n NAS Parallel Benchmarks 3.4 -- MG Benchmark\n\n");
        timeron = check_timer_flag();
    }
    MPI_Bcast(&timeron, 1, MPI_INT, 0, comm_work);

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

    MPI_Barrier(comm_work);
    timer_start(T_BENCH);

    resid(u, v, r, n1, n2, n3, a_coeff, k);
    norm2u3(r, n1, n2, n3, &rnm2, &rnmu, nx[lt], ny[lt], nz[lt]);

    for (it = 1; it <= nit; it++) {
        if (it == 1 || it == nit || (it % 5) == 0) {
            if (me == 0)
                printf("  iter %4d\n", it);
        }
        mg3P(u, v, r, a_coeff, c_coeff, n1, n2, n3, k);
        resid(u, v, r, n1, n2, n3, a_coeff, k);
    }

    norm2u3(r, n1, n2, n3, &rnm2, &rnmu, nx[lt], ny[lt], nz[lt]);
    timer_stop(T_BENCH);

    t0 = timer_read(T_BENCH);
    MPI_Reduce(&t0, &t, 1, MPI_DOUBLE, MPI_MAX, 0, comm_work);

    free_space();

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
            mflops = 58.0 * 1.0e-6 * (double)nit * nn / t;
        } else {
            mflops = 0.0;
        }

        c_print_results("MG", CLASS, nx[lt], ny[lt], nz[lt], nit, comm_size, np_total, t, mflops,
                        "          floating point", verified, NPBVERSION, COMPILETIME, MPICC, CLINK,
                        CMPI_LIB, CMPI_INC, CFLAGS, CLINKFLAGS);
    }

    MPI_Finalize();
    return 0;
}
