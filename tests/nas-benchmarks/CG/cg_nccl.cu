/*************************************************************************
 *                                                                       *
 *       N  A  S     P A R A L L E L     B E N C H M A R K S  3.4        *
 *                                                                       *
 *                       N C C L     V E R S I O N                       *
 *                                                                       *
 *                                  C G                                  *
 *                                                                       *
 *************************************************************************
 *                                                                       *
 *   This benchmark is an NCCL port of the NAS Parallel Benchmarks 3.4   *
 *   CG (Conjugate Gradient) code. All MPI collectives and point-to-     *
 *   point exchanges have been replaced with their NCCL equivalents      *
 *   operating on pinned host memory mapped into the CUDA address space. *
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

static double cg_wtime(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#include "c_timers.h"
#include "npbparams.h"

/*****************************************************************/
/* Class-dependent parameters (NPB 3.4.4 CG)                     */
/*****************************************************************/
#if CLASS == 'S'
#define NA                1400
#define NONZER            7
#define NITER             15
#define SHIFT             10.0
#define ZETA_VERIFY_VALUE 8.5971775078648
#elif CLASS == 'W'
#define NA                7000
#define NONZER            8
#define NITER             15
#define SHIFT             12.0
#define ZETA_VERIFY_VALUE 10.362595087124
#elif CLASS == 'A'
#define NA                14000
#define NONZER            11
#define NITER             15
#define SHIFT             20.0
#define ZETA_VERIFY_VALUE 17.130235054029
#elif CLASS == 'B'
#define NA                75000
#define NONZER            13
#define NITER             75
#define SHIFT             60.0
#define ZETA_VERIFY_VALUE 22.712745482631
#elif CLASS == 'C'
#define NA                150000
#define NONZER            15
#define NITER             75
#define SHIFT             110.0
#define ZETA_VERIFY_VALUE 28.973605592845
#elif CLASS == 'D'
#define NA                1500000
#define NONZER            21
#define NITER             100
#define SHIFT             500.0
#define ZETA_VERIFY_VALUE 52.514532105794
#else
#error "Unknown CLASS for CG benchmark"
#endif

#define MIN_PROCS 1
#define MAX_PROCS 131072
#define RCOND     0.1

/*****************************************************************/
/* Timing constants                                              */
/*****************************************************************/
#define T_TOTAL  1
#define T_CONJG  2
#define T_RCOMM  3
#define T_NCOMM  4
#define T_LAST   4

/*****************************************************************/
/* Global variables                                              */
/*****************************************************************/
static int my_rank, comm_size, np_total;
static int me, nprows, npcols;
static int num_proc_rows, num_proc_cols;
static int naa, nzz, nz;
static int firstrow, lastrow, firstcol, lastcol;
static int exch_proc, exch_recv_length, send_start, send_len;
static int l2npcols;
static int reduce_exch_proc[32];
static int reduce_send_starts[32];
static int reduce_send_lengths[32];
static int reduce_recv_starts[32];
static int reduce_recv_lengths[32];

static double tran, amult;
static int timeron;

/* NCCL and CUDA handles */
static ncclComm_t nccl_comm;
static cudaStream_t cuda_stream;
static ncclUniqueId nccl_id;

/* Arrays (1-based indexing matching Fortran NPB 3.4.4 CG) */
static int *colidx;
static int *rowstr;
static int *iv;
static int *arow;
static int *acol;

static double *a;
static double *x;
static double *z;
static double *p;
static double *q;     /* pinned for NCCL */
static double *r;     /* pinned for NCCL */
static double *w;     /* pinned for NCCL */
static double *v;
static double *aelt;

static double *send_buf; /* pinned for scalar/pair NCCL exchanges */
static double *recv_buf; /* pinned for scalar/pair NCCL exchanges */

extern "C" void c_print_results(char *name, char class_npb, int n1, int n2, int n3, int niter,
                                int nprocs_active, int nprocs_total, double t, double mops,
                                char *optype, int passed_verification, char *npbversion,
                                char *compiletime, char *mpicc, char *clink, char *cmpi_lib,
                                char *cmpi_inc, char *cflags, char *clinkflags);

/*****************************************************************/
/* Random number generator (NPB randlc)                          */
/*****************************************************************/
static double randlc(double *x_seed, double a_mult) {
    const double r23 = 1.1920928955078125e-07; /* 0.5^23 */
    const double r46 = r23 * r23;
    const double t23 = 8388608.0;              /* 2.0^23 */
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

/*****************************************************************/
/* Setup processor grid                                          */
/*****************************************************************/
static void setup_proc_info(void) {
    int i = comm_size;
    int log2nprocs = 0;

    while (i > 1) {
        log2nprocs++;
        i /= 2;
    }

    num_proc_rows = 1 << (log2nprocs / 2);
    num_proc_cols = 1 << (log2nprocs - log2nprocs / 2);
    nprows = num_proc_rows;
    npcols = num_proc_cols;
    me = my_rank;
}

/*****************************************************************/
/* Setup submatrix partitioning and butterfly schedule           */
/*****************************************************************/
static void setup_submatrix_info(void) {
    int col_size, row_size;
    int i, j, div_factor;
    int proc_row, proc_col;

    proc_row = me / npcols;
    proc_col = me - proc_row * npcols;

    if ((naa / npcols) * npcols == naa) {
        col_size = naa / npcols;
        firstcol = proc_col * col_size + 1;
        lastcol = firstcol - 1 + col_size;
        row_size = naa / nprows;
        firstrow = proc_row * row_size + 1;
        lastrow = firstrow - 1 + row_size;
    } else {
        if (proc_row < naa - (naa / nprows) * nprows) {
            row_size = naa / nprows + 1;
            firstrow = proc_row * row_size + 1;
            lastrow = firstrow - 1 + row_size;
        } else {
            row_size = naa / nprows;
            firstrow = (naa - (naa / nprows) * nprows) * (row_size + 1) +
                       (proc_row - (naa - (naa / nprows) * nprows)) * row_size + 1;
            lastrow = firstrow - 1 + row_size;
        }
        if (npcols == nprows) {
            if (proc_col < naa - (naa / npcols) * npcols) {
                col_size = naa / npcols + 1;
                firstcol = proc_col * col_size + 1;
                lastcol = firstcol - 1 + col_size;
            } else {
                col_size = naa / npcols;
                firstcol = (naa - (naa / npcols) * npcols) * (col_size + 1) +
                           (proc_col - (naa - (naa / npcols) * npcols)) * col_size + 1;
                lastcol = firstcol - 1 + col_size;
            }
        } else {
            if ((proc_col / 2) < naa - (naa / (npcols / 2)) * (npcols / 2)) {
                col_size = naa / (npcols / 2) + 1;
                firstcol = (proc_col / 2) * col_size + 1;
                lastcol = firstcol - 1 + col_size;
            } else {
                col_size = naa / (npcols / 2);
                firstcol = (naa - (naa / (npcols / 2)) * (npcols / 2)) * (col_size + 1) +
                           ((proc_col / 2) - (naa - (naa / (npcols / 2)) * (npcols / 2))) *
                               col_size +
                           1;
                lastcol = firstcol - 1 + col_size;
            }
            if ((me % 2) == 0) {
                lastcol = firstcol - 1 + (col_size - 1) / 2 + 1;
            } else {
                firstcol = firstcol + (col_size - 1) / 2 + 1;
                lastcol = firstcol - 1 + col_size / 2;
            }
        }
    }

    if (npcols == nprows) {
        send_start = 1;
        send_len = lastrow - firstrow + 1;
    } else {
        if ((me % 2) == 0) {
            send_start = 1;
            send_len = (1 + lastrow - firstrow + 1) / 2;
        } else {
            send_start = (1 + lastrow - firstrow + 1) / 2 + 1;
            send_len = (lastrow - firstrow + 1) / 2;
        }
    }

    if (npcols == nprows) {
        exch_proc = (me % nprows) * nprows + me / nprows;
    } else {
        exch_proc = 2 * (((me / 2) % nprows) * nprows + (me / 2) / nprows) + (me % 2);
    }

    i = npcols / 2;
    l2npcols = 0;
    while (i > 0) {
        l2npcols++;
        i /= 2;
    }

    div_factor = npcols;
    for (i = 1; i <= l2npcols; i++) {
        j = ((proc_col + div_factor / 2) % div_factor) + (proc_col / div_factor) * div_factor;
        reduce_exch_proc[i] = proc_row * npcols + j;
        div_factor /= 2;
    }

    for (i = l2npcols; i >= 1; i--) {
        if (nprows == npcols) {
            reduce_send_starts[i] = send_start;
            reduce_send_lengths[i] = send_len;
            reduce_recv_lengths[i] = lastrow - firstrow + 1;
        } else {
            reduce_recv_lengths[i] = send_len;
            if (i == l2npcols) {
                reduce_send_lengths[i] = lastrow - firstrow + 1 - send_len;
                if ((me / 2) * 2 == me) {
                    reduce_send_starts[i] = send_start + send_len;
                } else {
                    reduce_send_starts[i] = 1;
                }
            } else {
                reduce_send_lengths[i] = send_len;
                reduce_send_starts[i] = send_start;
            }
        }
        reduce_recv_starts[i] = send_start;
    }

    exch_recv_length = lastcol - firstcol + 1;
}

/*****************************************************************/
/* Memory allocation and cleanup                                 */
/*****************************************************************/
static void alloc_space(void) {
    /* Allocate enough space for intermediate uncompressed matrix elements in sparse() */
    long nz_calc = (long)NA * (NONZER + 1) * (NONZER + 1);
    if (nz_calc < 2000000L) {
        nz_calc = 2000000L;
    }
    nz = (int)nz_calc;
    naa = NA;
    nzz = nz;

    size_t vec_len = (size_t)(NA / num_proc_rows + 8);

    colidx = (int *)malloc((size_t)(nz + 2) * sizeof(int));
    rowstr = (int *)malloc((size_t)(NA + 4) * sizeof(int));
    iv = (int *)malloc((size_t)(2 * NA + 4) * sizeof(int));
    arow = (int *)malloc((size_t)(NA + 2) * sizeof(int));
    acol = (int *)malloc((size_t)NA * (NONZER + 1) * sizeof(int) + 16);

    a = (double *)malloc((size_t)(nz + 2) * sizeof(double));
    x = (double *)malloc(vec_len * sizeof(double));
    z = (double *)malloc(vec_len * sizeof(double));
    p = (double *)malloc(vec_len * sizeof(double));
    v = (double *)malloc((size_t)(NA + 4) * sizeof(double));
    aelt = (double *)malloc((size_t)NA * (NONZER + 1) * sizeof(double) + 16);

    cudaMallocHost((void **)&q, vec_len * sizeof(double));
    cudaMallocHost((void **)&r, vec_len * sizeof(double));
    cudaMallocHost((void **)&w, vec_len * sizeof(double));
    cudaMallocHost((void **)&send_buf, 8 * sizeof(double));
    cudaMallocHost((void **)&recv_buf, 8 * sizeof(double));

    if (!colidx || !rowstr || !iv || !arow || !acol || !a || !x || !z || !p || !v || !aelt || !q ||
        !r || !w || !send_buf || !recv_buf) {
        fprintf(stderr, "Rank %d: Failed to allocate memory for CG\n", my_rank);
        exit(1);
    }
}

static void free_space(void) {
    free(colidx);
    free(rowstr);
    free(iv);
    free(arow);
    free(acol);
    free(a);
    free(x);
    free(z);
    free(p);
    free(v);
    free(aelt);

    cudaFreeHost(q);
    cudaFreeHost(r);
    cudaFreeHost(w);
    cudaFreeHost(send_buf);
    cudaFreeHost(recv_buf);
}

/*****************************************************************/
/* Conjugate Gradient solver                                     */
/*****************************************************************/
static void conj_grad(double *rnorm) {
    int cgit, cgitmax = 25;
    int i, j, k;
    double d, sum, rho, rho0, alpha, beta;

    if (timeron)
        timer_start(T_CONJG);

    for (j = 1; j <= naa / nprows + 1; j++) {
        q[j] = 0.0;
        z[j] = 0.0;
        r[j] = x[j];
        p[j] = r[j];
        w[j] = 0.0;
    }

    /* rho = r.r */
    sum = 0.0;
    for (j = 1; j <= lastcol - firstcol + 1; j++) {
        sum = sum + r[j] * r[j];
    }
    rho = sum;

    for (i = 1; i <= l2npcols; i++) {
        if (timeron)
            timer_start(T_RCOMM);
        send_buf[0] = rho;
        cudaDeviceSynchronize();
        ncclGroupStart();
        ncclSend(send_buf, 1, ncclFloat64, reduce_exch_proc[i], nccl_comm, cuda_stream);
        ncclRecv(recv_buf, 1, ncclFloat64, reduce_exch_proc[i], nccl_comm, cuda_stream);
        ncclGroupEnd();
        cudaStreamSynchronize(cuda_stream);
        if (timeron)
            timer_stop(T_RCOMM);
        rho = rho + recv_buf[0];
    }

    /* The conj grad iteration loop */
    for (cgit = 1; cgit <= cgitmax; cgit++) {
        /* q = A.p */
        for (j = 1; j <= lastrow - firstrow + 1; j++) {
            sum = 0.0;
            for (k = rowstr[j]; k <= rowstr[j + 1] - 1; k++) {
                sum = sum + a[k] * p[colidx[k]];
            }
            w[j] = sum;
        }

        /* Sum the submatrix-vector products across columns */
        for (i = l2npcols; i >= 1; i--) {
            if (timeron)
                timer_start(T_RCOMM);
            cudaDeviceSynchronize();
            ncclGroupStart();
            ncclSend(&w[reduce_send_starts[i]], reduce_send_lengths[i], ncclFloat64,
                     reduce_exch_proc[i], nccl_comm, cuda_stream);
            ncclRecv(&q[reduce_recv_starts[i]], reduce_recv_lengths[i], ncclFloat64,
                     reduce_exch_proc[i], nccl_comm, cuda_stream);
            ncclGroupEnd();
            cudaStreamSynchronize(cuda_stream);
            if (timeron)
                timer_stop(T_RCOMM);

            for (j = send_start; j <= send_start + reduce_recv_lengths[i] - 1; j++) {
                w[j] = w[j] + q[j];
            }
        }

        /* Exchange piece of q with transpose processor */
        if (l2npcols != 0) {
            if (timeron)
                timer_start(T_RCOMM);
            cudaDeviceSynchronize();
            ncclGroupStart();
            ncclSend(&w[send_start], send_len, ncclFloat64, exch_proc, nccl_comm, cuda_stream);
            ncclRecv(&q[1], exch_recv_length, ncclFloat64, exch_proc, nccl_comm, cuda_stream);
            ncclGroupEnd();
            cudaStreamSynchronize(cuda_stream);
            if (timeron)
                timer_stop(T_RCOMM);
        } else {
            for (j = 1; j <= exch_recv_length; j++) {
                q[j] = w[j];
            }
        }

        /* Clear w for reuse */
        j = lastrow - firstrow + 1;
        if (lastcol - firstcol + 1 > j)
            j = lastcol - firstcol + 1;
        for (i = 1; i <= j; i++) {
            w[i] = 0.0;
        }

        /* Obtain p.q */
        sum = 0.0;
        for (j = 1; j <= lastcol - firstcol + 1; j++) {
            sum = sum + p[j] * q[j];
        }
        d = sum;

        for (i = 1; i <= l2npcols; i++) {
            if (timeron)
                timer_start(T_RCOMM);
            send_buf[0] = d;
            cudaDeviceSynchronize();
            ncclGroupStart();
            ncclSend(send_buf, 1, ncclFloat64, reduce_exch_proc[i], nccl_comm, cuda_stream);
            ncclRecv(recv_buf, 1, ncclFloat64, reduce_exch_proc[i], nccl_comm, cuda_stream);
            ncclGroupEnd();
            cudaStreamSynchronize(cuda_stream);
            if (timeron)
                timer_stop(T_RCOMM);
            d = d + recv_buf[0];
        }

        alpha = rho / d;
        rho0 = rho;

        for (j = 1; j <= lastcol - firstcol + 1; j++) {
            z[j] = z[j] + alpha * p[j];
            r[j] = r[j] - alpha * q[j];
        }

        /* rho = r.r */
        sum = 0.0;
        for (j = 1; j <= lastcol - firstcol + 1; j++) {
            sum = sum + r[j] * r[j];
        }
        rho = sum;

        for (i = 1; i <= l2npcols; i++) {
            if (timeron)
                timer_start(T_RCOMM);
            send_buf[0] = rho;
            cudaDeviceSynchronize();
            ncclGroupStart();
            ncclSend(send_buf, 1, ncclFloat64, reduce_exch_proc[i], nccl_comm, cuda_stream);
            ncclRecv(recv_buf, 1, ncclFloat64, reduce_exch_proc[i], nccl_comm, cuda_stream);
            ncclGroupEnd();
            cudaStreamSynchronize(cuda_stream);
            if (timeron)
                timer_stop(T_RCOMM);
            rho = rho + recv_buf[0];
        }

        beta = rho / rho0;

        for (j = 1; j <= lastcol - firstcol + 1; j++) {
            p[j] = r[j] + beta * p[j];
        }
    }

    /* Compute residual norm explicitly: ||r|| = ||x - A.z|| */
    for (j = 1; j <= lastrow - firstrow + 1; j++) {
        sum = 0.0;
        for (k = rowstr[j]; k <= rowstr[j + 1] - 1; k++) {
            sum = sum + a[k] * z[colidx[k]];
        }
        w[j] = sum;
    }

    for (i = l2npcols; i >= 1; i--) {
        if (timeron)
            timer_start(T_RCOMM);
        cudaDeviceSynchronize();
        ncclGroupStart();
        ncclSend(&w[reduce_send_starts[i]], reduce_send_lengths[i], ncclFloat64,
                 reduce_exch_proc[i], nccl_comm, cuda_stream);
        ncclRecv(&r[reduce_recv_starts[i]], reduce_recv_lengths[i], ncclFloat64,
                 reduce_exch_proc[i], nccl_comm, cuda_stream);
        ncclGroupEnd();
        cudaStreamSynchronize(cuda_stream);
        if (timeron)
            timer_stop(T_RCOMM);

        for (j = send_start; j <= send_start + reduce_recv_lengths[i] - 1; j++) {
            w[j] = w[j] + r[j];
        }
    }

    if (l2npcols != 0) {
        if (timeron)
            timer_start(T_RCOMM);
        cudaDeviceSynchronize();
        ncclGroupStart();
        ncclSend(&w[send_start], send_len, ncclFloat64, exch_proc, nccl_comm, cuda_stream);
        ncclRecv(&r[1], exch_recv_length, ncclFloat64, exch_proc, nccl_comm, cuda_stream);
        ncclGroupEnd();
        cudaStreamSynchronize(cuda_stream);
        if (timeron)
            timer_stop(T_RCOMM);
    } else {
        for (j = 1; j <= exch_recv_length; j++) {
            r[j] = w[j];
        }
    }

    sum = 0.0;
    for (j = 1; j <= lastcol - firstcol + 1; j++) {
        d = x[j] - r[j];
        sum = sum + d * d;
    }
    d = sum;

    for (i = 1; i <= l2npcols; i++) {
        if (timeron)
            timer_start(T_RCOMM);
        send_buf[0] = d;
        cudaDeviceSynchronize();
        ncclGroupStart();
        ncclSend(send_buf, 1, ncclFloat64, reduce_exch_proc[i], nccl_comm, cuda_stream);
        ncclRecv(recv_buf, 1, ncclFloat64, reduce_exch_proc[i], nccl_comm, cuda_stream);
        ncclGroupEnd();
        cudaStreamSynchronize(cuda_stream);
        if (timeron)
            timer_stop(T_RCOMM);
        d = d + recv_buf[0];
    }

    if (me == 0) {
        *rnorm = sqrt(d);
    }

    if (timeron)
        timer_stop(T_CONJG);
}

/*****************************************************************/
/* Sparse matrix generation helpers (makea, sparse, sprnvc, etc.)*/
/*****************************************************************/
static int icnvrt(double x_val, int ipwr2) {
    return (int)(ipwr2 * x_val);
}

static void vecset(int n, double v_arr[], int iv_arr[], int *nzv, int i, double val) {
    int k;
    int set = 0;
    for (k = 1; k <= *nzv; k++) {
        if (iv_arr[k] == i) {
            v_arr[k] = val;
            set = 1;
        }
    }
    if (!set) {
        *nzv = *nzv + 1;
        v_arr[*nzv] = val;
        iv_arr[*nzv] = i;
    }
}

static void sprnvc(int n, int nz_req, int nn1, double v_arr[], int iv_arr[]) {
    int nzv = 0;
    int ii, i;
    double vecelt, vecloc;

    while (nzv < nz_req) {
        vecelt = randlc(&tran, amult);
        vecloc = randlc(&tran, amult);
        i = icnvrt(vecloc, nn1) + 1;
        if (i > n)
            continue;

        int dup = 0;
        for (ii = 1; ii <= nzv; ii++) {
            if (iv_arr[ii] == i) {
                dup = 1;
                break;
            }
        }
        if (dup)
            continue;

        nzv++;
        v_arr[nzv] = vecelt;
        iv_arr[nzv] = i;
    }
}

static void sparse(double a_arr[], int colidx_arr[], int rowstr_arr[], int n, int nz_max,
                   int nonzer_val, int arow_arr[], int acol_arr[], double aelt_arr[],
                   int firstrow_val, int lastrow_val, double x_loc[], int mark[], int nzloc[],
                   double rcond_val, double shift_val) {
    int nrows = lastrow_val - firstrow_val + 1;
    int i, j, j1, j2, nza, k, kk, nzrow, jcol;
    double size, scale, ratio, va;
    int ilow = firstrow_val;
    int ihigh = lastrow_val;
    int stride = nonzer_val + 1;

    for (j = 1; j <= nrows + 1; j++) {
        rowstr_arr[j] = 0;
    }

    for (i = 1; i <= n; i++) {
        for (nza = 1; nza <= arow_arr[i]; nza++) {
            j = acol_arr[(i - 1) * stride + nza];
            if (j >= ilow && j <= ihigh) {
                j = j - ilow + 2;
                rowstr_arr[j] = rowstr_arr[j] + arow_arr[i];
            }
        }
    }

    rowstr_arr[1] = 1;
    for (j = 2; j <= nrows + 1; j++) {
        rowstr_arr[j] = rowstr_arr[j] + rowstr_arr[j - 1];
    }
    nza = rowstr_arr[nrows + 1] - 1;

    if (nza > nz_max) {
        printf("Space for matrix elements exceeded in sparse\n");
        printf("nza, nzmax = %d, %d\n", nza, nz_max);
        exit(1);
    }

    for (j = 1; j <= nrows; j++) {
        x_loc[j] = 0.0;
        mark[j] = 0;
        nzloc[j] = 0;
    }

    for (j = 1; j <= nrows; j++) {
        for (k = rowstr_arr[j]; k <= rowstr_arr[j + 1] - 1; k++) {
            a_arr[k] = 0.0;
            colidx_arr[k] = 0;
        }
    }

    size = 1.0;
    ratio = pow(rcond_val, (1.0 / (double)n));

    for (i = 1; i <= n; i++) {
        for (nza = 1; nza <= arow_arr[i]; nza++) {
            j = acol_arr[(i - 1) * stride + nza];
            if (j < ilow || j > ihigh)
                continue;

            scale = size * aelt_arr[(i - 1) * stride + nza];
            for (nzrow = 1; nzrow <= arow_arr[i]; nzrow++) {
                jcol = acol_arr[(i - 1) * stride + nzrow];
                va = aelt_arr[(i - 1) * stride + nzrow] * scale;
                if (jcol == j && j == i) {
                    va = va + rcond_val - shift_val;
                }

                for (k = rowstr_arr[j - ilow + 1]; k <= rowstr_arr[j - ilow + 2] - 1; k++) {
                    if (colidx_arr[k] > jcol) {
                        for (kk = rowstr_arr[j - ilow + 2] - 2; kk >= k; kk--) {
                            if (colidx_arr[kk] > 0) {
                                a_arr[kk + 1] = a_arr[kk];
                                colidx_arr[kk + 1] = colidx_arr[kk];
                            }
                        }
                        colidx_arr[k] = jcol;
                        a_arr[k] = 0.0;
                        goto L40;
                    } else if (colidx_arr[k] == 0) {
                        colidx_arr[k] = jcol;
                        goto L40;
                    } else if (colidx_arr[k] == jcol) {
                        nzloc[j - ilow + 1] = nzloc[j - ilow + 1] + 1;
                        goto L40;
                    }
                }
                printf("internal error in sparse: i=%d\n", i);
                exit(1);
            L40:
                a_arr[k] = a_arr[k] + va;
            }
        }
        size = size * ratio;
    }

    for (j = 2; j <= nrows; j++) {
        nzloc[j] = nzloc[j] + nzloc[j - 1];
    }

    for (j = 1; j <= nrows; j++) {
        if (j > 1) {
            j1 = rowstr_arr[j] - nzloc[j - 1];
        } else {
            j1 = 1;
        }
        j2 = rowstr_arr[j + 1] - nzloc[j] - 1;
        nza = rowstr_arr[j];
        for (k = j1; k <= j2; k++) {
            a_arr[k] = a_arr[nza];
            colidx_arr[k] = colidx_arr[nza];
            nza = nza + 1;
        }
    }
    for (j = 2; j <= nrows + 1; j++) {
        rowstr_arr[j] = rowstr_arr[j] - nzloc[j - 1];
    }
}

static void makea(int n, int nz_max, double a_arr[], int colidx_arr[], int rowstr_arr[],
                  int nonzer_val, int firstrow_val, int lastrow_val, int firstcol_val,
                  int lastcol_val, double rcond_val, int arow_arr[], int acol_arr[],
                  double aelt_arr[], double v_arr[], int iv_arr[], double shift_val) {
    int iouter, ivelt, nzv, nn1;
    int ivc[NONZER + 2];
    double vc[NONZER + 2];
    int stride = nonzer_val + 1;

    nn1 = 1;
    do {
        nn1 = 2 * nn1;
    } while (nn1 < n);

    for (iouter = 1; iouter <= n; iouter++) {
        nzv = nonzer_val;
        sprnvc(n, nzv, nn1, vc, ivc);
        vecset(n, vc, ivc, &nzv, iouter, 0.50);
        arow_arr[iouter] = nzv;
        for (ivelt = 1; ivelt <= nzv; ivelt++) {
            acol_arr[(iouter - 1) * stride + ivelt] = ivc[ivelt];
            aelt_arr[(iouter - 1) * stride + ivelt] = vc[ivelt];
        }
    }

    int nrows = lastrow_val - firstrow_val + 1;
    sparse(a_arr, colidx_arr, rowstr_arr, n, nz_max, nonzer_val, arow_arr, acol_arr, aelt_arr,
           firstrow_val, lastrow_val, v_arr, &iv_arr[0], &iv_arr[nrows], rcond_val, shift_val);
}

/*****************************************************************/
/* Main program                                                  */
/*****************************************************************/
int main(int argc, char **argv) {
    int i, j, k, it, active;
    double zeta = 0.0, rnorm = 0.0;
    double norm_temp11, norm_temp12;
    double t, tmax, mflops;
    int verified;
    double epsilon, err;

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

    /* Initialize NCCL for active ranks */
    const char *id_file = getenv("CG_NCCL_ID_FILE");
    if (id_file == NULL || id_file[0] == '\0')
        id_file = getenv("IS_NCCL_ID_FILE");
    if (id_file == NULL || id_file[0] == '\0')
        id_file = "/tmp/cg_nccl_id.bin";

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

    setup_proc_info();
    alloc_space();
    setup_submatrix_info();

    for (i = 1; i <= T_LAST; i++) {
        timer_clear(i);
    }

    if (me == 0) {
        printf("\n\n NAS Parallel Benchmarks 3.4 -- CG Benchmark\n\n");
        printf(" Size: %11d\n", NA);
        printf(" Iterations: %5d\n", NITER);
        printf(" Number of active processes: %5d\n", comm_size);
        printf(" Number of nonzeroes per row: %8d\n", NONZER);
        printf(" Eigenvalue shift: %8.3e\n", (double)SHIFT);
        timeron = check_timer_flag();
    }

    /* Initialize random number generator */
    tran = 314159265.0;
    amult = 1220703125.0;
    zeta = randlc(&tran, amult);

    /* Construct random sparse matrix A */
    makea(naa, nzz, a, colidx, rowstr, NONZER, firstrow, lastrow, firstcol, lastcol, RCOND, arow,
          acol, aelt, v, iv, SHIFT);

    /* Shift column indices to local 1-based range */
    for (j = 1; j <= lastrow - firstrow + 1; j++) {
        for (k = rowstr[j]; k <= rowstr[j + 1] - 1; k++) {
            colidx[k] = colidx[k] - firstcol + 1;
        }
    }

    /* Set starting vector to (1, 1, ..., 1) */
    for (i = 1; i <= NA / num_proc_rows + 1; i++) {
        x[i] = 1.0;
    }
    zeta = 0.0;

    /* One untimed iteration to warm up code and data pages */
    for (it = 1; it <= 1; it++) {
        conj_grad(&rnorm);

        norm_temp11 = 0.0;
        norm_temp12 = 0.0;
        for (j = 1; j <= lastcol - firstcol + 1; j++) {
            norm_temp11 = norm_temp11 + x[j] * z[j];
            norm_temp12 = norm_temp12 + z[j] * z[j];
        }

        for (i = 1; i <= l2npcols; i++) {
            send_buf[0] = norm_temp11;
            send_buf[1] = norm_temp12;
            cudaDeviceSynchronize();
            ncclGroupStart();
            ncclSend(send_buf, 2, ncclFloat64, reduce_exch_proc[i], nccl_comm, cuda_stream);
            ncclRecv(recv_buf, 2, ncclFloat64, reduce_exch_proc[i], nccl_comm, cuda_stream);
            ncclGroupEnd();
            cudaStreamSynchronize(cuda_stream);
            norm_temp11 = norm_temp11 + recv_buf[0];
            norm_temp12 = norm_temp12 + recv_buf[1];
        }

        norm_temp12 = 1.0 / sqrt(norm_temp12);
        for (j = 1; j <= lastcol - firstcol + 1; j++) {
            x[j] = norm_temp12 * z[j];
        }
    }

    /* Reset starting vector to (1, 1, ..., 1) for timed benchmark */
    for (i = 1; i <= NA / num_proc_rows + 1; i++) {
        x[i] = 1.0;
    }
    zeta = 0.0;

    for (i = 1; i <= T_LAST; i++) {
        timer_clear(i);
    }

    const char *sleep_env = getenv("CG_ITER_SLEEP");
    if (!sleep_env)
        sleep_env = getenv("IS_ITER_SLEEP");
    unsigned iter_sleep = sleep_env ? (unsigned)atoi(sleep_env) : 1;

    const char *repeat_env = getenv("CG_REPEAT");
    if (!repeat_env)
        repeat_env = getenv("IS_REPEAT");
    const char *target_env = getenv("CG_TARGET_SECONDS");
    if (!target_env)
        target_env = getenv("IS_TARGET_SECONDS");
    double target = target_env ? atof(target_env) : 0.0;
    int repeat = repeat_env ? atoi(repeat_env) : 1;
    if (repeat < 1)
        repeat = 1;

    if (me == 0) {
        printf("\n   iteration           ||r||                 zeta\n");
    }

    timer_start(T_TOTAL);

    /* Main timed inverse power iteration loop */
    for (int pass = 0; pass < repeat; pass++) {
        double pass_start = cg_wtime();
        if (pass > 0) {
            for (i = 1; i <= NA / num_proc_rows + 1; i++) {
                x[i] = 1.0;
            }
            zeta = 0.0;
        }

        for (it = 1; it <= NITER; it++) {
            conj_grad(&rnorm);

            norm_temp11 = 0.0;
            norm_temp12 = 0.0;
            for (j = 1; j <= lastcol - firstcol + 1; j++) {
                norm_temp11 = norm_temp11 + x[j] * z[j];
                norm_temp12 = norm_temp12 + z[j] * z[j];
            }

            for (i = 1; i <= l2npcols; i++) {
                if (timeron)
                    timer_start(T_NCOMM);
                send_buf[0] = norm_temp11;
                send_buf[1] = norm_temp12;
                cudaDeviceSynchronize();
                ncclGroupStart();
                ncclSend(send_buf, 2, ncclFloat64, reduce_exch_proc[i], nccl_comm, cuda_stream);
                ncclRecv(recv_buf, 2, ncclFloat64, reduce_exch_proc[i], nccl_comm, cuda_stream);
                ncclGroupEnd();
                cudaStreamSynchronize(cuda_stream);
                if (timeron)
                    timer_stop(T_NCOMM);
                norm_temp11 = norm_temp11 + recv_buf[0];
                norm_temp12 = norm_temp12 + recv_buf[1];
            }

            norm_temp12 = 1.0 / sqrt(norm_temp12);

            if (me == 0) {
                zeta = SHIFT + 1.0 / norm_temp11;
                printf("    %5d       %20.14e%20.13f\n", pass * NITER + it, rnorm, zeta);
            }

            for (j = 1; j <= lastcol - firstcol + 1; j++) {
                x[j] = norm_temp12 * z[j];
            }

            if (iter_sleep > 0)
                sleep(iter_sleep);
        }

        if (pass == 0 && target > 0) {
            double pass_t = cg_wtime() - pass_start;
            send_buf[0] = pass_t;
            cudaDeviceSynchronize();
            ncclAllReduce(send_buf, recv_buf, 1, ncclFloat64, ncclMax, nccl_comm, cuda_stream);
            cudaStreamSynchronize(cuda_stream);
            pass_t = recv_buf[0];
            repeat = pass_t > 0 ? (int)ceil(target / pass_t) : 1;
            if (repeat < 1)
                repeat = 1;
            if (repeat > 1000000)
                repeat = 1000000;
            if (me == 0)
                printf(" CG_TARGET_SECONDS=%.1f: pass time %.4f s -> %d passes\n", target, pass_t,
                       repeat);
        }
    }

    timer_stop(T_TOTAL);
    t = timer_read(T_TOTAL);

    send_buf[0] = t;
    cudaDeviceSynchronize();
    ncclReduce(send_buf, recv_buf, 1, ncclFloat64, ncclMax, 0, nccl_comm, cuda_stream);
    cudaStreamSynchronize(cuda_stream);
    tmax = recv_buf[0];

    if (me == 0) {
        printf(" Benchmark completed\n");

        epsilon = 1.0e-10;
        err = fabs(zeta - ZETA_VERIFY_VALUE) / ZETA_VERIFY_VALUE;
        if (err <= epsilon) {
            verified = 1;
            printf(" VERIFICATION SUCCESSFUL\n");
            printf(" Zeta is    %20.13e\n", zeta);
            printf(" Error is   %20.13e\n", err);
        } else {
            verified = 0;
            printf(" VERIFICATION FAILED\n");
            printf(" Zeta                %20.13e\n", zeta);
            printf(" The correct zeta is %20.13e\n", (double)ZETA_VERIFY_VALUE);
        }

        if (tmax != 0.0) {
            mflops = (double)(2 * NITER * repeat * (double)NA) *
                     (3.0 + (double)(NONZER * (NONZER + 1)) +
                      25.0 * (5.0 + (double)(NONZER * (NONZER + 1))) + 3.0) /
                     tmax / 1000000.0;
        } else {
            mflops = 0.0;
        }

        c_print_results((char *)"CG", CLASS, NA, 0, 0, NITER * repeat, comm_size, np_total, tmax,
                        mflops, (char *)"          floating point", verified, (char *)NPBVERSION,
                        (char *)COMPILETIME, (char *)MPICC, (char *)CLINK, (char *)CMPI_LIB,
                        (char *)CMPI_INC, (char *)CFLAGS, (char *)CLINKFLAGS);
    }

    free_space();

    ncclCommDestroy(nccl_comm);
    cudaStreamDestroy(cuda_stream);

    if (my_rank == 0)
        remove(id_file);

    return 0;
}
