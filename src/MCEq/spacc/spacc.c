/*Interface to the SParse blas functions in Apple's ACCelerate framework,
called SPACC. This seems to be the fastest gemv on Apple Silicon Macs.

This code is part of MCEq https://github.com/afedynitch/MCEq licensed under the BSD 3-clause.

Author: Anatoli Fedynitch, 2022
*/

#include <stdio.h>
#include <Accelerate/Accelerate.h>

#define SIZE_MSTORE 10
#define DEBUG false

static void *mstore[SIZE_MSTORE];

void free_mstore_at(int idx)
{
    if (mstore[idx])
    {
        sparse_matrix_destroy(mstore[idx]);
        mstore[idx] = NULL;
        if (DEBUG)
            printf("Matrix destroyed at %i\n", idx);
    }
}

void free_mstore()
{
    for (int i = 0; i < SIZE_MSTORE; ++i)
    {
        if (mstore[i])
        {
            free_mstore_at(i);
        }
    }
}

int gemv(double alpha, int ia, double *x, double *y)
{
    if (!mstore[ia])
    {
        printf("Matrix with index %i not found.\n", ia);
        return -1;
    }

    if (sparse_matrix_vector_product_dense_double(
            CblasNoTrans, alpha, mstore[ia], x, 1, y, 1) != SPARSE_SUCCESS)
    {
        printf("Error in sparse matrix-vector multiplication.\n");
        return -1;
    };

    return 0;
}

// SpMM: C := alpha * A * B + C  (accumulate, no beta), B is (n_rows_A, nrhs),
// C is (n_rows_A, nrhs). Column-major layout (CblasColMajor) so a single
// stride of ldX = n_rows_A walks columns the same way numpy's (dim, K)
// Fortran-contiguous layout does. Caller is responsible for zeroing C
// before the first accumulating call if a non-accumulating result is wanted.
int gemm(double alpha, int ia, int nrhs, double *B, int ldb, double *C, int ldc)
{
    if (!mstore[ia])
    {
        printf("Matrix with index %i not found.\n", ia);
        return -1;
    }

    if (sparse_matrix_product_dense_double(
            CblasColMajor, CblasNoTrans, nrhs, alpha, mstore[ia],
            B, ldb, C, ldc) != SPARSE_SUCCESS)
    {
        printf("Error in sparse matrix-matrix multiplication.\n");
        return -1;
    };

    return 0;
}

// fp32 variants of gemv/gemm. The mstore entries are typed at creation
// (sparse_matrix_create_double vs _float), so the caller has to keep
// fp32 and fp64 matrices in separate store slots. ``gemv_f32`` and
// ``gemm_f32`` cast the handle to ``sparse_matrix_float`` blindly —
// the wrapper-side Python layer enforces matching dtypes.
int gemv_f32(float alpha, int ia, float *x, float *y)
{
    if (!mstore[ia])
    {
        printf("Matrix with index %i not found.\n", ia);
        return -1;
    }
    if (sparse_matrix_vector_product_dense_float(
            CblasNoTrans, alpha, (sparse_matrix_float)mstore[ia],
            x, 1, y, 1) != SPARSE_SUCCESS)
    {
        printf("Error in sparse matrix-vector multiplication (f32).\n");
        return -1;
    }
    return 0;
}

int gemm_f32(float alpha, int ia, int nrhs, float *B, int ldb, float *C, int ldc)
{
    if (!mstore[ia])
    {
        printf("Matrix with index %i not found.\n", ia);
        return -1;
    }
    if (sparse_matrix_product_dense_float(
            CblasColMajor, CblasNoTrans, nrhs, alpha,
            (sparse_matrix_float)mstore[ia],
            B, ldb, C, ldc) != SPARSE_SUCCESS)
    {
        printf("Error in sparse matrix-matrix multiplication (f32).\n");
        return -1;
    }
    return 0;
}

// fp32 sparse-matrix construction. Mirrors ``create_sparse_matrix`` but
// targets sparse_matrix_create_float / sparse_insert_entries_float.
int create_sparse_matrix_f32(
    int store_idx, int M, int N, int nnz,
    const long long *row, const long long *col,
    const float *values)
{
    if (store_idx < -1)
    {
        printf("store_idx variable must be >= -1");
        return store_idx;
    }
    else if (store_idx == -1)
    {
        for (int i = 0; i < SIZE_MSTORE; ++i)
        {
            if (!mstore[i])
            {
                store_idx = i;
                break;
            }
        }
        if (store_idx == -1)
        {
            printf("Matrix store full, increase SIZE_MSTORE\n");
            return -1;
        }
    }
    else if (mstore[store_idx])
    {
        sparse_matrix_destroy(mstore[store_idx]);
    }

    mstore[store_idx] = (void *)sparse_matrix_create_float(M, N);
    if (sparse_insert_entries_float(
            mstore[store_idx], nnz, values, row, col) != SPARSE_SUCCESS)
    {
        printf("Failed to insert values into sparse matrix (f32).\n");
        return -1;
    }
    if (sparse_commit(mstore[store_idx]) != SPARSE_SUCCESS)
    {
        printf("Failed to commit inserted values into sparse matrix (f32).\n");
        return -1;
    }
    return store_idx;
}
// Performs y = alpha*x + y
void daxpy(int N, double alpha, double *x, double *y)
{
    cblas_daxpy(N, alpha, x, 1, y, 1);
}

// ETD2 post-apply fused kernels.
//
// Replace the 4-ufunc chain `np.multiply(eD[:, None], phc, out=a);
// np.multiply(phi1[:, None], F_phi, out=scratch); scratch *= h;
// np.add(a, scratch, out=a)` with a single pass that hits every
// (dim, K) element once. Same idea as PriNCe's `post_apply1`
// ElementwiseKernel on cupy. Cuts per-step (dim, K) memory traffic
// from ~8 ufuncs (read+write each) down to a single fused read+write,
// and replaces eD[i]+phi1[i] reads with stride-1 inner-loop scans.
//
// Layout: all (dim, K) arrays are column-major (Fortran-contiguous);
// the inner loop walks rows with stride 1. eD/phi1/phi2 may be (dim,)
// (multirhs — shared across K columns) or (dim, K) (multipath — per
// column). h may be scalar (multirhs) or (K,) (multipath).
//
// The `_multipath` variants are written as four small kernels rather
// than a single dispatch because the inner loop body changes (extra
// `eD_NK[k*dim + i]` load per cell vs the cached `eD[i]`).

// post_apply1 multirhs: a = eD * phc + h * phi1 * F_phi
// eD, phi1: (dim,) shared; h: scalar.
void etd2_post_apply1_multirhs(
    int dim, int K, double h,
    const double *eD, const double *phi1,
    const double *phc, const double *F_phi, double *a)
{
    for (int k = 0; k < K; ++k)
    {
        const double *phc_k = phc + (size_t)k * dim;
        const double *Fp_k = F_phi + (size_t)k * dim;
        double *a_k = a + (size_t)k * dim;
        for (int i = 0; i < dim; ++i)
        {
            a_k[i] = eD[i] * phc_k[i] + h * phi1[i] * Fp_k[i];
        }
    }
}

// post_apply2 multirhs: phc = a + h * phi2 * (F_a - F_phi)
void etd2_post_apply2_multirhs(
    int dim, int K, double h,
    const double *phi2,
    const double *a, const double *F_a, const double *F_phi, double *phc)
{
    for (int k = 0; k < K; ++k)
    {
        const double *a_k = a + (size_t)k * dim;
        const double *Fa_k = F_a + (size_t)k * dim;
        const double *Fp_k = F_phi + (size_t)k * dim;
        double *phc_k = phc + (size_t)k * dim;
        for (int i = 0; i < dim; ++i)
        {
            phc_k[i] = a_k[i] + h * phi2[i] * (Fa_k[i] - Fp_k[i]);
        }
    }
}

// post_apply1 multipath: a = eD_NK * phc + h_K * phi1_NK * F_phi
// eD_NK, phi1_NK: (dim, K) column-major; h_K: (K,).
void etd2_post_apply1_multipath(
    int dim, int K,
    const double *h_K,
    const double *eD_NK, const double *phi1_NK,
    const double *phc, const double *F_phi, double *a)
{
    for (int k = 0; k < K; ++k)
    {
        double h_k = h_K[k];
        const double *eD_k = eD_NK + (size_t)k * dim;
        const double *phi1_k = phi1_NK + (size_t)k * dim;
        const double *phc_k = phc + (size_t)k * dim;
        const double *Fp_k = F_phi + (size_t)k * dim;
        double *a_k = a + (size_t)k * dim;
        for (int i = 0; i < dim; ++i)
        {
            a_k[i] = eD_k[i] * phc_k[i] + h_k * phi1_k[i] * Fp_k[i];
        }
    }
}

// post_apply2 multipath: phc = a + h_K * phi2_NK * (F_a - F_phi)
void etd2_post_apply2_multipath(
    int dim, int K,
    const double *h_K,
    const double *phi2_NK,
    const double *a, const double *F_a, const double *F_phi, double *phc)
{
    for (int k = 0; k < K; ++k)
    {
        double h_k = h_K[k];
        const double *phi2_k = phi2_NK + (size_t)k * dim;
        const double *a_k = a + (size_t)k * dim;
        const double *Fa_k = F_a + (size_t)k * dim;
        const double *Fp_k = F_phi + (size_t)k * dim;
        double *phc_k = phc + (size_t)k * dim;
        for (int i = 0; i < dim; ++i)
        {
            phc_k[i] = a_k[i] + h_k * phi2_k[i] * (Fa_k[i] - Fp_k[i]);
        }
    }
}

// ---- fp32 variants of the post-apply kernels ----
// Halves memory traffic on the (dim, K) buffers vs the fp64 versions.
// Accuracy budget is enforced by the test suite — see test_etd2_fp32_stability.

void etd2_post_apply1_multirhs_f32(
    int dim, int K, float h,
    const float *eD, const float *phi1,
    const float *phc, const float *F_phi, float *a)
{
    for (int k = 0; k < K; ++k)
    {
        const float *phc_k = phc + (size_t)k * dim;
        const float *Fp_k = F_phi + (size_t)k * dim;
        float *a_k = a + (size_t)k * dim;
        for (int i = 0; i < dim; ++i)
        {
            a_k[i] = eD[i] * phc_k[i] + h * phi1[i] * Fp_k[i];
        }
    }
}

void etd2_post_apply2_multirhs_f32(
    int dim, int K, float h,
    const float *phi2,
    const float *a, const float *F_a, const float *F_phi, float *phc)
{
    for (int k = 0; k < K; ++k)
    {
        const float *a_k = a + (size_t)k * dim;
        const float *Fa_k = F_a + (size_t)k * dim;
        const float *Fp_k = F_phi + (size_t)k * dim;
        float *phc_k = phc + (size_t)k * dim;
        for (int i = 0; i < dim; ++i)
        {
            phc_k[i] = a_k[i] + h * phi2[i] * (Fa_k[i] - Fp_k[i]);
        }
    }
}

void etd2_post_apply1_multipath_f32(
    int dim, int K,
    const float *h_K,
    const float *eD_NK, const float *phi1_NK,
    const float *phc, const float *F_phi, float *a)
{
    for (int k = 0; k < K; ++k)
    {
        float h_k = h_K[k];
        const float *eD_k = eD_NK + (size_t)k * dim;
        const float *phi1_k = phi1_NK + (size_t)k * dim;
        const float *phc_k = phc + (size_t)k * dim;
        const float *Fp_k = F_phi + (size_t)k * dim;
        float *a_k = a + (size_t)k * dim;
        for (int i = 0; i < dim; ++i)
        {
            a_k[i] = eD_k[i] * phc_k[i] + h_k * phi1_k[i] * Fp_k[i];
        }
    }
}

void etd2_post_apply2_multipath_f32(
    int dim, int K,
    const float *h_K,
    const float *phi2_NK,
    const float *a, const float *F_a, const float *F_phi, float *phc)
{
    for (int k = 0; k < K; ++k)
    {
        float h_k = h_K[k];
        const float *phi2_k = phi2_NK + (size_t)k * dim;
        const float *a_k = a + (size_t)k * dim;
        const float *Fa_k = F_a + (size_t)k * dim;
        const float *Fp_k = F_phi + (size_t)k * dim;
        float *phc_k = phc + (size_t)k * dim;
        for (int i = 0; i < dim; ++i)
        {
            phc_k[i] = a_k[i] + h_k * phi2_k[i] * (Fa_k[i] - Fp_k[i]);
        }
    }
}

int create_sparse_matrix(
    int store_idx, int M, int N, int nnz,
    const long long *row, const long long *col,
    const double *values)
{
    if (store_idx < -1)
    {
        printf("store_idx variable must be >= -1");
        return store_idx;
    }
    else if (store_idx == -1)
    {
        // find free slot
        for (int i = 0; i < SIZE_MSTORE; ++i)
        {
            if (!mstore[i])
            {
                store_idx = i;
                if (DEBUG)
                    printf("Assigned free store_idx %i.\n", store_idx);
                break;
            }
        }
        if (store_idx == -1)
        {
            printf("Matrix store full, increase SIZE_MSTORE\n");
            return -1;
        }
    }
    else if (mstore[store_idx])
    {
        if (DEBUG)
            printf("Overwriting existing matrix @ store_idx %i.\n", store_idx);
        sparse_matrix_destroy(mstore[store_idx]);
    }

    mstore[store_idx] = (void *)sparse_matrix_create_double(M, N);

    if (sparse_insert_entries_double(
            mstore[store_idx], nnz,
            values,
            row,
            col) != SPARSE_SUCCESS)
    {

        printf("Failed to insert values into sparse matrix.\n");
        return -1;
    };
    if (sparse_commit(mstore[store_idx]) != SPARSE_SUCCESS)
    {
        printf("Failed to commit inserted values into sparse matrix.\n");
        return -1;
    }

    if (DEBUG)
        printf("Matrix added at %i\n", store_idx);

    return store_idx;
}

int test()
{
    const long long row[6] = {0, 1, 2, 0, 1, 2};
    const long long col[6] = {0, 1, 2, 2, 2, 0};
    const double values[6] = {1., 10., 100., 200., 300., 4};

    create_sparse_matrix(0,
                         3, 3, 6, row, col, values);

    double phi[3] = {1., 1., 1.};
    double res[3] = {0., 0., 0.};
    gemv(1., 0, phi, res);

    double expected[3] = {201., 310., 104.};

    for (int i = 0; i < 3; i++)
    {
        printf("Res: %f\n", res[i]);
        if (res[i] != expected[i])
        {
            printf("Result at %i do not match %f!=%f\n",
                   i, res[i], expected[i]);
            return -1;
        }
    }
    free_mstore();

    return 0;
}