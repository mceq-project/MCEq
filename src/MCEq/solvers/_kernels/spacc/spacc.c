/*Interface to the SParse blas functions in Apple's ACCelerate framework,
called SPACC. This seems to be the fastest gemv on Apple Silicon Macs.

This code is part of MCEq https://github.com/afedynitch/MCEq licensed under the BSD 3-clause.

Author: Anatoli Fedynitch, 2022
*/

#include <stdio.h>
#include <stdlib.h>
#include <Accelerate/Accelerate.h>

#define SIZE_MSTORE 10
#define DEBUG false

static void *mstore[SIZE_MSTORE];

void free_mstore_at(int idx)
{
    if (idx >= 0 && idx < SIZE_MSTORE && mstore[idx])
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

// SpMM: C := alpha * A * B + C (accumulate, no beta).
// The dense layout and leading dimensions belong to the caller: CblasRowMajor
// uses row strides, CblasColMajor uses column strides. Zero C before the first
// call in a non-accumulating chain.
int gemm(int order, double alpha, int ia, int nrhs, double *B, int ldb, double *C, int ldc)
{
    if (!mstore[ia])
    {
        printf("Matrix with index %i not found.\n", ia);
        return -1;
    }

    if (sparse_matrix_product_dense_double(
            (enum CBLAS_ORDER)order, CblasNoTrans, nrhs, alpha, mstore[ia],
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

int gemm_f32(int order, float alpha, int ia, int nrhs, float *B, int ldb, float *C, int ldc)
{
    if (!mstore[ia])
    {
        printf("Matrix with index %i not found.\n", ia);
        return -1;
    }
    if (sparse_matrix_product_dense_float(
            (enum CBLAS_ORDER)order, CblasNoTrans, nrhs, alpha,
            (sparse_matrix_float)mstore[ia],
            B, ldb, C, ldc) != SPARSE_SUCCESS)
    {
        printf("Error in sparse matrix-matrix multiplication (f32).\n");
        return -1;
    }
    return 0;
}

// Build from CSR rows, using only one row of temporary 64-bit indices.
// The bulk COO insertion path requires full row/column index arrays and a
// values copy, and queues them before commit. Row insertion avoids that peak.
static int create_csr_matrix(int store_idx, int M, int N,
                             const long long *indptr, const int *indices,
                             const void *values, bool use_float)
{
    if (store_idx < -1 || store_idx >= SIZE_MSTORE)
        return -1;
    if (store_idx == -1)
    {
        for (int i = 0; i < SIZE_MSTORE; ++i)
            if (!mstore[i])
            {
                store_idx = i;
                break;
            }
        if (store_idx == -1)
        {
            printf("Matrix store full, increase SIZE_MSTORE\n");
            return -1;
        }
    }
    else
        free_mstore_at(store_idx);

    void *matrix = use_float ? (void *)sparse_matrix_create_float(M, N)
                             : (void *)sparse_matrix_create_double(M, N);
    if (!matrix)
        return -1;

    long long max_nnz = 0;
    for (int row = 0; row < M; ++row)
        if (indptr[row + 1] - indptr[row] > max_nnz)
            max_nnz = indptr[row + 1] - indptr[row];
    sparse_index *columns = malloc((max_nnz ? max_nnz : 1) * sizeof(*columns));
    if (!columns)
    {
        sparse_matrix_destroy(matrix);
        return -1;
    }
    for (int row = 0; row < M; ++row)
    {
        long long start = indptr[row], nnz = indptr[row + 1] - start;
        if (!nnz)
            continue;
        for (long long j = 0; j < nnz; ++j)
            columns[j] = indices[start + j];
        sparse_status status = use_float
            ? sparse_insert_row_float(matrix, row, nnz, (const float *)values + start, columns)
            : sparse_insert_row_double(matrix, row, nnz, (const double *)values + start, columns);
        if (status != SPARSE_SUCCESS)
        {
            free(columns);
            sparse_matrix_destroy(matrix);
            return -1;
        }
    }
    free(columns);
    if (sparse_commit(matrix) != SPARSE_SUCCESS)
    {
        sparse_matrix_destroy(matrix);
        return -1;
    }
    mstore[store_idx] = matrix;
    return store_idx;
}

int create_sparse_matrix(int store_idx, int M, int N,
                         const long long *indptr, const int *indices,
                         const double *values)
{
    return create_csr_matrix(store_idx, M, N, indptr, indices, values, false);
}

int create_sparse_matrix_f32(int store_idx, int M, int N,
                             const long long *indptr, const int *indices,
                             const float *values)
{
    return create_csr_matrix(store_idx, M, N, indptr, indices, values, true);
}
