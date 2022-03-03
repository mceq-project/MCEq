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
    sparse_matrix_destroy(mstore[idx]);
    mstore[idx] = NULL;
    if (DEBUG)
        printf("Matrix destroyed at %i\n", idx);
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
// Performs y = alpha*x + y
void daxpy(int N, double alpha, double *x, double *y)
{
    cblas_daxpy(N, alpha, x, 1, y, 1);
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