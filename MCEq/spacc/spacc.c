/*This collection of wrappers around Apple's Accelerate framework
simplifies calling the Sparse BLAS routines via ctypes from Python.
*/

#include <stdio.h>
#include <Accelerate/Accelerate.h>

#define SIZE_MSTORE 10

static void *mstore[SIZE_MSTORE];

void free_mstore_at(int idx)
{
    sparse_matrix_destroy(mstore[idx]);
    printf("Matrix destroyed at %i\n", idx);
}

void spacc_free_mstore()
{
    for (int i = 0; i < SIZE_MSTORE; ++i)
    {
        if (mstore[i])
        {
            free_mstore_at(i);
        }
    }
}

int spacc_gemv(double alpha, int ia, double *x, double *y)
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

void spacc_daxpy(int N, double alpha, double *x, double *y)
{
    cblas_daxpy(N, alpha, x, 1, y, 1);
}

int spacc_create_sparse_matrix(
    int store_idx, int M, int N, int nnz,
    const long long *row, const long long *col,
    const double *values)
{
    sparse_status status;
    if (store_idx < -1){
        printf("store_idx variable must be >= -1");
    }
    else if (store_idx == -1)
    {
        // find free slot
        for (int i = 0; i < SIZE_MSTORE; ++i)
        {
            if (!mstore[store_idx])
            {
                store_idx = i;
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

    printf("Matrix added at %i\n", store_idx);

    return store_idx;
}

int test()
{
    const long long row[6] = {0, 1, 2, 0, 1, 2};
    const long long col[6] = {0, 1, 2, 2, 2, 0};
    const double values[6] = {1., 10., 100., 200., 300., 4};

    spacc_create_sparse_matrix(0,
                               3, 3, 6, row, col, values);

    double phi[3] = {1., 1., 1.};
    double res[3] = {0., 0., 0.};
    spacc_gemv(1., 0, phi, res);

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
    spacc_free_mstore();

    return 0;
}

int main()
{
    return test();
}