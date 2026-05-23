/* Fused ETD2 post-apply kernels — platform-neutral.
 *
 * Lifted out of src/MCEq/spacc/spacc.c so the same kernels are available on
 * Linux/Windows for the MKL Sparse BLAS multi-RHS path (which doesn't link
 * Apple Accelerate). Same symbol names and ABI as the spacc-side copies;
 * the Mac path can switch to importing these once Stage-4 settles. No
 * dependency on any sparse backend — these are loop kernels over the
 * (dim, K) column-major buffers produced by the multi-RHS/multipath
 * solvers.
 *
 * Layout: all (dim, K) arrays are column-major (Fortran-contiguous);
 * the inner loop walks rows with stride 1. eD/phi1/phi2 may be (dim,)
 * (multirhs — shared across K columns) or (dim, K) (multipath — per
 * column). h may be scalar (multirhs) or (K,) (multipath).
 */

#include <stddef.h>

/* ---- fp64 ---- */

/* post_apply1 multirhs: a = eD * phc + h * phi1 * F_phi  (eD/phi1: (dim,)) */
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

/* post_apply2 multirhs: phc = a + h * phi2 * (F_a - F_phi)  (phi2: (dim,)) */
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

/* post_apply1 multipath: per-column h_K and per-column (dim, K) eD/phi1. */
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

/* post_apply2 multipath: per-column h_K and per-column (dim, K) phi2. */
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

/* ---- fp32 ---- */

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
