/* The fused ETD2RK predictor and corrector — platform-neutral.
 *
 * One macro body per stage, instantiated at fp64 and fp32, so the two
 * precisions cannot drift apart. Pure loop kernels with no sparse-backend
 * dependency: the same compiled module serves the scipy, MKL and Accelerate
 * bindings of MCEq.solvers.backends.host on Mac, Linux and Windows.
 *
 * Layout: the state is the row-major (dim, K) plane of
 * MCEq.solvers.etd2.etd2_driver, so the lane axis is contiguous and the
 * inner loop walks it. `per_lane` says how to read the factors eD / hphi1 /
 * hphi2. With one integration path shared by every lane they are (dim,), one
 * value per row, and the loop lifts that value out of the lane loop so the
 * lane axis vectorises; with one path per lane they are (dim, K) in the same
 * row-major layout as the state, so they are indexed alongside it. The step
 * size is already folded into the phi factors, so a stage is an elementwise
 * product of three arrays.
 *
 * Aliasing: the output buffer is distinct from every input, which is what
 * `restrict` states. The driver holds to it — the predictor writes the
 * predictor buffer while reading the state, and the corrector writes the
 * state while reading the predictor — and an in-place call would be wrong
 * for the corrector in any case, which reads `a` after writing `x`.
 *
 * ETD2_PREDICT and ETD2_CORRECT are the formula table of
 * MCEq.solvers.numerics verbatim (PREDICTOR_EXPR, CORRECTOR_EXPR); the cupy
 * kernels of the CUDA backend are generated from those same two strings, and
 * tests/test_solvers.py pins the match.
 */

#include <stddef.h>

/* Windows: export each function from the DLL. Mirrors the pattern in
 * src/MCEq/geometry/{corsikaatm,nrlmsise00}/*.c. On non-MSVC platforms
 * the attribute resolves to nothing and default ELF/Mach-O symbol
 * visibility applies. */
#if defined(_MSC_VER) && _MSC_VER >= 1200
#  define MCEQ_EXPORT __declspec(dllexport)
#else
#  define MCEQ_EXPORT
#endif

/* a  = eD x + hphi1 F            */
#define ETD2_PREDICT(eD, x, hphi1, F) ((eD) * (x) + (hphi1) * (F))
/* x+ = a + hphi2 (F_a - F)       */
#define ETD2_CORRECT(a, hphi2, F_a, F) ((a) + (hphi2) * ((F_a) - (F)))

#define ETD2_STAGES(SUFFIX, T)                                                \
    MCEQ_EXPORT                                                               \
    void etd2_predictor_##SUFFIX(                                             \
        int dim, int K, int per_lane,                                         \
        const T *restrict eD, const T *restrict hphi1,                        \
        const T *restrict x, const T *restrict F, T *restrict a)              \
    {                                                                         \
        if (K == 1)                                                           \
        { /* single axis: one flat pass, no lane loop */                      \
            for (int i = 0; i < dim; ++i)                                     \
                a[i] = ETD2_PREDICT(eD[i], x[i], hphi1[i], F[i]);             \
            return;                                                           \
        }                                                                     \
        if (per_lane)                                                         \
        { /* factors in the state's own layout, indexed alongside it */       \
            const size_t n = (size_t)dim * (size_t)K;                         \
            for (size_t j = 0; j < n; ++j)                                    \
                a[j] = ETD2_PREDICT(eD[j], x[j], hphi1[j], F[j]);             \
            return;                                                           \
        }                                                                     \
        for (int i = 0; i < dim; ++i)                                         \
        { /* one factor per row, lifted out of the lane loop */               \
            const size_t r = (size_t)i * (size_t)K;                           \
            const T eD_i = eD[i], hphi1_i = hphi1[i];                         \
            for (int k = 0; k < K; ++k)                                       \
                a[r + k] =                                                    \
                    ETD2_PREDICT(eD_i, x[r + k], hphi1_i, F[r + k]);          \
        }                                                                     \
    }                                                                         \
                                                                              \
    MCEQ_EXPORT                                                               \
    void etd2_corrector_##SUFFIX(                                             \
        int dim, int K, int per_lane,                                         \
        const T *restrict hphi2,                                              \
        const T *restrict a, const T *restrict F_a, const T *restrict F,      \
        T *restrict x)                                                        \
    {                                                                         \
        if (K == 1)                                                           \
        {                                                                     \
            for (int i = 0; i < dim; ++i)                                     \
                x[i] = ETD2_CORRECT(a[i], hphi2[i], F_a[i], F[i]);            \
            return;                                                           \
        }                                                                     \
        if (per_lane)                                                         \
        {                                                                     \
            const size_t n = (size_t)dim * (size_t)K;                         \
            for (size_t j = 0; j < n; ++j)                                    \
                x[j] = ETD2_CORRECT(a[j], hphi2[j], F_a[j], F[j]);            \
            return;                                                           \
        }                                                                     \
        for (int i = 0; i < dim; ++i)                                         \
        {                                                                     \
            const size_t r = (size_t)i * (size_t)K;                           \
            const T hphi2_i = hphi2[i];                                       \
            for (int k = 0; k < K; ++k)                                       \
                x[r + k] =                                                    \
                    ETD2_CORRECT(a[r + k], hphi2_i, F_a[r + k], F[r + k]);    \
        }                                                                     \
    }

ETD2_STAGES(f64, double)
ETD2_STAGES(f32, float)
