#include <pymic_kernel.h>
#include <mkl.h>

PYMIC_KERNEL

void mceq_kernel(const int *m, int nsteps,
                 const double *phi, double *delta_phi,
                 const double *rho_inv, const double *dX,
                 const double *int_m_data, const int *int_m_ci,
                 const int *int_m_pb, const int *int_m_pe,
                 const double *dec_m_data, const int *dec_m_ci,
                 const int *dec_m_pb, const int *dec_m_pe) {

    // Create matdescra control variable
    char matdescra[6] = "G  C  ";
    char trans[1] = "n";
    int grid_step = 0;
    double one = 1.;
    double zero = 0.;
    int ione = 1;
    int step = 0;

    for (step = 0; step < nsteps; ++step) {

        // delta_phi = int_m.dot(phi)
        mkl_dcsrmv(trans, m, m, &one, matdescra,
        int_m_data, int_m_ci, int_m_pb, int_m_pe,
        phi, &zero, delta_phi);

        // delta_phi = rho_inv * dec_m.dot(phi) + delta_phi
        mkl_dcsrmv(trans, m, m, &rho_inv[step], matdescra,
        dec_m_data, dec_m_ci, dec_m_pb, dec_m_pe,
        phi, &one, delta_phi);

        // phi = delta_phi * dX + phi
        cblas_axpy(m, dX[step], delta_phi, &ione, phi, &ione);

        }
}