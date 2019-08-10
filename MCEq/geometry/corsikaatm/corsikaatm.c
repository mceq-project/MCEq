
#include <math.h>


double
#if _MSC_VER >= 1200
__declspec(dllexport) 
#endif
planar_rho_inv(double X, double cos_theta,
               double *a, double *b, double *c,
               double *t, double *hl)
{
    /*Optimized calculation of :math:`1/\\rho(X,\\theta)` in
    planar approximation.

    This function can be used for calculations where
    :math:`\\theta < 70^\\circ`.

    Args:
        X (float): slant depth in g/cm**2
        cos_theta (float): :math:`\\cos(\\theta)`

    Returns:
        float: :math:`1/\\rho(X,\\theta)` in cm**3/g
    */

    double res = 0.0;
    double x_v = X * cos_theta;
    int layer = 0;
    int i;
    for (i = 0; i < 5; ++i)
        if (!(x_v >= t[i]))
            layer = i;

    if (layer == 4)
        res = c[4] / b[4];
    else
        res = c[layer] / (x_v - a[layer]);

    return res;
}

double
#if _MSC_VER >= 1200
__declspec(dllexport) 
#endif
corsika_get_density(double h_cm, double *a,
                    double *b, double *c,
                    double *t, double *hl)
{
    /*Optimized calculation of :math:`\\rho(h)` in
    according to CORSIKA type parameterization.

    Args:
      h_cm (float): height above surface in cm
      param (numpy.array): 5x5 parameter array from
                        :class:`CorsikaAtmosphere`

    Returns:
      float: :math:`\\rho(h)` in g/cm**3
    */

    double res = 0.0;
    int layer = 0;
    int i;
    for (i = 0; i < 5; ++i)
        if (!(h_cm <= hl[i]))
            layer = i;
    if (layer == 4)
        res = b[4] / c[4];
    else
        res = b[layer] / c[layer] * exp(-h_cm / c[layer]);

    return res;
}
double
#if _MSC_VER >= 1200
__declspec(dllexport) 
#endif
corsika_get_m_overburden(double h_cm, double *a,
                         double *b, double *c,
                         double *t, double *hl)
{
    /*Optimized calculation of :math:`\\T(h)` in
    according to CORSIKA type parameterization.

    Args:
      h_cm (float): height above surface in cm
      param (numpy.array): 5x5 parameter array from
                        :class:`CorsikaAtmosphere`

    Returns:
      float: :math:`\\rho(h)` in g/cm**3
    */

    double res = 0.0;
    int layer = 0;
    int i;
    for (i = 0; i < 5; ++i)
        if (!(h_cm <= hl[i]))
            layer = i;

    if (layer == 4)
        res = a[4] - b[4] / c[4] * h_cm;
    else
        res = a[layer] + b[layer] * exp(-h_cm / c[layer]);

    return res;
}

