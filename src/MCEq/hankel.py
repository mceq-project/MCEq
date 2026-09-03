"""Inverse zeroth-order Hankel transform.

The 2D solver evolves Hankel-mode amplitudes F(kappa); this module maps
them back to angular densities f(theta). ``MCEqRun.convert_to_theta_space``
wraps :func:`inverse_hankel_legacy` with the state-vector indexing.
"""

import numpy as np
import scipy.special
from scipy.interpolate import interp1d


def inverse_hankel_legacy(
    F_k, k_grid, theta, oversample_res=5, return_oversampled=False
):
    """Inverse zeroth-order Hankel transform via cubic interpolation +
    trapezoidal rule on a uniform oversampled k-grid.

    Args:
        F_k: shape ``(..., n_k)`` — Hankel amplitudes at ``k_grid``;
            leading axes are transformed independently.
        k_grid: shape ``(n_k,)`` — non-negative, strictly increasing.
            Non-integer grids are supported; the oversampled point count
            is ``int(max(k_grid) * oversample_res)``.
        theta: shape ``(n_theta,)`` — angles to recover at.
        oversample_res: oversampling factor for the k-grid.
        return_oversampled: also return the oversampled k-grid and
            interpolated amplitudes.

    Returns:
        ``f_theta``: shape ``(..., n_theta)`` — recovered real-space
        amplitudes; with ``return_oversampled=True`` the tuple
        ``(k_oversampled, F_oversampled, f_theta)``.
    """
    F_k = np.asarray(F_k, dtype=np.float64)
    k_grid = np.asarray(k_grid, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    oversample_pts = int(np.max(k_grid) * oversample_res)
    k_oversampled = np.linspace(k_grid.min(), k_grid.max(), oversample_pts)
    F_oversampled = interp1d(k_grid, F_k, kind="cubic", axis=-1)(k_oversampled)
    j0_kth = scipy.special.j0(np.outer(k_oversampled, theta))
    # trapezoidal rule as a weighted matrix product so batched inputs
    # reduce to one GEMM: integral J0(k theta) F(k) k dk
    w = np.gradient(k_oversampled) * k_oversampled
    w[0] *= 0.5
    w[-1] *= 0.5
    f_theta = (F_oversampled * w) @ j0_kth
    if return_oversampled:
        return k_oversampled, F_oversampled, f_theta
    return f_theta
