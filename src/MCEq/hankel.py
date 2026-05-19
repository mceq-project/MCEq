"""Inverse zeroth-order Hankel transform implementations.

Two methods (currently one):
    inverse_hankel_legacy  — PR #48 cubic-interp + np.trapz baseline
    inverse_hankel_filon   — Filon-J₀ quadrature (Task 2.2)
"""

import numpy as np
import scipy.special
from scipy.interpolate import interp1d


def inverse_hankel_legacy(F_k, k_grid, theta, oversample_res=5):
    """Inverse zeroth-order Hankel transform via cubic interpolation +
    trapezoidal rule on a uniform oversampled k-grid.

    This reproduces PR #48's ``MCEqRun.convert_to_theta_space`` algorithm
    in standalone form. Used as a baseline for accuracy comparisons against
    the Filon-J₀ alternative (Task 2.2).

    Args:
        F_k: shape ``(n_k,)`` — Hankel amplitudes at ``k_grid``.
        k_grid: shape ``(n_k,)`` — non-negative, strictly increasing.
        theta: shape ``(n_theta,)`` — angles to recover at.
        oversample_res: oversampling factor; total samples =
            ``int(max(k_grid) * oversample_res)``.

    Returns:
        ``f_theta``: shape ``(n_theta,)`` — recovered real-space amplitudes.
    """
    F_k = np.asarray(F_k, dtype=np.float64)
    k_grid = np.asarray(k_grid, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    oversample_pts = int(np.max(k_grid) * oversample_res)
    k_oversampled = np.linspace(k_grid.min(), k_grid.max(), oversample_pts)
    F_oversampled = interp1d(k_grid, F_k, kind="cubic")(k_oversampled)
    j0_kth_k = scipy.special.j0(np.outer(k_oversampled, theta)) * k_oversampled[:, None]
    return np.trapezoid(j0_kth_k * F_oversampled[:, None], k_oversampled, axis=0)
