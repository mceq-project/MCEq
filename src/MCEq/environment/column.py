"""Slant-depth integration and the rho(X) splines shared by all atmospheres."""

from __future__ import annotations

import numpy as np


def fit_column_splines(atmosphere, rho_vec, dl_vec, h_intp, *, max_den):
    """Integrate the slant column and fit the depth/height/density splines.

    Sets ``_max_X``, ``_max_den``, ``_s_h2X`` (h -> log X), ``_s_X2rho``
    (X -> rho) and ``_s_lX2h`` (log X -> h) on *atmosphere*; all three are
    quadratic interpolating splines. *h_intp* must increase, so upgoing
    columns above a raised observation level are not supported.

    Args:
      atmosphere: the :class:`~MCEq.environment.base.EarthsAtmosphere` to update.
      rho_vec: density in g/cm**3 at each sample of *dl_vec*.
      dl_vec: path length in cm from the atmosphere top,
        ``linspace(0, path_len, n_steps)``.
      h_intp: height in cm at ``dl_vec[2:]``, in reverse path order.
      max_den: density at the atmosphere top in g/cm**3.

    Returns:
      ``X_int = cumulative_trapezoid(rho_vec, dl_vec)`` in g/cm**2, where
      ``X_int[i]`` is the depth at ``dl_vec[i+1]``.
    """
    from scipy.integrate import cumulative_trapezoid
    from scipy.interpolate import UnivariateSpline

    h_intp = np.asarray(h_intp)
    X_int = cumulative_trapezoid(rho_vec, dl_vec)

    atmosphere._max_X = X_int[-1]
    atmosphere._max_den = max_den

    log_X_intp = np.log(X_int[1:])[::-1]
    atmosphere._s_h2X = UnivariateSpline(h_intp, log_X_intp, k=2, s=0.0)
    atmosphere._s_X2rho = UnivariateSpline(X_int, rho_vec[1:], k=2, s=0.0)
    atmosphere._s_lX2h = UnivariateSpline(log_X_intp[::-1], h_intp[::-1], k=2, s=0.0)

    return X_int
