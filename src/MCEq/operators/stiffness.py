"""Continuous-loss step cap of the ETD2RK step.

The largest absolute off-diagonal row sum of the assembled loss bands,
without their monotone upwind closure rows and scaled by the largest
sec(theta) eigenvalue, is the rate that caps the step at
``LOSS_STEP_SAFETY / rate`` (:ref:`solver-mathematics`, the loss cap). A
conservative step policy, not an error estimator.
"""

import numpy as np

LOSS_STEP_SAFETY = 0.5


def continuous_loss_rate(bands, secant_eigenvalues=None, upwind_rows=None):
    """The inverse-depth rate of the explicit part of the loss bands.

    Args:
      bands: the assembled loss stencils, one dense band per species.
      secant_eigenvalues: eigenvalues of the sec(theta) coupling; the rate
        is scaled by their maximum (at least 1). ``None``: paraxial.
      upwind_rows: per band, the number of leading rows that are its
        monotone upwind closure and do not enter the rate. ``None`` counts
        every row.

    Empty or diagonal bands impose no restriction; nonfinite coefficients
    raise ``ValueError``.
    """
    rate = 0.0
    if upwind_rows is None:
        upwind_rows = (0 for _ in bands)
    for band, n_up in zip(bands, upwind_rows, strict=True):
        band = np.asarray(band, dtype=np.float64)
        if not np.isfinite(band).all():
            raise ValueError("Nonfinite continuous-loss stencil coefficients")
        off = band.copy()
        np.fill_diagonal(off, 0.0)
        row_sums = np.sum(np.abs(off), axis=1)
        rate = max(rate, float(np.max(row_sums[int(n_up) :], initial=0.0)))
    if secant_eigenvalues is not None:
        lam = np.asarray(secant_eigenvalues)
        if not np.isfinite(lam).all() or np.any(lam <= 0):
            raise ValueError("Secant eigenvalues must be finite and positive")
        rate *= max(1.0, float(np.max(lam, initial=1.0)))
    return rate
