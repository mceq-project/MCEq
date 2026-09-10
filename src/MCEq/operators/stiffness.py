"""Continuous-loss timescale for the explicit part of diagonal ETD2.

The loss bands are independent of Hankel mode. In the secant eigenbasis
those bands are multiplied by each eigenvalue of S. The largest absolute
row sum of their off-diagonal part bounds their explicit coupling rate;
short decay lifetimes (exponentiated by the kernel) do not enter this rate.
This is a conservative step policy, not an error estimator or a proof of
stability of the complete, generally non-normal cascade operator.
"""

import numpy as np

LOSS_STEP_SAFETY = 0.5


def continuous_loss_rate(bands, secant_eigenvalues=None):
    """Return an inverse-depth rate from the actual assembled loss stencils.

    Read the stored bands rather than reconstructing from dE/dX: boundary
    closures, grid spacing, operator averaging and the chosen stencil all
    affect the explicitly integrated coefficients. Empty/diagonal bands
    impose no explicit restriction. Nonfinite coefficients fail closed.
    """
    rate = 0.0
    for band in bands:
        band = np.asarray(band, dtype=np.float64)
        if not np.isfinite(band).all():
            raise ValueError("Nonfinite continuous-loss stencil coefficients")
        off = band.copy()
        np.fill_diagonal(off, 0.0)
        rate = max(rate, float(np.max(np.sum(np.abs(off), axis=1), initial=0.0)))
    if secant_eigenvalues is not None:
        lam = np.asarray(secant_eigenvalues)
        if not np.isfinite(lam).all() or np.any(lam <= 0):
            raise ValueError("Secant eigenvalues must be finite and positive")
        rate *= max(1.0, float(np.max(lam, initial=1.0)))
    return rate
