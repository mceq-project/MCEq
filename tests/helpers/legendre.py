"""Legendre moments of sampled distributions, for the reference values of the tests."""

import numpy as np


def moments(mu, ls, weights=None, chunk=1 << 16):
    """``sum_i w_i P_l(mu_i)`` for every ``l`` in the integer array ``ls``.

    The recurrence runs to ``max(l)`` over chunks of ``mu``, so sparse high
    ``l`` sets cost ``O(max(l) n_mu)`` time and ``O(chunk)`` memory per row.
    """
    mu = np.ravel(np.asarray(mu, dtype=float))
    w = np.ones_like(mu) if weights is None else np.ravel(weights).astype(float)
    ls = np.asarray(ls, dtype=int)
    out = np.zeros(ls.shape)
    rows = {int(li): np.flatnonzero(ls == li) for li in np.unique(ls)}
    for lo in range(0, mu.size, chunk):
        x, wx = mu[lo : lo + chunk], w[lo : lo + chunk]
        p_prev, p = np.ones_like(x), x.copy()
        for li in range(int(ls.max()) + 1):
            if li in rows:
                out[rows[li]] += wx @ (p_prev if li == 0 else p)
            if li >= 1:
                p_prev, p = p, ((2 * li + 1) * x * p - li * p_prev) / (li + 1)
    return out
