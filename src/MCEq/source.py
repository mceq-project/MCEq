"""External point sources of the cascade state, dΦ/dX = MΦ + S(X), for
:meth:`MCEq.core.MCEqRun.solve_batch` (``source=``)."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Source:
    """Point sources: ``val`` is added to state row ``row`` of column ``col`` at slant depth
    ``x`` (g/cm²). Rows are in the run's state layout (``species.lidx + bin``)."""

    col: np.ndarray
    x: np.ndarray
    row: np.ndarray
    val: np.ndarray


def bin_on_path(src, X_start, dX):
    """Coalesce a :class:`Source` onto the boundaries of a path.

    Each entry goes to the nearest step boundary ``X_start + sum(dX[:b])``; entries beyond the
    last boundary are dropped. Returns ``(starts, rows, cols, vals)`` sorted by boundary, with
    ``starts`` of length ``len(dX) + 2`` (entries of boundary ``b`` are ``starts[b]:starts[b+1]``)
    and no repeated ``(row, col)`` within a boundary.
    """
    xb = X_start + np.concatenate([[0.0], np.cumsum(dX)])
    mid = 0.5 * (xb[1:] + xb[:-1])
    b = np.searchsorted(mid, src.x)
    inside = src.x <= xb[-1]
    b, row, col, val = b[inside], src.row[inside], src.col[inside], src.val[inside]
    n_rows = int(row.max()) + 1 if len(row) else 1
    n_cols = int(col.max()) + 1 if len(col) else 1
    key = (b.astype(np.int64) * n_rows + row) * n_cols + col
    ukey, inv = np.unique(key, return_inverse=True)
    vals = np.bincount(inv, weights=val)
    bb, rest = np.divmod(ukey, n_rows * n_cols)
    rows, cols = np.divmod(rest, n_cols)
    starts = np.searchsorted(bb, np.arange(len(dX) + 2))
    return starts, rows, cols, vals
