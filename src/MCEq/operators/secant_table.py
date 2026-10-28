r"""sec(theta) path elongation for 1D solves, from a table of 2D solutions.

The l = 0 row of the Legendre transport (:mod:`MCEq.operators.legendre`) is
the 1D equation with every column of species ``j`` -- interaction, decay and
continuous loss -- multiplied by

    s_j(E, X) = (G a_j)_0 / a_{j,0} = < g(mu) >_j,   g = 1 / max(mu, mu_cap),

the mean path elongation of the current angular distribution of ``j``
(:func:`MCEq.operators.secant.legendre_coupling`, row 0). Charging the parent
column keeps the daughter production of a parent in equilibrium unchanged, so
neutrinos are unaffected. The 1D image of the 2D solve is therefore

    dN/dX = (A_int + rho_inv A_dec) diag(s) N,

exact up to the backward removal of produced particles (relevant below
~0.3 GeV) and the variation of ``s`` within a depth segment.

``s`` of hadrons is close to universal in slant depth; muon ``s`` depends on
depth and zenith (wide-angle fresh muons vs collimated survivors), so the
table holds ``s(species, E, X)`` at a few zenith angles, interpolated
linearly in log X and in cos(theta). It is generated from 2D solves of the
inclusive flux (:func:`generate_secant_table`) and transfers across
atmospheres, observation levels and primary spectra at the 1-2 % level for
sub-GeV muons, better above. Muons of single-primary showers have a different
angular history: an inclusive table leaves them 2-9 % high at 0.1-0.3 GeV.
Not covered: EM species (``s = 1``), media other than air, and the backward
removal below ~0.3 GeV.
"""

import h5py
import numpy as np

from MCEq import config
from MCEq.misc import info
from MCEq.operators.secant import legendre_coupling

__all__ = ["SecantTable", "generate_secant_table", "solve_segments", "table_for"]


def elongation_factors(mceq_2d):
    """``s`` of every state column at every recorded depth of a solved 2D run,
    shape ``(n_X, dim)``; 1 above ``config.secant_theta_e_max``, clipped to
    ``[1, sec(theta_cap)]`` (empty bins carry Gibbs noise)."""
    db = mceq_2d._mceq_db
    T0 = legendre_coupling(np.asarray(db.k_grid), config.secant_theta_cap())[0]
    a = np.asarray(mceq_2d.grid_sol).reshape(-1, db.n_k, mceq_2d.dim_states)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = 1 + np.einsum("l,xln->xn", T0, a) / a[:, 0]
    e_max = config.secant_theta_e_max or np.inf
    gate = np.tile(mceq_2d.e_grid < e_max, mceq_2d.dim_states // len(mceq_2d.e_grid))
    s[:, ~gate] = 1.0
    s_cap = 1 / np.cos(np.radians(config.secant_theta_cap()))
    return np.clip(np.nan_to_num(s, nan=1.0), 1.0, s_cap)


def generate_secant_table(mceq_2d, fname, zeniths=(0, 60, 75, 85, 90), n_X=48):
    """Solve ``mceq_2d`` (a 2D :class:`MCEq.core.MCEqRun` with its primary
    flux set) at each zenith on ``n_X`` log-spaced depths and write the
    elongation table to ``fname``."""
    if not mceq_2d._mceq_db.is_2d:
        raise ValueError("generate_secant_table needs a 2D database")
    names = [p.name for p in mceq_2d.pman.all_particles if p.mceqidx >= 0]
    idx = [mceq_2d.pman.pname2pref[n].mceqidx for n in names]
    ne = len(mceq_2d.e_grid)
    with h5py.File(fname, "w") as f:
        f["e_grid"] = mceq_2d.e_grid
        f["zeniths"] = np.asarray(zeniths, float)
        f["species"] = np.array(names, dtype="S")
        f.attrs["theta_cap_deg"] = config.secant_theta_cap()
        for i, zen in enumerate(zeniths):
            mceq_2d.set_zenith_azimuth(zen)
            max_X = mceq_2d.density_model.max_X
            X = np.geomspace(1.0, max_X * (1 - 1e-9), n_X)
            mceq_2d.solve(int_grid=X)
            s = elongation_factors(mceq_2d).reshape(n_X, -1, ne)[:, idx]
            f[f"X_{i}"] = X
            f[f"s_{i}"] = s.astype(np.float32)
            info(1, f"secant table: zenith {zen:g} deg done")


class SecantTable:
    """Elongation table of :func:`generate_secant_table`; :meth:`factors`
    returns ``s`` on the state vector of a 1D run."""

    def __init__(self, fname):
        path = config.data_dir / fname
        with h5py.File(path if path.exists() else fname, "r") as f:
            self.e_grid = f["e_grid"][:]
            self.zeniths = f["zeniths"][:]
            self.species = [n.decode() for n in f["species"][:]]
            self.X = [f[f"X_{i}"][:] for i in range(len(self.zeniths))]
            self.s = [f[f"s_{i}"][:].astype(float) for i in range(len(self.zeniths))]

    def _bracket(self, theta_deg):
        """Two zenith tables around ``theta_deg`` and the cos(theta) weight of the first."""
        c, cz = np.cos(np.radians(theta_deg)), np.cos(np.radians(self.zeniths))
        hi = int(np.clip(np.searchsorted(-cz, -c), 1, len(cz) - 1))
        lo = hi - 1
        w = float(np.clip((c - cz[hi]) / (cz[lo] - cz[hi]), 0.0, 1.0))
        return lo, hi, w

    def depth_grid(self, theta_deg):
        """Depths of the two zenith tables around ``theta_deg``, merged."""
        lo, hi, _ = self._bracket(theta_deg)
        return np.unique(np.concatenate([self.X[lo], self.X[hi]]))

    def factors(self, pman, dim, e_grid, theta_deg, X):
        """``s`` at slant depth ``X`` on the 1D state vector (length ``dim``) of ``pman``."""
        if not np.allclose(e_grid, self.e_grid):
            raise ValueError("secant table energy grid differs from the database")
        lo, hi, w = self._bracket(theta_deg)
        lx = np.log(X)

        def at(i):
            ix = np.interp(lx, np.log(self.X[i]), np.arange(len(self.X[i])))
            k = int(ix)
            f = ix - k
            k1 = min(k + 1, len(self.X[i]) - 1)
            return (1 - f) * self.s[i][k] + f * self.s[i][k1]

        s_sp = w * at(lo) + (1 - w) * at(hi)
        ne = len(e_grid)
        s = np.ones(dim)
        for j, name in enumerate(self.species):
            p = pman.pname2pref.get(name)
            if p is not None and p.mceqidx >= 0:
                s[p.mceqidx * ne : (p.mceqidx + 1) * ne] = s_sp[j]
        return s


def table_for(mceq):
    """The :class:`SecantTable` of ``config.secant_1d_table`` for a 1D run, cached
    per file name; ``None`` if unset or on a 2D database."""
    fname = config.secant_1d_table
    if fname is None or mceq._mceq_db.is_2d:
        return None
    cached = mceq.__dict__.get("_secant_table")
    if cached is None or cached[0] != fname:
        cached = mceq.__dict__["_secant_table"] = (fname, SecantTable(fname))
    return cached[1]


#: Depth segments per decade of slant depth; ``s`` is constant within a segment.
SEGMENTS_PER_DECADE = 10


def _scale_columns(M, s):
    """``M @ diag(s)`` for a CSR matrix, on its sparsity pattern."""
    out = M.tocsr(copy=True)
    out.data *= s[out.indices]
    return out


def solve_segments(mceq, nsteps, dX, rho_inv, grid_idcs, X_start, dtype):
    """Integrate the 1D path with ``int_m`` and ``dec_m`` right-multiplied by
    ``diag(s)``, ``s`` constant within log-spaced depth segments
    (:data:`SEGMENTS_PER_DECADE`) and evaluated at their geometric midpoints.
    Returns ``(final_state, grid_states)`` like the plain solve."""
    table = table_for(mceq)
    X_end = X_start + np.cumsum(dX)
    n_seg = max(1, int(np.ceil(SEGMENTS_PER_DECADE * np.log10(X_end[-1] / X_end[0]))))
    bounds = np.geomspace(X_end[0], X_end[-1], n_seg + 1)[1:-1]
    cut = np.unique(np.searchsorted(X_end, bounds, side="right"))
    edges = [0, *cut[(cut > 0) & (cut < nsteps)], nsteps]

    int_m, dec_m = mceq.int_m, mceq.dec_m
    phi, grid = mceq._stack_modes(mceq._phi0), []
    grid_idcs = np.asarray(grid_idcs, dtype=int)
    try:
        for a, b in zip(edges[:-1], edges[1:]):
            X_mid = np.sqrt(X_end[max(a - 1, 0)] * X_end[b - 1])
            s = table.factors(
                mceq.pman, mceq.dim_states, mceq.e_grid, mceq.theta_deg, X_mid
            )
            mceq.int_m, mceq.dec_m = _scale_columns(int_m, s), _scale_columns(dec_m, s)
            local = list(grid_idcs[(grid_idcs >= a) & (grid_idcs < b)] - a)
            phi, g = mceq._run_etd2(b - a, dX[a:b], rho_inv[a:b], phi, local, dtype)
            if local:
                grid.extend(np.asarray(g).reshape(len(local), -1))
    finally:
        mceq.int_m, mceq.dec_m = int_m, dec_m
    info(3, f"secant table: {len(edges) - 1} depth segments")
    return phi, np.asarray(grid)
