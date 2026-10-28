r"""sec(theta) path elongation for 1D solves, from tables of 2D solutions.

The l = 0 row of the Legendre transport (:mod:`MCEq.operators.legendre`) is
the 1D equation with every column of species ``j`` -- interaction, decay and
continuous loss -- multiplied by

    s_j(E, X) = (G a_j)_0 / a_{j,0} = < g(mu) >_j,   g = 1 / max(mu, mu_cap),

the mean path elongation of the current angular distribution of ``j``
(:func:`MCEq.operators.secant.legendre_coupling`, row 0). Charging the parent
column keeps the daughter production of a parent in equilibrium unchanged, so
neutrinos are unaffected. The 1D image of the 2D solve is

    dN/dX = (A_int + rho_inv A_dec) diag(s) N,

which :func:`MCEq.solvers.etd2.etd2_driver` integrates with ``s`` exact on the
diagonal and the remainder applied to ``s N`` (``col_scale``). It is exact up
to the backward removal of produced particles (relevant below ~0.3 GeV) and
the variation of ``s`` within a depth segment.

``s`` depends on depth, zenith and the kind of cascade: muons and sub-GeV
nucleons differ between the inclusive flux of a primary spectrum and the
shower of a single primary. A table holds both, ``s(species, E, X)`` at a few
zenith angles, from 2D solves (:func:`generate_secant_table`), interpolated
linearly in log X and in cos(theta); the run's initial condition selects the
group (:func:`run_mode`). The angles come from the low-energy model's tables,
so a table belongs to one low-energy model and one 2D database. It transfers
across atmospheres, observation levels, primary spectra and handover energies
from 50 GeV up at the 1-2 % level for sub-GeV muons, better above. Not
covered: EM species (``s = 1``), media other than air, and the backward
removal below ~0.3 GeV.
"""

import warnings

import h5py
import numpy as np

from MCEq import config
from MCEq.misc import info
from MCEq.operators.secant import legendre_coupling

__all__ = [
    "MODES",
    "SecantTable",
    "column_scale",
    "generate_secant_table",
    "run_mode",
    "table_for",
    "warn_per_direction",
]

#: Table groups: the inclusive flux of a primary spectrum, the shower of a
#: single primary.
MODES = ("inclusive", "shower")

#: ``config.secant_1d_table = "auto"`` applies a table only for ``e_min`` below
#: this energy (GeV); above it ``s - 1`` stays below ~1 %.
E_MIN_AUTO = 5.0

#: Handover energy (GeV) below which the high-energy model's angles, absent
#: from the tables, start to matter (tested down to 50 GeV).
HANDOVER_MIN = 50.0

#: Depth segments per decade of slant depth; ``s`` is constant within one.
SEGMENTS_PER_DECADE = 20


def run_mode(mceq):
    """``"shower"`` if the initial condition holds a single primary, else
    ``"inclusive"`` (primary flux model or user spectrum)."""
    kinds = {entry[0] for entry in getattr(mceq, "_initial_condition", ())}
    return "shower" if "single" in kinds else "inclusive"


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


def generate_secant_table(
    mceq_2d, fname, zeniths=(0, 60, 75, 85, 90), n_X=48, source_sha256=None
):
    """Solve ``mceq_2d`` (a 2D :class:`MCEq.core.MCEqRun` with its initial
    condition set) at each zenith on ``n_X`` log-spaced depths and write the
    group of its :func:`run_mode` to ``fname``, replacing that group if
    present. ``source_sha256`` (the 2D database's digest) is stored as
    ``split_from_sha256`` for the data manifest's freshness check."""
    db = mceq_2d._mceq_db
    if not db.is_2d:
        raise ValueError("generate_secant_table needs a 2D database")
    mode = run_mode(mceq_2d)
    names = [p.name for p in mceq_2d.pman.all_particles if p.mceqidx >= 0]
    idx = [mceq_2d.pman.pname2pref[n].mceqidx for n in names]
    ne = len(mceq_2d.e_grid)
    provenance = {
        "low_energy_model": str(db.low_energy_model),
        "he_le_transition": float(db.he_le_transition),
        "he_le_trwidth": float(db.he_le_trwidth),
        "theta_cap_deg": config.secant_theta_cap(),
        "secant_e_max": float(config.secant_theta_e_max or np.inf),
        "database": str(config.mceq_db_fname),
    }
    if source_sha256 is not None:
        provenance["split_from_sha256"] = source_sha256
    with h5py.File(fname, "a") as f:
        if "e_grid" in f:
            same = np.allclose(f["e_grid"][:], mceq_2d.e_grid) and all(
                f.attrs.get(k) == v
                for k, v in provenance.items()
                if k in ("low_energy_model", "database")
            )
            if not same:
                raise ValueError(f"{fname} holds a table of another database")
        else:
            f["e_grid"] = mceq_2d.e_grid
            f["species"] = np.array(names, dtype="S")
        f.attrs.update(provenance)
        if mode in f:
            del f[mode]
        g = f.create_group(mode)
        g.attrs["initial_condition"] = repr(mceq_2d._initial_condition)[:200]
        g["zeniths"] = np.asarray(zeniths, float)
        for i, zen in enumerate(zeniths):
            mceq_2d.set_zenith_azimuth(zen)
            X = np.geomspace(1.0, mceq_2d.density_model.max_X * (1 - 1e-9), n_X)
            mceq_2d.solve(int_grid=X)
            s = elongation_factors(mceq_2d).reshape(n_X, -1, ne)[:, idx]
            g[f"X_{i}"] = X
            g[f"s_{i}"] = s.astype(np.float32)
            info(1, f"secant table [{mode}]: zenith {zen:g} deg done")


class SecantTable:
    """A table of :func:`generate_secant_table`: provenance in :attr:`attrs`,
    the groups of :data:`MODES` it holds in :attr:`modes`."""

    def __init__(self, path):
        with h5py.File(path, "r") as f:
            self.attrs = dict(f.attrs)
            self.e_grid = f["e_grid"][:]
            self.species = [n.decode() for n in f["species"][:]]
            self.modes = {}
            for mode in MODES:
                if mode in f:
                    g = f[mode]
                    n = len(g["zeniths"])
                    self.modes[mode] = (
                        g["zeniths"][:],
                        [g[f"X_{i}"][:] for i in range(n)],
                        [g[f"s_{i}"][:].astype(float) for i in range(n)],
                    )
        self._index = None

    def factors(self, mode, pman, dim, e_grid, theta_deg, X):
        """``s`` of group ``mode`` at zenith ``theta_deg`` and slant depths
        ``X`` on the 1D state vector of ``pman`` (length ``dim``), shape
        ``(len(X), dim)``."""
        if not np.allclose(e_grid, self.e_grid):
            raise ValueError("secant table energy grid differs from the database")
        zeniths, Xs, ss = self.modes[mode]
        c, cz = np.cos(np.radians(theta_deg)), np.cos(np.radians(zeniths))
        hi = int(np.clip(np.searchsorted(-cz, -c), 1, len(cz) - 1))
        lo = hi - 1
        w = float(np.clip((c - cz[hi]) / (cz[lo] - cz[hi]), 0.0, 1.0))

        def at(i):
            ix = np.interp(np.log(X), np.log(Xs[i]), np.arange(len(Xs[i])))
            k = np.minimum(ix.astype(int), len(Xs[i]) - 2)
            f = (ix - k)[:, None, None]
            return (1 - f) * ss[i][k] + f * ss[i][k + 1]

        s_sp = w * at(lo) + (1 - w) * at(hi)  # (len(X), species, E)
        if self._index is None or self._index[0] is not pman:
            ne = len(e_grid)
            rows = [
                (j, pman.pname2pref[n].mceqidx)
                for j, n in enumerate(self.species)
                if n in pman.pname2pref and pman.pname2pref[n].mceqidx >= 0
            ]
            src = np.array([j for j, _ in rows])
            dst = np.array([m * ne + np.arange(ne) for _, m in rows])
            self._index = (pman, src, dst)
        _, src, dst = self._index
        s = np.ones((len(X), dim))
        s[:, dst] = s_sp[:, src]
        return s


def _warn_once(mceq, key, message):
    seen = mceq.__dict__.setdefault("_secant_table_warned", set())
    if key not in seen:
        seen.add(key)
        warnings.warn(message, stacklevel=4)


def _manifest_table(mceq, le_model):
    """Name of the manifest's ``kind: secant_1d`` package for ``le_model``, or None."""
    from MCEq.data.download import ensure_manifest

    files = ensure_manifest(config)["files"]
    names = [
        name
        for name, entry in files.items()
        if entry.get("kind") == "secant_1d"
        and le_model in entry.get("models", {}).get("air", [])
    ]
    return names[0] if names else None


def table_for(mceq):
    """``(table, mode)`` that applies to the 1D run ``mceq``, or None.

    ``config.secant_1d_table`` is ``"auto"`` (the data manifest's table of the
    run's low-energy model, applied with the sec(theta) transport switched on
    and ``config.e_min < E_MIN_AUTO``), a file in ``data_dir`` or a path
    (always applied), or None. A table of another low-energy model is
    refused; a handover below :data:`HANDOVER_MIN` warns.
    """
    choice = config.secant_1d_table
    db = mceq._mceq_db
    if choice is None or db.is_2d:
        return None
    flag = config.secant_theta_transport
    if choice == "auto":
        if not (flag is True or str(flag).lower() == "auto"):
            return None
        if config.e_min >= E_MIN_AUTO:
            return None
        le_model = db.low_energy_model
        # the manifest lookup parses YAML; resolve once per manifest and model
        key = (str(config.data_dir), config.mceq_db_manifest, le_model)
        resolved = mceq.__dict__.get("_secant_table_auto")
        if resolved is None or resolved[0] != key:
            name = _manifest_table(mceq, le_model) if le_model else None
            if name is not None:
                from MCEq.data.download import ensure_package

                name = ensure_package(config, name)
            resolved = mceq.__dict__["_secant_table_auto"] = (key, name)
        path = resolved[1]
        if path is None:
            _warn_once(
                mceq,
                ("missing", le_model),
                f"No 1D sec(theta) table for the low-energy model {le_model}: "
                "sub-GeV hadrons and muons are solved without the path elongation.",
            )
            return None
    else:
        path = config.data_dir / choice
        path = path if path.exists() else choice

    cached = mceq.__dict__.get("_secant_table")
    if cached is None or cached[0] != str(path):
        table = SecantTable(path)
        if str(table.attrs.get("low_energy_model")) != str(db.low_energy_model):
            raise ValueError(
                f"1D sec(theta) table {path} is built with the low-energy model "
                f"{table.attrs.get('low_energy_model')}, the run uses "
                f"{db.low_energy_model}: regenerate it "
                "(MCEq.operators.secant_table.generate_secant_table)."
            )
        cached = mceq.__dict__["_secant_table"] = (str(path), table)
    table = cached[1]
    if db.he_le_transition < HANDOVER_MIN:
        _warn_once(
            mceq,
            ("handover", db.he_le_transition),
            f"he_le_transition = {db.he_le_transition:g} GeV: below "
            f"{HANDOVER_MIN:g} GeV the high-energy model's production angles, "
            "absent from the 1D sec(theta) table, are untested.",
        )
    mode = run_mode(mceq)
    if mode not in table.modes:
        raise ValueError(f"1D sec(theta) table {path} has no '{mode}' group")
    return table, mode


def warn_per_direction(mceq):
    """Per-direction batch and full-sky solves run without the table (one
    ``s`` per lane is not implemented); warn once if one would apply."""
    if table_for(mceq) is not None:
        _warn_once(
            mceq,
            "per-direction",
            "Per-direction batch and full-sky solves ignore the 1D sec(theta) "
            "table; use MCEqRun.solve or a shared-path batch for sub-GeV fluxes.",
        )


def column_scale(mceq, dX, X_start):
    """``(S, starts)`` of :func:`MCEq.solvers.etd2.etd2_driver` for the path
    ``dX`` from ``X_start``: ``s`` at the geometric midpoint of each
    log-spaced depth segment (:data:`SEGMENTS_PER_DECADE`); None without a
    table."""
    found = table_for(mceq)
    if found is None:
        return None
    table, mode = found
    X_end = X_start + np.cumsum(dX)
    n_seg = int(np.ceil(SEGMENTS_PER_DECADE * np.log10(X_end[-1] / X_end[0])))
    bounds = np.geomspace(X_end[0], X_end[-1], max(n_seg, 1) + 1)[1:-1]
    starts = np.unique([0, *np.searchsorted(X_end, bounds, side="right")])
    starts = starts[starts < len(dX)]
    ends = np.append(starts[1:], len(dX))
    X_mid = np.sqrt(X_end[np.maximum(starts - 1, 0)] * X_end[ends - 1])
    key = (id(table), mode, mceq.theta_deg, X_mid.tobytes())
    cached = mceq.__dict__.get("_secant_column_scale")
    if cached is None or cached[0] != key:
        S = table.factors(
            mode, mceq.pman, mceq.dim_states, mceq.e_grid, mceq.theta_deg, X_mid
        )
        cached = mceq.__dict__["_secant_column_scale"] = (key, S)
    S = cached[1]
    info(
        2,
        f"1D sec(theta) table [{mode}], {len(starts)} depth segments, "
        f"low-energy model {table.attrs.get('low_energy_model')}",
    )
    return S, starts
