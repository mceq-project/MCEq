"""Bin-averaged rates for stencil-averaged databases (``MCEq.bin_average``).

A database whose tables act on the parent count density reconstructed by the
quartic neighbour stencil marks the parents it averaged; their decay and
interaction rates become banded operators. Databases without the marks are
unchanged.
"""

import shutil

import h5py
import numpy as np
import pytest

from MCEq import bin_average, config
from MCEq.data.hdf5_backend import HDF5Backend  # noqa: F401  (import check)
from MCEq.misc import energy_grid
from MCEq.particlemanager import MCEqParticle

H = 0.1
E_BINS = np.logspace(-1, 6, 71)


def _cell_average(f, e_bins, n=2000):
    """(1/w_j) int_bin f dE, midpoint in log E."""
    lb = np.log10(e_bins)
    u = (np.arange(n) + 0.5) / n
    e = 10 ** (lb[:-1, None] + np.diff(lb)[:, None] * u[None])
    de = e * np.log(10) * np.diff(lb)[:, None] / n
    return (f(e) * de).sum(1) / np.diff(e_bins)


def test_constant_rate_is_the_identity():
    op = bin_average.rate_operator(E_BINS, lambda e: np.full_like(e, 2.5), 2)
    assert np.allclose(op, 2.5 * np.eye(70), rtol=0, atol=1e-13)


@pytest.mark.parametrize("gamma", [1.0, 2.7, 3.7, 4.7])
def test_rate_stencil_integrates_rate_times_flux(gamma):
    """Pion decay rate (1/p) on a power law: banded R closes the bin integral of
    r * phi to < 4e-4 (quartic truncation at E^-4.7); the centre product is off
    by up to a few per cent."""
    m, ctau = 0.13957, 780.45

    def rate(e):
        return m / ctau / np.sqrt((e + m) ** 2 - m**2)

    phi = _cell_average(lambda e: e**-gamma, E_BINS)
    truth = _cell_average(lambda e: rate(e) * e**-gamma, E_BINS)
    op = bin_average.rate_operator(E_BINS, rate, 2)
    c = np.sqrt(E_BINS[1:] * E_BINS[:-1])
    inner = slice(2, -2)
    assert np.allclose((op @ phi)[inner], truth[inner], rtol=4e-4)
    if gamma > 2:
        assert np.max(np.abs(rate(c) * phi / truth - 1)[inner]) > 5e-3


def test_apply_right_is_the_matrix_product():
    rng = np.random.default_rng(3)
    op = bin_average.rate_operator(E_BINS, lambda e: 1 / e, 2)
    for block in (rng.random((70, 70)), rng.random((4, 70, 70))):
        assert np.allclose(bin_average.apply_right(block, op, 2), block @ op)


def test_rate_operator_refuses_a_non_uniform_grid():
    with pytest.raises(ValueError, match="uniform"):
        bin_average.rate_operator(np.r_[E_BINS[:-1], 2e6], lambda e: e, 2)


def _particle(pdg):
    c = np.sqrt(E_BINS[1:] * E_BINS[:-1])
    grid = energy_grid(c=c, b=E_BINS, w=np.diff(E_BINS), d=len(c))
    return MCEqParticle(pdg, 0, energy_grid=grid)


def test_particle_operators_need_the_marks():
    p = _particle(211)
    assert p.decay_rate_operator() is None
    assert p.interaction_rate_operator() is None
    p._dec_stencil = 2
    op = p.decay_rate_operator()
    lam = p.inverse_decay_length()
    assert np.allclose(np.diag(op)[2:-2], lam[2:-2], rtol=0.02)
    assert np.allclose(np.diag(op)[[0, 1, -2, -1]], lam[[0, 1, -2, -1]], rtol=0)
    stable = _particle(2212)
    stable._dec_stencil = 2
    assert stable.decay_rate_operator() is None


def _strip_marks(f):
    for obj in [f["decays"]] + [
        f["hadronic_interactions/air"][m]
        for m in f["hadronic_interactions/air"]
        if not m.endswith("_indptrs")
    ]:
        for key in [k for k in obj.attrs if k.startswith("bin_average")]:
            del obj.attrs[key]


def _unmarked_db(tmp_path, src):
    """Copy of a CI database without its stencil marks: the same tables, read
    with diagonal rates and the three-bin single-primary deposition."""
    dst = tmp_path / "unmarked.h5"
    shutil.copy(src, dst)
    with h5py.File(dst, "r+") as f:
        _strip_marks(f)
    return dst


def _marked_db(tmp_path, src):
    """Copy of a CI database with stencil marks on pions/kaons (decays) and on
    every projectile of every air model (yields) only, in place of the ones it
    ships with; the tables stay as they are."""
    dst = tmp_path / "marked.h5"
    shutil.copy(src, dst)
    with h5py.File(dst, "r+") as f:
        _strip_marks(f)
        f["decays"].attrs["bin_average_radius"] = 2
        f["decays"].attrs["bin_average_parents"] = np.array([211, -211, 321, -321])
        air = f["hadronic_interactions/air"]
        for m in (x for x in air if not x.endswith("_indptrs")):
            air[m].attrs["bin_average_radius"] = 2
            air[m].attrs["bin_average_projectiles"] = np.array([2212, 2112, 211, -211])
    return dst


@pytest.fixture
def ci_db_path():
    from importlib.resources import files

    path = files("MCEq") / "data" / "mceq_ci_base_air_1d_v2.h5"
    if not path.is_file():
        pytest.skip("CI database not present")
    return str(path)


def _run(db, model="SIBYLL23E"):
    import crflux.models as pm

    from MCEq.core import MCEqRun

    config.mceq_db_fname = db
    return MCEqRun(
        interaction_model=model,
        theta_deg=0.0,
        primary_model=(pm.HillasGaisser2012, "H3a"),
    )


def test_marked_database_bands_only_marked_parents(tmp_path, ci_db_path):
    plain = _run(str(_unmarked_db(tmp_path, ci_db_path)))
    marked = _run(str(_marked_db(tmp_path, ci_db_path)))
    pman = marked.pman
    pi = pman[(211, 0)]
    assert pi._dec_stencil == 2 and pi._int_stencil == 2
    assert pman[(13, 1)]._dec_stencil == 0 if (13, 1) in pman.pdg2pref else True
    d = marked.dim
    for name in ("dec_m", "int_m"):
        a = getattr(plain.matrix_builder, name).toarray()
        b = getattr(marked.matrix_builder, name).toarray()
        changed = np.flatnonzero(np.any(a != b, axis=0)) // d
        touched = {pman.cascade_particles[i].pdg_id[0] for i in set(changed)}
        allowed = {211, -211, 321, -321} if name == "dec_m" else {2212, 2112, 211, -211}
        assert touched and touched <= allowed, (name, touched)
        # the loss diagonal of the pion block is close to the plain rate
        i0 = pi.mceqidx * d
        blk_a = a[i0 : i0 + d, i0 : i0 + d]
        blk_b = b[i0 : i0 + d, i0 : i0 + d]
        assert np.allclose(np.diag(blk_b)[2:-2], np.diag(blk_a)[2:-2], rtol=0.05)


def test_ci_database_carries_the_stencil_marks(ci_db_path):
    """The CI packages are cut from the stencil-averaged release databases."""
    with h5py.File(ci_db_path, "r") as f:
        assert f["decays"].attrs["bin_average_radius"] == 2
        assert 211 in set(f["decays"].attrs["bin_average_parents"])
        for m in ("FLUKA20251", "SIBYLL23E"):
            ds = f["hadronic_interactions/air"][m]
            assert ds.attrs["bin_average_radius"] == 2
            assert 2212 in set(ds.attrs["bin_average_projectiles"])


def test_switch_off_restores_the_plain_operator(tmp_path, ci_db_path, monkeypatch):
    plain = _run(str(_unmarked_db(tmp_path, ci_db_path)))
    monkeypatch.setattr(config, "bin_average_rates", False)
    marked = _run(str(_marked_db(tmp_path, ci_db_path)))
    for name in ("dec_m", "int_m"):
        a = getattr(plain.matrix_builder, name)
        b = getattr(marked.matrix_builder, name)
        assert (a != b).nnz == 0


def _toy_tables():
    """Yield-like toy kernel dN/dx ~ (1 - x)^3 / x: bin-centre table, stencil table."""
    lx = np.linspace(-4, 0, 20001)[:-1] + 1e-4
    fx = (1 - 10**lx) ** 3 * np.log(10) * 2e-4
    lb = np.log10(E_BINS)
    n = len(E_BINS) - 1
    c = np.sqrt(E_BINS[1:] * E_BINS[:-1])

    def hist(e):
        i = np.searchsorted(lb, np.log10(e) + lx, side="right") - 1
        ok = (i >= 0) & (i < n)
        return np.bincount(i[ok], weights=fx[ok], minlength=n)

    nu = 100
    u = -H / 2 + (np.arange(nu) + 0.5) * H / nu
    mom = np.zeros((5, n, n))
    for j in range(n):
        for uu in u:
            h = hist(c[j] * 10**uu)
            for k in range(5):
                mom[k, :, j] += uu**k * h / nu
    w = bin_average.stencil_weights(2) * (H ** -np.arange(5))[:, None]
    table = np.zeros((n, n))
    for j in range(2, n - 2):
        for mi, d in enumerate(range(-2, 3)):
            table[:, j + d] += np.einsum("k,ki->i", w[:, mi], mom[:, :, j])
    return hist, table


@pytest.mark.parametrize("e0", [1000.0, 1071.5, 912.0])
def test_delta_deposition_closes_the_first_generation(e0):
    """Stencil tables + delta deposition: the top daughter band (x 0.5-1) within
    0.8 % of the kernel at e0 (three-bin centre deposition: ~4 % low); lower
    daughters to 0.2 % (away from the toy table's grid edge)."""
    hist, table = _toy_tables()
    first, s = bin_average.delta_deposition(E_BINS, e0, 2)
    assert s.sum() == pytest.approx(1.0, rel=1e-12)
    dep = np.zeros(len(E_BINS) - 1)
    dep[first : first + 5] = s
    truth = hist(e0)
    top = (E_BINS[:-1] >= 0.5 * e0) & (E_BINS[1:] <= e0 * 1.0001)
    assert (table @ dep)[top].sum() / truth[top].sum() == pytest.approx(1.0, abs=8e-3)
    low = slice(5, np.argmax(top) - 3)
    assert np.allclose((table @ dep)[low], truth[low], rtol=2e-3)


def test_delta_deposition_needs_neighbours_on_the_grid():
    with pytest.raises(ValueError, match="edge"):
        bin_average.delta_deposition(E_BINS, 0.11, 2)


def test_single_primary_uses_the_stencil_deposition_on_marked_databases(
    tmp_path, ci_db_path
):
    plain = _run(str(_unmarked_db(tmp_path, ci_db_path)))
    marked = _run(str(_marked_db(tmp_path, ci_db_path)))
    e0 = 2.0e4
    for m in (plain, marked):
        m.set_single_primary_particle(e0, pdg_id=2212)
    p_plain, p_marked = (
        m._phi0[m.pman[2212].lidx : m.pman[2212].lidx + m.dim] for m in (plain, marked)
    )
    w = marked.e_widths
    assert np.count_nonzero(p_plain) == 3
    assert np.count_nonzero(p_marked) == 5
    assert (p_marked * w).sum() == pytest.approx(1.0, rel=1e-10)
