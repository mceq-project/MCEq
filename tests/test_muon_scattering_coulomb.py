"""Screened Coulomb muon scattering: rate values, composition reading and
operator placement."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from scipy.special import k1

from MCEq import config
from MCEq.data.hdf5_backend import HDF5Backend
from MCEq.data.hdf5_store import HDF5Store
from MCEq.operators import scattering
from MCEq.operators.matrix_builder import MatrixBuilder

#: Air as in the mceq-EM generator that produced the reference values below.
AIR_INTEGER_A = [[7, 14.0, 0.7847874810295787], [8, 16.0, 0.210518910117893],
                 [18, 40.0, 0.004693608852528217]]  # fmt: skip
#: CODATA 2018 muon mass [GeV], used by that generator.
MUON_MASS = 0.1056583755

#: (kinetic energy [GeV], kappa [1/rad], planar rate [cm2/g]) from the generator.
REFERENCE = [
    (0.011220184543019636, 1, 0.006631083974020151),
    (0.011220184543019636, 43, 7.2701139234472745),
    (0.011220184543019636, 2000, 4817.608988250182),
    (11.220184543019636, 1, 3.750463643107822e-08),
    (11.220184543019636, 43, 5.158614620957443e-05),
    (11.220184543019636, 2000, 0.07237545250018884),
    (11220.18454301964, 1, 5.615368663764808e-14),
    (11220.18454301964, 43, 8.573567172665234e-11),
    (11220.18454301964, 2000, 1.4551770193293214e-07),
]


def _one_minus_z_k1(z):
    """``1 - z K_1(z)``, with its series below ``z = 1e-3``; zero at ``z = 0``."""
    out = np.zeros_like(z)
    small = (z > 0) & (z < 1e-3)
    x = z[small]
    log = np.log(x / 2) + np.euler_gamma
    out[small] = -x * x / 2 * (log - 0.5) - x**4 / 16 * (log - 1.25)
    large = z >= 1e-3
    out[large] = 1 - z[large] * k1(z[large])
    return out


def _planar_rate(e_kin, kappa, composition, muon_mass=scattering.MUON_MASS):
    """Small-angle limit of the screened Coulomb rate on the flat sky,
    ``sum_i R_i [1 - kappa a_i K_1(kappa a_i)]`` with ``a_i = 2 sqrt(A_i)``,
    from the element parameters of ``scattering._elements``."""
    e_kin, kappa = np.asarray(e_kin, dtype=float), np.asarray(kappa, dtype=float)
    rate = np.zeros((kappa.size, e_kin.size))
    for n, screening, c, charge, _ in scattering._elements(
        e_kin, composition, muon_mass
    ):
        strength = c * charge * (charge + 1)
        z = kappa[:, None] * (2 * np.sqrt(screening))
        rate += np.pi * n * strength / screening * _one_minus_z_k1(z)
    return rate


@pytest.mark.parametrize(("e_kin", "kappa", "expected"), REFERENCE)
def test_element_parameters_match_generator(e_kin, kappa, expected):
    """Screening and strength per element against the mceq-EM generator,
    through the planar limit it tabulated."""
    rate = _planar_rate([e_kin], [kappa], AIR_INTEGER_A, MUON_MASS)
    np.testing.assert_allclose(rate[0, 0], expected, rtol=1e-9)


def test_rate_zero_mode_and_monotone():
    e_kin = np.geomspace(1e-2, 1e6, 30)
    ls = np.array([0, 1, 2, 3, 5, 8, 13, 21, 34, 64, 102, 486, 2000])
    rate = scattering.screened_coulomb_rate_legendre(e_kin, ls, AIR_INTEGER_A)
    assert rate.shape == (ls.size, e_kin.size)
    assert np.all(rate[0] == 0)
    assert np.all(np.diff(rate, axis=0) > 0)
    assert np.all(np.diff(rate[1:], axis=1) < 0)


def _backend(path, medium="air"):
    backend = HDF5Backend.__new__(HDF5Backend)
    backend.had_fname = str(path)
    backend._had = HDF5Store(path)
    backend.medium = medium
    return backend


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "composition.h5"
    with h5py.File(path, "w") as f:
        f["material_scattering/air/composition"] = AIR_INTEGER_A
    return path


def test_composition_read_and_missing_medium(db):
    np.testing.assert_array_equal(_backend(db).scattering_composition(), AIR_INTEGER_A)
    with pytest.raises(ValueError, match="absent"):
        _backend(db, "water").scattering_composition()


@pytest.mark.parametrize("corrupt", ["shape", "fraction", "z", "nan"])
def test_reject_invalid_composition(tmp_path, corrupt):
    comp = np.array(AIR_INTEGER_A)
    if corrupt == "shape":
        comp = comp[:, :2]
    elif corrupt == "fraction":
        comp[0, 2] = 0.5
    elif corrupt == "z":
        comp[0, 0] = 7.5
    else:
        comp[1, 1] = np.nan
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as f:
        f["material_scattering/air/composition"] = comp
    with pytest.raises(ValueError, match="Invalid"):
        _backend(path).scattering_composition()


def test_all_muon_charges_helicities_and_operator_placement(monkeypatch, db):
    k_grid = np.array([0.0, 10.0])
    e_kin = np.array([1.0, 10.0])
    builder = MatrixBuilder.__new__(MatrixBuilder)
    builder.is_2d, builder.n_k, builder.k_grid = True, 2, k_grid
    builder._layout = _backend(db)
    monkeypatch.setattr(config, "muon_multiple_scattering", True)
    monkeypatch.setattr(config, "muon_scattering_model", "screened-coulomb")
    monkeypatch.setattr(config, "muon_scattering_finite_nuclei", True)
    monkeypatch.setattr(config, "floatlen", np.float64)
    builder._energy_grid = SimpleNamespace(c=e_kin, d=2)
    particles = {
        (pdg, hel): SimpleNamespace(lidx=2 * i, mceqidx=i, mass=MUON_MASS)
        for i, (pdg, hel) in enumerate(((p, h) for p in (13, -13) for h in (0, -1, 1)))
    }

    class Pman(dict):
        pass

    pman = Pman(particles)
    pman.pdg2pref, pman.dim_states, pman.dim = particles, 12, 2
    builder._pman = pman
    matrix = builder._csr_from_blocks({}, apply_muon_scattering=True)
    rate = scattering.screened_coulomb_rate_legendre(
        e_kin, k_grid, AIR_INTEGER_A, MUON_MASS
    )
    expected = np.zeros(24)
    expected[12:24] = np.tile(-rate[1], 6)
    np.testing.assert_allclose(matrix.diagonal(), expected, rtol=1e-15)
    assert matrix.nnz == 12


def test_unknown_model_raises(monkeypatch, db):
    builder = MatrixBuilder.__new__(MatrixBuilder)
    builder.is_2d, builder.k_grid = True, np.array([0.0, 10.0])
    monkeypatch.setattr(config, "muon_multiple_scattering", True)
    monkeypatch.setattr(config, "muon_scattering_model", "moliere")
    monkeypatch.setattr(config, "muon_scattering_finite_nuclei", True)
    builder._energy_grid = SimpleNamespace(c=np.array([1.0, 10.0]))

    class Pman(dict):
        pass

    muon = SimpleNamespace(lidx=0, mceqidx=0, mass=MUON_MASS)
    builder._pman = Pman({(13, 0): muon})
    builder._pman.pdg2pref = {(13, 0): muon}
    with pytest.raises(ValueError, match="Unknown muon_scattering_model"):
        builder._muon_scattering_damping()


def test_scattering_model_invalidates_split_assembly(monkeypatch):
    builder = MatrixBuilder.__new__(MatrixBuilder)
    builder.is_2d = True
    monkeypatch.setattr(config, "floatlen", None)
    monkeypatch.setattr(config, "muon_multiple_scattering", True)
    monkeypatch.setattr(config, "muon_scattering_model", "gaussian")
    monkeypatch.setattr(config, "muon_scattering_finite_nuclei", True)
    old_key = builder._current_assembly_key()
    monkeypatch.setattr(config, "muon_scattering_model", "screened-coulomb")
    assert old_key != builder._current_assembly_key()


# Legendre modes: Goudsmit-Saunderson rates of the Wentzel kernel.


def _quadrature_sigma(screening, ls, panels, order=32):
    """``2 int (1 - P_l(mu)) / (1 - mu + 2A)^2 dmu`` by composite Gauss-Legendre
    in ``u = ln(1 - mu + 2A)``, with ``1 - P_l`` from its own recurrence."""
    xg, wg = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(np.log(2 * screening), np.log(2 + 2 * screening), panels + 1)
    half = np.diff(edges)[:, None] / 2
    u = (half * xg + (edges[1:] + edges[:-1])[:, None] / 2).ravel()
    w = (half * wg).ravel()
    y = np.exp(u)
    t = y - 2 * screening
    d_prev, d, out = np.zeros_like(t), t.copy(), {}
    for n in range(1, max(ls) + 1):
        if n in ls:
            out[n] = 2 * np.sum(w * d / y)
        d_prev, d = d, ((2 * n + 1) * (t + (1 - t) * d) - n * d_prev) / (n + 1)
    return out


@pytest.mark.parametrize(
    ("screening", "l_max", "panels"),
    [
        (1e-12, 500, 1000),
        (1e-8, 500, 1000),
        (1e-4, 500, 1000),
        (1e-2, 1000, 4000),
        (0.3, 2000, 4000),
    ],
)
def test_transport_cross_sections_match_quadrature(screening, l_max, panels):
    """Both branches (``l sqrt(A)`` below and above one) against quadrature."""
    sigma = scattering.transport_cross_sections(screening, l_max)
    ls = {1, 2, 5, 64, l_max // 4, l_max}
    for ell, value in _quadrature_sigma(screening, ls, panels).items():
        np.testing.assert_allclose(sigma[ell], value, rtol=1e-9, err_msg=f"l={ell}")


def test_transport_cross_sections_l1_and_zero_mode():
    """``sigma_0 = 0`` and the textbook ``2 ln((1+A)/A) - 2/(1+A)`` at l = 1."""
    a = np.geomspace(1e-30, 1.0, 61)
    sigma = scattering.transport_cross_sections(a, 3)
    assert np.all(sigma[0] == 0)
    np.testing.assert_allclose(
        sigma[1], 2 * np.log1p(1 / a) - 2 / (1 + a), rtol=4e-16, atol=0
    )
    assert np.all(np.diff(sigma, axis=0) > 0)


def test_transport_cross_sections_branches_agree():
    """Where the polynomial form is still accurate (``l sqrt(A) <= 1``) the
    Neumann form with Miller ratios gives the same values, and a mixed batch
    of screenings equals the screenings taken one by one."""
    a = np.array([1e-4])
    r = scattering._q_ratios(a, 100)
    q_prev = np.log1p(1 / a) / 2 * np.cumprod(np.vstack([[1.0], r[:-1]]), axis=0)
    ls = np.arange(1, 101)[:, None]
    neumann = (1 - ls * q_prev * (1 - (1 + 2 * a) * r)) / (a * (1 + a))
    poly = scattering.transport_cross_sections(a, 100)[1:]
    np.testing.assert_allclose(poly[9:], neumann[9:], rtol=1e-11)
    mixed = np.array([1e-12, 1e-5, 3e-3, 0.3])
    batch = scattering.transport_cross_sections(mixed, 1000)
    for j, value in enumerate(mixed):
        single = scattering.transport_cross_sections(value, 1000)
        np.testing.assert_array_equal(batch[:, j], single)


def test_single_scattering_moments_mc():
    """``<P_l> = 1 - sigma_l / sigma_tot`` against samples of the Wentzel
    kernel (inverse CDF ``mu = 1 - 2 A u / (1 + A - u)``)."""
    from MCEq.operators.legendre import moments

    a, n = 1e-3, 400_000
    u = np.random.default_rng(7).random(n)
    mu = 1 - 2 * a * u / (1 + a - u)
    ls = np.array([1, 2, 5, 10, 30, 64])
    mc = moments(mu, ls) / n
    exact = 1 - scattering.transport_cross_sections(a, 64)[ls] * a * (1 + a)
    assert np.all(np.abs(mc - exact) < 5 / np.sqrt(n)), mc - exact


def test_legendre_rate_small_angle_limit_and_l1_pin():
    """The planar rate at ``kappa = l + 1/2`` is the small-angle limit of the
    exact rate: equal to 2e-4 at l = 64 and 1e-5 from l = 256. At l = 1 it is
    12.7 % high in air, nearly independent of energy (regression pin)."""
    e_kin = np.geomspace(0.1, 10.0, 7)
    ls = np.array([0, 1, 2, 64, 256, 1000, 2000])
    exact = scattering.screened_coulomb_rate_legendre(
        e_kin, ls, AIR_INTEGER_A, MUON_MASS, finite_nuclei=False
    )
    planar = _planar_rate(e_kin, ls[1:] + 0.5, AIR_INTEGER_A, MUON_MASS)
    assert np.all(exact[0] == 0)
    ratio = planar / exact[1:]
    np.testing.assert_allclose(ratio[0], 1.127, atol=0.002)
    np.testing.assert_allclose(ratio[1], 1.042, atol=0.002)
    assert np.all((ratio[2] >= 1) & (ratio[2] <= 1 + 2e-4))
    assert np.all(np.abs(ratio[3:] - 1) < 1e-5)


def _d_l(ell, z):
    """``1 - P_l(1 - z)`` by the forward recurrence, precise at small ``z``."""
    d_prev, d = 0.0, z
    for n in range(1, ell):
        d_prev, d = d, ((2 * n + 1) * (z + (1 - z) * d) - n * d_prev) / (n + 1)
    return d if ell else 0.0


@pytest.mark.parametrize("e_kin", [0.1, 2.0, 1e4])
def test_finite_nuclei_rate_matches_quadrature(e_kin):
    """Form factor ``(1 + b z)^-2`` on Z^2 and the electron limit z <= z_e on Z,
    against adaptive quadrature of the cross section in log(chi)."""
    from scipy import integrate

    mu, me = MUON_MASS, scattering.ELECTRON_MASS
    p2, e_tot = e_kin * (e_kin + 2 * mu), e_kin + mu
    z_e = 2 * me**2 / (mu**2 + me**2 + 2 * me * e_tot)
    ls = np.array([1, 5, 40])
    rate = scattering.screened_coulomb_rate_legendre([e_kin], ls, AIR_INTEGER_A, mu)
    for j, ell in enumerate(ls):
        total = 0.0
        for n, a, c, charge, mass in scattering._elements(
            np.array([e_kin]), AIR_INTEGER_A, mu
        ):
            a, c = a[0], c[0]
            b = scattering.NUCLEAR_FORM_FACTOR * mass**0.54 * p2

            def f(u, part):
                chi = np.exp(u)
                z = 2 * np.sin(chi / 2) ** 2
                w = charge**2 / (1 + b * z) ** 2 if part else charge * (z <= z_e)
                return (
                    2
                    * np.pi
                    * c
                    * _d_l(ell, z)
                    / (z + 2 * a) ** 2
                    * np.sin(chi)
                    * chi
                    * w
                )

            cuts = [np.log(np.sqrt(a)) - 14, np.log(np.sqrt(a)), np.log(2 * np.arcsin(np.sqrt(z_e / 2))),
                    np.log(min(1.0, np.sqrt(2 / b))), np.log(1 / (ell + 1)), np.log(np.pi)]  # fmt: skip
            cuts = sorted(
                set(cuts) | set(np.log(np.linspace(1 / (ell + 1), np.pi, 60)))
            )
            for part in (True, False):
                for lo, hi in zip(cuts[:-1], cuts[1:]):
                    total += (
                        n
                        * integrate.quad(
                            f, lo, hi, args=(part,), limit=200, epsrel=1e-11
                        )[0]
                    )
        np.testing.assert_allclose(rate[j, 0], total, rtol=1e-8)


def test_finite_nuclei_limits():
    """Below the point-nucleus rate everywhere, approaching it where the form
    factor and electron limit only touch the large-angle tail (low E, l = 1 is
    most affected); finite_nuclei=False is the point-nucleus kernel."""
    e_kin, ls = np.geomspace(0.05, 1e6, 12), np.array([0, 1, 10, 89])
    fin = scattering.screened_coulomb_rate_legendre(e_kin, ls, AIR_INTEGER_A)
    pt = scattering.screened_coulomb_rate_legendre(
        e_kin, ls, AIR_INTEGER_A, finite_nuclei=False
    )
    assert np.all(fin[0] == 0)
    ratio = fin[1:] / pt[1:]
    assert np.all((ratio > 0.3) & (ratio < 1))
    above = e_kin[1:] > 1.0  # below, the electron limit keeps l = 89 flat to 0.4 %
    assert np.all(np.diff(ratio, axis=1)[:, above] < 0)  # tail cut grows with energy
    assert np.all(np.diff(ratio, axis=0) > 0)  # high l is set by small angles


def test_legendre_rate_composition_sum_and_integer_modes():
    e_kin, ls = np.array([0.01, 1.0, 1e4]), np.array([0, 1, 7, 64, 500])
    total = scattering.screened_coulomb_rate_legendre(e_kin, ls, AIR_INTEGER_A)
    mean_a = sum(row[1] * row[2] for row in AIR_INTEGER_A)
    parts = sum(
        scattering.screened_coulomb_rate_legendre(
            e_kin, ls, [[z, mass, 1.0]]
        ) * fraction * mass / mean_a
        for z, mass, fraction in AIR_INTEGER_A
    )  # fmt: skip
    np.testing.assert_allclose(total, parts, rtol=1e-13)
    with pytest.raises(ValueError, match="integer"):
        scattering.screened_coulomb_rate_legendre(e_kin, [0, 1.5], AIR_INTEGER_A)


def test_gaussian_mode_rates():
    """The diffusion eigenvalue ``l (l + 1) theta_s^2 / 4``."""
    e_kin, ell = np.array([0.3, 3.0]), np.array([0.0, 1.0, 4.0])
    theta2 = scattering.theta_s_squared(e_kin, MUON_MASS)
    rate = scattering.mode_rates("gaussian", e_kin, ell, None, MUON_MASS)
    np.testing.assert_allclose(rate, np.outer(ell * (ell + 1), theta2) / 4, rtol=1e-15)


@pytest.mark.parametrize("model", ["screened-coulomb", "gaussian"])
def test_legendre_operator_placement(monkeypatch, db, model):
    """The muon diagonals carry the exact per-l rate of either model."""
    l_grid, e_kin = np.array([0.0, 1.0, 67.0]), np.array([0.5, 50.0])
    builder = MatrixBuilder.__new__(MatrixBuilder)
    builder.is_2d, builder.n_k, builder.k_grid = True, 3, l_grid
    builder._layout = _backend(db)
    monkeypatch.setattr(config, "muon_multiple_scattering", True)
    monkeypatch.setattr(config, "muon_scattering_model", model)
    monkeypatch.setattr(config, "muon_scattering_finite_nuclei", True)
    monkeypatch.setattr(config, "floatlen", np.float64)
    builder._energy_grid = SimpleNamespace(c=e_kin, d=2)
    muon = SimpleNamespace(lidx=0, mceqidx=0, mass=MUON_MASS)

    class Pman(dict):
        pass

    pman = Pman({(13, 0): muon})
    pman.pdg2pref, pman.dim_states, pman.dim = {(13, 0): muon}, 2, 2
    builder._pman = pman
    matrix = builder._csr_from_blocks({}, apply_muon_scattering=True)
    if model == "gaussian":
        theta2 = scattering.theta_s_squared(e_kin, MUON_MASS)
        rate = np.outer(l_grid * (l_grid + 1), theta2) / 4
    else:
        rate = scattering.screened_coulomb_rate_legendre(
            e_kin, l_grid, AIR_INTEGER_A, MUON_MASS
        )
    np.testing.assert_allclose(matrix.diagonal(), -rate.ravel(), rtol=1e-15)
