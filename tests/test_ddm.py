import numpy as np
import pytest
from scipy.interpolate import splrep

from MCEq.particlemanager import _pdata

# DDMEntry


def test_DDMEntry_knot_sigma(ddm_entry):
    expected_result = np.array([0.1, 0.1, 0.1])

    assert np.allclose(ddm_entry.knot_sigma, expected_result)


def test_DDMEntry_fl_ebeam(ddm_entry):
    assert ddm_entry.fl_ebeam == 2.0


def test_DDMEntry_n_knots(ddm_entry):
    assert ddm_entry.n_knots == 3


def test_DDMEntry_x_min(ddm_entry):
    assert ddm_entry.x_min == _pdata.mass(211) / 2


# DataDrivenModel


def test_DataDrivenModel_channels(data_driven_model):
    assert list(data_driven_model.channels) == [(2212, 211)]
    entries = data_driven_model.find_channel(2212, 211)
    assert [e.ebeam for e in entries] == ["31.0", "158.0"]


def test_DataDrivenModel_get_entry(data_driven_model):
    assert data_driven_model.get_entry(2212, 211, ebeam=158).ebeam == "158.0"
    assert data_driven_model.get_entry(2212, 211, idx=0).ebeam == "31.0"
    with pytest.raises(ValueError):
        data_driven_model.get_entry(2212, 211, ebeam=123.0)


def test_DataDrivenModel_apply_tuning(data_driven_model):
    data_driven_model.apply_tuning(2212, 211, ebeam=158.0, tv=0.5, te=0.8)
    entry = data_driven_model.get_entry(2212, 211, ebeam=158.0)
    assert entry.tv == 0.5
    assert entry.te == 0.8


def test_DataDrivenModel_dn_dxl(data_driven_model):
    x = np.linspace(0, 1, 100)
    dn_dxl, error = data_driven_model.dn_dxl(x, 2212, 211, 158.0, return_error=True)
    assert len(dn_dxl) == len(x)
    assert len(error) == len(x)
    assert dn_dxl[0] == 0
    assert error[0] == 0


def test_DataDrivenModel_repr(data_driven_model):
    expected_repr = (
        "DDM channels:\n\t2212 -> 211:\n\t\t0: ebeam = 31.0 GeV, x17=False, "
        + "tune v|e=1.000|1.000\n\t\t1: ebeam = 158.0 GeV, x17=False,"
        + " tune v|e=1.000|1.000\n\n"
    )
    assert repr(data_driven_model) == expected_repr


# ddm_utils


def test_ddm_utils_fmteb():
    from MCEq import ddm_utils

    ebeam_float = 2.5
    ebeam_str = "2.5"
    ebeam_int = 3
    formatted_float = ddm_utils.fmteb(ebeam_float)
    formatted_str = ddm_utils.fmteb(ebeam_str)
    formatted_int = ddm_utils.fmteb(ebeam_int)

    assert formatted_float == "2.5"
    assert formatted_str == "2.5"
    assert formatted_int == "3.0"


def test_ddm_utils_spline_min_max_at_knot():
    from MCEq import ddm_utils

    tck = (
        np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        3,
    )
    sigma = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    iknot = 2
    tck_min, tck_max, h = ddm_utils._spline_min_max_at_knot(tck, iknot, sigma)
    expected_tck_min = (np.array([0, 1, 2, 3, 4]), np.array([1, 2, 2.7, 4, 5]), 3)
    expected_tck_max = (np.array([0, 1, 2, 3, 4]), np.array([1, 2, 3.3, 4, 5]), 3)
    expected_h = 0.3

    assert tck_min[0] == pytest.approx(expected_tck_min[0])
    assert tck_min[1] == pytest.approx(expected_tck_min[1])
    assert tck_min[2] == expected_tck_min[2]
    assert tck_max[0] == pytest.approx(expected_tck_max[0])
    assert tck_max[1] == pytest.approx(expected_tck_max[1])
    assert tck_max[2] == expected_tck_max[2]
    assert h == expected_h


def test_generate_DDM_matrix(mceq_qgs, data_driven_model):
    from MCEq import ddm_utils

    entries = data_driven_model.find_channel(2212, 211)

    generated_matrix = ddm_utils._generate_DDM_matrix(
        entries, mceq_qgs, e_min=20, e_max=50
    )

    # Values on the stencil-averaged v2 CI data. No projectile energy of the
    # 31-bin window is in 20..50 GeV, so this is the table's own corner; the
    # stencil leaves small negative cells at x -> 1 on the diagonal.
    expected_matrix = np.array(
        [
            [-4.357559950827538e-06, 3.557753560490889e-04],
            [3.0378289191345977e-06, -9.953006029864926e-06],
        ]
    )

    assert generated_matrix == pytest.approx(expected_matrix)


def test_ddm_utils_eval_spline():
    from MCEq import ddm_utils

    x = np.linspace(0.1, 0.99, 10)
    x_fit = np.linspace(0.1, 0.99, 20)
    y_fit = np.exp(-x_fit)
    tck = splrep(x_fit, y_fit, k=3)
    cov = np.eye(len(tck[1]))

    x17 = True

    # Shape + non-negativity alone are satisfied by an _eval_spline that
    # returns all zeros, so each block below also asserts a relation that a
    # degenerate return cannot meet: the y_fit = exp(-x) fit, divided by
    # x**1.7 under x17, must come back strictly decreasing and non-zero.

    # Test without error
    res = ddm_utils._eval_spline(x, tck, x17, cov)
    assert res.shape == (10,)
    assert np.all(res > 0)
    assert np.all(np.diff(res) < 0)

    # Test with error
    res, err = ddm_utils._eval_spline(x, tck, x17, cov, return_error=True)
    assert res.shape == (10,)
    assert err.shape == (10,)
    assert np.all(res > 0)
    assert np.all(np.diff(res) < 0)
    assert np.all(err > 0)

    # x17 is not a no-op: the same spline evaluated without it differs.
    assert not np.allclose(res, ddm_utils._eval_spline(x, tck, False, cov))

    # Test with gamma_zfac
    gamma_zfac = 0.5
    res_zfac = ddm_utils._eval_spline(x, tck, x17, cov, gamma_zfac=gamma_zfac)
    assert res_zfac.shape == (10,)
    assert np.all(res_zfac > 0)
    assert not np.allclose(res_zfac, res)


def test_gen_averaged_dndx(data_driven_model):
    from MCEq import ddm_utils

    # Log binning down to 1e-3 puts entry.x_min (0.0045 for 211 at 31 GeV)
    # inside the grid, so the x < x_min mask below is not empty.
    xbins = np.logspace(-3, 0, 11)
    entry = data_driven_model.get_entry(2212, 211, 31)
    averaged_dndx = ddm_utils._gen_averaged_dndx(xbins, entry)

    # Check the shape of the output
    assert averaged_dndx.shape == xbins[:-1].shape

    # Check that averaged dndx values are non-negative
    assert np.all(averaged_dndx >= 0)

    # _gen_averaged_dndx zeroes a bin only when its *right* edge is below
    # x_min; a straddling bin is integrated from x_min upwards instead.
    below = xbins[1:] < entry.x_min
    assert below.sum() == 2, "the mask must be non-empty or the next line is vacuous"
    assert np.all(averaged_dndx[below] == 0)
    assert np.all(averaged_dndx[~below] > 0)


def test_calc_zfactor_and_error(data_driven_model):
    projectile = 2212
    secondary = 211
    ebeam = 158.0

    entry = data_driven_model.get_entry(projectile, secondary, ebeam=ebeam)

    z_factor, z_error = entry.calc_zfactor_and_error()

    # Check that Z-factor values are non-negative
    assert np.all(z_factor >= 0)

    # Check zfactor result
    assert np.allclose(0.0485328868, z_factor)

    # Check zerror result
    assert np.allclose(0.0081361078, z_error)


def test_calc_zfactor_and_error2(data_driven_model):
    projectile = 2212
    secondary = 211
    ebeam = 158.0

    entry = data_driven_model.get_entry(projectile, secondary, ebeam=ebeam)

    z_factor, z_error = entry.calc_zfactor_and_error2()

    # Check that Z-factor values are non-negative
    assert np.all(z_factor >= 0)

    # Check zfactor result
    assert np.allclose(0.0590098529, z_factor)

    # Check zerror result
    assert np.allclose(0.010476966, z_error)


def test_default_spline_file_follows_data_dir(tmp_path, monkeypatch):
    from MCEq import config
    from MCEq.ddm import DataDrivenModel

    monkeypatch.setattr(config, "data_dir", tmp_path)
    with pytest.raises(FileNotFoundError, match=str(tmp_path)):
        DataDrivenModel()


def test_ddm_matrices_replace_only_columns_inside_the_energy_range(mceq_sib21):
    from MCEq.ddm import DataDrivenModel

    kw = dict(enable_channels=[(2212, 211)], enable_K0_from_isospin=False)
    full = DataDrivenModel(**kw).ddm_matrices(mceq_sib21)[(2212, 211)]
    part = DataDrivenModel(e_min=1e2, e_max=1e4, **kw).ddm_matrices(mceq_sib21)[
        (2212, 211)
    ]
    p = mceq_sib21.pman[2212]
    orig = np.asarray(p.hadr_yields[mceq_sib21.pman[211]])
    e = mceq_sib21.e_grid + p.mass
    inside = (e >= 1e2) & (e <= 1e4)
    assert inside.any() and not inside.all()
    assert np.array_equal(part[:, inside], full[:, inside])
    assert np.array_equal(part[:, ~inside], orig[:, ~inside])


def test_ddm_matrices_include_the_isospin_partners(mceq_sib21):
    from MCEq.ddm import DataDrivenModel

    mats = DataDrivenModel(
        enable_channels=[(2212, 211)], enable_K0_from_isospin=False
    ).ddm_matrices(mceq_sib21)
    assert set(mats) == {(2212, 211), (2112, -211)}
    assert mats[(2112, -211)] is mats[(2212, 211)]


def test_gen_matrix_variations_shift_one_knot_by_one_sigma(mceq_sib21):
    from MCEq.ddm import DataDrivenModel
    from MCEq.ddm_utils import gen_matrix_variations

    model = DataDrivenModel(
        enable_channels=[(2212, 321), (2212, -321)], enable_K0_from_isospin=True
    )
    variations, iso = gen_matrix_variations(model, mceq_sib21)
    assert set(variations) == {(2212, 321), (2212, -321)}
    central = model.ddm_matrices(mceq_sib21)
    entry = model.get_entry(2212, 321, idx=0)
    iknot, (mat_max, mat_min, h) = next(iter(variations[(2212, 321)][0].items()))
    assert h == entry.knot_sigma[iknot]
    # the splines are left as they were
    assert np.array_equal(
        model.ddm_matrices(mceq_sib21)[(2212, 321)], central[(2212, 321)]
    )
    assert not np.allclose(mat_max, mat_min)
    assert iso[(2212, 321)][0] == (310, 130)
    imax, imin, ih = iso[(2212, 321)][1][0][iknot]
    assert np.array_equal(imax, 0.5 * (mat_max + central[(2212, -321)]))
    assert ih == h
