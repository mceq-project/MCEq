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


# DDMChannel
def test_DDMChannel_entries(ddm_channel):
    assert len(ddm_channel.entries) == 1


def test_DDMChannel_total_n_knots(ddm_channel):
    assert ddm_channel.total_n_knots == 3


def test_DDMChannel_n_splines(ddm_channel):
    assert ddm_channel.n_splines == 1


def test_DDMChannel_spline_indices(ddm_channel):
    assert ddm_channel.spline_indices == [0]


def test_DDMChannel_get_entry_by_ebeam(ddm_channel):
    entry = ddm_channel.get_entry(ebeam=2.0)
    assert entry.projectile == 2212
    assert entry.secondary == 211


def test_DDMChannel_get_entry_by_idx(ddm_channel):
    entry = ddm_channel.get_entry(idx=0)
    assert entry.projectile == 2212
    assert entry.secondary == 211


def test_DDMChannel_get_entry_raises_error(ddm_channel):
    with pytest.raises(ValueError):
        ddm_channel.get_entry(ebeam=3.0)


def test_DDMChannel_str_representation(ddm_channel):
    expected_output = (
        "\t2212 -> 211:\n\t\t0: ebeam = 2.0 GeV, x17=False, tune v|e=1.000|1.000\n"
    )
    assert str(ddm_channel) == expected_output


# DDMSplineDB


def test_DDMSplineDB_clone_entry(ddm_spline_db):
    ddm_spline_db.clone_entry(2212, 211, 158.0, 1000)
    entry = ddm_spline_db.get_entry(2212, 211, ebeam=1000)
    assert entry.projectile == 2212
    assert entry.secondary == 211


def test_DDMSplineDB_get_spline_indices(ddm_spline_db):
    indices = ddm_spline_db.get_spline_indices(2212, 211)
    assert indices == [0, 1]


def test_DDMSplineDB_channels(ddm_spline_db):
    channels = list(ddm_spline_db.channels)
    assert len(channels) == 1
    channel = channels[0]
    assert channel.projectile == 2212
    assert channel.secondary == 211


def test_DDMSplineDB_get_entry_by_ebeam(ddm_spline_db):
    from MCEq import ddm_utils

    entry = ddm_spline_db.get_entry(2212, 211, ebeam=158)
    assert entry.projectile == 2212
    assert entry.secondary == 211
    assert entry.ebeam == ddm_utils.fmteb(158.0)


def test_DDMSplineDB_get_entry_by_idx(ddm_spline_db):
    from MCEq import ddm_utils

    entry = ddm_spline_db.get_entry(2212, 211, idx=0)
    assert entry.projectile == 2212
    assert entry.secondary == 211
    assert entry.ebeam == ddm_utils.fmteb(31.0)


def test_DDMSplineDB_get_entry_raises_error(ddm_spline_db):
    with pytest.raises(ValueError):
        ddm_spline_db.get_entry(2212, 211, ebeam=123.0)


def test_DDMSplineDB_mk_channel(ddm_spline_db):
    channel = ddm_spline_db._mk_channel(2212, 211)
    assert channel == "2212-211"


# DataDrivenModel


def test_DataDrivenModel_apply_tuning(data_driven_model):
    data_driven_model.apply_tuning(2212, 211, ebeam=158.0, tv=0.5, te=0.8)
    entry = data_driven_model.spline_db.get_entry(2212, 211, ebeam=158.0)
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

    formatted_float == "2.5"
    formatted_str == "2.5"
    formatted_int == "3.0"


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

    channel = data_driven_model.find_channel(2212, 211)

    generated_matrix = ddm_utils._generate_DDM_matrix(
        channel, mceq_qgs, e_min=20, e_max=50, average=True
    )

    expected_matrix = np.array(
        [
            [
                8.00000000e-06,
                4.70861343e-04,
            ],
            [
                0.00000000e00,
                4.00016770e-06,
            ],
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

    # Test without error
    res = ddm_utils._eval_spline(x, tck, x17, cov)
    assert res.shape == (10,)
    assert np.all(res >= 0)

    # Test with error
    res, err = ddm_utils._eval_spline(x, tck, x17, cov, return_error=True)
    assert res.shape == (10,)
    assert err.shape == (10,)
    assert np.all(res >= 0)
    assert np.all(err >= 0)

    # Test with gamma_zfac
    gamma_zfac = 0.5
    res = ddm_utils._eval_spline(x, tck, x17, cov, gamma_zfac=gamma_zfac)
    assert res.shape == (10,)
    assert np.all(res >= 0)


def test_gen_dndx(data_driven_model):
    from MCEq import ddm_utils

    xbins = np.linspace(0.1, 1, 11)
    entry = data_driven_model.spline_db.get_entry(2212, 211, 31)
    dndx = ddm_utils._gen_dndx(xbins, entry)

    # Check the shape of the output
    assert len(dndx) == len(xbins) - 1

    # Check that dndx values are non-negative
    assert np.all(dndx >= 0)

    # Check that dndx values are zero where x < entry.x_min
    assert np.all(dndx[xbins[:-1] < entry.x_min] == 0)


def test_gen_averaged_dndx(data_driven_model):
    from MCEq import ddm_utils

    xbins = np.linspace(0.1, 1, 11)
    entry = data_driven_model.spline_db.get_entry(2212, 211, 31)
    averaged_dndx = ddm_utils._gen_averaged_dndx(xbins, entry)

    # Check the shape of the output
    assert averaged_dndx.shape == xbins[:-1].shape

    # Check that averaged dndx values are non-negative
    assert np.all(averaged_dndx >= 0)

    # Check that averaged dndx values are zero where x < entry.x_min
    assert np.all(averaged_dndx[xbins[:-1] < entry.x_min] == 0)


def test_calc_zfactor_and_error(data_driven_model):
    projectile = 2212
    secondary = 211
    ebeam = 158.0

    entry = data_driven_model.spline_db.get_entry(projectile, secondary, ebeam=ebeam)

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

    entry = data_driven_model.spline_db.get_entry(projectile, secondary, ebeam=ebeam)

    z_factor, z_error = entry.calc_zfactor_and_error2()

    # Check that Z-factor values are non-negative
    assert np.all(z_factor >= 0)

    # Check zfactor result
    assert np.allclose(0.0590098529, z_factor)

    # Check zerror result
    assert np.allclose(0.010476966, z_error)
