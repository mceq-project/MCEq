import numpy as np
import numpy.testing as npt
import pytest

from MCEq import ddm, ddm_utils
from MCEq.particlemanager import _pdata


@pytest.fixture(scope="module")
def ddm_fix():
    return ddm.DataDrivenModel(
        e_min=5.0,
        e_max=500.0,
        enable_channels=[(2212, 211)],
        exclude_projectiles=[111, 2112],
        enable_K0_from_isospin=True,
    )


@pytest.fixture(scope="module")
def ddm_entry():
    return ddm._DDMEntry(
        ebeam=ddm_utils.fmteb(2.0),
        projectile=2212,
        secondary=211,
        x17=False,
        tck=(np.array([1, 2, 3]), np.array([4, 5, 6]), 3),
        cov=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        tv=1.0,
        te=0.1,
        spl_idx=1,
    )


@pytest.fixture(scope="module")
def ddm_channel() -> ddm._DDMChannel:
    channel = ddm._DDMChannel(projectile=2212, secondary=211)
    channel.add_entry(
        ebeam=ddm_utils.fmteb(2.0),
        projectile=2212,
        secondary=211,
        x17=False,
        tck=(np.array([1, 2, 3]), np.array([4, 5, 6]), 3),
        cov=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        tv=1.0,
        te=1.0,
    )
    return channel


@pytest.fixture(scope="module")
def ddm_spline_db() -> ddm.DDMSplineDB:
    return ddm.DDMSplineDB(
        enable_channels=[(2212, 211)],
        exclude_projectiles=[111, 2112],
    )


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ("knot_sigma", np.array([0.1, 0.1, 0.1])),
        ("fl_ebeam", 2.0),
        ("n_knots", 3),
        ("x_min", _pdata.mass(211) / 2.0),
    ],
)
def test_ddm_entry_properties(ddm_entry, test_input, expected) -> None:
    if test_input == "knot_sigma":
        npt.assert_array_equal(getattr(ddm_entry, test_input), expected)
    else:
        assert getattr(ddm_entry, test_input) == pytest.approx(expected)


def test_ddm_channel_entries(ddm_channel: ddm._DDMChannel) -> None:
    assert len(ddm_channel.entries) == 1


def test_ddm_channel_total_n_knots(ddm_channel: ddm._DDMChannel) -> None:
    assert ddm_channel.total_n_knots == 3


def test_ddm_channel_n_splines(ddm_channel: ddm._DDMChannel) -> None:
    assert ddm_channel.n_splines == 1


def test_ddm_channel_spline_indices(ddm_channel: ddm._DDMChannel) -> None:
    assert ddm_channel.spline_indices == [0]


def test_ddm_channel_get_entry_by_ebeam(ddm_channel: ddm._DDMChannel) -> None:
    entry = ddm_channel.get_entry(ebeam=2.0)
    assert entry.projectile == 2212
    assert entry.secondary == 211


def test_ddm_channel_get_entry_by_idx(ddm_channel: ddm._DDMChannel) -> None:
    entry = ddm_channel.get_entry(idx=0)
    assert entry.projectile == 2212
    assert entry.secondary == 211


def test_ddm_channel_get_entry_raises_error(ddm_channel: ddm._DDMChannel) -> None:
    with pytest.raises(ValueError, match="No entry for ebeam = 3.0 GeV"):
        ddm_channel.get_entry(ebeam=3.0)


def test_ddm_channel_str_representation(ddm_channel: ddm._DDMChannel) -> None:
    expected_output = (
        "\t2212 -> 211:\n\t\t0: ebeam = 2.0 GeV, x17=False, tune v|e=1.000|1.000\n"
    )
    assert str(ddm_channel) == expected_output


def test_ddm_spline_db_clone_entry(ddm_spline_db: ddm.DDMSplineDB) -> None:
    ddm_spline_db.clone_entry(2212, 211, 158.0, 1000)
    entry = ddm_spline_db.get_entry(2212, 211, ebeam=1000)
    assert entry.projectile == 2212
    assert entry.secondary == 211


def test_ddm_spline_db_get_spline_indices(ddm_spline_db: ddm.DDMSplineDB) -> None:
    indices = ddm_spline_db.get_spline_indices(2212, 211)
    assert indices == [0, 1]


def test_ddm_spline_db_channels(ddm_spline_db: ddm.DDMSplineDB) -> None:
    channels = list(ddm_spline_db.channels)
    assert len(channels) == 1
    channel = channels[0]
    assert channel.projectile == 2212
    assert channel.secondary == 211


def test_ddm_spline_db_get_entry_by_ebeam(ddm_spline_db: ddm.DDMSplineDB) -> None:
    entry = ddm_spline_db.get_entry(2212, 211, ebeam=158)
    assert entry.projectile == 2212
    assert entry.secondary == 211
    assert entry.ebeam == ddm_utils.fmteb(158.0)


def test_ddm_spline_db_get_entry_by_idx(ddm_spline_db: ddm.DDMSplineDB) -> None:
    entry = ddm_spline_db.get_entry(2212, 211, idx=0)
    assert entry.projectile == 2212
    assert entry.secondary == 211
    assert entry.ebeam == ddm_utils.fmteb(31.0)


def test_ddm_spline_db_get_entry_raises_error(ddm_spline_db: ddm.DDMSplineDB) -> None:
    with pytest.raises(ValueError, match="No entry for ebeam = 123.0 GeV"):
        ddm_spline_db.get_entry(2212, 211, ebeam=123.0)


def test_ddm_spline_db_mk_channel(ddm_spline_db: ddm.DDMSplineDB) -> None:
    channel = ddm_spline_db._mk_channel(2212, 211)
    assert channel == "2212-211"


def test_eval_spline() -> None:
    x = np.linspace(0.1, 0.99, 10)
    tck = (
        np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.0]),
        3,
    )
    x17 = True
    cov = np.eye(6)

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


def test_gen_dndx(ddm_fix: ddm.DataDrivenModel) -> None:
    xbins = np.linspace(0.1, 1, 11)
    entry = ddm_fix.spline_db.get_entry(2212, 211, 31)
    dndx = ddm_utils._gen_dndx(xbins, entry)

    # Check the shape of the output
    assert len(dndx) == len(xbins) - 1

    # Check that dndx values are non-negative
    assert np.all(dndx >= 0)

    # Check that dndx values are zero where x < entry.x_min
    assert np.all(dndx[xbins[:-1] < entry.x_min] == 0)


def test_gen_averaged_dndx(ddm_fix: ddm.DataDrivenModel) -> None:
    xbins = np.linspace(0.1, 1, 11)
    entry = ddm_fix.spline_db.get_entry(2212, 211, 31)
    averaged_dndx = ddm_utils._gen_averaged_dndx(xbins, entry)

    # Check the shape of the output
    assert averaged_dndx.shape == xbins[:-1].shape

    # Check that averaged dndx values are non-negative
    assert np.all(averaged_dndx >= 0)

    # Check that averaged dndx values are zero where x < entry.x_min
    assert np.all(averaged_dndx[xbins[:-1] < entry.x_min] == 0)


def test_calc_zfactor_and_error(ddm_fix: ddm.DataDrivenModel) -> None:
    projectile = 2212
    secondary = 211
    ebeam = 158.0

    entry = ddm_fix.spline_db.get_entry(projectile, secondary, ebeam=ebeam)

    z_factor, z_error = entry.calc_zfactor_and_error()

    # Check that Z-factor values are non-negative
    assert np.all(z_factor >= 0)

    # Check zfactor result
    assert np.allclose(0.0485328868, z_factor)

    # Check zerror result
    assert np.allclose(0.0081361078, z_error)


def test_calc_zfactor_and_error2(ddm_fix: ddm.DataDrivenModel) -> None:
    projectile = 2212
    secondary = 211
    ebeam = 158.0

    entry = ddm_fix.spline_db.get_entry(projectile, secondary, ebeam=ebeam)

    z_factor, z_error = entry.calc_zfactor_and_error2()

    # Check that Z-factor values are non-negative
    assert np.all(z_factor >= 0)

    # Check zfactor result
    assert np.allclose(0.0590098529, z_factor)

    # Check zerror result
    assert np.allclose(0.010476966, z_error)
