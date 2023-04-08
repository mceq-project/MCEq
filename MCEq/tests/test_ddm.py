import pytest
from MCEq import ddm
import numpy as np


@pytest.fixture(scope="module")
def ddm_obs():
    return ddm.DataDrivenModel()


# Test cases
def test_knot_sigma():
    cov = np.array([[1, 0], [0, 4]])
    ddm_entry = ddm._DDMEntry(
        ebeam=1,
        projectile=2,
        secondary=3,
        x17=False,
        tck=(None, None, None),
        cov=cov,
        tv=1,
        te=2,
        spl_idx=0,
    )
    assert np.allclose(ddm_entry.knot_sigma, np.array([2, 4]))


def test_n_knots():
    tck = (None, [1, 2, 3], None)
    ddm_entry = ddm._DDMEntry(
        ebeam=1,
        projectile=2,
        secondary=3,
        x17=False,
        tck=tck,
        cov=None,  # type: ignore
        tv=1,
        te=2,
        spl_idx=0,
    )
    assert ddm_entry.n_knots == 3


# Test cases
def test_total_n_knots():
    channel = ddm._DDMChannel(projectile=1, secondary=2)
    entry1 = ddm._DDMEntry(
        ebeam=1,
        projectile=1,
        secondary=2,
        x17=False,
        tck=(None, [1, 2], None),
        cov=None,  # type: ignore
        tv=1,
        te=2,
        spl_idx=0,
    )
    entry2 = ddm._DDMEntry(
        ebeam=2,
        projectile=1,
        secondary=2,
        x17=False,
        tck=(None, [1, 2, 3], None),
        cov=None,  # type: ignore
        tv=1,
        te=2,
        spl_idx=1,
    )
    channel._entries = [entry1, entry2]
    assert channel.total_n_knots == 5


def test_n_splines():
    channel = ddm._DDMChannel(projectile=1, secondary=2)
    entry1 = ddm._DDMEntry(
        ebeam=1,
        projectile=1,
        secondary=2,
        x17=False,
        tck=(None, [1, 2], None),
        cov=None,  # type: ignore
        tv=1,
        te=2,
        spl_idx=0,
    )
    entry2 = ddm._DDMEntry(
        ebeam=2,
        projectile=1,
        secondary=2,
        x17=False,
        tck=(None, [1, 2, 3], None),
        cov=None,  # type: ignore
        tv=1,
        te=2,
        spl_idx=1,
    )
    channel._entries = [entry1, entry2]
    assert channel.n_splines == 2


def test_spline_indices():
    channel = ddm._DDMChannel(projectile=1, secondary=2)
    entry1 = ddm._DDMEntry(
        ebeam=1,
        projectile=1,
        secondary=2,
        x17=False,
        tck=(None, [1, 2], None),
        cov=None,  # type: ignore
        tv=1,
        te=2,
        spl_idx=0,
    )
    entry2 = ddm._DDMEntry(
        ebeam=2,
        projectile=1,
        secondary=2,
        x17=False,
        tck=(None, [1, 2, 3], None),
        cov=None,  # type: ignore
        tv=1,
        te=2,
        spl_idx=1,
    )
    channel._entries = [entry1, entry2]
    assert channel.spline_indices == [0, 1]


def test_get_entry():
    channel = ddm._DDMChannel(projectile=1, secondary=2)
    entry1 = ddm._DDMEntry(
        ebeam=1,
        projectile=1,
        secondary=2,
        x17=False,
        tck=(None, [1, 2], None),
        cov=None,  # type: ignore
        tv=1,
        te=2,
        spl_idx=0,
    )
    entry2 = ddm._DDMEntry(
        ebeam=2,
        projectile=1,
        secondary=2,
        x17=False,
        tck=(None, [1, 2, 3], None),
        cov=None,  # type: ignore
        tv=1,
        te=2,
        spl_idx=1,
    )
    channel._entries = [entry1, entry2]
    assert channel.get_entry(ebeam=1) == entry1
    assert channel.get_entry(idx=1) == entry2

    try:
        channel.get_entry(ebeam=3)
    except ValueError as e:
        assert str(e) == "No entry for ebeam = 3 GeV."

    try:
        channel.get_entry(idx=2)
    except ValueError as e:
        assert str(e) == "No entry for spl_idx = 2."
