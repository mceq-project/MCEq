"""Tabulated cutoff maps of the named sites and the average over azimuth."""

import numpy as np
import pytest

import MCEq.geometry.density_profiles as dp
from MCEq.batch import build_phi0_with_cutoff
from MCEq.geometry.cutoff import _tabulated_maps, get_cutoff_map


def test_every_named_site_has_a_downgoing_map():
    from MCEq.geometry.atmosphere_parameters import KM3NET_DETECTORS, LOCATIONS

    tab = _tabulated_maps()
    assert tab["zenith_deg"] == list(np.arange(0.0, 90.1, 5.0))
    assert tab["azimuth_deg"] == list(np.arange(0.0, 360.0, 5.0))
    assert set(tab["cutoff_GV"]) == set(LOCATIONS) | set(KM3NET_DETECTORS)
    for rc in tab["cutoff_GV"].values():
        assert np.shape(rc) == (19, 72)


def test_site_atmospheres_read_the_table_at_the_nodes():
    tab = _tabulated_maps()
    atm = dp.CorsikaAtmosphere("ANTARES/KM3NeT-ORCA", "Summer")
    rc = get_cutoff_map(atm, np.array([0.0, 60.0]), np.array([90.0, 270.0]))
    ref = np.array(tab["cutoff_GV"]["ORCA"])[[0, 12]][:, [18, 54]]
    assert np.allclose(rc, ref)
    # positive primaries from the East are cut harder than from the West
    assert rc[1, 0] > rc[1, 1]


def test_site_atmospheres_refuse_upgoing_directions():
    atm = dp.CorsikaAtmosphere("SouthPole", "June")
    with pytest.raises(ValueError, match="downgoing"):
        get_cutoff_map(atm, np.array([100.0]), np.array([0.0]))


def test_azimuth_samples_average_the_transmitted_flux(mceq_sib21):
    rc = np.array([[0.0, 1e9]])  # half of the sky open, half closed
    half = build_phi0_with_cutoff(mceq_sib21, mceq_sib21.pmodel, rc)
    full = build_phi0_with_cutoff(mceq_sib21, mceq_sib21.pmodel, np.zeros(1))
    assert np.allclose(half, 0.5 * full, rtol=1e-12, atol=0)
