"""Loader test for the 2D regression fixture.

The fixture at ``tests/data/2d_baseline_solution.npz`` is a snapshot of the
production 2D configuration — FLUKA 2D database (48 Hankel modes), secant
transport, muon multiple scattering, helicity-dependent decays — produced
by ``tests/data/make_2d_baseline_fixture.py``. This test only verifies that
the fixture file is well-formed and loadable so that
``test_2d_baseline_validation.py`` can rely on its presence and structure.
"""

import pathlib

import numpy as np

FIXTURE = pathlib.Path(__file__).parent / "data" / "2d_baseline_solution.npz"


def test_baseline_fixture_loads():
    d = np.load(FIXTURE, allow_pickle=True)
    assert "phi_hankel" in d.files
    assert "f_theta" in d.files
    assert "save_depths" in d.files
    assert "k_grid" in d.files
    assert "e_grid" in d.files
    assert "theta_grid" in d.files
    assert d["k_grid"].shape == (48,)
    assert d["phi_hankel"].size > 0
    assert d["f_theta"].size > 0


def test_baseline_fixture_shapes_consistent():
    """Cross-check that the saved arrays' shapes line up with each other."""
    d = np.load(FIXTURE, allow_pickle=True)
    n_depths = d["save_depths"].shape[0]
    n_k = d["k_grid"].shape[0]
    n_e = d["e_grid"].shape[0]
    n_theta = d["theta_grid"].shape[0]

    # phi_hankel: (n_depths, n_k, dim_states); dim_states is a multiple of n_e
    assert d["phi_hankel"].shape[0] == n_depths
    assert d["phi_hankel"].shape[1] == n_k
    assert d["phi_hankel"].shape[2] % n_e == 0

    # f_theta: (n_depths, n_e, n_theta) for the canonical (numu, hel=0) entry
    assert d["f_theta"].shape == (n_depths, n_e, n_theta)
    for key in (
        "f_theta_14_0",
        "f_theta_12_0",
        "f_theta_13_m1",
        "f_theta_13_0",
        "f_theta_13_p1",
        "f_theta_pdg14_summed",
        "f_theta_pdg12_summed",
        "f_theta_pdg13_summed",
    ):
        assert d[key].shape == (n_depths, n_e, n_theta), key


def test_baseline_fixture_metadata():
    """The provenance baked into the fixture pins the production 2D setup."""
    d = np.load(FIXTURE, allow_pickle=True)
    assert str(d["db_fname"]) == "mceq_db_v2_fluka2d_rc7.h5"
    assert str(d["interaction_model"]) == "FLUKA20251"
    assert str(d["kernel_config"]) == "numpy_etd2"
    assert float(d["theta_deg"]) == 30.0
    assert int(d["primary_pdg"]) == 2212
    assert len(str(d["mceq_commit"])) > 0
