"""Loader test for the PR #48 2D baseline regression fixture.

The fixture was captured by running ``examples/Angular_shower_development.ipynb``
on the unmodified PR #48 branch (commit ba916fa) using the URQMD 2D database.
Future tasks (1.6, 2.3) will compare the v2-based 2D solver against this
fixture. This test only verifies that the fixture file is well-formed and
loadable so that downstream tests can rely on its presence and structure.
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
    assert d["k_grid"].shape == (24,)
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
    """Document the provenance baked into the fixture."""
    d = np.load(FIXTURE, allow_pickle=True)
    assert str(d["pr_commit"]) == "ba916fa"
    assert str(d["db_fname"]) == "mceq_db_URQMD_150GeV_2D.h5"
    assert str(d["interaction_model"]) == "EPOSLHC"
    assert float(d["theta_deg"]) == 30.0
    assert int(d["primary_pdg"]) == 2212
    assert float(d["primary_energy_gev"]) == 100.0
