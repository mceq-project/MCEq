"""Automatic path choices and explicit precision guidance."""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from MCEq.driver.batch import _warn_low_energy_cuda_fp32
from MCEq.driver.paths import calculate_integration_path
from MCEq.operators.matrix_builder import MatrixBuilder


def path_run(is_2d=False, electrons=False):
    return SimpleNamespace(
        config=SimpleNamespace(
            solver=SimpleNamespace(
                X_start=0.0,
                etd2_path=dict(eps=None, dX_max=None, dX_min=0.01, fd_span=0.01),
            ),
            em=SimpleNamespace(adaptive_step=False),
        ),
        _mceq_db=SimpleNamespace(is_2d=is_2d),
        pman=SimpleNamespace(
            all_particles=[SimpleNamespace(pdg_id=(11 if electrons else 13, 0))]
        ),
        matrix_builder=SimpleNamespace(_contloss_bands={}, _contloss_upwind_rows={}),
        int_m=object(),
        _resolve_secant=lambda: None,
        _em_cascade_step_scale=lambda: 0.0,
        integration_path=None,
        density_model=SimpleNamespace(r_X2rho=lambda x: np.ones_like(x), max_X=10.0),
    )


@pytest.mark.parametrize(
    "is_2d,electrons,expected",
    [
        (False, False, (0.03, 5.0)),
        (True, False, (0.01, 2.0)),
        (False, True, (0.01, 2.0)),
    ],
)
def test_automatic_settings_follow_retained_physics(is_2d, electrons, expected):
    run = path_run(is_2d, electrons)
    calculate_integration_path(run, None, "X")
    assert run._cached_etd2_path_params[1:3] == expected


def test_call_and_config_overrides_survive_automatic_selection():
    run = path_run(True)
    run.config.solver.etd2_path["eps"] = 0.08
    calculate_integration_path(run, None, "X", dX_max=4.0)
    assert run._cached_etd2_path_params[1:3] == (0.08, 4.0)
    calculate_integration_path(run, None, "X", eps=0.02)
    assert run._cached_etd2_path_params[1:3] == (0.02, 2.0)


@pytest.mark.parametrize(
    "kernel,twod,secant,floor,dtype,expected",
    [
        ("cuda_etd2", True, True, 0.1, np.float32, True),
        ("CUDA", True, True, 0.05, np.float32, True),
        ("cuda_etd2", True, True, 1.0, np.float32, False),
        ("cuda_etd2", True, True, 0.1, np.float64, False),
        ("cuda_etd2", False, False, 0.1, np.float32, False),
        ("cuda_etd2", True, False, 0.1, np.float32, False),
        ("numpy_etd2", True, True, 0.1, np.float32, False),
    ],
)
def test_precision_warning_is_informative(kernel, twod, secant, floor, dtype, expected):
    run = SimpleNamespace(
        config=SimpleNamespace(backend=SimpleNamespace(kernel_config=kernel)),
        _mceq_db=SimpleNamespace(is_2d=twod),
        e_bins=np.array([floor, 2 * floor]),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_low_energy_cuda_fp32(run, np.dtype(dtype), {} if secant else None)
    assert len(caught) == int(expected)
    if expected:
        assert caught[0].category is RuntimeWarning
        assert "same integration path" in str(caught[0].message)
    assert run.e_bins[0] == floor


@pytest.mark.parametrize("twod", [False, True])
def test_sparse_assembly_matches_dense_blocks(twod):
    """Rectangular species blocks, empty rows and signed entries retain values."""
    builder = MatrixBuilder.__new__(MatrixBuilder)
    prefs = {
        0: SimpleNamespace(lidx=0, uidx=2, name="a"),
        1: SimpleNamespace(lidx=2, uidx=5, name="b"),
    }
    builder._pman = SimpleNamespace(dim=2, dim_states=5, mceqidx2pref=prefs)
    builder._grid = SimpleNamespace(dtype=np.float64)
    builder.is_2d = twod
    builder.n_k = 2 if twod else 1
    blocks = {
        (0, 1): np.array([[0.0, 2.0, 0.0], [-1.0, 0.0, 3.0]]),
        (1, 1): np.diag([0.0, -2.0, 1.0]),
    }
    expected = np.zeros((5, 5))
    expected[:2, 2:] = blocks[0, 1]
    expected[2:, 2:] = blocks[1, 1]
    if twod:
        blocks = {k: np.stack([v, 2 * v]) for k, v in blocks.items()}
        from scipy.sparse import block_diag

        expected = block_diag([expected, 2 * expected]).toarray()
    result = builder._csr_from_blocks(blocks)
    np.testing.assert_array_equal(result.toarray(), expected)
    assert result.has_canonical_format
    assert not np.any(result.data == 0)
