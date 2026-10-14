"""Automatic path choices and explicit precision guidance."""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from MCEq import config
from MCEq.batch import resolve_dtype
from MCEq.core import MCEqRun
from MCEq.operators.matrix_builder import MatrixBuilder


class _Run(SimpleNamespace):
    _calculate_integration_path = MCEqRun._calculate_integration_path
    _continuous_loss_dx_cap = MCEqRun._continuous_loss_dx_cap


def path_run(monkeypatch, is_2d=False, electrons=False):
    monkeypatch.setattr(config, "X_start", 0.0)
    monkeypatch.setattr(
        config,
        "etd2_path",
        dict(eps=None, dX_max=None, dX_min=0.01, fd_span=0.01),
    )
    monkeypatch.setattr(config, "em_adaptive_step", False)
    return _Run(
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
def test_automatic_settings_follow_retained_physics(
    monkeypatch, is_2d, electrons, expected
):
    run = path_run(monkeypatch, is_2d, electrons)
    run._calculate_integration_path(None, "X")
    assert run._cached_etd2_path_params[1:3] == expected


def test_call_and_config_overrides_survive_automatic_selection(monkeypatch):
    run = path_run(monkeypatch, True)
    config.etd2_path["eps"] = 0.08
    run._calculate_integration_path(None, "X", dX_max=4.0)
    assert run._cached_etd2_path_params[1:3] == (0.08, 4.0)
    run._calculate_integration_path(None, "X", eps=0.02)
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
def test_precision_warning_is_informative(
    monkeypatch, kernel, twod, secant, floor, dtype, expected
):
    monkeypatch.setattr(config, "kernel_config", kernel)
    run = SimpleNamespace(
        _mceq_db=SimpleNamespace(is_2d=twod),
        e_bins=np.array([floor, 2 * floor]),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        MCEqRun._warn_low_energy_cuda_fp32(run, np.dtype(dtype), {} if secant else None)
    assert len(caught) == int(expected)
    if expected:
        assert caught[0].category is RuntimeWarning
        assert "same integration path" in str(caught[0].message)
    assert run.e_bins[0] == floor


@pytest.mark.parametrize(
    "kernel,fp,dtype,expected",
    [
        ("cuda_etd2", 32, None, np.float32),
        ("CUDA", 32, None, np.float32),
        ("cuda_etd2", 64, None, np.float64),
        ("cuda_etd2", 32, np.float64, np.float64),
        ("numpy_etd2", 32, None, np.float64),
        ("mkl_etd2", 32, np.float32, np.float32),
    ],
)
def test_state_precision_follows_cuda_fp_precision(
    monkeypatch, kernel, fp, dtype, expected
):
    """``config.cuda_fp_precision`` selects the CUDA state dtype when the call
    names none; an explicit dtype wins; host kernels ignore the setting."""
    monkeypatch.setattr(config, "kernel_config", kernel)
    monkeypatch.setattr(config, "cuda_fp_precision", fp)
    assert resolve_dtype(dtype) == np.dtype(expected)


def test_invalid_cuda_fp_precision_raises(monkeypatch):
    monkeypatch.setattr(config, "kernel_config", "cuda_etd2")
    monkeypatch.setattr(config, "cuda_fp_precision", 16)
    with pytest.raises(ValueError, match="cuda_fp_precision"):
        resolve_dtype()


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
