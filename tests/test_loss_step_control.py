"""The time step follows the actual explicit loss stencil, including 2D."""

from types import SimpleNamespace

import numpy as np
import pytest

from MCEq.driver.paths import continuous_loss_dx_cap
from MCEq.operators.stiffness import continuous_loss_rate
from MCEq.solvers.path import etd2_nonuniform_path


def test_stiffness_tracks_grid_and_secant():
    # An upwind d/dE loss band has rate dE/dX / deltaE.
    def band(de):
        return np.array([[-1.0, 1.0], [0.0, -1.0]]) * 0.002 / de

    rate = continuous_loss_rate([band(0.01)])
    assert continuous_loss_rate([band(0.001)]) == pytest.approx(10 * rate)
    assert continuous_loss_rate([band(0.01)], [1.0, 4.0]) == pytest.approx(4 * rate)
    assert continuous_loss_rate([np.diag([-1e20, -1e30])]) == 0
    assert continuous_loss_rate([]) == 0


def test_nonfinite_stencil_rejected():
    with pytest.raises(ValueError, match="Nonfinite"):
        continuous_loss_rate([np.array([[0.0, np.nan], [0.0, 0.0]])])


def test_cap_rebuilt_with_operator_and_secant():
    bands = {(0, 0): np.array([[-1.0, 2.0], [0.0, -1.0]])}
    sec = None
    run = SimpleNamespace(
        matrix_builder=SimpleNamespace(_contloss_bands=bands),
        int_m=object(),
        _resolve_secant=lambda: sec,
    )
    assert continuous_loss_dx_cap(run) == 0.25
    sec = {"lam": np.array([1.0, 4.0])}
    assert continuous_loss_dx_cap(run) == 0.0625
    bands[(0, 0)] *= 2
    run.int_m = object()
    assert continuous_loss_dx_cap(run) == 0.03125


def test_path_floor_cannot_exceed_ceiling():
    settings = SimpleNamespace(
        X_start=0.0, etd2_path=dict(eps=0.01, dX_max=0.001, dX_min=0.01, fd_span=0.01)
    )
    density = SimpleNamespace(r_X2rho=lambda x: np.ones_like(x), max_X=1.0)
    with pytest.raises(ValueError, match="dX_min"):
        etd2_nonuniform_path(density, step=settings)


def test_run_guard_clamps_override_and_floor_and_refreshes_settings():
    from MCEq.driver.paths import calculate_integration_path

    settings = SimpleNamespace(
        X_start=0.0, etd2_path=dict(eps=0.01, dX_max=20.0, dX_min=0.01, fd_span=0.01)
    )
    run = SimpleNamespace(
        config=SimpleNamespace(
            solver=settings, em=SimpleNamespace(adaptive_step=False)
        ),
        matrix_builder=SimpleNamespace(
            _contloss_bands={0: np.array([[-1e3, 1e3], [0.0, -1e3]])}
        ),
        int_m=object(),
        _resolve_secant=lambda: None,
        _em_cascade_step_scale=lambda: 0.0,
        integration_path=None,
        density_model=SimpleNamespace(r_X2rho=lambda x: np.ones_like(x), max_X=0.002),
    )
    calculate_integration_path(run, None, "X", dX_max=1e4)
    assert max(run.integration_path[1]) <= 0.0005
    before = run._cached_etd2_path_params
    settings.etd2_path["eps"] = 0.005
    calculate_integration_path(run, None, "X", dX_max=1e4)
    assert run._cached_etd2_path_params != before


def test_temporal_tolerances_do_not_change_spatial_stencil(monkeypatch):
    from MCEq import config
    from MCEq.operators.matrix_builder import MatrixBuilder

    builder = MatrixBuilder.__new__(MatrixBuilder)
    edges = np.geomspace(0.001, 1e6, 91)
    builder._energy_grid = SimpleNamespace(
        b=edges, c=np.sqrt(edges[:-1] * edges[1:]), d=90
    )
    builder._pman = {2212: SimpleNamespace(is_em=False, dEdX=np.full(90, -0.01))}
    monkeypatch.setattr(config, "etd2_path", dict(config.etd2_path, dX_max=20.0))
    coarse = builder._upwind_rows_for(2212)
    monkeypatch.setattr(config, "etd2_path", dict(config.etd2_path, dX_max=0.001))
    assert coarse > 8  # Exercise the species-adaptive region, not just its floor.
    assert builder._upwind_rows_for(2212) == coarse
