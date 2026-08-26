"""Production defaults for 2D databases: the "auto" sec(theta) transport
tri-state.

The tri-state resolution is a pure function of (config flag, is_2d) and is
tested unbound with a stub. End-to-end secant solves are covered by the
2026-08-10 secant runs, not the unit suite (the operator build is
minutes-scale without its disk cache).

(The former FLUKA-only 2D model restriction and its refusal tests were
removed when runtime per-kappa HE/LE blending on multi-model 2D databases
became the production design, superseding the postponed 1D-HE/2D-LE
kappa-window hybrid.)
"""

from types import SimpleNamespace

import numpy as np
import pytest

from MCEq import config
from MCEq.core import MCEqRun


def _mode_for(flag, is_2d):
    stub = SimpleNamespace(_mceq_db=SimpleNamespace(is_2d=is_2d))
    saved = config.secant_theta_transport
    config.secant_theta_transport = flag
    try:
        return MCEqRun._secant_mode(stub)
    finally:
        config.secant_theta_transport = saved


def test_secant_auto_cap_clips_at_50():
    """The 'auto' cap is 90 - zenith + 5 snapped to 5-deg steps, clipped
    to [50, 75]: below ~45 deg the S_P eigenbasis is numerically
    defective (near-nilpotent coupling), so inclined zeniths run at 50."""

    def cap_for(theta_z):
        stub = SimpleNamespace(density_model=SimpleNamespace(theta_deg=theta_z))
        return MCEqRun._secant_theta_cap_deg(stub)

    saved = config.secant_theta_cap_deg
    config.secant_theta_cap_deg = "auto"
    try:
        assert cap_for(0.0) == 75.0
        assert cap_for(30.0) == 65.0
        assert cap_for(45.0) == 50.0
        assert cap_for(60.0) == 50.0  # not 35 — defective below ~45
        assert cap_for(85.0) == 50.0
        config.secant_theta_cap_deg = 62.5
        assert cap_for(60.0) == 62.5  # explicit float passes through
    finally:
        config.secant_theta_cap_deg = saved


def test_secant_auto_caps_cover_heterogeneous_fullsky_and_both_hemispheres():
    zeniths = np.array([0, 25, 30, 35, 40, 45, 60, 85, 90, 95, 120, 180], dtype=float)
    expected = np.array([75, 70, 65, 60, 55, 50, 50, 50, 50, 50, 50, 50])
    stub = object.__new__(MCEqRun)
    stub.density_model = SimpleNamespace(theta_deg=17.0)

    saved = config.secant_theta_cap_deg
    try:
        config.secant_theta_cap_deg = "auto"
        resolved = np.array([stub._secant_theta_cap_deg(theta_deg=z) for z in zeniths])
        np.testing.assert_array_equal(resolved, expected)

        conditions = [{"zenith_deg": float(z)} for z in zeniths]
        np.testing.assert_array_equal(
            stub._secant_caps_for_conditions(conditions, len(conditions)),
            expected,
        )

        config.secant_theta_cap_deg = 62.5
        np.testing.assert_array_equal(
            stub._secant_caps_for_conditions(conditions, len(conditions)),
            np.full(len(conditions), 62.5),
        )
    finally:
        config.secant_theta_cap_deg = saved


def test_secant_mode_resolution():
    # Default "auto": on for 2D, no-op for 1D.
    assert _mode_for("auto", True) == "auto"
    assert _mode_for("auto", False) == "off"
    # Explicit True: required (unsupported paths raise); still off in 1D.
    assert _mode_for(True, True) == "require"
    assert _mode_for(True, False) == "off"
    # Explicit False: off everywhere.
    assert _mode_for(False, True) == "off"
    assert _mode_for(False, False) == "off"


def test_batch_secant_backend_semantics_never_downgrade():
    stub = object.__new__(MCEqRun)
    stub._mceq_db = SimpleNamespace(is_2d=True)
    stub._int_m_stack = None
    stub._em_rho_grid = None

    saved_flag = config.secant_theta_transport
    saved_kernel = config.kernel_config
    try:
        for flag in ("auto", True):
            config.secant_theta_transport = flag
            config.kernel_config = "numpy_etd2"
            assert stub._require_batch_secant_supported("test") is True
            config.kernel_config = "cuda_etd2"
            assert stub._require_batch_secant_supported("test") is True

            for unsupported in ("mkl_etd2", "accelerate_etd2"):
                config.kernel_config = unsupported
                with pytest.raises(NotImplementedError, match="No paraxial downgrade"):
                    stub._require_batch_secant_supported("test")

        config.secant_theta_transport = False
        config.kernel_config = "accelerate_etd2"
        assert stub._require_batch_secant_supported("test") is False

        # A 1-D database always resolves off, independent of the requested
        # flag and backend.
        stub._mceq_db = SimpleNamespace(is_2d=False)
        config.secant_theta_transport = True
        config.kernel_config = "mkl_etd2"
        assert stub._require_batch_secant_supported("test") is False
    finally:
        config.secant_theta_transport = saved_flag
        config.kernel_config = saved_kernel


def test_batch_secant_rho_stack_reports_clear_error():
    stub = object.__new__(MCEqRun)
    stub._mceq_db = SimpleNamespace(is_2d=True)
    stub._int_m_stack = [object()]
    stub._em_rho_grid = np.array([1.0])

    saved_flag = config.secant_theta_transport
    saved_kernel = config.kernel_config
    try:
        config.secant_theta_transport = "auto"
        config.kernel_config = "numpy_etd2"
        with pytest.raises(NotImplementedError, match="rho-stack"):
            stub._require_batch_secant_supported("test")
    finally:
        config.secant_theta_transport = saved_flag
        config.kernel_config = saved_kernel
