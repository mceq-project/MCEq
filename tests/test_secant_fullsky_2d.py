"""Optional production-DB regression for secant full-sky transport.

Set ``MCEQ_SECANT_2D_REGRESSION=1`` and ``MCEQ_SECANT_2D_DB=/path/to/db.h5``
to run it.  The synthetic tests in :mod:`test_secant_multirhs` are the fast CI
coverage; this gate checks real particle indexing, all 48 kappa modes, both
hemispheres, auto/fixed caps, and the exact-J1 observable.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import CubicSpline
from scipy.special import j1

pytestmark = pytest.mark.skipif(
    os.environ.get("MCEQ_SECANT_2D_REGRESSION") != "1",
    reason="set MCEQ_SECANT_2D_REGRESSION=1 for the production 48-mode gate",
)


ZENITHS = np.array([0, 25, 30, 35, 40, 45, 60, 85, 90, 95, 120, 180], dtype=float)
EXPECTED_CAPS = np.array([75, 70, 65, 60, 55, 50, 50, 50, 50, 50, 50, 50])


def _j1_integral(amplitudes, k_grid):
    theta_max = np.pi / 2.0
    k_fine = np.linspace(k_grid[0], k_grid[-1], 10001)
    dk = np.diff(k_fine)
    trap = np.empty_like(k_fine)
    trap[0] = 0.5 * dk[0]
    trap[-1] = 0.5 * dk[-1]
    trap[1:-1] = 0.5 * (dk[:-1] + dk[1:])
    weights = theta_max * j1(k_fine * theta_max) * trap
    fine = CubicSpline(k_grid, amplitudes, axis=0)(k_fine)
    return np.tensordot(weights, fine, axes=(0, 0))


def _particle_modes(mceq, state, keys):
    n_k = int(mceq._mceq_db.n_k)
    K = state.shape[1]
    modes = state.reshape(n_k, mceq.dim_states, K)
    out = np.zeros((n_k, len(mceq.e_grid), K))
    for key in keys:
        first = mceq.pman.pdg2mceqidx[key] * len(mceq.e_grid)
        out += modes[:, first : first + len(mceq.e_grid), :]
    return out


def _single_axis_to_fraction(mceq, phi0, fraction, *, eps, dX_max):
    """Run the production single-axis kernel only to a saved endpoint.

    ``solve(int_grid=[endpoint])`` records at the requested depth but then
    propagates through ``max_X``. Truncate that exact integration path before
    dispatch so this optional real-matrix test stays bounded while retaining an
    independent single-axis secant kernel as its reference.
    """
    endpoint = fraction * float(mceq.density_model.max_X)
    mceq._calculate_integration_path(
        [endpoint],
        "X",
        eps=eps,
        dX_max=dX_max,
    )
    path = mceq._truncate_path_at_last_snapshot(
        mceq.integration_path, "test_secant_fullsky_2d"
    )
    mceq.integration_path = path
    mceq._phi0[:] = phi0
    mceq.solve(skip_integration_path=True)
    return mceq._solution.copy(), path[0]


def test_real_48mode_fullsky_auto_fixed_caps_and_per_rhs_primaries(monkeypatch):
    db = Path(os.environ.get("MCEQ_SECANT_2D_DB", ""))
    if not db.is_file():
        pytest.skip(f"MCEQ_SECANT_2D_DB is not a file: {db}")

    import crflux.models as crf

    from MCEq import config
    from MCEq.core import MCEqRun
    from MCEq.geometry.density_profiles import MSIS00IceCubeCentered

    kernel = os.environ.get("MCEQ_SECANT_2D_KERNEL", "numpy_etd2")
    if kernel not in ("numpy_etd2", "cuda_etd2"):
        pytest.fail(
            f"MCEQ_SECANT_2D_KERNEL must be numpy_etd2 or cuda_etd2, got {kernel!r}"
        )
    if kernel == "cuda_etd2":
        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() < 1:
                pytest.skip("CUDA regression requested but no device is visible")
        except Exception as exc:
            pytest.skip(f"CUDA regression requested but unavailable: {exc}")

    monkeypatch.setattr(config, "data_dir", db.resolve().parent)
    monkeypatch.setattr(config, "mceq_db_fname", db.name)
    monkeypatch.setattr(config, "kernel_config", kernel)
    monkeypatch.setattr(config, "cuda_fp_precision", 64)
    monkeypatch.setattr(config, "e_min", 0.1)
    monkeypatch.setattr(config, "e_max", 100.0)
    monkeypatch.setattr(config, "debug_level", 0)
    monkeypatch.setattr(config, "secant_theta_transport", "auto")
    monkeypatch.setattr(config, "secant_theta_cap_deg", "auto")
    monkeypatch.setattr(config, "secant_theta_e_gate", 31.6)
    monkeypatch.setattr(config, "muon_multiple_scattering", True)
    monkeypatch.setattr(
        config,
        "low_energy_extension",
        dict(
            config.low_energy_extension,
            model="FLUKA20251",
            he_le_transition=80,
            he_le_trwidth=0.3,
        ),
    )

    mceq = MCEqRun(
        interaction_model="SIBYLL23E",
        density_model=MSIS00IceCubeCentered("SouthPole", "January"),
        primary_model=(crf.GaisserHonda, None),
        theta_deg=0.0,
    )
    try:
        assert mceq._mceq_db.is_2d and mceq._mceq_db.n_k == 48
        base = mceq._phi0.copy()
        phi0 = np.broadcast_to(base[:, None], (mceq.dim_states, len(ZENITHS))).copy()
        # Cutoff-like, deliberately distinct primary columns. The real
        # production gate uses gtracr-derived states; these masks make
        # cross-talk/order failures obvious without invoking gtracr here.
        initial_3 = phi0.reshape(-1, len(mceq.e_grid), len(ZENITHS))
        for rhs, threshold in enumerate(np.linspace(0.1, 8.0, len(ZENITHS))):
            initial_3[:, mceq.e_grid < threshold, rhs] = 0.0
            initial_3[:, :, rhs] *= 0.7 + 0.05 * rhs

        solve_kwargs = dict(
            carousel_K=4,
            eps=0.02,
            dX_max=2.0,
            # Keep this optional CPU gate bounded while retaining the real
            # 48-mode matrices and distinct atmospheric paths. The full-depth
            # production gate below the test suite uses 0.9999 and dX_max=0.4.
            X_stop_fraction=0.01,
        )
        result = mceq.solve_fullsky(
            ZENITHS,
            phi0=phi0,
            geomagnetic_cutoff=False,
            **solve_kwargs,
        )
        batched = np.asarray(result.sol)
        assert batched.shape == (48 * mceq.dim_states, len(ZENITHS))
        np.testing.assert_array_equal(
            np.array([mceq._secant_theta_cap_deg(z) for z in ZENITHS]),
            EXPECTED_CAPS,
        )
        np.testing.assert_array_equal(
            result.pixel_index,
            np.column_stack(
                (
                    np.arange(len(ZENITHS), dtype=np.int32),
                    np.zeros(len(ZENITHS), dtype=np.int32),
                )
            ),
        )

        serial = np.empty_like(batched)
        serial_nsteps = []
        for rhs, zenith in enumerate(ZENITHS):
            mceq.set_zenith_azimuth(float(zenith), None)
            serial[:, rhs], nsteps = _single_axis_to_fraction(
                mceq,
                phi0[:, rhs],
                solve_kwargs["X_stop_fraction"],
                eps=solve_kwargs["eps"],
                dX_max=solve_kwargs["dX_max"],
            )
            serial_nsteps.append(nsteps)
        np.testing.assert_array_equal(result.nsteps_per_col, serial_nsteps)

        species = {
            "proton": ((2212, 0),),
            "muon": tuple(
                key
                for key in ((13, 0), (13, -1), (13, 1))
                if key in mceq.pman.pdg2mceqidx
            ),
            "numu": ((14, 0), (-14, 0)),
            "nue": ((12, 0), (-12, 0)),
        }
        for keys in species.values():
            batch_modes = _particle_modes(mceq, batched, keys)
            serial_modes = _particle_modes(mceq, serial, keys)
            np.testing.assert_allclose(
                batch_modes, serial_modes, rtol=3e-11, atol=3e-14
            )
            np.testing.assert_allclose(
                _j1_integral(batch_modes, mceq._mceq_db.k_grid),
                _j1_integral(serial_modes, mceq._mceq_db.k_grid),
                rtol=3e-10,
                atol=3e-13,
            )

        # A fixed cap collapses every column onto one operator and still
        # preserves each atmosphere path/initial state.
        config.secant_theta_cap_deg = 65.0
        fixed_axes = np.array([0.0, 60.0, 90.0])
        fixed_phi0 = phi0[:, [0, 6, 8]]
        fixed = mceq.solve_fullsky(
            fixed_axes,
            phi0=fixed_phi0,
            geomagnetic_cutoff=False,
            **solve_kwargs,
        ).sol
        fixed_serial = []
        for rhs, zenith in enumerate(fixed_axes):
            mceq.set_zenith_azimuth(float(zenith), None)
            column, _ = _single_axis_to_fraction(
                mceq,
                fixed_phi0[:, rhs],
                solve_kwargs["X_stop_fraction"],
                eps=solve_kwargs["eps"],
                dX_max=solve_kwargs["dX_max"],
            )
            fixed_serial.append(column)
        np.testing.assert_allclose(
            fixed, np.stack(fixed_serial, axis=1), rtol=3e-11, atol=3e-14
        )
        if kernel == "cuda_etd2":
            cache = mceq._cuda_etd2_multirhs_cache
            contexts = [entry["ctx"] for entry in cache.values()]
            assert {ctx.K for ctx in contexts}.issuperset({1, 3, 4})
            anchor = contexts[0]
            for ctx in contexts[1:]:
                assert ctx.cu_int_off is anchor.cu_int_off
                assert ctx.cu_dec_off is anchor.cu_dec_off
                assert ctx.cu_d_int is anchor.cu_d_int
                assert ctx.cu_d_dec is anchor.cu_d_dec

        # Explicit False must stay on the pre-existing paraxial carousel.
        # Compare it with independent single-axis paraxial solves on a real
        # 48-mode matrix and fail immediately if a secant operator is built.
        config.secant_theta_transport = False
        monkeypatch.setattr(
            mceq,
            "_build_secant_ops",
            lambda *args, **kwargs: pytest.fail(
                "secant operator built with secant_theta_transport=False"
            ),
        )
        assert mceq._require_batch_secant_supported("test") is False
        paraxial = mceq.solve_fullsky(
            fixed_axes,
            phi0=fixed_phi0,
            geomagnetic_cutoff=False,
            **solve_kwargs,
        ).sol
        paraxial_serial = []
        for rhs, zenith in enumerate(fixed_axes):
            mceq.set_zenith_azimuth(float(zenith), None)
            column, _ = _single_axis_to_fraction(
                mceq,
                fixed_phi0[:, rhs],
                solve_kwargs["X_stop_fraction"],
                eps=solve_kwargs["eps"],
                dX_max=solve_kwargs["dX_max"],
            )
            paraxial_serial.append(column)
        np.testing.assert_allclose(
            paraxial,
            np.stack(paraxial_serial, axis=1),
            rtol=3e-11,
            atol=3e-14,
        )
    finally:
        mceq.close()
