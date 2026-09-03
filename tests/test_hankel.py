"""Inverse Hankel transform tests — legacy method baseline.

These tests pin the accuracy of PR #48's cubic-interp + np.trapz inverse
Hankel transform on an analytic round-trip. Task 2.2 introduces a
Filon-J₀ alternative that must beat these baselines by ≥ 5×.
"""

import numpy as np
import pytest

# 2D database k-grid — geometric integer-valued
K_GRID = np.array(
    [
        0,
        1,
        2,
        3,
        4,
        6,
        9,
        12,
        17,
        23,
        32,
        44,
        61,
        84,
        115,
        158,
        217,
        299,
        410,
        563,
        773,
        1061,
        1457,
        2000,
    ]
)


def gaussian_F(k, sigma):
    """Closed-form zeroth-order Hankel transform of f(θ)=exp(−θ²/(2σ²)).

    For axisymmetric f(θ): F(k) = ∫₀^∞ f(θ) J₀(kθ) θ dθ.
    Derivation: F(k) = σ² · exp(-(σk)²/2).
    """
    return sigma**2 * np.exp(-((sigma * k) ** 2) / 2.0)


def gaussian_f(theta, sigma):
    return np.exp(-(theta**2) / (2.0 * sigma**2))


@pytest.mark.parametrize("sigma", [0.001, 0.005, 0.01, 0.05, 0.1])
def test_legacy_round_trip_baseline(sigma):
    """Document the legacy method's recovery error on a Gaussian round-trip.

    The error is largest for narrow Gaussians (small σ) where J₀ oscillates
    rapidly across the irregular k-grid. Loose pass thresholds — these are
    *baselines* to be beaten by the Filon method (Task 2.2).
    """
    from MCEq.hankel import inverse_hankel_legacy

    F_k = gaussian_F(K_GRID, sigma)
    theta = np.linspace(0, np.pi / 2, 600)
    f_recovered = inverse_hankel_legacy(F_k, K_GRID, theta, oversample_res=5)
    f_true = gaussian_f(theta, sigma)
    err_max = np.max(np.abs(f_recovered - f_true))
    f_norm = max(np.abs(f_true).max(), 1e-30)
    rel_err = err_max / f_norm
    # Generous bounds; this test is meant to *document*, not constrain.
    assert rel_err < 2.0, f"sigma={sigma}: rel_err={rel_err:.3e}"


def test_legacy_returns_correct_shape():
    """The legacy function returns an array with the same shape as `theta`."""
    from MCEq.hankel import inverse_hankel_legacy

    F_k = gaussian_F(K_GRID, 0.05)
    theta = np.linspace(0, 1.0, 100)
    out = inverse_hankel_legacy(F_k, K_GRID, theta, oversample_res=5)
    assert out.shape == theta.shape
    assert np.all(np.isfinite(out))


def test_legacy_baseline_summary(capsys):
    """Print the σ vs rel-err table to stdout (visible with pytest -s)."""
    from MCEq.hankel import inverse_hankel_legacy

    print()
    print(f"{'sigma':>8}  {'rel_err_max':>14}")
    for sigma in (0.001, 0.005, 0.01, 0.05, 0.1):
        F_k = gaussian_F(K_GRID, sigma)
        theta = np.linspace(0, np.pi / 2, 600)
        f_rec = inverse_hankel_legacy(F_k, K_GRID, theta)
        f_true = gaussian_f(theta, sigma)
        err = np.max(np.abs(f_rec - f_true)) / max(np.abs(f_true).max(), 1e-30)
        print(f"{sigma:>8.3g}  {err:>14.3e}")
