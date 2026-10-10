"""Bin-averaged decay rate: ``<E_j/E>`` over a parent bin for a flux ``E^-gamma``.

The decay source of bin j is the bin integral of ``phi/lambda_dec``; with
``lambda_dec`` proportional to E the table value ``Phi_j/lambda_dec(E_j)`` is low
by ``<E_j/E>``. Databases carrying ``decays/bin_average_index`` (the parent
decay-spectrum index g, flux index g - 1) switch the factor on.
"""

import numpy as np
import pytest

from MCEq import config
from MCEq.misc import energy_grid
from MCEq.species.particle import MCEqParticle, bin_average_inverse_energy

H = 0.1  # decades per bin
L = H * np.log(10.0)


def closed_form(gamma):
    if abs(gamma - 1.0) < 1e-12:
        return 2.0 * np.sinh(L / 2) / L
    if abs(gamma) < 1e-12:
        return (L / 2) / np.sinh(L / 2)
    return (gamma - 1) / gamma * np.sinh(gamma * L / 2) / np.sinh((gamma - 1) * L / 2)


@pytest.mark.parametrize("gamma", [0.0, 1.0, 1.5, 2.2, 2.7, 3.2, 3.7])
def test_factor_matches_the_closed_form(gamma):
    assert bin_average_inverse_energy(gamma)[0] == pytest.approx(closed_form(gamma), rel=5e-6)  # 64-node midpoint rule


def test_factor_is_vectorised_and_monotonic():
    g = np.linspace(-1.0, 6.0, 15)
    f = bin_average_inverse_energy(g)
    assert f.shape == g.shape
    assert np.all(np.diff(f) > 0)
    assert bin_average_inverse_energy(3.2)[0] == pytest.approx(1.0119, abs=1e-4)


def _particle(pdg):
    e_bins = np.logspace(-1, 6, 71)
    c = np.sqrt(e_bins[1:] * e_bins[:-1])
    grid = energy_grid(c=c, b=e_bins, w=np.diff(e_bins), d=len(c))
    return MCEqParticle(pdg, 0, energy_grid=grid, physics=config.physics)


def test_inverse_decay_length_uses_the_factor_when_set():
    p = _particle(211)
    base = p.inverse_decay_length()
    p._decay_rate_factor = bin_average_inverse_energy(np.full(len(base), 2.7))
    assert np.allclose(p.inverse_decay_length(), base * p._decay_rate_factor, rtol=0, atol=0)
    p._decay_rate_factor = None
    assert np.array_equal(p.inverse_decay_length(), base)


def test_stable_and_zero_ctau_paths_are_untouched():
    p = _particle(2212)
    p._decay_rate_factor = np.full(70, 1.01)
    assert np.array_equal(p.inverse_decay_length(), np.zeros(70))
    q = _particle(211)
    q.ctau = 0.0
    q._decay_rate_factor = np.full(70, 1.01)
    assert np.all(np.isinf(q.inverse_decay_length()))


def test_config_switch_is_a_physics_field():
    from MCEq.config.groups import GROUPS

    assert "bin_averaged_decay_rate" in GROUPS["physics"]
    assert config.bin_averaged_decay_rate is True
