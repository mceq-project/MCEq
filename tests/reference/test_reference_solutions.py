"""The stored results must still come out of the code.

Fluxes are compared where they carry weight -- within four decades of the
peak -- because the outermost bins of the CI energy window hold values near
zero, where a relative comparison measures nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from . import values

#: Relative tolerance on the solved spectra. A BLAS or libm difference moves
#: them by ~1e-13; this leaves room for that and nothing else.
FLUX_RTOL = 1e-6
#: Relative tolerance on operator norms, which are sums over many terms.
NORM_RTOL = 1e-10
#: Bins below this fraction of a spectrum's peak are not compared.
FLOOR = 1e-4

pytestmark = pytest.mark.xdist_group("reference")


@pytest.fixture(scope="module")
def produced():
    if not values.STORE.exists():
        pytest.skip(f"no stored reference values; run `python -m {values.__name__}`")
    return values.compute()


@pytest.fixture(scope="module")
def stored():
    if not values.STORE.exists():
        pytest.skip("no stored reference values")
    return values.load()


def test_stored_and_produced_have_the_same_keys(stored, produced):
    assert sorted(stored) == sorted(produced)


def test_energy_grid_unchanged(stored, produced):
    np.testing.assert_array_equal(stored["e_grid"], produced["e_grid"])


def test_tracked_species_unchanged(stored, produced):
    np.testing.assert_array_equal(stored["pdg_ids"], produced["pdg_ids"])
    assert int(stored["dim_states"]) == int(produced["dim_states"])


@pytest.mark.parametrize("matrix", ["int_m", "dec_m"])
def test_operator_size_and_norm_unchanged(stored, produced, matrix):
    assert int(stored[f"{matrix}/nnz"]) == int(produced[f"{matrix}/nnz"])
    assert float(produced[f"{matrix}/norm"]) == pytest.approx(
        float(stored[f"{matrix}/norm"]), rel=NORM_RTOL
    )


@pytest.mark.parametrize("theta", values.ZENITHS)
@pytest.mark.parametrize("lepton", values.LEPTONS)
def test_lepton_spectrum_unchanged(stored, produced, theta, lepton):
    key = f"flux/{theta:g}/{lepton}"
    want, got = stored[key], produced[key]
    assert want.shape == got.shape
    compared = want > want.max() * FLOOR
    assert compared.any(), f"{key} carries no flux"
    rel = np.abs(got[compared] - want[compared]) / want[compared]
    worst = int(np.argmax(rel))
    assert rel.max() < FLUX_RTOL, (
        f"{key} moved by {rel.max():.3e} at "
        f"E = {produced['e_grid'][compared][worst]:.4e} GeV "
        f"(stored {want[compared][worst]:.6e}, here {got[compared][worst]:.6e})"
    )
