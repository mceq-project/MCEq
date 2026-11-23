"""``enable_cont_rad_loss=False`` keeps the ionization loss band."""

import crflux.models as pm
import numpy as np

from MCEq import config
from MCEq.core import MCEqRun


def _run(monkeypatch, cont_rad):
    monkeypatch.setattr(config, "enable_cont_rad_loss", cont_rad)
    return MCEqRun(
        interaction_model="SIBYLL21",
        theta_deg=0.0,
        primary_model=(pm.HillasGaisser2012, "H3a"),
    )


def _muon_band(run):
    """Dense ``mu-`` diagonal block of the run's continuous-loss band."""
    mu = run.pman["mu-"]
    sl = slice(mu.lidx, mu.uidx)
    return run.matrix_builder.dEdx_band[sl, sl].toarray(), mu


def test_muons_lose_energy_without_radiative_losses(monkeypatch):
    run = _run(monkeypatch, cont_rad=False)
    band, mu = _muon_band(run)
    ionization = run._mceq_db.continuous_loss_db()["index_d"][(13, 0)]
    np.testing.assert_allclose(np.abs(mu.dEdX), np.abs(ionization))
    assert np.count_nonzero(band) > 0
    np.testing.assert_allclose(
        band, run.matrix_builder.cont_loss_operator(mu.pdg_id), rtol=1e-12
    )


def test_radiative_losses_add_to_the_ionization_band(monkeypatch):
    ion = np.abs(_run(monkeypatch, cont_rad=False).pman["mu-"].dEdX)
    tot = np.abs(_run(monkeypatch, cont_rad=True).pman["mu-"].dEdX)
    assert np.all(tot >= ion)
    assert tot[-1] > 2 * ion[-1]
