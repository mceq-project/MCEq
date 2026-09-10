"""Installed-package smoke checks for the separately staged release databases."""

import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

BUNDLES = os.environ.get("MCEQ_RELEASE_BUNDLES")
pytestmark = pytest.mark.skipif(not BUNDLES, reason="release bundles not staged")


def bundle_path(package):
    root = Path(BUNDLES)
    manifest = json.loads((root / "bundle_manifest_v2.0.0-rc1.json").read_text())
    return root / manifest["packages"][package]["file"]


def test_inventory():
    with (
        h5py.File(bundle_path("base")) as base,
        h5py.File(bundle_path("extra_models")) as extra,
    ):
        assert "DPMJETIII193" in base["hadronic_interactions/air"]
        assert "FLUKA20251" in base["hadronic_interactions/air"]
        for suffix in ("RHO", "BAR", "MIXED", "STRANGE"):
            name = "SIBYLL23ESTAR" + suffix
            assert name not in base["hadronic_interactions/air"]
            assert name in extra["hadronic_interactions/air"]
        for medium in base["continuous_losses"]:
            for kind in ("ionization", "total"):
                for particle, ds in base[f"continuous_losses/{medium}/{kind}"].items():
                    np.testing.assert_array_equal(
                        ds[:], extra[f"continuous_losses/{medium}/{kind}/{particle}"][:]
                    )


@pytest.mark.parametrize(
    "package,model,blend",
    [
        ("base", "SIBYLL23E", None),
        ("base", "SIBYLL23E", "FLUKA20251"),
        ("base", "SIBYLL23E", "DPMJETIII193"),
        ("extra_models", "SIBYLL23ESTARRHO", None),
        ("extra_models", "SIBYLL21", None),
        ("2d", "FLUKA20251", None),
    ],
)
def test_solve(package, model, blend, monkeypatch, tmp_path):
    import crflux.models as pm

    from MCEq import config
    from MCEq.core import MCEqRun

    for name, value in {
        "mceq_db_fname": str(bundle_path(package)),
        "data_dir": tmp_path,
        "kernel_config": "numpy_etd2",
        "e_min": 10.0 if package == "2d" else 1.0,
        "e_max": 100.0 if package == "2d" else 1e5,
        "enable_em": False,
        "enable_default_tracking": False,
        "density_model": ("CORSIKA", ("USStd", None)),
        "debug_level": 0,
    }.items():
        monkeypatch.setattr(config, name, value)
    run = MCEqRun(
        interaction_model=model,
        low_energy_model=blend,
        primary_model=(pm.HillasGaisser2012, "H3a"),
        theta_deg=0.0,
    )
    run.solve()
    for species in ("total_mu+", "total_mu-", "total_numu", "total_antinumu"):
        flux = run.get_solution(species)
        assert np.isfinite(flux).all()
        assert np.any(flux > 0)
