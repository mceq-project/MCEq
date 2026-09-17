"""Installed-package smoke checks against the staged v2 data packages.

Staged by the release-bundle workflow into ``MCEQ_RELEASE_BUNDLES`` (the
manifest, the 1D packages, the 2D bases). The SIBYLL23E 2D model file is
deliberately *not* staged: its test makes the loader resolve it through the
manifest and fetch it. The draft release needs an authenticated download,
which the loader's ``_download_file`` does not do, so the ``gh_fetch`` fixture
routes that one fetch through ``gh release download``; the loader still
verifies the manifest sha256 and publishes the file atomically.
"""

import os
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

BUNDLES = os.environ.get("MCEQ_RELEASE_BUNDLES")
MANIFEST = "mceq_db_manifest_v2.yaml"
EXTRA = "mceq_extra_models_air_1d_v2.h5"
SIBYLL_2D = "mceq_model_SIBYLL23E_air_2d_v2.h5"
RC7 = "mceq_base_air_2d_rc7_v2.h5"
pytestmark = pytest.mark.skipif(not BUNDLES, reason="release bundles not staged")


def manifest():
    return yaml.safe_load((Path(BUNDLES) / MANIFEST).read_text())


def package(kind):
    """Default package of a grid family (``1d``/``2d``) or a file name, staged."""
    m = manifest()
    return Path(BUNDLES) / m["defaults"].get(kind, kind)


@pytest.fixture
def gh_fetch(monkeypatch):
    """Serve loader fetches from the (draft) release through ``gh``."""
    from MCEq.data import download

    def fetch(url, outfile, checksum=None):
        name = url.rsplit("/", 1)[-1]
        staged = Path(BUNDLES) / name
        if not staged.is_file():
            repo = os.environ.get("GITHUB_REPOSITORY", "mceq-project/MCEq")
            subprocess.run(
                [
                    "gh",
                    "release",
                    "download",
                    manifest()["release"],
                    "--repo",
                    repo,
                    "--dir",
                    BUNDLES,
                    "--pattern",
                    name,
                ],
                check=True,
            )
        digest = download.FileIntegrityCheck(staged).get_file_checksum()
        if checksum is not None and digest != checksum:
            raise OSError(f"{name}: sha256 {digest} != manifest {checksum}")
        if staged.resolve() != Path(outfile).resolve():
            shutil.copy(staged, outfile)

    monkeypatch.setattr(download, "_download_file", fetch)


def _configure(monkeypatch, primary, data_dir, **extra):
    from MCEq import config

    settings = {
        "mceq_db_fname": str(primary),
        "data_dir": Path(data_dir),
        "mceq_db_manifest": MANIFEST,
        "kernel_config": "numpy_etd2",
        "enable_em": False,
        "enable_default_tracking": False,
        "density_model": ("CORSIKA", ("USStd", None)),
        "debug_level": 0,
    }
    settings.update(extra)
    for name, value in settings.items():
        monkeypatch.setattr(config, name, value)


def _finite_positive(run, species):
    for s in species:
        flux = run.get_solution(s)
        assert np.isfinite(flux).all()
        assert np.any(flux > 0)


def test_inventory():
    m = manifest()
    assert m["schema"] == 1
    base_1d, extra = package("1d"), Path(BUNDLES) / EXTRA
    entry = m["files"][base_1d.name]
    assert entry["spine"] and entry["models"]["air"] == ["FLUKA20251", "SIBYLL23E"]
    with h5py.File(base_1d) as base, h5py.File(extra) as ext:
        assert base.attrs["mceq_db_manifest"] == MANIFEST
        fluka = base["cross_sections/air/FLUKA20251"]
        parents = list(fluka.attrs["parents"])
        assert {2212, -2212, 2112, -2112, 211, -211, 321, -321, 130, 310} <= set(
            parents
        )
        for positive in (2212, 2112, 211, 321):
            assert not np.array_equal(
                fluka[:, parents.index(positive)], fluka[:, parents.index(-positive)]
            )
        for name in m["files"][EXTRA]["models"]["air"]:
            assert name not in base["hadronic_interactions/air"]
            assert name in ext["hadronic_interactions/air"]
        for medium in base["continuous_losses"]:
            for kind in ("ionization", "total"):
                for particle, ds in base[f"continuous_losses/{medium}/{kind}"].items():
                    np.testing.assert_array_equal(
                        ds[:], ext[f"continuous_losses/{medium}/{kind}/{particle}"][:]
                    )
    # every model of the 1D grid is in exactly one 1D package
    grid = entry["grid"]
    owner = {}
    for name, e in m["files"].items():
        if e["grid"] == grid:
            for model in e["models"]["air"]:
                assert model not in owner, (model, owner[model], name)
                owner[model] = name
    assert len(owner) == 15


@pytest.mark.parametrize(
    "model,blend",
    [
        ("SIBYLL23E", None),
        ("SIBYLL23E", "FLUKA20251"),
        ("FLUKA20251", None),
        ("SIBYLL23ESTARRHO", "FLUKA20251"),  # resolved through the manifest
        ("DPMJETIII193", None),  # resolved through the manifest
        ("SIBYLL21", None),
    ],
)
def test_solve_1d(model, blend, monkeypatch):
    import crflux.models as pm

    from MCEq.core import MCEqRun

    _configure(monkeypatch, package("1d"), BUNDLES, e_min=1.0, e_max=1e5)
    run = MCEqRun(
        interaction_model=model,
        low_energy_model=blend,
        primary_model=(pm.HillasGaisser2012, "H3a"),
        theta_deg=0.0,
    )
    run.solve()
    _finite_positive(run, ("total_mu+", "total_mu-", "total_numu", "total_antinumu"))


def test_extra_package_loads_standalone(monkeypatch):
    """The extra-models package carries the spine and loads as a primary."""
    import crflux.models as pm

    from MCEq.core import MCEqRun

    _configure(monkeypatch, Path(BUNDLES) / EXTRA, BUNDLES, e_min=1.0, e_max=1e5)
    run = MCEqRun(
        interaction_model="QGSJETIII",
        primary_model=(pm.HillasGaisser2012, "H3a"),
        theta_deg=0.0,
    )
    run.solve()
    _finite_positive(run, ("total_numu",))


def test_missing_model_error_names_manifest(monkeypatch):
    from MCEq.core import MCEqRun

    _configure(monkeypatch, package("1d"), BUNDLES, e_min=1.0, e_max=1e5)
    with pytest.raises(Exception, match=MANIFEST):
        MCEqRun(interaction_model="NOSUCHMODEL", primary_model=None, theta_deg=0.0)


@pytest.mark.parametrize("primary", ["2d", RC7])
def test_solve_2d_fluka(primary, monkeypatch, tmp_path):
    import crflux.models as pm

    from MCEq.core import MCEqRun

    _configure(monkeypatch, package(primary), tmp_path, e_min=10.0, e_max=100.0)
    run = MCEqRun(
        interaction_model="FLUKA20251",
        low_energy_model=None,
        primary_model=(pm.HillasGaisser2012, "H3a"),
        theta_deg=0.0,
    )
    run.solve()
    _finite_positive(run, ("total_mu+", "total_mu-", "total_numu", "total_antinumu"))


def test_solve_2d_sibyll_via_model_file(gh_fetch, monkeypatch):
    """SIBYLL23E 2D lives in a model file the loader has to fetch."""
    import crflux.models as pm

    from MCEq.core import MCEqRun

    _configure(monkeypatch, package("2d"), BUNDLES, e_min=10.0, e_max=100.0)
    run = MCEqRun(
        interaction_model="SIBYLL23E",
        low_energy_model="FLUKA20251",
        primary_model=(pm.HillasGaisser2012, "H3a"),
        theta_deg=0.0,
    )
    assert (Path(BUNDLES) / SIBYLL_2D).is_file()
    run.solve()
    _finite_positive(run, ("total_mu+", "total_numu"))


def test_2d_low_energy_convergence(monkeypatch, tmp_path):
    """Full production grid; finite output alone missed the previous blowup."""
    from MCEq import config
    from MCEq.config.run import RunConfig
    from MCEq.core import MCEqRun

    monkeypatch.setattr(config.paths, "mceq_db_fname", str(package("2d")))
    monkeypatch.setattr(config.paths, "data_dir", tmp_path)
    monkeypatch.setattr(config.debug, "level", 0)
    run = MCEqRun(
        interaction_model="FLUKA20251",
        low_energy_model=None,
        primary_model=None,
        theta_deg=30.0,
        density_model=("CORSIKA", ("USStd", None)),
        config=RunConfig.snapshot(),
    )
    try:
        assert run.e_grid[0] < 0.15
        run.set_single_primary_particle(1e3, pdg_id=2212)
        run.solve()
        h = float(max(run.integration_path[1]))
        species = ("total_mu+", "total_mu-", "total_numu", "total_antinumu")
        default = {p: run.get_solution(p).copy() for p in species}
        run.solve(eps=0.003, dX_max=h / 2)
        for p in species:
            ref = run.get_solution(p)
            assert np.isfinite(ref).all() and np.isfinite(default[p]).all()
            mask = (run.e_grid < 100.0) & (ref > 1e-10 * np.max(ref))
            assert mask.any()
            assert (default[p][mask] > 0).all()
            np.testing.assert_allclose(default[p][mask], ref[mask], rtol=0.01, atol=0)
    finally:
        run.close()
