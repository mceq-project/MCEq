"""Manifest-described data packages: resolution, fetch safety, equality.

Runs on the fixture of ``tests/data/make_manifest_fixtures.py`` -- the CI
extra-models package cut into a plain monolith, a spine package (QGSJETII04),
a model file (SIBYLL21) and a schema-1 manifest. Pins:

* a model served from a package decodes and solves bitwise equal to the
  monolith;
* the monolith itself never consults the manifest;
* a model no package on the grid provides fails with an error naming the
  model, the manifest and the models the release does offer -- no network;
* a listed-but-absent package is fetched once, atomically, with its
  manifest sha256; a held lock fails fast with the CLI hint; a wrong payload
  leaves nothing behind;
* a package whose ``common/e_grid`` differs from the primary is refused;
* ``ensure_db_available`` fetches a manifest-listed primary from the data
  release and leaves legacy names on the legacy channel.
"""

import importlib.util
import pathlib
import shutil
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import yaml

import MCEq.config as cfg
from MCEq.core import MCEqRun
from MCEq.data import db_manifest, download
from MCEq.data.hdf5_backend import HDF5Backend

_SPEC = importlib.util.spec_from_file_location(
    "make_manifest_fixtures",
    pathlib.Path(__file__).parent / "data" / "make_manifest_fixtures.py",
)
fixtures = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fixtures)

REFERENCE_DB = fixtures.REFERENCE_DB
MONOLITH = fixtures.MONOLITH_FILE
BASE_MODEL = fixtures.BASE_MODEL
MODEL = fixtures.MODEL
MEDIUM = fixtures.MEDIUM
BASE_FILE = fixtures.BASE_FILE
MODEL_FILE = fixtures.MODEL_FILE
MANIFEST = fixtures.MANIFEST

pytestmark = pytest.mark.skipif(
    not REFERENCE_DB.is_file(),
    reason="CI data packages not present (see _run_tests.yml)",
)


@pytest.fixture(scope="session")
def release_dir(tmp_path_factory):
    return fixtures.build_fixture(tmp_path_factory.mktemp("db_manifest"))


@pytest.fixture
def no_network(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("test attempted a network fetch")

    monkeypatch.setattr(download, "_download_file", _boom)


def _paths(db_fname, data_dir, manifest=MANIFEST):
    return SimpleNamespace(
        data_dir=pathlib.Path(data_dir),
        mceq_db_fname=db_fname,
        em_db_fname="mceq_db_EM_Tsai-Max_Z7.31.h5",
        mceq_db_manifest=manifest,
        debug_level=0,
    )


def _backend(db_fname, data_dir, manifest=MANIFEST):
    grid = SimpleNamespace(e_min=None, e_max=None, dtype=None, em_standalone_grid=False)
    physics = SimpleNamespace(
        filters={
            "disabled_particles": [],
            "forced_int_cs": None,
            "replace_meson_cross_sections_with": None,
        },
        assume_nucleon_interactions_for_exotics=False,
        enable_em=False,
        enable_cont_rad_loss=True,
        fallback_to_air_cs=True,
        interaction_medium="air",
        muon_helicity_dependence=False,
        generic_losses_all_charged=False,
    )
    em = SimpleNamespace(air_density=None)
    return HDF5Backend(
        medium="air",
        paths=_paths(db_fname, data_dir, manifest),
        grid=grid,
        physics=physics,
        em=em,
    )


def _copy_release(release_dir, dest, names):
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy(release_dir / name, dest / name)
    return dest


# ---------------------------------------------------------------------------
# fixture shape and manifest
# ---------------------------------------------------------------------------


def test_fixture_is_a_release_cut(release_dir):
    manifest = db_manifest.load(release_dir / MANIFEST)
    assert set(manifest["files"]) == {BASE_FILE, MODEL_FILE}
    with h5py.File(release_dir / MONOLITH, "r") as mono:
        assert "mceq_db_manifest" not in mono.attrs
        assert f"hadronic_interactions/{MEDIUM}/{MODEL}" in mono
    with h5py.File(release_dir / BASE_FILE, "r") as base:
        assert base.attrs["mceq_db_manifest"] == MANIFEST
        assert f"hadronic_interactions/{MEDIUM}/{BASE_MODEL}" in base
        assert f"hadronic_interactions/{MEDIUM}/{MODEL}" not in base
        assert "decays" in base and "continuous_losses" in base
    with h5py.File(release_dir / MODEL_FILE, "r") as mod:
        assert f"hadronic_interactions/{MEDIUM}/{MODEL}" in mod
        assert "decays" not in mod
        grid = db_manifest.grid_signature(mod["common"].attrs)
    assert manifest["grids"][fixtures.GRID_ID] == grid
    assert db_manifest.files_for(manifest, grid, MODEL, MEDIUM) == [MODEL_FILE]
    assert db_manifest.files_for(manifest, grid, BASE_MODEL, MEDIUM) == [BASE_FILE]
    assert db_manifest.files_for(manifest, grid, MODEL, "water") == []
    other = dict(grid, e_dim=grid["e_dim"] + 1)
    assert db_manifest.files_for(manifest, other, MODEL, MEDIUM) == []


def test_monolith_never_consults_the_manifest(release_dir, no_network):
    backend = _backend(MONOLITH, release_dir, manifest="does-not-exist.yaml")
    assert backend._packages is None
    assert backend.interaction_db(MODEL)["index_d"]


# ---------------------------------------------------------------------------
# equality against the monolith
# ---------------------------------------------------------------------------


def _assert_packs_equal(store_a, store_b, tree, medium, model):
    a = store_a.read_channel_pack(f"{tree}/{medium}", model)
    b = store_b.read_channel_pack(f"{tree}/{medium}", model)
    np.testing.assert_array_equal(a.data, b.data)
    np.testing.assert_array_equal(a.indptrs, b.indptrs)
    np.testing.assert_array_equal(a.len_data, b.len_data)
    np.testing.assert_array_equal(a.tuple_idcs, b.tuple_idcs)
    assert a.description == b.description


def test_backend_reads_equal_monolith(release_dir, no_network):
    mono = _backend(MONOLITH, release_dir)
    base = _backend(BASE_FILE, release_dir)
    assert base._packages is not None
    _assert_packs_equal(
        mono._had, base._had, "hadronic_interactions", MEDIUM, BASE_MODEL
    )
    store = base._model_store(MODEL, MEDIUM, "hadronic_interactions")
    assert store is not None and store is not base._had
    _assert_packs_equal(mono._had, store, "hadronic_interactions", MEDIUM, MODEL)
    cs_mono = mono._had.read_table(f"cross_sections/{MEDIUM}/{MODEL}")
    cs_pkg = store.read_table(f"cross_sections/{MEDIUM}/{MODEL}")
    np.testing.assert_array_equal(cs_mono[0], cs_pkg[0])
    for model in (BASE_MODEL, MODEL):
        idx_mono, idx_pkg = mono.interaction_db(model), base.interaction_db(model)
        assert idx_mono["parents"] == idx_pkg["parents"]
        assert set(idx_mono["index_d"]) == set(idx_pkg["index_d"])
        for key in idx_mono["index_d"]:
            np.testing.assert_array_equal(
                idx_mono["index_d"][key], idx_pkg["index_d"][key]
            )
        cs_m, cs_p = mono.cs_db(model), base.cs_db(model)
        assert cs_m["parents"] == cs_p["parents"]
        for p in cs_m["index_d"]:
            np.testing.assert_array_equal(cs_m["index_d"][p], cs_p["index_d"][p])


def test_full_solve_bitwise_package_vs_monolith(release_dir, monkeypatch, no_network):
    import crflux.models as pm

    def solve(db_fname, data_dir):
        monkeypatch.setattr(cfg, "mceq_db_fname", db_fname)
        monkeypatch.setattr(cfg, "data_dir", pathlib.Path(data_dir))
        monkeypatch.setattr(cfg, "mceq_db_manifest", MANIFEST)
        monkeypatch.setattr(cfg, "debug_level", 0)
        monkeypatch.setattr(cfg, "enable_em", False)
        run = MCEqRun(
            interaction_model=MODEL,
            theta_deg=0.0,
            primary_model=(pm.HillasGaisser2012, "H3a"),
        )
        run.solve()
        return np.asarray(run.get_solution("numu", mag=0))

    mono = solve(MONOLITH, release_dir)
    pkg = solve(BASE_FILE, release_dir)
    assert np.array_equal(mono, pkg)
    assert pkg.sum() != 0.0


# ---------------------------------------------------------------------------
# missing model, fetch, lock, checksum, grid
# ---------------------------------------------------------------------------


def test_missing_model_error_is_actionable(release_dir, tmp_path, no_network):
    bare = _copy_release(release_dir, tmp_path / "bare", [BASE_FILE, MODEL_FILE])
    fixtures.write_manifest(bare, {BASE_FILE: (True, {MEDIUM: [BASE_MODEL]})})
    backend = _backend(BASE_FILE, bare)
    with pytest.raises(Exception) as exc:
        backend.interaction_db(MODEL)
    message = str(exc.value)
    assert MODEL in message and MANIFEST in message and BASE_MODEL in message
    with pytest.raises(Exception, match=MODEL):
        backend.cs_db(MODEL)


def _fake_download(release_dir, calls):
    """Stand-in for the streaming downloader: serve fixture files by name."""

    def fake(url, outfile, checksum=None):
        calls.append(url)
        name = url.rsplit("/", 1)[-1]
        if (release_dir / name).is_file():
            shutil.copy(release_dir / name, outfile)
        else:
            pathlib.Path(outfile).write_bytes(b"legacy channel payload")

    return fake


def test_listed_package_is_fetched_once_atomically(release_dir, tmp_path, monkeypatch):
    bare = _copy_release(release_dir, tmp_path / "bare", [BASE_FILE, MANIFEST])
    calls = []
    monkeypatch.setattr(download, "_download_file", _fake_download(release_dir, calls))
    backend = _backend(BASE_FILE, bare)
    idx = backend.interaction_db(MODEL)
    assert idx["index_d"]
    assert calls == [fixtures.URL_BASE + MODEL_FILE]
    assert (bare / MODEL_FILE).is_file()
    assert not list(bare.glob("*.lock")) and not list(bare.glob("*.part"))
    backend.cs_db(MODEL)  # cached store, no second fetch
    assert len(calls) == 1
    # a fresh backend finds the file on disk: no network
    monkeypatch.setattr(
        download, "_download_file", lambda *a, **k: pytest.fail("network")
    )
    assert _backend(BASE_FILE, bare).interaction_db(MODEL)["index_d"]


def test_lock_collision_fails_fast_with_cli_hint(release_dir, tmp_path, monkeypatch):
    bare = _copy_release(release_dir, tmp_path / "bare", [BASE_FILE, MANIFEST])
    (bare / f"{MODEL_FILE}.lock").write_bytes(b"")
    monkeypatch.setattr(
        download, "_download_file", lambda *a, **k: pytest.fail("network")
    )
    backend = _backend(BASE_FILE, bare)
    with pytest.raises(Exception) as exc:
        backend.interaction_db(MODEL)
    message = str(exc.value)
    assert "another process is already fetching" in message
    assert "python -m MCEq.data.download" in message and MODEL_FILE in message
    assert not (bare / MODEL_FILE).exists()


def test_checksum_mismatch_installs_nothing(release_dir, tmp_path, monkeypatch):
    bare = _copy_release(release_dir, tmp_path / "bare", [BASE_FILE])
    manifest = yaml.safe_load((release_dir / MANIFEST).read_text())
    manifest["files"][MODEL_FILE]["sha256"] = "0" * 64
    (bare / MANIFEST).write_text(yaml.safe_dump(manifest))
    monkeypatch.setattr(
        download, "_download_file_orig", download._download_file, raising=False
    )

    def wrong_payload(url, outfile, checksum=None):
        # go through the real atomic publisher with a stand-in response
        class Response:
            headers = {"content-length": "4"}

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def raise_for_status(self):
                pass

            def iter_content(self, size):
                yield b"junk"

        monkeypatch.setattr("requests.get", lambda *a, **kw: Response())
        return download._download_file_orig(url, outfile, checksum=checksum)

    monkeypatch.setattr(download, "_download_file", wrong_payload)
    backend = _backend(BASE_FILE, bare)
    with pytest.raises(Exception, match="sha256"):
        backend.interaction_db(MODEL)
    assert not (bare / MODEL_FILE).exists()
    assert not list(bare.glob("*.lock")) and not list(bare.glob("*.part"))


def test_e_grid_mismatch_is_refused(release_dir, tmp_path, no_network):
    bad = _copy_release(
        release_dir, tmp_path / "bad", [BASE_FILE, MODEL_FILE, MANIFEST]
    )
    with h5py.File(bad / MODEL_FILE, "a") as f:
        f["common"].attrs["e_grid"] = np.asarray(f["common"].attrs["e_grid"]) * 1.0001
    backend = _backend(BASE_FILE, bad)
    with pytest.raises(Exception, match="different energy grid"):
        backend.interaction_db(MODEL)


# ---------------------------------------------------------------------------
# ensure_db_available and the CLI
# ---------------------------------------------------------------------------


def test_ensure_db_available_fetches_listed_primary(release_dir, tmp_path, monkeypatch):
    data_dir = _copy_release(release_dir, tmp_path / "data", [MANIFEST])
    calls = []
    monkeypatch.setattr(download, "_download_file", _fake_download(release_dir, calls))
    download.ensure_db_available(_paths(BASE_FILE, data_dir))
    assert calls == [fixtures.URL_BASE + BASE_FILE]
    assert (data_dir / BASE_FILE).is_file()
    # present: no fetch; legacy name: legacy channel untouched by the manifest
    download.ensure_db_available(_paths(BASE_FILE, data_dir))
    assert len(calls) == 1
    download.ensure_db_available(_paths("mceq_db_custom.h5", data_dir))
    assert calls[-1] == download.base_url + download.release_tag + "mceq_db_custom.h5"


def test_manifest_local_override_path(release_dir, tmp_path, no_network):
    """A hand-registered entry with a local ``path`` is used in place."""
    data_dir = _copy_release(release_dir, tmp_path / "data", [BASE_FILE])
    manifest = yaml.safe_load((release_dir / MANIFEST).read_text())
    manifest["files"][MODEL_FILE]["path"] = str(release_dir / MODEL_FILE)
    (data_dir / MANIFEST).write_text(yaml.safe_dump(manifest))
    backend = _backend(BASE_FILE, data_dir)
    assert backend.interaction_db(MODEL)["index_d"]
    assert not (data_dir / MODEL_FILE).exists()


def test_cli_prefetches_model_and_defaults(release_dir, tmp_path, monkeypatch):
    bare = _copy_release(release_dir, tmp_path / "bare", [MANIFEST])
    calls = []
    monkeypatch.setattr(download, "_download_file", _fake_download(release_dir, calls))
    argv = ["--dir", str(bare), "--manifest-name", MANIFEST]
    assert download.cli_main(["--model", MODEL, "--defaults", *argv]) == 0
    assert sorted(calls) == sorted(
        fixtures.URL_BASE + n for n in (BASE_FILE, MODEL_FILE)
    )
    assert (bare / BASE_FILE).is_file() and (bare / MODEL_FILE).is_file()
    with pytest.raises(SystemExit):
        download.cli_main(["--model", "NOSUCHMODEL", *argv])


def test_config_default_matches_download_constant():
    """The data layer cannot import config, so the literal lives twice."""
    assert cfg.mceq_db_manifest == download.DEFAULT_MANIFEST
