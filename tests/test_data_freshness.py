"""A data file that is not the one its manifest or pin lists is refused."""

import importlib.util
import pathlib
import shutil

import h5py
import pytest
import yaml

from MCEq import config
from MCEq.data import download

CI_DB = "mceq_ci_base_air_1d_v2.h5"
CI_MANIFEST = "mceq_db_manifest_ci_v2.yaml"

_spec = importlib.util.spec_from_file_location(
    "ci_assets", pathlib.Path(__file__).parent / "data" / "ci_assets.py"
)
ci_assets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci_assets)


class _Cfg:
    """The attributes download.* reads, on a scratch data directory."""

    def __init__(self, data_dir, verify=True):
        self.data_dir = pathlib.Path(data_dir)
        self.mceq_db_manifest = CI_MANIFEST
        self.mceq_db_fname = CI_DB
        self.debug_level = 0
        self.verify_data_packages = verify


@pytest.fixture
def scratch(tmp_path):
    for name in (CI_DB, CI_MANIFEST):
        shutil.copy(config.data_dir / name, tmp_path / name)
    return tmp_path


def _entry(data_dir):
    return download.load_manifest(data_dir / CI_MANIFEST)["files"][CI_DB]


def _restamp(path, split_from):
    with h5py.File(path, "a") as handle:
        handle.attrs["split_from_sha256"] = split_from


def test_the_listed_file_passes(scratch):
    download.check_package(scratch / CI_DB, _entry(scratch))
    assert download.ensure_package(_Cfg(scratch), CI_DB) == scratch / CI_DB


def test_a_copy_cut_from_another_database_is_refused(scratch):
    entry = _entry(scratch)
    _restamp(scratch / CI_DB, "36eb21e6" + "0" * 56)
    entry = dict(entry, size=(scratch / CI_DB).stat().st_size)  # size alone agrees
    with pytest.raises(download.StaleDataError, match="cut from a database"):
        download.check_package(scratch / CI_DB, entry)


def test_a_copy_of_another_size_is_refused_on_resolution(scratch):
    with open(scratch / CI_DB, "ab") as handle:
        handle.write(b"\0" * 16)
    with pytest.raises(download.StaleDataError, match="bytes, listed"):
        download.ensure_package(_Cfg(scratch), CI_DB)


def test_the_check_can_be_switched_off(scratch):
    with open(scratch / CI_DB, "ab") as handle:
        handle.write(b"\0" * 16)
    assert download.ensure_package(_Cfg(scratch, verify=False), CI_DB).is_file()


def test_verify_compares_the_full_digest(scratch, capsys):
    assert (
        download.cli_main(
            ["--verify", "--dir", str(scratch), "--manifest-name", CI_MANIFEST]
        )
        == 0
    )
    with open(scratch / CI_DB, "r+b") as handle:  # same size, other bytes
        handle.seek(-1, 2)
        last = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([last[0] ^ 1]))
    assert (
        download.cli_main(
            ["--verify", "--dir", str(scratch), "--manifest-name", CI_MANIFEST]
        )
        == 1
    )
    assert "STALE" in capsys.readouterr().out


def test_an_older_manifest_in_a_separate_data_dir_warns(tmp_path):
    shipped = download.packaged_manifest(CI_MANIFEST)
    manifest = yaml.safe_load(shipped.read_text())
    manifest["created"] = "2000-01-01T00:00:00"
    (tmp_path / CI_MANIFEST).write_text(yaml.safe_dump(manifest))
    with pytest.warns(download.StaleManifestWarning, match="older than"):
        download.local_manifest(_Cfg(tmp_path))


def test_ci_assets_names_the_stale_and_missing_pins(tmp_path):
    shutil.copy(config.data_dir / CI_DB, tmp_path / CI_DB)
    with open(tmp_path / CI_DB, "ab") as handle:
        handle.write(b"\0")
    found = ci_assets.stale(tmp_path)
    assert found[CI_DB].startswith("sha256")
    assert found[CI_MANIFEST] == "missing"
    assert ci_assets.url(CI_DB).endswith(
        f"mceq_ci_base_air_1d_v2_{ci_assets.PINS[CI_DB][:12]}.h5"
    )


def test_the_session_runs_on_the_pinned_files():
    assert ci_assets.stale() == {}
