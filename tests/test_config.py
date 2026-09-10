from unittest.mock import patch

import pytest

import MCEq.config as cfg
from MCEq.data import download

DEFAULT_DB = "mceq_db_lext_dpm193_v142.h5"
CUSTOM_DB = "mceq_db_v140reduced_compact.h5"
OLD_DBS = ("mceq_db_lext_dpm193_v140.h5", "mceq_db_lext_dpm191.h5")


# ---------------------------------------------------------------------------
# ensure_db_available
# ---------------------------------------------------------------------------


def test_no_download_when_default_db_checksum_ok(tmp_path, monkeypatch):
    (tmp_path / DEFAULT_DB).write_bytes(b"fake")
    monkeypatch.setattr(cfg, "data_dir", tmp_path)
    monkeypatch.setattr(cfg, "mceq_db_fname", DEFAULT_DB)

    with (
        patch("MCEq.data.download._download_file") as mock_dl,
        patch("MCEq.data.download.FileIntegrityCheck") as mock_fic,
    ):
        mock_fic.return_value.succeeded.return_value = True
        download.ensure_db_available(cfg)

    mock_dl.assert_not_called()
    mock_fic.assert_called_once_with(tmp_path / DEFAULT_DB, download.file_checksum)


def test_downloads_when_default_db_checksum_fails(tmp_path, monkeypatch):
    (tmp_path / DEFAULT_DB).write_bytes(b"corrupt")
    monkeypatch.setattr(cfg, "data_dir", tmp_path)
    monkeypatch.setattr(cfg, "mceq_db_fname", DEFAULT_DB)

    with (
        patch("MCEq.data.download._download_file") as mock_dl,
        patch("MCEq.data.download.FileIntegrityCheck") as mock_fic,
    ):
        mock_fic.return_value.succeeded.return_value = False
        download.ensure_db_available(cfg)

    mock_dl.assert_called_once_with(
        download.base_url + download.release_tag + DEFAULT_DB,
        tmp_path / DEFAULT_DB,
    )


def test_downloads_when_default_db_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "data_dir", tmp_path)
    monkeypatch.setattr(cfg, "mceq_db_fname", DEFAULT_DB)

    with patch("MCEq.data.download._download_file") as mock_dl:
        download.ensure_db_available(cfg)

    mock_dl.assert_called_once_with(
        download.base_url + download.release_tag + DEFAULT_DB,
        tmp_path / DEFAULT_DB,
    )


def test_no_download_for_existing_custom_db(tmp_path, monkeypatch):
    """Non-default DB that exists must not trigger a download or checksum check."""
    (tmp_path / CUSTOM_DB).write_bytes(b"fake")
    monkeypatch.setattr(cfg, "data_dir", tmp_path)
    monkeypatch.setattr(cfg, "mceq_db_fname", CUSTOM_DB)

    with (
        patch("MCEq.data.download._download_file") as mock_dl,
        patch("MCEq.data.download.FileIntegrityCheck") as mock_fic,
    ):
        download.ensure_db_available(cfg)

    mock_dl.assert_not_called()
    mock_fic.assert_not_called()


def test_downloads_missing_custom_db(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "data_dir", tmp_path)
    monkeypatch.setattr(cfg, "mceq_db_fname", CUSTOM_DB)

    with patch("MCEq.data.download._download_file") as mock_dl:
        download.ensure_db_available(cfg)

    mock_dl.assert_called_once_with(
        download.base_url + download.release_tag + CUSTOM_DB,
        tmp_path / CUSTOM_DB,
    )


@pytest.mark.parametrize("old_name", OLD_DBS)
def test_removes_old_db(tmp_path, monkeypatch, old_name):
    (tmp_path / CUSTOM_DB).write_bytes(b"fake")
    old = tmp_path / old_name
    old.write_bytes(b"old")
    monkeypatch.setattr(cfg, "data_dir", tmp_path)
    monkeypatch.setattr(cfg, "mceq_db_fname", CUSTOM_DB)

    with patch("MCEq.data.download._download_file"):
        download.ensure_db_available(cfg)

    assert not old.exists()


def test_no_error_when_old_db_absent(tmp_path, monkeypatch):
    (tmp_path / CUSTOM_DB).write_bytes(b"fake")
    monkeypatch.setattr(cfg, "data_dir", tmp_path)
    monkeypatch.setattr(cfg, "mceq_db_fname", CUSTOM_DB)

    with patch("MCEq.data.download._download_file"):
        download.ensure_db_available(cfg)  # must not raise


# ---------------------------------------------------------------------------
# MKL library load
# ---------------------------------------------------------------------------


class PreFspathCDLL:
    """``ctypes.CDLL.__init__`` as CPython < 3.12 has it, on Windows.

    The ``os.fspath`` call landed in 3.12; before it, the DLL-search-path
    branch subscripts the argument directly, so anything but a string raises
    ``TypeError: argument of type 'WindowsPath' is not iterable``. Standing in
    for the real loader lets a Linux run reject a ``Path`` the way Windows on
    3.10/3.11 does.
    """

    def __init__(self, name, *args, **kwargs):
        assert isinstance(name, str), (
            f"cdll.LoadLibrary got a {type(name).__name__}; ctypes calls "
            "os.fspath on it only from CPython 3.12"
        )
        self.loaded = "/" in name or "\\" in name  # TypeError on a PathLike


def test_load_mkl_passes_a_string_path(monkeypatch):
    """``mkl_runtime.load`` hands ``LoadLibrary`` a ``str``.

    ``detect.mkl_library_path`` returns a ``Path``, which the Windows loader
    on Python 3.10/3.11 cannot take. Retargeted from ``config._load_mkl``
    at M4: the memo and the load moved to :mod:`MCEq.mkl_runtime`; the
    config entry point is a thin delegate and ``config.mkl`` a read-only
    view of the runtime handle.
    """
    import ctypes

    from MCEq import mkl_runtime

    monkeypatch.setattr(mkl_runtime, "_LIB", None)
    monkeypatch.setattr(cfg.detect, "has_mkl", lambda: True)
    monkeypatch.setattr(ctypes.cdll, "_dlltype", PreFspathCDLL)

    cfg._load_mkl()

    assert isinstance(cfg.mkl, PreFspathCDLL)


@pytest.mark.parametrize("failure", [None, "truncated", "stream"])
def test_download_publishes_only_complete_file(tmp_path, monkeypatch, failure):
    """Readers keep the old database throughout a download, including failures."""
    outfile = tmp_path / "custom.h5"
    outfile.write_bytes(b"old database")

    class Response:
        headers = {"content-length": "6"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, size):
            yield b"new"
            assert outfile.read_bytes() == b"old database"
            if failure == "stream":
                raise OSError("connection lost")
            if failure != "truncated":
                yield b" db"
            assert outfile.read_bytes() == b"old database"

    monkeypatch.setattr("requests.get", lambda *a, **kw: Response())
    if failure:
        with pytest.raises(OSError):
            download._download_file("https://example.invalid/database", outfile)
        assert outfile.read_bytes() == b"old database"
    else:
        download._download_file("https://example.invalid/database", outfile)
        assert outfile.read_bytes() == b"new db"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["custom.h5"]
