from unittest.mock import patch

import MCEq.config as cfg
from MCEq import download

DEFAULT_DB = "mceq_db_lext_dpm193_v140.h5"
CUSTOM_DB = "mceq_db_v140reduced_compact.h5"
OLD_DB = "mceq_db_lext_dpm191.h5"


# ---------------------------------------------------------------------------
# ensure_db_available
# ---------------------------------------------------------------------------


def test_no_download_when_default_db_checksum_ok(tmp_path, monkeypatch):
    (tmp_path / DEFAULT_DB).write_bytes(b"fake")
    monkeypatch.setattr(cfg, "data_dir", tmp_path)
    monkeypatch.setattr(cfg, "mceq_db_fname", DEFAULT_DB)

    with (
        patch("MCEq.download._download_file") as mock_dl,
        patch("MCEq.download.FileIntegrityCheck") as mock_fic,
    ):
        mock_fic.return_value.succeeded.return_value = True
        download.ensure_db_available()

    mock_dl.assert_not_called()
    mock_fic.assert_called_once_with(tmp_path / DEFAULT_DB, download.file_checksum)


def test_downloads_when_default_db_checksum_fails(tmp_path, monkeypatch):
    (tmp_path / DEFAULT_DB).write_bytes(b"corrupt")
    monkeypatch.setattr(cfg, "data_dir", tmp_path)
    monkeypatch.setattr(cfg, "mceq_db_fname", DEFAULT_DB)

    with (
        patch("MCEq.download._download_file") as mock_dl,
        patch("MCEq.download.FileIntegrityCheck") as mock_fic,
    ):
        mock_fic.return_value.succeeded.return_value = False
        download.ensure_db_available()

    mock_dl.assert_called_once_with(
        download.base_url + download.release_tag + DEFAULT_DB,
        tmp_path / DEFAULT_DB,
    )


def test_downloads_when_default_db_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "data_dir", tmp_path)
    monkeypatch.setattr(cfg, "mceq_db_fname", DEFAULT_DB)

    with patch("MCEq.download._download_file") as mock_dl:
        download.ensure_db_available()

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
        patch("MCEq.download._download_file") as mock_dl,
        patch("MCEq.download.FileIntegrityCheck") as mock_fic,
    ):
        download.ensure_db_available()

    mock_dl.assert_not_called()
    mock_fic.assert_not_called()


def test_downloads_missing_custom_db(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "data_dir", tmp_path)
    monkeypatch.setattr(cfg, "mceq_db_fname", CUSTOM_DB)

    with patch("MCEq.download._download_file") as mock_dl:
        download.ensure_db_available()

    mock_dl.assert_called_once_with(
        download.base_url + download.release_tag + CUSTOM_DB,
        tmp_path / CUSTOM_DB,
    )


def test_removes_old_db(tmp_path, monkeypatch):
    (tmp_path / CUSTOM_DB).write_bytes(b"fake")
    old = tmp_path / OLD_DB
    old.write_bytes(b"old")
    monkeypatch.setattr(cfg, "data_dir", tmp_path)
    monkeypatch.setattr(cfg, "mceq_db_fname", CUSTOM_DB)

    with patch("MCEq.download._download_file"):
        download.ensure_db_available()

    assert not old.exists()


def test_no_error_when_old_db_absent(tmp_path, monkeypatch):
    (tmp_path / CUSTOM_DB).write_bytes(b"fake")
    monkeypatch.setattr(cfg, "data_dir", tmp_path)
    monkeypatch.setattr(cfg, "mceq_db_fname", CUSTOM_DB)

    with patch("MCEq.download._download_file"):
        download.ensure_db_available()  # must not raise
