"""Fetching and verifying the MCEq database files.

`MCEqRun.__init__` calls :func:`ensure_db_available` so the download happens
when a database is actually needed, which lets a caller point
``config.mceq_db_fname`` somewhere else first.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

# Download database file from github
base_url = "https://github.com/afedynitch/MCEq/releases/download/"
release_tag = "builds_on_azure/"
# sha256 checksum of the default database file (config.mceq_db_fname),
# https://github.com/afedynitch/MCEq/releases/download/builds_on_azure/mceq_db_lext_dpm193_v142.h5
file_checksum = "247f40203436c69431db4d393316940636badb2c231f68d51870fb49bf476946"


class FileIntegrityCheck:
    """
    A class to check a file integrity against provided checksum

    Attributes
    ----------
    filename : str
        path to the file
    checksum : str
        hex of sha256 checksum
    Methods
    -------
    succeeded():
        returns True if checksum and calculated checksum of the file are equal

    get_file_checksum():
        returns checksum of the file
    """

    def __init__(self, filename, checksum=""):
        self.filename = filename
        self.checksum = checksum
        self.sha256_hash = hashlib.sha256()
        self.hash_is_calculated = False

    def _calculate_hash(self):
        if not self.hash_is_calculated:
            try:
                with open(self.filename, "rb") as file:
                    for byte_block in iter(lambda: file.read(4096), b""):
                        self.sha256_hash.update(byte_block)
                self.hash_is_calculated = True
            except OSError as ex:
                print(f"FileIntegrityCheck: {ex}")

    def succeeded(self):
        self._calculate_hash()
        return self.hash_is_calculated and self.sha256_hash.hexdigest() == self.checksum

    def get_file_checksum(self):
        self._calculate_hash()
        return self.sha256_hash.hexdigest()


def _download_file(url, outfile):
    """Publish a complete download atomically so concurrent readers stay safe."""
    import math

    import requests
    from tqdm import tqdm

    outfile = Path(outfile)
    temporary = None
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024 * 1024
            wrote = 0
            with tempfile.NamedTemporaryFile(
                dir=outfile.parent,
                prefix=outfile.name + ".",
                suffix=".part",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                for data in tqdm(
                    response.iter_content(block_size),
                    total=math.ceil(total_size / block_size),
                    unit="MB",
                    unit_scale=True,
                ):
                    wrote += len(data)
                    stream.write(data)
            if total_size and wrote != total_size:
                raise OSError("Incomplete MCEq database download")
        digest = FileIntegrityCheck(temporary).get_file_checksum()
        if outfile.name == "mceq_db_lext_dpm193_v142.h5" and digest != file_checksum:
            raise OSError("MCEq database download checksum mismatch")
        try:
            os.replace(temporary, outfile)
        except PermissionError:
            # Windows may refuse replacement while another reader has the
            # concurrently published identical file open. It is already ready.
            if not FileIntegrityCheck(outfile, digest).succeeded():
                raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ensure_db_available(cfg):
    """Download the MCEq database if not already present.

    Called by ``MCEqRun.__init__`` with the run's configuration — the live
    module or a run snapshot — so the download is deferred until the database
    is actually needed and this module imports no configuration itself.
    ``cfg`` must expose flat ``data_dir``, ``mceq_db_fname`` and
    ``debug_level`` attributes; a paths-only group does not supply that
    interface. Direct callers pass ``MCEq.config`` (or the module they
    override):

        from MCEq import config
        config.mceq_db_fname = "my_db.h5"
        ensure_db_available(config)

    The integrity check only applies to the default database; non-default
    files are accepted as-is if they exist.
    """
    data_dir = cfg.data_dir
    mceq_db_fname = cfg.mceq_db_fname
    debug_level = cfg.debug_level

    _url = base_url + release_tag + mceq_db_fname
    filepath = data_dir / mceq_db_fname
    if filepath.exists():
        is_complete = (
            FileIntegrityCheck(filepath, file_checksum).succeeded()
            if mceq_db_fname == "mceq_db_lext_dpm193_v142.h5"
            else True
        )
    else:
        is_complete = False

    if not is_complete:
        print(f"Downloading MCEq database file {mceq_db_fname}.")
        if debug_level >= 2:
            print(_url)
        _download_file(_url, filepath)

    for old_name in ("mceq_db_lext_dpm193_v140.h5", "mceq_db_lext_dpm191.h5"):
        old_db = data_dir / old_name
        if old_db.exists():
            print(f"Removing previous database {old_db.name}.")
            os.unlink(old_db)
