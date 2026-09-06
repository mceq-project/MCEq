"""Fetching and verifying the MCEq database files.

`MCEqRun.__init__` calls :func:`ensure_db_available` so the download happens
when a database is actually needed, which lets a caller point
``config.mceq_db_fname`` somewhere else first.
"""

from __future__ import annotations

import hashlib
import os

# Download database file from github
base_url = "https://github.com/afedynitch/MCEq/releases/download/"
release_tag = "builds_on_azure/"
# sha256 checksum of the default database file
# https://github.com/afedynitch/MCEq/releases/download/builds_on_azure/mceq_db_lext_dpm191_v12.h5
file_checksum = "5da415e9bcf81926b1061d5792d75cb3aceb9de173beccb4695fd3909a0bfdd0"


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
    """Downloads the MCEq database from github"""

    import math

    import requests
    from tqdm import tqdm

    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(r.headers.get("content-length", 0))
    block_size = 1024 * 1024
    wrote = 0
    with open(outfile, "wb") as f:
        for data in tqdm(
            r.iter_content(block_size),
            total=math.ceil(total_size // block_size),
            unit="MB",
            unit_scale=True,
        ):
            wrote = wrote + len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size:
        raise Exception("ERROR, something went wrong")


def ensure_db_available():
    """Download the MCEq database if not already present.

    Called by MCEqRun.__init__ so that the download is deferred until the
    database is actually needed.  This allows tests (and other callers) to
    override ``config.mceq_db_fname`` before a download is attempted.

    The integrity check only applies to the default database; non-default
    files are accepted as-is if they exist.
    """
    from MCEq import config

    data_dir = config.data_dir
    mceq_db_fname = config.mceq_db_fname
    debug_level = config.debug_level

    _url = base_url + release_tag + mceq_db_fname
    filepath = data_dir / mceq_db_fname
    if filepath.exists():
        is_complete = (
            FileIntegrityCheck(filepath, file_checksum).succeeded()
            if mceq_db_fname == "mceq_db_lext_dpm193_v140.h5"
            else True
        )
    else:
        is_complete = False

    if not is_complete:
        print(f"Downloading MCEq database file {mceq_db_fname}.")
        if debug_level >= 2:
            print(_url)
        _download_file(_url, filepath)

    old_db = data_dir / "mceq_db_lext_dpm191.h5"
    if old_db.exists():
        print(f"Removing previous database {old_db.name}.")
        os.unlink(old_db)
