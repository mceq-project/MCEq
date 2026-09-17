"""Fetching and verifying the MCEq database files.

`MCEqRun.__init__` calls :func:`ensure_db_available` so the download happens
when a database is actually needed, which lets a caller point
``config.mceq_db_fname`` somewhere else first.

Two channels: the legacy monolith release (``base_url`` + ``release_tag``,
one file, checksum pinned here for the default name) and the v2 data release
described by a YAML manifest (:mod:`MCEq.data.db_manifest`) -- packages are
fetched by name with the sha256 the manifest records. Package fetches are
race-safe for cluster jobs that start together: a ``<file>.lock`` sentinel
taken with ``O_CREAT | O_EXCL`` lets exactly one process download while the
others fail fast with the pre-fetch CLI (``python -m MCEq.data.download``)
in the message, and :func:`_download_file` publishes only a complete,
checksum-verified file via an atomic rename.
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


def _download_file(url, outfile, checksum=None):
    """Publish a complete download atomically so concurrent readers stay safe.

    ``checksum`` (hex sha256) is verified before the rename when given; the
    legacy default database is verified against ``file_checksum`` by name.
    """
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
        if checksum is not None and digest != checksum.lower():
            raise OSError(
                f"MCEq data package {outfile.name} failed its sha256 check "
                f"(expected {checksum}, got {digest}); nothing was installed."
            )
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
    files are accepted as-is if they exist. A missing file that the local
    data manifest lists (a v2 release package) is fetched from the data
    release with the manifest's sha256; every other missing name goes to the
    legacy monolith release.
    """
    data_dir = cfg.data_dir
    mceq_db_fname = cfg.mceq_db_fname
    debug_level = cfg.debug_level

    _url = base_url + release_tag + mceq_db_fname
    filepath = data_dir / mceq_db_fname
    if not filepath.exists():
        manifest = local_manifest(cfg)
        if manifest is not None and mceq_db_fname in manifest["files"]:
            ensure_package(cfg, mceq_db_fname, manifest)
            _remove_old_dbs(data_dir)
            return
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

    _remove_old_dbs(data_dir)


def _remove_old_dbs(data_dir):
    for old_name in ("mceq_db_lext_dpm193_v140.h5", "mceq_db_lext_dpm191.h5"):
        old_db = data_dir / old_name
        if old_db.exists():
            print(f"Removing previous database {old_db.name}.")
            os.unlink(old_db)


# ---------------------------------------------------------------------------
# v2 data release: manifest-described packages
# ---------------------------------------------------------------------------

#: Where a manifest missing from ``data_dir`` is fetched from. The manifest's
#: own ``url_base`` governs the package URLs, so this only bootstraps a
#: custom data directory on a fresh machine; a copy of the manifest ships
#: inside the package (``MCEq/data``) and is used when present.
data_release_url = "https://github.com/mceq-project/MCEq/releases/download/v2.0.0/"
#: Manifest file name of the current data release; ``config.mceq_db_manifest``
#: defaults to the same literal (the data layer does not import config).
DEFAULT_MANIFEST = "mceq_db_manifest_v2.yaml"


class PackageFetchError(Exception):
    """A package fetch was attempted and could not be completed safely.

    Concurrent-download collision (another process holds ``<file>.lock``),
    transport failure, a name the manifest does not list, or a sha256
    mismatch. The message says how to pre-fetch with the CLI.
    """


def manifest_path(cfg):
    """``data_dir/<mceq_db_manifest>``: the manifest a run reads and edits."""
    return Path(cfg.data_dir) / cfg.mceq_db_manifest


def packaged_manifest(name):
    """The copy shipped inside the package, or None."""
    here = Path(__file__).resolve().parent / name
    return here if here.is_file() else None


def local_manifest(cfg, preferred=None):
    """Parsed manifest without network, or None.

    Search order: ``preferred`` in ``data_dir`` (the manifest a package
    declares in its root attribute ``mceq_db_manifest`` -- a CI or custom
    package set names its own), then ``cfg.mceq_db_manifest`` in
    ``data_dir``, then the copy packaged with MCEq.
    """
    from MCEq.data import db_manifest

    candidates = []
    if preferred:
        candidates.append(Path(cfg.data_dir) / preferred)
    candidates += [manifest_path(cfg), packaged_manifest(cfg.mceq_db_manifest)]
    for path in candidates:
        if path is not None and path.is_file():
            return db_manifest.load(path)
    return None


def ensure_manifest(cfg, preferred=None):
    """The manifest (see :func:`local_manifest`), fetched into ``data_dir``
    from the data release when no local copy exists."""
    from MCEq.data import db_manifest

    manifest = local_manifest(cfg, preferred)
    if manifest is not None:
        return manifest
    dest = manifest_path(cfg)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _fetch_asset(
        dest,
        data_release_url + cfg.mceq_db_manifest,
        checksum=None,
        hint="--manifest",
        debug_level=getattr(cfg, "debug_level", 1),
    )
    return db_manifest.load(dest)


def _acquire_lock(dest, hint):
    """Take the ``O_CREAT | O_EXCL`` sentinel ``<dest>.lock`` for one fetch.

    At most one process creates it, so N cluster jobs missing the same file
    never download it N times or read a half-written copy: the others fail
    with the pre-fetch instructions. A crashed fetch leaves the sentinel
    behind; the message says to delete it.
    """
    lock = dest.parent / (dest.name + ".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise PackageFetchError(
            f"{dest.name} is missing and another process is already fetching "
            f"it (lock file {lock} exists). Wait for that fetch to finish and "
            "rerun, or fetch once before submitting jobs:\n"
            f"    python -m MCEq.data.download {hint}\n"
            "    python -m MCEq.data.download --all\n"
            "If no fetch is in progress, delete the stale lock file and retry."
        ) from None
    os.close(fd)
    return lock


def _fetch_asset(dest, url, checksum, hint, debug_level=1):
    """Lock, then :func:`_download_file` (temp file, sha256, atomic rename)."""
    dest = Path(dest)
    lock = _acquire_lock(dest, hint)
    try:
        if debug_level >= 2:
            print(url)
        try:
            _download_file(url, dest, checksum=checksum)
        except OSError as ex:
            raise PackageFetchError(f"Fetching {url} failed: {ex}") from ex
    finally:
        lock.unlink(missing_ok=True)
    return dest


def ensure_package(cfg, name, manifest=None):
    """Path of package ``name`` in ``data_dir``, fetching it once if missing.

    A manifest entry with a top-level ``path`` (a file the user registered
    by hand) is used in place when it exists. Otherwise the URL is the
    entry's ``url`` or ``url_base + name``, verified against ``sha256``.
    """
    dest = Path(cfg.data_dir) / name
    if dest.exists():
        return dest
    manifest = manifest if manifest is not None else ensure_manifest(cfg)
    entry = manifest["files"].get(name)
    if entry is None:
        raise PackageFetchError(
            f"{name} is not listed in the data manifest {cfg.mceq_db_manifest}."
        )
    if entry.get("path") and Path(entry["path"]).is_file():
        return Path(entry["path"])
    url = entry.get("url") or (manifest.get("url_base") or data_release_url) + name
    size = entry.get("size")
    print(
        f"Downloading MCEq data package {name}"
        + (f" ({size / 1e6:.0f} MB)." if size else ".")
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    return _fetch_asset(
        dest,
        url,
        checksum=entry.get("sha256"),
        hint=f"--file {name}",
        debug_level=getattr(cfg, "debug_level", 1),
    )


def cli_main(argv=None):
    """``python -m MCEq.data.download``: pre-fetch data packages.

    Run once on the submission node so cluster jobs never race on a first
    fetch. ``--model`` resolves through the manifest on the grid of the
    default 1D (or, with ``--dim 2d``, 2D) package; ``--file`` names a
    package directly; ``--all`` fetches every package of the release.
    """
    import argparse

    from MCEq.data import db_manifest

    parser = argparse.ArgumentParser(
        prog="python -m MCEq.data.download",
        description="Fetch MCEq data packages listed in the release manifest.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="MODEL",
        help="hadronic model to make available (repeatable)",
    )
    parser.add_argument(
        "--medium", default="air", help="medium for --model (default: air)"
    )
    parser.add_argument(
        "--dim",
        choices=("1d", "2d"),
        default="1d",
        help="grid family for --model: that of the default 1d/2d package",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        metavar="NAME",
        help="package file to fetch (repeatable)",
    )
    parser.add_argument(
        "--defaults", action="store_true", help="fetch the default 1D and 2D packages"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="fetch_all",
        help="fetch every package in the manifest",
    )
    parser.add_argument(
        "--manifest",
        action="store_true",
        help="only make sure the manifest itself is present",
    )
    parser.add_argument(
        "--dir",
        default=None,
        help="data directory (default: the package data directory, "
        "i.e. config.data_dir out of the box)",
    )
    parser.add_argument(
        "--manifest-name",
        default=DEFAULT_MANIFEST,
        help=f"manifest file name (default: {DEFAULT_MANIFEST})",
    )
    args = parser.parse_args(argv)

    class _Paths:
        data_dir = Path(args.dir) if args.dir else Path(__file__).resolve().parent
        mceq_db_manifest = args.manifest_name
        mceq_db_fname = None
        debug_level = 1

    cfg = _Paths()
    manifest = ensure_manifest(cfg)
    print(
        f"Manifest: {manifest_path(cfg) if manifest_path(cfg).is_file() else packaged_manifest(cfg.mceq_db_manifest)}"
    )
    wanted = list(args.file)
    if args.fetch_all:
        wanted += list(manifest["files"])
    if args.defaults:
        wanted += list(manifest.get("defaults", {}).values())
    if args.model:
        default = manifest.get("defaults", {}).get(args.dim)
        if default is None:
            parser.error(f"manifest names no default {args.dim} package")
        grid = manifest["grids"][manifest["files"][default]["grid"]]
        for model in args.model:
            names = db_manifest.files_for(manifest, grid, model, args.medium)
            if not names:
                parser.error(
                    f"no {args.dim} package provides {model} in {args.medium}; "
                    f"on this grid: {db_manifest.models_on_grid(manifest, grid)}"
                )
            wanted += names
    if not wanted and not args.manifest:
        parser.error("give --model, --file, --defaults, --all or --manifest")
    for name in dict.fromkeys(wanted):
        print(f"Ready: {ensure_package(cfg, name, manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
