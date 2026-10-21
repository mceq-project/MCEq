"""The CI databases the test suite runs on, pinned by sha256.

This file is the one place that says which data the tests expect. The suite
checks the files in ``src/MCEq/data`` against it at session start (see
``tests/conftest.py``) and replaces any that differ, so a branch never runs on
a stale copy; the workflows fetch through it too, and their cache key is this
file's hash. The 2D baseline solution is pinned in ``baseline_asset.py``.

When the CI data is re-cut, upload the new files under digest-stamped names
(``<stem>_<sha256[:12]>.<ext>``) to the ``ci-assets`` release and update
:data:`PINS` (``python tests/data/ci_assets.py --print-digests`` prints them).

    python tests/data/ci_assets.py            # fetch what is missing or stale
    python tests/data/ci_assets.py --check    # report only; exit 1 if stale
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import sys
import tempfile
import urllib.request

RELEASE_URL = "https://github.com/mceq-project/MCEq/releases/download/ci-assets/"

#: File name in ``src/MCEq/data`` -> sha256.
PINS = {
    "mceq_db_manifest_ci_v2.yaml": (
        "21865c8cff3c6b82fad89c04e6fc91cb010924273d69d46087fb285e23036bd9"
    ),
    "mceq_ci_base_air_1d_v2.h5": (
        "f20f913a20e6c9b25ad438c94d94cdb45f81bf9bb44092064ee47d0fa46caf63"
    ),
    "mceq_ci_extra_models_air_1d_v2.h5": (
        "363d7c85edd96a98e187251f7c067ea6625939c2a92c7d3c608bee4e9c933e43"
    ),
    "mceq_ci_base_air_2d_legendre_v2.h5": (
        "22a6614270807eaa33b2d6d448030d2eb55e87efe2c7bf0bc7f08b63125a3838"
    ),
}

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "src" / "MCEq" / "data"


def url(name):
    """Release URL of ``name``: published under a digest-stamped name, so an
    older commit keeps finding its own files."""
    stem, ext = name.rsplit(".", 1)
    return f"{RELEASE_URL}{stem}_{PINS[name][:12]}.{ext}"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stale(data_dir=DATA_DIR):
    """``{name: reason}`` for the pinned files that are missing or differ."""
    found = {}
    for name, pin in PINS.items():
        path = pathlib.Path(data_dir) / name
        if not path.is_file():
            found[name] = "missing"
        elif (got := digest(path)) != pin:
            found[name] = f"sha256 {got[:12]}, pinned {pin[:12]}"
    return found


def fetch(name, data_dir=DATA_DIR, timeout=300):
    """Download ``name``, verify it and put it in place atomically.

    A symlink at the destination (a worktree sharing another checkout's data)
    is replaced by the file, so the shared copy is never overwritten.
    """
    dest = pathlib.Path(data_dir) / name
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=name + ".", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out:
            with urllib.request.urlopen(url(name), timeout=timeout) as response:
                while block := response.read(1 << 20):
                    out.write(block)
        if digest(tmp) != PINS[name]:
            raise OSError(f"{name}: the download does not match its pin")
        os.replace(tmp, dest)
    finally:
        pathlib.Path(tmp).unlink(missing_ok=True)


def ensure(data_dir=DATA_DIR):
    """Fetch every missing or stale file; return ``{name: reason}`` for those
    that could not be fetched (no network, for example)."""
    failed = {}
    for name, reason in stale(data_dir).items():
        print(f"CI data {name}: {reason}; fetching {url(name)}", file=sys.stderr)
        try:
            fetch(name, data_dir)
        except Exception as err:  # noqa: BLE001 -- reported to the caller
            failed[name] = f"{reason}; fetch failed: {err}"
    return failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="report only")
    parser.add_argument(
        "--print-digests",
        action="store_true",
        help="print the sha256 of the files in the data directory",
    )
    parser.add_argument("--dir", type=pathlib.Path, default=DATA_DIR)
    args = parser.parse_args(argv)
    if args.print_digests:
        for name in PINS:
            path = args.dir / name
            print(f'"{name}": "{digest(path) if path.is_file() else "missing"}",')
        return 0
    bad = stale(args.dir) if args.check else ensure(args.dir)
    for name in PINS:
        print(f"{'STALE' if name in bad else 'OK   '} {name} {bad.get(name, '')}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
