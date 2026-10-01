"""Fetch the stored 2D solution the baseline tests compare against.

The file is 13 MB, so it is not kept in the repository. It lives with the
other test data on the ``ci-assets`` release and is downloaded once into
``tests/data/``, where it stays. Regenerate it with
``tests/data/make_2d_baseline_fixture.py`` and upload the new file when the
2D transport changes on purpose.
"""

from __future__ import annotations

import hashlib
import pathlib
import urllib.request

NAME = "2d_baseline_solution.npz"
SHA256 = "21feb7f019f36ceac49ef52e5f34f1a4034ad90af4f0715f9e444e48919fc7d7"
#: Published under a digest-stamped name, so an older commit keeps its file.
URL = (
    "https://github.com/mceq-project/MCEq/releases/download/ci-assets/"
    f"2d_baseline_solution_{SHA256[:12]}.npz"
)

PATH = pathlib.Path(__file__).parent / NAME


def _digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ensure(timeout=120):
    """The local path of the stored solution, downloading it if needed.

    Returns ``None`` when it is neither present nor reachable, so the tests
    that use it can skip rather than fail on a machine without a network.
    """
    if PATH.is_file() and _digest(PATH) == SHA256:
        return PATH
    tmp = PATH.with_suffix(".npz.part")
    try:
        with urllib.request.urlopen(URL, timeout=timeout) as response:
            tmp.write_bytes(response.read())
    except Exception:
        tmp.unlink(missing_ok=True)
        return None
    if _digest(tmp) != SHA256:
        tmp.unlink(missing_ok=True)
        return None
    tmp.replace(PATH)
    return PATH
