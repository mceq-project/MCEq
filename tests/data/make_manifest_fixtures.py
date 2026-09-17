"""Build a release-package fixture (monolith + base + model file + manifest).

Source: the CI extra-models package ``mceq_ci_extra_models_air_1d_v2.h5`` (a
spine package: grid, decays, losses and thirteen 1D models on the 31-bin CI
window). It is cut the way the release packager (mceq-maintenance-tools
step 6) cuts the production databases:

* ``mceq_monolith_test.h5`` -- the source without its ``mceq_db_*`` root
  attributes, i.e. a plain monolithic database the loader resolves against
  itself only;
* ``mceq_base_test.h5`` -- spine + QGSJETII04;
* ``mceq_model_SIBYLL21_air_test.h5`` -- SIBYLL21 with a ``common`` mirror;
* ``mceq_db_manifest_test.yaml`` -- schema-1 manifest listing both on one grid.

Dataset values, dtypes, shapes and attributes are copied unchanged, so the
loader tests pin bitwise equality against the monolith without a binary
fixture in the repository. Imported by path from
``tests/test_data_db_manifest.py`` (``tests/data`` is not a package).
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import h5py
import yaml

from MCEq.data import db_manifest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
REFERENCE_DB = REPO_ROOT / "src" / "MCEq" / "data" / "mceq_ci_extra_models_air_1d_v2.h5"
BASE_MODEL = "QGSJETII04"
MODEL = "SIBYLL21"
MEDIUM = "air"
MONOLITH_FILE = "mceq_monolith_test.h5"
BASE_FILE = "mceq_base_test.h5"
MODEL_FILE = f"mceq_model_{MODEL}_{MEDIUM}_test.h5"
MANIFEST = "mceq_db_manifest_test.yaml"
URL_BASE = "https://example.invalid/mceq-test-release/"
GRID_ID = "1d-test"
PACKAGE_ATTRS = (
    "mceq_db_package",
    "mceq_db_manifest",
    "mceq_db_spine",
    "mceq_db_models",
    "split_from",
    "split_from_sha256",
    "packaged_on",
)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_package(src, dest, models, spine):
    with h5py.File(dest, "w") as out:
        for k, v in src.attrs.items():
            if k not in PACKAGE_ATTRS:
                out.attrs[k] = v
        out.attrs["mceq_db_package"] = dest.stem
        out.attrs["mceq_db_manifest"] = MANIFEST
        out.attrs["mceq_db_spine"] = spine
        out.attrs["mceq_db_models"] = json.dumps(models)
        for grp in ("common", "decays", "continuous_losses"):
            if grp in src and (spine or grp == "common"):
                out.copy(src[grp], grp)
        for medium, names in models.items():
            for name in names:
                had = f"hadronic_interactions/{medium}"
                out.require_group(had).copy(src[had][name], name)
                out[had].copy(src[had][f"{name}_indptrs"], f"{name}_indptrs")
                cs = f"cross_sections/{medium}"
                if name in src[cs]:
                    out.require_group(cs).copy(src[cs][name], name)


def _write_monolith(src, dest):
    """The source as a plain monolith: same datasets, no package attributes."""
    with h5py.File(dest, "w") as out:
        for k, v in src.attrs.items():
            if k not in PACKAGE_ATTRS:
                out.attrs[k] = v
        for grp in src:
            out.copy(src[grp], grp)


def write_manifest(directory, entries, url_base=URL_BASE):
    """A schema-1 manifest for ``entries`` = ``{file: (spine, models)}``."""
    directory = pathlib.Path(directory)
    with h5py.File(REFERENCE_DB, "r") as src:
        grid = db_manifest.grid_signature(src["common"].attrs)
    files = {}
    for name, (spine, models) in entries.items():
        path = directory / name
        files[name] = {
            "sha256": _sha256(path),
            "size": path.stat().st_size,
            "grid": GRID_ID,
            "spine": spine,
            "models": models,
        }
    manifest = {
        "schema": db_manifest.SCHEMA,
        "release": "test",
        "url_base": url_base,
        "defaults": {"1d": BASE_FILE},
        "grids": {GRID_ID: grid},
        "files": files,
    }
    (directory / MANIFEST).write_text(yaml.safe_dump(manifest, sort_keys=False))
    return manifest


def build_fixture(directory):
    """Write the monolith, ``BASE_FILE``, ``MODEL_FILE`` and ``MANIFEST`` into ``directory``."""
    if not REFERENCE_DB.is_file():
        raise FileNotFoundError(
            f"{REFERENCE_DB} not found; the CI workflow caches it from the "
            "ci-assets release (see .github/workflows/_run_tests.yml)."
        )
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    base_models = {MEDIUM: [BASE_MODEL]}
    model_models = {MEDIUM: [MODEL]}
    with h5py.File(REFERENCE_DB, "r") as src:
        _write_monolith(src, directory / MONOLITH_FILE)
        _write_package(src, directory / BASE_FILE, base_models, spine=True)
        _write_package(src, directory / MODEL_FILE, model_models, spine=False)
    write_manifest(
        directory,
        {BASE_FILE: (True, base_models), MODEL_FILE: (False, model_models)},
    )
    return directory
