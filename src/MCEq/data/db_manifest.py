"""Data-release manifest: which package file carries which hadronic model.

The v2 data release is a set of HDF5 packages plus one YAML manifest
(``config.mceq_db_manifest``). Every package entry records the energy grid
(dims and byte digests of ``e_grid``/``k_grid``), sha256, size and the models
it carries per medium::

    schema: 1
    release: v2.0.0
    url_base: https://github.com/mceq-project/MCEq/releases/download/v2.0.0/
    defaults: {1d: mceq_base_air_1d_v2.h5, 2d: mceq_base_air_2d_v2.h5}
    grids:
      1d-150: {e_dim: 150, k_dim: null, e_grid_sha256: ...}
    files:
      mceq_base_air_1d_v2.h5:
        sha256: ...
        size: ...
        grid: 1d-150
        spine: true          # carries common/decays/continuous_losses
        models: {air: [FLUKA20251, SIBYLL23E], hydrogen: [SIBYLL23E]}
      mceq_extra_models_air_1d_v2.h5:
        grid: 1d-150
        spine: true
        models: {air: [DPMJETIII193, EPOSLHC, ...]}

``config.mceq_db_fname`` names the *primary* file: it supplies the grid,
decays and continuous losses (so it must be a ``spine`` package) and whatever
models it holds. When a run asks for a model the primary lacks, the backend
comes here: :class:`ModelStoreCache` picks the manifest entry on the **same
grid** that lists the model for the medium, fetches the file once if it is
missing (:mod:`MCEq.data.download`, sha256 from the manifest, race-safe),
checks that its ``common/e_grid`` is byte-identical to the primary's, and
hands back the store. Registering a new model is one manifest entry pointing
at a single-model file; nothing here knows file names.

Only release packages take this path: the packager stamps the root attribute
``mceq_db_manifest`` on every package, and the backend consults this module
only when the primary file carries it. A monolithic database is resolved
against its own file exactly as before.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from MCEq.data import hdf5_store

SCHEMA = 1


def array_sha256(values) -> str:
    """sha256 of the raw C-order bytes of an array: the spine digest."""
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(values)).tobytes()
    ).hexdigest()


def grid_signature(common_attrs) -> dict:
    """Grid identity of one file, in the manifest's ``grids`` form."""
    sig = {
        "e_dim": int(common_attrs["e_dim"]),
        "k_dim": int(common_attrs["k_dim"]) if "k_dim" in common_attrs else None,
        "e_grid_sha256": array_sha256(common_attrs["e_grid"]),
    }
    if sig["k_dim"] is not None:
        sig["k_grid_sha256"] = array_sha256(common_attrs["k_grid"])
    return sig


def load(path) -> dict:
    """Parse and sanity-check a manifest file."""
    manifest = yaml.safe_load(Path(path).read_text())
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ValueError(
            f"{path}: not a schema-{SCHEMA} MCEq data manifest "
            f"(schema={manifest.get('schema') if isinstance(manifest, dict) else None})"
        )
    for key in ("grids", "files"):
        if not isinstance(manifest.get(key), dict):
            raise ValueError(f"{path}: manifest lacks a '{key}' mapping")
    return manifest


def files_for(manifest, grid_sig, model, medium) -> list[str]:
    """Package names on ``grid_sig`` whose ``models[medium]`` list ``model``."""
    return [
        name
        for name, entry in manifest["files"].items()
        if manifest["grids"].get(entry.get("grid")) == grid_sig
        and model in entry.get("models", {}).get(medium, [])
    ]


def models_on_grid(manifest, grid_sig) -> dict[str, list[str]]:
    """``{medium: sorted models}`` the release offers on ``grid_sig``."""
    out: dict[str, set] = {}
    for entry in manifest["files"].values():
        if manifest["grids"].get(entry.get("grid")) != grid_sig:
            continue
        for medium, models in entry.get("models", {}).items():
            out.setdefault(medium, set()).update(models)
    return {m: sorted(v) for m, v in sorted(out.items())}


class ModelStoreCache:
    """Per-backend resolution of models the primary package does not carry.

    One instance per :class:`~MCEq.data.hdf5_backend.HDF5Backend`. The
    manifest is loaded on the first miss; each resolved file is opened once
    and cached by name, and a model that no package on the grid provides is
    remembered so repeated lookups do not re-read the manifest.
    """

    def __init__(self, paths, primary: hdf5_store.HDF5Store, fetcher):
        # ``fetcher`` supplies ``ensure_manifest(paths, name)`` and
        # ``ensure_package(paths, name, manifest)`` (MCEq.data.download);
        # injected so the fetch layer, which reads this module's schema,
        # is not imported back from here.
        self._paths = paths
        self._primary = primary
        self._fetch = fetcher
        # The packager stamps the manifest a package belongs to; a CI or
        # custom set names its own, which takes precedence over the config
        # default when it is present in data_dir.
        declared = primary.attrs().get("mceq_db_manifest")
        self.manifest_name = str(declared) if declared else paths.mceq_db_manifest
        common = primary.attrs("common")
        self._grid = grid_signature(common)
        self._e_grid = np.ascontiguousarray(np.asarray(common["e_grid"]))
        self._manifest = None
        self._stores: dict[str, hdf5_store.HDF5Store] = {}
        self._unavailable: set = set()

    @property
    def manifest(self) -> dict:
        if self._manifest is None:
            self._manifest = self._fetch.ensure_manifest(
                self._paths, self.manifest_name
            )
        return self._manifest

    def get(self, model, medium, tree):
        """Store holding ``tree/<medium>/<model>``, or None if no package on
        this grid lists the model for the medium."""
        key = (model, medium)
        if key in self._unavailable:
            return None
        for name in files_for(self.manifest, self._grid, model, medium):
            store = self._store(name)
            if store.has(f"{tree}/{medium}/{model}"):
                return store
        self._unavailable.add(key)
        return None

    def _store(self, name) -> hdf5_store.HDF5Store:
        store = self._stores.get(name)
        if store is None:
            path = self._fetch.ensure_package(self._paths, name, self.manifest)
            store = hdf5_store.HDF5Store(str(path))
            self._check_grid(store, name)
            self._stores[name] = store
        return store

    def _check_grid(self, store, name) -> None:
        """Refuse a package whose ``common/e_grid`` is not byte-identical to
        the primary's: the two would index each other's bins silently."""
        other = np.ascontiguousarray(np.asarray(store.attrs("common")["e_grid"]))
        if (
            other.shape != self._e_grid.shape
            or other.tobytes() != self._e_grid.tobytes()
        ):
            raise Exception(
                f"Data package {name} ({store.fname}) has a different energy grid "
                f"than the primary database {self._paths.mceq_db_fname}. The "
                "manifest lists it on the same grid, so one of the two files is "
                "stale: delete it and fetch again with "
                f"python -m MCEq.data.download --file {name}"
            )


def missing_model_error(paths, cache: ModelStoreCache, model, medium) -> Exception:
    """The error raised when neither the primary file nor any package on its
    grid can serve ``model`` in ``medium``."""
    try:
        offered = models_on_grid(cache.manifest, cache._grid)
        manifest_note = (
            f"Models on this grid per {cache.manifest_name}: {json.dumps(offered)}."
        )
    except Exception as ex:  # manifest unavailable: say so, still actionable
        manifest_note = f"(release manifest not readable: {ex})"
    return Exception(
        f"Interaction model {model} for medium {medium!r} is not in the "
        f"primary database {paths.mceq_db_fname} and no package of the data "
        f"release provides it on the same grid. {manifest_note} To register a "
        f"single-model file, add an entry to {Path(paths.data_dir) / cache.manifest_name} "
        "(see the data installation docs)."
    )
