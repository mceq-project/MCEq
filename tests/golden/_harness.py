"""Shared machinery for the golden regression harness.

A *golden section* is one `.npz` file under `tests/golden/data/` holding named
arrays plus a `__provenance__` entry (a JSON string). Sections are produced by
the generators in this package and compared by `test_goldens.py`.

Comparison modes per key, resolved by :func:`tolerance_for`:

``bitwise``   `np.array_equal(..., equal_nan=True)` — what the host backends
              (numpy, MKL) deliver today for every section.
``rel_l2``    `||y - x||_2 / ||x||_2 <= rtol` — for backends whose reduction
              order is not fixed (CUDA), and the escape hatch for phases that
              deliberately change a summation or association order.
``per_species_max``
              the worst `max_E |dphi/phi_ref|` over the species of a state, on
              the bins above `floor * peak` of the grid less its top
              `trim_top_bins` bins, counting only the species a `guard` admits
              there — see :mod:`._flux_metric`. For the flux keys, where an L2
              hides a minor species behind the muon rows and an element-wise
              ratio measures the ~1e-16 components instead of the solver. The
              entry carries `floor`, `guard` and `trim_top_bins` alongside
              `rtol`, plus a `fallback_rtol` that bounds everything the maximum
              does not cover: the species the guard leaves unscored, contained
              per lane by relative L2, and the whole array of a key it empties.

The default is ``bitwise``; keys are moved to another mode only by an explicit
entry in the section's `__provenance__["tolerances"]` table, so a phase that
loosens a tolerance has to say so in its diff.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import platform
import sys

import numpy as np

from . import _flux_metric

DATA_DIR = pathlib.Path(__file__).parent / "data"

#: Relative-L2 budget for backends without a fixed reduction order. The plan
#: allows 1e-9 for CUDA; measured worst case at the Phase-0 tree is 9.8e-14.
CUDA_RTOL = 1e-9

#: Relative-L2 budget for a host change that legitimately reorders a sum
#: (plan decision D4: Phase 2 may move host results by this much).
HOST_RTOL = 1e-12

PROVENANCE_KEY = "__provenance__"


# --------------------------------------------------------------------------
# digests
# --------------------------------------------------------------------------


def array_digest(arr) -> str:
    """sha256 over dtype, shape and the C-contiguous bytes of `arr`.

    dtype is part of the digest, so a `float32` run is distinguishable from a
    `float64` one rather than colliding with it.
    """
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def sparse_digest(mat) -> dict:
    """Digest a scipy sparse matrix without disturbing it.

    Copies before any canonicalisation: `sort_indices()` mutates in place and
    MKL sparse handles hold raw pointers into the live `data`/`indices`
    buffers, so sorting a matrix that a backend has bound corrupts that handle.
    """
    m = mat.tocsr(copy=True)
    m.sort_indices()
    return {
        "shape": list(m.shape),
        "nnz": int(m.nnz),
        "dtype": m.dtype.str,
        "data": array_digest(m.data),
        "indices": array_digest(m.indices),
        "indptr": array_digest(m.indptr),
    }


def file_digest(path) -> str:
    """sha256 of a file, resolving symlinks (the DBs in `src/MCEq/data` are links)."""
    p = pathlib.Path(path).resolve()
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(obj):
    """Convert a value tree so `json.dumps` accepts it.

    Three config values need it: `mceqidx` is a mix of `int` and `numpy.int64`
    (`add_tracking_particle` uses `np.max(...) + 1`) and channel-index keys are
    tuples of them; `config.data_dir` is a `pathlib.Path`; `config.floatlen` is
    a numpy *type object* (`np.float32`), not an instance.
    """
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, pathlib.PurePath):
        return str(obj)
    if isinstance(obj, type):
        return f"{obj.__module__}.{obj.__qualname__}"
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return repr(obj)


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def environment_stanza() -> dict:
    """Versions and host facts that can move a golden without any MCEq change."""
    import scipy

    import MCEq
    from MCEq import config

    stanza = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "mceq_path": str(pathlib.Path(MCEq.__file__).parent),
        "has_mkl": bool(getattr(config, "has_mkl", False)),
        "has_cuda": bool(getattr(config, "has_cuda", False)),
        "has_accelerate": bool(getattr(config, "has_accelerate", False)),
    }
    for name in ("h5py", "crflux", "nrlmsis", "cupy"):
        try:
            stanza[name] = __import__(name).__version__
        except Exception:
            stanza[name] = None
    return stanza


#: Every config global that any generator in this package reads, directly or
#: through MCEq. A generator records all of them so a golden that moves can be
#: explained by diffing the stanza instead of re-deriving the inputs.
PINNED_CONFIG_KEYS = (
    "data_dir",
    "mceq_db_fname",
    "em_db_fname",
    "e_min",
    "e_max",
    "floatlen",
    "kernel_config",
    "cuda_fp_precision",
    "cuda_gpu_id",
    "enable_em",
    "enable_em_ion",
    "enable_energy_loss",
    "enable_cont_rad_loss",
    "generic_losses_all_charged",
    "muon_helicity_dependence",
    "muon_multiple_scattering",
    "enable_default_tracking",
    "standard_particles",
    "use_isospin_sym",
    "assume_nucleon_interactions_for_exotics",
    "fallback_to_air_cs",
    "interaction_medium",
    "prompt_ctau",
    "minimal_primary_energy",
    "return_as",
    "excpt_on_missing_particle",
    "average_loss_operator",
    "loss_step_for_average",
    "loss_stencil_method",
    "loss_stencil_alpha0",
    "loss_stencil_low_upwind_rows",
    "X_start",
    "etd2_path",
    "em_adaptive_step",
    "em_step_safety",
    "em_step_dense_eig_max",
    "em_air_density",
    "secant_theta_transport",
    "secant_theta_cap_deg",
    "secant_theta_row_kmax",
    "secant_theta_lam_rel",
    "secant_theta_w_flat",
    "secant_theta_e_max",
    "density_model",
    "h_obs",
    "h_atm",
    "r_E",
)


def config_stanza() -> dict:
    """Snapshot of `PINNED_CONFIG_KEYS` and `adv_set` as they stand right now."""
    from MCEq import config

    snap = {k: jsonable(getattr(config, k, "<undefined>")) for k in PINNED_CONFIG_KEYS}
    snap["adv_set"] = jsonable(dict(config.adv_set))
    return snap


def db_stanza() -> dict:
    """sha256 of the HDF5 databases in play.

    The `.h5` files are gitignored symlinks into other checkouts, so a golden
    that cannot be reproduced elsewhere is identified by these digests rather
    than by a path.
    """
    import os

    from MCEq import config

    out = {}
    for key in ("mceq_db_fname", "em_db_fname"):
        name = getattr(config, key, None)
        if not name:
            continue
        path = os.path.join(config.data_dir, name)
        out[key] = {"name": name, "exists": os.path.exists(path)}
        if os.path.exists(path):
            out[key]["sha256"] = file_digest(path)
    return out


def make_provenance(
    section: str, *, note: str = "", tolerances=None, extra=None
) -> dict:
    """Assemble the `__provenance__` payload for one section."""
    import subprocess

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=pathlib.Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        commit = None

    return {
        "section": section,
        "note": note,
        "git_commit": commit,
        "environment": environment_stanza(),
        "config": config_stanza(),
        "databases": db_stanza(),
        "tolerances": tolerances or {},
        "extra": jsonable(extra or {}),
    }


# --------------------------------------------------------------------------
# section IO
# --------------------------------------------------------------------------


def section_path(section: str) -> pathlib.Path:
    return DATA_DIR / f"{section}.npz"


def save_section(section: str, arrays: dict, provenance: dict) -> pathlib.Path:
    """Write one golden section.

    Values are stored uncompressed so the file is byte-stable for an identical
    regeneration and diffs do not churn.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {k: np.asarray(v) for k, v in arrays.items()}
    payload[PROVENANCE_KEY] = np.array(json.dumps(provenance, indent=1, sort_keys=True))
    path = section_path(section)
    with open(path, "wb") as fh:
        np.savez(fh, **payload)
    return path


def load_section(section: str):
    """Return `(arrays, provenance)` for a stored section."""
    path = section_path(section)
    if not path.exists():
        raise FileNotFoundError(
            f"golden section {section!r} missing at {path}. "
            f"Regenerate with `python -m tests.golden.make_goldens {section}` "
            f"or `pytest tests/golden --regenerate-goldens`."
        )
    with np.load(path, allow_pickle=False) as npz:
        prov = json.loads(str(npz[PROVENANCE_KEY]))
        arrays = {k: npz[k] for k in npz.files if k != PROVENANCE_KEY}
    return arrays, prov


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------


def tolerance_entry_for(key: str, provenance: dict) -> dict:
    """The tolerance entry governing one key, or the bitwise default.

    A key matches an entry if the entry is the key itself or a prefix of it
    ending in `/`, so a whole subtree can be loosened with one line; the
    longest matching prefix wins, which is how a section keeps `state/` on the
    flux metric while its per-mode norms under the same prefix stay on L2.
    """
    table = provenance.get("tolerances") or {}
    if key in table:
        return dict(table[key])
    for prefix, entry in sorted(table.items(), key=lambda kv: -len(kv[0])):
        if prefix.endswith("/") and key.startswith(prefix):
            return dict(entry)
    return {"mode": "bitwise", "rtol": 0.0}


def tolerance_for(key: str, provenance: dict):
    """Resolve `(mode, rtol)` for one key — the two fields every mode needs."""
    entry = tolerance_entry_for(key, provenance)
    return entry.get("mode", "rel_l2"), float(entry.get("rtol", HOST_RTOL))


def compare_key(
    key, expected, actual, mode, rtol, *, layout=None, entry=None, rtol_floor=0.0
):
    """Return `None` when `actual` matches `expected`, else an explanation.

    Shape and dtype-kind are checked before any numeric work so a structural
    change reports as a structural change rather than raising from `np.asarray`.

    `layout` and `entry` are read only by ``per_species_max``: the species
    table to slice the state with, which `compare_section` resolves from the
    golden, and the tolerance entry's `floor`, `guard`, `trim_top_bins` and
    `fallback_rtol` fields.

    `rtol_floor` is also read only by ``per_species_max``, and only by the two
    relative-L2 sub-bounds inside it -- the containment of the species the guard
    leaves unscored, and the whole-array fallback for a key it empties. Those
    are norms, so a backend that does not fix its reduction order moves them;
    the per-species maximum itself is the backend-independent statement and
    stays at its stored `rtol`. `compare_section` handles the floor for every
    other mode before calling.
    """
    exp = np.asarray(expected)
    act = np.asarray(actual)

    if exp.shape != act.shape:
        return f"{key}: shape {act.shape} != golden {exp.shape}"

    if exp.dtype.kind in "SUO" or act.dtype.kind in "SUO":
        if not np.array_equal(exp, act):
            return f"{key}: text/object value {act!r} != golden {exp!r}"
        return None

    if exp.dtype.kind != act.dtype.kind:
        return f"{key}: dtype kind {act.dtype.kind!r} != golden {exp.dtype.kind!r}"

    if mode == "per_species_max":
        if layout is None:
            return (
                f"{key}: per_species_max needs a species layout and the "
                f"section records none (meta/species arrays or "
                f"extra.species_layout)"
            )
        fields = entry or {}
        fallback = max(float(fields.get("fallback_rtol", rtol)), rtol_floor)
        try:
            return _flux_metric.compare(
                key,
                exp,
                act,
                rtol,
                layout=layout,
                floor=float(fields.get("floor", _flux_metric.FLOOR)),
                guard=fields.get("guard", _flux_metric.DEFAULT_GUARD),
                trim=int(fields.get("trim_top_bins", _flux_metric.TRIM_TOP_BINS)),
                fallback_rtol=fallback,
            )
        except _flux_metric.Unscorable:
            # The guard admits no species of this key, so the per-species
            # maximum does not exist for it. Fall through to relative L2 at
            # `fallback_rtol` — a bound that is always defined — rather than
            # let the key pass unchecked or demand bitwise equality of a
            # reduction whose order the backend chooses.
            mode = "rel_l2"
            rtol = fallback

    if mode == "bitwise":
        if exp.dtype != act.dtype:
            return f"{key}: dtype {act.dtype} != golden {exp.dtype} (bitwise mode)"
        if np.array_equal(exp, act, equal_nan=exp.dtype.kind == "f"):
            return None
        return f"{key}: not bitwise equal ({_summarise(exp, act)})"

    # rel_l2
    ef = np.asarray(exp, dtype=np.float64)
    af = np.asarray(act, dtype=np.float64)
    if not np.array_equal(np.isfinite(ef), np.isfinite(af)):
        n_e = int((~np.isfinite(ef)).sum())
        n_a = int((~np.isfinite(af)).sum())
        return f"{key}: non-finite pattern differs (golden {n_e}, actual {n_a})"
    finite = np.isfinite(ef)
    ef, af = ef[finite], af[finite]
    denom = np.linalg.norm(ef)
    if denom == 0.0:
        # An all-zero reference cannot support a relative bound; require exact.
        if np.array_equal(ef, af):
            return None
        return f"{key}: golden is all-zero and actual is not"
    rel = float(np.linalg.norm(af - ef) / denom)
    if rel <= rtol:
        return None
    return f"{key}: rel-L2 {rel:.3e} > {rtol:.1e}"


def _summarise(exp, act):
    ef = np.asarray(exp, dtype=np.float64).ravel()
    af = np.asarray(act, dtype=np.float64).ravel()
    finite = np.isfinite(ef) & np.isfinite(af)
    if not finite.any():
        return "no jointly finite entries"
    ef, af = ef[finite], af[finite]
    denom = np.linalg.norm(ef)
    rel = float(np.linalg.norm(af - ef) / denom) if denom else float("inf")
    with np.errstate(divide="ignore", invalid="ignore"):
        maxrel = float(
            np.nanmax(np.abs(af - ef) / np.where(ef == 0, np.nan, np.abs(ef)))
        )
    return f"rel-L2 {rel:.3e}, max elementwise {maxrel:.3e}"


def compare_section(section: str, produced: dict, *, rtol_floor: float = 0.0):
    """Compare freshly produced arrays against the stored section.

    Returns the list of mismatch descriptions; empty means the section matches.

    `rtol_floor` relaxes every key to at least that relative-L2 budget. It is
    how a backend without a fixed reduction order is compared against a host
    golden — CUDA at `CUDA_RTOL` — without loosening the stored tolerances,
    which stay bitwise for the host.

    A key already on `per_species_max` is exempt from that floor: its stored
    bound is the backend-independent one (a per-species maximum on the bins
    that carry flux does not depend on a reduction order the way an
    element-wise ratio does), so CUDA is judged on the flux keys by the
    per-species bound and on the rest by `rtol_floor`. The floor is still
    handed to `compare_key`, which applies it to the relative-L2 sub-bounds
    inside that mode -- those are norms and do move with the reduction order.
    """
    golden, prov = load_section(section)
    problems = []

    missing = sorted(set(golden) - set(produced))
    extra = sorted(set(produced) - set(golden))
    if missing:
        problems.append(f"{len(missing)} golden key(s) not produced: {missing[:8]}")
    if extra:
        problems.append(f"{len(extra)} new key(s) not in golden: {extra[:8]}")

    for key in sorted(set(golden) & set(produced)):
        entry = tolerance_entry_for(key, prov)
        mode = entry.get("mode", "rel_l2")
        rtol = float(entry.get("rtol", HOST_RTOL))
        if mode != "per_species_max" and rtol_floor > rtol:
            mode, rtol = "rel_l2", rtol_floor
        problem = compare_key(
            key,
            golden[key],
            produced[key],
            mode,
            rtol,
            layout=(
                _flux_metric.layout_for(key, golden, prov)
                if mode == "per_species_max"
                else None
            ),
            entry=entry,
            rtol_floor=rtol_floor,
        )
        if problem:
            problems.append(problem)
    return problems
