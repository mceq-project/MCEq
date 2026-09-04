"""Golden section ``paths``: ETD2 integration paths and the zenith contract.

Pins ``(nsteps, dX, rho_inv, grid_idcs)`` from
:func:`MCEq.solvers.etd2_nonuniform_path` for CORSIKA / MSIS00 / MSIS21 at
zenith 0 / 60 / 85 / 90 deg, one snapshot-grid variant, the density-model name
table of :meth:`MCEqRun.set_density_model`, and the ``set_theta`` guard.

Built from standalone atmosphere objects. The builder touches only
``density_model.r_X2rho`` and ``density_model.max_X``, so this section needs no
MCEqRun and no HDF5 database. The output is bitwise equal to
``MCEqRun._calculate_integration_path(..., force=True)`` for all twelve
(model, zenith) pairs while ``config.em_adaptive_step`` is False:
``etd2_nonuniform_path`` never reads that flag, but the MCEqRun route replaces
``dX_max`` with ``min(dX_max, _em_cascade_dx_cap())`` when it is True, so the
equivalence holds only for the flag value recorded in the config stanza.

Paths are labelled by ``density_model.theta_deg``. That was originally forced:
``MCEqRun.theta_deg`` kept its constructor value forever (bug B13, fixed at
f087d25) and would have mislabelled every angle but the first. The atmosphere
remains the label source because it is also where the azimuth lives.

Everything is compared bitwise. The values reproduce across processes, across
OMP / MKL / OPENBLAS thread counts and through a fork pool, and are invariant
to atmosphere-object reuse and to the order the zeniths are visited.

Golden inputs beyond numpy/scipy versions: the FITPACK splines behind
``r_X2rho`` and behind the cumulative per-step mean, the hard-coded hybrid
sample sizes ``geomspace(..., 6001)`` and ``linspace(..., 4001)``
(solvers.py:172/175, not config), the ctypes ``nrlmsise00`` shared library
under ``MCEq/geometry/nrlmsise00`` (MSIS00) and the ``nrlmsis`` Fortran
extension (MSIS21).
"""

from __future__ import annotations

import numpy as np

from ._harness import make_provenance

SECTION = "paths"

#: (golden name, class in MCEq.geometry.density_profiles, constructor args).
#: The classes are named rather than imported so this module stays free of MCEq
#: imports until :func:`build` runs. ``MSIS21Atmosphere`` is re-exported from
#: ``density_profiles`` through its PEP-562 ``__getattr__``.
DENSITY_MODEL_SPECS = (
    ("CORSIKA", "CorsikaAtmosphere", ("BK_USStd", None)),
    ("MSIS00", "MSIS00Atmosphere", ("SouthPole", "January")),
    ("MSIS21", "MSIS21Atmosphere", ("SouthPole", "January")),
)

ZENITHS_DEG = (0.0, 60.0, 85.0, 90.0)

#: Snapshot depths in g/cm^2 for the ``int_grid`` case. All twelve default
#: paths have ``grid_idcs == []``, so the truncation logic that lands a step
#: exactly on a requested depth is otherwise unpinned.
SNAPSHOT_GRID_X = (10.0, 100.0, 500.0, 1000.0, 2000.0)

#: First zenith past ``max_theta`` (= 90.0 for these models).
OUT_OF_RANGE_ZENITH_DEG = 90.0001

NOTE = """\
Integration paths for CORSIKA/MSIS00/MSIS21 at zenith 0/60/85/90 deg, built
from standalone atmosphere objects (no MCEqRun, no HDF5 DB) and bitwise equal
to the MCEqRun route while config.em_adaptive_step is False.

Zenith guard: max_theta is 90.0 for all three models, theta=90.0 is accepted,
and anything above it raises a BARE Exception("Zenith angle not in allowed
range.") from ``EarthsAtmosphere.set_theta`` (``MSIS21Atmosphere.set_theta``
for MSIS21; ValueError from ``LocationCenteredMixin.set_theta`` in the
LocationCentered variants). pytest.raises(ValueError)
does not catch it. Pinned as the exception type name in
zenith_guard/exception_above_max_theta.

available_density_models is MCEq.geometry.registry.available_models(), the same
table set_density_model dispatches through, so a name can no longer be offered
without being buildable or the reverse. It used to be a list literal local to
that method with a parallel elif ladder beside it, and the two disagreed:
"MSIS00_KM3NeT" was absent although MSIS00KM3NeTCentered is implemented, which
was bug B15. Fixed, so this key is (10,) where the first build of this section
recorded (9,). The list is the golden rather than the ValueError it raises,
because every unknown model name raises the identical
ValueError("Choose a different profile.").

max_den_type_before_set_theta pins the type of config.environment.max_density,
which EarthsAtmosphere.__init__ copies into _max_den: the public max_den
property serves that float until the first set_theta overwrites it. A tuple
here is the signature of bug B14, a second max_density assignment in config
shadowing the float.

max_den itself is not stored per path: density_profiles.py:117 sets it to the
density at the top of the atmosphere, which is constant over zenith (CORSIKA
1.0e-9, MSIS00 5.06e-11, MSIS21 4.32e-11 g/cm^3) and discriminates only the
model, which max_X already does.
"""


def _available_density_models() -> list[str]:
    """The model names ``set_density_model`` accepts.

    Until the registry existed this had to be recovered by parsing the
    ``available_models`` list literal out of the method's source with ``ast``,
    because it was a local variable. It is now a module-level table, so the
    section reads it directly -- and reads the same object the dispatch uses,
    which is what made bug B15 possible: the literal and the ``elif`` ladder
    could disagree, and did.
    """
    from MCEq.geometry.registry import available_models

    return available_models()


def _path_arrays(prefix: str, path, max_X) -> dict:
    """Flatten one ``(nsteps, dX, rho_inv, grid_idcs)`` tuple into golden keys.

    ``nsteps`` is a python int and ``grid_idcs`` a python list (empty for a path
    without snapshots); both are cast so the npz round-trip keeps int64 instead
    of promoting an empty list to float64.
    """
    nsteps, dX, rho_inv, grid_idcs = path
    return {
        f"{prefix}/nsteps": np.int64(nsteps),
        f"{prefix}/dX": dX,
        f"{prefix}/rho_inv": rho_inv,
        f"{prefix}/grid_idcs": np.asarray(grid_idcs, dtype=np.int64),
        f"{prefix}/max_X": np.float64(max_X),
    }


def _zenith_guard_exception(atmosphere, zenith_deg: float) -> str:
    """Name of the exception type ``set_theta`` raises past ``max_theta``."""
    try:
        atmosphere.set_theta(zenith_deg)
    except Exception as exc:
        return type(exc).__name__
    return "<none>"


def build() -> tuple[dict, dict]:
    """Produce (arrays, provenance) for this section."""
    from MCEq import config
    from MCEq.geometry import density_profiles as dprof
    from MCEq.solvers import etd2_nonuniform_path

    from .make_goldens import SectionUnavailable

    # MSIS21 rides on the optional `nrlmsis` package, which only the `test`
    # dependency group installs.
    try:
        import nrlmsis  # noqa: F401
    except ImportError as exc:
        raise SectionUnavailable(
            "the MSIS21 paths need the optional nrlmsis package"
        ) from exc

    arrays = {}
    # The paths come from the atmosphere and the integration grid, not from the
    # hadronic yields, but `MCEqRun` opens a database to report which density
    # models it accepts -- and the provenance records that file's sha256. Pin
    # the reduced database the other sections use, or the section is generated
    # against whatever the host happens to have configured and can never be
    # verified anywhere else.
    saved = {k: getattr(config, k) for k in ("debug_level", "mceq_db_fname")}
    config.debug_level = 0
    config.mceq_db_fname = "mceq_db_v140reduced_compact.h5"
    try:
        atmospheres = {
            name: getattr(dprof, cls_name)(*model_config)
            for name, cls_name, model_config in DENSITY_MODEL_SPECS
        }

        # Read before any set_theta, while _max_den still holds the value
        # EarthsAtmosphere.__init__ copied out of config.max_density.
        arrays["max_den_type_before_set_theta"] = np.asarray(
            type(atmospheres["CORSIKA"].max_den).__name__
        )

        for name, atmosphere in atmospheres.items():
            arrays[f"{name}/max_theta"] = np.float64(atmosphere.max_theta)
            for zenith in ZENITHS_DEG:
                atmosphere.set_theta(zenith)
                arrays.update(
                    _path_arrays(
                        f"{name}/theta{zenith:g}",
                        etd2_nonuniform_path(atmosphere),
                        atmosphere.max_X,
                    )
                )

        corsika = atmospheres["CORSIKA"]
        corsika.set_theta(60.0)
        int_grid = np.asarray(SNAPSHOT_GRID_X)
        nsteps, dX, rho_inv, grid_idcs = etd2_nonuniform_path(
            corsika, int_grid=int_grid
        )
        prefix = "CORSIKA/theta60_snapshots"
        arrays.update(
            _path_arrays(prefix, (nsteps, dX, rho_inv, grid_idcs), corsika.max_X)
        )
        arrays[f"{prefix}/int_grid"] = int_grid
        # Depths reached after the flagged steps; equals int_grid exactly when
        # the truncation is correct (X_start = 0).
        arrays[f"{prefix}/X_at_grid_idcs"] = np.cumsum(dX)[np.asarray(grid_idcs)]

        arrays["zenith_guard/exception_above_max_theta"] = np.asarray(
            _zenith_guard_exception(corsika, OUT_OF_RANGE_ZENITH_DEG)
        )
        arrays["available_density_models"] = np.asarray(_available_density_models())

        provenance = make_provenance(
            SECTION,
            note=NOTE,
            tolerances={},
            extra={
                "density_model_specs": [
                    {"name": name, "class": cls_name, "config": model_config}
                    for name, cls_name, model_config in DENSITY_MODEL_SPECS
                ],
                "zeniths_deg": list(ZENITHS_DEG),
                "snapshot_grid_X": list(SNAPSHOT_GRID_X),
                "out_of_range_zenith_deg": OUT_OF_RANGE_ZENITH_DEG,
                "hybrid_sample_sizes": {"geomspace": 6001, "linspace": 4001},
                "fixed_bugs": ["B13", "B15"],
            },
        )
    finally:
        for key, value in saved.items():
            setattr(config, key, value)

    return arrays, provenance
