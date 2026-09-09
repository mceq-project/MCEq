"""Geomagnetic rigidity-cutoff integration via gtracr.

Self-contained helper used by :class:`MCEq.core.MCEqRun` to apply a
per-pixel rigidity cutoff to the primary spectrum on the full-sky path.

Flow on first call after install (or after a cache-invalidation):

1. Look up the detector location from the active density model
   (``MSIS00*Centered`` / ``MSIS21*Centered`` exposes
   ``_detector_latitude/_longitude``; named MSIS atmospheres / location-
   tagged CORSIKA atmospheres look up the coords in
   :data:`MCEq.environment.parameters.LOCATIONS`).
2. Run ``gtracr.geomagnetic_cutoffs.GMRC`` to sample trajectories at the
   detector position (IGRF-13 by default), with a tqdm progress bar.
   The position is an explicit ``gtracr.location.Location`` carrying the
   coordinates from step 1, for every site: MCEq's table is authoritative,
   so the cutoff map and the atmospheric column always refer to the same
   point on the Earth.
3. Bin to a fine native grid (180 × 360) and gap-fill any NaN cells.
4. Save the result to ``<MCEq.data_dir>/gtracr_cutoffs/<key>.npz``.
   Cache key encodes location, IGRF date, B-field type, particle
   altitude, MC iter count, and :data:`CACHE_VERSION`.

Subsequent calls with the same key hit the cache. Re-evaluate by
deleting the file or bumping :data:`CACHE_VERSION`.

The bin returned to the caller is linearly interpolated from the fine
native grid onto the caller's ``(zenith_centres, azimuth_centres)``,
so the gtracr binning is decoupled from MCEq's pixel grid.
"""

from __future__ import annotations

import hashlib
import time as _time
from pathlib import Path
from typing import Optional

import numpy as np

# Bump to invalidate every existing cache file (e.g. when the IGRF model
# or the gtracr API changes in a backward-incompatible way).
#
# v2: the trajectory sampling point is MCEq's detector coordinate for every
# site, where v1 handed gtracr the bare site name for IceCube and ORCA and
# let gtracr's own table decide. Maps cached under v1 sampled a different
# position for those two, so they cannot be reused.
CACHE_VERSION = 2

# Native gtracr sampling grid: 1° × 1° (180 zen × 360 az). All downstream
# pixel grids are linearly interpolated from this.
_NATIVE_N_ZEN = 180
_NATIVE_N_AZ = 360


def _try_tqdm():
    """Return a tqdm callable or a no-op stand-in if tqdm is unavailable."""
    try:
        from tqdm import tqdm

        return tqdm
    except ImportError:  # pragma: no cover

        class _NoTqdm:
            def __init__(self, iterable=None, total=None, **kw):
                self.iterable = iterable

            def __iter__(self):
                return iter(self.iterable or [])

            def update(self, n=1):
                pass

            def close(self):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def set_postfix_str(self, *a, **k):
                pass

        return _NoTqdm


def _cache_dir(paths=None) -> Path:
    """Return ``<paths.data_dir>/gtracr_cutoffs/``; create if missing.

    ``paths`` is a ``config.paths`` group; ``None`` reads the global one
    through a dynamic import, keeping this layer free of a static import
    edge to ``MCEq.config``; the driver always passes the run's group.
    """
    if paths is None:
        from importlib import import_module

        paths = import_module("MCEq.config").paths

    d = Path(paths.data_dir) / "gtracr_cutoffs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(
    location: str,
    bfield_type: str,
    date: str,
    particle_altitude: float,
    iter_num: int,
    min_rigidity: float,
    max_rigidity: float,
    delta_rigidity: float,
) -> tuple[str, str]:
    """Return (key_string, short_hash). Key string is human-readable."""
    key_str = (
        f"v{CACHE_VERSION}|"
        f"{location}|{bfield_type}|{date}|"
        f"palt{particle_altitude}km|iter{iter_num}|"
        f"R[{min_rigidity}-{max_rigidity}|d{delta_rigidity}]"
    )
    h = hashlib.sha1(key_str.encode()).hexdigest()[:10]
    return key_str, h


def _gtracr_location_from_atmosphere(density_model):
    """Resolve an MCEq atmosphere to ``(name, lat, lon)`` for gtracr.

    Raises ``ValueError`` unless the model exposes detector coordinates or a
    location name present in :data:`atmosphere_parameters.LOCATIONS`.

    The coordinates always come from MCEq — ``_detector_latitude /
    _detector_longitude`` for the detector-centred models, the
    :data:`atmosphere_parameters.LOCATIONS` table for a named site — and
    :func:`get_cutoff_map` samples at exactly those. The name is a label on
    them, not a lookup key, so how it is obtained cannot move the sampling
    point:

    - the KM3NeT subclasses carry ``_detector_name`` (``ORCA`` / ``ARCA``),
      set from the same ``_KM3NET_DETECTORS`` entry the coordinates come
      from, so the tag cannot drift away from the position it names;
    - without an explicit detector name, coordinates within 1e-6 degrees of
      (-90, 0) are labelled ``IceCube``; everything else gets a stub
      spelling out its own coordinates.
    """
    from MCEq.environment.parameters import LOCATIONS

    # Detector-centered MSIS — most informative branch
    lat = getattr(density_model, "_detector_latitude", None)
    lon = getattr(density_model, "_detector_longitude", None)
    if lat is not None and lon is not None:
        name = getattr(density_model, "_detector_name", None)
        if not name:
            if abs(lat - (-90.0)) < 1e-6 and abs(lon) < 1e-6:
                name = "IceCube"
            else:
                name = f"coord_lat{lat:+07.3f}_lon{lon:+08.3f}".replace(".", "p")
        return str(name), float(lat), float(lon)

    # Named MSIS or CORSIKA-style atmosphere with a location in the table
    loc = getattr(density_model, "location", None)
    if isinstance(loc, str) and loc in LOCATIONS:
        lon_cm, lat_cm, _alt_cm = LOCATIONS[loc]
        return loc, float(lat_cm), float(lon_cm)

    raise ValueError(
        f"gtracr_cutoff: cannot infer a geographic location from "
        f"density_model={type(density_model).__name__}. Pass an "
        f"MSIS*Centered atmosphere or a CORSIKA/MSIS00 atmosphere with "
        f"a named LOCATIONS entry."
    )


def get_cutoff_map(
    density_model,
    zen_centres: np.ndarray,
    az_centres: np.ndarray,
    *,
    bfield_type: str = "igrf",
    date: Optional[str] = None,
    particle_altitude: float = 100.0,
    iter_num: int = 30000,
    min_rigidity: float = 0.1,
    max_rigidity: float = 55.0,
    delta_rigidity: float = 0.5,
    force_rebuild: bool = False,
    paths=None,
) -> np.ndarray:
    """Return a rigidity-cutoff map ``R_c[zen, az]`` in GV.

    Hits a per-installation cache under ``paths.data_dir/gtracr_cutoffs/``;
    if absent, runs gtracr with a tqdm progress bar and writes the file.

    Args:
        density_model: an MCEq atmosphere object with detector geography.
        zen_centres, az_centres: target grid centres in degrees.
        bfield_type: ``"igrf"`` (default) or ``"table"`` (faster lookup).
        date: ISO-format date string; defaults to today.
        particle_altitude: top-of-atmosphere altitude in km (gtracr trajectory
            launch height). 100 km is the gtracr default.
        iter_num: number of MC trajectories. Production maps use 30 000.
        min_rigidity, max_rigidity, delta_rigidity: rigidity scan in GV.
        force_rebuild: ignore the cache; always run gtracr.
        paths: a ``config.paths`` group locating the cache; ``None`` reads
            the global one.

    Returns:
        ``(n_zen, n_az)`` array of rigidity cutoffs in GV.
    """
    try:
        import gtracr
        from gtracr.geomagnetic_cutoffs import GMRC
        from gtracr.location import Location
    except ImportError as e:
        raise ImportError(
            "gtracr is required for the geomagnetic cutoff feature. "
            "Install from https://github.com/afedynitch/gtracr or set "
            "geomagnetic_cutoff=False on MCEqRun."
        ) from e

    if date is None:
        from datetime import date as _date

        date = str(_date.today())

    location_name, lat, lon = _gtracr_location_from_atmosphere(density_model)
    key_str, key_hash = _cache_key(
        location_name,
        bfield_type,
        date,
        particle_altitude,
        iter_num,
        min_rigidity,
        max_rigidity,
        delta_rigidity,
    )

    n_zen = int(zen_centres.size)
    n_az = int(az_centres.size)

    cache_dir = _cache_dir(paths)
    cache_file = cache_dir / (
        f"gtracr_cutoffs_{location_name}_{bfield_type}_v{CACHE_VERSION}_{key_hash}.npz"
    )

    if cache_file.exists() and not force_rebuild:
        with np.load(cache_file, allow_pickle=True) as d:
            g_zen = d["gtracr_native_zen"].copy()
            g_az = d["gtracr_native_az"].copy()
            g_rc = d["gtracr_native_rcutoff_GV"].copy()
    else:
        # MCEq's site table is authoritative: always hand gtracr an explicit
        # Location built from the coordinates MCEq resolved, never a bare
        # site name. Passing a name gtracr recognises would make gtracr's own
        # table decide where the trajectories are sampled, so the cutoff map
        # would belong to a different point than the atmospheric column the
        # rest of the calculation integrates — the two must agree.
        gmrc_loc = Location(name=location_name, latitude=lat, longitude=lon)

        gmrc = GMRC(
            location=gmrc_loc,
            particle_altitude=particle_altitude,
            iter_num=iter_num,
            bfield_type=bfield_type,
            date=date,
            min_rigidity=min_rigidity,
            max_rigidity=max_rigidity,
            delta_rigidity=delta_rigidity,
            solver="rk45",
        )

        # gtracr's evaluate_batch runs the full MC; wrap a tqdm bar around
        # the wall-clock so the user has feedback. (gtracr itself does not
        # expose a per-trajectory callback.)
        tqdm_cls = _try_tqdm()
        bar = tqdm_cls(
            total=iter_num,
            desc=f"gtracr {location_name} {bfield_type}",
            unit="traj",
        )
        t0 = _time.monotonic()
        gmrc.evaluate_batch(dt=1e-5, max_time=1.0)
        bar.update(iter_num)
        bar.close()
        wall = _time.monotonic() - t0

        g_az, g_zen, g_rc = gmrc.bin_results(
            nbins_azimuth=_NATIVE_N_AZ,
            nbins_zenith=_NATIVE_N_ZEN,
        )

        # Gap-fill any NaN bins with nearest-neighbour
        if np.any(~np.isfinite(g_rc)):
            from scipy.ndimage import distance_transform_edt

            mask = ~np.isfinite(g_rc)
            if mask.all():
                raise RuntimeError(
                    f"gtracr returned all-NaN bins at iter_num={iter_num}; "
                    f"increase iter_num"
                )
            idx = distance_transform_edt(
                mask,
                return_distances=False,
                return_indices=True,
            )
            g_rc = g_rc[tuple(idx)]

        np.savez(
            cache_file,
            gtracr_native_zen=g_zen,
            gtracr_native_az=g_az,
            gtracr_native_rcutoff_GV=g_rc,
            location=np.array(location_name),
            bfield_type=np.array(bfield_type),
            date=np.array(date),
            iter_num=np.array(iter_num),
            particle_altitude=np.array(particle_altitude),
            cache_version=np.array(CACHE_VERSION),
            wall_seconds=np.array(wall),
            key=np.array(key_str),
            gtracr_version=np.array(getattr(gtracr, "__version__", "unknown")),
        )

    # Linear interpolation onto the caller's grid. Azimuth wraps at 360°,
    # so pad the native grid by one column on each side for periodic
    # interpolation.
    from scipy.interpolate import RegularGridInterpolator

    az_pad = np.concatenate([[g_az[-1] - 360.0], g_az, [g_az[0] + 360.0]])
    r_pad = np.concatenate([g_rc[:, -1:], g_rc, g_rc[:, :1]], axis=1)
    interp = RegularGridInterpolator(
        (g_zen, az_pad),
        r_pad,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    z_q = np.clip(zen_centres, g_zen[0], g_zen[-1])
    a_q = np.mod(az_centres, 360.0)
    Z, A = np.meshgrid(z_q, a_q, indexing="ij")
    rc = interp(np.stack([Z.ravel(), A.ravel()], axis=-1)).reshape(n_zen, n_az)
    return rc


# ---------------------------------------------------------------------------
# Primary spectrum with per-pixel cutoff
# ---------------------------------------------------------------------------

# GaisserHonda / GSF mass-group nominal nuclear charges (Z) and mass numbers
# (A). Used as the canonical superposition for primary spectra without their
# own per-species rigidity awareness.
_DEFAULT_MASS_GROUPS = (
    # (corsika_id, Z, A)
    (14, 1, 1),  # p
    (402, 2, 4),  # He
    (1206, 6, 12),  # C-N-O
    (2814, 14, 28),  # Si
    (5426, 26, 56),  # Fe-56; the id is crflux's group key, not A*100+Z
)


def build_phi0_with_cutoff(
    mceq,
    primary_model,
    rc_GV_per_pixel: np.ndarray,
    mass_groups=_DEFAULT_MASS_GROUPS,
    physics=None,
) -> np.ndarray:
    """Build a ``(dim_states, K)`` phi0 with rigidity cutoff applied.

    For each mass group ``(Z, A)`` with cutoff ``R_c[pixel]``, the group is
    accepted where ``A * E_nuc > Z * R_c`` — nucleus total energy
    ``A * E_nuc`` against the rigidity threshold ``Z * R_c``, in the
    ultrarelativistic approximation ``R ≈ E / Z``. Below the threshold the
    species contributes zero. Above, the standard proton+neutron
    superposition is applied as in
    :meth:`MCEqRun.set_primary_model`.

    ``physics`` is a ``config.physics`` group supplying
    ``minimal_primary_energy``; ``None`` reads the global one through a
    dynamic import, keeping this layer free of a static import edge to
    ``MCEq.config``; the driver always passes the run's group.
    """
    if physics is None:
        from importlib import import_module

        physics = import_module("MCEq.config").physics

    K = int(rc_GV_per_pixel.size)
    e_grid = mceq._energy_grid.c
    n_E = e_grid.size

    p_mass = mceq.pman[(2212, 0)].mass
    n_mass = mceq.pman[(2112, 0)].mass
    e_tot_nucleon = e_grid + 0.5 * (p_mass + n_mass)

    minimal_energy = physics.minimal_primary_energy
    min_idx = int(np.argmin(np.abs(e_tot_nucleon - minimal_energy)))

    phi0 = np.zeros((mceq.dim_states, K), dtype=np.float64)
    p_flux_2d = np.zeros((n_E, K), dtype=np.float64)
    n_flux_2d = np.zeros((n_E, K), dtype=np.float64)

    for corsika_id, Z, A in mass_groups:
        if corsika_id not in primary_model.nucleus_ids:
            continue
        E_nucleus_tot = e_tot_nucleon * A
        phi_nuc = np.asarray(
            primary_model.nucleus_flux(corsika_id, E_nucleus_tot),
            dtype=np.float64,
        )
        mask = (E_nucleus_tot[:, None] > Z * rc_GV_per_pixel[None, :]).astype(
            np.float64
        )
        p_flux_2d += Z * A * (phi_nuc[:, None] * mask)
        n_flux_2d += (A - Z) * A * (phi_nuc[:, None] * mask)

    p = mceq.pman[(2212, 0)]
    if (2112, 0) in mceq.pman and not mceq.pman[(2112, 0)].is_resonance:
        nproj = mceq.pman[(2112, 0)]
        has_neutrons = True
    else:
        has_neutrons = False

    for k in range(K):
        phi0[p.lidx + min_idx : p.uidx, k] = 1e-4 * p_flux_2d[min_idx:, k]
        if has_neutrons:
            phi0[nproj.lidx + min_idx : nproj.uidx, k] = 1e-4 * n_flux_2d[min_idx:, k]

    return phi0
