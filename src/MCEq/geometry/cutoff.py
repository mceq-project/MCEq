"""Geomagnetic rigidity cutoffs from gtracr (cached per site)."""

from __future__ import annotations

import hashlib
import time as _time
from pathlib import Path
from typing import Optional

import numpy as np

from MCEq import config
from MCEq.geometry.atmosphere_parameters import site_coordinates
from MCEq.misc import info

# Bump to invalidate every existing cache file (e.g. when the IGRF model
# or the gtracr API changes in a backward-incompatible way).
CACHE_VERSION = 2

# Native gtracr sampling grid: 1° × 1° (180 zen × 360 az). All downstream
# pixel grids are linearly interpolated from this.
_NATIVE_N_ZEN = 180
_NATIVE_N_AZ = 360


def _cache_dir() -> Path:
    """Return ``<config.data_dir>/gtracr_cutoffs/``; create if missing."""
    d = Path(config.data_dir) / "gtracr_cutoffs"
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

    Coordinates come from ``_detector_latitude/_longitude`` of detector-centred
    models, or from :func:`MCEq.geometry.atmosphere_parameters.site_coordinates`
    for a named site. The name is only a cache label: ``_detector_name`` if set, ``IceCube``
    at (-90, 0), otherwise a string encoding the coordinates.

    Raises:
        ValueError: if neither detector coordinates nor a known location exist.
    """
    # Detector-centred MSIS
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

    site = site_coordinates(getattr(density_model, "location", None))
    if site is not None:
        name, lon, lat = site
        return name, float(lat), float(lon)

    raise ValueError(
        f"gtracr_cutoff: cannot infer a geographic location from "
        f"density_model={type(density_model).__name__}. Pass an "
        f"MSIS*Centered atmosphere or an MSIS/CORSIKA atmosphere of a "
        f"named site."
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
) -> np.ndarray:
    """Return a rigidity-cutoff map ``R_c[zen, az]`` in GV.

    Hits a per-installation cache under ``config.data_dir/gtracr_cutoffs/``;
    if absent, runs gtracr and writes the file.

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

    cache_dir = _cache_dir()
    cache_file = cache_dir / (
        f"gtracr_cutoffs_{location_name}_{bfield_type}_v{CACHE_VERSION}_{key_hash}.npz"
    )

    if cache_file.exists() and not force_rebuild:
        with np.load(cache_file, allow_pickle=True) as d:
            g_zen = d["gtracr_native_zen"].copy()
            g_az = d["gtracr_native_az"].copy()
            g_rc = d["gtracr_native_rcutoff_GV"].copy()
    else:
        # Explicit coordinates, so gtracr samples at the same point as the
        # atmospheric column (a bare site name would use gtracr's own table).
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

        info(1, f"gtracr: {iter_num} trajectories at {location_name} ({bfield_type})")
        t0 = _time.monotonic()
        gmrc.evaluate_batch(dt=1e-5, max_time=1.0)
        wall = _time.monotonic() - t0
        info(1, f"gtracr: done in {wall:.0f}s")

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
