"""NRLMSISE-00 atmospheres, including the detector-centred variants."""

import numpy as np

from MCEq.environment.base import EarthsAtmosphere
from MCEq.environment.column import fit_column_splines
from MCEq.environment.location_centered import LocationCenteredMixin
from MCEq.environment.parameters import KM3NET_DETECTORS
from MCEq.misc import info


class MSIS00Atmosphere(EarthsAtmosphere):
    """Wrapper class for a python interface to the NRLMSISE-00 model.

    `NRLMSISE-00 <http://ccmc.gsfc.nasa.gov/modelweb/atmos/nrlmsise00.html>`_
    is an empirical model of the Earth's atmosphere. It is available as
    a FORTRAN 77 code or as a verson traslated into
    `C by Dominik Borodowski <http://www.brodo.de/english/pub/nrlmsise/>`_.
    Here a PYTHON wrapper has been used.

    Attributes:
      _msis : NRLMSISE-00 python wrapper object handler

    Args:
      location (str): see :func:`init_parameters`
      season (str,optional): months (January, February, etc.)
      doy (int, optional): day of year
    """

    def __init__(
        self, location, season=None, doy=None, use_loc_altitudes=False, environment=None
    ):
        from MCEq.environment.msis00_backend import cNRLMSISE00

        msis_atmospheres = [
            "SouthPole",
            "Karlsruhe",
            "Geneva",
            "Tokyo",
            "GranSasso",
            "TelAviv",
            "KSC",
            "SoudanMine",
            "Tsukuba",
            "LynnLake",
            "PeaceRiver",
            "FtSumner",
        ]
        assert location in msis_atmospheres, (
            f"{location} not available for MSIS00Atmosphere"
        )

        self._msis = cNRLMSISE00()

        # Base class first: it creates self.geom, which init_parameters needs
        # when ``use_loc_altitudes`` moves the observation level.  Calling it
        # afterwards made ``use_loc_altitudes=True`` raise AttributeError.
        EarthsAtmosphere.__init__(self, environment=environment)

        self.init_parameters(location, season, doy, use_loc_altitudes)

    def init_parameters(self, location, season, doy, use_loc_altitudes):
        """Sets location and season in :class:`NRLMSISE-00`.

        Translates location and season into day of year
        and geo coordinates.

        Args:
          location (str): Supported are "SouthPole" and "Karlsruhe"
          season (str): months of the year: January, February, etc.
          use_loc_altitudes (bool): If to use default altitudes from location
        """
        self._msis.set_location(location)
        if season is not None:
            self._msis.set_season(season)
        else:
            self._msis.set_doy(doy)
        self.location, self.season = location, season
        # Clear cached value to force spline recalculation
        self.theta_deg = None
        if use_loc_altitudes:
            info(0, "Using loc altitude", self._msis.alt_surface, "cm")
            # set_h_obs, not a bare attribute write: the geometry caches
            # r_obs and the maximal zenith angle off the observation level.
            self.geom.set_h_obs(self._msis.alt_surface)

    def _clear_cache(self):
        """Clears the density model cache so that density profiles can be recalculated

        It is a private method to wrap the logic of cache cleaning
        """
        self.theta_deg = None

    def update_parameters(self, **kwargs):
        """Updates parameters of the density model

        Args:
          location_coord (tuple of str): (longitude, latitude)
          season (str): months of the year: January, February, etc.
          doy (int): day of the year. 'doy' takes precedence over 'season' if both are set
        """

        self._clear_cache()
        if not kwargs:
            return

        if "location_coord" in kwargs:
            self.set_location_coord(*kwargs.get("location_coord"))

        if "season" in kwargs:
            self.set_season(kwargs.get("season"))

        if "doy" in kwargs:
            self.set_doy(kwargs.get("doy"))
            if "season" in kwargs:
                info(
                    2,
                    "Both 'season' and 'doy' are set in parameter list.\n'doy' takes precedence over 'season'",
                )

    def get_density(self, h_cm):
        """Returns the density of air in g/cm**3.

        Wraps around ctypes calls to the NRLMSISE-00 C library.

        Args:
          h_cm (float): height in cm

        Returns:
          float: density :math:`\\rho(h_{cm})` in g/cm**3
        """
        return self._msis.get_density(h_cm)

    def set_location(self, location):
        """Changes MSIS location by strings defined in _msis_wrapper.

        Args:
          location (str): location as defined in :class:`NRLMSISE-00.`

        """
        self._msis.set_location(location)
        self._clear_cache()

    def set_location_coord(self, longitude, latitude):
        """Changes MSIS location by longitude, latitude in _msis_wrapper

        Args:
          longitude (float): longitude of the location with abs(longitude) <= 180
          latitude (float): latitude of the location with abs(latitude) <= 90

        """
        self._msis.set_location_coord(longitude, latitude)
        self._clear_cache()

    def set_season(self, month):
        """Changes MSIS location by month strings defined in _msis_wrapper.

        Args:
          location (str): month as defined in :class:`NRLMSISE-00.`

        """
        self._msis.set_season(month)
        self._clear_cache()

    def set_doy(self, day_of_year):
        """Changes MSIS season by day of year.

        Args:
          day_of_year (int): 1. Jan.=0, 1.Feb=32

        """
        self._msis.set_doy(day_of_year)
        self._clear_cache()

    def get_temperature(self, h_cm):
        """Returns the temperature of air in K.

        Wraps around ctypes calls to the NRLMSISE-00 C library.

        Args:
          h_cm (float): height in cm

        Returns:
          float: density :math:`T(h_{cm})` in K
        """
        return self._msis.get_temperature(h_cm)


class MSIS00LocationCentered(LocationCenteredMixin, MSIS00Atmosphere):
    """MSIS atmosphere model coupled to an arbitrary detector location.

    This is the general base class for detector-centred atmosphere models.
    It computes the geographic coordinates (latitude, longitude) of the
    shower impact point on Earth's surface for a given zenith and azimuth
    angle using a spherical-Earth ECEF (Earth-Centred, Earth-Fixed) geometry,
    then feeds those coordinates to the NRLMSISE-00 empirical atmosphere model
    so that the density profile reflects the actual atmosphere above the
    shower column.

    Azimuth convention: 0° = geographic North, 90° = East (meteorological /
    clockwise from North).  Zenith: 0° = directly above detector, 90° =
    horizontal, > 90° = upgoing (shower source below horizon) provided
    *max_theta* is set to 180.

    When *set_theta* is called **without** an azimuth angle the model
    computes an azimuth-averaged density profile by sampling *n_azimuth*
    equally-spaced directions around the compass and averaging the MSIS
    density at every height step.  This provides a single representative
    profile that accounts for the latitudinal variation around the detector.

    Subclasses (e.g. :class:`MSIS00IceCubeCentered`,
    :class:`MSIS00KM3NeTCentered`) specialise this for concrete detector
    sites.

    The zenith angle is interpreted as the angle *at the detector*.  Since
    the detector sits below (or above, for elevated sites) the local
    surface, the local zenith angle where the shower axis crosses the
    surface differs from the detector-frame angle.  For a straight line on
    a spherical Earth the impact parameter :math:`r\\,\\sin\\theta` is
    conserved, so the surface-frame angle used for the atmospheric column
    is :math:`\\theta_s = \\arcsin((r_{det}/r_{surf})\\sin\\theta)`.  The
    difference is negligible below ~80° but reaches ~1.4° at the horizon
    for a detector 2 km below the surface (a ~60% slant-depth effect), and
    it maps the grazing window just below the geometric horizon
    (:math:`90° < \\theta \\le 90° + \\arccos(r_{det}/r_{surf})`) onto
    near-side downgoing columns instead of mis-classifying it as upgoing.

    Args:
        detector_coord (tuple): ``(longitude, latitude)`` of the detector
            in degrees.  Longitude in (−180, 180], latitude in [−90, 90].
        depth_m (float): Depth of the detector below the local surface in
            metres (positive value = below surface).
        season (str, optional): Month name (e.g. ``"January"``).  If both
            *season* and *doy* are ``None`` the MSIS default day is used.
        doy (int, optional): Day of year (1–365).  Takes precedence over
            *season* when both are supplied.
        n_azimuth (int): Number of azimuth samples used for azimuth-
            averaging (default 36, i.e. every 10°).
        max_theta (float): Maximum allowed zenith angle in degrees.
            Use 90.0 (default) for downgoing-only models, 180.0 to also
            accept grazing and upgoing angles.
        surface_elevation_m (float): Elevation of the local surface above
            sea level in metres (default 0).  For detectors buried in an
            elevated ice sheet (e.g. IceCube: glacier top at ~2835 m,
            detector 1948 m below it) the shower impacts the glacier
            surface, and the atmospheric column ends there.  The
            observation level of the geometry is set to this elevation for
            downgoing/grazing showers and to sea level for upgoing showers
            (whose column ends on the far side of the Earth).
    """

    #: Preserve max_theta across set_h_obs calls (see EarthsAtmosphere).
    _preserve_max_theta: bool = True

    #: Density profile depends on azimuth (impact point bound to the
    #: detector location); ``set_theta`` accepts ``azimuth_deg``.
    depends_on_azimuth: bool = True

    def _set_backend_location(self, longitude, latitude):
        """MSISE-00 keeps its position inside the C library object."""
        self._msis.set_location_coord(longitude, latitude)

    def __init__(
        self,
        detector_coord,
        depth_m,
        season=None,
        doy=None,
        n_azimuth=36,
        max_theta=90.0,
        surface_elevation_m=0.0,
        environment=None,
    ):
        from MCEq.environment.msis00_backend import cNRLMSISE00

        longitude, latitude = detector_coord

        # Bypass MSIS00Atmosphere.__init__ (which requires a named location
        # string) and set up the C library object directly via coordinates.
        self._msis = cNRLMSISE00()
        self._msis.set_location_coord(longitude, latitude)
        if season is not None:
            self._msis.set_season(season)
        elif doy is not None:
            self._msis.set_doy(doy)

        # Base class first (creates self.geom), then the shared detector
        # geometry -- see LocationCenteredMixin._init_detector_geometry.
        EarthsAtmosphere.__init__(self, environment=environment)
        self._init_detector_geometry(
            longitude,
            latitude,
            depth_m,
            season,
            n_azimuth,
            max_theta,
            surface_elevation_m,
        )

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Density
    # ------------------------------------------------------------------

    def get_density(self, h_cm):
        """Return air density in g/cm³ at height *h_cm*.

        In single-azimuth mode (azimuth was passed to :meth:`set_theta`)
        delegates directly to the MSIS C library whose location is already
        set to the impact point.

        In azimuth-averaging mode (no azimuth passed) computes the mean
        density over the pre-computed :attr:`_azimuth_avg_coords` impact
        points.  The impact points are fixed for a given zenith angle and
        are cached by :meth:`set_theta`, so this method only iterates over
        the cached coordinates without recomputing them.
        """
        if self._azimuth_averaging:
            total = 0.0
            for lat, lon in self._azimuth_avg_coords:
                self._msis.set_location_coord(lon, lat)
                total += self._msis.get_density(h_cm)
            return total / len(self._azimuth_avg_coords)
        return self._msis.get_density(h_cm)

    # ------------------------------------------------------------------
    # Density spline (azimuth-averaging optimisation)
    # ------------------------------------------------------------------

    def calculate_density_spline(self, n_steps=2000):
        """Calculate and store a spline of :math:`\\rho(X)`.

        In single-azimuth mode delegates to the base-class implementation.

        In azimuth-averaging mode uses an azimuth-major loop for efficiency:
        for each of the :attr:`_n_azimuth` pre-computed impact points the
        MSIS location is set **once** and the full height profile is sampled
        in a single vectorised call, so the C library benefits from staying
        at the same latitude/longitude across all height steps.  The 36
        per-azimuth profiles are averaged before spline fitting.
        """
        if not self._azimuth_averaging:
            super().calculate_density_spline(n_steps)
            return

        from time import time

        thrad = self.thrad
        path_length = self.geom.path_len(thrad)
        dl_vec = np.linspace(0, path_length, n_steps)
        h_vec = np.array([self.geom.h(dl, thrad) for dl in dl_vec])

        info(
            5,
            f"Calculating azimuth-averaged spline for zenith {self.theta_deg:4.1f}\u00b0"
            f" ({len(self._azimuth_avg_coords)} directions).",
        )
        now = time()

        # Azimuth-major loop: set MSIS location once per direction, then
        # vectorise over all heights.  This is far more efficient than the
        # height-major order because the C library stays at the same
        # lat/lon across all altitude queries.
        rho_sum = np.zeros(n_steps)
        msis_vec = np.vectorize(self._msis.get_density)
        for lat, lon in self._azimuth_avg_coords:
            self._msis.set_location_coord(lon, lat)
            rho_sum += msis_vec(h_vec)
        rho_vec = rho_sum / len(self._azimuth_avg_coords)

        info(5, f".. took {time() - now:1.2f}s")

        max_den = float(rho_vec[0])

        # Scalar ``geom.h``, as in the base tail this branch replaces. These
        # are the same values as ``h_vec[2:][::-1]`` -- same function, same
        # arguments -- so the recomputation is redundant, but removing it is a
        # change of its own and not part of this extraction.
        h_intp = [self.geom.h(dl, thrad) for dl in reversed(dl_vec[2:])]
        fit_column_splines(self, rho_vec, dl_vec, h_intp, max_den=max_den)


class MSIS00IceCubeCentered(MSIS00LocationCentered):
    """Atmosphere model centred on the IceCube detector at South Pole.

    Specialisation of :class:`MSIS00LocationCentered` for IceCube.  The
    detector sits 1948 m below the top of the Antarctic ice sheet, whose
    surface at the South Pole is at ~2835 m elevation — the detector
    centre is therefore ~887 m *above* sea level, and downgoing showers
    impact the glacier surface at 2835 m.  Upgoing angles up to 180° are
    supported.

    The public interface is identical to the original implementation so
    that existing code continues to work unchanged.  The ``location``
    argument is accepted for backward compatibility but is always
    overridden to ``"SouthPole"``.

    Args:
      location (str): Ignored (kept for backward compatibility).
      season (str): Month name, e.g. ``"January"``.
    """

    def __init__(self, location, season, environment=None):
        if location != "SouthPole":
            info(2, "location forced to the South Pole")
        super().__init__(
            detector_coord=(0.0, -90.0),
            depth_m=1948.0,
            season=season,
            max_theta=180.0,
            surface_elevation_m=2835.0,
            environment=environment,
        )

    def _latitude(self, det_zenith_deg):
        """Return the geographic latitude of the shower impact point.

        Backward-compatible wrapper around :meth:`_impact_point`.
        The azimuth is irrelevant at the South Pole (all directions are
        equivalent), so 0° is passed.

        Args:
          det_zenith_deg (float): zenith angle at detector in degrees

        Returns:
          float: latitude of the impact point in degrees
        """
        lat, _lon = self._impact_point(det_zenith_deg, 0.0)
        return lat


class MSIS00KM3NeTCentered(MSIS00LocationCentered):
    """MSIS atmosphere model coupled to a KM3NeT detector location.

    Convenience subclass of :class:`MSIS00LocationCentered` for the two
    KM3NeT deep-sea neutrino telescope sites in the Mediterranean:

    * **ORCA** (Oscillation Research with Cosmics in the Abyss) —
      offshore Toulon, France (6.033°E, 42.803°N, ~2450 m depth).
    * **ARCA** (Astroparticle Research with Cosmics in the Abyss) —
      offshore Capo Passero, Sicily (16.1°E, 36.267°N, ~3500 m depth).

    Both sites are at non-polar latitudes so the azimuth angle matters:
    the same zenith angle can probe very different atmospheric columns
    depending on direction.  Upgoing neutrino angles up to 180° are
    supported (``max_theta=180``).

    When *set_theta* is called without an azimuth, the density profile is
    averaged over *n_azimuth* equally-spaced azimuth directions (default
    36), providing an isotropically-averaged column for the given zenith.

    Args:
        detector (str): ``"ORCA"`` or ``"ARCA"``.
        season (str, optional): Month name (e.g. ``"January"``).
        doy (int, optional): Day of year (1–365).
        n_azimuth (int): Azimuth steps for averaging (default 36).
    """

    def __init__(self, detector, season=None, doy=None, n_azimuth=36, environment=None):
        if detector not in KM3NET_DETECTORS:
            raise ValueError(
                f"Unknown KM3NeT detector '{detector}'. "
                f"Choose from {list(KM3NET_DETECTORS.keys())}."
            )
        det = KM3NET_DETECTORS[detector]
        super().__init__(
            detector_coord=(det["longitude"], det["latitude"]),
            depth_m=det["depth_m"],
            season=season,
            doy=doy,
            n_azimuth=n_azimuth,
            max_theta=180.0,
            environment=environment,
        )
        self._detector_name = detector


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 4))
    plt.title("NRLMSISE-00 atmospheres")
    msis_atmospheres = [
        ("SouthPole", "January"),
        ("Karlsruhe", "January"),
        ("Geneva", "January"),
        ("Tokyo", "January"),
        ("GranSasso", "January"),
        ("TelAviv", "January"),
        ("KSC", "January"),
        ("SoudanMine", "January"),
        ("Tsukuba", "January"),
        ("LynnLake", "January"),
        ("PeaceRiver", "January"),
        ("FtSumner", "January"),
    ]
    msis_surf_100 = []
    for loc, season in msis_atmospheres:
        msis_obj = MSIS00Atmosphere(loc, season)
        msis_obj.set_theta(0.0)
        x_vec = np.linspace(0, msis_obj.max_X, 5000)
        plt.plot(x_vec, 1 / msis_obj.r_X2rho(x_vec), lw=1.5, label=f"{loc}")
        msis_surf_100.append((msis_obj.max_X, 1.0 / msis_obj.r_X2rho(100.0)))
    plt.ylabel(r"Density $\rho$ (g/cm$^3$)")
    plt.xlabel(r"Depth (g/cm$^2$)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
