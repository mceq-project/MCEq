from ctypes import byref, c_double, c_int, pointer

import MCEq.environment._ext.nrlmsise00 as cmsis
from MCEq.environment.parameters import (
    DAY_TIMES_SEC,
    DEFAULT_AP,
    DEFAULT_F107,
    DEFAULT_F107A,
    LOCATIONS,
    MONTH_TO_DAY_OF_YEAR,
)
from MCEq.misc import info


class NRLMSISE00Base:
    def __init__(self):
        # Cache altitude value of last call
        self.last_alt = None

        self.inp = cmsis.nrlmsise_input()
        self.output = cmsis.nrlmsise_output()
        self.flags = cmsis.nrlmsise_flags()
        # Use imported constants
        self.month2doy = MONTH_TO_DAY_OF_YEAR
        self.locations = LOCATIONS
        self.daytimes = DAY_TIMES_SEC

        self.current_location = "SouthPole"
        self.init_default_values()

    def surface_vert_depth(self, loc="SouthPole", month="June"):
        self.set_location("SouthPole")
        self.set_season("June")

    def height2depth(self, altitude_cm):
        from scipy.integrate import quad

        return quad(self.get_density, altitude_cm, 112.8 * 1e5, epsrel=0.001)[0]

    def _retrieve_result(self, *args, **kwargs):
        """Calls NRLMSISE library's main function"""
        raise Exception("Not implemented for the base class")

    def get_temperature(self, altitude_cm):
        """Returns temperature in K"""
        self._retrieve_result(altitude_cm)
        return self.output.t[1]

    def get_density(self, altitude_cm):
        """Returns density in g/cm^3"""
        self._retrieve_result(altitude_cm)
        return self.output.d[5]


class cNRLMSISE00(NRLMSISE00Base):
    def init_default_values(self):
        """Sets default to June at South Pole"""

        self.inp.doy = c_int(self.month2doy["June"])  # Day of year
        self.inp.year = c_int(0)  # No effect
        self.inp.sec = c_double(self.daytimes["day"])  # 12:00
        self.inp.alt = c_double(self.locations[self.current_location][2])
        self.inp.g_lat = c_double(self.locations[self.current_location][1])
        self.inp.g_long = c_double(self.locations[self.current_location][0])
        self._update_lst()
        # Do not touch this except you know what you are doing
        # Use imported constants
        self.inp.f107A = c_double(DEFAULT_F107A)
        self.inp.f107 = c_double(DEFAULT_F107)
        self.inp.ap = c_double(DEFAULT_AP)
        self.inp.ap_a = pointer(cmsis.ap_array())
        self.alt_surface = self.locations[self.current_location][2]

        self.flags.switches[0] = c_int(0)
        for i in range(1, 24):
            self.flags.switches[i] = c_int(1)

    def set_location(self, tag):
        if tag == "SanGrasso":
            raise ValueError(
                "Location 'SanGrasso' does not exist — did you mean 'GranSasso'? "
                "(Gran Sasso d'Italia is in Abruzzo, not San Grasso, wherever that is.)"
            )
        if tag not in list(self.locations):
            raise Exception(
                f"NRLMSISE00::set_location(): Unknown location tag '{tag}'."
            )

        self.inp.alt = c_double(self.locations[tag][2])
        self.set_location_coord(*self.locations[tag][:2])
        self.current_location = tag
        self.alt_surface = self.locations[self.current_location][2]

    def set_location_coord(self, longitude, latitude):
        info(5, f"long={longitude:5.2f}, lat={latitude:5.2f}")
        if abs(latitude) > 90 or abs(longitude) > 180:
            raise Exception("NRLMSISE00::set_location_coord(): Invalid inp.")
        self.inp.g_lat = c_double(latitude)
        self.inp.g_long = c_double(longitude)
        self._update_lst()
        self._invalidate()

    def _update_lst(self):
        """Keep local apparent solar time consistent with the longitude.

        NRLMSISE-00 takes ``lst`` as a separate input from ``sec`` and
        ``g_long``. Its own documentation (``nrlmsise-00.h``, NOTES ON INPUT
        VARIABLES) says the three "are used independently in the model" but
        that "for the most physically realistic calculation these three
        variables should be consistent (lst=sec/3600 + g_long/15)" -- a
        recommendation, not a hard requirement, and it notes that Equation of
        Time departures are legitimate. The model uses ``lst`` for the diurnal
        and semidiurnal tides.

        Before this was called from :meth:`set_location_coord`, ``lst`` was
        written once at construction and never again, so every site was
        evaluated at the *default* location's solar time -- 12.0, since the
        default is the South Pole at longitude 0.

        Measured over the 12 shipped ``LOCATIONS`` x 12 months on this host,
        comparing against ``lst`` pinned at 12.0:

        * up to 9.34 h of solar time (Tsukuba, longitude 140.1);
        * 0.41 % at sea level in the worst case (LynnLake, January), above
          0.25 % at 14 of the 144 site-months -- consistent with ``max_X``
          moving by at most 4.1e-03, since the column is low-altitude
          dominated;
        * 22.97 % at exactly 95 km (LynnLake, July), and up to 46 % near
          107 km (KSC, July), where the tide is strongest.

        It was exactly zero at the South Pole, which is why the flagship MSIS00
        use case never showed it.

        **What time of day this now means.** ``sec`` is seconds of day in *UT*
        and is fixed at 43200 with no setter, so consistency is restored by
        moving ``lst``, i.e. every site is evaluated at 12:00 UT
        (``lst = 12 + longitude/15``), not at local noon. That is a choice: the
        alternative is to hold ``lst = 12`` and derive
        ``sec = 43200 - 240*longitude``, keeping local noon everywhere and the
        ``DAY_TIMES_SEC`` day/night naming. This one was taken because it is
        what MSIS 2.1 already does (``nrlmsis/globe.py`` computes
        ``lst = utsec/3600 + lon/15``), so the two backends now describe the
        same instant rather than two different ones.
        """
        self.inp.lst = c_double(
            self.inp.sec.value / 3600.0 + self.inp.g_long.value / 15.0
        )

    def set_season(self, tag):
        if tag not in self.month2doy:
            raise Exception("NRLMSISE00::set_location(): Unknown season tag.")
        info(5, "Season", tag, "doy=", self.month2doy[tag])
        self.inp.doy = self.month2doy[tag]
        self._invalidate()

    def set_doy(self, doy):
        if doy < 0 or doy > 365:
            raise Exception("NRLMSISE00::set_doy(): Day of year out of range.")
        info(5, "day of year", doy)
        self.inp.doy = c_int(doy)
        self._invalidate()

    def _invalidate(self):
        """Drop the altitude memo of :meth:`_retrieve_result`.

        The memo keys on altitude alone, so every setter that changes latitude,
        longitude or day of year has to clear it: without this a
        ``get_density(h)`` repeated at the same altitude returns the previous
        location's or season's value (measured 12.5 % low for a doy change from
        1 to 200 at 20 km, and it is what corrupts the azimuth-averaged
        ``MSIS00LocationCentered.get_density``).
        """
        self.last_alt = None

    def _retrieve_result(self, altitude_cm):
        if self.last_alt == altitude_cm:
            return

        inp = self.inp
        inp.alt = c_double(altitude_cm / 1e5)
        cmsis.msis.gtd7_py(
            inp.year,
            inp.doy,
            inp.sec,
            inp.alt,
            inp.g_lat,
            inp.g_long,
            inp.lst,
            inp.f107A,
            inp.f107,
            inp.ap,
            inp.ap_a,
            byref(self.flags),
            byref(self.output),
        )

        self.last_alt = altitude_cm


def test():
    import matplotlib.pyplot as plt
    import numpy as np

    msis = cNRLMSISE00()
    den = np.vectorize(msis.get_density)

    plt.figure(figsize=(16, 5))
    plt.suptitle("NRLMSISE-00")

    h_vec = np.linspace(0, 112.8 * 1e5, 500)
    msis.set_season("January")
    msis.set_location("SouthPole")
    den_sp_jan = den(h_vec)

    msis.set_season("January")
    msis.set_location("Karlsruhe")
    den_ka_jan = den(h_vec)

    plt.subplot(131)
    plt.semilogy(h_vec / 1e5, den_sp_jan, label="MSIS00: SP Jan.")
    plt.semilogy(h_vec / 1e5, den_ka_jan, label="MSIS00: KA Jan.")
    plt.legend()
    plt.xlabel("vertical height in km")
    plt.ylabel(r"density $\rho(h)$ in g/cm$^3$")

    plt.subplot(132)
    plt.plot(h_vec / 1e5, den_ka_jan / den_sp_jan, label="MSIS00: KA/SP")
    plt.xlabel("vertical height in km")
    plt.ylabel(r"density ratio")
    plt.legend(loc="upper left")

    plt.subplot(133)
    msis.set_location("SouthPole")
    for i in range(360 / 30):
        msis.inp.doy = i * 30
        plt.plot(h_vec / 1e5, den(h_vec) / den_sp_jan, label=str(i + 1))
    plt.legend(ncol=2, loc=3)
    plt.title("MSIS00: SouthPole")
    plt.xlabel("vertical height in km")
    plt.ylabel(r"$\rho$(Month) / $\rho$(January)")
    plt.ylim(ymin=0.6)
    plt.tight_layout()

    plt.figure(figsize=(6, 5))
    h2d = np.vectorize(msis.height2depth)
    plt.semilogy(h_vec / 1e5, h2d(h_vec))
    plt.ylabel(r"Slant depth X [g/cm$^2$]")
    plt.xlabel(r"Atmospheric height $h$ [km]")
    plt.subplots_adjust(left=0.15, bottom=0.11)
    plt.show()


if __name__ == "__main__":
    test()
