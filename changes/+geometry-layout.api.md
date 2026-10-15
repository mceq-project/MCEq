The atmospheres are in ``MCEq.geometry`` again under their v1.4.2 module names
(``density_profiles``, ``atmosphere_parameters``, ``nrlmsise00_mceq``,
``geometry``, ``nrlmsise00``, ``corsikaatm``); the ``MCEq.environment`` package is
gone. The NRLMSIS 2.1 atmospheres are in ``MCEq.geometry.msis21``, the material
target in ``MCEq.geometry.target`` (also importable from ``density_profiles``)
and the geomagnetic cutoff in ``MCEq.geometry.cutoff``. All MSIS atmospheres
share ``MSISAtmosphere`` and ``MSISLocationCentered`` with a backend object;
``MSIS00Atmosphere`` without ``season`` or ``doy`` now defaults to June instead
of failing. ``MCEq.core.DENSITY_MODELS`` replaces the registry module.
``AIRSAtmosphere`` takes its tables from ``table_path`` (no default location).
