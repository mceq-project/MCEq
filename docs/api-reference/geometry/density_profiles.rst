.. _data:
******************************************************************
density_profiles (:mod:`MCEq.geometry.density_profiles`)
******************************************************************
.. currentmodule:: MCEq.geometry.density_profiles


This module includes classes and functions modeling the Earth’s atmosphere. 
Currently, three different types models are supported:

#.  Linsley-type/CORSIKA-style parameterization
#.  Numerical atmosphere via external routine (NRLMSISE-00 / NRLMSIS 2.1)
#.  Tabulated vertical profiles read from a CSV table
    (:class:`MCEq.geometry.density_profiles.TabulatedAtmosphere`)

Both implementations have to inherit from the abstract class :class:`MCEq.geometry.density_profiles.EarthsAtmosphere`,
which provides the functions for other parts of the program. In particular the function :func:`MCEq.geometry.density_profiles.EarthsAtmosphere.get_density()`.

Typical interaction:

.. code-block::

   atm_object = CorsikaAtmosphere("BK_USStd")
   atm_object.set_theta(90)
   print(density at X=100, atm_object.X2rho(100.))

The class :class:`MCEq.core.MCEqRun` uses only a small part of the interface:

* ``EarthsAtmosphere.set_theta()``, called with an extra ``azimuth_deg`` argument
  when the model sets ``depends_on_azimuth``
* ``EarthsAtmosphere.r_X2rho()`` and ``EarthsAtmosphere.max_X``
* the attributes ``theta_deg``, ``_current_azimuth_deg`` and ``depends_on_azimuth``,
  which :func:`MCEq.core.MCEqRun.set_zenith_azimuth` reads to cache the current
  direction, and ``location``, used to detect sites for the geomagnetic cutoff

If you are extending this module make sure to provide these without breaking compatibility.

Reference/API
=============
.. automodapi:: MCEq.geometry.density_profiles
  :inherited-members:


