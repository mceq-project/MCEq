.. _densities:

Geometry package
================

In MCEq, geometry is everything related to the medium in which the particle
cascade develops. The very basic geometrical functions for the polar coordinate
system of the Earth - no it's not flat, but just azimuth symmetric - are located
in :mod:`MCEq.geometry.geometry`. The density parameterizations and interfaces
are in :mod:`MCEq.geometry.density_profiles`


:mod:`MCEq.geometry.density_profiles`
.....................................

This module includes classes and functions modeling the Earth's atmosphere.
Currently, two different types models are supported:

- Linsley-type/CORSIKA-style parameterization
- Numerical atmosphere via external routine (NRLMSISE-00)

Both implementations have to inherit from the abstract class
:class:`EarthsAtmosphere`, which provides the functions for other parts of
the program. In particular the function :func:`EarthsAtmosphere.get_density`

Typical interaction::

      $ atm_object = CorsikaAtmosphere("BK_USStd")
      $ atm_object.set_theta(90)
      $ print 'density at X=100', atm_object.X2rho(100.)

The class :class:`MCEqRun` will only the following routines::
    - :func:`EarthsAtmosphere.set_theta`,
    - :func:`EarthsAtmosphere.r_X2rho`.

If you are extending this module make sure to provide these
functions without breaking compatibility.

Example:
  An example can be run by executing the module::

      $ python MCEq/atmospheres.py

.. automodule:: MCEq.geometry.density_profiles
   :members:
   :no-special-members:


:mod:`MCEq.geometry.geometry`
.............................

The module contains the geometry for an azimuth symmetric Earth.

.. automodule:: MCEq.geometry.geometry
   :members:
   :no-special-members:

:mod:`MCEq.geometry.nrlmsise00`
...............................

CTypes interface to the C-version of the NRLMSISE-00 code, originally
developed by `Picone et al. <https://doi.org/10.1029/2002JA009430>`_.
The C-translation is by `Dominik Brodowski <https://www.brodo.de/space/nrlmsise/index.html>_`.


.. automodule:: MCEq.geometry.nrlmsise00
   :members:
   :no-special-members:
   
:mod:`MCEq.geometry.corsikaatm`
...............................

This set of functions are C implementations of the piecewise defined exponential
profiles as used in CORSIKA. An efficient implementation is difficult in plain
numpy. 

.. automodule:: MCEq.geometry.corsikaatm
   :members:
   :no-special-members: