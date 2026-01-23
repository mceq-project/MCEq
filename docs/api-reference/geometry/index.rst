.. _geometry:

*******************************
geometry (:mod:`MCEq.geometry`)
*******************************

.. currentmodule:: MCEq.geometry

In MCEq, geometry is everything related to the medium in which the particle cascade develops. 
The very basic geometrical functions for the polar coordinate system of the 
Earth - no it’s not flat, but just azimuth symmetric - are located in :mod:`MCEq.geometry.geometry`.
The density parameterizations and interfaces are in :mod:`MCEq.geometry.density_profiles`.

Submodules
==========
.. toctree::
  :maxdepth: 1
  :glob:

  atmosphere_parameters
  corsikaatm/index
  density_profiles
  geometry
  nrlmsise00/index
  nrlmsise00_mceq


Reference/API
=============
.. automodapi:: MCEq.geometry
    :no-inheritance-diagram:
