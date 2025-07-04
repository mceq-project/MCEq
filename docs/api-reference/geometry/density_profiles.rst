.. _data:
******************************************************************
density_profiles (:mod:`MCEq.geometry.density_profiles`)
******************************************************************
.. currentmodule:: MCEq.geometry.density_profiles


This module includes classes and functions modeling the Earth’s atmosphere. 
Currently, two different types models are supported:

#.  Linsley-type/CORSIKA-style parameterization
#.  Numerical atmosphere via external routine (NRLMSISE-00)

Both implementations have to inherit from the abstract class :class:`MCEq.geometry.density_profiles.EarthsAtmosphere`,
which provides the functions for other parts of the program. In particular the function :func:`MCEq.geometry.density_profiles.EarthsAtmosphere.get_density()`.

Typical interaction:

.. code-block::

   atm_object = CorsikaAtmosphere("BK_USStd")
   atm_object.set_theta(90)
   print(density at X=100, atm_object.X2rho(100.))

The class :class:`MCEq.core.MCEqRun` will only the following routines:

* EarthsAtmosphere.set_theta()
* EarthsAtmosphere.r_X2rho()

If you are extending this module make sure to provide these functions without breaking compatibility.

Reference/API
=============
.. automodapi:: MCEq.geometry.density_profiles
  :inherited-members:


