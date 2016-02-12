.. CRFluxModels documentation master file, created by
   sphinx-quickstart on Sun Nov 30 18:10:29 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation for :mod:`CRFluxModels`
=====================================

Historically, this module was part of the research code for a paper 
`A. Fedynitch, J. Becker Tjus, and P. Desiati, Phys. Rev. D 86, 114024 
(2012) <http://journals.aps.org/prd/abstract/10.1103/PhysRevD.86.114024>`_, 
where we compared the effects of different primary Cosmic Ray Flux models - therefore
the name. Later the module became an integral part of my research tools and has been
recently cleaned up and documented. It

- implements numerical models of high energy cosmic ray fluxes, 
- converts from all-particle into all-nucleon flux,
- includes some convenience functions for semi-analytical atmospheric lepton flux calculations.

Since this module never was a performance bottle-neck, the formulae are written in a human-readable 
without any attempt to optimize for speed or elegance. If you find it slow and improve
something, feel free to send a pull request or just branch/fork and let me know.


.. toctree::
   :maxdepth: 2

````````````````````
Module documentation
````````````````````

.. automodule:: CRFluxModels
   :members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

