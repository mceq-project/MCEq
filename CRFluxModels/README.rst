`CRFluxModels` --- models of the high-energy cosmic ray flux
------------------------------------------------------------

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
awy without any attempt to optimize for speed or elegance. If you find it slow and improve
something, feel free to send a pull request or just branch/fork and let me know.

The module is used as a submodule in the `Matrix Cascade Equation <https://github.com/afedynitch/MCEq>`_ 
app.

Documentation
-------------

Please follow `this link to the documentation <http://crfluxmodels.readthedocs.org/en/latest/index.html#>`_.

Installation
------------

The module was developed using 
`Python2.7 <http://python.org>`_ and `NumPy <http://www.numpy.org>`_. It doesn't use any kind of fancy functionality.
It should therefore work with all versions. Some models require `SciPy <http://www.scipy.org>`_.

Example
-------
Run `python CRFluxModels.py` from a shell to generate a set of standard plots.

