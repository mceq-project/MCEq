Physical models
===============

The cascade equation can be applied whenever particles interact and decay. The
main purpose of the current program version is to calculate inclusive fluxes 
of leptons (:math:`\mu,\ \nu_{\mu},\ \nu_e\ \text{and}\ \nu_{\tau}`) in the Earth's
atmosphere. The calculation requires a spherical model of the Earth's atmosphere
which on hand is based on its :mod:`MCEq.geometry` and on the other hand parameterizations
or numerical models of the :mod:`MCEq.geometry.density_profiles`.

Alternatiely, this code could be used for calculations of the high energy hadron 
and lepton flux in astrophysical environments, where the gas density and the 
(re-)interaction probability are very low. Prediction of detailed photon spectra
is possible but additional extensions.

At very high energies, i.e. beyond 100 TeV, the decays of very short-lived particles
become an important contribution to the flux. However, heavy-flavor production at these
energies is not well known and it can not be consitently predicted within theoretical
frameworks. Some of the default interaction models (``SIBYLL-2.3`` or ``SIBYLL-2.3c``)
contain models of charmed particle production but they represent only hypotheses based
on experimental data and phenomenology. Other :mod:`MCEq.charm_models` exist, such as 
the :class:`MCEq.charm_models.MRS_charm`, which can be coupled with any other interaction
model for normal hadron production. In practice any kind of model which predicts a
:math:`x` distribution can be employed in this code as extension of the 
:mod:`MCEq.charm_models` module.  

----------

.. automodule:: MCEq.geometry.density_profiles
   :members:

----------

.. automodule:: MCEq.geometry.geometry
   :members: 

--------------------

.. automodule:: MCEq.geometry.msis_wrapper
   :members:

--------------------

.. automodule:: MCEq.charm_models
   :members:
