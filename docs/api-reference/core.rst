.. _core:
************************************************
core (:mod:`MCEq.core`)
************************************************
.. currentmodule:: MCEq.core


This module contains the main program features. Instantiating :class:`MCEq.core.MCEqRun` will 
initialize the data structures and particle tables, create and fill the interaction and decay 
matrix and check if all information for the calculation of inclusive fluxes in the atmosphere is available.
The batched solves and their result container are in :mod:`MCEq.batch`.

Reference/API
=============
.. automodapi:: MCEq.core
  :inherited-members:

.. automodapi:: MCEq.batch

.. automodule:: MCEq.initial_state
   :members:
