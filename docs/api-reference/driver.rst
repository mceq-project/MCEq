.. _driver:

************************************************
driver (:mod:`MCEq.driver`)
************************************************
.. currentmodule:: MCEq.driver


The driver layer's internals, documented where they live. The
:class:`MCEqRun` façade, :class:`MCEqBatchResult`, and the batch
functions the façade binds (``solve_batch``, ``solve_fullsky``,
``get_solution``) are documented on the permanent
:ref:`core` page, which ``MCEq.core`` re-exports. What is documented
here is the run-coupled path machinery and the counting / Z-factor
observables, none of which are import names on ``MCEq.core``:

Reference/API
=============
.. automodule:: MCEq.driver.paths
   :members:

.. automodule:: MCEq.driver.observables
   :members:
