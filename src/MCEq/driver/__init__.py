"""The driver layer: the MCEqRun facade and its collaborators.

``mceq_run`` carries the facade class (``MCEq.core`` is a permanent
re-export shim over it, §8.6); ``system`` the ``CascadeSystem``,
``initial_state`` the ``InitialState``, ``results`` the batch result + 2D
Hankel back-transform, ``batch`` the batch/full-sky orchestration, and
``paths`` the run-coupled path machinery.
"""
