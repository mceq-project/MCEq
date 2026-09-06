"""The driver layer: the MCEqRun facade and its collaborators.

Plan section 1's ``driver/``, landed fully in Phase 6. ``mceq_run`` carries
the facade class (``MCEq.core`` is a permanent re-export shim over it, plan
§8.6); ``system`` the ``CascadeSystem``, ``initial_state`` the
``InitialState``, ``results`` the batch result + 2D Hankel back-transform
and ``batch`` the batch/full-sky orchestration (all moved verbatim from
``MCEq/core.py``).
"""
