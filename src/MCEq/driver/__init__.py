"""The driver layer: the MCEqRun facade and its collaborators.

Plan section 1's ``driver/``, opened in Phase 6. ``results`` holds the batch
result object and the 2D Hankel back-transform (moved verbatim from
``MCEq.hankel``, which stays as a compat shim); the facade split (``mceq_run``,
``system``, ``initial_state``, ``batch``) continues in later commits.
"""
