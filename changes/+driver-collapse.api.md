``MCEqRun`` is defined in ``MCEq.core`` again, as in v1.4.2, and the
``MCEq.driver`` package is gone. The batched solves are in ``MCEq.batch``
together with ``MCEqBatchResult`` and ``build_phi0_with_cutoff`` (previously
``MCEq.environment.geomagnetic.cutoff.build_phi0_with_cutoff``). Removed during
the 2.0 development cycle: ``MCEqRun.initial_state(components)`` (compose
columns with ``set_single_primary_particle`` / ``set_initial_spectrum`` and
``get_initial_state``) and the ``skip_integration_path`` keyword of ``solve``
(use ``solve_from_integration_path``). ``get_solution(return_as=None)`` now reads
``config.return_as`` at call time instead of at import time.
