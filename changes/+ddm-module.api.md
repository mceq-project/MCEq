The Data-Driven Model is importable from its v1.4.2 paths again,
``MCEq.ddm`` and ``MCEq.ddm_utils``; the ``MCEq.models`` package is gone.
``DDMSplineDB`` and ``_DDMChannel`` are removed: ``DataDrivenModel.channels``
maps ``(projectile, secondary)`` to the beam-energy-sorted spline entries, and
``DataDrivenModel.get_entry`` replaces ``spline_db.get_entry``.
``DataDrivenModel.ddm_matrices`` now also returns the isospin-partner channels
that ``MCEqRun.inject_ddm`` installs.
