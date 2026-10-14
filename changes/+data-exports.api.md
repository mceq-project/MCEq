``MCEq.data`` exports the v1.4.2 names again: ``HDF5Backend``, ``Interactions``,
``Decays``, ``InteractionCrossSections``, ``ContinuousLosses`` and
``equivalences``. The other names it re-exported during the 2.0 development
cycle (``EnergyGrid``, ``family_of``, ``apply_equivalences``,
``reverse_equivalences``, the ``blend_*`` helpers, ``isfile``, ``join``, ``info``)
are gone; ``energy_grid``, ``gen_xmat`` and ``normalize_hadronic_model_name``
live in ``MCEq.misc`` as in v1.4.2.
