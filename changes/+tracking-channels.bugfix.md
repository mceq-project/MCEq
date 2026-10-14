Tracking particles no longer double-count. ``set_interaction_model(...,
update_particle_list=False)`` appended every interaction-tracked channel a
second time, doubling the ``prres_*`` and all other ``from_interactions`` tracked
fluxes, and raised when the model changed; both are fixed. A tracking particle
also no longer shares the full yields of its species until the next
``regenerate_matrices``: it is transported like its species (cross section,
lifetime, losses) but produces only itself, so tracking an interacting species
(muons with ``enable_em``, or hadrons) no longer adds real secondaries. Default
runs without ``enable_em`` are unchanged.
