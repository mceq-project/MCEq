The particle species live in ``MCEq.particlemanager`` again, as in v1.4.2; the
``MCEq.species`` package is gone and the bin-averaged rate helpers are in
``MCEq.bin_average``. ``MCEqParticle.inel_cross_section`` is restored as the
v1.4.2 name of ``prod_cross_section``. ``MCEqParticle`` objects compare by
identity instead of by name. Removed: ``MCEqParticle.is_secondary`` and
``is_child`` (use ``sec in proj.hadr_yields`` / ``sec in proj.decay_dists``),
``track_decays``, ``track_interactions``, ``init_custom_particle_data``,
``set_hadronic_channels`` and ``set_decay_channels`` (the ``ParticleManager``
wires the channels), and ``ParticleManager.set_cross_sections_db``, ``keys``
and ``nspec``. ``hadr_secondaries`` and ``children`` are read-only views of
``hadr_yields`` and ``decay_dists``. Photons, e± and π⁰ are left out of the
species list by the ``ParticleManager`` unless ``config.enable_em`` is set
(``misc.strip_em_species`` and ``misc.EM_PDG_IDS`` are removed).
