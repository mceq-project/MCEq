Fixed the five mechanical bug-ledger entries sanctioned by the Phase-4 ruling.
Three are user-visible: ``Interactions.set_mod_pprod`` with a neutron primary
now writes the isospin copies under the true mirror-partner keys (a ``K+``
modification on a neutron primary applies its factor once, not twice);
``InteractionCrossSections.get_cs`` returns a zero vector — never the scalar
``0.0`` — when no model matches the family; ``MCEqParticle.inverse_decay_length``
returns a length-``d`` vector of ``inf`` (and its docstring now says so) where
it previously raised or returned a scalar for a zero-width particle. The other
two (``DDMSplineDB`` caching per instance instead of on the class, and
``MCEqParticle.prod_cross_section(mbarn=True)`` dividing by the single shared
unit constant, a one-ulp move) change no shipped-database result.
