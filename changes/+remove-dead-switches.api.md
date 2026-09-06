Deleted ``config.adv_set["disable_leading_mesons"]`` and the leading-meson
inversion in ``Interactions.get_matrix``, and deleted the unreachable
unflavoured-coupling branch (eta/omega/phi) in ``_set_mod_pprod``. Both were
dead since the interaction index gained helicity: the veto guard
``abs(child) < 2000`` raises ``TypeError`` on the ``(pdg, helicity)`` tuple
before its hard-coded ``ie = 50`` window is ever evaluated, and the coupling
guard tested bare PDG ints against ``(pdg, helicity)`` keys, so it never fired
for any shipped database. No behaviour changes; setting the knob in user code
now raises ``KeyError`` instead of being silently accepted.
