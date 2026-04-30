Reverted the ``MCEqParticle.inel_cross_section()`` →
``prod_cross_section()`` rename from PR #159. The interaction-cross-section
database in MCEq 2 stores **inelastic** cross-sections again, so the v1
name is restored: use ``MCEqParticle.inel_cross_section()``. Code that was
updated to ``prod_cross_section()`` for a v1.4.x release must switch back.
