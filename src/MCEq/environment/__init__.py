"""Atmospheres, Earth geometry and geomagnetic cutoffs.

The environment layer sits beside the cascade stack rather than in it: it knows
about air density, column depth and where a shower enters the atmosphere, and
nothing about particles, interactions or the solver.

Deliberately empty of re-exports. :mod:`MCEq.environment.registry` resolves
model names to classes through dotted strings so that selecting an MSIS00 model
does not import the MSIS21 tree or vice versa; an eager ``__init__`` would
defeat that. Import the concrete module you need.

``MCEq.geometry.*`` provides compatibility re-exports over this package.
"""
