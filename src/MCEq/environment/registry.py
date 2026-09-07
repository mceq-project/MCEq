"""Name to atmosphere class: the table behind ``MCEqRun.set_density_model``.

The dispatch used to be a list literal and an ``elif`` ladder inside that
method, which had two consequences worth removing. A name could be present in
one and absent from the other -- ``MSIS00_KM3NeT`` was, bug B15 -- and the only
way to ask the code which models exist was to parse the literal out of the
method's source with :mod:`ast`, which ``tests/golden/gen_paths.py`` did.

Entries are dotted ``module:qualname`` strings resolved on first use, so
importing this module costs nothing and selecting an MSIS00 model does not drag
in the MSIS21 tree or vice versa. That laziness was a property of the ``elif``
ladder (it ran ``import MCEq.geometry.density_profiles`` inside the method) and
is kept deliberately.


"""

from __future__ import annotations

from importlib import import_module

#: Public model name -> ``"module:qualname"``. Order is the order
#: :func:`available_models` reports and the error message lists, so it is the
#: user-visible ordering; keep the MSIS families together and grouped by
#: backend generation, as the original literal did.
DENSITY_MODELS: dict[str, str] = {
    "MSIS00": "MCEq.environment.msis00:MSIS00Atmosphere",
    "MSIS00_IC": "MCEq.environment.msis00:MSIS00IceCubeCentered",
    "MSIS00_KM3NeT": "MCEq.environment.msis00:MSIS00KM3NeTCentered",
    "MSIS21": "MCEq.environment.msis21:MSIS21Atmosphere",
    "MSIS21_IC": "MCEq.environment.msis21:MSIS21IceCubeCentered",
    "MSIS21_KM3NeT": "MCEq.environment.msis21:MSIS21KM3NeTCentered",
    "CORSIKA": "MCEq.environment.corsika:CorsikaAtmosphere",
    "AIRS": "MCEq.environment.tabulated:AIRSAtmosphere",
    "Isothermal": "MCEq.environment.isothermal:IsothermalAtmosphere",
    "GeneralizedTarget": "MCEq.environment.target:GeneralizedTarget",
}

#: Models whose constructor takes no configuration. ``set_density_model``
#: discarded ``model_config`` for these, and still does.
_NO_CONFIG = frozenset({"GeneralizedTarget"})


def available_models() -> list[str]:
    """The model names :func:`build` accepts, in user-visible order."""
    return list(DENSITY_MODELS)


def density_model_class(name: str) -> type:
    """Resolve one registry entry to its class, importing it on first use."""
    target = DENSITY_MODELS[name]
    module_name, _, qualname = target.partition(":")
    return getattr(import_module(module_name), qualname)


def build(name: str, model_config, environment=None):
    """Instantiate the atmosphere *name* with *model_config*.

    ``model_config`` is splatted into the constructor, except for the models in
    :data:`_NO_CONFIG`, which are constructed with no arguments -- the
    behaviour of the ``elif`` ladder this replaces.

    ``environment`` is the caller's ``environment`` settings group, forwarded
    as a keyword to every model: a driver-built atmosphere then reads the
    run's settings (snapshot-aware, D3/C5), not the process globals. ``None``
    keeps the live-read fallback for user-built models.

    Raises:
        KeyError: if *name* is not a registered model. The caller owns the
            user-facing message, since it also has the debug channel.
    """
    cls = density_model_class(name)
    if name in _NO_CONFIG:
        return cls(environment=environment)
    return cls(*model_config, environment=environment)
