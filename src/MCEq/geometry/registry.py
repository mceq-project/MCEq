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

Destined for ``MCEq.environment.registry``.
"""

from __future__ import annotations

from importlib import import_module

#: Public model name -> ``"module:qualname"``. Order is the order
#: :func:`available_models` reports and the error message lists, so it is the
#: user-visible ordering; keep the MSIS families together and grouped by
#: backend generation, as the original literal did.
DENSITY_MODELS: dict[str, str] = {
    "MSIS00": "MCEq.geometry.density_profiles:MSIS00Atmosphere",
    "MSIS00_IC": "MCEq.geometry.density_profiles:MSIS00IceCubeCentered",
    "MSIS00_KM3NeT": "MCEq.geometry.density_profiles:MSIS00KM3NeTCentered",
    "MSIS21": "MCEq.geometry.msis21_atmosphere:MSIS21Atmosphere",
    "MSIS21_IC": "MCEq.geometry.msis21_atmosphere:MSIS21IceCubeCentered",
    "MSIS21_KM3NeT": "MCEq.geometry.msis21_atmosphere:MSIS21KM3NeTCentered",
    "CORSIKA": "MCEq.geometry.density_profiles:CorsikaAtmosphere",
    "AIRS": "MCEq.geometry.density_profiles:AIRSAtmosphere",
    "Isothermal": "MCEq.geometry.density_profiles:IsothermalAtmosphere",
    "GeneralizedTarget": "MCEq.geometry.density_profiles:GeneralizedTarget",
}

#: Models whose constructor takes no configuration. ``set_density_model``
#: discarded ``model_config`` for these, and still does.
_NO_CONFIG = frozenset({"GeneralizedTarget"})


def available_models() -> list[str]:
    """The model names :func:`build` accepts, in user-visible order."""
    return list(DENSITY_MODELS)


def density_model_class(name: str) -> type:
    """Resolve one registry entry to its class, importing it on first use."""
    try:
        target = DENSITY_MODELS[name]
    except KeyError:
        raise KeyError(name) from None
    module_name, _, qualname = target.partition(":")
    return getattr(import_module(module_name), qualname)


def build(name: str, model_config):
    """Instantiate the atmosphere *name* with *model_config*.

    ``model_config`` is splatted into the constructor, except for the models in
    :data:`_NO_CONFIG`, which are constructed with no arguments -- the
    behaviour of the ``elif`` ladder this replaces.

    Raises:
        KeyError: if *name* is not a registered model. The caller owns the
            user-facing message, since it also has the debug channel.
    """
    cls = density_model_class(name)
    if name in _NO_CONFIG:
        return cls()
    return cls(*model_config)
