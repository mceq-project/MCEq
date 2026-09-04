"""``MCEq.geometry.*`` is a shim layer over ``MCEq.environment``; pin its surface.

Phase 3 moved the atmospheres into ``MCEq.environment`` and left every old
module behind as a re-export. ``__all__`` in a shim is a *declaration*; this
file is the gate that makes it a contract, the way
``tests/test_operators_import.py`` does for the Phase 5 shims.

The property that matters is not "the import works" but "the same object is
reachable by both paths". A shim that rebound a name to a different object
would pass a smoke test and silently break ``isinstance`` for user code that
mixes the two import paths.

DB-free: nothing here constructs an ``MCEqRun``.
"""

from __future__ import annotations

import importlib

import pytest

#: shim module -> (new home, the names the shim must re-export).
MOVED = {
    "MCEq.geometry.column": ("MCEq.environment.column", ["fit_column_splines"]),
    "MCEq.geometry.location_centered": (
        "MCEq.environment.location_centered",
        ["LocationCenteredMixin"],
    ),
    # ``np`` is deliberate here as well: the ``EarthGeometry`` docstring embeds a
    # ``.. plot::`` directive whose body does ``from MCEq.geometry.geometry
    # import *`` and then calls ``np.linspace``.
    "MCEq.geometry.geometry": (
        "MCEq.environment.geometry",
        ["EarthGeometry", "chirkin_cos_theta_star", "np"],
    ),
    "MCEq.geometry.registry": (
        "MCEq.environment.registry",
        ["DENSITY_MODELS", "available_models", "build", "density_model_class"],
    ),
    "MCEq.geometry.nrlmsise00_mceq": (
        "MCEq.environment.msis00_backend",
        ["NRLMSISE00Base", "cNRLMSISE00", "test"],
    ),
    "MCEq.geometry.msis21_atmosphere": (
        "MCEq.environment.msis21",
        [
            "MSIS21Atmosphere",
            "MSIS21IceCubeCentered",
            "MSIS21KM3NeTCentered",
            "MSIS21LocationCentered",
        ],
    ),
    "MCEq.geometry.gtracr_cutoff": (
        "MCEq.environment.geomagnetic.cutoff",
        ["CACHE_VERSION", "build_phi0_with_cutoff", "get_cutoff_map"],
    ),
    "MCEq.geometry.corsikaatm": (
        "MCEq.environment._ext.corsikaatm",
        [
            "corsika_acc",
            "corsika_get_density",
            "corsika_get_m_overburden",
            "planar_rho_inv",
        ],
    ),
    "MCEq.geometry.nrlmsise00": (
        "MCEq.environment._ext.nrlmsise00",
        ["ap_array", "msis", "nrlmsise_flags", "nrlmsise_input", "nrlmsise_output"],
    ),
}

#: What ``from MCEq.geometry.density_profiles import *`` bound before the split.
#: Verified against ``git show d587b4b^`` when the shim was written. The five
#: non-atmosphere entries leak out of the module's own imports rather than by
#: design, but ``docs/examples/Plot_density_depth_relations.ipynb`` calls
#: ``np.linspace`` without importing numpy itself, so dropping them would break
#: a notebook that the ``notebooks`` workflow does NOT ignore.
DENSITY_PROFILES_STAR = [
    "ABCMeta",
    "AIRSAtmosphere",
    "CorsikaAtmosphere",
    "EarthsAtmosphere",
    "GeneralizedTarget",
    "IsothermalAtmosphere",
    "LocationCenteredMixin",
    "MONTH_TO_DAY_OF_YEAR",
    "MSIS00Atmosphere",
    "MSIS00IceCubeCentered",
    "MSIS00KM3NeTCentered",
    "MSIS00LocationCentered",
    "abstractmethod",
    "fit_column_splines",
    "get_atmosphere_parameters",
    "info",
    "join",
    "list_available_corsika_atmospheres",
    "np",
]

#: Reachable as attributes of the shim and via ``dir()``, but deliberately NOT
#: bound by ``import *`` -- a module ``__getattr__`` does not extend a
#: star-import, and that was true before the split too.
MSIS21_LAZY = [
    "MSIS21Atmosphere",
    "MSIS21IceCubeCentered",
    "MSIS21KM3NeTCentered",
    "MSIS21LocationCentered",
]


@pytest.mark.parametrize("shim", sorted(MOVED))
def test_shim_all_is_the_pinned_surface(shim):
    mod = importlib.import_module(shim)
    assert sorted(mod.__all__) == sorted(MOVED[shim][1])


@pytest.mark.parametrize("shim", sorted(MOVED))
def test_shim_names_are_the_same_objects(shim):
    """Re-export, not redefinition."""
    new_name, names = MOVED[shim]
    shim_mod = importlib.import_module(shim)
    new_mod = importlib.import_module(new_name)
    for name in names:
        assert getattr(shim_mod, name) is getattr(new_mod, name), (
            f"{shim}.{name} is not the same object as {new_name}.{name}"
        )


def test_density_profiles_star_import_is_unchanged():
    """The 19 names, exactly -- ``np`` included. See DENSITY_PROFILES_STAR."""
    ns: dict = {}
    exec("from MCEq.geometry.density_profiles import *", ns)  # noqa: S102
    bound = sorted(k for k in ns if not k.startswith("__"))
    assert bound == sorted(DENSITY_PROFILES_STAR)


def test_msis21_names_stay_out_of_the_star_import():
    ns: dict = {}
    exec("from MCEq.geometry.density_profiles import *", ns)  # noqa: S102
    for name in MSIS21_LAZY:
        assert name not in ns, (
            f"{name} entered the star-import; it was lazy-only before the split"
        )


@pytest.mark.parametrize("name", MSIS21_LAZY)
def test_msis21_names_stay_reachable_and_listed(name):
    import MCEq.geometry.density_profiles as dp
    from MCEq.environment import msis21

    assert getattr(dp, name) is getattr(msis21, name)
    assert name in dir(dp)


def test_density_profiles_rejects_an_unknown_attribute():
    """The ``__getattr__`` must not answer for names it does not own."""
    import MCEq.geometry.density_profiles as dp

    with pytest.raises(AttributeError):
        dp.NoSuchAtmosphere


def test_the_km3net_table_is_one_object_on_every_path():
    """Both MSIS backends must read one site table, not two copies."""
    import MCEq.geometry.density_profiles as dp
    import MCEq.geometry.msis21_atmosphere as m21
    from MCEq.environment.parameters import KM3NET_DETECTORS

    assert dp._KM3NET_DETECTORS is KM3NET_DETECTORS
    assert m21._KM3NET_DETECTORS is KM3NET_DETECTORS


def test_atmosphere_parameters_shim_keeps_its_documented_functions():
    """The six ``get_*``/``list_*`` names sphinx-automodapi would otherwise drop.

    ``automodapi`` only disables its "local to this module?" filter when
    ``__all__`` is present. A ``from ... import *`` shim has none, so every
    re-exported *function* is judged foreign and silently vanishes from the
    rendered page, while dicts and lists survive (they have no ``__name__``).
    """
    import MCEq.geometry.atmosphere_parameters as ap

    for name in (
        "get_atmosphere_parameters",
        "get_day_time_seconds",
        "get_location_data",
        "get_month_day_of_year",
        "get_nrlmsise00_defaults",
        "list_available_corsika_atmospheres",
    ):
        assert name in ap.__all__, f"{name} would vanish from the API docs"
        assert callable(getattr(ap, name))
