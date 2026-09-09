"""Grouped views over the flat configuration names.

Each group exposes the settings needed by one component, so a component
receives ``HDF5Backend(paths, grid)`` rather than reaching into the
:mod:`MCEq.config` module.

They are **views, not snapshots**. A group reads the flat name from
:mod:`MCEq.config` at attribute access, so a component that holds one still
sees ``config.e_min = ...`` written after it was constructed — which is what
the tests, the notebooks and the golden generators all rely on. That also means
a group costs nothing to build, and there is exactly one place a value lives.
Reading a group does not rebuild objects already constructed from earlier
values.

Mutation goes the same way::

    grid = config.grid
    grid.e_min = 1e-1  # identical to config.e_min = 1e-1

Dict-valued settings (``adv_set``, ``etd2_path``, ``low_energy_extension``) are
handed through by identity, so ``physics.filters["disabled_particles"]`` and
``config.adv_set["disabled_particles"]`` are the same object.
"""

from __future__ import annotations

import sys


class GroupView:
    """A named subset of the flat config namespace.

    ``fields`` maps each group attribute onto the flat config name it reads,
    for example ``dtype`` to ``floatlen`` and ``filters`` to ``adv_set``.
    """

    __slots__ = ("_fields", "_name", "_over")

    def __init__(self, name, fields, over=None):
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_fields", fields)
        object.__setattr__(self, "_over", dict(over) if over else {})

    def __getattr__(self, attr):
        over = object.__getattribute__(self, "_over")
        if attr in over:
            return over[attr]
        fields = object.__getattribute__(self, "_fields")
        try:
            flat = fields[attr]
        except KeyError:
            raise AttributeError(
                f"{object.__getattribute__(self, '_name')} has no setting {attr!r}"
            ) from None
        return getattr(sys.modules["MCEq.config"], flat)

    def __setattr__(self, attr, value):
        fields = object.__getattribute__(self, "_fields")
        if attr not in fields:
            raise AttributeError(
                f"{object.__getattribute__(self, '_name')} has no setting {attr!r}"
            )
        if object.__getattribute__(self, "_over"):
            raise AttributeError(
                f"{object.__getattribute__(self, '_name')} copy is read-only; "
                f"cannot set {attr!r}"
            )
        setattr(sys.modules["MCEq.config"], fields[attr], value)

    def __dir__(self):
        return sorted(object.__getattribute__(self, "_fields"))

    def __repr__(self):
        fields = object.__getattribute__(self, "_fields")
        shown = ", ".join(f"{a}={getattr(self, a)!r}" for a in sorted(fields))
        return f"{object.__getattribute__(self, '_name')}({shown})"

    def as_dict(self):
        """The group's settings as a plain dict, resolved now."""
        fields = object.__getattribute__(self, "_fields")
        return {attr: getattr(self, attr) for attr in sorted(fields)}

    def with_overrides(self, **values):
        """A view with ``values`` supplied and live reads for every other field.

        Overridden fields return the supplied value; every other field stays
        a live read of the flat namespace, exactly as on the base view.
        A nonempty override set refuses writes — an override made here can
        never leak back into the flat config; an empty override set remains
        writable. Used by :func:`MCEq.driver.system.run_physics_view`, which
        forces ``muon_helicity_dependence=False`` for one EM run without
        touching the defaults.
        """
        fields = object.__getattribute__(self, "_fields")
        unknown = sorted(set(values) - set(fields))
        if unknown:
            raise AttributeError(
                f"{object.__getattribute__(self, '_name')} has no setting "
                f"{unknown[0]!r}"
            )
        return GroupView(object.__getattribute__(self, "_name"), fields, over=values)


#: group name -> {attribute: flat config name}
GROUPS = {
    "paths": {
        "data_dir": "data_dir",
        "mceq_db_fname": "mceq_db_fname",
        "em_db_fname": "em_db_fname",
    },
    "grid": {
        "e_min": "e_min",
        "e_max": "e_max",
        "dtype": "floatlen",
        "em_standalone_grid": "em_standalone_grid",
    },
    "physics": {
        "enable_em": "enable_em",
        "enable_energy_loss": "enable_energy_loss",
        "enable_cont_rad_loss": "enable_cont_rad_loss",
        "enable_em_ion": "enable_em_ion",
        "generic_losses_all_charged": "generic_losses_all_charged",
        "muon_helicity_dependence": "muon_helicity_dependence",
        "muon_multiple_scattering": "muon_multiple_scattering",
        "enable_default_tracking": "enable_default_tracking",
        "prompt_ctau": "prompt_ctau",
        "minimal_primary_energy": "minimal_primary_energy",
        "standard_particles": "standard_particles",
        "use_isospin_sym": "use_isospin_sym",
        "assume_nucleon_interactions_for_exotics": "assume_nucleon_interactions_for_exotics",
        "fallback_to_air_cs": "fallback_to_air_cs",
        "interaction_medium": "interaction_medium",
        "A_target": "A_target",
        "filters": "adv_set",
        "low_energy": "low_energy_extension",
    },
    "losses": {
        "stencil_method": "loss_stencil_method",
        "stencil_alpha0": "loss_stencil_alpha0",
        "stencil_low_upwind_rows": "loss_stencil_low_upwind_rows",
        "average_operator": "average_loss_operator",
        "step_for_average": "loss_step_for_average",
    },
    "em": {
        "air_density": "em_air_density",
        "adaptive_step": "em_adaptive_step",
        "step_safety": "em_step_safety",
        "step_dense_eig_max": "em_step_dense_eig_max",
    },
    "environment": {
        "density_model": "density_model",
        "r_E": "r_E",
        "h_obs": "h_obs",
        "h_atm": "h_atm",
        "max_density": "max_density",
        "len_target": "len_target",
        "env_density": "env_density",
        "env_name": "env_name",
    },
    "secant": {
        "transport": "secant_theta_transport",
        "cap_deg": "secant_theta_cap_deg",
        "row_kmax": "secant_theta_row_kmax",
        "lam_rel": "secant_theta_lam_rel",
        "w_flat": "secant_theta_w_flat",
        "e_max": "secant_theta_e_max",
    },
    # X_start and etd2_path are read when a path is planned. They are NOT
    # picked up by a later solve() on a live instance: _calculate_integration_path
    # keys its cache on the unresolved kwargs, so a post-construction write is
    # only seen after force=True or an explicit solve(eps=...) argument.
    "solver": {
        "X_start": "X_start",
        "etd2_path": "etd2_path",
    },
    "backend": {
        "kernel_config": "kernel_config",
        "cuda_gpu_id": "cuda_gpu_id",
        "cuda_fp_precision": "cuda_fp_precision",
        "mkl_threads": "mkl_threads",
        "mkl_path": "mkl_path",
    },
    "output": {
        "return_as": "return_as",
        "excpt_on_missing_particle": "excpt_on_missing_particle",
    },
    "debug": {
        "level": "debug_level",
        "override_fcn": "override_debug_fcn",
        "override_max_level": "override_max_level",
        "print_module": "print_module",
    },
}

#: Flat config name -> the group and group attribute that own it.
FLAT_TO_GROUP = {
    flat: (group, attr)
    for group, fields in GROUPS.items()
    for attr, flat in fields.items()
}


def build():
    """One view per group, for installation on the config module."""
    return {name: GroupView(name, fields) for name, fields in GROUPS.items()}
