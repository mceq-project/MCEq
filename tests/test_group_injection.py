"""Phase 6 commit 7 — group injection is unconditional.

Pins the contract established when the seven function-local
``None -> config`` fallbacks were deleted: the constructors of the
below-driver classes take their settings groups as required keyword
arguments, the EM/helicity rule is a validator returning a corrected
*copy* (flat config untouched, plan §9), and :mod:`MCEq.misc` reads its
settings from views installed by :mod:`MCEq.config` at import time
instead of importing it.

The pre-deletion reachability probe (every fallback confirmed to read
live config before removal) ran against the parent commit; see the
commit message. What is pinned here is the contract after removal.
"""

import ast
import pathlib

import pytest

import MCEq.config as config
from MCEq.config.groups import GroupView
from MCEq.config.run import RunConfig
from MCEq.driver.system import run_physics_view


def _source(module_name):
    import importlib.util

    origin = importlib.util.find_spec(module_name).origin
    return pathlib.Path(origin).read_text()


# --- the override-copy mechanism (plan §9: "returns a corrected copy") ------


def test_with_overrides_freezes_only_the_named_fields():
    base = config.physics
    frozen = base.with_overrides(muon_helicity_dependence=False)
    assert frozen is not base
    assert frozen.muon_helicity_dependence is False
    assert base.muon_helicity_dependence is True  # flat untouched
    # a non-overridden field still reads live from the flat namespace
    medium = base.interaction_medium
    assert frozen.interaction_medium == medium
    config.interaction_medium = "water"
    try:
        assert frozen.interaction_medium == "water"
    finally:
        config.interaction_medium = medium
    assert base.muon_helicity_dependence is True


def test_with_overrides_copy_rejects_mutation_and_keeps_the_api_surface():
    frozen = config.physics.with_overrides(enable_em=False)
    with pytest.raises(AttributeError, match="read-only"):
        frozen.enable_em = True
    with pytest.raises(AttributeError, match="no setting"):
        frozen.nope = 1
    assert "muon_helicity_dependence" in dir(frozen)
    assert frozen.as_dict()["enable_em"] is False
    # non-overridden fields stay live reads, not freezes of the override-time value
    assert frozen.as_dict()["muon_helicity_dependence"] is (
        config.muon_helicity_dependence
    )


def test_validator_forces_unpolarized_copy_without_touching_flat():
    saved = (config.enable_em, config.muon_helicity_dependence)
    try:
        config.enable_em = True
        config.muon_helicity_dependence = True
        view = run_physics_view()
        assert view.muon_helicity_dependence is False
        assert config.muon_helicity_dependence is True  # no flat write
        assert view is not config.physics
        assert isinstance(view, GroupView)
    finally:
        config.enable_em, config.muon_helicity_dependence = saved


def test_validator_passes_through_when_no_conflict():
    saved = config.enable_em
    try:
        config.enable_em = False
        assert run_physics_view() is config.physics
        config.enable_em = True
        config.muon_helicity_dependence = False
        assert run_physics_view() is config.physics
    finally:
        config.enable_em = saved


def test_validator_warning_names_the_rule(capfd):
    saved = (config.enable_em, config.muon_helicity_dependence, config.debug_level)
    try:
        config.enable_em = True
        config.muon_helicity_dependence = True
        config.debug_level = 1
        run_physics_view()
        out = capfd.readouterr().out
        assert "muon_helicity_dependence" in out and "enable_em" in out
    finally:
        (
            config.enable_em,
            config.muon_helicity_dependence,
            config.debug_level,
        ) = saved


# --- misc takes views, never imports config ----------------------------------


def test_misc_no_longer_imports_config():
    tree = ast.parse(_source("MCEq.misc"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module != "MCEq.config"
            assert node.module != "config" or node.level == 0
            assert not (node.module is None and "config" in node.names)
        if isinstance(node, ast.Import):
            assert all(a.name != "MCEq.config" for a in node.names)


def test_misc_views_are_live_once_resolved():
    import MCEq.misc as misc

    misc._views()  # the lazy path; config's tail install is best-effort only
    assert misc._DEBUG is config.debug
    assert misc._PHYSICS is config.physics
    saved = config.debug_level
    try:
        config.debug_level = 7
        assert misc._DEBUG.level == 7
    finally:
        config.debug_level = saved


def test_average_A_target_auto_still_resolves_medium():
    from MCEq.misc import average_A_target

    saved = config.interaction_medium
    try:
        config.interaction_medium = "air"
        assert average_A_target("auto") == pytest.approx(14.543071650401519)
        config.interaction_medium = "water"
        assert average_A_target("auto") == 6.0
    finally:
        config.interaction_medium = saved


# --- the constructors require their groups ------------------------------------


def test_bare_groupless_construction_is_TypeError():
    from MCEq.data import ContinuousLosses, Decays, HDF5Backend, Interactions
    from MCEq.operators.matrix_builder import MatrixBuilder
    from MCEq.particlemanager import MCEqParticle, ParticleManager

    db = object()
    with pytest.raises(TypeError):
        HDF5Backend()
    with pytest.raises(TypeError):
        Interactions(db)
    with pytest.raises(TypeError):
        ContinuousLosses(db)
    with pytest.raises(TypeError):
        Decays(db)
    with pytest.raises(TypeError):
        ParticleManager([2212], None, db)
    with pytest.raises(TypeError):
        MCEqParticle(2212, 0, energy_grid=None, cs_db=None)
    with pytest.raises(TypeError):
        MatrixBuilder(None, None)


def test_system_injects_one_physics_view_everywhere():
    """CascadeSystem resolves ``run_physics()`` once and hands the copy down.

    Parsed from source: every group argument of the five table constructors
    must name ``self._physics`` — no constructor may keep reading
    ``config.physics`` — so an EM run's view (helicity forced off) is the
    one every table sees.
    """
    src = _source("MCEq.driver.system")
    assert src.count("self._physics = run_physics_view(cfg)") == 1
    for cls in ("HDF5Backend", "Interactions", "ContinuousLosses", "Decays"):
        assert "physics=self._physics" in src, cls
    # since D3 every group read goes through the run's cfg; the only direct
    # config.* read left in system.py is the validator's live-default branch
    reads = [
        ast.unparse(node)
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Attribute) and ast.unparse(node).startswith("config.")
    ]
    assert reads == ["config.physics"]
    cfg_reads = [
        ast.unparse(node)
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Attribute)
        and ast.unparse(node).startswith("self._cfg.")
    ]
    assert "self._cfg.paths" in cfg_reads and "self._cfg.grid" in cfg_reads


def test_helicity_write_is_gone_from_the_driver():
    """No Assign to the flat flag (nor to the physics view) anywhere in the
    driver anymore — only reads of the group may remain."""
    tree = ast.parse(_source("MCEq.driver.mceq_run"))
    written = {
        node.targets[0].attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Attribute)
    }
    assert "muon_helicity_dependence" not in written


# --- D3: the detached snapshot ------------------------------------------------------------


def _run_kwargs(primary):
    return dict(
        interaction_model="SIBYLL21",
        primary_model=primary,
        theta_deg=20.0,
    )


def test_default_run_reads_live_config():
    import crflux.models as pm

    from MCEq.core import MCEqRun

    m = MCEqRun(**_run_kwargs((pm.HillasGaisser2012, "H3a")))
    assert m.config is config


def test_snapshot_run_is_detached_from_later_writes():
    import crflux.models as pm

    from MCEq.config.run import RunConfig
    from MCEq.core import MCEqRun

    snap = RunConfig.snapshot()
    saved = config.e_min
    try:
        config.e_min = 7e0
        m = MCEqRun(config=snap, **_run_kwargs((pm.HillasGaisser2012, "H3a")))
        # the frozen grid (e_min 0.1 at snapshot time) reached the energy grid
        assert m.e_bins[0] < 1.0, m.e_bins[0]
    finally:
        config.e_min = saved


def test_snapshot_refuses_writes_and_bad_types():
    from MCEq.config.run import RunConfig
    from MCEq.core import MCEqRun

    snap = RunConfig.snapshot()
    with pytest.raises(AttributeError, match="read-only"):
        snap.e_min = 1.0
    with pytest.raises(AttributeError, match="no setting"):
        snap.nope
    with pytest.raises(TypeError, match="RunConfig"):
        MCEqRun("SIBYLL21", {}, 20.0, config={"e_min": 1})


def test_snapshot_copies_dicts_not_aliases():
    snap = RunConfig.snapshot()
    adv = config.adv_set
    key = "disabled_particles"
    adv[key].append(2212)
    try:
        assert 2212 not in snap.physics.filters[key]
    finally:
        adv[key].remove(2212)
