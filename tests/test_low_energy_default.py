"""MCEqRun blends in FLUKA20251 below the transition unless told otherwise."""

import crflux.models as pm

from MCEq import config
from MCEq.core import MCEqRun


def _run(**kwargs):
    return MCEqRun(
        "SIBYLL21",
        (pm.HillasGaisser2012, "H3a"),
        0.0,
        build_matrices=False,
        **kwargs,
    )


def test_default_blends_fluka():
    assert config.low_energy_extension["model"] == "FLUKA20251"
    assert _run()._mceq_db.low_energy_model == "FLUKA20251"


def test_explicit_none_switches_the_blend_off():
    assert _run(low_energy_model=None)._mceq_db.low_energy_model is None


def test_default_skipped_for_a_medium_without_fluka():
    """The CI data has SIBYLL23E for hydrogen but no FLUKA: the default must
    not borrow FLUKA's air tables for it."""
    run = MCEqRun(
        "SIBYLL23E",
        (pm.HillasGaisser2012, "H3a"),
        0.0,
        medium="hydrogen",
        build_matrices=False,
    )
    assert run._mceq_db.low_energy_model is None
