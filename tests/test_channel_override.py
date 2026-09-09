"""R2 deliverable B: the single channel-override hook on ``Interactions``.

Plan §6.1 / §11 Phase 6, closing data-§5 risk 18 ("injections survive
``set_interaction_model``"). Before the hook, ``MCEqRun.inject_ddm``
wrote into ``pman[p].hadr_yields`` — a particle-dict copy that the next
``regenerate_matrices``/model reload silently wiped, and that only
reached ``int_m`` because ``inject_ddm`` happened to rebuild matrices
itself. The override lives on the table, below the wiring, so
``get_matrix``, the particle wiring built from it, ``dN_dxlab``, and
the assembled ``int_m`` agree by construction, and the two injection
mechanisms (``inject_ddm`` → absolute override; ``set_mod_pprod`` →
multiplicative factor) compose instead of competing.
"""

import numpy as np
import pytest

from MCEq import config

pytestmark = pytest.mark.usefixtures("ci_reduced_db")


@pytest.fixture
def ci_reduced_db(monkeypatch):
    monkeypatch.setattr(config, "mceq_db_fname", "mceq_db_v140reduced_compact.h5")


@pytest.fixture(scope="module")
def run_sib21():
    import crflux.models as pm

    from MCEq import config as _cfg
    from MCEq.core import MCEqRun

    saved = _cfg.mceq_db_fname
    _cfg.mceq_db_fname = "mceq_db_v140reduced_compact.h5"
    try:
        return MCEqRun(
            interaction_model="SIBYLL21",
            theta_deg=0.0,
            primary_model=(pm.HillasGaisser2012, "H3a"),
        )
    finally:
        _cfg.mceq_db_fname = saved


class TestOverrideMechanics:
    def test_bare_int_keys_normalise_to_helicity_zero(self, run_sib21):
        table = run_sib21._interactions
        m = np.ones_like(table.index_d[((2212, 0), (211, 0))]) * 7.0
        assert table.set_channel_override(2212, 211, m) is True
        assert ((2212, 0), (211, 0)) in table.channel_overrides
        assert table.get_matrix((2212, 0), (211, 0)) is m
        table.clear_channel_overrides()

    def test_tuple_keys_accepted_verbatim(self, run_sib21):
        table = run_sib21._interactions
        m = np.zeros_like(table.index_d[((2212, 0), (211, 0))])
        table.set_channel_override((2212, 0), (211, 0), m)
        assert table.get_matrix((2212, 0), (211, 0)) is m
        assert table.clear_channel_overrides() == 1

    def test_get_matrix_gate_still_guards_unknown_channels(self, run_sib21):
        """The relations gate runs before the override lookup: the hook
        overrides a channel the model knows, it does not add one."""
        table = run_sib21._interactions
        with pytest.raises(Exception, match="empty matrix"):
            table.get_matrix((2212, 0), (3312, 0))

    def test_override_survives_load(self, run_sib21):
        """``load`` refills ``index_d`` from the file; the override dict
        is untouched — that is the survival property the dict write into
        ``hadr_yields`` never had."""
        table = run_sib21._interactions
        m = np.ones_like(table.index_d[((2212, 0), (211, 0))]) * 3.0
        table.set_channel_override((2212, 0), (211, 0), m)
        table.load("SIBYLL21")
        assert table.get_matrix((2212, 0), (211, 0)) is m
        table.clear_channel_overrides()
        table.load("SIBYLL21")

    def test_mod_pprod_multiplies_on_top(self, run_sib21):
        """The two mechanisms compose: the Barr-style factor applies to
        the injected matrix, not to the file matrix. (``_gen_mod_matrix``
        zeroes the lower triangle, so the factor is 2 above the diagonal
        and 0 on/below it.)"""
        table = run_sib21._interactions
        base = table.index_d[((2212, 0), (211, 0))]
        m = np.ones_like(base) * 2.0
        table.set_channel_override((2212, 0), (211, 0), m)

        def twice(xmat, egrid, value):
            return np.full(xmat.shape, float(value))

        table._set_mod_pprod(2212, 211, twice, (2.0,))
        out = table.get_matrix((2212, 0), (211, 0))
        kmat = table.mod_pprod[(2212, 211)][("twice", (2.0,))]
        np.testing.assert_allclose(out, m * kmat)
        table.mod_pprod.clear()
        table.clear_channel_overrides()

    def test_clear_returns_count_and_reverts(self, run_sib21):
        table = run_sib21._interactions
        base = table.index_d[((2212, 0), (211, 0))]
        m = np.ones_like(base) * 5.0
        table.set_channel_override((2212, 0), (211, 0), m)
        assert table.clear_channel_overrides() == 1
        assert table.get_matrix((2212, 0), (211, 0)) is base
        assert table.clear_channel_overrides() == 0


class TestInjectDdmRoutesThroughHook:
    """inject_ddm is now a caller of the hook, not a dict write."""

    def test_wiring_sees_override(self, run_sib21):
        from MCEq.models.ddm import ddm

        model = ddm.DataDrivenModel()
        ref = model.ddm_matrices(run_sib21)[(2212, 211)]
        run_sib21.inject_ddm(model)
        np.testing.assert_array_equal(
            run_sib21.pman[2212].hadr_yields[run_sib21.pman[211]], ref
        )
        # get_matrix, the wiring, and int_m all agree
        np.testing.assert_array_equal(
            run_sib21._interactions.get_matrix((2212, 0), (211, 0)), ref
        )
        run_sib21._interactions.clear_channel_overrides()
        run_sib21.regenerate_matrices()

    def test_survives_regenerate_matrices(self, run_sib21):
        """The regression the hook exists to close: the pre-R2-B code
        wrote hadr_yields directly and regenerate_matrices wiped it."""
        from MCEq.models.ddm import ddm

        model = ddm.DataDrivenModel()
        ref = model.ddm_matrices(run_sib21)[(2212, 211)]
        run_sib21.inject_ddm(model)
        run_sib21.regenerate_matrices()
        np.testing.assert_array_equal(
            run_sib21.pman[2212].hadr_yields[run_sib21.pman[211]], ref
        )
        run_sib21._interactions.clear_channel_overrides()
        run_sib21.regenerate_matrices()

    def test_survives_same_model_reload_cleared_on_model_change(self, run_sib21):
        from MCEq.models.ddm import ddm

        model = ddm.DataDrivenModel()
        ref = model.ddm_matrices(run_sib21)[(2212, 211)]
        run_sib21.inject_ddm(model)

        run_sib21.set_interaction_model("SIBYLL21", force=True)
        np.testing.assert_array_equal(
            run_sib21.pman[2212].hadr_yields[run_sib21.pman[211]], ref
        )

        # A genuine model change honours the docstring ("Calling
        # set_interaction_model overwrites DDM with a different model").
        run_sib21.set_interaction_model("QGSJET-II-04")
        assert run_sib21._interactions.channel_overrides == {}
        run_sib21.set_interaction_model("SIBYLL21")
