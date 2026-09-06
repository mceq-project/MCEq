"""Pins for the data-layer entries of the Phase 4 bug ledger.

Every test here records what the code does **today**, defects included, so the
commit that fixes one of them shows up as a failing pin rather than a silently
regenerated golden. Each docstring names its ledger id (section 9 of
``wiki/_meta/plan-mceq-v2-layered-architecture.md``) and says what the corrected
behaviour would be, which is the expectation the fix commit inherits.

The probes build ``Interactions`` / ``InteractionCrossSections`` with
``object.__new__`` and inject the handful of attributes the method under test
reads, the way ``tests/test_low_energy_blending.py`` does for ``HDF5Backend``.
B2 and B5 read a real database (the reduced DB directly, no ``MCEqRun``) and
B4 reads synthetic fixture files built by ``tests/data/make_hdf5_fixtures.py``,
loaded by file path the way ``tests/test_data_hdf5_decode.py`` does it -- B4
is the only ledger entry whose probe needs an EM database.
"""

import importlib.util
import inspect
import pathlib
from collections import defaultdict
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import MCEq.misc as misc
from MCEq.data import InteractionCrossSections, Interactions
from MCEq.misc import energy_grid, gen_xmat

_spec = importlib.util.spec_from_file_location(
    "make_hdf5_fixtures",
    pathlib.Path(__file__).parent / "data" / "make_hdf5_fixtures.py",
)
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)

#: Five bins is enough for every matrix here and keeps ``_gen_mod_matrix`` cheap.
GRID = energy_grid(c=np.logspace(0.0, 3.0, 5), b=None, w=None, d=5)


def interactions(parents):
    """An ``Interactions`` carrying only what ``_set_mod_pprod`` reads."""
    obj = object.__new__(Interactions)
    obj.energy_grid = GRID
    obj.mod_pprod = defaultdict(dict)
    obj.parents = list(parents)
    obj._physics = SimpleNamespace(use_isospin_sym=True)
    return obj


def cross_sections(index_d):
    obj = object.__new__(InteractionCrossSections)
    obj.energy_grid = GRID
    obj.index_d = dict(index_d)
    return obj


def scale(xmat, egrid, _name, value):
    """A stand-in for the ``MCEq.charm_models``-style x-modification funcs."""
    return np.full_like(xmat, value)


# B1 -- _set_mod_pprod isospin partner


def test_b1_fixed_neutron_primary_gets_the_proton_as_partner():
    """B1 FIXED (R3): the dispatch reads the caller's primary, not a rebound local.

    The pre-fix code bound ``prim_pdg, symm_pdg = 2212, 2112`` and then tested
    the overwritten name, so the swap could never fire and a 2112 primary was
    treated as a proton: its isospin copy landed on ``(2112, -211)`` -- the
    proton's partner key -- instead of ``(2212, -211)``. The fix dispatches on
    the argument before binding (interaction_tables.py). Pinned here: a
    2112/211 modification writes ``{(2112, 211), (2212, -211)}``, the
    mirror image of the proton case, and the partner keys of the two runs
    differ (they were byte-identical before). Pin history
    ``test_b1_neutron_primary_lands_on_the_proton_partner_keys`` in
    ``ae6830d``/`ba03d9e` era, flipped here.
    """
    args = ("scale", 0.1)

    proton = interactions([(2212, 0), (2112, 0)])
    assert proton._set_mod_pprod(2212, 211, scale, args) is True
    assert set(proton.mod_pprod) == {(2212, 211), (2112, -211)}

    neutron = interactions([(2212, 0), (2112, 0)])
    assert neutron._set_mod_pprod(2112, 211, scale, args) is True
    assert set(neutron.mod_pprod) == {(2112, 211), (2212, -211)}

    # Same partner key held by neither case before (2112,-211 by both); now
    # each primary owns its own mirror partner.
    assert set(neutron.mod_pprod) & set(proton.mod_pprod) == set()


def test_b1_fixed_neutron_kaon_partner_gets_its_own_key():
    """B1 FIXED (R3), kaon leg: the isospin copy lands on (2212, 321).

    Before the fix the copy was written back onto ``(2112, 321)`` -- the
    primary's own entry -- so that channel carried both a ``("scale", args)``
    and an ``("isospin", args)`` label for the same matrix and the true
    partner ``(2212, 321)`` never appeared. Now the entry holds only the
    scale and the partner key holds the isospin copy; the K0L/K0S loop names
    both nucleons explicitly and is unchanged. Pin history
    ``test_b1_neutron_kaon_partner_collides_with_the_primary_key``.
    """
    args = ("scale", 0.1)
    neutron = interactions([(2212, 0), (2112, 0)])
    assert neutron._set_mod_pprod(2112, 321, scale, args) is True

    assert sorted(neutron.mod_pprod[(2112, 321)]) == [("scale", args)]
    assert sorted(neutron.mod_pprod[(2212, 321)]) == [("isospin", args)]
    # K0L/K0S do get both nucleons, because that loop names both explicitly.
    for key in [(2112, 130), (2112, 310), (2212, 130), (2212, 310)]:
        assert key in neutron.mod_pprod


# B20 -- deleted by R3: the eta/omega/phi coupling branch in _set_mod_pprod
# tested bare ints against the (pdg, helicity) keys of ``parents``, so it never
# fired, and no shipped database carries 221/223/333 in any tuple_idcs. The
# pins recorded that (ae6830d); the deletion removes them, and what remains is
# the key-set regression guard below.


def test_b20_deleted_no_unflavoured_keys_written():
    """B20 DELETED (R3): the pion modification writes two keys, never eight.

    With the branch gone, a 2212/211 modification with (221, 0), (223, 0),
    (333, 0) present as parents writes exactly the isospin pair -- no
    unflavoured keys, no crash from the removed code path. The pre-deletion
    pins ``test_b20_eta_omega_phi_branch_is_unreachable`` (wrote nothing) and
    its bare-int mechanism twin are replaced by this key-set assertion.
    """
    args = ("scale", 0.1)
    obj = interactions([(2212, 0), (2112, 0), (221, 0), (223, 0), (333, 0)])
    assert obj._set_mod_pprod(2212, 211, scale, args) is True
    assert set(obj.mod_pprod) == {(2212, 211), (2112, -211)}


# B2 leg (iii) -- disabled_particles child filter in Interactions.load


def _load_interactions(disabled):
    """Load SIBYLL21 from the reduced DB with ``disabled_particles=disabled``.

    ``conftest``'s autouse fixture restores ``config.adv_set`` and
    ``config.mceq_db_fname`` afterwards, so mutating them here is safe.
    """
    from MCEq import config
    from MCEq.data import HDF5Backend

    config.mceq_db_fname = "mceq_db_v140reduced_compact.h5"
    config.adv_set["disabled_particles"] = list(disabled)
    inter = Interactions(HDF5Backend())
    inter.load("SIBYLL21")
    return inter


def test_b2_disabled_child_is_dropped_from_relations_but_kept_in_particles():
    """B2 leg (iii): the ``particles`` child filter compares int to tuple.

    ``self.particles += [d for d in self.relations[p] if d not in
    disabled_particles]`` puts ``d = (pdg, helicity)`` up against a list of
    bare PDG ints, so it never matches and a disabled child stays in
    ``Interactions.particles``. Twelve lines further down the ``relations``
    filter gets it right (``c[0] not in disabled_particles``), so the two
    containers disagree.

    The observable needs a *negative* id. The HDF5 read filter (``data.py``,
    ``_load_and_convert_matrix``) excludes on ``abs(pdg) in exclude``, so
    ``[11]`` drops both signs before ``load`` ever sees them (measured:
    22 -> 20 particles, no e-11 anywhere) and leaves nothing for the buggy
    filter to miss. ``[-11]`` matches no ``abs()`` at all, so both e+ and e-
    are read in and the Python-level filter is the only thing acting -- and it
    drops ``(-11, 0)`` from ``relations`` while ``particles`` keeps all 22.

    Correct behaviour: with ``[-11]`` the count should fall to 21 and
    ``(-11, 0)`` should be absent from ``particles`` too. (The read filter's
    ``abs()`` vs the literal match here is B2's other half; this pin covers
    only leg (iii).)
    """
    baseline = _load_interactions([])
    assert (-11, 0) in baseline.particles
    assert (11, 0) in baseline.particles

    inter = _load_interactions([-11])

    def em_children(obj):
        seen = set()
        for children in obj.relations.values():
            seen.update(c for c in children if abs(c[0]) == 11)
        return seen

    # relations: correctly filtered.
    assert (-11, 0) not in em_children(inter)
    assert (11, 0) in em_children(inter)
    # particles: the bug -- the positron survives, and the count is unchanged.
    assert (-11, 0) in inter.particles
    assert len(inter.particles) == len(baseline.particles) == 22


def test_b2_positive_id_excludes_both_signs_at_hdf5_read_time():
    """B2, the ``abs()`` half, recorded so the pin above cannot be misread.

    ``disabled_particles=[11]`` removes e+ *and* e- in the backend, two
    particles fewer than the unfiltered load. Correct behaviour under a literal
    match (which is what ``config``'s comment on ``disabled_particles``
    promises) would be to drop ``(11, 0)`` only.
    """
    baseline = _load_interactions([])
    inter = _load_interactions([11])

    assert not [p for p in inter.particles if abs(p[0]) == 11]
    assert len(inter.particles) == len(baseline.particles) - 2 == 20


# B6 -- InteractionCrossSections.get_cs unknown-family fall-through


@pytest.fixture
def cs_db():
    """Three model columns; nothing above |pdg| = 5000."""
    return cross_sections(
        {
            211: np.full(GRID.d, 20.0),
            321: np.full(GRID.d, 15.0),
            2212: np.full(GRID.d, 30.0),
        }
    )


@pytest.mark.parametrize("pdg", [5122, 5132, 5232, 5332])
def test_b6_fixed_unknown_family_returns_a_zero_vector(cs_db, pdg):
    """B6 FIXED (R3): the fall-through now returns ``zeros(d)`` like the lepton branch.

    Before the fix this branch returned the float ``0.0`` (shape ``()``, both
    unit modes), so a bottom-baryon projectile that reaches it -- the elif
    chain leaves only ``|pdg| >= 5000`` -- got a scalar where every other
    ``get_cs`` return is a length-``d`` vector, and the caller's broadcast
    changed shape under it. Pin history: ``test_b6_unknown_family_returns_a_
    scalar_not_a_vector`` in ``ba03d9e``; the fix commit flipped it to this
    expectation. The 511/521 boundary test below guards that the elif chain's
    ordering did NOT move with the fix.
    """
    for mbarn in (True, False):
        cs = cs_db.get_cs(pdg, mbarn=mbarn)
        assert isinstance(cs, np.ndarray)
        assert cs.shape == (GRID.d,)
        assert np.all(cs == 0.0)


def test_b6_the_lepton_branch_already_returns_a_vector(cs_db):
    """B6, contrast: ``5 < |pdg| < 23`` returns ``zeros(d)`` as it should."""
    for pdg in (13, -11, 22):
        cs = cs_db.get_cs(pdg)
        assert cs.shape == (GRID.d,)
        assert np.array_equal(cs, np.zeros(GRID.d))


def test_b6_b_mesons_do_not_reach_the_else_branch(cs_db):
    """B6, boundary: 511/521 land in the KAON branch, not the fall-through.

    Recorded because it is the natural wrong guess about which ids are
    affected. A ``get_cs`` fix must not change this -- it is the elif chain's
    ordering, untouched by B6.
    """
    for pdg in (511, 521):
        assert np.array_equal(cs_db.get_cs(pdg, mbarn=True), np.full(GRID.d, 15.0))


# misc.gen_xmat -- module-global cache keyed on shape only


@pytest.fixture
def clean_xmat_cache():
    """Isolate the ``misc._xmat`` global; other tests share it."""
    saved = misc._xmat
    misc._xmat = None
    try:
        yield
    finally:
        misc._xmat = saved


def test_gen_xmat_aliases_across_different_grids_of_equal_size(clean_xmat_cache):
    """``misc.gen_xmat`` caches on ``(d, d)`` alone, so grids collide.

    The guard is ``if _xmat is None or _xmat.shape != dims``: nothing in it
    looks at ``energy_grid.c``. Two five-decade-apart grids with the same bin
    count therefore get back the *same object*, still holding the first grid's
    x values -- ``x[2, 3]`` is 0.1 (one decade per bin, from ``logspace(0, 3,
    4)``) where the second grid's ratio is 0.01. The returned array is the
    cache itself, not a copy, so a caller that writes into it corrupts every
    later caller.

    Correct behaviour (plan section 5, ``data/energy_grid.py``): the x matrix
    becomes a cached method on ``EnergyGrid``, keyed by the grid it belongs to,
    so each grid gets its own -- ``a is not b``, and ``b[2, 3] == 0.01``.
    """
    coarse = energy_grid(c=np.logspace(0.0, 3.0, 4), b=None, w=None, d=4)
    steep = energy_grid(c=np.logspace(0.0, 6.0, 4), b=None, w=None, d=4)

    first = gen_xmat(coarse)
    snapshot = first.copy()
    assert first[2, 3] == pytest.approx(coarse.c[2] / coarse.c[3])

    second = gen_xmat(steep)
    assert second is first
    assert np.array_equal(second, snapshot)
    # The value the second grid should have produced, and did not.
    assert steep.c[2] / steep.c[3] == pytest.approx(0.01)
    assert second[2, 3] == pytest.approx(0.1)


def test_gen_xmat_recomputes_when_only_the_size_changes(clean_xmat_cache):
    """The shape key is the *only* thing that invalidates the cache."""
    four = energy_grid(c=np.logspace(0.0, 3.0, 4), b=None, w=None, d=4)
    five = energy_grid(c=np.logspace(0.0, 6.0, 5), b=None, w=None, d=5)

    first = gen_xmat(four)
    second = gen_xmat(five)
    assert second is not first
    assert second.shape == (5, 5)
    assert second[3, 4] == pytest.approx(five.c[3] / five.c[4])
    # ...and the cache now holds the new one, so the old grid re-derives.
    assert misc._xmat is second


def test_gen_xmat_returns_the_cache_itself_not_a_copy(clean_xmat_cache):
    """In-place writes by one caller are visible to the next.

    Pinned as the consequence that makes the aliasing more than cosmetic. The
    per-grid cached method the plan calls for should hand out an array no
    caller can reach across a boundary like this -- at minimum, mutating one
    return value must not change a later one.
    """
    grid = energy_grid(c=np.logspace(0.0, 3.0, 4), b=None, w=None, d=4)
    first = gen_xmat(grid)
    first[0, 0] = -12345.0
    assert gen_xmat(grid)[0, 0] == -12345.0


# B4 -- ice->water EM medium substitution rewrites a local, then reads the
# unswapped medium


def em_backend(had_path, em_path, medium):
    """An ``HDF5Backend`` with ``enable_em`` on, all four groups injected.

    The idiom of ``tests/test_data_hdf5_decode.py::backend``, with two
    changes the B4 pins need: ``enable_em`` is True and both database files
    exist, so the EM reads happen; ``interaction_medium`` and the constructor
    ``medium`` agree, and nothing comes from ``MCEq.config`` except the
    ``info`` logger's own settings.
    """
    from MCEq.data import HDF5Backend

    paths = SimpleNamespace(
        data_dir=had_path.parent,
        mceq_db_fname=had_path.name,
        em_db_fname=em_path.name,
    )
    grid = SimpleNamespace(e_min=None, e_max=None, dtype=None, em_standalone_grid=False)
    physics = SimpleNamespace(
        filters={
            "disabled_particles": [],
            "forced_int_cs": None,
            "replace_meson_cross_sections_with": None,
        },
        assume_nucleon_interactions_for_exotics=True,
        enable_em=True,
        enable_cont_rad_loss=False,
        fallback_to_air_cs=True,
        interaction_medium=medium,
        muon_helicity_dependence=False,
    )
    em = SimpleNamespace(air_density=None)
    return HDF5Backend(medium=medium, paths=paths, grid=grid, physics=physics, em=em)


def b4_dbs(tmp_path):
    """Hadronic file with an ``ice`` medium, EM file with ``ice`` AND ``water``.

    The two EM media carry distinguishable content on purpose: the ``ice``
    pack stores the reversed channel list (so every matrix value differs) and
    the ``ice`` cs table a different scale.
    """
    had = fixtures.build_database(tmp_path / "b4_had.h5", medium="ice")
    em = fixtures.build_em_database(
        tmp_path / "b4_em.h5",
        {
            "water": (fixtures.EM_CHANNELS, 2.0),
            "ice": (list(reversed(fixtures.EM_CHANNELS)), 1.0),
        },
    )
    return had, em


def test_b4_interaction_db_reads_ice_after_saying_water(tmp_path):
    """B4: the ice->water swap in ``interaction_db`` is dead.

    The block at the head of the EM section rewrites the LOCAL ``medium`` and
    logs "replaced by water", but every lookup below it uses ``self.medium``,
    which still says ``ice``. Pin: the matrices that arrive are the ICE
    group's (channel ``(11, 0) -> (11, 0)`` decodes from the reversed list,
    position 6, not from the water list, position 0). The fix -- making the
    EM lookups use the swapped medium, or deleting the swap and substituting
    at the source -- must flip this expect to the water numbers.

    The hadronic file carries an ``ice`` subgroup for the *cs* leg of the
    pair (without one, ``_cs_db_single`` raises ``Unknown selections.`` at
    its own subgroup guard before the EM section runs); this leg alone would
    also pass on an air-only hadronic file, where the ``fallback_to_air_cs``
    rewrite of the local ``medium`` to ``air`` skips the swap block but the
    EM lookups still use ``self.medium``.
    """
    had, em = b4_dbs(tmp_path)
    db = em_backend(had, em, "ice").interaction_db(fixtures.MODEL)
    ice_pos = list(reversed(fixtures.EM_CHANNELS)).index((11, 11))
    assert ice_pos == 6
    matrix = db["index_d"][((11, 0), (11, 0))]
    assert np.array_equal(matrix, fixtures.channel_block(ice_pos))
    assert not np.array_equal(matrix, fixtures.channel_block(0))


def test_b4_cs_db_has_no_ice_to_water_substitution_at_all(tmp_path):
    """B4, second leg: ``_cs_db_single`` never mentions ice or water.

    Where ``interaction_db`` at least tries a substitution (a dead one, per
    the pin above), ``_cs_db_single`` looks the EM ``cs`` table up under the
    local ``medium`` verbatim, so an ice run reads the ICE table. Pin: the
    merged EM columns equal the scale-1.0 (ice) table, not scale 2.0 (water).
    A fix that unifies both paths through one substitution must update this
    expect to water -- and should say which path is right, since today the
    interaction path logs a substitution it does not perform. The cs leg's discriminator is the scale alone (the ``cs``
    table is keyed by its own ``projectiles`` attribute, independently of
    the pack order); the interaction pin above discriminates the same fix
    through pack order alone, so a fixture regression that flattened the
    scales cannot hide a real substitution fix from both pins.
    """
    had, em = b4_dbs(tmp_path)
    with h5py.File(em, "r") as f:
        assert np.array_equal(
            f["electromagnetic/water/cs"][:], fixtures.em_cs_table(2.0)
        )
        assert np.array_equal(f["electromagnetic/ice/cs"][:], fixtures.em_cs_table(1.0))
    cs = em_backend(had, em, "ice").cs_db(fixtures.MODEL)
    ice_table = fixtures.em_cs_table(1.0)
    water_table = fixtures.em_cs_table(2.0)
    for ip, p in enumerate(fixtures.EM_CS_PARENTS):
        assert np.array_equal(cs["index_d"][p], ice_table[ip, :])
    assert not np.array_equal(cs["index_d"][11], water_table[0, :])
    assert cs["parents"][-len(fixtures.EM_CS_PARENTS) :] == fixtures.EM_CS_PARENTS


# B5 -- disable_interactions_of_unstable removes the nucleons instead of the
# unstable hadrons


def _load_b5(disable_unstable):
    """Load SIBYLL21 from the reduced DB with the B5 flag set as given.

    ``conftest``'s autouse fixture restores ``config.adv_set`` and
    ``config.mceq_db_fname`` afterwards, so mutating them here is safe.
    """
    from MCEq import config
    from MCEq.data import HDF5Backend

    config.mceq_db_fname = "mceq_db_v140reduced_compact.h5"
    config.adv_set["disable_interactions_of_unstable"] = disable_unstable
    inter = Interactions(HDF5Backend())
    inter.load("SIBYLL21")
    return inter


def test_b5_flag_removes_the_nucleon_projectiles_and_keeps_the_mesons():
    """B5: the flag is inverted -- it strips nucleons and keeps the mesons.

    The config doc says ``#: Disable particle production by all hadrons,
    except nucleons``, i.e. keep the nucleons and strip the unstable hadrons,
    but the branch in ``Interactions.load`` removes exactly the four nucleon
    charge states ``[2212, 2112, -2212, -2112]`` and leaves every pi/K parent
    in place. Measured on the reduced DB (2026-09-06): baseline parents carry
    both nucleon signs plus the meson families and a 3122 hyperon family;
    with the flag on the same list survives minus the four nucleon tuples,
    ``relations`` still matches ``parents`` exactly, and ``particles`` keeps
    all 20 entries either way -- the filter acts on the parent list only, so
    nucleons stay as secondaries.

    Correct behaviour: the flag keeps nucleons and strips the unstable
    parents, which flips the parent-set assertion below; the hyperons' fate
    (3122 is unstable but not a meson) is decided by whatever the fixed rule
    is, so the fix owns that part of the expectation too. A fix that rewrote
    the filter as a *child* filter instead would also change the
    ``particles`` pin here, which is why it is pinned.
    """
    baseline = _load_b5(False)
    baseline_pdgs = {p[0] for p in baseline.parents}
    for nucleon in (2212, 2112, -2212, -2112):
        assert (nucleon, 0) in baseline.parents
    assert 211 in baseline_pdgs and -211 in baseline_pdgs
    assert 321 in baseline_pdgs and -321 in baseline_pdgs

    inter = _load_b5(True)

    # The bug: the surviving set is the baseline minus the four nucleons.
    assert set(inter.parents) == {
        (-3122, 0),
        (-321, 0),
        (-211, 0),
        (111, 0),
        (130, 0),
        (211, 0),
        (310, 0),
        (321, 0),
        (3122, 0),
    }
    for nucleon in (2212, 2112, -2212, -2112):
        assert (nucleon, 0) not in inter.parents
    assert set(inter.relations) == set(inter.parents)
    # Parent-list filter only: the proton survives as a secondary.
    assert (2212, 0) in inter.particles
    assert len(inter.particles) == len(baseline.particles) == 20


def test_b5_filter_is_a_hardcoded_nucleon_list_not_a_stability_check():
    """B5, mechanism check: the branch is a literal, not a lifetime lookup.

    ``inspect.getsource(Interactions.load)`` normalized to single spaces
    contains the branch name and, right after it, the four-int list
    ``[2212, 2112, -2212, -2112]`` -- the exact set the ``adv_set`` comment's
    ``except nucleons`` intends to *keep*. Like the B20 mechanism test, this
    pins the mechanism, not the intent: it goes red when the branch is deleted
    or the literal set changes. A fix that keeps the header and the list but
    inverts the predicate survives here -- the behavioural test above is what
    catches that.
    """
    source = " ".join(inspect.getsource(Interactions.load).split())
    branch = 'if filters["disable_interactions_of_unstable"]:'
    assert source.count(branch) == 1
    after_branch = source.split(branch, 1)[1]
    assert "[2212, 2112, -2212, -2112]" in after_branch


# B3 -- deleted by R3: the veto block, its config key, and the crash evidence
# all went together in the deletion commit. What stays pinned is the deletion
# itself -- the key must not come back, and get_matrix must not re-grow a
# filter read that the physics views no longer provide.


def test_b3_deleted_key_and_guard():
    """B3 DELETED (R3): no `disable_leading_mesons` key exists anywhere.

    The veto raised TypeError on `abs(child) < 2000` for a (pdg, helicity)
    tuple before its hard-coded `ie = 50` window was ever evaluated -- dead
    since the helicity keys landed. The deletion commit removed the block in
    ``get_matrix``, the ``adv_set`` default, and the ``disable_leading_
    mesons`` entry from all six golden generators' pin blocks. A re-addition
    of the key without restoring working code fails the first assert; a
    restored read of a missing key in get_matrix fails the second leg below.
    """
    from MCEq import config

    assert "disable_leading_mesons" not in config.adv_set
    assert not hasattr(config, "disable_leading_mesons")

    obj = interactions([(2212, 0)])
    obj.index_d = {((2212, 0), (211, 0)): np.ones((GRID.d, GRID.d))}
    obj.relations = {(2212, 0): [(211, 0)]}
    obj._physics = SimpleNamespace(use_isospin_sym=True, filters={})
    # get_matrix must survive an EMPTY filters dict -- no veto read left.
    assert obj.get_matrix((2212, 0), (211, 0)).shape == (GRID.d, GRID.d)
