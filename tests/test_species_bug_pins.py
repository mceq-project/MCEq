"""MCEqParticle rates and cross sections at the edges of the particle table (no database)."""

import numpy as np

from MCEq.misc import energy_grid
from MCEq.particlemanager import MCEqParticle, _pdata

#: Five bins, one decade apart -- enough to tell a vector from a scalar.
GRID = energy_grid(c=np.logspace(0.0, 3.0, 5), b=None, w=None, d=5)


def particle(pdg_id, helicity=0):
    return MCEqParticle(pdg_id, helicity, energy_grid=GRID)


# B9 -- inverse_decay_length ZeroDivisionError branch


def test_b9_fixed_ctau_zero_returns_a_length_d_vector_of_inf():
    """B9 FIXED (R3): the ``except ZeroDivisionError`` branch returns ``(d,)`` inf.

    Before the fix the branch passed the bin *count* to ``np.ones_like``
    (``ones_like(int)`` is the 0-d ``array(1)``, ``* np.inf`` a
    ``np.float64``), so a zero-ctau particle got a scalar where every other
    path returns a length-``d`` vector. The fix passes the centre-node array
    (``ones_like(self._energy_grid.c)``), matching the docstring. Pin
    history: ``test_b9_ctau_zero_returns_a_zero_d_scalar_not_a_vector`` in
    ``ba03d9e``; flipped here.
    """
    p = particle(211)
    p.ctau = 0.0
    result = p.inverse_decay_length()

    assert isinstance(result, np.ndarray)
    assert result.shape == (GRID.d,)
    assert np.all(np.isinf(result))
    assert np.all(result > 0)


def test_b9_the_healthy_paths_all_return_length_d_vectors():
    """B9, contrast: what the except branch is supposed to look like.

    A decaying particle and a stable one both come back as ``(d,)``; only the
    ctau-zero path drops to a scalar.
    """
    decaying = particle(211).inverse_decay_length()
    assert decaying.shape == (GRID.d,)
    assert np.all(decaying > 0.0)

    stable = particle(2212).inverse_decay_length()
    assert stable.shape == (GRID.d,)
    assert np.array_equal(stable, np.zeros(GRID.d))


def test_b9_stable_returns_zero():
    """``ctau = inf`` gives a zero inverse decay length."""
    for pdg_id in (2212, 11, 22):
        p = particle(pdg_id)
        assert p.ctau == np.inf
        assert np.array_equal(p.inverse_decay_length(), np.zeros(GRID.d))


def test_b9_the_branch_is_reachable_from_the_particle_table_via_k0():
    """B9: the ledger calls the branch unreachable; K0 (311) reaches it.

    ``particletools``' table gives 311 ``ctau = 0.0`` (and mass 0.49761), so
    ``MCEqParticle(311, 0, ...)`` hits the ZeroDivisionError with no attribute
    patching. 142 of the 733 int ids in that table carry ``ctau == 0``, 100 of
    them flagged ``is_hadron`` (mostly diquarks: -5503, -5403, ...). None of
    them is a species MCEq propagates -- SIBYLL21 in the reduced DB yields 22
    particles, K0L/K0S (130/310) among them but not 311 -- which is the sense
    in which the branch is unreachable in practice.

    Correct behaviour: as in the pin above, a length-``d`` vector of ``inf``.
    """
    assert _pdata.ctau(311) == 0.0
    assert _pdata.is_hadron(311)

    k0 = particle(311)
    assert k0.name == "K0"
    assert k0.ctau == 0.0
    result = k0.inverse_decay_length()
    assert result.shape == (GRID.d,)
    assert np.all(np.isinf(result))


def test_b9_the_except_branch_depends_on_mass_and_ctau_being_python_floats():
    """B9, boundary: the whole defect hangs on ``float`` vs ``np.float64``.

    ``self.mass`` and ``self.ctau`` come off the particletools table as Python
    floats (measured: ``type(...) is float``), so ``mass / ctau / p_lab`` is
    Python's division and ``ctau = 0.0`` raises ZeroDivisionError before numpy
    is reached -- that raise is what lands on the except branch. With
    ``ctau = np.float64(0.0)`` nothing raises at all: the healthy path returns
    ``array([inf] * d)``, and with ``mass`` numpy-zero too it returns
    ``array([nan] * d)``.

    B9 FIXED (R3): the three shapes now agree where they should. The Python-
    float ``ctau == 0`` case reaches the except branch, which now returns a
    length-``d`` vector of ``inf`` -- byte-equal to the numpy-``ctau`` path
    that never raised. Before the fix the except branch dropped to a 0-d
    ``np.float64(inf)`` and the two paths disagreed in shape; pin history
    ``test_b9_ctau_zero_returns_a_zero_d_scalar_not_a_vector`` in ``ba03d9e``.
    """
    p = particle(211)
    assert type(p.mass) is float
    assert type(p.ctau) is float

    p.ctau = 0.0
    p.mass = 0.0
    py_zero = p.inverse_decay_length()
    assert py_zero.shape == (GRID.d,)
    assert np.all(np.isinf(py_zero))

    numpy_ctau = particle(211)
    numpy_ctau.ctau = np.float64(0.0)
    with np.errstate(divide="ignore"):
        no_raise = numpy_ctau.inverse_decay_length()
    assert np.array_equal(no_raise, np.full(GRID.d, np.inf))

    numpy_both = particle(211)
    numpy_both.ctau = np.float64(0.0)
    numpy_both.mass = np.float64(0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        nans = numpy_both.inverse_decay_length()
    assert nans.shape == (GRID.d,)
    assert np.all(np.isnan(nans))


# B10 -- prod_cross_section's duplicated unit constant, inverted name


def test_b10_fixed_prod_cross_section_divides_the_single_unit_constant():
    """The mb value divides by the one unit constant ``mbarn2cm2``."""
    from MCEq.data.tables import InteractionCrossSections

    p = particle(211)
    p.cs = 3.0e-27  # an arbitrary hadronic cross section in cm^2

    assert p.prod_cross_section(mbarn=False) == 3.0e-27
    assert p.prod_cross_section(mbarn=True) == 3.0000000000000004
    assert (
        p.prod_cross_section(mbarn=True) == 3.0e-27 / InteractionCrossSections.mbarn2cm2
    )
