"""Pins for the species-layer entries of the Phase 4 bug ledger.

Records today's behaviour of ``MCEqParticle`` -- defects included -- so that the
commit fixing one produces a failing pin instead of a quiet change of numbers.
Docstrings name the ledger id (section 9 of
``wiki/_meta/plan-mceq-v2-layered-architecture.md``) and state the corrected
behaviour, which is the expectation the fix commit inherits.

DB-free: ``MCEqParticle`` is built straight from the particletools table with a
five-bin grid; nothing here constructs an ``MCEqRun``.
"""

import numpy as np
import pytest

from MCEq.misc import energy_grid
from MCEq.particlemanager import MCEqParticle, _pdata

#: Five bins, one decade apart -- enough to tell a vector from a scalar.
GRID = energy_grid(c=np.logspace(0.0, 3.0, 5), b=None, w=None, d=5)


def particle(pdg_id, helicity=0):
    return MCEqParticle(pdg_id, helicity, energy_grid=GRID)


# B9 -- inverse_decay_length ZeroDivisionError branch


def test_b9_ctau_zero_returns_a_zero_d_scalar_not_a_vector():
    """B9: the ``except ZeroDivisionError`` branch returns a 0-d value.

    ``np.ones_like(self._energy_grid.d)`` is passed the bin *count* -- a plain
    int -- not the grid array, so ``ones_like`` yields the 0-d ``array(1)`` and
    ``* np.inf`` collapses it to ``np.float64(inf)``. Measured: ``shape ()``,
    ``ndim 0``, ``dtype float64``, and not an ``ndarray`` instance at all.
    Every other return path from this method is a length-``d`` vector.

    Correct behaviour: ``np.ones_like(self._energy_grid.c) * np.inf`` (or
    ``np.full(self._energy_grid.d, np.inf)``) -- shape ``(d,)``, so callers
    that index or broadcast the result keep working.
    """
    p = particle(211)
    p.ctau = 0.0
    result = p.inverse_decay_length()

    assert np.ndim(result) == 0
    assert np.shape(result) == ()
    assert not isinstance(result, np.ndarray)
    assert isinstance(result, np.float64)
    assert result == np.inf


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


def test_b9_the_stable_case_returns_zero_while_the_docstring_says_infinity():
    """B9, second leg: ``ctau = inf`` never reaches the ``inf`` branch.

    The method's own docstring promises "infinity (np.inf), if particle is
    stable", but ``mass / inf / p_lab`` is 0.0 in IEEE arithmetic and raises
    nothing, so a stable particle gets a vector of *zeros* -- the opposite
    end of the scale. That is the physically right value for an inverse decay
    length, so this pin is on the docstring, not the number: measured 0.0 for
    p, e-, gamma.

    Correct behaviour: the docstring's "or infinity" clause goes, or is
    rewritten to say the ``inf`` return is the ``ctau == 0`` case.
    """
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
    assert np.ndim(k0.inverse_decay_length()) == 0
    assert k0.inverse_decay_length() == np.inf


def test_b9_the_except_branch_depends_on_mass_and_ctau_being_python_floats():
    """B9, boundary: the whole defect hangs on ``float`` vs ``np.float64``.

    ``self.mass`` and ``self.ctau`` come off the particletools table as Python
    floats (measured: ``type(...) is float``), so ``mass / ctau / p_lab`` is
    Python's division and ``ctau = 0.0`` raises ZeroDivisionError before numpy
    is reached. With ``ctau = np.float64(0.0)`` nothing raises at all: the
    healthy path returns ``array([inf] * d)`` -- the shape B9's fix is supposed
    to produce -- and with ``mass`` numpy-zero too it returns ``array([nan] *
    d)``. Recorded because a fix that reaches for ``np.errstate`` or coerces
    the scalars, rather than repairing the ``ones_like`` argument, changes
    which of these three answers a caller gets without touching the pin above.
    """
    p = particle(211)
    assert type(p.mass) is float
    assert type(p.ctau) is float

    p.ctau = 0.0
    p.mass = 0.0
    assert p.inverse_decay_length() == np.inf
    assert np.ndim(p.inverse_decay_length()) == 0

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


def test_b10_local_mbarn2cm2_is_the_inverse_of_the_class_constant():
    """B10: two constants share the name `mbarn2cm2` with reciprocal values.

    ``MCEqParticle.prod_cross_section`` re-derives the unit factors method-
    locally (particle.py) and sets ``mbarn2cm2 = GeV2mbarn / GeVcm**2`` — the
    cm^2->mb direction — while ``InteractionCrossSections.mbarn2cm2``
    (cross_sections.py, the class attribute) is ``GeVcm**2 / GeV2mbarn``, the
    mb->cm^2 direction the name states. The *result* is correct either way:
    multiplying cm^2 by the inverse factor converts to mb, which is what the
    ``if mbarn`` branch wants. Pinned numerically so the rename/dedup commit
    cannot change a number: prod_cross_section(mbarn=True) equals
    cs / InteractionCrossSections.mbarn2cm2 exactly, and the two constants
    are reciprocals of each other.

    Correct behaviour (R3: pin, then the dedup): one set of constants — the
    ``InteractionCrossSections`` class attributes, already the single copy
    after §13.2 item 4 — reached from ``prod_cross_section``, with the local
    inverse either deleted (branch divided by the class constant) or renamed
    to what it computes (``cm22mbarn``). The value the method returns must
    not move; this test's numbers are the regression.
    """
    from MCEq.data import InteractionCrossSections

    p = particle(211)
    p.cs = 3.0e-27  # an arbitrary hadronic cross section in cm^2

    GeVfm = 0.19732696312541853
    GeVcm = GeVfm * 1e-13
    GeV2mbarn = 10.0 * GeVfm**2
    cm2_per_mbarn = GeVcm**2 / GeV2mbarn  # what the name should mean

    assert p.prod_cross_section(mbarn=False) == 3.0e-27
    assert p.prod_cross_section(mbarn=True) == 3.0e-27 * (GeV2mbarn / GeVcm**2)
    # ...which is the exact reciprocal of the correctly-named class constant:
    assert InteractionCrossSections.mbarn2cm2 == cm2_per_mbarn
    # rel tol 1e-12: the two routes differ in the last ulp of the float chain
    assert p.prod_cross_section(mbarn=True) == pytest.approx(
        3.0e-27 / InteractionCrossSections.mbarn2cm2, rel=1e-12
    )
