"""Angular modes of a 2D database as seen by the operator compilation.

2D databases carry Legendre moments. Their operators compile the hadronic
part and the continuous-loss band separately, so that the removal of
backward production (:func:`MCEq.operators.legendre.backward_removal`) acts
on produced particles only; energy loss keeps the direction.
"""

from types import SimpleNamespace

from MCEq import config
from MCEq.operators.legendre import backward_removal, dense_count


def _removal(run):
    if not getattr(getattr(run, "_mceq_db", None), "is_2d", False):
        return False
    return bool(getattr(run, "_cfg", config).secant.legendre_removal)


def inputs(run):
    """``(int_m, dec_m)`` for :func:`MCEq.operators.compiled.compile_operator`."""
    if not _removal(run):
        return run.int_m, run.dec_m
    return run.matrix_builder.int_m_hadr, run.dec_m


def extras(run):
    """``band`` and ``mode_mix`` keywords of ``compile_operator``, empty for
    1D databases or with ``secant.legendre_removal`` off."""
    if not _removal(run):
        return {}
    n_d = dense_count(run._mceq_db.k_grid)
    mix = SimpleNamespace(M=backward_removal(n_d - 1), n_D=n_d, N=run.dim_states)
    return dict(band=run.matrix_builder.dEdx_band, mode_mix=mix)
