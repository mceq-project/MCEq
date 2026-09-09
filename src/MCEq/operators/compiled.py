"""Operator assembly for the ETD2RK solvers.

:class:`~MCEq.operators.matrix_builder.MatrixBuilder` produces the cascade
operator as two constant sparse matrices, ``A = int_m`` and ``B = dec_m``.
The ETD2RK step loop does not consume them directly: it integrates the
diagonal of ``A + ri B`` exactly and the off-diagonal explicitly, in a state
layout of its choosing, and with the sec(theta) transport it needs the
constant mode-coupling operators of :mod:`MCEq.operators.secant` alongside.
This module is the layer in between — host-only and backend-agnostic.
:func:`compile_operator` turns the matrices (and the optional coupling
operator set) into one :class:`CompiledOperator`, to be treated as read-only
while a backend or cache uses it; the backends in
:mod:`MCEq.solvers` place that object onto their library handles or device
and execute the step loop of :func:`MCEq.solvers.etd2_driver` against it.
:func:`em_step_scale` is the other property of ``A`` the step loop needs and
the matrices do not carry: the stiffness scale that caps ``dX``.

Numerics: the off-diagonals are CSR with every row's nonzeros kept in
their build order, also after the layout permutation (the column indices
are then no longer sorted within a row — deliberately: sorting would change
the summation order). Every backend therefore sums the same products in
the same order, and the cross-backend agreement of the kernels is a
property of this object.
"""

from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp


def split_diagonal(int_m, dec_m):
    """``(d_int, d_dec, int_off, dec_off)``: the diagonals of A and B and
    their off-diagonal remainders (explicit zeros dropped). Both pieces are
    constant in X — only ``ri`` modulates how they combine per step."""
    d_int = int_m.diagonal()
    d_dec = dec_m.diagonal()
    int_off = int_m - sp.diags(d_int, format=int_m.format)
    dec_off = dec_m - sp.diags(d_dec, format=dec_m.format)
    int_off.eliminate_zeros()
    dec_off.eliminate_zeros()
    return d_int, d_dec, int_off, dec_off


def identity_layout(dim):
    """The layout of the paraxial kernels: the state as built, no coupling."""
    return SimpleNamespace(
        n_k=1, N=dim, n_P=0, n_g=0, perm=None, inv_perm=None, coupled=False
    )


def secant_layout(sec_ops, dim):
    """Low-E-first state layout of the secant kernels.

    The state is ``(n_k, N)`` row-major over (Hankel mode, species x
    energy). The coupling acts on the modes ``P`` — a leading block, since
    kappa is sorted — and on the low-E columns ``g``. Moving ``g`` to the
    front of every mode block, ``x' = x[perm]``, makes the coupled plane
    the corner block ``x'.reshape(n_k, N)[:n_P, :n_g]``: a strided view,
    so the step needs no gather and no scatter. Returns a namespace with
    ``n_k, N, n_P, n_g, perm, inv_perm, coupled``.
    """
    n_k = int(sec_ops["n_k"])
    N = dim // n_k
    if n_k * N != dim:
        raise ValueError(f"secant_layout: dim {dim} is not divisible by n_k {n_k}")
    P = np.asarray(sec_ops["P"])
    g = np.asarray(sec_ops["low_e_idx"])
    if not np.array_equal(P, np.arange(len(P))):
        raise ValueError(
            "secant_layout: the coupled modes must be the leading block of "
            "the mode axis (P == arange(n_P))"
        )
    if g.size and not np.all(np.diff(g) > 0):
        raise ValueError("secant_layout: low_e_idx must be sorted and unique")
    order = np.concatenate([g, np.setdiff1d(np.arange(N), g)])
    perm = (np.arange(n_k)[:, None] * N + order[None, :]).ravel()
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(dim)
    return SimpleNamespace(
        n_k=n_k,
        N=N,
        n_P=len(P),
        n_g=len(g),
        perm=perm,
        inv_perm=inv_perm,
        coupled=True,
    )


def secant_coupling(sec_ops):
    """The constant mode-coupling operators as contiguous fp64 arrays:
    ``T_P`` (n_P, n_k), ``T_PP`` (n_P, n_P), the eigenbasis ``V``, ``Vi`` and
    eigenvalues ``lam`` of ``S_P = I + T_PP`` (see :mod:`MCEq.operators.secant`)."""
    return SimpleNamespace(
        **{
            k: np.ascontiguousarray(sec_ops[k], dtype=np.float64)
            for k in ("T_P", "T_PP", "V", "Vi", "lam")
        }
    )


def _permute_csr(off, perm, inv_perm):
    """Symmetric row/column permutation of a CSR matrix that keeps each
    row's nonzeros in their original order (the SpMM sums as before)."""
    off = off.tocsr()[perm]
    return sp.csr_matrix((off.data, inv_perm[off.indices], off.indptr), shape=off.shape)


class CompiledOperator:
    """The ETD2RK operator, assembled for the step loop.

    Treat the object as read-only while a backend or cache uses it; nothing
    here enforces that.

    Attributes:
      dim: state dimension (``n_k * dim_states`` for 2D databases).
      d_int, d_dec: ``(dim,)`` fp64 diagonals of A and B in the layout.
      int_off, dec_off: CSR off-diagonals in the layout, preserving the
        original within-row entry order (column indices need not be sorted).
      layout: :func:`identity_layout` or :func:`secant_layout` namespace.
      coupling: :func:`secant_coupling` namespace, or ``None`` (paraxial).
      sec_ops: the operator set the coupling was built from (identity is
        the cache key of the device copies), or ``None``.
    """

    def __init__(self, d_int, d_dec, int_off, dec_off, layout, coupling, sec_ops):
        self.dim = int(d_int.shape[0])
        self.d_int = np.ascontiguousarray(d_int, dtype=np.float64)
        self.d_dec = np.ascontiguousarray(d_dec, dtype=np.float64)
        self.int_off = int_off
        self.dec_off = dec_off
        self.layout = layout
        self.coupling = coupling
        self.sec_ops = sec_ops

    @property
    def coupled(self):
        return self.coupling is not None

    @property
    def split(self):
        return self.d_int, self.d_dec, self.int_off, self.dec_off


def compile_operator(int_m, dec_m, sec_ops=None):
    """Assemble the ETD2RK operator from the matrices of ``MatrixBuilder``.

    Without ``sec_ops`` this is the diagonal / off-diagonal split in the
    state's own order (paraxial transport). With ``sec_ops`` the split is
    permuted once into the low-E-first layout of :func:`secant_layout`
    and the coupling operators are attached. Pure function of its inputs;
    :class:`MCEq.core.MCEqRun` caches the result against the identity of
    the matrices and the operator set.
    """
    d_int, d_dec, int_off, dec_off = split_diagonal(int_m, dec_m)
    dim = int(d_int.shape[0])
    if sec_ops is None:
        layout, coupling = identity_layout(dim), None
        int_off, dec_off = int_off.tocsr(), dec_off.tocsr()
    else:
        layout, coupling = secant_layout(sec_ops, dim), secant_coupling(sec_ops)
        perm, inv_perm = layout.perm, layout.inv_perm
        d_int, d_dec = d_int[perm], d_dec[perm]
        int_off = _permute_csr(int_off, perm, inv_perm)
        dec_off = _permute_csr(dec_off, perm, inv_perm)
    return CompiledOperator(d_int, d_dec, int_off, dec_off, layout, coupling, sec_ops)


def em_step_scale(int_m, particles, dense_eig_max=4000):
    """EM stiffness-based step limit r_EM of the e+/-/gamma block of ``int_m``.

    Returns ``spec(int_off_EM)``: the spectral radius of the *off-diagonal*
    part of ``int_m`` restricted to the e+/-/gamma rows and columns, in
    1/(g/cm^2). This is the same quantity (``spec(int_off)``) that fixes the
    legacy ``config.etd2_path['dX_max']`` for the hadronic matrix, evaluated
    on the stiffer EM block. The spectral radius — not a matrix norm — is the
    right scale: the EM off-diagonal is a near-singular, highly non-normal
    difference+production operator whose 1-norm is dominated by the
    near-cancelling continuous-loss bands and is meaningless as a step scale.
    ``int_m`` is X-constant, so r_EM is one global scalar;
    :meth:`MCEq.driver.mceq_run.MCEqRun._em_cascade_step_scale` memoises it
    against the identity of the matrix it was taken from. Returns 0.0 when
    ``particles`` holds no e+/-/gamma (EM cascade inactive).

    ``particles`` are the cascade species, read for ``is_em`` and for the
    ``lidx``/``uidx`` of those that are. Those offsets index one ``dim_states``
    block, so on the ``(n_k * dim_states)`` operator the 2D assembly stitches
    this reads Hankel mode 0 alone. See the corresponding test in
    ``tests/test_operators_pin.py``.
    """
    col_ranges = [
        np.arange(p.lidx, p.uidx) for p in particles if getattr(p, "is_em", False)
    ]
    if not col_ranges:
        return 0.0
    idx = np.concatenate(col_ranges)
    diag = int_m.diagonal()
    int_off = (int_m - sp.diags(diag, format="csr")).tocsc()
    block = int_off[idx][:, idx].tocsc().astype(np.float64)
    # Spectral radius of the EM off-diagonal. The block is a small sub-system
    # (a few e+/-/gamma species x dim_e, typically < ~2000) and strongly
    # NON-NORMAL (||A||_2 / rho ~ 3). Sparse ``eigs(k=1, 'LM')`` routinely
    # FAILS to converge on it (ARPACK DNAUPD finds no eigenvalue to tolerance),
    # and an except-branch that substitutes a matrix norm then overestimates
    # the spectral radius AND is nondeterministic, capping dX tighter than
    # intended. Dense eigendecomposition is deterministic and cheap at this
    # size and avoids those convergence failures; it produces a numerical
    # estimate. ARPACK + norm survive only as a fallback for an unexpectedly
    # huge block.
    n = block.shape[0]
    try:
        if n <= int(dense_eig_max):
            ev = np.linalg.eigvals(block.toarray())
        else:
            from scipy.sparse.linalg import eigs

            ev = eigs(block, k=1, which="LM", return_eigenvectors=False, maxiter=10000)
        return float(np.max(np.abs(ev)))
    except Exception:
        # Last-resort fall-back (huge block + ARPACK failure): the spectral
        # radius is bounded above by both induced norms; take the smaller so we
        # never under-cap.
        n1 = float(np.abs(block).sum(axis=0).max())
        ninf = float(np.abs(block).sum(axis=1).max())
        return min(n1, ninf)
