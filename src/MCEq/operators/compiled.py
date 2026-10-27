"""Diagonal/off-diagonal split and secant layout of the operator for the ETD2RK step loop.

The off-diagonals are CSR with every row's nonzeros kept in build order, also
after the layout permutation, so every backend sums the same products in the
same order. The math is in :ref:`solver-math`.
"""

from types import SimpleNamespace

import numpy as np
import scipy.linalg
import scipy.sparse as sp


def split_diagonal(int_m, dec_m):
    """``(d_int, d_dec, int_off, dec_off)``: the diagonals of A and B and
    their off-diagonal remainders (explicit zeros dropped)."""
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

    The state is ``(n_k, N)`` row-major over (Legendre mode, species x
    energy). The coupling acts on the modes ``P`` (the leading dense block
    of the l grid) and on the low-E columns ``g``. Moving ``g`` to the
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


def coupled_corner(layout, a):
    """The coupled plane of a state-shaped array ``a`` as a strided view:
    ``a.reshape(n_k, N, ...)[:n_P, :n_g]`` for ``a`` of shape ``(dim,)`` or
    ``(dim, K)``."""
    return a.reshape((layout.n_k, layout.N) + a.shape[1:])[: layout.n_P, : layout.n_g]


def secant_coupling(sec_ops, schur=False):
    """The mode-coupling operators of the coupled corner in a real basis of
    ``S_P = I + T_PP`` (see :mod:`MCEq.operators.secant`), as contiguous fp64
    arrays: ``V``, ``Vi``, ``lam`` (the basis, its inverse and the exactly
    integrated diagonal) and ``W = [S_P V | T_P[:, P^c]]`` of shape
    ``(n_P, n_k)``, which maps the low-E block of the state onto the corner
    of the operand ``S Phi``. The basis is the eigenbasis of ``sec_ops``, or
    with ``schur=True`` the orthogonal real Schur basis ``S_P = V R V^T``,
    ``lam = diag(R)``, whose strictly upper part joins the remainder."""
    V, Vi, lam, T_P = (
        np.ascontiguousarray(sec_ops[k], dtype=np.float64)
        for k in ("V", "Vi", "lam", "T_P")
    )
    n_P = V.shape[0]
    if schur:
        R, V = scipy.linalg.schur(np.eye(n_P) + T_P[:, :n_P], output="real")
        Vi, lam = np.ascontiguousarray(V.T), np.diag(R).copy()
    W = np.hstack([V + T_P[:, :n_P] @ V, T_P[:, n_P:]])
    return SimpleNamespace(V=V, Vi=Vi, lam=lam, W=np.ascontiguousarray(W))


def _permute_csr(off, perm, inv_perm):
    """Symmetric row/column permutation of a CSR matrix that keeps each
    row's nonzeros in their original order."""
    off = off.tocsr()[perm]
    return sp.csr_matrix((off.data, inv_perm[off.indices], off.indptr), shape=off.shape)


class CompiledOperator:
    """The ETD2RK operator, assembled for the step loop (read-only).

    Attributes:
      dim: state dimension (``n_k * dim_states`` for 2D databases).
      d_int, d_dec: ``(dim,)`` fp64 diagonals of A and B in the layout.
      int_off, dec_off: CSR off-diagonals in the layout, preserving the
        original within-row entry order (column indices need not be sorted).
      layout: :func:`identity_layout` or :func:`secant_layout` namespace.
      coupling: :func:`secant_coupling` namespace, or ``None`` (paraxial).
      sec_ops: the operator set the coupling was built from (identity is
        the cache key of the device copies), or ``None``.
      band_off, mode_mix: with a Legendre database, the continuous-loss
        off-diagonal (kept out of ``int_off``) and the ``(n_D, n_D)``
        removal of backward production applied to the leading modes of
        ``int_off x + ri dec_off x``; ``None`` otherwise.
    """

    def __init__(
        self,
        d_int,
        d_dec,
        int_off,
        dec_off,
        layout,
        coupling,
        sec_ops,
        band_off=None,
        mode_mix=None,
    ):
        self.dim = int(d_int.shape[0])
        self.band_off = band_off
        self.mode_mix = mode_mix
        self.d_int = np.ascontiguousarray(d_int, dtype=np.float64)
        self.d_dec = np.ascontiguousarray(d_dec, dtype=np.float64)
        self.int_off = int_off
        self.dec_off = dec_off
        self.layout = layout
        self.coupling = coupling
        self.sec_ops = sec_ops
        #: :func:`column_bases` of the corner columns integrated in their
        #: own basis (e+- with multiple scattering), or ``None``.
        self.column_basis = None

    @property
    def coupled(self):
        return self.coupling is not None

    @property
    def split(self):
        return self.d_int, self.d_dec, self.int_off, self.dec_off

    @property
    def corner_diagonals(self):
        """``(d_int, d_dec)`` on the coupled plane, ``(n_P, n_g)`` fp64
        arrays: the l-dependent diagonal of the coupled modes."""
        return tuple(
            np.ascontiguousarray(coupled_corner(self.layout, d))
            for d in (self.d_int, self.d_dec)
        )

    @property
    def exact_slot_diagonals(self):
        """``(L_int, L_dec)``, ``(n_P, n_g)`` fp64 arrays: the diagonal of
        the exactly integrated coupled operator in the coupling basis.
        ``L_dec = lam_j d_dec_0i`` with ``D0`` the l = 0 row of the
        diagonals; ``L_int = lam_j d_int_0i`` plus the basis diagonal
        of the l-dependent part of ``d_int`` on the corner (the muon
        multiple-scattering damping), so that only its mode mixing is left
        to the remainder (:ref:`solver-math`)."""
        lay, c = self.layout, self.coupling
        lam = c.lam[:, None]
        d_int0, d_dec0 = (
            d.reshape(lay.n_k, lay.N)[0, : lay.n_g][None, :]
            for d in (self.d_int, self.d_dec)
        )
        delta = coupled_corner(lay, self.d_int) - d_int0
        # diagonal of Vi diag(delta_i) V for every column i
        damping = (c.Vi * c.V.T) @ delta * lam
        L_int, L_dec = lam * d_int0 + damping, lam * d_dec0
        cb = self.column_basis
        if cb is not None:
            L_int[:, cb.idx] = cb.lam
            L_dec[:, cb.idx] = 0.0
        return L_int, L_dec


def column_bases(op, columns, tol=1e-8):
    """Per-column exact bases of the coupled corner for the state columns
    ``columns`` (indices within one mode block).

    The corner block of column ``i`` is ``A_i = diag(d_i) S_P`` with ``d_i``
    the l-dependent diagonal of ``A`` (for e+- it carries the screened Coulomb
    damping, up to ~1e4 cm^2/g at low energy and high l). In the shared basis
    of ``S_P`` only the diagonal of ``Vi diag(d_i) V`` is integrated exactly and
    the rest is explicit, which is unstable for such rates. ``A_i`` is
    ``-|D| N G`` with ``N G`` the Legendre form of a positive multiplication
    operator, so it has a real, negative spectrum: each column gets its own
    eigenbasis, in which ``A_i`` is integrated exactly. A decay diagonal
    (``ri d_dec``, mode independent and small: low-energy muons) stays in the
    explicit remainder of these columns (``L_dec = 0``).

    Returns a namespace ``idx`` (corner column numbers), ``V, Vi, SPV``
    (``(m, n_P, n_P)``, ``SPV = S_P V``) and ``lam`` (``(n_P, m)``), or None
    when no column of ``columns`` is in the corner.
    """
    lay = op.layout
    g = np.asarray(op.sec_ops["low_e_idx"])
    idx = np.flatnonzero(np.isin(g, np.asarray(columns)))
    if idx.size == 0:
        return None
    d_int = coupled_corner(lay, op.d_int)[:, idx]
    T_P = np.asarray(op.sec_ops["T_P"], dtype=np.float64)
    n_P = lay.n_P
    S_P = np.eye(n_P) + T_P[:, :n_P]
    m = idx.size
    V, Vi = np.empty((m, n_P, n_P)), np.empty((m, n_P, n_P))
    lam = np.empty((n_P, m))
    for j in range(m):
        A = d_int[:, j, None] * S_P
        w, v = np.linalg.eig(A)
        if np.abs(w.imag).max() > tol * np.abs(w).max() or w.real.max() > 0:
            raise RuntimeError(
                f"column_bases: corner column {idx[j]} has a non-real or growing "
                f"spectrum (max imag {np.abs(w.imag).max():.3g}, max real {w.real.max():.3g})"
            )
        order = np.argsort(w.real)
        v = v[:, order].real
        v /= np.linalg.norm(v, axis=0)
        vi = np.linalg.inv(v)
        rec = np.abs(v @ (w.real[order, None] * vi) - A).max()
        if rec > 1e-6 * np.abs(A).max():
            raise RuntimeError(f"column_bases: column {idx[j]} basis error {rec:.3g}")
        V[j], Vi[j], lam[:, j] = v, vi, w.real[order]
    return SimpleNamespace(idx=idx, V=V, Vi=Vi, lam=lam, SPV=S_P @ V)


def compile_operator(
    int_m, dec_m, sec_ops=None, band=None, mode_mix=None, exact_columns=None
):
    """Assemble the ETD2RK operator from the matrices of ``MatrixBuilder``.

    Args:
      int_m, dec_m: interaction and decay matrices in the state's own order.
      sec_ops: sec(theta) operator set of :mod:`MCEq.operators.secant`; the
        split is then permuted into the layout of :func:`secant_layout` and
        the coupling operators attached. ``None``: paraxial transport.
      band, mode_mix: Legendre databases only. ``int_m`` is then the
        hadronic part (``MatrixBuilder.int_m_hadr``) and ``band`` the
        continuous-loss band; the band's diagonal joins ``d_int`` and its
        off-diagonal bypasses ``mode_mix``, since energy loss does not
        change the direction.
      exact_columns: state columns (within one mode block) whose coupled
        corner is integrated in its own basis (:func:`column_bases`).
    """
    d_int, d_dec, int_off, dec_off = split_diagonal(int_m, dec_m)
    band_off = None
    if band is not None:
        d_band, _, band_off, _ = split_diagonal(band, band)
        d_int = d_int + d_band
    dim = int(d_int.shape[0])
    if sec_ops is None:
        layout, coupling = identity_layout(dim), None
        int_off, dec_off = int_off.tocsr(), dec_off.tocsr()
        band_off = None if band_off is None else band_off.tocsr()
    else:
        layout, coupling = secant_layout(sec_ops, dim), secant_coupling(sec_ops)
        perm, inv_perm = layout.perm, layout.inv_perm
        d_int, d_dec = d_int[perm], d_dec[perm]
        int_off = _permute_csr(int_off, perm, inv_perm)
        dec_off = _permute_csr(dec_off, perm, inv_perm)
        if band_off is not None:
            band_off = _permute_csr(band_off, perm, inv_perm)
    op = CompiledOperator(
        d_int, d_dec, int_off, dec_off, layout, coupling, sec_ops, band_off, mode_mix
    )
    if op.coupled and op.exact_slot_diagonals[0].max() > 0:
        # The eigenvectors of a near-degenerate cluster of S_P are arbitrary;
        # a skewed basis turns part of the multiple-scattering damping into
        # an exactly integrated growth. In the orthogonal Schur basis the
        # damping slot is a convex combination of the mode damping.
        eig_max = op.exact_slot_diagonals[0].max()
        op.coupling = secant_coupling(sec_ops, schur=True)
        if op.exact_slot_diagonals[0].max() >= eig_max:
            op.coupling = coupling
    if op.coupled and exact_columns is not None and len(exact_columns):
        op.column_basis = column_bases(op, exact_columns)
    return op
