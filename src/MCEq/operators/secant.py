"""sec(theta) path-elongation operator for the 2D transport.

Problem: the 2D solver books interaction, decay and continuous losses per
unit AXIS-projected slant depth X, while a particle travelling at angle
theta to the shower axis physically crosses ``sec(theta)`` more air (the
paraxial approximation). For populations in local production/loss
equilibrium (sub-GeV hadrons and muons) this over-predicts the wide-angle
density by exactly ``sec(theta)``.

Solution: multiplication by ``g(mu) = 1 / max(mu, cos theta_cap)`` in angle
space is the constant mode-coupling matrix ``S = I + T = M[g]`` on the
Legendre moments (:func:`MCEq.operators.legendre.path_elongation`, exact by
quadrature). The corrected transport right-multiplies the per-mode blocks::

    da_l/dX = M_l  sum_l' (I + T)_{ll'} a_l'

so the elongation acts on the flux entering the transport operator: it
rescales the parent's path length, not the daughter's emission angle. In
equilibrium the parent density becomes ``source x lambda x cos(theta)``
while its interaction/decay rate per unit axis depth grows by
``sec(theta)``; the product (the daughter production rate) is
unchanged, so loss-free daughters (neutrinos) are preserved identically.
The correction must cover ALL loss channels of a species or
none: elongating the ionization path but not the decay path would change
the mu -> nu balance by cos(theta).

All angles are relative to the shower axis, like the angular modes
themselves, so the operator and the choice of ``theta_cap`` are
independent of the axis' zenith angle.

The coupling is applied in the step loop, not stitched into the CSR: the
coupled same-(species, E) block ``d_i * S_P`` is integrated exactly in a real
basis of ``S_P`` (:func:`MCEq.operators.compiled.compile_operator`), which
reproduces the S-corrected equilibrium in the stiff limit.
"""

import numpy as np

from MCEq.misc import info

#: Max ``|imag|/real`` accepted for an eigenvalue of ``S_P``.
_PAIR_SPLIT_MAX = 1e-5

#: Relative gap between consecutive eigenvalues of ``S_P`` (sorted by real
#: part) below which they form one cluster.
_CLUSTER_RTOL = 3 * _PAIR_SPLIT_MAX


def _canonical_basis(S_P, lam, V, l_P):
    """A real basis of ``S_P`` that does not depend on round-off.

    Directions beyond the cap all see the same ``sec(theta_cap)``, so a large
    cluster of ``S_P`` is degenerate (33 of 65 eigenvalues on the 90-mode
    grid at 75 deg) and LAPACK returns an arbitrary basis of it, which
    changes with the BLAS thread count; so does the phase of a complex
    eigenvector, and with it the ``[Re v, Im v]`` basis of a conjugate pair.
    The basis decides how the muon mode damping splits into the exactly
    integrated diagonal and the remainder.

    Each cluster of eigenvalues (``_CLUSTER_RTOL``) is spanned instead by the
    orthonormal invariant subspace of an ordered real Schur form, rotated to
    diagonalise its projection of ``diag(l (l + 1))``, the shape of the
    multiple-scattering damping. Columns have unit norm and their largest
    component positive; the order is by eigenvalue. The spread within a
    cluster stays off the diagonal.

    Returns ``(V, Vi, lam)`` with ``lam = diag(Vi S_P V)``.
    """
    import scipy.linalg

    n = len(lam)
    order = np.lexsort((lam.imag, lam.real))
    lam, V = lam[order], V[:, order]
    breaks = np.flatnonzero(np.abs(np.diff(lam)) > _CLUSTER_RTOL * np.abs(lam[:-1]))
    Vr = np.empty((n, n))
    damping = l_P * (l_P + 1.0)
    for g in np.split(np.arange(n), breaks + 1):
        if len(g) == 1:
            Vr[:, g[0]] = V[:, g[0]].real
            continue
        c = lam[g].real.mean()
        reach = np.abs(lam[g] - c).max() * (1 + 1e-6) + 1e-12 * c
        _, Z, sdim = scipy.linalg.schur(
            S_P, output="real", sort=lambda re, im: abs(complex(re, im) - c) <= reach
        )
        if sdim != len(g):
            raise RuntimeError(
                f"secant coupling: cluster at {c:.6f} has {len(g)} eigenvalues "
                f"but a {sdim}-dimensional invariant subspace"
            )
        Q = Z[:, :sdim]
        _, U = np.linalg.eigh(Q.T @ (damping[:, None] * Q))
        Vr[:, g] = Q @ U
    Vr /= np.linalg.norm(Vr, axis=0)
    Vr *= np.sign(Vr[np.abs(Vr).argmax(axis=0), np.arange(n)])
    Vi = np.linalg.inv(Vr)
    return Vr, Vi, np.einsum("ij,ji->i", Vi, S_P @ Vr)


def legendre_coupling(l_grid, theta_cap_deg):
    """``T = G - 1`` on the dense rows of the Legendre grid, zero elsewhere.

    ``G`` is the exact multiplication by ``1 / max(mu, mu_cap)`` of
    :func:`MCEq.operators.legendre.path_elongation`. Its dense rows reach
    far beyond the dense block (``g`` has a kink at ``mu_cap``), so they are
    taken against every ``l <= l_grid[-1]`` and mapped onto all modes by the
    cubic interpolation in ``l`` of
    :func:`MCEq.operators.legendre.l_interpolation`. The sparse rows stay
    zero: the sparse high-l modes describe narrow cones, which are not
    elongated.
    """
    from MCEq.operators.legendre import dense_count, l_interpolation, path_elongation

    l_grid = np.asarray(l_grid)
    n_d = dense_count(l_grid)
    l_int = np.arange(int(l_grid[-1]) + 1)
    T = np.zeros((len(l_grid), len(l_grid)))
    G = path_elongation(len(l_int) - 1, theta_cap_deg, n_rows=n_d)
    T[:n_d] = G @ l_interpolation(l_grid, l_int)
    T[:n_d, :n_d] -= np.eye(n_d)
    return T


def build_secant_kernel_ops(l_grid, e_centers, n_species, spec, theta_cap_deg=None):
    """Assemble the constant data the secant ETD2RK kernel needs.

    ``spec`` is the ``secant`` config group (``cap_deg``, ``e_max``);
    ``theta_cap_deg`` overrides ``spec.cap_deg`` and must lie in [0, 90).

    Returns a dict with::

      P          -- indices of the coupled modes (the dense block of l_grid)
      T_P        -- T restricted to coupled rows, all columns (n_P, n_k)
      T_PP       -- the (n_P, n_P) same-subspace block
      V, Vi, lam -- real basis of S_P = I + T_PP, ``lam = diag(Vi S_P V)``:
                    the eigenbasis, with every eigenvalue cluster (and
                    near-real pair a ± ib) spanned by a canonical basis of
                    its invariant subspace (:func:`_canonical_basis`); the
                    cluster spread and pair rotation stay off-diagonal
      low_e_idx  -- state columns (species x energy) with
                    E_kin < spec.e_max
      n_k        -- number of angular modes
    """
    from MCEq.operators.legendre import dense_count

    if theta_cap_deg is None:
        theta_cap_deg = spec.cap_deg
    theta_cap_deg = float(theta_cap_deg)
    T = legendre_coupling(l_grid, theta_cap_deg)
    P = np.arange(dense_count(l_grid))
    T_P = T[P, :]
    T_PP = T[np.ix_(P, P)]
    S_P = np.eye(len(P)) + T_PP
    lam, V = np.linalg.eig(S_P)
    split = np.abs(lam.imag) / np.abs(lam.real)
    if split.max() > _PAIR_SPLIT_MAX or lam.real.min() <= 0:
        raise RuntimeError(
            "secant coupling: S_P eigenvalues not positive-real "
            f"(min real {lam.real.min():.3e}, "
            f"max |imag|/real {split.max():.3e}) at "
            f"theta_cap_deg = {theta_cap_deg:g}."
        )
    V, Vi, lam = _canonical_basis(S_P, lam, V, l_grid[P])
    info(
        2,
        f"secant kernel ops: {len(P)} coupled modes, eig(S_P) in "
        f"[{lam.min():.3f}, {lam.max():.3f}], cond(V) "
        f"{np.linalg.cond(V):.1e}",
    )

    # e_max is optional: absent or None couples every energy.
    e_max = getattr(spec, "e_max", None)
    e_centers = np.asarray(e_centers)
    if e_max is None:
        low_e = np.ones(len(e_centers), dtype=bool)
    else:
        low_e = e_centers < e_max
    low_e_idx = np.where(np.tile(low_e, n_species))[0]

    return {
        "P": P,
        "T_P": np.ascontiguousarray(T_P),
        "T_PP": np.ascontiguousarray(T_PP),
        "V": np.ascontiguousarray(V),
        "Vi": np.ascontiguousarray(Vi),
        "lam": lam,
        "low_e_idx": low_e_idx,
        "n_k": len(l_grid),
    }
