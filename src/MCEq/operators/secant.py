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
``sec(theta)``; the product — the daughter production rate — is
unchanged, so loss-free daughters (neutrinos) are preserved identically.
Consequently the correction must cover ALL loss channels of a species or
none: elongating the ionization path but not the decay path would change
the mu -> nu balance by cos(theta).

All angles are relative to the shower axis, like the angular modes
themselves, so the operator and the choice of ``theta_cap`` are
independent of the axis' zenith angle.

Solver integration: NOT via the matrices. Stitching ``M_l S`` into the
block CSR puts stiff mode-coupled loss terms (short-lived species at
altitude, ``rate/rho`` unbounded) into ETD2RK's explicit part; the
stiff-limit update is then a Jacobi-like iteration with matrix
``D_S^-1 (S - D_S)``, whose spectral radius exceeds 1 at wide caps, so the
stitched iteration diverges there. Instead the step loop
(``MCEq.solvers.etd2_driver``, on any backend, once this operator set is
compiled into the ETD2 operator) treats the coupled same-(species,E) block
``d_i * S_P`` exactly through the eigendecomposition of ``S_P``
(constant, shared by every state), which is stable at any cap and
reproduces the S-corrected equilibrium exactly in the stiff limit.
"""

import numpy as np

from MCEq.misc import info


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

    Returns a dict with:
      P          -- indices of the coupled modes (the dense block of l_grid)
      T_P        -- T restricted to coupled rows, all columns (n_P, n_k)
      T_PP       -- the (n_P, n_P) same-subspace block
      V, Vi, lam -- eigendecomposition of S_P = I + T_PP (real)
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
    if np.abs(lam.imag).max() > 1e-9 or lam.real.min() <= 0:
        raise RuntimeError(
            "secant coupling: S_P eigenvalues not positive-real "
            f"(min real {lam.real.min():.3e}, "
            f"max |imag| {np.abs(lam.imag).max():.3e}) at "
            f"theta_cap_deg = {theta_cap_deg:g}."
        )
    # A conjugate pair with |imag| <= 1e-9 (a split of the degenerate
    # eigenvalue 1/mu_cap) spans its invariant plane with Re v and Im v; the
    # real parts of the pair alone would be one vector twice.
    pair = np.flatnonzero(lam.imag > 0)
    V = V.copy()
    V[:, pair + 1] = V[:, pair].imag
    lam = lam.real
    V = V.real
    Vi = np.linalg.inv(V)
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
