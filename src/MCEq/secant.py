"""Hankel-space sec(theta) path-elongation operator for the 2D transport.

Problem: the 2D solver books interaction, decay and continuous losses per
unit AXIS-projected slant depth X, while a particle travelling at angle
theta to the shower axis physically crosses ``sec(theta)`` more air (the
paraxial approximation). For populations in local production/loss
equilibrium (sub-GeV hadrons and muons) this over-predicts the wide-angle
density by exactly ``sec(theta)``.

Solution: multiplication by ``g(theta) = min(sec theta, sec theta_cap)``
in angle space is a constant mode-coupling matrix ``S = I + T`` in Hankel
space. The corrected transport right-multiplies the per-mode blocks::

    dF_k/dX = M_k  sum_k' (I + T)_{kk'} F_k'

so the elongation acts on the flux entering the transport operator: it
rescales the parent's path length, not the daughter's emission angle. In
equilibrium the parent density becomes ``source x lambda x cos(theta)``
while its interaction/decay rate per unit axis depth grows by
``sec(theta)``; the product — the daughter production rate — is
unchanged, so loss-free daughters (neutrinos) are preserved identically.
Consequently the correction must cover ALL loss channels of a species or
none: elongating the ionization path but not the decay path would change
the mu -> nu balance by cos(theta).

All angles are relative to the shower axis, like the Hankel modes
themselves, so the operator and the choice of ``theta_cap`` are
independent of the axis' zenith angle.

Construction of ``T`` (per-column ridge with a flat-state damping term)::

    obj(T) = || W (R T - diag(g-1) R) ||_F^2
             + || T (lam2 I + w_f 1 1^T)^(1/2) ||_F^2

with ``R`` the readout linear map (cubic kappa-oversampling + trapz, the
same convention as :func:`MCEq.hankel.inverse_hankel_legacy`) on a dense
theta grid, ``W`` the sqrt(theta dtheta) measure, and the rank-one term
damping ``T @ 1`` (kappa-flat = collimated states are not elongated).
Rows with ``kappa > row_kmax`` are zeroed: the correction has no support
at narrow angular scales. Accuracy of the operator: rms 1-2 % on
near-isotropic profiles, angle-integrated action exact to 0.1 %,
S@1 = 1 to 0.15 %, eig(S) real and positive.

Solver integration: NOT via the matrices. Stitching ``M_k S`` into the
block CSR puts stiff mode-coupled loss terms (short-lived species at
altitude, ``rate/rho`` unbounded) into ETD2RK's explicit part; the
stiff-limit update is then a Jacobi-like iteration with matrix
``D_S^-1 (S - D_S)``. Its spectral radius grows monotonically with the
cap — 0.28 / 0.46 / 0.57 at caps 50 / 60 / 65 deg, 1.07 at the default
75, 2.65 at 85 (current operator, 48-mode grid) — so at production
settings the iteration diverges, and even below the rho = 1 crossover
(~70 deg) the stitched form showed transient blow-ups at moderate
stiffness in prototyping. Instead the kernels
(``solv_numpy_etd2_secant`` / ``solv_mkl_etd2_secant`` /
``solv_cuda_etd2_secant``) treat the coupled same-(species,E) block
``d_i * S_P`` exactly through the eigendecomposition of ``S_P``
(constant, shared by every state), which is unconditionally stable at
any cap and reproduces the S-corrected equilibrium exactly in the
stiff limit.
"""

import numpy as np

from MCEq.misc import info

_T_CACHE = {}


def _readout_matrix(k_grid, theta, oversample_res=10, chunk=1000):
    """Linear map of ``convert_to_theta_space``: F(k_grid) -> f(theta)."""
    import scipy.special
    from scipy.interpolate import interp1d

    n_k = len(k_grid)
    pts = int(np.max(k_grid) * oversample_res)
    k_ov = np.linspace(k_grid.min(), k_grid.max(), pts)
    interp_cols = np.empty((pts, n_k))
    for i in range(n_k):
        e = np.zeros(n_k)
        e[i] = 1.0
        interp_cols[:, i] = interp1d(k_grid, e, kind="cubic")(k_ov)
    R = np.empty((len(theta), n_k))
    for lo in range(0, len(theta), chunk):
        hi = min(lo + chunk, len(theta))
        j0 = scipy.special.j0(np.outer(k_ov, theta[lo:hi])) * k_ov[:, None]
        R[lo:hi, :] = np.trapezoid(
            j0[:, :, None] * interp_cols[:, None, :], k_ov, axis=0
        )
    return R


def secant_coupling_matrix(
    k_grid,
    theta_cap_deg=75.0,
    row_kmax=50.0,
    lam_rel=1e-9,
    w_flat=1.0,
    n_theta=24001,
    entry_tol=1e-4,
    paths=None,
):
    """Return ``T`` such that ``S = I + T`` represents multiplication by
    ``min(sec theta, sec theta_cap)`` on the Hankel mode amplitudes.

    Rows with ``k_grid > row_kmax`` and entries below ``entry_tol`` are
    zeroed. The result is cached per (k_grid, parameters).

    ``paths`` is the ``paths`` config group; its ``data_dir`` holds the
    on-disk operator cache. Without it only the in-process cache applies.
    """
    key = (
        tuple(np.asarray(k_grid, dtype=float)),
        theta_cap_deg,
        row_kmax,
        lam_rel,
        w_flat,
        n_theta,
        entry_tol,
    )
    if key in _T_CACHE:
        return _T_CACHE[key]

    # Disk cache under paths.data_dir: the dense-grid construction takes
    # ~4 minutes. The MCEq version is part of the hash so a version bump
    # invalidates stale operators. Without paths there is nowhere to put
    # it and only the in-process cache applies.
    cache_dir = cache_file = None
    if paths is not None:
        import hashlib

        from MCEq.version import __version__

        digest = hashlib.sha256(repr((__version__, key)).encode()).hexdigest()[:16]
        cache_dir = paths.data_dir / "secant_cache"
        cache_file = cache_dir / f"T_{digest}.npy"
        if cache_file.exists():
            T = np.load(cache_file)
            _T_CACHE[key] = T
            return T

    k_grid = np.asarray(k_grid, dtype=np.float64)
    n_k = len(k_grid)
    theta = np.linspace(0, np.pi / 2, n_theta)
    g = 1.0 / np.cos(np.minimum(theta, np.radians(theta_cap_deg)))

    info(
        2,
        f"Building sec(theta) coupling operator: cap {theta_cap_deg} deg,"
        f" rows kappa <= {row_kmax}, lam_rel {lam_rel:g}",
    )
    R = _readout_matrix(k_grid, theta)
    w = np.gradient(theta)
    w[0] *= 0.5
    w[-1] *= 0.5
    sqw = np.sqrt(theta * w)
    R_w = sqw[:, None] * R
    Y_w = sqw[:, None] * ((g - 1.0)[:, None] * R)
    G = R_w.T @ R_w
    lam2 = lam_rel * np.linalg.svd(G, compute_uv=False).max()

    # Sylvester form G T + T (lam2 I + w_f 1 1^T) = R_w^T Y_w: rotate the
    # column space so the rank-one flat-damping term diagonalizes.
    ones = np.ones(n_k)
    RHS = R_w.T @ Y_w
    Q = np.eye(n_k)
    Q[:, 0] = ones / np.sqrt(n_k)
    Q, _ = np.linalg.qr(Q)
    if Q[0, 0] < 0:
        Q[:, 0] *= -1
    dvals = np.full(n_k, lam2)
    dvals[0] = lam2 + w_flat * n_k
    RHSQ = RHS @ Q
    Tp = np.empty((n_k, n_k))
    for j in range(n_k):
        Tp[:, j] = np.linalg.solve(G + dvals[j] * np.eye(n_k), RHSQ[:, j])
    T = Tp @ Q.T
    T[k_grid > row_kmax, :] = 0.0
    T[np.abs(T) < entry_tol] = 0.0
    _T_CACHE[key] = T
    if cache_file is not None:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, T)
        except OSError:
            pass
    return T


def build_secant_kernel_ops(
    k_grid, e_centers, n_species, spec, theta_cap_deg=None, paths=None
):
    """Assemble the constant data the secant ETD2RK kernel needs.

    ``spec`` is the ``secant`` config group (``cap_deg``, ``row_kmax``,
    ``lam_rel``, ``w_flat``, ``e_max``); ``paths`` is the ``paths`` group,
    forwarded to :func:`secant_coupling_matrix` for its disk cache.

    ``theta_cap_deg`` overrides ``spec.cap_deg``; it must be a number in
    [50, 90) (see the config docstring).

    Returns a dict with:
      P          -- indices of the coupled modes (kappa <= row_kmax)
      T_P        -- T restricted to coupled rows, all columns (n_P, n_k)
      T_PP       -- the (n_P, n_P) same-subspace block
      V, Vi, lam -- eigendecomposition of S_P = I + T_PP (real)
      low_e_idx  -- state columns (species x energy) with
                    E_kin < spec.e_max
      n_k        -- number of Hankel modes
    """
    if theta_cap_deg is None:
        theta_cap_deg = spec.cap_deg
    theta_cap_deg = float(theta_cap_deg)
    T = secant_coupling_matrix(
        np.asarray(k_grid, dtype=np.float64),
        theta_cap_deg=theta_cap_deg,
        row_kmax=spec.row_kmax,
        lam_rel=spec.lam_rel,
        w_flat=spec.w_flat,
        paths=paths,
    )
    k_grid = np.asarray(k_grid, dtype=np.float64)
    P = np.where(k_grid <= spec.row_kmax)[0]
    T_P = T[P, :]
    T_PP = T[np.ix_(P, P)]
    S_P = np.eye(len(P)) + T_PP
    lam, V = np.linalg.eig(S_P)
    if np.abs(lam.imag).max() > 1e-9 or lam.real.min() <= 0:
        raise RuntimeError(
            "secant coupling: S_P eigenvalues not positive-real "
            f"(min real {lam.real.min():.3e}, "
            f"max |imag| {np.abs(lam.imag).max():.3e}). This happens "
            f"for small caps (theta_cap_deg = {theta_cap_deg:g}): below "
            "~45 deg the coupling is nearly nilpotent and S_P becomes "
            "numerically defective. Use theta_cap_deg >= 50."
        )
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
        "n_k": len(k_grid),
    }
