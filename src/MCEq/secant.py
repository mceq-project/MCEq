"""Hankel-space sec(theta) path-elongation operator for the 2D transport.

The 2D solver books interaction, decay and continuous losses per unit
AXIS-projected slant depth X, while a particle travelling at angle theta to
the shower axis physically crosses ``sec(theta)`` more air (the paraxial
approximation). For populations in local production/loss equilibrium
(sub-GeV hadrons and muons) this over-predicts the wide-angle density by
exactly ``sec(theta)`` — measured as C/M(theta) = cos(theta) against
CORSIKA-7, see mceq-em-integration
``wiki/results/lowe-excess-secant-theta.md``.

The fix: multiplication by ``g(theta) = min(sec theta, sec theta_cap)`` in
angle space is a constant mode-coupling matrix ``S = I + T`` in Hankel
space. The corrected transport right-multiplies the per-mode blocks::

    dF_k/dX = M_k  sum_k' (I + T)_{kk'} F_k'

i.e. the elongation charges the flux BEFORE the yield kick — the parent's
path, not the child's emission angle. In equilibrium the parent density
becomes ``source x lambda cos(theta)`` while its interaction/decay rate
per axis depth gains ``sec(theta)``; the product (daughter production) is
unchanged, so loss-free daughters (neutrinos), which already agree with
CORSIKA, are preserved identically. For the same reason the correction
must cover ALL loss channels of a species or none: correcting ionization
but not decay would break the mu -> nu balance by cos(theta).

Construction (per-column ridge with a flat-state damping term, prototype
v6 in mceq-em-integration ``runs/2026-08-10_secant-transport-kernel/``)::

    obj(T) = || W (R T - diag(g-1) R) ||_F^2
             + || T (lam2 I + w_f 1 1^T)^(1/2) ||_F^2

with ``R`` the readout linear map (cubic kappa-oversampling + trapz, the
same convention as ``MCEqRun.convert_to_theta_space``) on a dense theta
grid, ``W`` the sqrt(theta dtheta) measure, and the rank-one term damping
``T @ 1`` (kappa-flat = collimated states must pass through untouched).
Rows with ``kappa > row_kmax`` are zeroed: the correction has no business
at narrow angular scales. Validated: rms 1-2 % on near-isotropic sub-GeV
profiles, angle-integrated action exact to 0.1 %, S@1 = 1 to 0.15 %,
eig(S) real in [1.0, 3.9].

Solver integration: NOT via the matrices. Stitching ``M_k S`` into the
block CSR puts stiff mode-coupled loss terms (short-lived species at
altitude, ``rate/rho`` unbounded) into ETD2RK's explicit part; the
stiff-limit update is then a Jacobi-like iteration with matrix
``D_S^-1 (S - D_S)`` whose spectral radius is 1.75 — it NaNs. Instead the
kernel (``solv_numpy_etd2_secant``) treats the coupled same-(species,E)
block ``d_i * S_P`` exactly through the eigendecomposition of ``S_P``
(constant, shared by every state), which is unconditionally stable and
reproduces the S-corrected equilibrium exactly in the stiff limit.
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
):
    """Return ``T`` such that ``S = I + T`` represents multiplication by
    ``min(sec theta, sec theta_cap)`` on the Hankel mode amplitudes.

    Rows with ``k_grid > row_kmax`` and entries below ``entry_tol`` are
    zeroed. The result is cached per (k_grid, parameters).
    """
    key = (tuple(np.asarray(k_grid, dtype=float)), theta_cap_deg, row_kmax,
           lam_rel, w_flat, n_theta, entry_tol)
    if key in _T_CACHE:
        return _T_CACHE[key]

    # Disk cache: the dense-grid construction takes ~4 minutes.
    import hashlib
    import pathlib
    import tempfile

    digest = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
    cache_dir = pathlib.Path(tempfile.gettempdir()) / "mceq_secant_cache"
    cache_file = cache_dir / f"T_{digest}.npy"
    if cache_file.exists():
        T = np.load(cache_file)
        _T_CACHE[key] = T
        return T

    k_grid = np.asarray(k_grid, dtype=np.float64)
    n_k = len(k_grid)
    theta = np.linspace(0, np.pi / 2, n_theta)
    g = 1.0 / np.cos(np.minimum(theta, np.radians(theta_cap_deg)))

    info(2, f"Building sec(theta) coupling operator: cap {theta_cap_deg} deg,"
            f" rows kappa <= {row_kmax}, lam_rel {lam_rel:g}")
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
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, T)
    except OSError:
        pass
    return T


def build_secant_kernel_ops(k_grid, e_centers, n_species, config):
    """Assemble the constant data the secant ETD2RK kernel needs.

    Returns a dict with:
      P          -- indices of the coupled modes (kappa <= row_kmax)
      T_P        -- T restricted to coupled rows, all columns (n_P, n_k)
      T_PP       -- the (n_P, n_P) same-subspace block
      V, Vi, lam -- eigendecomposition of S_P = I + T_PP (real)
      gate_idx   -- state columns (species x energy) with E < e_gate
      n_k        -- number of Hankel modes
    """
    T = secant_coupling_matrix(
        np.asarray(k_grid, dtype=np.float64),
        theta_cap_deg=config.secant_theta_cap_deg,
        row_kmax=config.secant_theta_row_kmax,
        lam_rel=config.secant_theta_lam_rel,
        w_flat=config.secant_theta_w_flat,
    )
    k_grid = np.asarray(k_grid, dtype=np.float64)
    P = np.where(k_grid <= config.secant_theta_row_kmax)[0]
    T_P = T[P, :]
    T_PP = T[np.ix_(P, P)]
    S_P = np.eye(len(P)) + T_PP
    lam, V = np.linalg.eig(S_P)
    if np.abs(lam.imag).max() > 1e-9 or lam.real.min() <= 0:
        raise RuntimeError(
            "secant coupling: S_P eigenvalues not positive-real "
            f"(min real {lam.real.min():.3e}, "
            f"max |imag| {np.abs(lam.imag).max():.3e})"
        )
    lam = lam.real
    V = V.real
    Vi = np.linalg.inv(V)
    info(2, f"secant kernel ops: {len(P)} coupled modes, eig(S_P) in "
            f"[{lam.min():.3f}, {lam.max():.3f}], cond(V) "
            f"{np.linalg.cond(V):.1e}")

    e_gate = getattr(config, "secant_theta_e_gate", None)
    e_centers = np.asarray(e_centers)
    if e_gate is None:
        gate_e = np.ones(len(e_centers), dtype=bool)
    else:
        gate_e = e_centers < e_gate
    gate = np.tile(gate_e, n_species)
    gate_idx = np.where(gate)[0]

    return {
        "P": P,
        "T_P": np.ascontiguousarray(T_P),
        "T_PP": np.ascontiguousarray(T_PP),
        "V": np.ascontiguousarray(V),
        "Vi": np.ascontiguousarray(Vi),
        "lam": lam,
        "gate_idx": gate_idx,
        "n_k": len(k_grid),
    }
