"""The ETD2RK step loop and the entry point that drives it.

:func:`etd2_driver` is one loop for every route -- paraxial and
sec(theta)-coupled, single axis, shared-path multi-RHS and the LPT carousel
-- on numpy, MKL, Accelerate and CUDA.
:func:`MCEq.operator_assembly.compile_operator` prepares the operator; a
backend object places it on its library / device and executes the stages
there. Nothing else differs between backends. :func:`solve_etd2` is the
single route from the matrices of ``MatrixBuilder`` to a solution: compile,
bind the backend named in :data:`_BACKENDS`, run the loop, release the
handles.

Stage logic and route dispatch belong here. A stage's implementation on a
particular library or device belongs in :mod:`MCEq.solvers.backends`; the
scalar formulas it calls belong in :mod:`MCEq.solvers.numerics`.
"""

import numpy as np

from MCEq.misc import info
from MCEq.operator_assembly import compile_operator
from MCEq.solvers.backends import (
    accelerate_backend,
    cuda_backend,
    mkl_backend,
    numpy_backend,
)


def etd2_driver(
    nsteps, dX, rho_inv, be, phi, grid_idcs, schedule=None, phi0_per_pixel=None
):
    """ETD2RK step loop — every route, every backend.

    Integrates ``dPhi/dX = (A + ri B) S Phi`` with the Cox–Matthews
    exponential RK2: the diagonal ``D`` of ``A + ri B`` is treated exactly
    through an integrating factor, the off-diagonal remainder explicitly.
    ``S = I`` is the paraxial transport. With the sec(theta) transport
    ``S = I + T Pi``, where ``T`` is the constant Hankel-space representation
    of multiplication by ``min(sec theta, sec theta_cap)`` (see
    :mod:`MCEq.secant`) and ``Pi`` the projector onto the state columns with
    ``E_kin < config.secant_theta_e_max``. The operator is then split, per
    state column i in the support of Pi and the coupled mode subspace P
    (S_P = (I+T)[P,P]):

      exact slot   D0_i S_P      D0_i = diagonal of A + ri B at kappa = 0
                                 (the k-independent part)
      remainder    everything else: off-diagonal production acting on
                   w = S Phi, the k-dependent diagonal spread on the
                   coupled rows, and the one-way cross coupling from the
                   uncoupled modes.

    The exact slot is evaluated in the eigenbasis ``S_P = V diag(lam)
    V^-1`` (constant, shared by every column), where exp/phi1/phi2 of
    ``h D0_i S_P`` are elementwise. Unconditionally stable at any
    stiffness; stitching the coupling into the CSR instead puts stiff
    coupled loss terms in the explicit part and diverges.

    The operator behind ``be`` is a :class:`~MCEq.operator_assembly.
    CompiledOperator`; with coupling, the state lives in its low-E-first
    layout (``phi`` and the results are in the original layout), the
    coupled plane is the corner block ``x.reshape(n_k, N, K)[:n_P, :n_g]``
    and the operand of ``T_P`` the low-E block ``[:, :n_g]`` — strided
    views.

    Stages per step (state x = Phi_n, corner C(.) of a full-state array;
    the corner terms are absent for the paraxial transport):

      1. factors      eD, hphi1, hphi2 = f(h D), D = d_int + ri d_dec, with
                      the step size folded into the phi factors
                      (:mod:`MCEq.solvers.numerics`); block factors
                      f(h D0_i lam_j) on the corner, without it
      2. operand      x_c = C(x); Y = T_P x[:, G];  C(x) <- x_c + Y
                      (x now holds w = S x)
      3. remainder    F = A_off w + ri B_off w  (SpMM);
                      C(F) += Df (x_c + Y) - D0 (x_c + T_PP x_c)
      4. predictor    a = eD x + hphi1 F on the full state;
                      C(a) = V eDB Vi x_c + h V phi1B Vi C(F)
      5. operand and remainder (2-3) at a: a_c = C(a), Y_a, F_a
      6. corrector    x = a + hphi2 (F_a - F) on the full state;
                      C(x) = a_c + h V phi2B Vi (C(F_a) - C(F))
      7. harvest      carousel harvest/reset, or int_grid snapshot

    In 4 and 6 the full-state formula also writes the corner, using the
    operand w there instead of the state; that block is discarded and
    replaced by the exact-slot update, so no copy of the state is needed
    to form w. Batching: ``phi`` is ``(dim,)`` or ``(dim, K)``; ``dX`` /
    ``rho_inv`` are ``(nsteps,)`` (one shared path, the multi-RHS route)
    or ``(nsteps, K)`` (per-lane paths; lanes with ``h == 0`` are pinned
    to exact identity). A :class:`MCEq.solvers.schedule.CarouselSchedule`
    with
    ``phi0_per_pixel`` turns the per-lane form into the LPT carousel and
    the return value into the ``(dim, K_total)`` per-pixel solution.
    Single-axis is K = 1 without a schedule.

    Precision: the loop carries the step size and ``rho_inv`` twice, in
    fp64 for stage 1 (``h64`` / ``ri64``, which feed the diagonals and the
    exact slot) and in ``be.dtype`` for the stages that touch the state --
    ``ri`` for the SpMM, and ``h`` for the coupled corner, which is the one
    place the step size still multiplies state-dtype arrays. At fp64 they
    are the same values. See
    :data:`MCEq.solvers.backends.base._PRECISION_CONTRACT`.
    """
    xp = be.xp
    dtype = be.dtype
    fp64 = dtype == np.float64
    lay = be.op.layout
    coupled = lay.coupled
    phi = np.asarray(phi)
    batched = phi.ndim == 2
    per_lane = np.ndim(dX) == 2
    dim = phi.shape[0]
    K = phi.shape[1] if batched else 1
    if per_lane and not batched:
        raise ValueError("etd2_driver: (nsteps, K) dX requires a (dim, K) state")
    if schedule is not None:
        if not per_lane:
            raise ValueError(
                "etd2_driver: a carousel schedule requires (T, K) dX / rho_inv"
            )
        if grid_idcs:
            raise ValueError(
                "etd2_driver: carousel runs do not support int_grid snapshots"
            )
        if phi0_per_pixel.shape != (dim, schedule.K_total):
            raise ValueError(
                "etd2_driver: phi0_per_pixel must be (dim, K_total) "
                f"= ({dim}, {schedule.K_total}); got {phi0_per_pixel.shape}"
            )

    be.bind(dim, K, per_lane, nsteps)
    phc, F_phi, F_a, a = be.state_buffers(dim, K)

    def to_layout(x):
        return x if lay.perm is None else x[xp.asarray(lay.perm)]

    def from_layout(x):
        return x if lay.inv_perm is None else x[xp.asarray(lay.inv_perm)]

    phc[:] = to_layout(be.asarray(phi.reshape(dim, K)))

    if per_lane:
        dX_64 = be.asarray(dX, np.float64)
        ri_64 = be.asarray(rho_inv, np.float64)
        dX_b = dX_64 if fp64 else be.asarray(dX)
        ri_b = ri_64 if fp64 else be.asarray(rho_inv)

    if coupled:
        n_k, N, n_P, n_g = lay.n_k, lay.N, lay.n_P, lay.n_g
        T_P, T_PP, V, Vi, lam = be.coupling()
        lmm = be.left_matmul

        def corner(x):
            return x.reshape(n_k, N, K)[:n_P, :n_g]

        def low_e(x):
            return x.reshape(n_k, N, K)[:, :n_g]

        # Constant diagonals of the coupled plane and of the kappa = 0 mode.
        d_int_c, d_dec_c = (d.reshape(n_k, N)[:n_P, :n_g] for d in (be.d_int, be.d_dec))
        d_int_0, d_dec_0 = (d.reshape(n_k, N)[0, :n_g] for d in (be.d_int, be.d_dec))

        plane = (n_P, n_g, K)
        Y = xp.empty(plane, dtype=dtype)
        x_c = xp.empty(plane, dtype=dtype)
        a_c = xp.empty(plane, dtype=dtype)
        F_c = xp.empty(plane, dtype=dtype)
        tmp = xp.empty(plane, dtype=dtype)
        mode_tmp = xp.empty(plane, dtype=dtype)

        def block_action(factors, source, out):
            """``V diag(factors) V^-1 source`` on a coupled plane."""
            lmm(Vi, source, out=mode_tmp)
            xp.multiply(factors, mode_tmp, out=mode_tmp)
            lmm(V, mode_tmp, out=out)

        def eval_F(x, x_c, F, ri, Df, D0):
            """Stages 2-3 at the state x whose corner is held in x_c: F <- the
            remainder at x; x itself becomes the operand w = S x."""
            lmm(T_P, low_e(x), out=Y)
            xp.add(x_c, Y, out=corner(x))
            be.apply_off(x, F, ri)
            lmm(T_PP, x_c, out=tmp)
            xp.add(x_c, tmp, out=tmp)
            xp.multiply(Df, corner(x), out=Y)
            xp.multiply(D0, tmp, out=tmp)
            xp.subtract(Y, tmp, out=Y)
            xp.add(corner(F), Y, out=corner(F))

    if schedule is not None:
        sol_pixel = xp.empty((dim, schedule.K_total), dtype=dtype)
        phi0_pp = to_layout(be.asarray(phi0_per_pixel))
        rs, cs = schedule.reset_t_starts, schedule.record_t_starts
        rj, rp = (xp.asarray(schedule.reset_j), xp.asarray(schedule.reset_pixel))
        cj, cpix = (xp.asarray(schedule.record_j), xp.asarray(schedule.record_pixel))

    grid_sol = []
    grid_step = 0

    from time import time

    start = time()

    # See :data:`MCEq.solvers.path._EM_BLOWUP_CAVEAT` for the errstate contract.
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(nsteps):
            # 1. diagonal factors of the full state and of the exact slot
            if per_lane:
                h64, ri64 = dX_64[k], ri_64[k]  # (K,) lane rows, fp64
                ri = ri_b[k]  # the same row in the state dtype
            else:
                h64, ri64 = np.float64(dX[k]), np.float64(rho_inv[k])
                ri = dtype(ri64)
            eD, hphi1, hphi2 = be.diag_factors(h64, ri64)
            if coupled:
                # The exact slot scales the coupled corner by h after the
                # eigenbasis GEMMs, so it is the one stage that still needs
                # the step size in the state dtype.
                if per_lane:
                    h_c = dX_b[k][None, None, :]
                    frozen = (h64 == 0.0)[None, None, :]
                    Df = d_dec_c[:, :, None] * ri64 + d_int_c[:, :, None]
                    D0 = d_dec_0[:, None] * ri64 + d_int_0[:, None]
                    ZB = lam[:, None, None] * (D0 * h64)
                    D0_b = D0[None]
                    eDB, phi1B, phi2B = be.block_factors(ZB)
                else:
                    h_c = dtype(h64)
                    Df = (d_dec_c * ri64 + d_int_c)[:, :, None]
                    D0 = d_dec_0 * ri64 + d_int_0
                    ZB = lam[:, None] * (D0 * h64)
                    D0_b = D0[None, :, None]
                    eDB, phi1B, phi2B = (f[:, :, None] for f in be.block_factors(ZB))

            # 2-3. operand and remainder at the state
            if coupled:
                xp.copyto(x_c, corner(phc))
                eval_F(phc, x_c, F_phi, ri, Df, D0_b)
                xp.copyto(F_c, corner(F_phi))
            else:
                be.apply_off(phc, F_phi, ri)

            # 4. predictor a = eD x + hphi1 F, exact slot on the corner
            be.predictor(eD, phc, hphi1, F_phi, out=a)
            if coupled:
                block_action(eDB, x_c, a_c)
                block_action(phi1B, F_c, tmp)
                xp.multiply(tmp, h_c, out=tmp)
                xp.add(a_c, tmp, out=a_c)
                if per_lane:
                    xp.copyto(a_c, x_c, where=frozen)
                xp.copyto(corner(a), a_c)

            # 5. operand and remainder at the predictor
            if coupled:
                eval_F(a, a_c, F_a, ri, Df, D0_b)
            else:
                be.apply_off(a, F_a, ri)

            # 6. corrector x = a + hphi2 (F_a - F), exact slot on the corner
            be.corrector(a, F_a, F_phi, hphi2, out=phc)
            if coupled:
                xp.subtract(corner(F_a), F_c, out=tmp)
                block_action(phi2B, tmp, tmp)
                xp.multiply(tmp, h_c, out=tmp)
                xp.add(a_c, tmp, out=tmp)
                if per_lane:
                    xp.copyto(tmp, x_c, where=frozen)
                xp.copyto(corner(phc), tmp)

            # 7. harvest finished lanes BEFORE the reset reloads them
            if schedule is not None:
                lo, hi = int(cs[k]), int(cs[k + 1])
                if hi > lo:
                    sol_pixel[:, cpix[lo:hi]] = phc[:, cj[lo:hi]]
                lo, hi = int(rs[k]), int(rs[k + 1])
                if hi > lo:
                    phc[:, rj[lo:hi]] = phi0_pp[:, rp[lo:hi]]
            elif grid_idcs and grid_step < len(grid_idcs) and grid_idcs[grid_step] == k:
                grid_sol.append(phc.copy())
                grid_step += 1

    be.synchronize()
    elapsed = time() - start
    info(
        2,
        f"Performance ({be.name} K={K}): "
        f"{1e3 * elapsed / float(nsteps):6.2f}ms/iteration "
        f"({1e3 * elapsed / float(nsteps) / float(K):6.3f}ms/iteration/RHS)",
    )

    if schedule is not None:
        return be.to_host(from_layout(sol_pixel))
    sol = be.to_host(from_layout(phc))
    grid = np.array([])
    if grid_sol:
        grid = xp.stack(grid_sol)
        grid = be.to_host(
            grid if lay.inv_perm is None else grid[:, xp.asarray(lay.inv_perm)]
        )
    if not batched:
        sol = sol[:, 0]
        if grid.size:
            grid = grid[:, :, 0]
    return sol, grid


# --- the entry point -------------------------------------------------------


#: Backend factories :func:`solve_etd2` binds by name. Each takes the
#: compiled operator, the device and the state precision; the host
#: backends ignore the device.
_BACKENDS = {
    "numpy": lambda op, device_id, fp_precision: numpy_backend(op, fp_precision),
    "mkl": lambda op, device_id, fp_precision: mkl_backend(
        op, fp_precision=fp_precision
    ),
    "accelerate": lambda op, device_id, fp_precision: accelerate_backend(
        op, fp_precision=fp_precision
    ),
    "cuda": cuda_backend,
}


def solve_etd2(
    nsteps,
    dX,
    rho_inv,
    int_m,
    dec_m,
    phi,
    grid_idcs=(),
    *,
    backend="numpy",
    sec_ops=None,
    schedule=None,
    phi0_per_pixel=None,
    device_id=0,
    fp_precision=64,
):
    """Integrate ``dPhi/dX`` with :func:`etd2_driver` on one backend.

    The single route from the matrices of ``MatrixBuilder`` to a solution:
    compile the operator, bind ``backend`` to it, run the step loop, and
    release the backend's library handles / device buffers. Every route the
    driver offers is a keyword here, not a separate function.

    Args:
      nsteps, dX, rho_inv: the integration path. ``dX`` / ``rho_inv`` are
        ``(nsteps,)`` for a shared path or ``(nsteps, K)`` per lane.
      int_m, dec_m: interaction and decay matrices in the state's own order.
      phi: initial state, ``(dim,)`` or ``(dim, K)``; the solution has the
        same rank.
      grid_idcs: step indices to snapshot.
      backend: ``"numpy"``, ``"mkl"``, ``"accelerate"`` or ``"cuda"``.
      sec_ops: sec(theta) operator set of :mod:`MCEq.secant`, or ``None``
        for the paraxial transport.
      schedule, phi0_per_pixel: LPT carousel of
        :func:`MCEq.solvers.schedule.compile_carousel_schedule`. With a
        schedule the step count is
        the schedule's own ``T`` and ``nsteps`` is ignored; the return is the
        harvested ``(dim, K_total)`` pixel matrix rather than a pair.
      device_id: CUDA device index; ignored by the host backends.
      fp_precision: 32 or 64, the state precision on every backend — the
        state, the scratch buffers and the off-diagonal operator are
        stored in it while the diagonals and the phi factors stay fp64.
        See :data:`MCEq.solvers.backends.base._PRECISION_CONTRACT`.

    Returns:
      ``(solution, grid_snapshots)``, or the pixel matrix with a schedule.
    """
    try:
        make_backend = _BACKENDS[backend]
    except KeyError:
        raise ValueError(
            f"solve_etd2: unknown backend {backend!r}; "
            f"choose one of {', '.join(sorted(_BACKENDS))}"
        ) from None
    be = make_backend(compile_operator(int_m, dec_m, sec_ops), device_id, fp_precision)
    try:
        return etd2_driver(
            schedule.T if schedule is not None else nsteps,
            dX,
            rho_inv,
            be,
            phi,
            grid_idcs,
            schedule=schedule,
            phi0_per_pixel=phi0_per_pixel,
        )
    finally:
        be.close()
