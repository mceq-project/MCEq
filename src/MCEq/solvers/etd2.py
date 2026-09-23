"""The ETD2RK step loop and the entry point that drives it.

:func:`etd2_driver` is one loop for every route -- paraxial and
sec(theta)-coupled, single axis, shared-path multi-RHS and the LPT carousel
-- on numpy, MKL, Accelerate and CUDA.
:func:`MCEq.operators.compiled.compile_operator` prepares the operator; a
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
from MCEq.operators.compiled import compile_operator, coupled_corner
from MCEq.solvers.backends import (
    accelerate_backend,
    cuda_backend,
    mkl_backend,
    numpy_backend,
)

#: ``log2`` of the largest initial value the fp32 state carries.
_FP32_PEAK_EXP = 76


def _fp32_scale(phi, phi0_per_pixel=None):
    """The power of two ``s`` that puts ``max |phi|`` at
    ``2**_FP32_PEAK_EXP``: the fp32 state carries ``s phi``, which keeps the
    high-energy fluxes clear of the subnormals. 1.0 for a zero state."""
    peak = max(
        float(np.max(np.abs(p), initial=0.0))
        for p in (phi, phi0_per_pixel)
        if p is not None
    )
    if not np.isfinite(peak) or peak == 0.0:
        return 1.0
    return 2.0 ** (_FP32_PEAK_EXP - int(np.ceil(np.log2(peak))))


def etd2_driver(
    nsteps, dX, rho_inv, be, phi, grid_idcs, schedule=None, phi0_per_pixel=None
):
    """ETD2RK step loop — every route, every backend.

    Integrates ``dPhi/dX = (A + ri B) S Phi`` with the Cox–Matthews
    exponential RK2: the diagonal ``D`` exactly through its exp / phi
    factors, the off-diagonal remainder ``F`` explicitly. ``S = I`` is the
    paraxial transport; with the sec(theta) transport the coupled corner
    of the state is carried in the eigenbasis of the mode coupling, rotated
    in once before the loop and back once after it. Formulas and stage
    order: :ref:`solver-mathematics`.

    Args:
      nsteps: number of steps.
      dX, rho_inv: the path, ``(nsteps,)`` for one path shared by every
        lane or ``(nsteps, K)`` for one path per lane; a lane with
        ``h == 0`` is left unchanged.
      be: the backend, built on the compiled operator.
      phi: initial state in the state's own order, ``(dim,)`` or
        ``(dim, K)``.
      grid_idcs: step indices at which to snapshot the state.
      schedule, phi0_per_pixel: a
        :class:`MCEq.solvers.schedule.CarouselSchedule` and the per-pixel
        initial states ``(dim, K_total)``; requires per-lane paths and no
        snapshots.

    Returns:
      ``(solution, grid_snapshots)`` of ``phi``'s rank, or the harvested
      ``(dim, K_total)`` pixel matrix with a schedule.

    The step size and ``rho_inv`` reach the factor stage in fp64 and the
    SpMM in the state dtype
    (:data:`MCEq.solvers.backends.base._PRECISION_CONTRACT`).
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

    # fp32 carries s phi (see :func:`_fp32_scale`); undone on the way out
    scale = 1.0 if fp64 else _fp32_scale(phi, phi0_per_pixel)
    phc[:] = to_layout(be.asarray(scale * phi.reshape(dim, K)))

    # the path, uploaded once: fp64 for the factor stage, the state dtype
    # for the SpMM
    dX_64 = be.asarray(dX, np.float64)
    ri_64 = be.asarray(rho_inv, np.float64)
    ri_b = ri_64 if fp64 else be.asarray(rho_inv)

    if coupled:
        W, V, Vi = be.coupling()
        lmm = be.left_matmul

        def corner(x):
            return coupled_corner(lay, x)

        def low_e(x):
            return x.reshape((lay.n_k, lay.N) + x.shape[1:])[:, : lay.n_g]

        def rotate(x, M):
            """``C(x) <- M C(x)`` for a state-shaped ``x``."""
            xp.copyto(corner(x), lmm(M, corner(x)))

        # constant parts of the corner diagonals Df = d_int_c + ri d_dec_c
        # and lam_j D0_i = L_int + ri L_dec
        d_int_c, d_dec_c = (
            be.asarray(d, np.float64)[:, :, None] for d in be.op.corner_diagonals
        )
        L_int, L_dec = (
            be.asarray(d, np.float64)[:, :, None] for d in be.op.exact_slot_diagonals
        )
        Df = xp.empty((lay.n_P, lay.n_g, K if per_lane else 1), dtype=np.float64)
        D0lam = xp.empty_like(Df)
        plane = (lay.n_P, lay.n_g, K)
        psi = xp.empty(plane, dtype=dtype)
        work = xp.empty(plane, dtype=dtype)
        rot = xp.empty(plane, dtype=dtype)

        def eval_F(x, F, ri):
            """``F <- F(x)`` for a state ``x`` whose corner holds ``Psi``."""
            xc, Fc = corner(x), corner(F)
            xp.copyto(psi, xc)
            # the corner of the operand S x, in place of Psi around the SpMM
            lmm(W, low_e(x), out=work)
            xp.copyto(xc, work)
            be.apply_off(x, F, ri)
            # Vi (F_c + Df C(S x)) - lam_j D0_i Psi
            xp.multiply(Df, xc, out=work)
            xp.add(Fc, work, out=work)
            lmm(Vi, work, out=rot)
            xp.multiply(D0lam, psi, out=work)
            xp.subtract(rot, work, out=Fc)
            xp.copyto(xc, psi)

        rotate(phc, Vi)
    else:

        def eval_F(x, F, ri):
            be.apply_off(x, F, ri)

    if schedule is not None:
        sol_pixel = xp.empty((dim, schedule.K_total), dtype=dtype)
        phi0_pp = to_layout(be.asarray(scale * phi0_per_pixel))
        if coupled:
            rotate(phi0_pp, Vi)
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
            # 1. factors of the full state, corner included
            h64, ri64 = dX_64[k], ri_64[k]  # (K,) lane rows or one value, fp64
            ri = ri_b[k]  # the same in the state dtype
            eD, hphi1, hphi2 = be.diag_factors(h64, ri64)
            if coupled:
                xp.multiply(d_dec_c, ri64, out=Df)
                xp.add(Df, d_int_c, out=Df)
                xp.multiply(L_dec, ri64, out=D0lam)
                xp.add(D0lam, L_int, out=D0lam)

            # 2-4. remainder, predictor, remainder, corrector
            eval_F(phc, F_phi, ri)
            be.predictor(eD, phc, hphi1, F_phi, out=a)
            eval_F(a, F_a, ri)
            be.corrector(a, F_a, F_phi, hphi2, out=phc)

            # 5. harvest finished lanes BEFORE the reset reloads them
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
        if coupled:
            rotate(sol_pixel, V)
        return be.to_host(from_layout(sol_pixel)) / scale
    if coupled:
        rotate(phc, V)
        for snapshot in grid_sol:
            rotate(snapshot, V)
    sol = be.to_host(from_layout(phc)) / scale
    grid = np.array([])
    if grid_sol:
        grid = xp.stack(grid_sol)
        grid = (
            be.to_host(
                grid if lay.inv_perm is None else grid[:, xp.asarray(lay.inv_perm)]
            )
            / scale
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
      sec_ops: sec(theta) operator set of :mod:`MCEq.operators.secant`, or ``None``
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
