"""IMEX integrator on :class:`MCEq.core.MCEqRun`: which integrator a run takes, and the IMEX solve."""

from contextlib import nullcontext

import numpy as np

import MCEq.solvers as solvers
from MCEq import config
from MCEq.config import detect
from MCEq.operators import secant_table
from MCEq.operators.legendre import backward_removal, dense_count
from MCEq.solvers.path import em_cascade_dx_cap


class ImexRun:
    """Mixin of :class:`MCEq.core.MCEqRun` for :mod:`MCEq.solvers.imex`."""

    def _integrator(self, shared=True):
        """The integrator of this run, ``"etd2"`` or ``"imex"``.

        ``config.integrator = "auto"`` takes IMEX on 2D runs whose ETD2
        stiffness cap is below ``config.imex_auto_dx`` g/cm^2 (EM showers), and
        ETD2 otherwise; IMEX needs a ``shared`` path and no 1D secant table.
        """
        name = config.integrator.lower()
        if name not in ("auto", "etd2", "imex"):
            raise ValueError(
                f"config.integrator must be 'auto', 'etd2' or 'imex', got {name!r}"
            )
        supported = shared and secant_table.table_for(self) is None
        if name == "imex" and not supported:
            raise NotImplementedError(
                "The IMEX integrator needs one shared path and no 1D secant "
                "table (config.secant_1d_table = None)."
            )
        if name != "auto":
            return name
        if not (supported and self._mceq_db.is_2d):
            return "etd2"
        return "imex" if self._stiffness_dx_cap() < config.imex_auto_dx else "etd2"

    def _stiffness_dx_cap(self):
        """ETD2 step cap of the continuous-loss and EM stiffness in g/cm^2."""
        return min(
            em_cascade_dx_cap(self._em_cascade_step_scale()),
            self._continuous_loss_dx_cap(),
        )

    @staticmethod
    def _imex_path_defaults(eps, dX_max):
        """``eps`` and ``dX_max`` with ``config.imex_path`` for None."""
        p = config.imex_path
        return (
            p["eps"] if eps is None else eps,
            p["dX_max"] if dX_max is None else dX_max,
        )

    def _solve_imex(self, nsteps, dX, rho_inv, grid_idcs, sec_ops):
        """:meth:`MCEq.core.MCEqRun.solve_from_integration_path` with IMEX."""
        self._solution, self.grid_sol = self._run_imex(
            nsteps, dX, rho_inv, self._stack_modes(self._phi0), grid_idcs, sec_ops
        )

    def _run_imex(
        self, nsteps, dX, rho_inv, phi0, grid_idcs, sec_ops=None, tangent=None
    ):
        """Integrate a ``(dim,)`` or ``(dim, K)`` state with
        :mod:`MCEq.solvers.imex`, on the GPU when ``kernel_config`` is CUDA;
        ``tangent`` (:class:`~MCEq.solvers.TangentOperator`) adds its sources
        to the derivative columns. The stepper is cached per identity of
        ``int_m``, ``dec_m`` and ``sec_ops``."""
        kernel = config.kernel_config.lower()
        on_cuda = detect.ALIASES.get(kernel, kernel) == "cuda_etd2"
        key = (self.int_m, self.dec_m, sec_ops, on_cuda)
        cached = getattr(self, "_imex_cache", None)
        device = nullcontext()
        if on_cuda:
            import cupy

            device = cupy.cuda.Device(config.cuda_gpu_id)
        with device:
            if cached is None or any(a is not b for a, b in zip(cached[0], key)):
                self._imex_cache = cached = (key, self._imex_stepper(sec_ops, on_cuda))
            stepper = cached[1]
            stepper.tangent = (
                None if tangent is None else solvers.TangentSources(tangent, stepper.op)
            )
            return solvers.imex_driver(nsteps, dX, rho_inv, stepper, phi0, grid_idcs)

    def _imex_stepper(self, sec_ops, on_cuda):
        """:class:`MCEq.solvers.imex.ImexStepper` of the current matrices."""
        N = self.dim_states
        light, e_tot = np.zeros(N, dtype=bool), np.zeros(N)
        for p in self.pman.all_particles:
            if p.mceqidx >= 0:
                light[p.lidx : p.uidx] = p.is_em or p.is_lepton
                e_tot[p.lidx : p.uidx] = np.asarray(self.e_grid) + p.mass
        db = self._mceq_db
        n_k = db.n_k if db.is_2d else 1
        mix = None
        if db.is_2d and config.legendre_backward_removal:
            mix = backward_removal(dense_count(db.k_grid) - 1)
        coupling = None
        if sec_ops is not None:
            coupling = (sec_ops["T_P"], sec_ops["low_e_idx"])
        if on_cuda:
            import cupy as xp
        else:
            xp = np
        builder = self.matrix_builder
        op = solvers.ImexOperator(
            builder.int_m_hadr,
            self.dec_m,
            builder.dEdx_band,
            light,
            species=np.arange(N) // len(self.e_grid),
            n_k=n_k,
            mix=mix,
            coupling=coupling,
            xp=xp,
        )
        return solvers.ImexStepper(op, e_tot)
