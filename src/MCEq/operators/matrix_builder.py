r"""Assembly of the cascade operator from the tabulated species data.

The transport equation is

.. math::

    \frac{d\Phi}{dX} = \left(\boldsymbol{M}_{int}
        + \frac{1}{\rho(X)}\boldsymbol{M}_{dec}\right)\Phi,

.. math::

    \boldsymbol{M}_{int} = (-\boldsymbol{1} + \boldsymbol{C})
        \boldsymbol{\Lambda}_{int}, \quad
    \boldsymbol{M}_{dec} = (-\boldsymbol{1} + \boldsymbol{D})
        \boldsymbol{\Lambda}_{dec},

with :math:`\boldsymbol{C}` the secondary-production matrix,
:math:`\boldsymbol{D}` the decay matrix and :math:`\boldsymbol{\Lambda}` the
diagonal of inverse interaction / decay lengths.

:class:`MatrixBuilder` fills ``C`` and ``D`` per (child, parent) channel from
the yields of a :class:`~MCEq.particlemanager.ParticleManager`, folds
resonances into their parents, adds the continuous-loss band and the per-mode
muon multiple-scattering rate, and assembles the sparse matrices ``int_m`` and
``dec_m``. ``int_m`` is also kept as its two parts, :attr:`MatrixBuilder.int_m_hadr`
+ :attr:`MatrixBuilder.dEdx_band`.
"""

from collections import defaultdict
from itertools import product

import numpy as np
import scipy.sparse as sp

from MCEq import bin_average, config
from MCEq.misc import info
from MCEq.operators import loss_stencil, scattering

#: Stiffness threshold of the species-adaptive upwind layer, see
#: :meth:`MatrixBuilder._upwind_rows_for`. The row count covers every row whose
#: explicit one-step stiffness ``dX_max * |dEdX| / (E * dlnE)`` reaches this
#: fraction; the expfit interior row is contractive below it and mixed-sign
#: (unstable in isolation) above.
UPWIND_STIFFNESS_THRESHOLD = 0.5

#: Minimum number of monotone upwind rows of the adaptive layer.
UPWIND_ROWS_FLOOR = 8


class MatrixBuilder:
    """Builds ``int_m`` and ``dec_m`` from a ParticleManager's yields.

    ``pman`` holds the cascade species, their yields and the energy grid.
    ``layout`` supplies ``is_2d``, ``n_k`` and ``k_grid``: a per-channel block
    is ``(dim, dim)`` or one slab per Legendre mode.
    """

    def __init__(self, pman, layout):
        self._pman = pman
        self._layout = layout
        self.is_2d = layout.is_2d
        self.n_k = layout.n_k
        self.k_grid = layout.k_grid
        self._energy_grid = self._pman._energy_grid
        self.int_m = None
        self.dec_m = None
        #: ``int_m`` without the continuous-loss band
        self.int_m_hadr = None
        #: the continuous-loss band of ``int_m`` (fp64)
        self.dEdx_band = None
        self._contloss_bands, self._preband_blocks = {}, {}
        self._contloss_upwind_rows = {}
        self._construct_differential_operator()

    def construct_matrices(self, skip_decay_matrix=False):
        r"""Constructs the matrices for calculation.

        These are:

        - :math:`\boldsymbol{M}_{int} = (-\boldsymbol{1} +
            \boldsymbol{C}){\boldsymbol{\Lambda}}_{int}`,
        - :math:`\boldsymbol{M}_{dec} = (-\boldsymbol{1} +
            \boldsymbol{D}){\boldsymbol{\Lambda}}_{dec}`.

        Matrix-block diagnostics are logged at level 5. ``C_blocks`` and
        ``D_blocks`` remain stored after assembly.

        Set the ``skip_decay_matrix`` flag to avoid recreating the decay
        matrix. This is not necessary if, for example, particle production
        is modified, or the interaction model is changed.

        Args:
          skip_decay_matrix (bool): Omit re-creating D matrix

        """

        info(
            3,
            f"Start filling matrices. Skip_decay_matrix = {skip_decay_matrix}",
        )

        self._contloss_bands, self._preband_blocks = {}, {}
        self._contloss_upwind_rows = {}
        self._fill_matrices(skip_decay_matrix=skip_decay_matrix)

        cparts = self._pman.cascade_particles

        # interaction part
        # -I + C
        int_rates = {}

        for parent, child in product(cparts, cparts):
            idx = (child.mceqidx, parent.mceqidx)
            # Main diagonal
            if child.mceqidx == parent.mceqidx and parent.can_interact:
                # Subtract unity from the main diagonals
                info(10, "subtracting main C diagonal from", child.name, parent.name)
                if self.is_2d:
                    self.C_blocks[idx][
                        :, np.diag_indices(self.dim)[0], np.diag_indices(self.dim)[1]
                    ] -= 1.0
                else:
                    self.C_blocks[idx][np.diag_indices(self.dim)] -= 1.0

            if idx in self.C_blocks:
                self.C_blocks[idx] = bin_average.apply_rate(
                    self.C_blocks[idx], parent, "int", int_rates, config.floatlen
                )

            if child.mceqidx == parent.mceqidx and parent.has_contloss:
                pid = abs(parent.pdg_id[0])
                if config.enable_energy_loss:
                    if (
                        pid == 13
                        or (config.enable_em_ion and pid == 11)
                        or (config.generic_losses_all_charged and pid != 11)
                    ):
                        info(5, "Cont. loss for", parent.name)
                        if config.enable_cont_rad_loss:
                            # Kept for the int_m_hadr / dEdx_band split.
                            band = self.cont_loss_operator(parent.pdg_id)
                            self._contloss_bands[idx] = band
                            self._contloss_upwind_rows[idx] = self._closure_rows_for(
                                parent.pdg_id
                            )
                            self._preband_blocks[idx] = self.C_blocks[idx].copy()
                            if self.is_2d:
                                self.C_blocks[idx] += band[None, :, :]
                            else:
                                self.C_blocks[idx] += band

        self.int_m = self._csr_from_blocks(self.C_blocks, apply_muon_scattering=True)
        self.int_m_hadr = self._csr_from_blocks(
            {**self.C_blocks, **self._preband_blocks}, apply_muon_scattering=True
        )
        self.dEdx_band = self._csr_from_bands()
        # -I + D

        if not skip_decay_matrix or self.dec_m is None:
            dec_rates = {}
            for parent, child in product(cparts, cparts):
                idx = (child.mceqidx, parent.mceqidx)
                # Main diagonal
                if child.mceqidx == parent.mceqidx and not parent.is_stable:
                    # Subtract unity from the main diagonals
                    info(
                        10, "subtracting main D diagonal from", child.name, parent.name
                    )
                    if self.is_2d:
                        self.D_blocks[idx][
                            :,
                            np.diag_indices(self.dim)[0],
                            np.diag_indices(self.dim)[1],
                        ] -= 1.0
                    else:
                        self.D_blocks[idx][np.diag_indices(self.dim)] -= 1.0
                if idx not in self.D_blocks:
                    info(25, parent.pdg_id[0], child.pdg_id, "not in D_blocks")
                    continue
                self.D_blocks[idx] = bin_average.apply_rate(
                    self.D_blocks[idx], parent, "dec", dec_rates, config.floatlen
                )

            self.dec_m = self._csr_from_blocks(self.D_blocks)

        for mname, mat in [("C", self.int_m), ("D", self.dec_m)]:
            mat_density = float(mat.nnz) / float(np.prod(mat.shape))
            info(5, f"{mname} Matrix info:")
            info(5, f"    density    : {mat_density:3.2%}")
            info(5, "    shape      : {0} x {1}".format(*mat.shape))
            info(5, f"    nnz        : {mat.nnz}")
            info(10, "    sum        :", mat.sum())

        info(3, "Done filling matrices.")

        return self.int_m, self.dec_m

    def _average_operator(self, op_mat):
        """Averages the continuous loss operator by performing
        ``int(1 / config.loss_step_for_average)`` repeated finite updates of the
        configured size."""

        n_steps = int(1.0 / config.loss_step_for_average)
        info(
            10,
            f"Averaging continuous loss using {n_steps} intermediate steps.",
        )

        op_step = np.eye(self._energy_grid.d) + op_mat * config.loss_step_for_average
        return np.linalg.matrix_power(op_step, n_steps) - np.eye(self._energy_grid.d)

    def cont_loss_operator(self, pdg_id):
        """Returns continuous loss operator that can be summed with appropriate
        position in the C matrix.

        The derivative operator is :meth:`_differential_operator_for`.
        """
        op = self._differential_operator_for(pdg_id)
        op_mat = -np.diag(1 / self._energy_grid.c).dot(
            op.dot(np.diag(self._pman[pdg_id].dEdX))
        )

        if config.average_loss_operator:
            return self._average_operator(op_mat)
        return op_mat

    def _upwind_rows_for(self, pdg_id):
        """Number of monotone upwind rows for one particle's loss stencil.

        Covers every row whose one-step stiffness
        ``dX * |dEdX| / (E * dlnE)`` at ``dX = 20 g/cm^2`` reaches
        :data:`UPWIND_STIFFNESS_THRESHOLD`, clamped to
        ``[UPWIND_ROWS_FLOOR, dim - 2]``. EM species return 0.
        """
        particle = self._pman[pdg_id]
        if getattr(particle, "is_em", False):
            return 0
        d_x_max = 20.0  # fixed stencil-design depth
        dedx = np.abs(np.asarray(particle.dEdX, dtype=np.float64))
        dln_e = float(
            np.mean(loss_stencil.log_spacings(np.asarray(self._energy_grid.b, float)))
        )
        stiffness = d_x_max * dedx / (np.asarray(self._energy_grid.c, float) * dln_e)
        hits = np.flatnonzero(stiffness >= UPWIND_STIFFNESS_THRESHOLD)
        n_rows = 0 if hits.size == 0 else int(hits.max()) + 1
        return min(max(n_rows, UPWIND_ROWS_FLOOR), int(self._energy_grid.d) - 2)

    def _closure_rows_for(self, pdg_id):
        """Leading upwind-closure rows of one particle's loss band (0 unless ``expfit_low_upwind*``)."""
        method = config.loss_stencil_method
        if not str(method).startswith("expfit_low_upwind"):
            return 0
        return self._upwind_rows_for(pdg_id)

    def _differential_operator_for(self, pdg_id):
        """The ``d/du`` operator for one particle: :attr:`op_matrix`, or a cached
        ``expfit_low_upwind*`` variant with that particle's upwind row count."""
        method = config.loss_stencil_method
        if not str(method).startswith("expfit_low_upwind"):
            return self.op_matrix
        n_rows = self._upwind_rows_for(pdg_id)
        if n_rows == 0:
            return self.op_matrix
        if n_rows not in self._stencils:
            self._stencils[n_rows] = loss_stencil.differential_operator(
                self._energy_grid.b,
                int(self._energy_grid.d),
                method=method,
                alpha0=config.loss_stencil_alpha0,
                low_upwind_rows=n_rows,
                dtype=config.floatlen,
            )
        return self._stencils[n_rows]

    def _csr_from_bands(self):
        """fp64 CSR of the stashed loss bands at their species offsets, repeated per mode."""
        shape = (self.dim_states, self.dim_states)
        parts = []
        for (c, p), band in self._contloss_bands.items():
            rc, rp = self._pman.mceqidx2pref[c], self._pman.mceqidx2pref[p]
            r, cc = np.nonzero(band)
            parts.append((band[r, cc], r + rc.lidx, cc + rp.lidx))
        if parts:
            data, rows, cols = (np.concatenate(a) for a in zip(*parts))
            one = sp.coo_matrix(
                (data.astype(np.float64, copy=False), (rows, cols)), shape=shape
            ).tocsr()
        else:
            one = sp.csr_matrix(shape, dtype=np.float64)
        stitched = one if self.n_k == 1 else sp.block_diag([one] * self.n_k, "csr")
        stitched.eliminate_zeros()
        stitched.sort_indices()
        return stitched

    @property
    def dim(self):
        """Energy grid (dimension)"""
        return int(self._pman.dim)

    @property
    def dim_states(self):
        """Number of cascade particles times dimension of grid
        (dimension of the equation system)"""
        return int(self._pman.dim_states)

    def _zero_mat(self):
        """Returns a new square zero valued matrix with dimensions of grid.

        For 2D databases the per-channel block carries an extra leading
        ``n_k`` axis (one slab per Legendre mode), so the returned shape is
        ``(n_k, dim, dim)`` instead of ``(dim, dim)``.
        """
        if self.is_2d:
            return np.zeros(
                (self.n_k, self._pman.dim, self._pman.dim),
                dtype=config.floatlen,
            )
        return np.zeros((self._pman.dim, self._pman.dim), dtype=config.floatlen)

    def _muon_scattering_damping(self):
        """Return muon offsets and negative rates, shape (n_k, n_energy).

        The rate of the selected model (:mod:`MCEq.operators.scattering`)
        enters the interaction diagonal, which ETD2 integrates exactly.
        """
        if not (self.is_2d and config.muon_multiple_scattering):
            return None
        # Prefer pman's (13, 0) mass; fall back to the PDG value.
        try:
            mu_mass = float(self._pman[(13, 0)].mass)
        except (KeyError, AttributeError):
            mu_mass = scattering.MUON_MASS
        muon_lidcs = scattering.muon_state_offsets(self._pman.pdg2pref)
        if not muon_lidcs:
            return None
        damping = -scattering.mode_rates(
            config.muon_scattering_model,
            self._energy_grid.c,
            self.k_grid,
            getattr(self, "_layout", None),
            mu_mass,
            config.muon_scattering_finite_nuclei,
        )
        return muon_lidcs, damping

    def _csr_from_blocks(self, blocks, apply_muon_scattering=False):
        """Assemble the per-channel blocks into the global CSR operator.

        2D blocks ``(n_k, dim, dim)`` give one CSR per mode, joined block-diagonally.
        ``apply_muon_scattering`` adds :meth:`_muon_scattering_damping` to the muon diagonals.
        """
        mu_damp = None
        if self.is_2d and apply_muon_scattering:
            mu_damp = self._muon_scattering_damping()
        n_e = self.dim
        shape = (self.dim_states, self.dim_states)
        per_mode_csr = []
        for k in range(self.n_k if self.is_2d else 1):
            rows, cols, vals = [], [], []
            for (c, p), d in iter(blocks.items()):
                rc, rp = self._pman.mceqidx2pref[c], self._pman.mceqidx2pref[p]
                slab = d[k] if self.is_2d else d
                if slab.shape != (rc.uidx - rc.lidx, rp.uidx - rp.lidx):
                    _d = self.dim_states
                    raise Exception(
                        "Dimension mismatch: matrix "
                        + f"{_d}x{_d}, p={rp.name}:({rp.lidx},{rp.uidx}),"
                        + f" c={rc.name}:({rc.lidx},{rc.uidx})"
                    )
                r, cc = np.nonzero(slab)
                rows.append(r + rc.lidx)
                cols.append(cc + rp.lidx)
                vals.append(slab[r, cc])
            if mu_damp is not None and self.k_grid[k] != 0:
                muon_lidcs, all_damping = mu_damp
                damping = all_damping[k]
                for lidx in muon_lidcs:
                    diag = np.arange(lidx, lidx + n_e)
                    rows.append(diag)
                    cols.append(diag)
                    vals.append(damping)
            if rows:
                m = sp.coo_matrix(
                    (
                        np.concatenate(vals).astype(config.floatlen),
                        (np.concatenate(rows), np.concatenate(cols)),
                    ),
                    shape=shape,
                ).tocsr()
            else:
                m = sp.csr_matrix(shape, dtype=config.floatlen)
            m.eliminate_zeros()
            m.sort_indices()
            per_mode_csr.append(m)
        if not self.is_2d:
            return per_mode_csr[0]
        stitched = sp.block_diag(per_mode_csr, format="csr")
        stitched.eliminate_zeros()
        stitched.sort_indices()
        return stitched

    def _follow_chains(self, p, pprod_mat, p_orig, propmat, reclev=0):
        """Add ``p_orig``'s production through ``p``'s children to ``propmat``, recursing into resonances."""
        info(40, reclev * "\t", "entering with", p.name)
        for d in p.children:
            info(40, reclev * "\t", "following to", d.name)
            if not d.is_resonance:
                dprop = self._zero_mat()
                dprop[:, :] = p.decay_dists[d][:, :]
                propmat[(d.mceqidx, p_orig.mceqidx)] += dprop.dot(pprod_mat)
                info(20, reclev * "\t", "\t terminating at", d.name)
            else:
                dres = self._zero_mat()
                dres[:, :] = p.decay_dists[d][:, :]
                self._follow_chains(d, dres.dot(pprod_mat), p_orig, propmat, reclev + 1)

    def _fill_matrices(self, skip_decay_matrix=False):
        """Generates the interaction and decay matrices from scratch."""
        # Fill decay matrix blocks
        if not skip_decay_matrix or self.dec_m is None:
            # Initialize empty D matrix
            self.D_blocks = defaultdict(lambda: self._zero_mat())
            for p in self._pman.cascade_particles:
                # Fill parts of the D matrix related to p as mother
                if not p.is_stable and bool(p.children) and not p.is_tracking:
                    self._follow_chains(
                        p,
                        np.diag(np.ones(self.dim)).astype(config.floatlen),
                        p,
                        self.D_blocks,
                        reclev=0,
                    )
                else:
                    info(20, p.name, "stable or not added to D matrix")

        # Initialize empty C blocks
        self.C_blocks = defaultdict(lambda: self._zero_mat())
        for p in self._pman.cascade_particles:
            # if p doesn't interact, skip interaction matrices
            if not p.is_projectile:
                if p.is_hadron:
                    info(1, f"No interactions by {p.name} ({p.pdg_id}).")
                continue
            for s in p.hadr_secondaries:
                cmat = self._zero_mat()
                cmat[:, :] = p.hadr_yields[s][:, :]
                if not s.is_resonance:
                    # s has its own state-vector slot.
                    self.C_blocks[(s.mceqidx, p.mceqidx)] += cmat
                else:
                    # s is folded: recurse into its children.
                    self._follow_chains(s, cmat, p, self.C_blocks, reclev=1)

    def _construct_differential_operator(self):
        """Build the shared ``d/du`` operator :attr:`op_matrix`
        (:mod:`MCEq.operators.loss_stencil`) and reset its per-species variants."""
        method = config.loss_stencil_method
        low_upwind_rows = config.loss_stencil_low_upwind_rows
        self.op_matrix = loss_stencil.differential_operator(
            self._energy_grid.b,
            int(self._energy_grid.d),
            method=method,
            alpha0=config.loss_stencil_alpha0,
            low_upwind_rows=low_upwind_rows,
            dtype=config.floatlen,
        )
        # Variants by upwind row count; op_matrix under its count after the
        # clamp of differential_operator.
        rows = min(max(3, int(low_upwind_rows)), int(self._energy_grid.d) - 2)
        self._stencils = {rows: self.op_matrix}
