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
the yields a :class:`~MCEq.particlemanager.ParticleManager` carries, folds
resonances into their parents, adds the continuous-loss band and the
:math:`\kappa^2` muon multiple-scattering damping, and assembles the result
into the two constant sparse matrices ``int_m`` and ``dec_m`` that
:mod:`MCEq.operators.compiled` turns into a compiled operator for the solvers.

``int_m`` is also available split in two, as
:attr:`MatrixBuilder.int_m_hadr` + :attr:`MatrixBuilder.dEdx_band`, which the
block-ETD research item (plan §7 R3) needs. Both halves are assembled lazily
from what the accumulation stashed as it ran -- ``int_m`` is never recomputed
from them, so it stays bitwise what it was, and the hot path pays one dense
block copy per species carrying losses.

Settings reach the builder as the ``grid``, ``losses`` and ``physics`` group
views of :mod:`MCEq.config.groups`, not as reads off the config module, so this
layer stays below the driver (contract C5 in ``.importlinter``). ``grid`` is
the third view because ``config.floatlen`` -- the dtype every channel block and
every assembled CSR is built in -- is mapped into the grid group as
``grid.dtype`` rather than into ``losses`` or ``physics``. The plan's decision
D27 names only the latter two, which does not cover those reads; its constraint
is that no new spec dataclass appears, and three existing views satisfy it.
"""

from itertools import product

import numpy as np
import scipy.sparse as sp

from MCEq.misc import info
from MCEq.operators import loss_stencil, scattering


class MatrixBuilder:
    """This class constructs the interaction and decay matrices.

    ``pman`` is the particle manager holding the cascade species, their
    tabulated yields and the energy grid. ``layout`` supplies the mode layout
    of the database (``is_2d``, ``n_k``, ``k_grid``), which decides whether a
    per-channel block is ``(dim, dim)`` or one slab per Hankel mode.

    ``grid``, ``losses`` and ``physics`` are the settings groups this builder
    reads (see the module docstring); a group left as ``None`` falls back to
    the live view MCEq.config publishes, which reads the same flat names as
    before.
    """

    def __init__(self, pman, layout, grid=None, losses=None, physics=None):
        from MCEq import config

        # A group left as None resolves to the live view on MCEq.config, so an
        # un-injected builder reads exactly what it read before, per read.
        self._grid = config.grid if grid is None else grid
        self._losses = config.losses if losses is None else losses
        self._physics = config.physics if physics is None else physics
        self._pman = pman
        self._layout = layout
        self.is_2d = layout.is_2d
        self.n_k = layout.n_k
        self.k_grid = layout.k_grid
        self._energy_grid = self._pman._energy_grid
        self.int_m = None
        self.dec_m = None
        self._reset_band_split()
        self._construct_differential_operator()

    def construct_matrices(self, skip_decay_matrix=False):
        r"""Constructs the matrices for calculation.

        These are:

        - :math:`\boldsymbol{M}_{int} = (-\boldsymbol{1} +
            \boldsymbol{C}){\boldsymbol{\Lambda}}_{int}`,
        - :math:`\boldsymbol{M}_{dec} = (-\boldsymbol{1} +
            \boldsymbol{D}){\boldsymbol{\Lambda}}_{dec}`.

        For debug_levels >= 2 some general information about matrix
        shape and the number of non-zero elements is printed. The
        intermediate matrices :math:`\boldsymbol{C}` and
        :math:`\boldsymbol{D}` are deleted afterwards to save memory.

        Set the ``skip_decay_matrix`` flag to avoid recreating the decay
        matrix. This is not necessary if, for example, particle production
        is modified, or the interaction model is changed.

        Also invalidates and re-stashes the :attr:`int_m_hadr` /
        :attr:`dEdx_band` split, which is read off this accumulation.

        Args:
          skip_decay_matrix (bool): Omit re-creating D matrix

        """

        info(
            3,
            f"Start filling matrices. Skip_decay_matrix = {skip_decay_matrix}",
        )

        self._reset_band_split()
        self._fill_matrices(skip_decay_matrix=skip_decay_matrix)

        cparts = self._pman.cascade_particles

        # interaction part
        # -I + C
        # In first interaction mode it is just C
        self.max_lint = 0.0

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
                # Multiply with Lambda_int and keep track the maximal
                # interaction length for the calculation of integration steps
                self.max_lint = np.max(
                    [self.max_lint, np.max(parent.inverse_interaction_length())]
                )
                self.C_blocks[idx] *= np.asarray(
                    parent.inverse_interaction_length(), dtype=self._grid.dtype
                )

            if child.mceqidx == parent.mceqidx and parent.has_contloss:
                pid = abs(parent.pdg_id[0])
                if self._physics.enable_energy_loss:
                    if (
                        pid == 13
                        or (self._physics.enable_em_ion and pid == 11)
                        or (self._physics.generic_losses_all_charged and pid != 11)
                    ):
                        info(5, "Cont. loss for", parent.name)
                        if self._physics.enable_cont_rad_loss:
                            # Stash the band and the block it lands on, so the
                            # int_m_hadr / dEdx_band split needs neither a
                            # second cont_loss_operator call nor a change here.
                            band = self.cont_loss_operator(parent.pdg_id)
                            self._contloss_bands[idx] = band
                            self._preband_blocks[idx] = self.C_blocks[idx].copy()
                            if self.is_2d:
                                self.C_blocks[idx] += band[None, :, :]
                            else:
                                self.C_blocks[idx] += band

        self.int_m = self._csr_from_blocks(self.C_blocks, apply_muon_scattering=True)
        self._assembly_key = self._current_assembly_key()
        # -I + D

        if not skip_decay_matrix or self.dec_m is None:
            self.max_ldec = 0.0
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
                # Multiply with Lambda_dec and keep track of the
                # maximal decay length for the calculation of integration steps
                self.max_ldec = max(
                    [self.max_ldec, np.max(parent.inverse_decay_length())]
                )

                self.D_blocks[idx] *= np.asarray(
                    parent.inverse_decay_length(), dtype=self._grid.dtype
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
        1/max_step explicit euler steps"""

        n_steps = int(1.0 / self._losses.step_for_average)
        info(
            10,
            f"Averaging continuous loss using {n_steps} intermediate steps.",
        )

        op_step = np.eye(self._energy_grid.d) + op_mat * self._losses.step_for_average
        return np.linalg.matrix_power(op_step, n_steps) - np.eye(self._energy_grid.d)

    def cont_loss_operator(self, pdg_id):
        """Returns continuous loss operator that can be summed with appropriate
        position in the C matrix."""
        op_mat = -np.diag(1 / self._energy_grid.c).dot(
            self.op_matrix.dot(np.diag(self._pman[pdg_id].dEdX))
        )

        if self._losses.average_operator:
            return self._average_operator(op_mat)
        return op_mat

    # ----------------------------------------------------------------------
    # the int_m_hadr + dEdx_band split (D27)
    # ----------------------------------------------------------------------

    def _reset_band_split(self):
        """Drop the band stashes and both assembled halves of ``int_m``."""
        self._contloss_bands, self._preband_blocks = {}, {}
        self._int_m_hadr = self._dEdx_band = self._assembly_key = None

    def _current_assembly_key(self):
        """The two settings an assembly reads that the blocks do not carry: the
        CSR dtype (``np.dtype`` normalises the ``floatlen = None`` spelling of
        fp64) and whether ``_csr_from_blocks`` finds muon damping to apply."""
        damping = self.is_2d and getattr(
            self._physics, "muon_multiple_scattering", False
        )
        return np.dtype(self._grid.dtype), bool(damping)

    def _require_live_assembly(self, name):
        """Raise unless ``name`` would assemble into a part of the live ``int_m``."""
        if self.int_m is None:
            raise RuntimeError(
                f"{name} needs construct_matrices() to have run: it is read off "
                f"the accumulation that produces int_m."
            )
        current = self._current_assembly_key()
        if current != self._assembly_key:
            raise RuntimeError(
                f"{name} would not be a part of the int_m in hand: (grid.dtype, "
                f"muon damping) is now {current} and int_m was assembled at "
                f"{self._assembly_key}. Call construct_matrices() first."
            )

    @property
    def int_m_hadr(self):
        r"""``int_m`` without the continuous-loss band, assembled on demand.

        Hadronic production, the :math:`-\boldsymbol{1}` main diagonal, the
        :math:`\boldsymbol{\Lambda}_{int}` scaling and the
        :math:`-\kappa^2\theta_s^2(E)/4` muon damping, which belongs to neither
        name: ``int_m == int_m_hadr + dEdx_band``, the sum taken in the band's
        fp64 and rounded once to ``grid.dtype``.

        Bitwise at fp64, the default. Below it the identity holds except where
        the band and the damping meet -- nowhere in 1D, the muon diagonals in
        2D. There the assembly adds the band into the dense block and the
        damping over it as a CSR duplicate, so ``int_m`` is
        ``fl(fl(hadronic + band) + damping)`` against this half's
        ``fl(hadronic + damping)``; float addition is not associative, and a
        few ulp of those entries is the difference (measured: 1542 of 8742, 1
        to 7 ulp, on the 2D rc7 fixture at float32).

        ``int_m`` is not reassembled from the halves -- it is this same
        accumulation read one block earlier. Cached until the next
        ``construct_matrices``.
        """
        self._require_live_assembly("int_m_hadr")
        if self._int_m_hadr is None:
            blocks = dict(self.C_blocks)
            blocks.update(self._preband_blocks)
            self._int_m_hadr = self._csr_from_blocks(blocks, apply_muon_scattering=True)
        return self._int_m_hadr

    @property
    def dEdx_band(self):
        """The continuous-loss band of ``int_m`` alone, assembled on demand.

        The complement of :attr:`int_m_hadr`, in fp64, from the band arrays
        ``cont_loss_operator`` returned during the assembly -- reused, not
        recomputed, so ``losses.average_operator`` never re-enters its
        ``np.linalg.matrix_power`` and a stencil changed since cannot leak in.
        Empty when the continuous loss is off. Cached like the other half.
        """
        self._require_live_assembly("dEdx_band")
        if self._dEdx_band is None:
            self._dEdx_band = self._csr_from_bands()
        return self._dEdx_band

    def _csr_from_bands(self):
        """Place the stashed loss bands at their species offsets, per mode.

        fp64, the dtype ``cont_loss_operator`` produces: a band cast to
        ``grid.dtype`` first would round twice, where ``block += band`` rounds
        the promoted sum once. The band is mode-independent -- added to every
        Hankel slab of a species through ``band[None, :, :]`` -- so the 2D
        operator is ``n_k`` copies of the one-mode matrix. No ``(row, column)``
        is written twice, the species blocks being disjoint, so the COO
        conversion has no duplicates to sum in an unspecified order.
        """
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
        ``n_k`` axis (one slab per Hankel mode), so the returned shape is
        ``(n_k, dim, dim)`` instead of ``(dim, dim)``.
        """
        if self.is_2d:
            return np.zeros(
                (self.n_k, self._pman.dim, self._pman.dim),
                dtype=self._grid.dtype,
            )
        return np.zeros((self._pman.dim, self._pman.dim), dtype=self._grid.dtype)

    def _muon_scattering_damping(self):
        """Per-energy Gaussian multiple-scattering damping data for muons.

        Returns ``(muon_lidcs, theta_s_sq)`` — the state-vector offsets of
        all muon species present (PDG ±13, helicities 0, ±1) and the
        squared scattering angle per unit depth — or ``None`` when muon
        multiple scattering does not apply. Physics and formulas:
        :mod:`MCEq.operators.scattering`. The per-mode diagonal
        contribution is ``-kappa^2 * theta_s^2(E) / 4``; it sits on the
        diagonal D so ETD2RK's ``e^{h*D}`` integrates it exactly, without
        a per-step operator split.
        """
        if not (
            self.is_2d and getattr(self._physics, "muon_multiple_scattering", False)
        ):
            return None
        # Prefer pman's (13, 0) mass; fall back to the PDG value.
        try:
            mu_mass = float(self._pman[(13, 0)].mass)
        except (KeyError, AttributeError):
            mu_mass = scattering.MUON_MASS
        theta_s_sq = scattering.theta_s_squared(self._energy_grid.c, mu_mass)
        muon_lidcs = scattering.muon_state_offsets(self._pman.pdg2pref)
        if not muon_lidcs:
            return None
        return muon_lidcs, theta_s_sq

    def _csr_from_blocks(self, blocks, apply_muon_scattering=False):
        """Assemble the per-channel blocks into the global CSR operator.

        For 1D databases each block is a dense ``(dim, dim)`` channel
        matrix placed at its (child, parent) offsets in a single
        ``(dim_states, dim_states)`` sparse matrix.

        For 2D databases each block carries a leading ``n_k`` axis (one
        slab per Hankel mode, shape ``(n_k, dim, dim)``). The Hankel modes
        are mutually decoupled, so the operator is block-diagonal in
        kappa: per mode, the nonzero entries of every channel slab are
        scattered (with their global row/column offsets) into one COO
        triplet set and converted to CSR in one shot — no dense
        ``(dim_states, dim_states)`` intermediate. ``scipy.sparse.
        block_diag`` then stitches the ``n_k`` mode matrices into the
        final ``(n_k * dim_states, n_k * dim_states)`` CSR that the
        dimension-agnostic ETD2RK kernels consume.

        When ``apply_muon_scattering`` is True (the interaction-matrix
        path) and ``physics.muon_multiple_scattering`` is on, muon-row
        diagonals receive the per-mode Gaussian multiple-scattering
        damping ``-kappa^2 * theta_s^2(E) / 4`` (see
        :meth:`_muon_scattering_damping`), added as extra COO entries
        (duplicates are summed on CSR conversion).

        The sec(theta) transport setting (``secant.transport``) does not
        alter these matrices; the mode coupling is applied inside the
        ETD2RK secant kernels (see :mod:`MCEq.operators.secant` for why it
        is not stitched into the CSR).
        """
        from scipy.sparse import coo_matrix, csr_matrix

        if self.is_2d:
            mu_damp = None
            if apply_muon_scattering:
                mu_damp = self._muon_scattering_damping()
            n_e = self.dim
            shape = (self.dim_states, self.dim_states)
            per_mode_csr = []
            for k in range(self.n_k):
                rows, cols, vals = [], [], []
                for (c, p), d in iter(blocks.items()):
                    rc, rp = self._pman.mceqidx2pref[c], self._pman.mceqidx2pref[p]
                    slab = d[k]
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
                kappa = self.k_grid[k]
                if mu_damp is not None and kappa != 0:
                    muon_lidcs, theta_s_sq = mu_damp
                    damping = scattering.mode_damping(theta_s_sq, kappa)
                    for lidx in muon_lidcs:
                        diag = np.arange(lidx, lidx + n_e)
                        rows.append(diag)
                        cols.append(diag)
                        vals.append(damping)
                if rows:
                    m = coo_matrix(
                        (
                            np.concatenate(vals).astype(self._grid.dtype),
                            (np.concatenate(rows), np.concatenate(cols)),
                        ),
                        shape=shape,
                    ).tocsr()
                else:
                    m = csr_matrix(shape, dtype=self._grid.dtype)
                m.eliminate_zeros()
                m.sort_indices()
                per_mode_csr.append(m)
            stitched = sp.block_diag(per_mode_csr, format="csr")
            stitched.eliminate_zeros()
            stitched.sort_indices()
            return stitched

        new_mat = np.zeros((self.dim_states, self.dim_states), dtype=self._grid.dtype)
        for (c, p), d in iter(blocks.items()):
            rc, rp = self._pman.mceqidx2pref[c], self._pman.mceqidx2pref[p]
            try:
                new_mat[rc.lidx : rc.uidx, rp.lidx : rp.uidx] = d
            except ValueError:
                _d = self.dim_states
                _n = rp.name
                _l = rp.lidx
                _u = rp.uidx
                _nc = rc.name
                _lc = rc.lidx
                _uc = rc.uidx
                raise Exception(
                    "Dimension mismatch: matrix "
                    + f"{_d}x{_d}, p={_n}:({_l},{_u}), c={_nc}:({_lc},{_uc})"
                )
        return csr_matrix(new_mat)

    def _follow_chains(self, p, pprod_mat, p_orig, propmat, reclev=0):
        """Recursively project ``p_orig``'s production through resonance
        children of ``p`` into ``propmat``.

        For each child ``d`` of ``p``:

        * If ``d`` is *not* a resonance, ``d`` has its own state-vector slot,
          so we add a direct contribution ``propmat[d, p_orig] += d's
          production matrix · pprod_mat`` and stop.
        * If ``d`` *is* a resonance (set via ``adv_set["force_resonance"]``),
          ``d`` has no slot of its own, so we fold its production into
          ``p_orig``'s row by multiplying through and recursing into ``d``'s
          own children.
        """
        info(40, reclev * "\t", "entering with", p.name)
        for d in p.children:
            info(40, reclev * "\t", "following to", d.name)
            if not d.is_resonance:
                dprop = self._zero_mat()
                p._assign_decay_dist(d, dprop)
                propmat[(d.mceqidx, p_orig.mceqidx)] += dprop.dot(pprod_mat)
                info(20, reclev * "\t", "\t terminating at", d.name)
            else:
                dres = self._zero_mat()
                p._assign_decay_dist(d, dres)
                self._follow_chains(d, dres.dot(pprod_mat), p_orig, propmat, reclev + 1)

    def _fill_matrices(self, skip_decay_matrix=False):
        """Generates the interaction and decay matrices from scratch."""
        from collections import defaultdict

        # Fill decay matrix blocks
        if not skip_decay_matrix or self.dec_m is None:
            # Initialize empty D matrix
            self.D_blocks = defaultdict(lambda: self._zero_mat())
            for p in self._pman.cascade_particles:
                # Fill parts of the D matrix related to p as mother
                if not p.is_stable and bool(p.children) and not p.is_tracking:
                    self._follow_chains(
                        p,
                        np.diag(np.ones(self.dim)).astype(self._grid.dtype),
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
                p._assign_hadr_dist(s, cmat)
                if not s.is_resonance:
                    # s has its own state-vector slot — direct entry.
                    self.C_blocks[(s.mceqidx, p.mceqidx)] += cmat
                else:
                    # s is folded — recurse into its children.
                    self._follow_chains(s, cmat, p, self.C_blocks, reclev=1)

    def _construct_differential_operator(self):
        """Constructs a derivative operator for the continuous losses.

        Builds a (dim_e x dim_e) banded matrix that approximates d/du with
        u = ln E on the (log-uniform) energy grid. Families, exactness
        conditions and the boundary-row caveat:
        :mod:`MCEq.operators.loss_stencil`. The interior 7-point stencil is
        selected by ``losses.stencil_method``, anchored for the ``expfit*``
        families at ``losses.stencil_alpha0``, and the composites replace
        ``losses.stencil_low_upwind_rows`` low-energy rows.

        Called from :meth:`__init__` and nowhere else, so
        :meth:`MCEq.core.MCEqRun.regenerate_matrices` rebuilds the blocks
        against the ``op_matrix`` the constructor left behind; a stencil change
        needs this method called explicitly.
        """
        self.op_matrix = loss_stencil.differential_operator(
            self._energy_grid.b,
            int(self._energy_grid.d),
            method=getattr(self._losses, "stencil_method", "expfit_low_upwind2"),
            alpha0=getattr(self._losses, "stencil_alpha0", 3.0),
            low_upwind_rows=getattr(self._losses, "stencil_low_upwind_rows", 8),
            dtype=self._grid.dtype,
        )
