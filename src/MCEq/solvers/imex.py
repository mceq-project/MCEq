"""IMEX cascade solver: the stiff part of the operator implicit, production between species explicit.

ARS(2,2,2) steps whose implicit stages are solved by GMRES, preconditioned with a level-scheduled block sweep of
the transfer graph; no factorization, steps limited by accuracy. The method is described in :ref:`solver-math`
(section "IMEX step").
"""

from contextlib import nullcontext
from graphlib import TopologicalSorter

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from MCEq import config
from MCEq.operators.legendre import backward_removal, dense_count

#: ARS(2,2,2) coefficients.
GAMMA = 1.0 - 1.0 / np.sqrt(2.0)
DELTA = 1.0 - 1.0 / (2.0 * GAMMA)


def _device(xp):
    """``(asarray, csr)`` constructors of the array module ``xp``."""
    if xp is np:
        return np.asarray, sp.csr_matrix
    import cupyx.scipy.sparse as csp

    return xp.asarray, csp.csr_matrix


def _to_host(a):
    """``a`` as a numpy array."""
    return a if isinstance(a, np.ndarray) else a.get()


def _offdiagonal(m):
    """``m`` without its diagonal, CSR, explicit zeros dropped."""
    m = sp.csr_matrix(m, copy=True)
    m.setdiag(0)
    m.eliminate_zeros()
    return m


def _restrict(m, keep):
    """CSR of the entries of ``m`` (COO) where ``keep``."""
    return sp.csr_matrix((m.data[keep], (m.row[keep], m.col[keep])), shape=m.shape)


class ImexOperator:
    """The cascade operator, split into its implicit and explicit parts.

    States are MCEq's mode-major arrays ``(n_k, N)``, ``N`` states per mode, or ``(n_k, N, K)`` for ``K`` columns.

    Args:
      int_m: interaction matrix without the continuous-loss band (``MatrixBuilder.int_m_hadr``).
      dec_m: decay matrix.
      band: continuous-loss band (``MatrixBuilder.dEdx_band``).
      light: ``(N,)`` bool, EM particles and leptons.
      species: ``(N,)`` species index of every state.
      n_k: number of angular modes (1 on 1D runs).
      mix: ``(n_D, n_D)`` backward-removal matrix of the leading modes, or None.
      coupling: ``(T_P, low)``: the sec(theta) coupling ``T_P`` ``(n_P, n_k)`` of the leading modes and the
        low-energy states it acts on, or None.
      xp: array module of the run (numpy or cupy).
    """

    def __init__(
        self, int_m, dec_m, band, light, species, n_k=1, mix=None, coupling=None, xp=np
    ):
        asarray, csr = _device(xp)
        self.xp, self.n_k = xp, n_k
        self.N = N = int_m.shape[0] // n_k
        d_int = int_m.diagonal() + band.diagonal()
        self.d_int = asarray(d_int.reshape(n_k, N))
        self.d_dec = asarray(dec_m.diagonal().reshape(n_k, N))

        for name, m in (("int_m", int_m), ("dec_m", dec_m), ("band", band)):
            c = sp.coo_matrix(m)
            if np.any(c.row // N != c.col // N):
                raise ValueError(f"ImexOperator: {name} couples the angular modes")
        int_off = _offdiagonal(int_m).tocoo()
        # implicit: transfers inside a species (continuous losses, redistribution) and inside the light sector
        r, c = int_off.row % N, int_off.col % N
        species = np.asarray(species)
        light = np.asarray(light, dtype=bool)
        stiff = (species[r] == species[c]) | (light[r] & light[c])
        self.int_imp = _restrict(int_off, stiff)
        self.int_exp = _restrict(int_off, ~stiff)
        self.dec_off = _offdiagonal(dec_m)
        self.band_off = _offdiagonal(band)
        self._dev = {
            name: csr(getattr(self, name))
            for name in ("int_imp", "int_exp", "dec_off", "band_off")
        }

        self.mix = None if mix is None else asarray(mix)
        self.T_P, self.low = None, None
        if coupling is not None:
            T_P, low = coupling
            self.T_P, self.low = asarray(T_P), np.asarray(low)
            self._low_dev = asarray(self.low)

    def coupled(self, x):
        """The operand ``S x`` of the transport, mode-major."""
        if self.T_P is None:
            return x
        xs = x.copy()
        low = self._low_dev
        xs[: self.T_P.shape[0], low] += self.xp.tensordot(self.T_P, x[:, low], axes=1)
        return xs

    def _mixed(self, y):
        if self.mix is not None:
            n_D = self.mix.shape[0]
            y[:n_D] = self.xp.tensordot(self.mix, y[:n_D], axes=1)
        return y

    def _apply(self, name, xs):
        flat = xs.reshape(self.n_k * self.N, -1)
        return (self._dev[name] @ flat).reshape(xs.shape)

    @staticmethod
    def _diagonal(d, xs):
        """``d`` ``(n_k, N)`` times every column of ``xs``."""
        return d.reshape(d.shape + (1,) * (xs.ndim - 2)) * xs

    def implicit(self, x, ri):
        """``F_imp(x)`` at inverse density ``ri``."""
        xs = self.coupled(x)
        produced = self._apply("int_imp", xs) + ri * self._apply("dec_off", xs)
        return (
            self._diagonal(self.d_int + ri * self.d_dec, xs)
            + self._mixed(produced)
            + self._apply("band_off", xs)
        )

    def explicit(self, x, ri, sources=None):
        """``F_exp(x)``: hadronic production (``ri`` unused), plus the :class:`TangentSources` ``sources``."""
        xs = self.coupled(x)
        produced = self._apply("int_exp", xs)
        if sources is None:
            return self._mixed(produced)
        off, diagonal = sources.terms(xs)
        return self._mixed(produced + off) + diagonal

    def state_parts(self, *names):
        """The parts ``names`` on one state-level pattern with per-mode values.

        Returns ``(rows, cols, values)``: the union of the parts' entries over all mode blocks, sorted by
        ``(row, col)``, and their values ``(len(names), nnz, n_k)``.
        """
        N = self.N
        coos = [getattr(self, name).tocoo() for name in names]
        keys = [(m.row % N) * N + (m.col % N) for m in coos]
        uniq = np.unique(np.concatenate(keys))
        values = np.zeros((len(names), len(uniq), self.n_k))
        for k, (m, key) in enumerate(zip(coos, keys)):
            values[k, np.searchsorted(uniq, key), m.row // N] = m.data
        return uniq // N, uniq % N, values


class TangentSources:
    """Explicit sources of tangent columns (:class:`MCEq.solvers.etd2.TangentOperator`).

    Each group of columns holds the cascade, its ``n`` first derivatives, with ``curvature`` the ``n`` second
    derivatives, and ``extra`` passive columns; groups (``tangent.groups``, default 1) follow each other. The
    derivative columns get ``G_k S phi`` (second derivatives ``2 G_k S psi_k``), the off-diagonal of ``G_k`` mixed
    like the production, its diagonal not.

    Args:
      tangent: the derivative matrices ``tangent._mats`` (MCEq order), ``curvature``, ``extra``, optional ``groups``.
      op: the :class:`ImexOperator` of the run.
    """

    def __init__(self, tangent, op):
        asarray, csr = _device(op.xp)
        mats = [sp.csr_matrix(m) for m in tangent._mats]
        self.n, self.curvature = len(mats), bool(tangent.curvature)
        self.groups = int(getattr(tangent, "groups", 1))
        self.width = 1 + self.n * (2 if self.curvature else 1) + int(tangent.extra)
        self.dim = op.n_k * op.N
        offs = [_offdiagonal(m) for m in mats]
        self.diagonal = asarray(
            np.stack([m.diagonal() for m in mats], axis=1)
        )  # (dim, n)
        self.stacked = csr(sp.vstack(offs, format="csr"))  # (n dim, dim)
        self.block = csr(sp.block_diag(offs, format="csr")) if self.curvature else None
        self.xp = op.xp

    def _per_parameter(self, flat):
        """``(n dim, groups)`` -> ``(dim, groups, n)``."""
        return flat.reshape(self.n, self.dim, self.groups).transpose(1, 2, 0)

    def terms(self, xs):
        """``(off, diagonal)`` source terms for the coupled operand ``xs`` ``(n_k, N, K)``."""
        xp, n = self.xp, self.n
        x = xs.reshape(self.dim, self.groups, self.width)
        off, diagonal = xp.zeros_like(x), xp.zeros_like(x)
        phi = x[:, :, 0]
        off[:, :, 1 : 1 + n] = self._per_parameter(self.stacked @ phi)
        diagonal[:, :, 1 : 1 + n] = self.diagonal[:, None, :] * phi[:, :, None]
        if self.curvature:
            psi = x[:, :, 1 : 1 + n]
            flat = psi.transpose(2, 0, 1).reshape(n * self.dim, self.groups)
            off[:, :, 1 + n : 1 + 2 * n] = 2.0 * self._per_parameter(self.block @ flat)
            diagonal[:, :, 1 + n : 1 + 2 * n] = 2.0 * self.diagonal[:, None, :] * psi
        return off.reshape(xs.shape), diagonal.reshape(xs.shape)


class LevelSweep:
    """Exact solve of the lower part of ``I - c J`` (``J`` the implicit part): the GMRES preconditioner.

    The transfer graph of the implicit part (column feeds row) is cut into strongly connected components, taken in
    topological order, with the states inside a component in descending total energy. The entries inside a
    component that go up in that order are left out; the rest is lower block-triangular. States are grouped into
    levels (longest path in the graph) that are solved together: each state is one block over the modes,
    diagonal outside the sec(theta) corner and dense ``n_P x n_P`` inside it.

    Args:
      op: an :class:`ImexOperator`.
      e_tot: ``(N,)`` total energy of every state.
    """

    def __init__(self, op, e_tot):
        self.op, self.xp = op, op.xp
        asarray, csr = _device(self.xp)
        N = op.N
        rows, cols, (va, vd) = op.state_parts("int_imp", "dec_off")
        b_rows, b_cols, (vb,) = op.state_parts("band_off")
        pattern = sp.csr_matrix(
            (
                np.ones(len(rows) + len(b_rows)),
                (np.r_[rows, b_rows], np.r_[cols, b_cols]),
            ),
            shape=(N, N),
        )
        self.perm, self.levels = self._order(pattern, e_tot)
        self.inv_perm = np.argsort(self.perm)
        lvl = np.empty(N, dtype=int)
        for k, (a, b) in enumerate(self.levels):
            lvl[a:b] = k

        def lower(rows, cols, *values):
            """Entries from earlier levels in level order: ``(indptr, cols, *values)``."""
            r, c = self.inv_perm[rows], self.inv_perm[cols]
            keep = lvl[c] < lvl[r]
            order = np.lexsort((c[keep], r[keep]))
            r, c = r[keep][order], c[keep][order]
            indptr = np.searchsorted(r, np.arange(N + 1))
            return (indptr, c, *(asarray(v[keep][order]) for v in values))

        # the mixed group (EM/lepton transfers + decays) and the band, which keeps the direction
        self._mixed = lower(rows, cols, va, vd)
        self._band = lower(b_rows, b_cols, vb)
        self.corner = None
        if op.T_P is not None:
            low = np.zeros(N, dtype=bool)
            low[op.low] = True
            in_corner = low[self.perm]
            self.corner = np.flatnonzero(in_corner)
            slot = np.full(N, -1)
            slot[self.corner] = np.arange(len(self.corner))
            self._corner_slices = []
            for a, b in self.levels:
                local = np.flatnonzero(in_corner[a:b])
                self._corner_slices.append(
                    (asarray(local), asarray(slot[a + local])) if len(local) else None
                )
        self._prepared = None
        self._slices = [self._level_slices(a, b, csr) for a, b in self.levels]

    @staticmethod
    def _order(pattern, e_tot):
        """``(perm, levels)``: states in level order and the ``(start, stop)`` of every level."""
        N = len(e_tot)
        c = sp.coo_matrix(pattern)
        n_scc, scc = connected_components(c, directed=True, connection="strong")
        between = scc[c.row] != scc[c.col]
        feeders = {k: set() for k in range(n_scc)}
        for src, dst in zip(scc[c.col[between]], scc[c.row[between]]):
            feeders[dst].add(src)
        topo = np.empty(n_scc, dtype=int)
        topo[list(TopologicalSorter(feeders).static_order())] = np.arange(n_scc)
        order = np.lexsort((np.arange(N), -np.asarray(e_tot), topo[scc]))
        pos = np.empty(N, dtype=int)
        pos[order] = np.arange(N)
        keep = pos[c.col] < pos[c.row]
        deps = sp.csr_matrix(
            (np.ones(keep.sum()), (c.row[keep], c.col[keep])), shape=(N, N)
        )
        lvl = np.zeros(N, dtype=int)
        for i in order:
            d = deps.indices[deps.indptr[i] : deps.indptr[i + 1]]
            lvl[i] = lvl[d].max() + 1 if len(d) else 0
        perm = np.lexsort((pos, lvl))
        bounds = np.searchsorted(lvl[perm], np.arange(lvl.max() + 2))
        return perm, list(zip(bounds[:-1].tolist(), bounds[1:].tolist()))

    def _level_slices(self, a, b, csr):
        """Device data of one level: entry columns, values and a row-sum selector for each group."""
        xp = self.xp
        out = {}
        for name, (indptr, cols, *vals) in (
            ("mixed", self._mixed),
            ("band", self._band),
        ):
            p0, p1 = indptr[a], indptr[b]
            if p1 == p0:
                out[name] = None
                continue
            rows = csr(
                (
                    xp.ones(p1 - p0),
                    xp.arange(p1 - p0),
                    xp.asarray(indptr[a : b + 1] - p0),
                ),
                shape=(b - a, p1 - p0),
            )
            out[name] = (rows, xp.asarray(cols[p0:p1]), *(v[p0:p1] for v in vals))
        return out

    def prepare(self, c, ri):
        """Set the step: ``c = gamma h`` and inverse density ``ri``."""
        xp, op = self.xp, self.op
        d = (op.d_int + ri * op.d_dec).T[xp.asarray(self.perm)]  # (N, n_k), level order
        den = 1.0 - c * d
        w = d / den
        kinv = None
        if self.corner is not None:
            n_P = op.T_P.shape[0]
            wc = w[xp.asarray(self.corner), :n_P]
            K = xp.eye(n_P)[None] - c * op.T_P[None, :, :n_P] * wc[:, None, :]
            kinv = xp.linalg.inv(K)
        self._prepared = (c, ri, den, w, kinv)

    def solve(self, rhs):
        """Solve the lower system for a mode-major ``rhs`` ``(n_k, N)`` or ``(n_k, N, K)``; returns the same layout."""
        xp = self.xp
        b = xp.moveaxis(rhs.reshape(rhs.shape[:2] + (-1,)), 0, 1)[xp.asarray(self.perm)]
        x = self._solve_levels(b)
        return xp.moveaxis(x[xp.asarray(self.inv_perm)], 0, 1).reshape(rhs.shape)

    def _solve_levels(self, b):
        """Array version of the sweep: ``b`` and the result ``(N, n_k, K)`` in level order."""
        xp, op = self.xp, self.op
        c, ri, den, w, kinv = self._prepared
        x, xs = xp.zeros_like(b), xp.zeros_like(b)
        n_cols = b.shape[2]
        for k, (a, e) in enumerate(self.levels):
            lv = self._slices[k]
            r = b[a:e]
            if lv["mixed"] is not None:
                rows, cols, va, vd = lv["mixed"]
                terms = (va + ri * vd)[:, :, None] * xs[cols]
                prod = (rows @ terms.reshape(len(cols), -1)).reshape(e - a, -1, n_cols)
                if op.mix is not None:
                    n_D = op.mix.shape[0]
                    prod[:, :n_D] = xp.einsum("ij,rjk->rik", op.mix, prod[:, :n_D])
                r = r + c * prod
            if lv["band"] is not None:
                rows, cols, vb = lv["band"]
                terms = vb[:, :, None] * xs[cols]
                r = r + c * (rows @ terms.reshape(len(cols), -1)).reshape(
                    e - a, -1, n_cols
                )
            z = r / den[a:e, :, None]
            x[a:e] = z
            xs[a:e] = z
            corner = self._corner_slices[k] if self.corner is not None else None
            if corner is not None:
                local, slot = corner
                n_P = op.T_P.shape[0]
                u = xp.einsum("pl,rlk->rpk", op.T_P, z[local])
                y = xp.einsum("rij,rjk->rik", kinv[slot], u)
                xc = z[local]
                xc[:, :n_P] += c * w[a + local, :n_P, None] * y
                x[a + local] = xc
                xc = xc.copy()
                xc[:, :n_P] += y
                xs[a + local] = xc
        return x


def gmres(apply, b, precond, x0=None, rtol=1e-8, restart=40, maxiter=400, xp=np):
    """Right-preconditioned restarted GMRES on arrays of any shape and module ``xp``.

    Returns ``(x, iterations)``; stops at ``|b - apply(x)| <= rtol |b|`` or after ``maxiter`` iterations.
    """
    shape = b.shape
    b = b.ravel()
    x = xp.zeros_like(b) if x0 is None else x0.ravel().copy()
    b_norm = float(xp.linalg.norm(b))
    if b_norm == 0.0:
        return x.reshape(shape), 0
    its = 0
    V = xp.empty((restart + 1, b.size), dtype=b.dtype)
    Z = xp.empty((restart, b.size), dtype=b.dtype)
    while True:
        r = b - apply(x.reshape(shape)).ravel()
        beta = float(xp.linalg.norm(r))
        if beta <= rtol * b_norm or its >= maxiter:
            return x.reshape(shape), its
        H = np.zeros((restart + 1, restart))
        g = np.zeros(restart + 1)
        g[0] = beta
        V[0] = r / beta
        for j in range(restart):
            Z[j] = precond(V[j].reshape(shape)).ravel()
            v = apply(Z[j].reshape(shape)).ravel()
            for _ in range(2):  # classical Gram-Schmidt, reorthogonalized
                h = V[: j + 1] @ v
                v -= V[: j + 1].T @ h
                H[: j + 1, j] += _to_host(h)
            H[j + 1, j] = float(xp.linalg.norm(v))
            V[j + 1] = v / H[j + 1, j]
            its += 1
            y = np.linalg.lstsq(H[: j + 2, : j + 1], g[: j + 2], rcond=None)[0]
            res = np.linalg.norm(H[: j + 2, : j + 1] @ y - g[: j + 2])
            if res <= rtol * b_norm or its >= maxiter:
                break
        x += Z[: j + 1].T @ xp.asarray(y)


class ImexStepper:
    """ARS(2,2,2) steps of ``dPhi/dX = F_imp + F_exp`` (:class:`ImexOperator`) with GMRES / :class:`LevelSweep`
    stage solves.

    Args:
      op: an :class:`ImexOperator`.
      e_tot: ``(N,)`` total energy of every state (the sweep's order).
      rtol: relative residual of the stage solves.
      gpu: CUDA device of a CuPy ``op``, or None.

    ``tangent``: :class:`TangentSources` of tangent columns, or None.
    """

    def __init__(self, op, e_tot, rtol=1e-8, gpu=None):
        self.op, self.rtol, self.gpu = op, rtol, gpu
        self.sweep = LevelSweep(op, e_tot)
        self.tangent = None
        self.iterations = []

    def _solve(self, rhs, c, ri, x0):
        """``(I - c J) x = rhs``, every column scaled to unit norm (the columns are independent systems)."""
        op = self.op
        scale = op.xp.linalg.norm(rhs.reshape(op.n_k * op.N, -1), axis=0)
        scale[scale == 0.0] = 1.0
        x, its = gmres(
            lambda v: v - c * op.implicit(v, ri),
            rhs / scale,
            self.sweep.solve,
            x0=None if x0 is None else x0 / scale,
            rtol=self.rtol,
            xp=op.xp,
        )
        self.iterations.append(its)
        return x * scale

    def step(self, phi, h, ri):
        """Advance the mode-major state ``phi`` by ``h`` at inverse density ``ri``."""
        c = GAMMA * h
        self.sweep.prepare(c, ri)
        explicit_1 = self.op.explicit(phi, ri, self.tangent)
        stage_2 = self._solve(phi + c * explicit_1, c, ri, x0=phi)
        explicit_2 = self.op.explicit(stage_2, ri, self.tangent)
        implicit_2 = self.op.implicit(stage_2, ri)
        rhs_3 = phi + h * (
            DELTA * explicit_1 + (1.0 - DELTA) * explicit_2 + (1.0 - GAMMA) * implicit_2
        )
        return self._solve(rhs_3, c, ri, x0=stage_2)


def _device_context(stepper):
    """The CUDA device context of ``stepper``, or a null context."""
    if stepper.gpu is None:
        return nullcontext()
    import cupy

    return cupy.cuda.Device(stepper.gpu)


def imex_driver(nsteps, dX, rho_inv, stepper, phi, grid_idcs=(), tangent=None):
    """Run ``nsteps`` IMEX steps along a path.

    Args:
      nsteps, dX, rho_inv: the path, ``dX`` and ``rho_inv`` of shape ``(nsteps,)``.
      stepper: an :class:`ImexStepper`.
      phi: initial state, ``(dim,)`` or ``(dim, K)`` in MCEq's order.
      grid_idcs: step indices after which to snapshot the state.
      tangent: :class:`~MCEq.solvers.etd2.TangentOperator` of the columns of ``phi``, or None.

    Returns:
      ``(solution, grid_snapshots)`` on the host in float64, shaped like ``phi`` and
      ``(len(grid_idcs),) + phi.shape``.
    """
    op = stepper.op
    phi = np.asarray(phi, dtype=np.float64)
    snapshots = {int(k) for k in grid_idcs}
    grid_sol = []
    with _device_context(stepper):
        stepper.tangent = None if tangent is None else TangentSources(tangent, op)
        x = op.xp.asarray(phi.reshape((op.n_k, op.N) + phi.shape[1:]))
        for k in range(nsteps):
            x = stepper.step(x, float(dX[k]), float(rho_inv[k]))
            if k in snapshots:
                grid_sol.append(_to_host(x).reshape(phi.shape))
        return _to_host(x).reshape(phi.shape), np.array(grid_sol)


def imex_stepper(builder, dec_m, particles, e_grid, db, sec_ops=None, gpu=None):
    """:class:`ImexStepper` of an MCEq run.

    Args:
      builder: the run's ``MatrixBuilder`` (``int_m_hadr``, ``dEdx_band``).
      dec_m: decay matrix.
      particles: the run's particles (``mceqidx``, ``lidx``, ``uidx``, ``mass``, ``is_em``, ``is_lepton``).
      e_grid: kinetic-energy grid.
      db: the run's database (``is_2d``, ``n_k``, ``k_grid``); 2D runs mix the leading modes with the
        backward removal when ``config.legendre_backward_removal`` is set.
      sec_ops: sec(theta) operator set (``T_P``, ``low_e_idx``), or None.
      gpu: CUDA device to run on with CuPy, or None for numpy.
    """
    n_k = db.n_k if db.is_2d else 1
    mix = None
    if db.is_2d and config.legendre_backward_removal:
        mix = backward_removal(dense_count(db.k_grid) - 1)
    coupling = None if sec_ops is None else (sec_ops["T_P"], sec_ops["low_e_idx"])
    N = dec_m.shape[0] // n_k
    light, e_tot = np.zeros(N, dtype=bool), np.zeros(N)
    for p in particles:
        if p.mceqidx >= 0:
            light[p.lidx : p.uidx] = p.is_em or p.is_lepton
            e_tot[p.lidx : p.uidx] = np.asarray(e_grid) + p.mass
    if gpu is None:
        xp, device = np, nullcontext()
    else:
        import cupy as xp

        device = xp.cuda.Device(gpu)
    with device:
        op = ImexOperator(
            builder.int_m_hadr,
            dec_m,
            builder.dEdx_band,
            light,
            np.arange(N) // len(e_grid),
            n_k,
            mix,
            coupling,
            xp,
        )
        return ImexStepper(op, e_tot, gpu=gpu)


def choose_integrator(name, *, is_2d, supported, stiffness_cap):
    """``"etd2"`` or ``"imex"`` for ``config.integrator`` ``name``.

    ``"auto"`` takes IMEX on 2D runs whose ETD2 step cap ``stiffness_cap()`` (g/cm^2) is below
    ``config.imex_auto_dx``, ETD2 otherwise. ``supported`` is False where IMEX does not apply (no
    shared path, 1D secant table): ``"auto"`` then takes ETD2 and ``"imex"`` raises.
    """
    name = name.lower()
    if name not in ("auto", "etd2", "imex"):
        raise ValueError(
            f"config.integrator must be 'auto', 'etd2' or 'imex', got {name!r}"
        )
    if name == "imex" and not supported:
        raise NotImplementedError(
            "The IMEX integrator needs one shared path and no 1D secant table "
            "(config.secant_1d_table = None)."
        )
    if name != "auto":
        return name
    if supported and is_2d and stiffness_cap() < config.imex_auto_dx:
        return "imex"
    return "etd2"
