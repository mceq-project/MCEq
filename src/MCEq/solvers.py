import numpy as np

from MCEq import config
from MCEq.misc import info
from tqdm import tqdm


def solv_numpy(nsteps, dX, rho_inv, int_m, dec_m, phi, grid_idcs, **kwargs):
    """:mod:`numpy` implementation of forward-euler integration.

    Args:
      nsteps (int): number of integration steps
      dX (numpy.array[nsteps]): vector of step-sizes
        :math:`\\Delta X_i` in g/cm**2
      rho_inv (numpy.array[nsteps]): vector of density values
        :math:`\\frac{1}{\\rho(X_i)}`
      int_m (numpy.array): interaction matrix :eq:`int_matrix`
        in dense or sparse representation
      dec_m (numpy.array): decay  matrix :eq:`dec_matrix` in dense
        or sparse representation
      phi (numpy.array): initial state vector :math:`\\Phi(X_0)`
    Returns:
      numpy.array: state vector :math:`\\Phi(X_{nsteps})` after integration
    """

    grid_sol = []
    grid_step = 0

    imc = int_m
    dmc = dec_m
    dxc = dX
    ric = rho_inv
    phc = phi

    dXaccum = 0.0

    from time import time

    start = time()
    # TK: 2D solver (same approach as 1D, but looping over the Hankel modes k)
    if config.enable_2D:

        for step in tqdm(range(nsteps)):

            if config.muon_multiple_scattering:
                muon_mult_scat_kernel = kwargs['muon_scat_kernel'](dxc[step])
                for idx in kwargs['muon_inds']:
                    for k in range(len(config.k_grid)):
                        phc[k][idx:idx + kwargs['edim']] *= muon_mult_scat_kernel[k]

            int_deriv_k = [
            imc[k].dot(phc[k]) for k in range(len(config.k_grid))
            ]
            dec_deriv_k = [
                dmc[k].dot(phc[k]) * ric[step] for k in range(len(config.k_grid))
            ]
            full_deriv_k = [
                int_deriv_k[k] + dec_deriv_k[k] for k in range(len(config.k_grid))
            ]

            phc += np.array(full_deriv_k) * dxc[step]

            if (
                grid_idcs
                and grid_step < len(grid_idcs)
                and grid_idcs[grid_step] == step
            ):
                grid_sol.append(np.copy(phc))
                grid_step += 1
    # 1D solver
    else:

        for step in range(nsteps):
            phc += (imc.dot(phc) + dmc.dot(ric[step] * phc)) * dxc[step]

            dXaccum += dxc[step]

            if (
                grid_idcs
                and grid_step < len(grid_idcs)
                and grid_idcs[grid_step] == step
            ):
                grid_sol.append(np.copy(phc))
                grid_step += 1

    info(
        2,
        f"Performance: {1e3 * (time() - start) / float(nsteps):6.2f}ms/iteration",
    )

    return phc, np.array(grid_sol)


class CUDASparseContext:
    """This class handles the transfer between CPU and GPU memory,
    and the calling of GPU kernels. Initialized by :class:`MCEq.core.MCEqRun`
    and used by :func:`solv_CUDA_sparse`.
    """

    def __init__(self, int_m, dec_m, device_id=config.cuda_gpu_id):
        # Setup GPU stuff and upload data to it
        try:
            import cupy as cp
            import cupyx.scipy as cpx

            self.cp = cp
            self.cpx = cpx
        except ImportError:
            raise Exception(
                "solv_CUDA_sparse(): Numbapro CUDA libaries not "
                + "installed.\nCan not use GPU."
            )

        cp.cuda.Device(device_id).use()
        self.set_matrices(int_m, dec_m)

    def set_matrices(self, int_m, dec_m):
        """Upload sparce matrices to GPU memory"""
        if self.cpx.sparse.isspmatrix_csr(int_m):
            self.cu_int_m = int_m
        else:
            self.cu_int_m = self.cpx.sparse.csr_matrix(int_m, dtype=config.floatlen)
        if self.cpx.sparse.isspmatrix_csr(dec_m):
            self.cu_dec_m = dec_m
        else:
            self.cu_dec_m = self.cpx.sparse.csr_matrix(dec_m, dtype=config.floatlen)

    def alloc_grid_sol(self, dim, nsols):
        """Allocates memory for intermediate if grid solution requested."""
        self.curr_sol_idx = 0
        self.grid_sol = self.cp.zeros((nsols, dim))

    def dump_sol(self):
        """Saves current solution to a new index in grid solution memory."""
        self.cp.copyto(self.grid_sol[self.curr_sol_idx, :], self.cu_curr_phi)
        self.curr_sol_idx += 1
        # self.grid_sol[self.curr_sol, :] = self.cu_curr_phi

    def get_gridsol(self):
        """Downloads grid solution to main memory."""
        return self.cp.asnumpy(self.grid_sol)

    def set_phi(self, phi):
        """Uploads initial condition to GPU memory."""
        self.cu_curr_phi = self.cp.asarray(phi, dtype=config.floatlen)

    def get_phi(self):
        """Downloads current solution from GPU memory."""
        return self.cp.asnumpy(self.cu_curr_phi)

    def solve_step(self, rho_inv, dX):
        """Makes one solver step on GPU using cuSparse (BLAS)"""

        cu_delta_phi = self.cu_int_m @ self.cu_curr_phi
        cu_delta_phi += self.cu_dec_m @ (self.cu_curr_phi * rho_inv)
        self.cu_curr_phi += dX * cu_delta_phi


def solv_CUDA_sparse(nsteps, dX, rho_inv, context, phi, grid_idcs):
    """`NVIDIA CUDA cuSPARSE <https://developer.nvidia.com/cusparse>`_ implementation
    of forward-euler integration.

    Function requires a working :mod:`accelerate` installation.

    Args:
      nsteps (int): number of integration steps
      dX (numpy.array[nsteps]): vector of step-sizes
        :math:`\\Delta X_i` in g/cm**2
      rho_inv (numpy.array[nsteps]): vector of density values
        :math:`\\frac{1}{\\rho(X_i)}`
      context (object): Instance of :class:`CUDASparseContext`
      phi (numpy.array): initial state vector :math:`\\Phi(X_0)`
      grid_idcs (numpy.array): indices, when to save the state vector

    Returns:
      numpy.array: state vector :math:`\\Phi(X_{nsteps})` after final
        step
      numpy.array: state vector copies at `grid_idcs` or empty list
    """

    c = context
    c.set_phi(phi)

    grid_step = 0

    from time import time

    start = time()
    if len(grid_idcs) > 0:
        c.alloc_grid_sol(phi.shape[0], len(grid_idcs))

    for step in range(nsteps):
        c.solve_step(rho_inv[step], dX[step])

        if grid_idcs and grid_step < len(grid_idcs) and grid_idcs[grid_step] == step:
            c.dump_sol()
            grid_step += 1

    info(
        2,
        f"Performance: {1e3 * (time() - start) / float(nsteps):6.2f}ms/iteration",
    )

    return c.get_phi(), c.get_gridsol() if len(grid_idcs) > 0 else []


def solv_MKL_sparse(nsteps, dX, rho_inv, int_m, dec_m, phi, grid_idcs):
    # mu_loss_handler):
    """`Intel MKL sparse BLAS
    <https://software.intel.com/en-us/articles/
    intel-mkl-sparse-blas-overview?language=en>`_
    implementation of forward-euler integration.

    Function requires that the path to the MKL runtime library ``libmkl_rt.[so/dylib]``
    defined in the config file.

    Args:
      nsteps (int): number of integration steps
      dX (numpy.array[nsteps]): vector of step-sizes
        :math:`\\Delta X_i` in g/cm**2
      rho_inv (numpy.array[nsteps]): vector of density values
        :math:`\\frac{1}{\\rho(X_i)}`
      int_m (numpy.array): interaction matrix :eq:`int_matrix`
        in dense or sparse representation
      dec_m (numpy.array): decay  matrix :eq:`dec_matrix` in dense
        or sparse representation
      phi (numpy.array): initial state vector :math:`\\Phi(X_0)`
      grid_idcs (list): indices, when to save the state vector

    Returns:
      numpy.array: state vector :math:`\\Phi(X_{nsteps})` after integration
      numpy.array: state vector copies at `grid_idcs` or empty list
    """

    from ctypes import POINTER, byref, c_char, c_int

    from MCEq.config import mkl

    np_fl = config.floatlen

    if config.floatlen == np.float64:
        from ctypes import c_double as fl_pr

        # sparse CSR-matrix x dense vector
        gemv = mkl.mkl_dcsrmv
    else:
        from ctypes import c_float as fl_pr

        # sparse CSR-matrix x dense vector
        gemv = mkl.mkl_scsrmv

    # Prepare CTYPES pointers for MKL sparse CSR BLAS
    int_m_data = int_m.data.ctypes.data_as(POINTER(fl_pr))
    int_m_ci = int_m.indices.ctypes.data_as(POINTER(c_int))
    int_m_pb = int_m.indptr[:-1].ctypes.data_as(POINTER(c_int))
    int_m_pe = int_m.indptr[1:].ctypes.data_as(POINTER(c_int))

    dec_m_data = dec_m.data.ctypes.data_as(POINTER(fl_pr))
    dec_m_ci = dec_m.indices.ctypes.data_as(POINTER(c_int))
    dec_m_pb = dec_m.indptr[:-1].ctypes.data_as(POINTER(c_int))
    dec_m_pe = dec_m.indptr[1:].ctypes.data_as(POINTER(c_int))

    npphi = np.copy(phi).astype(np_fl)
    phi = npphi.ctypes.data_as(POINTER(fl_pr))
    npdelta_phi = np.zeros_like(npphi)
    delta_phi = npdelta_phi.ctypes.data_as(POINTER(fl_pr))

    trans = byref(c_char(b"n"))
    npmatd = np.chararray(6)
    npmatd[0] = b"G"
    npmatd[3] = b"C"
    matdsc = npmatd.ctypes.data_as(POINTER(c_char))
    m = byref(c_int(int_m.shape[0]))
    cdzero = byref(fl_pr(0.0))
    cdone = byref(fl_pr(1.0))

    grid_step = 0
    grid_sol = []

    from time import time

    start = time()

    for step in range(nsteps):
        # delta_phi = int_m.dot(phi)
        gemv(
            trans,
            m,
            m,
            cdone,
            matdsc,
            int_m_data,
            int_m_ci,
            int_m_pb,
            int_m_pe,
            phi,
            cdzero,
            delta_phi,
        )
        # delta_phi = rho_inv * dec_m.dot(phi) + delta_phi
        gemv(
            trans,
            m,
            m,
            byref(fl_pr(rho_inv[step])),
            matdsc,
            dec_m_data,
            dec_m_ci,
            dec_m_pb,
            dec_m_pe,
            phi,
            cdone,
            delta_phi,
        )
        npphi += npdelta_phi * dX[step]
        # phi = delta_phi * dX + phi
        # axpy(m, fl_pr(dX[step]), delta_phi, cione, phi, cione)
        # print(np.sum(npphi))

        if grid_idcs and grid_step < len(grid_idcs) and grid_idcs[grid_step] == step:
            grid_sol.append(np.copy(npphi))
            grid_step += 1

    info(
        2,
        f"Performance: {1e3 * (time() - start) / float(nsteps):6.2f}ms/iteration",
    )

    return npphi, np.asarray(grid_sol)


def solv_spacc_sparse(nsteps, dX, rho_inv, spacc_int_m, spacc_dec_m, phi, grid_idcs):
    # mu_loss_handler):
    """Apple Accelerate (vecLib) implementation.

    Args:
      nsteps (int): number of integration steps
      dX (numpy.array[nsteps]): vector of step-sizes
        :math:`\\Delta X_i` in g/cm**2
      rho_inv (numpy.array[nsteps]): vector of density values
        :math:`\\frac{1}{\\rho(X_i)}`
      int_m (numpy.array): interaction matrix :eq:`int_matrix`
        in dense or sparse representation
      dec_m (numpy.array): decay  matrix :eq:`dec_matrix` in
        dense or sparse representation
      phi (numpy.array): initial state vector :math:`\\Phi(X_0)`
      grid_idcs (list): indices at which longitudinal solutions
        have to be saved.

    Returns:
      numpy.array: state vector :math:`\\Phi(X_{nsteps})` after integration
    """

    from ctypes import POINTER, c_double

    from MCEq import spacc

    dim_phi = int(phi.shape[0])
    npphi = np.copy(phi)
    phi = npphi.ctypes.data_as(POINTER(c_double))
    npdelta_phi = np.zeros_like(npphi)
    delta_phi = npdelta_phi.ctypes.data_as(POINTER(c_double))

    grid_step = 0
    grid_sol = []

    from time import time

    start = time()

    for step in range(nsteps):
        # delta_phi = int_m.dot(phi)
        npdelta_phi *= 0
        spacc_int_m.gemv_ctargs(1.0, phi, delta_phi)

        # delta_phi = rho_inv * dec_m.dot(phi) + delta_phi
        spacc_dec_m.gemv_ctargs(rho_inv[step], phi, delta_phi)

        # phi = delta_phi * dX + phi
        spacc.daxpy(dim_phi, dX[step], delta_phi, phi)

        # axpy(m, fl_pr(dX[step]), delta_phi, cione, phi, cione)

        if grid_idcs and grid_step < len(grid_idcs) and grid_idcs[grid_step] == step:
            grid_sol.append(np.copy(npphi))
            grid_step += 1

    info(
        2,
        f"Performance: {1e3 * (time() - start) / float(nsteps):6.2f}ms/iteration",
    )

    return npphi, np.asarray(grid_sol)


# # TODO: Debug this and transition to BDF
# def _odepack(dXstep=.1,
#                 initial_depth=0.0,
#                 int_grid=None,
#                 grid_var='X',
#                 *args,
#                 **kwargs):
#     """Solves the transport equations with solvers from ODEPACK.

#     Args:
#         dXstep (float): external step size (adaptive sovlers make more
#           steps internally)
#         initial_depth (float): starting depth in g/cm**2
#         int_grid (list): list of depths at which results are recorded
# grid_var (str): Can be depth `X` or something else (currently only `X`
# supported)

#     """
#     from scipy.integrate import ode
#     ri = self.density_model.r_X2rho

#     if config.enable_muon_energy_loss:
#         raise NotImplementedError(
#             'Energy loss not imlemented for this solver.')

#     # Functional to solve
#     def dPhi_dX(X, phi, *args):
#         return self.int_m.dot(phi) + self.dec_m.dot(ri(X) * phi)

#     # Jacobian doesn't work with sparse matrices, and any precision
#     # or speed advantage disappear if used with dense algebra
#     def jac(X, phi, *args):
#         # print 'jac', X, phi
#         return (self.int_m + self.dec_m * ri(X)).todense()

#     # Initial condition
#     phi0 = np.copy(self.phi0)

#     # Initialize variables
#     grid_sol = []

#     # Setup solver
#     r = ode(dPhi_dX).set_integrator(
#         with_jacobian=False, **config.ode_params)

#     if int_grid is not None:
#         initial_depth = int_grid[0]
#         int_grid = int_grid[1:]
#         max_X = int_grid[-1]
#         grid_sol.append(phi0)

#     else:
#         max_X = self.density_model.max_X

#     info(
#         1,
#         'your X-grid is shorter then the material',
#         condition=max_X < self.density_model.max_X)
#     info(
#         1,
#         'your X-grid exceeds the dimentsions of the material',
#         condition=max_X > self.density_model.max_X)

#     # Initial value
#     r.set_initial_value(phi0, initial_depth)

#     info(
#         2, 'initial depth: {0:3.2e}, maximal depth {1:}'.format(
#             initial_depth, max_X))

#     start = time()
#     if int_grid is None:
#         i = 0
#         while r.successful() and (r.t + dXstep) < max_X - 1:
#             info(5, "Solving at depth X =", r.t, condition=(i % 5000) == 0)
#             r.integrate(r.t + dXstep)
#             i += 1
#         if r.t < max_X:
#             r.integrate(max_X)
#         # Do last step to make sure the rational number max_X is reached
#         r.integrate(max_X)
#     else:
#         for i, Xi in enumerate(int_grid):
#             info(5, 'integrating at X =', Xi, condition=i % 10 == 0)

#             while r.successful() and (r.t + dXstep) < Xi:
#                 r.integrate(r.t + dXstep)

#             # Make sure the integrator arrives at requested step
#             r.integrate(Xi)
#             # Store the solution on grid
#             grid_sol.append(r.y)

#     info(2,
#             'time elapsed during integration: {1} sec'.format(time() - start))

#     self.solution = r.y
#     self.grid_sol = grid_sol
