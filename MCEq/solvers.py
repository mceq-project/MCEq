
import numpy as np
import mceq_config as config
from MCEq.misc import info


def solv_numpy(nsteps, dX, rho_inv, int_m, dec_m, phi, grid_idcs):
    """:mod:`numpy` implementation of forward-euler integration.

    Args:
      nsteps (int): number of integration steps
      dX (numpy.array[nsteps]): vector of step-sizes :math:`\\Delta X_i` in g/cm**2
      rho_inv (numpy.array[nsteps]): vector of density values :math:`\\frac{1}{\\rho(X_i)}`
      int_m (numpy.array): interaction matrix :eq:`int_matrix` in dense or sparse representation
      dec_m (numpy.array): decay  matrix :eq:`dec_matrix` in dense or sparse representation
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

    dXaccum = 0.

    from time import time
    start = time()

    for step in range(nsteps):
        phc += (imc.dot(phc) + dmc.dot(ric[step] * phc)) * dxc[step]

        dXaccum += dxc[step]

        if (grid_idcs and grid_step < len(grid_idcs)
                and grid_idcs[grid_step] == step):
            grid_sol.append(np.copy(phc))
            grid_step += 1

    info(
        2, "Performance: {0:6.2f}ms/iteration".format(1e3 * (time() - start) /
                                                      float(nsteps)))

    return phc, np.array(grid_sol)


class CUDASparseContext(object):
    """This class handles the transfer between CPU and GPU memory, and the calling
    of GPU kernels. Initialized by :class:`MCEq.core.MCEqRun` and used by
    :func:`solv_CUDA_sparse`.
    """
    def __init__(self, int_m, dec_m, device_id=0):

        if config.cuda_fp_precision == 32:
            self.fl_pr = np.float32
        elif config.cuda_fp_precision == 64:
            self.fl_pr = np.float64
        else:
            raise Exception(
                "CUDASparseContext(): Unknown precision specified.")
        # Setup GPU stuff and upload data to it
        try:
            import cupy as cp
            import cupyx.scipy as cpx
            self.cp = cp
            self.cpx = cpx
            self.cubl = cp.cuda.cublas
        except ImportError:
            raise Exception("solv_CUDA_sparse(): Numbapro CUDA libaries not " +
                            "installed.\nCan not use GPU.")

        cp.cuda.Device(config.cuda_gpu_id).use()
        self.cubl_handle = self.cubl.create()
        self.set_matrices(int_m, dec_m)

    def set_matrices(self, int_m, dec_m):
        """Upload sparce matrices to GPU memory"""
        self.cu_int_m = self.cpx.sparse.csr_matrix(int_m, dtype=self.fl_pr)
        self.cu_dec_m = self.cpx.sparse.csr_matrix(dec_m, dtype=self.fl_pr)
        self.cu_delta_phi = self.cp.zeros(self.cu_int_m.shape[0],
                                          dtype=self.fl_pr)

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
        self.cu_curr_phi = self.cp.asarray(phi, dtype=self.fl_pr)

    def get_phi(self):
        """Downloads current solution from GPU memory."""
        return self.cp.asnumpy(self.cu_curr_phi)

    def solve_step(self, rho_inv, dX):
        """Makes one solver step on GPU using cuSparse (BLAS)"""

        self.cp.cusparse.csrmv(a=self.cu_int_m,
                               x=self.cu_curr_phi,
                               y=self.cu_delta_phi,
                               alpha=1.,
                               beta=0.)
        self.cp.cusparse.csrmv(a=self.cu_dec_m,
                               x=self.cu_curr_phi,
                               y=self.cu_delta_phi,
                               alpha=rho_inv,
                               beta=1.)
        self.cubl.saxpy(self.cubl_handle, self.cu_delta_phi.shape[0], dX,
                        self.cu_delta_phi.data.ptr, 1,
                        self.cu_curr_phi.data.ptr, 1)


def solv_CUDA_sparse(nsteps, dX, rho_inv, context, phi, grid_idcs):
    """`NVIDIA CUDA cuSPARSE <https://developer.nvidia.com/cusparse>`_ implementation
    of forward-euler integration.

    Function requires a working :mod:`accelerate` installation.

    Args:
      nsteps (int): number of integration steps
      dX (numpy.array[nsteps]): vector of step-sizes :math:`\\Delta X_i` in g/cm**2
      rho_inv (numpy.array[nsteps]): vector of density values :math:`\\frac{1}{\\rho(X_i)}`
      int_m (numpy.array): interaction matrix :eq:`int_matrix` in dense or sparse representation
      dec_m (numpy.array): decay  matrix :eq:`dec_matrix` in dense or sparse representation
      phi (numpy.array): initial state vector :math:`\\Phi(X_0)`
      mu_loss_handler (object): object of type :class:`SemiLagrangianEnergyLosses`
    Returns:
      numpy.array: state vector :math:`\\Phi(X_{nsteps})` after integration
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

        if (grid_idcs and grid_step < len(grid_idcs)
                and grid_idcs[grid_step] == step):
            c.dump_sol()
            grid_step += 1

    info(
        2, "Performance: {0:6.2f}ms/iteration".format(1e3 * (time() - start) /
                                                      float(nsteps)))

    return c.get_phi(), c.get_gridsol() if len(grid_idcs) > 0 else []


def solv_MKL_sparse(nsteps, dX, rho_inv, int_m, dec_m, phi, grid_idcs):
    # mu_loss_handler):
    """`Intel MKL sparse BLAS
    <https://software.intel.com/en-us/articles/intel-mkl-sparse-blas-overview?language=en>`_
    implementation of forward-euler integration.

    Function requires that the path to the MKL runtime library ``libmkl_rt.[so/dylib]``
    defined in the config file.

    Args:
      nsteps (int): number of integration steps
      dX (numpy.array[nsteps]): vector of step-sizes :math:`\\Delta X_i` in g/cm**2
      rho_inv (numpy.array[nsteps]): vector of density values :math:`\\frac{1}{\\rho(X_i)}`
      int_m (numpy.array): interaction matrix :eq:`int_matrix` in dense or sparse representation
      dec_m (numpy.array): decay  matrix :eq:`dec_matrix` in dense or sparse representation
      phi (numpy.array): initial state vector :math:`\\Phi(X_0)`
      grid_idcs (list): indices at which longitudinal solutions have to be saved.

    Returns:
      numpy.array: state vector :math:`\\Phi(X_{nsteps})` after integration
    """

    from ctypes import c_int, c_char, POINTER, byref
    from mceq_config import mkl

    gemv = None
    axpy = None
    np_fl = None
    from ctypes import c_double as fl_pr
    # sparse CSR-matrix x dense vector
    gemv = mkl.mkl_dcsrmv
    # dense vector + dense vector
    axpy = mkl.cblas_daxpy
    np_fl = np.float64

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

    trans = c_char(b'n')
    npmatd = np.chararray(6)
    npmatd[0] = b'G'
    npmatd[3] = b'C'
    matdsc = npmatd.ctypes.data_as(POINTER(c_char))
    m = c_int(int_m.shape[0])
    cdzero = fl_pr(0.)
    cdone = fl_pr(1.)
    cione = c_int(1)

    grid_step = 0
    grid_sol = []

    from time import time
    start = time()

    for step in range(nsteps):
        # delta_phi = int_m.dot(phi)
        gemv(byref(trans), byref(m), byref(m), byref(cdone),
             matdsc, int_m_data, int_m_ci, int_m_pb, int_m_pe, phi,
             byref(cdzero), delta_phi)
        # delta_phi = rho_inv * dec_m.dot(phi) + delta_phi
        gemv(byref(trans), byref(m), byref(m), byref(fl_pr(rho_inv[step])),
             matdsc, dec_m_data, dec_m_ci, dec_m_pb, dec_m_pe, phi,
             byref(cdone), delta_phi)
        # phi = delta_phi * dX + phi
        axpy(m, fl_pr(dX[step]), delta_phi, cione, phi, cione)

        if (grid_idcs and grid_step < len(grid_idcs)
                and grid_idcs[grid_step] == step):
            grid_sol.append(np.copy(npphi))
            grid_step += 1

    info(
        2, "Performance: {0:6.2f}ms/iteration".format(1e3 * (time() - start) /
                                                      float(nsteps)))

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
#         dXstep (float): external step size (adaptive sovlers make more steps internally)
#         initial_depth (float): starting depth in g/cm**2
#         int_grid (list): list of depths at which results are recorded
#         grid_var (str): Can be depth `X` or something else (currently only `X` supported)

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
