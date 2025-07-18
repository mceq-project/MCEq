import numpy as np

from MCEq import config
from MCEq.misc import info


def solv_numpy(nsteps, dX, rho_inv, int_m, dec_m, phi, grid_idcs):
    """:mod:`numpy` implementation of forward-euler integration.

    Args:
      nsteps (int): number of integration steps
      dX (:func:`numpy.array` [nsteps]): vector of step-sizes :math:`\\Delta X_i` in g/cm**2
      rho_inv (:func:`numpy.array` [nsteps]): vector of density values :math:`\\frac{1}{\\rho(X_i)}`
      int_m (:func:`numpy.array`): interaction matrix :eq:`int_matrix` in dense or sparse representation
      dec_m (:func:`numpy.array`): decay  matrix :eq:`dec_matrix` in dense or sparse representation
      phi (:func:`numpy.array`): initial state vector :math:`\\Phi(X_0)`
    Returns:
      :func:`numpy.array`: state vector :math:`\\Phi(X_{nsteps})` after integration
    """

    grid_sol = []
    grid_step = 0

    imc = int_m
    dmc = dec_m
    dxc = dX
    ric = rho_inv
    phc = phi.copy()  # Fix: create a copy to avoid modifying the input

    dXaccum = 0.0

    from time import time

    start = time()

    for step in range(nsteps):
        phc += (imc.dot(phc) + dmc.dot(ric[step] * phc)) * dxc[step]

        dXaccum += dxc[step]

        if grid_idcs and grid_step < len(grid_idcs) and grid_idcs[grid_step] == step:
            grid_sol.append(np.copy(phc))
            grid_step += 1

    info(
        2,
        f"Performance: {1e3 * (time() - start) / float(nsteps):6.2f}ms/iteration",
    )

    return phc, np.array(grid_sol)


class CUDASparseContext:
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
            raise Exception("CUDASparseContext(): Unknown precision specified.")
        # Setup GPU stuff and upload data to it
        try:
            import cupy as cp
            import cupyx.scipy as cpx

            self.cp = cp
            self.cpx = cpx
            self.cubl = cp.cuda.cublas
        except ImportError:
            raise Exception(
                "solv_CUDA_sparse(): CuPy not installed or not available.\n"
                + "Install with: pip install cupy-cuda12x>=12.0.0\n"
                + "CuPy 12.0+ is required for modern sparse matrix interface compatibility."
            )

        cp.cuda.Device(config.cuda_gpu_id).use()
        self.cubl_handle = self.cubl.create()
        self.set_matrices(int_m, dec_m)

    def set_matrices(self, int_m, dec_m):
        """Upload sparce matrices to GPU memory"""
        self.cu_int_m = self.cpx.sparse.csr_matrix(int_m, dtype=self.fl_pr)
        self.cu_dec_m = self.cpx.sparse.csr_matrix(dec_m, dtype=self.fl_pr)
        self.cu_delta_phi = self.cp.zeros(self.cu_int_m.shape[0], dtype=self.fl_pr)

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

        # Mimic the exact NumPy implementation:
        # phc += (imc.dot(phc) + dmc.dot(ric[step] * phc)) * dxc[step]
        
        # Calculate: int_m @ curr_phi + dec_m @ (rho_inv * curr_phi)
        int_result = self.cu_int_m @ self.cu_curr_phi
        dec_result = self.cu_dec_m @ (rho_inv * self.cu_curr_phi)
        delta = int_result + dec_result
        
        # Apply: curr_phi += delta * dX
        self.cu_curr_phi += delta * dX


def solv_CUDA_sparse(nsteps, dX, rho_inv, context, phi, grid_idcs):
    """`NVIDIA CUDA cuSPARSE <https://developer.nvidia.com/cusparse>`_ implementation
    of forward-euler integration.

    Function requires a working :mod:`accelerate` installation.

    Args:
      nsteps (int): number of integration steps
      dX (:func:`numpy.array` [nsteps]): vector of step-sizes :math:`\\Delta X_i` in g/cm**2
      rho_inv (:func:`numpy.array` [nsteps]): vector of density values :math:`\\frac{1}{\\rho(X_i)}`
      int_m (:func:`numpy.array`): interaction matrix :eq:`int_matrix` in dense or sparse representation
      dec_m (:func:`numpy.array`): decay  matrix :eq:`dec_matrix` in dense or sparse representation
      phi (:func:`numpy.array`): initial state vector :math:`\\Phi(X_0)`
      mu_loss_handler (object): object of type :class:`SemiLagrangianEnergyLosses`
    Returns:
      :func:`numpy.array`: state vector :math:`\\Phi(X_{nsteps})` after integration
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
    <https://software.intel.com/en-us/articles/intel-mkl-sparse-blas-overview?language=en>`_
    implementation of forward-euler integration.

    Function requires that the path to the MKL runtime library ``libmkl_rt.[so/dylib]``
    defined in the config file.

    Args:
      nsteps (int): number of integration steps
      dX (:func:`numpy.array` [nsteps]): vector of step-sizes :math:`\\Delta X_i` in g/cm**2
      rho_inv (:func:`numpy.array` [nsteps]): vector of density values :math:`\\frac{1}{\\rho(X_i)}`
      int_m (:func:`numpy.array`): interaction matrix :eq:`int_matrix` in dense or sparse representation
      dec_m (:func:`numpy.array`): decay  matrix :eq:`dec_matrix` in dense or sparse representation
      phi (:func:`numpy.array`): initial state vector :math:`\\Phi(X_0)`
      grid_idcs (list): indices at which longitudinal solutions have to be saved.

    Returns:
      :func:`numpy.array`: state vector :math:`\\Phi(X_{nsteps})` after integration
    """

    from ctypes import POINTER, byref, c_int, c_void_p, Structure
    from ctypes import c_double as fl_pr

    from MCEq.config import mkl

    # sparse CSR-matrix x dense vector
    create_csr = mkl.mkl_sparse_d_create_csr
    gemv = mkl.mkl_sparse_d_mv
    gemv_hint = mkl.mkl_sparse_set_mv_hint
    optimize = mkl.mkl_sparse_optimize
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

    m = c_int(int_m.shape[0])
    cdzero = fl_pr(0.0)
    cdone = fl_pr(1.0)
    cizero = c_int(0)
    cione = c_int(1)

    grid_step = 0
    grid_sol = []

    int_m_handle = c_void_p()
    matrix_status = create_csr(
        byref(int_m_handle), cizero, m, m, int_m_pb, int_m_pe, int_m_ci, int_m_data
    )

    assert (
        matrix_status == 0
    ), f"MKL create_csr failed with status {matrix_status} on interaction matrix"

    dec_m_handle = c_void_p()
    matrix_status = create_csr(
        byref(dec_m_handle), cizero, m, m, dec_m_pb, dec_m_pe, dec_m_ci, dec_m_data
    )

    assert (
        matrix_status == 0
    ), f"MKL create_csr failed with status {matrix_status} on decay matrix"

    # hints
    operation = int(10)  # SPARSE_OPERATION_NON_TRANSPOSE

    class MatrixDescr(Structure):
        _fields_ = [
            ("type", c_int),
            ("mode", c_int),
            ("diag", c_int),
        ]

    descr = MatrixDescr()
    descr.type = int(20)  # General matrix
    descr.mode = int(121)  # set but dont care since general matrix
    descr.diag = int(131)  # set but dont care since general matrix

    hint_status = gemv_hint(
        int_m_handle,
        operation,
        descr,
        nsteps,
    )

    assert (
        hint_status == 0
    ), f"MKL gemv_hint failed with status {hint_status} on interaction matrix"

    hint_status = gemv_hint(
        dec_m_handle,
        operation,
        descr,
        nsteps,
    )

    assert (
        hint_status == 0
    ), f"MKL gemv_hint failed with status {hint_status} on decay matrix"

    # add mkl_sparse_set_memory_hint???
    #

    optimize_status = optimize(int_m_handle)

    assert (
        optimize_status == 0
    ), f"MKL mkl_sparse_optimize failed with status {optimize_status} on interaction matrix"

    optimize_status = optimize(dec_m_handle)

    assert (
        optimize_status == 0
    ), f"MKL mkl_sparse_optimize failed with status {optimize_status} on decay matrix"

    from time import time

    start = time()

    for step in range(nsteps):
        # delta_phi = int_m.dot(phi)
        gemv(
            operation,
            cdone,
            int_m_handle,
            descr,
            phi,
            cdzero,
            delta_phi,
        )
        # delta_phi = rho_inv * dec_m.dot(phi) + delta_phi
        gemv(
            operation,
            fl_pr(rho_inv[step]),
            dec_m_handle,
            descr,
            phi,
            cdone,
            delta_phi,
        )
        # phi = delta_phi * dX + phi
        axpy(m, fl_pr(dX[step]), delta_phi, cione, phi, cione)

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
