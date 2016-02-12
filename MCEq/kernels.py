# -*- coding: utf-8 -*-
"""
:mod:`MCEq.kernels` --- calculation kernels for the forward-euler integrator
============================================================================

The module contains functions which are called by the forward-euler
integration routine :func:`MCEq.core.MCEqRun.forward_euler`. 

The integration is part of these functions. The single step

.. math::

  \Phi_{i + 1} = \\left[\\boldsymbol{M}_{int} + \\frac{1}{\\rho(X_i)}\\boldsymbol{M}_{dec}\\right]
  \\cdot \\Phi_i \\cdot \\Delta X_i

with

.. math::
  \\boldsymbol{M}_{int} = (-\\boldsymbol{1} + \\boldsymbol{C}){\\boldsymbol{\\Lambda}}_{int}
  :label: int_matrix
  
and

.. math::
  \\boldsymbol{M}_{dec} = (-\\boldsymbol{1} + \\boldsymbol{D}){\\boldsymbol{\\Lambda}}_{dec}.
  :label: dec_matrix

The functions use different libraries for sparse and dense linear algebra (BLAS): 

- The default for dense or sparse matrix representations is the function :func:`kern_numpy`.
  It uses the dot-product implementation of :mod:`numpy`. Depending on the details, your :mod:`numpy` 
  installation can be already linked to some BLAS library like as ATLAS or MKL, what typically accelerates 
  the calculation significantly.
- The fastest version, :func:`kern_MKL_sparse`, directly interfaces to the sparse BLAS routines 
  from `Intel MKL <https://software.intel.com/en-us/intel-mkl>`_ via :mod:`ctypes`. If you have the
  MKL runtime installed, this function is recommended for most purposes.
- The GPU accelerated versions :func:`kern_CUDA_dense` and :func:`kern_CUDA_sparse` are implemented
  using the cuBLAS or cuSPARSE libraries, respectively. They should be considered as experimental or
  implementation examples if you need extremely high performance. To keep Python as the main programming 
  language, these interfaces are accessed via the module :mod:`numbapro`, which is part of the 
  `Anaconda Accelerate <https://store.continuum.io/cshop/accelerate/>`_ package. It is free for
  academic use.

"""
import numpy as np
from mceq_config import config

def kern_numpy(nsteps, dX, rho_inv, int_m, dec_m,
               phi, grid_idcs, prog_bar=None):
    """:mod;`numpy` implementation of forward-euler integration.
    
    Args:
      nsteps (int): number of integration steps
      dX (numpy.array[nsteps]): vector of step-sizes :math:`\\Delta X_i` in g/cm**2
      rho_inv (numpy.array[nsteps]): vector of density values :math:`\\frac{1}{\\rho(X_i)}`
      int_m (numpy.array): interaction matrix :eq:`int_matrix` in dense or sparse representation
      dec_m (numpy.array): decay  matrix :eq:`dec_matrix` in dense or sparse representation
      phi (numpy.array): initial state vector :math:`\\Phi(X_0)` 
      prog_bar (object,optional): handle to :class:`ProgressBar` object
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

    if config['FP_precision'] == 32:
        imc = int_m.astype(np.float32)
        dmc = dec_m.astype(np.float32)
        dxc = dX.astype(np.float32)
        ric = rho_inv.astype(np.float32)
        phc = phi.astype(np.float32)

    for step in xrange(nsteps):
        if prog_bar and (step % 200 == 0):
            prog_bar.update(step)
        phc += (imc.dot(phc) + dmc.dot(ric[step] * phc)) * dxc[step]
        
        if (grid_idcs and grid_step < len(grid_idcs) 
            and grid_idcs[grid_step] == step):
            grid_sol.append(np.copy(phc))
            grid_step += 1

    return phc, grid_sol


def kern_CUDA_dense(nsteps, dX, rho_inv, int_m, dec_m,
                    phi, grid_idcs, prog_bar=None):
    """`NVIDIA CUDA cuBLAS <https://developer.nvidia.com/cublas>`_ implementation 
    of forward-euler integration.
    
    Function requires a working :mod:`numbapro` installation. It is typically slower
    compared to :func:`kern_MKL_sparse` but it depends on your hardware.
    
    Args:
      nsteps (int): number of integration steps
      dX (numpy.array[nsteps]): vector of step-sizes :math:`\\Delta X_i` in g/cm**2
      rho_inv (numpy.array[nsteps]): vector of density values :math:`\\frac{1}{\\rho(X_i)}`
      int_m (numpy.array): interaction matrix :eq:`int_matrix` in dense or sparse representation
      dec_m (numpy.array): decay  matrix :eq:`dec_matrix` in dense or sparse representation
      phi (numpy.array): initial state vector :math:`\\Phi(X_0)` 
      prog_bar (object,optional): handle to :class:`ProgressBar` object
    Returns:
      numpy.array: state vector :math:`\\Phi(X_{nsteps})` after integration
    """
    
    calc_precision = None
    if config['FP_precision'] == 32:
        calc_precision = np.float32
    elif config['FP_precision'] == 64:
        calc_precision = np.float64
    else:
        raise Exception("kern_CUDA_dense(): Unknown precision specified.")    
    
    #=======================================================================
    # Setup GPU stuff and upload data to it
    #=======================================================================
    try:
        from numbapro.cudalib.cublas import Blas  # @UnresolvedImport
        from numbapro import cuda, float32  # @UnresolvedImport
    except ImportError:
        raise Exception("kern_CUDA_dense(): Numbapro CUDA libaries not " + 
                        "installed.\nCan not use GPU.")
    cubl = Blas()
    m, n = int_m.shape
    stream = cuda.stream()
    cu_int_m = cuda.to_device(int_m.astype(calc_precision), stream)
    cu_dec_m = cuda.to_device(dec_m.astype(calc_precision), stream)
    cu_curr_phi = cuda.to_device(phi.astype(calc_precision), stream)
    cu_delta_phi = cuda.device_array(phi.shape, dtype=calc_precision)
    for step in xrange(nsteps):
        if prog_bar:
            prog_bar.update(step)
        cubl.gemv(trans='T', m=m, n=n, alpha=float32(1.0), A=cu_int_m,
            x=cu_curr_phi, beta=float32(0.0), y=cu_delta_phi)
        cubl.gemv(trans='T', m=m, n=n, alpha=float32(rho_inv[step]),
            A=cu_dec_m, x=cu_curr_phi, beta=float32(1.0), y=cu_delta_phi)
        cubl.axpy(alpha=float32(dX[step]), x=cu_delta_phi, y=cu_curr_phi)

    return cu_curr_phi.copy_to_host(), []

def kern_CUDA_sparse(nsteps, dX, rho_inv, int_m, dec_m,
                    phi, grid_idcs, prog_bar=None):
    """`NVIDIA CUDA cuSPARSE <https://developer.nvidia.com/cusparse>`_ implementation 
    of forward-euler integration.
    
    Function requires a working :mod:`numbapro` installation.
    
    Note:
      Currently some bug in :mod:`numbapro` introduces unnecessary array copies and
      slows down the execution tremendously. 
    
    Args:
      nsteps (int): number of integration steps
      dX (numpy.array[nsteps]): vector of step-sizes :math:`\\Delta X_i` in g/cm**2
      rho_inv (numpy.array[nsteps]): vector of density values :math:`\\frac{1}{\\rho(X_i)}`
      int_m (numpy.array): interaction matrix :eq:`int_matrix` in dense or sparse representation
      dec_m (numpy.array): decay  matrix :eq:`dec_matrix` in dense or sparse representation
      phi (numpy.array): initial state vector :math:`\\Phi(X_0)` 
      prog_bar (object,optional): handle to :class:`ProgressBar` object
    Returns:
      numpy.array: state vector :math:`\\Phi(X_{nsteps})` after integration
    """
    calc_precision = None
    if config['FP_precision'] == 32:
        calc_precision = np.float32
    elif config['FP_precision'] == 64:
        calc_precision = np.float64
    else:
        raise Exception("kern_CUDA_sparse(): Unknown precision specified.")    
    print ("kern_CUDA_sparse(): Warning, the performance is slower than " + 
           "dense cuBLAS or any type of MKL.")
    #=======================================================================
    # Setup GPU stuff and upload data to it
    #=======================================================================
    try:
        from numbapro.cudalib import cusparse  # @UnresolvedImport
        from numbapro.cudalib.cublas import Blas
        from numbapro import cuda, float32  # @UnresolvedImport
    except ImportError:
        raise Exception("kern_CUDA_sparse(): Numbapro CUDA libaries not " + 
                        "installed.\nCan not use GPU.")
    cusp = cusparse.Sparse()
    cubl = Blas()
    m, n = int_m.shape
    int_m_nnz = int_m.nnz
    int_m_csrValA = cuda.to_device(int_m.data.astype(calc_precision))
    int_m_csrRowPtrA = cuda.to_device(int_m.indptr)
    int_m_csrColIndA = cuda.to_device(int_m.indices)
    
    dec_m_nnz = dec_m.nnz
    dec_m_csrValA = cuda.to_device(dec_m.data.astype(calc_precision))
    dec_m_csrRowPtrA = cuda.to_device(dec_m.indptr)
    dec_m_csrColIndA = cuda.to_device(dec_m.indices)
    
    cu_curr_phi = cuda.to_device(phi.astype(calc_precision))
    cu_delta_phi = cuda.device_array(phi.shape, dtype=calc_precision)

    descr = cusp.matdescr()
    descr.indexbase = cusparse.CUSPARSE_INDEX_BASE_ZERO
    
    for step in xrange(nsteps):
        if prog_bar and (step % 5 == 0):
            prog_bar.update(step)
        cusp.csrmv(trans='T', m=m, n=n, nnz=int_m_nnz,
                   descr=descr,
                   alpha=float32(1.0),
                   csrVal=int_m_csrValA,
                   csrRowPtr=int_m_csrRowPtrA,
                   csrColInd=int_m_csrColIndA,
                   x=cu_curr_phi, beta=float32(0.0), y=cu_delta_phi)
        cusp.csrmv(trans='T', m=m, n=n, nnz=dec_m_nnz,
                   descr=descr,
                   alpha=float32(rho_inv[step]),
                   csrVal=dec_m_csrValA,
                   csrRowPtr=dec_m_csrRowPtrA,
                   csrColInd=dec_m_csrColIndA,
                   x=cu_curr_phi, beta=float32(1.0), y=cu_delta_phi)
        cubl.axpy(alpha=float32(dX[step]), x=cu_delta_phi, y=cu_curr_phi)

    return cu_curr_phi.copy_to_host(), []

def kern_MKL_sparse(nsteps, dX, rho_inv, int_m, dec_m,
                    phi, grid_idcs, prog_bar=None):
    """`Intel MKL sparse BLAS <https://software.intel.com/en-us/articles/intel-mkl-sparse-blas-overview?language=en>`_ 
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
      prog_bar (object,optional): handle to :class:`ProgressBar` object
    Returns:
      numpy.array: state vector :math:`\\Phi(X_{nsteps})` after integration
    """
    
    from ctypes import cdll, c_int, c_double, c_char, POINTER, byref
    try:
        mkl = cdll.LoadLibrary(config['MKL_path'])
    except OSError:
        raise Exception("kern_MKL_sparse(): MKL runtime library not " + 
                        "found. Please check path.")

    # sparse CSR-matrix x dense vector 
    gemv = mkl.mkl_dcsrmv
    # dense vector + dense vector
    axpy = mkl.cblas_daxpy

    # Set number of threads to sufficiently small number, since 
    # matrix-vector multiplication is memory bandwidth limited
    mkl.mkl_set_num_threads(byref(c_int(config['MKL_threads'])))

    # Prepare CTYPES pointers for MKL sparse CSR BLAS
    int_m_data = int_m.data.ctypes.data_as(POINTER(c_double))
    int_m_ci = int_m.indices.ctypes.data_as(POINTER(c_int))
    int_m_pb = int_m.indptr[:-1].ctypes.data_as(POINTER(c_int))
    int_m_pe = int_m.indptr[1:].ctypes.data_as(POINTER(c_int))

    dec_m_data = dec_m.data.ctypes.data_as(POINTER(c_double))
    dec_m_ci = dec_m.indices.ctypes.data_as(POINTER(c_int))
    dec_m_pb = dec_m.indptr[:-1].ctypes.data_as(POINTER(c_int))
    dec_m_pe = dec_m.indptr[1:].ctypes.data_as(POINTER(c_int))

    npphi = np.copy(phi)
    phi = npphi.ctypes.data_as(POINTER(c_double))
    npdelta_phi = np.zeros_like(npphi, dtype='double')
    delta_phi = npdelta_phi.ctypes.data_as(POINTER(c_double))

    trans = c_char('n')
    npmatd = np.chararray(6)
    npmatd[0] = 'G'
    npmatd[3] = 'C'
    matdsc = npmatd.ctypes.data_as(POINTER(c_char))
    m = c_int(int_m.shape[0])
    cdzero = c_double(0.)
    cdone = c_double(1.)
    cione = c_int(1)
    
    grid_step = 0
    grid_sol = []
    for step in xrange(nsteps):
        if prog_bar:
            prog_bar.update(step)
            
        # delta_phi = int_m.dot(phi)
        gemv(byref(trans), byref(m), byref(m),
             byref(cdone), matdsc,
             int_m_data, int_m_ci, int_m_pb, int_m_pe,
             phi, byref(cdzero), delta_phi)
        # delta_phi = rho_inv * dec_m.dot(phi) + delta_phi
        gemv(byref(trans), byref(m), byref(m),
             byref(c_double(rho_inv[step])), matdsc,
             dec_m_data, dec_m_ci, dec_m_pb, dec_m_pe,
             phi, byref(cdone), delta_phi)
        # phi = delta_phi * dX + phi
        axpy(m, c_double(dX[step]),
             delta_phi, cione, phi, cione)
        
        if (grid_idcs and grid_step < len(grid_idcs) 
            and grid_idcs[grid_step] == step):
            grid_sol.append(np.copy(npphi))
            grid_step += 1
            

    # Reset number of threads for MKL
    mkl.mkl_set_num_threads(byref(c_int(4)))
    return npphi, grid_sol
