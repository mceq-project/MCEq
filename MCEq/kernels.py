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
               phi, grid_idcs, 
               mu_egrid=None, mu_dEdX=None, mu_lidx_nsp=None,
               prog_bar=None):
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
    # Experimental code for Xeon Phi testing
    # if config['MKL_enable_mic']:
    #     from ctypes import cdll

    #     try:
    #         mkl = cdll.LoadLibrary(config['MKL_path'])
    #     except OSError:
    #         raise Exception("kern_MKL_sparse(): MKL runtime library not " + 
    #                         "found. Please check path.")

    #     print ("kern_MKL_sparse(): Automatic Xeon Phi offloading activated.")
    #     mkl.mkl_mic_enable()
    #     mkl.mkl_mic_set_offload_report()
    
    #     config['MKL_enable_mic'] = False

    grid_sol = []
    grid_step = 0
    
    imc = int_m
    dmc = dec_m
    dxc = dX
    ric = rho_inv
    phc = phi

    enmuloss = config['enable_muon_energy_loss']
    de = mu_egrid.size
    muloss_min_step = config['muon_energy_loss_min_step']
    lidx, nmuspec =  mu_lidx_nsp
    # Accumulate at least a few g/cm2 for energy loss steps
    # to avoid numerical errors
    dXaccum = 0.

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
        
        dXaccum += dxc[step]
        
        if (enmuloss and 
            (dXaccum > muloss_min_step or step == nsteps - 1)):
            for nsp in xrange(nmuspec):
                phc[lidx + de*nsp: lidx + de*(nsp+1)] = np.interp(
                    mu_egrid, mu_egrid + mu_dEdX*dXaccum, 
                    phc[lidx + de*nsp:lidx + de*(nsp+1)])

            dXaccum = 0.
        
        if (grid_idcs and grid_step < len(grid_idcs) 
            and grid_idcs[grid_step] == step):
            grid_sol.append(np.copy(phc))
            grid_step += 1



    return phc, grid_sol


def kern_CUDA_dense(nsteps, dX, rho_inv, int_m, dec_m,
                    phi, grid_idcs, 
                    mu_egrid=None, mu_dEdX=None, mu_lidx_nsp=None,
                    prog_bar=None):
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
    
    fl_pr = None
    if config['FP_precision'] == 32:
        fl_pr = np.float32
    elif config['FP_precision'] == 64:
        fl_pr = np.float64
    else:
        raise Exception("kern_CUDA_dense(): Unknown precision specified.")    
    
    # if config['enable_muon_energyloss']:
    #     raise NotImplementedError('kern_CUDA_dense(): ' + 
    #         'Energy loss not imlemented for this solver.')

    if config['enable_muon_energy_loss']:
        raise NotImplementedError('kern_CUDA_dense(): ' + 
            'Energy loss not imlemented for this solver.')

    #=======================================================================
    # Setup GPU stuff and upload data to it
    #=======================================================================
    try:
        from accelerate.cuda.blas import Blas
        from accelerate.cuda import cuda
    except ImportError:
        raise Exception("kern_CUDA_dense(): Numbapro CUDA libaries not " + 
                        "installed.\nCan not use GPU.")
    cubl = Blas()
    m, n = int_m.shape
    stream = cuda.stream()
    cu_int_m = cuda.to_device(int_m.astype(fl_pr), stream)
    cu_dec_m = cuda.to_device(dec_m.astype(fl_pr), stream)
    cu_curr_phi = cuda.to_device(phi.astype(fl_pr), stream)
    cu_delta_phi = cuda.device_array(phi.shape, dtype=fl_pr)
    for step in xrange(nsteps):
        if prog_bar:
            prog_bar.update(step)
        cubl.gemv(trans='N', m=m, n=n, alpha=fl_pr(1.0), A=cu_int_m,
            x=cu_curr_phi, beta=fl_pr(0.0), y=cu_delta_phi)
        cubl.gemv(trans='N', m=m, n=n, alpha=fl_pr(rho_inv[step]),
            A=cu_dec_m, x=cu_curr_phi, beta=fl_pr(1.0), y=cu_delta_phi)
        cubl.axpy(alpha=fl_pr(dX[step]), x=cu_delta_phi, y=cu_curr_phi)

    return cu_curr_phi.copy_to_host(), []


class CUDASparseContext(object):
    def __init__(self, int_m, dec_m, device_id=0):

        if config['FP_precision'] == 32:
            self.fl_pr = np.float32
        elif config['FP_precision'] == 64:
            self.fl_pr = np.float64
        else:
            raise Exception("CUDASparseContext(): Unknown precision specified.")    
        #=======================================================================
        # Setup GPU stuff and upload data to it
        #=======================================================================
        try:
            from accelerate.cuda.blas import Blas
            import accelerate.cuda.sparse as cusparse
            from accelerate.cuda import cuda
        except ImportError:
            raise Exception("kern_CUDA_sparse(): Numbapro CUDA libaries not " + 
                            "installed.\nCan not use GPU.")

        cuda.select_device(0)
        self.cuda = cuda
        self.cusp = cusparse.Sparse()
        self.cubl = Blas()
        self.set_matrices(int_m, dec_m)
        
    def set_matrices(self, int_m, dec_m):
        import accelerate.cuda.sparse as cusparse
        from accelerate.cuda import cuda    
        
        self.m, self.n = int_m.shape
        self.int_m_nnz = int_m.nnz
        self.int_m_csrValA = cuda.to_device(int_m.data.astype(self.fl_pr))
        self.int_m_csrRowPtrA = cuda.to_device(int_m.indptr)
        self.int_m_csrColIndA = cuda.to_device(int_m.indices)
        
        self.dec_m_nnz = dec_m.nnz
        self.dec_m_csrValA = cuda.to_device(dec_m.data.astype(self.fl_pr))
        self.dec_m_csrRowPtrA = cuda.to_device(dec_m.indptr)
        self.dec_m_csrColIndA = cuda.to_device(dec_m.indices)
        
        self.descr = self.cusp.matdescr()
        self.descr.indexbase = cusparse.CUSPARSE_INDEX_BASE_ZERO
    
    def set_phi(self, phi):
        self.cu_curr_phi = self.cuda.to_device(phi.astype(self.fl_pr))
        self.cu_delta_phi = self.cuda.device_array_like(phi.astype(self.fl_pr))

def kern_CUDA_sparse(nsteps, dX, rho_inv, context, phi, grid_idcs, 
                      mu_egrid=None, mu_dEdX=None, mu_lidx_nsp=None,
                      prog_bar=None):
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
      prog_bar (object,optional): handle to :class:`ProgressBar` object
    Returns:
      numpy.array: state vector :math:`\\Phi(X_{nsteps})` after integration
    """

    c = context
    c.set_phi(phi)

    enmuloss = config['enable_muon_energy_loss']
    de = mu_egrid.size
    mu_egrid = mu_egrid.astype(c.fl_pr)
    muloss_min_step = config['muon_energy_loss_min_step']
    lidx, nmuspec =  mu_lidx_nsp

    # Accumulate at least a few g/cm2 for energy loss steps
    # to avoid numerical errors
    dXaccum = 0.

    grid_step = 0
    grid_sol = []
    
    for step in xrange(nsteps):
        if prog_bar and (step % 5 == 0):
            prog_bar.update(step)
        c.cusp.csrmv(trans='N', m=c.m, n=c.n, nnz=c.int_m_nnz,
                   descr=c.descr,
                   alpha=c.fl_pr(1.0),
                   csrVal=c.int_m_csrValA,
                   csrRowPtr=c.int_m_csrRowPtrA,
                   csrColInd=c.int_m_csrColIndA,
                   x=c.cu_curr_phi, beta=c.fl_pr(0.0), y=c.cu_delta_phi)
        # print np.sum(cu_curr_phi.copy_to_host())
        c.cusp.csrmv(trans='N', m=c.m, n=c.n, nnz=c.dec_m_nnz,
                   descr=c.descr,
                   alpha=c.fl_pr(rho_inv[step]),
                   csrVal=c.dec_m_csrValA,
                   csrRowPtr=c.dec_m_csrRowPtrA,
                   csrColInd=c.dec_m_csrColIndA,
                   x=c.cu_curr_phi, beta=c.fl_pr(1.0), y=c.cu_delta_phi)
        c.cubl.axpy(alpha=c.fl_pr(dX[step]), x=c.cu_delta_phi, y=c.cu_curr_phi)
        
        dXaccum += dX[step]
        
        if (enmuloss and 
            (dXaccum > muloss_min_step or step == nsteps - 1)):
            # Download current solution vector to host
            phc = c.cu_curr_phi.copy_to_host()
            for nsp in xrange(nmuspec):
                phc[lidx + de*nsp: lidx + de*(nsp+1)] = np.interp(
                    mu_egrid, mu_egrid + mu_dEdX*dXaccum, 
                    phc[lidx + de*nsp:lidx + de*(nsp+1)])
            # Upload changed vector back..
            c.cu_curr_phi = c.cuda.to_device(phc)
            dXaccum = 0.

        if (grid_idcs and grid_step < len(grid_idcs) 
            and grid_idcs[grid_step] == step):
            grid_sol.append(c.cu_curr_phi.copy_to_host())
            grid_step += 1

    return c.cu_curr_phi.copy_to_host(), grid_sol

def kern_MKL_sparse(nsteps, dX, rho_inv, int_m, dec_m,
                    phi, grid_idcs, 
                    mu_egrid=None, mu_dEdX=None, mu_lidx_nsp=None,
                    prog_bar=None):
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
    
    from ctypes import cdll, c_int, c_char, POINTER, byref

    try:
        mkl = cdll.LoadLibrary(config['MKL_path'])
    except OSError:
        raise Exception("kern_MKL_sparse(): MKL runtime library not " + 
                        "found. Please check path.")
    
    # if config['MKL_enable_mic']:
    #     print ("kern_MKL_sparse(): Automatic Xeon Phi offloading activated.")
    #     mkl.mkl_mic_enable()
    #     config['MKL_enable_mic'] = False
    
    gemv = None
    axpy = None
    np_fl = None
    if config['FP_precision'] == 32:
        from ctypes import c_float as fl_pr
        # sparse CSR-matrix x dense vector 
        gemv = mkl.mkl_scsrmv
        # dense vector + dense vector
        axpy = mkl.cblas_saxpy
        np_fl = np.float32
    elif config['FP_precision'] == 64:
        from ctypes import c_double as fl_pr
        # sparse CSR-matrix x dense vector 
        gemv = mkl.mkl_dcsrmv
        # dense vector + dense vector
        axpy = mkl.cblas_daxpy
        np_fl = np.float64
    else:
        raise Exception("kern_MKL_sparse(): Unknown precision specified.")
        

    # Set number of threads
    mkl.mkl_set_num_threads(byref(c_int(config['MKL_threads'])))

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
    npdelta_phi = np.zeros_like(npphi, dtype=np_fl)
    delta_phi = npdelta_phi.ctypes.data_as(POINTER(fl_pr))

    trans = c_char('n')
    npmatd = np.chararray(6)
    npmatd[0] = 'G'
    npmatd[3] = 'C'
    matdsc = npmatd.ctypes.data_as(POINTER(c_char))
    m = c_int(int_m.shape[0])
    cdzero = fl_pr(0.)
    cdone = fl_pr(1.)
    cione = c_int(1)
    
    enmuloss = config['enable_muon_energy_loss']
    de = mu_egrid.size
    muloss_min_step = config['muon_energy_loss_min_step']
    lidx, nmuspec =  mu_lidx_nsp
    # Accumulate at least a few g/cm2 for energy loss steps
    # to avoid numerical errors
    dXaccum = 0.
    
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
             byref(fl_pr(rho_inv[step])), matdsc,
             dec_m_data, dec_m_ci, dec_m_pb, dec_m_pe,
             phi, byref(cdone), delta_phi)
        # phi = delta_phi * dX + phi
        axpy(m, fl_pr(dX[step]),
             delta_phi, cione, phi, cione)
        
        dXaccum += dX[step]
        
        if (enmuloss and 
            (dXaccum > muloss_min_step or step == nsteps - 1)):
            for nsp in xrange(nmuspec):
                npphi[lidx + de*nsp: lidx + de*(nsp+1)] = np.interp(
                    mu_egrid, mu_egrid + mu_dEdX*dXaccum, 
                    npphi[lidx + de*nsp:lidx + de*(nsp+1)])

            dXaccum = 0.


        if (grid_idcs and grid_step < len(grid_idcs) 
            and grid_idcs[grid_step] == step):
            grid_sol.append(np.copy(npphi))
            grid_step += 1
            

    # Reset number of threads for MKL
    # mkl.mkl_set_num_threads(byref(c_int(4)))
    return npphi, grid_sol
