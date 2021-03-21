import mceq_config as config

if config.kernel_config.lower() == "cuda":
    import cupy
    zeros = cupy.zeros
    ones = cupy.ones
    eye = cupy.eye
    diag = cupy.diag
    csr_matrix = cupy.sparse.csr_matrix
    linalg = cupy.linalg
    asarray = cupy.asarray
    floatlen = config.floatlen
else:
    import numpy as np
    import scipy as sp
    zeros = np.zeros
    ones = np.ones
    eye = np.eye
    diag = np.diag
    csr_matrix = sp.sparse.csr_matrix
    linalg = np.linalg
    asarray = np.asarray
    floatlen = config.floatlen