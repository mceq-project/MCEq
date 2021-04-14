import mceq_config as config

if config.kernel_config.lower() == "cuda":
    import cupy
    import cupyx
    zeros = cupy.zeros
    ones = cupy.ones
    eye = cupy.eye
    diag = cupy.diag
    csr_matrix = cupyx.scipy.sparse.csr_matrix
    linalg = cupy.linalg
    asarray = cupy.asarray
    floatlen = config.floatlen
else:
    import numpy as np
    from scipy.sparse import csr_matrix
    zeros = np.zeros
    ones = np.ones
    eye = np.eye
    diag = np.diag
    csr_matrix = csr_matrix
    linalg = np.linalg
    asarray = np.asarray
    floatlen = config.floatlen