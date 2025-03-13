import src.mceq_config as config
import MCEq.version

__version__ = MCEq.version.__version__


def set_backend(kernel_config):
    global zeros, ones, eye, diag, csr_matrix, linalg, asarray
    assert kernel_config in ["cuda", "numpy", "mkl", "accelerate"]

    import numpy as np

    if kernel_config == "cuda":
        import cupy
        import cupyx

        zeros = cupy.zeros
        ones = cupy.ones
        eye = cupy.eye
        diag = cupy.diag
        csr_matrix = cupyx.scipy.sparse.csr_matrix
        linalg = cupy.linalg
        asarray = cupy.asarray
        config.floatlen = np.float32
        config.kernel_config = "cuda"
    else:
        from scipy.sparse import csr_matrix

        zeros = np.zeros
        ones = np.ones
        eye = np.eye
        diag = np.diag
        csr_matrix = csr_matrix
        linalg = np.linalg
        asarray = np.asarray
        config.floatlen = np.float64
        config.kernel_config = kernel_config

    if config.debug_level >= 3:
        print("MCEq::__init__: Using {0} backend".format(config.kernel_config))


def set_mkl_threads(nthreads):
    from ctypes import cdll, c_int, byref

    config.mkl = cdll.LoadLibrary(config.mkl_path)
    # Set number of threads
    config.mkl_threads = nthreads
    config.mkl.mkl_set_num_threads(byref(c_int(nthreads)))
    if config.debug_level >= 5:
        print("MCEq::__init__: MKL threads limited to {0}".format(nthreads))


# CUDA is usually fastest, then MKL. Fallback to numpy.
if config.kernel_config == "auto":
    if config.has_cuda:
        config.kernel_config = "cuda"
    elif config.has_accelerate:
        config.kernel_config = "accelerate"
    elif config.has_mkl:
        config.kernel_config = "mkl"
    else:
        config.kernel_config = "numpy"
else:
    if config.kernel_config.lower() == "cuda" and not config.has_cuda:
        raise Exception("CUDA unavailable. Make sure cupy is installed.")
    elif config.kernel_config.lower() == "mkl" and not config.has_mkl:
        raise Exception("MKL unavailable. Make sure Intel MKL is installed.")
    elif config.kernel_config.lower() == "accelerate" and not config.has_accelerate:
        raise Exception("Apple Accelerate only available on Mac platforms.")

if config.debug_level >= 2 and config.kernel_config == "auto":
    print("MCEq::__init__: Auto-detected {0} solver.".format(config.kernel_config))

if config.has_mkl:
    set_mkl_threads(config.mkl_threads)

set_backend(config.kernel_config.lower())
