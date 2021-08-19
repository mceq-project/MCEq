import time
import cupy as cp
import numpy as np
import dask
import dask.array as da

def many_svd_np_vs_cp():
    device = cp.cuda.Device()

    N = 32      # number of desired SVDs, grouped.
    M = 2048  # size of each matrix, for SVD (MxM)
    A = np.asarray(np.random.randn(N, M, M), dtype=np.float32)
    
    # ----- Prime Pump, to eliminate CUDA overhead in timings. ----- 
    A_gpu = cp.asarray(A)
    for i in range(16):  
        sg = cp.linalg.svd(A_gpu[0], compute_uv=False)
    time.sleep(0.25)  # to separate this, in nvvp

    # ----- Grouped SVDs in numpy ----- 
    tm = time.time()
    s_npall = np.linalg.svd(A, compute_uv=False)  # 256 x 16
    elaps = time.time() - tm
    print('%20s: elaps=%f' % ('Numpy', elaps))

    # ----- Cupy-Loop: grouped SVDs in cupy ----- 
    sg_all = cp.asarray([])
    tm = time.time()
    for i in range(A_gpu.shape[0]):
        sg = cp.linalg.svd(A_gpu[i], compute_uv=False)
        sg_all = cp.concatenate((sg_all, sg), axis=0) # N*16 = 4096, but that's OK
    s_cpall = cp.asnumpy(sg_all)
    elaps = time.time() - tm
    print('%20s: elaps=%f' % ('Cupy-Loop', elaps))
    time.sleep(0.20)

    # ----- Cupy-ListComp: is List Comprehension Faster? -----
    sg_all = cp.asarray([])
    tm = time.time()
    sg_all = [cp.asnumpy(cp.linalg.svd(A_gpu[i], compute_uv=False)) for i in range(A_gpu.shape[0])]
    #s_cpall = cp.asnumpy(sg_all)
    elaps = time.time() - tm
    print('%20s: elaps=%f' % ('Cupy-ListComp', elaps))
    time.sleep(0.20)

    # ----- Cupy-Dask-Delayed: try using Dask.Delayed for parallelism/concurrency -----
    # TODO: not currently trying to retrieve the results, with this example.
    tm = time.time()
    tasks = [ dask.delayed(cp.linalg.svd)(A_gpu[i], compute_uv=False) for i in range(A_gpu.shape[0])]
    tasks_list = dask.delayed( list(tasks) )
    res = dask.compute(tasks_list)  # Does return a list of 256 x 16
    device.synchronize()
    elaps = time.time() - tm
    print('%20s: elaps=%f' % ('Cupy-Dask-Delayed', elaps))
    time.sleep(0.20)

    # ----- Cupy-Streams: try cupy streams for paralellism/concurrency -----
    # TODO: not currently trying to retrieve the results, with this example.
    device = cp.cuda.Device()
    map_streams = [cp.cuda.stream.Stream() for i in range(N)]
    tm = time.time()  # BUG: was start_time = time.time()
    for i, stream in enumerate(map_streams):
        with stream:
            sg = cp.linalg.svd(A_gpu[i], compute_uv=False)
            # This is a little worse:
            # C_gpu = cp.asarray(np.random.randn(M, M), dtype=np.float32) 
            # sg = cp.linalg.svd(C_gpu, compute_uv=False)
    device.synchronize()
    elaps = time.time() - tm
    print('%20s: elaps=%f' % ('Cupy-Streams', elaps))

if __name__ == "__main__":
    many_svd_np_vs_cp()