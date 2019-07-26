from __future__ import print_function


def test_config_and_file_download():
    import mceq_config as config
    import os
    # Import of config triggers data download
    assert config.mceq_db_fname in os.listdir(config.data_dir)


def test_some_angles():
    import mceq_config as config
    from MCEq.core import MCEqRun
    import crflux.models as pm
    import numpy as np

    config.debug_level = 5
    config.kernel_config = 'numpy'
    config.cuda_gpu_id = 0
    config.mkl_threads = 2

    mceq = MCEqRun(
        interaction_model='SIBYLL23C',
        theta_deg=0.,
        primary_model=(pm.HillasGaisser2012, 'H3a'))

    nmu = []
    for theta in [0., 30., 60., 90]:
        mceq.set_theta_deg(theta)
        mceq.solve()
        nmu.append(
            np.sum(
                mceq.get_solution('mu+', 0, integrate=True) +
                mceq.get_solution('mu-', 0, integrate=True)))
    print(nmu)
    assert np.allclose(nmu, [
        59787.31805017808, 60908.05990627792, 66117.91267025097,
        69664.26521920023
    ])
