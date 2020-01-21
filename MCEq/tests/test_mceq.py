from __future__ import print_function

import mceq_config as config
from MCEq.core import MCEqRun
import crflux.models as pm
import numpy as np

config.debug_level = 1
config.kernel_config = 'numpy'
config.cuda_gpu_id = 0
config.mkl_threads = 4

mceq = MCEqRun(
    interaction_model='SIBYLL23C',
    theta_deg=0.,
    primary_model=(pm.HillasGaisser2012, 'H3a'))


def test_config_and_file_download():
    import mceq_config as config
    import os
    # Import of config triggers data download
    assert config.mceq_db_fname in os.listdir(config.data_dir)


def test_some_angles():

    nmu = []
    for theta in [0., 30., 60., 90]:
        mceq.set_theta_deg(theta)
        mceq.solve()
        nmu.append(
            np.sum(
                mceq.get_solution('mu+', mag=0, integrate=True) +
                mceq.get_solution('mu-', mag=0, integrate=True)))
    assert np.allclose(nmu,
                       [0.004801787364145619,
                        0.0036615779742498415,
                        0.0012506912333444566,
                        8.179093198172332e-06],
                       rtol=1e-4)


def test_switch_interaction_models():
    mlist = [
        'DPMJETIII191',
        'DPMJETIII306',
        'QGSJET01C',
        'QGSJETII03',
        'QGSJETII04',
        'SIBYLL21',
        'SIBYLL23',
        'SIBYLL23C',
        'SIBYLL23CPP']
    count_part = []
    for m in mlist:
        mceq.set_interaction_model(m)
        count_part.append(len(mceq._particle_list))
    assert(count_part == [64, 64, 58, 44, 44, 48, 62, 62, 62])
    


def test_single_primary():
    energies = [1e3, 1e6, 1e9, 5e10]
    nmu, nnumu, nnue = [], [], []
    mceq.set_interaction_model('SIBYLL23C')
    mceq.set_theta_deg(0.)
    for e in energies:
        mceq.set_single_primary_particle(E=e, pdg_id=2212)
        mceq.solve()
        nmu.append(
            np.sum(
                mceq.get_solution('mu+', mag=0, integrate=True) +
                mceq.get_solution('mu-', mag=0, integrate=True)))
        nnumu.append(
            np.sum(
                mceq.get_solution('numu', mag=0, integrate=True) +
                mceq.get_solution('antinumu', mag=0, integrate=True)))
        nnue.append(
            np.sum(
                mceq.get_solution('nue', mag=0, integrate=True) +
                mceq.get_solution('antinue', mag=0, integrate=True)))

    assert(np.allclose([nmu, nnumu, nnue], 
        [[18.567350622641847, 10486.232899632832, 6094178.175780058, 226728040.8471465],
         [59.08402527783906, 21111.172321878876, 10057901.920254719, 346916423.1257473],
         [20.007070780314464, 5498.879277541484, 2266361.6016466515, 73347318.16002306]], rtol=1e-2))
