from __future__ import print_function

import mceq_config as config
from MCEq.core import MCEqRun
import crflux.models as pm
import numpy as np

config.debug_level = 1
config.kernel_config = 'numpy'
config.cuda_gpu_id = 0
config.set_mkl_threads(4)

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
                        [59.08402527783906, 21111.172321878876,
                            10057901.920254719, 346916423.1257473],
                        [20.007070780314464, 5498.879277541484, 2266361.6016466515, 73347318.16002306]], rtol=1e-2))


def test_corsika_atm():

    from MCEq.geometry.density_profiles import CorsikaAtmosphere

    # Depth at surface and density at X=100 g/cm2
    cka_surf_100 = [
        (1036.099233683902, 0.00015623258808300557),
        (1033.8094962133184, 0.00015782685585891685),
        (1055.861981113731, 0.00016209949387937668),
        (1037.9745770942316, 0.00015591979560837204),
        (1011.4568036341923, 0.00014626903051217024),
        (1019.974568696789, 0.0001464549375212421),
        (1019.9764946890782, 0.0001685608228906579)
    ]
    for iatm, (loc, season) in enumerate([
        ("USStd", None),
        ("BK_USStd", None),
        ("Karlsruhe", None),
        ("ANTARES/KM3NeT-ORCA", None),
        ('SouthPole', 'December'),
        ('PL_SouthPole', 'January'),
        ('PL_SouthPole', 'August'),
    ]):

        cka_obj = CorsikaAtmosphere(loc, season)
        assert np.allclose([cka_obj.max_X, 1. /
                            cka_obj.r_X2rho(100.)], cka_surf_100[iatm])


def test_msis_atm():

    from MCEq.geometry.density_profiles import MSIS00Atmosphere
    msis_surf_100 = [
        (1022.6914983678925, 0.00014380042112573175), (1041.2180457811605, 0.00016046129606232836),
        (1044.6608866969684, 0.00016063221634835724), (1046.427667371285, 0.00016041531186210874),
        (1048.6505423154006, 0.00016107650347480857), (1050.6431802896034, 0.00016342084740033518),
        (1050.2145039327452, 0.00016375664772178006), (1033.3640270683418, 0.00015614485659072835),
        (1045.785578319159, 0.00015970449150213374), (1019.9475650272982, 0.000153212909250962),
        (1020.3640351872195, 0.00015221038616604717), (1047.964376368261, 0.00016218804771381842)
    ]
    for iatm, (loc, season) in enumerate([
        ('SouthPole', "January"),
        ('Karlsruhe', "January"),
        ('Geneva', "January"),
        ('Tokyo', "January"),
        ('SanGrasso', "January"),
        ('TelAviv', "January"),
        ('KSC', "January"),
        ('SoudanMine', "January"),
        ('Tsukuba', "January"),
        ('LynnLake', "January"),
        ('PeaceRiver', "January"),
        ('FtSumner', "January")
    ]):

        msis_obj = MSIS00Atmosphere(loc, season)
        assert np.allclose([msis_obj.max_X, 1. /
                            msis_obj.r_X2rho(100.)], msis_surf_100[iatm])
