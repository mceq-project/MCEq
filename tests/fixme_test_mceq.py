from __future__ import print_function

import MCEq.config as config


import crflux.models as pm
import numpy as np

import pytest
import sys


config.debug_level = 1
config.kernel_config = "numpy"

import MCEq
import MCEq.core

if config.has_mkl:
    MCEq.set_mkl_threads(2)

config.e_min = 1
config.e_max = 5e7

# MCEq.set_backend("numpy")


def format_8_digits(a_list):
    return ["%.8e" % member for member in a_list]


if sys.platform.startswith("win") and sys.maxsize <= 2**32:
    pytest.skip("Skip model test on 32-bit Windows.", allow_module_level=True)


@pytest.fixture(scope="module")
def mceq_qgs():
    return MCEq.core.MCEqRun(
        interaction_model="QGSJETII04",
        theta_deg=0.0,
        primary_model=(pm.HillasGaisser2012, "H3a"),
    )


def test_config_and_file_download():
    import MCEq.config as config
    import os

    # Import of config triggers data download
    assert config.mceq_db_fname in os.listdir(config.data_dir)


def test_some_angles(mceq_qgs):
    nmu = []
    for theta in [0.0, 30.0, 60.0, 80]:
        mceq_qgs.set_theta_deg(theta)
        mceq_qgs.solve()
        nmu.append(
            np.sum(
                mceq_qgs.get_solution("mu+", mag=0, integrate=True)
                + mceq_qgs.get_solution("mu-", mag=0, integrate=True)
            )
        )
    assert format_8_digits(nmu), [
        "5.62504370e-03",
        "4.20479234e-03",
        "1.36630552e-03",
        "8.20255259e-06",
    ]


def test_switch_interaction_models():
    mceq = MCEq.core.MCEqRun(
        interaction_model="DPMJETIII191",
        theta_deg=0.0,
        primary_model=(pm.HillasGaisser2012, "H3a"),
    )
    mlist = [
        "DPMJETIII191",
        "DPMJETIII306",
        "QGSJET01D",
        "QGSJETII03",
        "QGSJETII04",
        "SIBYLL21",
        "SIBYLL23C",
        "SIBYLL23D",
    ]
    count_part = []
    for m in mlist:
        mceq.set_interaction_model(m)
        count_part.append(len(mceq._particle_list))
    assert count_part == [68, 68, 62, 48, 48, 52, 66, 66]


def test_single_primary(mceq_qgs):
    energies = [1e3, 1e6]#, 1e9, 5e10]
    nmu, nnumu, nnue = [], [], []
    mceq_qgs.set_interaction_model("SIBYLL23D")
    mceq_qgs.set_theta_deg(0.0)
    for e in energies:
        mceq_qgs.set_single_primary_particle(E=e, pdg_id=2212)
        mceq_qgs.solve()
        nmu.append(
            np.sum(
                mceq_qgs.get_solution("mu+", mag=0, integrate=True)
                + mceq_qgs.get_solution("mu-", mag=0, integrate=True)
            )
        )
        nnumu.append(
            np.sum(
                mceq_qgs.get_solution("numu", mag=0, integrate=True)
                + mceq_qgs.get_solution("antinumu", mag=0, integrate=True)
            )
        )
        nnue.append(
            np.sum(
                mceq_qgs.get_solution("nue", mag=0, integrate=True)
                + mceq_qgs.get_solution("antinue", mag=0, integrate=True)
            )
        )

    assert format_8_digits(nmu) == [
        "2.03134720e+01",
        "1.20365838e+04",
        # "7.09254150e+06",
        # "2.63982133e+08",
    ]
    assert format_8_digits(nnumu) == [
        "6.80367347e+01",
        "2.53158948e+04",
        # "1.20884925e+07",
        # "4.14935240e+08",
    ]
    assert format_8_digits(nnue) == [
        "2.36908717e+01",
        "6.91213253e+03",
        "2.87396649e+06",
        "9.27683105e+07",
    ]
