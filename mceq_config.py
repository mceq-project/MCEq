

import sys
import platform
import os.path as path
base = path.dirname(path.abspath(__file__))
sys.path.append(base)
sys.path.append(base+"/CRFluxModels")
sys.path.append(base+"/ParticleDataTool")
sys.path.append(base+"/Python-NRLMSISE-00")
sys.path.append(base+"/c-NRLMSISE-00")
#commented out since projects are far from beeing ready for prime time.
#sys.path.append("../Fedynitch2012")
#sys.path.append("../AnalyticalApproximation")

#detrmine shared library extension and MKL path
lib_ext = None
mkl_default = path.join(sys.prefix, 'lib', 'libmkl_rt')

if platform.platform().find('Linux') != -1:
    lib_ext = '.so'
elif platform.platform().find('Darwin') != -1:
    lib_ext = '.dylib'
else:
    #Windows case
    mkl_default = path.join(sys.prefix, 'pkgs', 'mkl-11.3.3-1',
                            'Library', 'bin', 'mkl_rt')
    lib_ext = '.dll'

config = {

    # Debug flag for verbose printing, 0 = minimum
    "debug_level": 1,

    # Use progress_bars
    "prog_bar": False,


    #=========================================================================
    # Paths and library locations
    #=========================================================================

    # Directory where the data files for the calculation are stored
    "data_dir": base+'/data',

    # File name of particle decay spectra
    "decay_fname": "decays_v1.ppd",

    # File name of the cross-sections tables
    "cs_fname": "crosssections.ppd",

    # File name of for energy losses
    "mu_eloss_fname": "dEdX_mu_air.ppl",

    # File where to cache interpolating splines of the atmosphere module
    'atm_cache_file':'atm_cache.ppd',

    # full path to libmkl_rt.[so/dylib] (only if kernel=='MKL')
    "MKL_path": mkl_default + lib_ext,
    #=========================================================================
    # Atmosphere and geometry settings
    #=========================================================================

    # Use file for caching calculated atmospheric rho(X) splines
    "use_atm_cache": True,

    # Atmospheric model in the format: (model, (arguments))
    "density_model": ('CORSIKA', ('BK_USStd', None)),
    # "density_model": ('MSIS00_IC',('SouthPole','January')),
    # "density_model": ('GeneralizedTarget', None),


    # Version of NRLMSISE-00 python library (ctypes, native)
    "msis_python": "ctypes",

    # List of particles which decay products will be scored
    # in the 'obs_' category
    "obs_ids": None,  # Example ["eta", "eta*", "etaC", "omega", "phi"],

    # Average mass of target (for cross section calculations)
    # Change parameter only in combination with interaction model setting.
    # By default all inclusive cross sections are calculated for air targets
    # expect those with '_pp' suffix.
    "A_target" : 14.5, # <A> = 14.5 for air

    #parameters for EarthGeometry
    "r_E": 6391.e3,  # Earth radius in m
    "h_obs": 0.,  # observation level in m
    "h_atm": 112.8e3,  # top of the atmosphere in m

    #Default parameters for GeneralizedTarget
    "len_target": 1000., # Total length of the target [m]
    "env_density": 0.001225, # density of default material in g/cm^3
    "env_name": "air",

    #===========================================================================
    # Parameters of numerical integration
    #===========================================================================

    # Selection of integrator (euler/odepack)
    "integrator": "euler",

    # euler kernel implementation (numpy/MKL/CUDA).
    # With serious nVidia GPUs CUDA a few times faster than MKL
    "kernel_config": "MKL",

    #parameters for the odepack integrator. More details at
    #http://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html#scipy.integrate.ode
    "ode_params": {'name':'vode',
                   'method':'adams',
                   'nsteps':10000,
                   'max_step':10.0},

    # Use sparse linear algebra (recommended!)
    "use_sparse": True,

    #Number of MKL threads (for sparse matrix multiplication the performance
    #advantage from using more than 1 thread is limited by memory bandwidth)
    "MKL_threads": 24,

    # Floating point precision: 32-bit results in speed-up with CUDA.
    # Do not use with MKL, it can result in false results and slow down.
    "FP_precision": 64,

    #=========================================================================
    # Advanced settings
    #=========================================================================

    # Ratio of decay_length/interaction_length where particle interactions
    # are neglected and the resonance approximation is used
    # 0.5 ~ precision loss <+3% speed gain ~ factor 10
    # If smoothness and shape accuracy for prompt flux is crucial, use smaller
    # values around 0.1 or 0.05
    "hybrid_crossover": 0.5,

    # Muon energy loss according to Kokoulin et al.
    "enable_muon_energy_loss": True,

    # Minimal stepsize for muon energy loss steps in g/cm2
    "muon_energy_loss_min_step": 5.,

    # First interaction mode
    # (stop particle production after one interaction length)
    "first_interaction_mode": False,

    # When using modified particle production matrices use
    # isospin symmetry to determine the modification to neutrons
    # and K0L/K0S
    "use_isospin_sym": True,

    # Possibilities to control the solver (some options are obsolete/not
    # working)
    "vetos": {
        # inhibit coupling/secondary production of mesons
        "veto_sec_interactions": False,
        # Disable resonance/prompt contribution
        "veto_resonance_decay": False,
        "veto_hadrons": [],
        "veto_resonances": [],
        "allow_resonances": [],

        # Allow only those particles to be projectiles (incl. anti-particles)
        # Faster initialization,
        # For inclusive lepton flux computations:
        # precision loss ~ 1%, for SIBYLL2.3.X with charm 5% above 10^7 GeV
        # Might be different for yields (set_single_primary_particle)
        # For full precision or if in doubt, use []
        "allowed_projectiles": [2212, 2112, 211, 321, 130],

        # Disable leptons coming from decays inside the interaction models
        "veto_direct_leptons":False,

        # Difficult to explain parameter
        'veto_forward_mesons':False,

        # Do not apply mixing to these particles
        "exclude_from_mixing": [],

        # Switch off decays. E.g., disable muon decay with [13,-13]
        "veto_decays": [],

        # Switch off particle production by charm projectiles
        "veto_charm_pprod": False,

        # Disable mixing between resonance approx. and full propagation
        "no_mixing": False
        }
}



dbg = config['debug_level']

def mceq_config_without(key_list):
    r = dict(config)  # make a copy
    for key in key_list:
        del r[key]
    return r
