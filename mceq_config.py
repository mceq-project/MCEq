import sys
import platform
import os.path as path
sys.path.append("./CRFluxModels")
sys.path.append("./ParticleDataTool")
sys.path.append("./Python-NRLMSISE-00")
sys.path.append("./c-NRLMSISE-00")
#commented out since projects are far from beeing ready for prime time.
#sys.path.append("../Fedynitch2012")
#sys.path.append("../AnalyticalApproximation")

#detrmine shared library extension
lib_ext = None
if platform.platform().find('Linux') != -1:
    lib_ext = '.so'
elif platform.platform().find('Darwin') != -1:
    lib_ext = '.dylib'
else:
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
"data_dir": './data',

# File name of particle production yields
"yield_fname": "yield_dict.ppd",

# File name of particle decay spectra
"decay_fname": "decay_dict.ppd",

# File name of the cross-sections tables
"cs_fname":"cs_dict.ppd",

# File where to cache interpolating splines of the atmosphere module
'atm_cache_file':'atm_cache.ppd',

# full path to libmkl_rt.[so/dylib] (only if kernel=='MKL')
"MKL_path": path.join(sys.prefix, 'lib', 'libmkl_rt') + lib_ext,

#=========================================================================
# Atmosphere and geometry settings
#=========================================================================

# Use file for caching calculated atmospheric rho(X) splines
"use_atm_cache": False,

# Atmospheric model in the format: (model, parametrise ation, options)
"atm_model": ('CORSIKA', 'BK_USStd', None),

# Version of NRLMSISE-00 python library (ctypes, native)
"msis_python": "ctypes",

# List of particles which decay products will be scored
# in a 'obs_' category
"obs_ids": None,  # Example ["eta", "eta*", "etaC", "omega", "phi"],

# Geometry
"r_E": 6391.e3,  # Earth radius in m
"h_obs": 0.,  # observation level in m
"h_atm": 112.8e3,  # top of the atmosphere in m

#===========================================================================
# Parameters of numerical integration
#===========================================================================
    
# Selection of integrator (euler/odepack)
"integrator": "euler",

# euler kernel implementation (numpy/MKL/CUDA)
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
#advantage from using more than 1 thread is only a few precent due to
#memory bandwidth limitations)
"MKL_threads": 4,

# CUDA float precision
"CUDA_precision": 32,

#=========================================================================
# Advanced settings
#=========================================================================

# Ratio of decay_length/interaction_length where particle interactions
# are neglected and the resonance approximation is used
"hybrid_crossover": 0.05,

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
    # Disable mixing of resonance approx. and propagation
    "no_mixing": False
    }
}



dbg = config['debug_level']

def mceq_config_without(key_list):
    r = dict(config)  # make a copy
    for key in key_list:
        del r[key]
    return r
