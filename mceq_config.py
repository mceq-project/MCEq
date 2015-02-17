import sys
sys.path.append("../AnalyticalApproximation")
sys.path.append("../CRFluxModels")
sys.path.append("../ParticleDataTool")
sys.path.append("../Fedynitch2012")


config = {

# Debug flag for verbose printing
"debug_level": 1,

#=========================================================================
# Paths and library locations
#=========================================================================

# Directory where the data files for the calculation are stored
"data_dir": '/Users/afedynitch/Documents/KIT/artifacts/matrix_method/data_files',

# File name of particle production yields
"yield_fname": "yield_dict.ppd",

# File name of particle decay spectra
"decay_fname": "decay_dict.ppd",

# File name of the cross-sections tables
"cs_fname":"cs_dict.ppd",

# File where to cache interpolating splines of the atmosphere module
'atm_cache_file':'atm_cache.ppd',

# full path to libmkl_rt.[so/dylib]
"MKL_path": "/Users/afedynitch/anaconda/lib/libmkl_rt.dylib",

#=========================================================================
# Basic calculation settings
#=========================================================================
# Zenith angle in degrees. 0 means vertical, 90 is horizontal
"theta_deg": 0.0,

# Use file for caching calculated atmospheric rho(X) splines
"use_atm_cache": True,

# Atmospheric model in the format: (model, parametrise ation, options)
"atm_model": ('CORSIKA', 'BK_USStd', None),

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

"ode_params": {'name':'vode',
               'method':'adams',
               'nsteps':10000,
               'max_step':10.0},

# Use sparse linear algebra (recommended!)
"use_sparse": True,

# CUDA float precision (has no effect on other solvers than CUDA)
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
