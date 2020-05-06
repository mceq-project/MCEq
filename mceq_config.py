from __future__ import print_function
import sys
import platform
import os.path as path
import warnings
base_path = path.dirname(path.abspath(__file__))

#: Debug flag for verbose printing, 0 silences MCEq entirely
debug_level = 1
#: Override debug prinput for functions listed here (just give the name,
#: "get_solution" for instance) Warning, this option slows down initialization
#: by a lot. Use only when needed.
override_debug_fcn = []
#: Override debug printout for debug levels < value for the functions above
override_max_level = 10
#: Print module name in debug output
print_module = False

# =================================================================
# Paths and library locations
# =================================================================

#: Directory where the data files for the calculation are stored
data_dir = path.join(base_path, 'MCEq', 'data')

#: File name of the MCEq database
mceq_db_fname = "mceq_db_lext_dpm191_v12.h5"

#: File name of the MCEq database
em_db_fname = "mceq_db_EM_Tsai-Max_Z7.31.h5"

# =================================================================
# Atmosphere and geometry settings
# =================================================================

#: The latest versions of MCEq work in kinetic energy not total energy
#: If you want the result to be compatible with the previous choose
#: 'total energy' else 'kinetic energy'
return_as = "kinetic energy"
#: Atmospheric model in the format: (model, (arguments))
density_model = ('CORSIKA', ('BK_USStd', None))
#: density_model = ('MSIS00_IC',('SouthPole','January'))
#: density_model = ('GeneralizedTarget', None)

#: Definition of prompt: default ctau < 0.123 cm (that of D0)
prompt_ctau = 0.123

#: Average mass of target (for interaction length calculations)
#: Change parameter only in combination with interaction model setting.
#: By default all particle production matrices are calculated for air targets
#: expect those for models with '_pp' suffix. These are valid for hydrogen targets.
#: <A> = 14.6568 for air as below (source https://en.wikipedia.org/wiki/Atmosphere_of_Earth)
A_target = sum([f[0]*f[1] for f in [(0.78084, 14), (0.20946, 16), (0.00934, 40)]])

#: parameters for EarthGeometry
r_E = 6391.e3  # Earth radius in m
h_obs = 0.  # observation level in m
h_atm = 112.8e3  # top of the atmosphere in m

#: Default parameters for GeneralizedTarget
#: Total length of the target [m]
len_target = 1000.
#: density of default material in g/cm^3
env_density = 0.001225
env_name = "air"
#: Approximate value for the maximum density expected. Needed for the
#: resonance approximation. Default value: air at the surface
max_density = 0.001225,
#: Material for ionization and radiation (=continuous) loss terms
#: Currently available choices: 'air', 'water', 'ice'
dedx_material = 'air'
# =================================================================
# Parameters of numerical integration
# =================================================================

#: Minimal energy for grid
#: The minimal energy (technically) is 1e-2 GeV. Currently you can run into
#: stability problems with the integrator with such low thresholds. Use with
#: care and check results for oscillations and feasibility.
e_min = .1

#: The maximal energy is 1e12 GeV, but not all interaction models run at such
#: high energies. If you are interested in lower energies, reduce this value
#: for inclusive calculations to max. energy of interest + 4-5 orders of
#: magnitude. For single primaries the maximal energy is directly limited by
#: this value. Smaller grids speed up the initialization and integration.
e_max = 1e11

#: Enable electromagnetic cascade with matrices from EmCA
enable_em = False

#: Selection of integrator (euler/odepack)
integrator = "euler"

#: euler kernel implementation (numpy/MKL/CUDA).
#: With serious nVidia GPUs CUDA a few times faster than MKL
#: autodetection of fastest kernel below
kernel_config = "numpy"

#: Select CUDA device ID if you have multiple GPUs
cuda_gpu_id = 0

#: CUDA Floating point precision (default 32-bit 'float')
cuda_fp_precision = 32

#: Number of MKL threads (for sparse matrix multiplication the performance
#: advantage from using more than a few threads is limited by memory bandwidth)
#: Irrelevant for GPU integrators, but can affect initialization speed if
#: numpy is linked to MKL. 
mkl_threads = 8

#: parameters for the odepack integrator. More details at
#: http://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html#scipy.integrate.ode
ode_params = {
    'name': 'lsoda',
    'method': 'bdf',
    'nsteps': 1000,
    # 'max_step': 10.0,
    'rtol': 0.01
}

# =========================================================================
# Advanced settings
# =========================================================================

#: The leading process is can be either "decays" or "interactions". This depends
#: on the target density and it is usually chosen automatically. For
#: advanced applications one can force "interactions" to be the dominant
#: process. Essentially this affects how the adaptive step size is computed.
#: There is also the choice of "auto" that takes both processes into account
leading_process = "auto"

#: Stability margin for the integrator. The default 0.95 means that step
#: sizes are chosen 5% away from the stability circle. Usually no need to
#: change, except you know what it does.
stability_margin = 0.95

#: Ratio of decay_length/interaction_length where particle interactions
#: are neglected and the resonance approximation is used
#: 0.5 ~ precision loss <+3% speed gain ~ factor 10
#: If smoothness and shape accuracy for prompt flux is crucial, use smaller
#: values around 0.1 or 0.05
hybrid_crossover = 0.5

#: Maximal integration step dX in g/cm2. No limit necessary in most cases,
#: use for debugging purposes when searching for stability issues.
dXmax = 10.

#: Enable default tracking particles, such as pi_numu, pr_mu+, etc.
#: If only total fluxes are of interest, disable this feature to gain
#: performance since the eqution system becomes smaller and sparser
enable_default_tracking = True

#: Muon energy loss according to Kokoulin et al.
enable_muon_energy_loss = True

#: enable EM ionization loss
enable_em_ion = False

#: Improve (explicit solver) stability by averaging the continous loss
#: operator
average_loss_operator = True

#: Step size (dX) for averaging
loss_step_for_average = 1e-1

#: Raise exception when requesting unknown particles from get_solution
excpt_on_missing_particle = False

#: When using modified particle production matrices use
#: isospin symmetries to determine the corresponding
#: modification for neutrons and K0L/K0S
use_isospin_sym = True

#: Helicity dependent muons decays from analytical expressions
muon_helicity_dependence = True

#: Assume nucleon, pion and kaon cross sections for interactions of
#: rare or exotic particles (mostly relevant for non-compact mode)
assume_nucleon_interactions_for_exotics = True

#: This is not used in the code as before, instead the low energy
#: extension is compiled into the HDF backend files.
low_energy_extension = {
    "he_le_transition": 80,  # GeV
    "nbins_interp": 3,
    "use_unknown_cs": True,
}

#: Advanced settings (some options might be obsolete/not working)
adv_set = {
    #: Disable particle production by all hadrons, except nucleons
    "disable_interactions_of_unstable": False,

    #: Disable particle production by charm *projectiles* (interactions)
    "disable_charm_pprod": False,

    #: Disable resonance/prompt contribution (this group of options
    #: is either obsolete or needs maintenance.)
    #: "disable_resonance_decay" : False,

    #: Allow only those particles to be projectiles (incl. anti-particles)
    #: Faster initialization,
    #: For inclusive lepton flux computations:
    #: precision loss ~ 1%, for SIBYLL2.3.X with charm 5% above 10^7 GeV
    #: Might be different for yields (set_single_primary_particle)
    #: For full precision or if in doubt, use []
    "allowed_projectiles": [],  # [2212, 2112, 211, 321, 130, 11, 22],

    #: Disable particle (production) 
    "disabled_particles": [], #20, 19, 18, 17, 97, 98, 99, 101, 102, 103

    #: Disable leptons coming from prompt hadron decays at the vertex
    "disable_direct_leptons": False,

    #: Difficult to explain parameter
    'disable_leading_mesons': False,

    #: Do not apply mixing to these particles
    "exclude_from_mixing": [13],

    #: Switch off decays. E.g., disable muon decay with [13,-13]
    "disable_decays": [],

    #: Force particles to be treated as resonance
    "force_resonance": [],

    #: Disable mixing between resonance approx. and full propagation
    "no_mixing": False
}

#: Particles for compact mode
standard_particles = [
    11, 12, 13, 14, 16, 211, 321, 2212, 2112, 3122, 411, 421, 431
]

#: Anti-particles
standard_particles += [-pid for pid in standard_particles]

#: unflavored particles
#: append 221, 223, 333, if eta, omega and phi needed directly
standard_particles += [22, 111, 130, 310]  #: , 221, 223, 333]

#: This construct provides access to the attributes as in previous
#: versions, using `from mceq_config import config`. The future versions
#: will access the module attributes directly.

#: Autodetect best solver
#: determine shared library extension and MKL path
pf = platform.platform()

if 'Linux' in pf:
    mkl_path = path.join(sys.prefix, 'lib', 'libmkl_rt.so')
elif 'Darwin' in pf:
    mkl_path = path.join(sys.prefix, 'lib', 'libmkl_rt.dylib')
else:
    # Windows case
    mkl_path = path.join(sys.prefix, 'Library', 'bin', 'mkl_rt.dll')

# mkl library handler
mkl = None

# Check if MKL library found
if path.isfile(mkl_path):
    has_mkl = True
else:
    has_mkl = False

# Look for cupy module
try:
    import cupy
    has_cuda = True
except ImportError:
    has_cuda = False

# CUDA is usually fastest, then MKL. Fallback to numpy.
if has_cuda:
    kernel_config = 'CUDA'
elif has_mkl:
    kernel_config = 'MKL'
else:
    kernel_config = 'numpy'
if debug_level >= 2:
    print('Auto-detected {0} solver.'.format(kernel_config))

def set_mkl_threads(nthreads):
    global mkl_threads, mkl
    from ctypes import cdll, c_int, byref
    mkl = cdll.LoadLibrary(mkl_path)
    # Set number of threads
    mkl_threads = nthreads
    mkl.mkl_set_num_threads(byref(c_int(nthreads)))
    if debug_level >= 5:
        print('MKL threads limited to {0}'.format(nthreads))

if has_mkl:
    set_mkl_threads(mkl_threads)



# Compatibility layer for dictionary access to config attributes
# This is deprecated and will be removed in future


class MCEqConfigCompatibility(dict):
    """This class provides access to the attributes of the module as a
    dictionary, as it was in the previous versions of MCEq

    This method is deprecated and will be removed in future.
    """

    def __init__(self, namespace):
        self.__dict__.update(namespace)
        if debug_level > 1:
            warn_str = ("Config dictionary is deprecated. " +
                        "Use config.variable instead of config['variable']")
            warnings.warn(warn_str, FutureWarning)

    def __setitem__(self, key, value):
        key = key.lower()
        if key not in self.__dict__:
            raise Exception('Unknown config key', key)
        return super(MCEqConfigCompatibility, self).__setitem__(key, value)


config = MCEqConfigCompatibility(globals())


def _download_file(url, outfile):
    """Downloads the MCEq database from github"""

    from tqdm import tqdm
    import requests
    import math

    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024 * 1024
    wrote = 0
    with open(outfile, 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size // block_size),
                         unit='MB', unit_scale=True):
            wrote = wrote + len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size:
        raise Exception("ERROR, something went wrong")


# Download database file from github
base_url = 'https://github.com/afedynitch/MCEq/releases/download/'
release_tag = 'builds_on_azure/'
url = base_url + release_tag + mceq_db_fname
if not path.isfile(path.join(data_dir, mceq_db_fname)):
    print('Downloading for mceq database file {0}.'.format(mceq_db_fname))
    if debug_level >= 2:
        print(url)
    _download_file(url, path.join(data_dir, mceq_db_fname))

if path.isfile(path.join(data_dir, 'mceq_db_lext_dpm191.h5')):
    import os
    print('Removing previous database {0}.'.format('mceq_db_lext_dpm191.h5'))
    os.unlink(path.join(data_dir, 'mceq_db_lext_dpm191.h5'))
