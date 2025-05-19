# import os.path as path
import importlib.util
import pathlib
import platform
import sys

base_path = pathlib.Path(__file__).parent.resolve()

# base_path = path.dirname(path.abspath(__file__))

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
data_dir = base_path / "data"

#: File name of the MCEq database
mceq_db_fname = "mceq_db_lext_dpm193_v150.h5"

#: File name of the MCEq database
em_db_fname = "mceq_db_EM_Tsai_Max_v150.h5"

#: Decay database name
decay_db_name = None

# =================================================================
# Atmosphere and geometry settings
# =================================================================

#: The latest versions of MCEq work in kinetic energy not total energy
#: If you want the result to be compatible with the previous choose
#: 'total energy' else 'kinetic energy'
return_as = "kinetic energy"
#: Atmospheric model in the format: (model, (arguments))
density_model = ("CORSIKA", ("BK_USStd", None))
#: density_model = ('MSIS00_IC',('SouthPole','January'))
#: density_model = ('GeneralizedTarget', None)

#: Definition of prompt (only for correct accounting). Leptons from parent particles
#: with ctau < prompt_ctau will be counted in the pr_[mu, numu, nue] category, whereas
#: everything else will be attributed to the "conventional" category
prompt_ctau = 2.6842  # cm (everything shorter-lived than K0s will be considered prompt)

#: Approximate value for the maximum density expected. Needed for the
#: resonance approximation. Default value: air at the surface
max_density = 0.001225
#: Material for interaction lengths, ionization and radiation (=continuous) loss terms
#: Currently available choices: 'air', 'water', 'ice', 'rock', 'co2', 'hydrogen', 'iron'
interaction_medium = "air"

#: Average target mass (for interaction length calculations)
#: Change parameter only in combination with interaction model setting.
#: By default, secondary particle production matrices are calculated for air targets
#: If set to 'auto', use default according to the "interaction_medium" settings below
A_target = "auto"

#: parameters for EarthGeometry
r_E = 6371.0e5  # Earth radius in cm
h_obs = 0.0  # observation level in cm
h_atm = 112.8e5  # top of the atmosphere in cm
X_start = 0.0  # starting slant depth in g/cm^-2


#: Default parameters for GeneralizedTarget
#: Total length of the target [m]
len_target = 1000.0
#: density of default material in g/cm^3
env_density = 0.001225
env_name = "air"

#: Check if densities requested outside of target dimensions
except_out_of_bounds = False

# =================================================================
# Parameters of numerical integration
# =================================================================

#: Minimal energy for grid
#: The minimal energy (technically) is 1e-2 GeV. Currently you can run into
#: stability problems with the integrator with such low thresholds. Use with
#: care and check results for oscillations and feasibility.
e_min = 0.1

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

#: euler kernel implementation (numpy/MKL/CUDA/accelerate).
#: With serious nVidia GPUs CUDA a few times faster than MKL
#: autodetection of fastest kernel below
kernel_config = "auto"

#: Select CUDA device ID if you have multiple GPUs
cuda_gpu_id = 0

#: Floating point precision (is set automatically)
floatlen = None

#: Number of MKL threads (for sparse matrix multiplication the performance
#: advantage from using more than a few threads is limited by memory bandwidth)
#: Irrelevant for GPU integrators, but can affect initialization speed if
#: numpy is linked to MKL.
mkl_threads = 8

#: parameters for the odepack integrator. More details at
#: http://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html#scipy.integrate.ode
ode_params = {
    "name": "lsoda",
    "method": "bdf",
    "nsteps": 1000,
    # 'max_step': 10.0,
    "rtol": 0.01,
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
#: use for debugging purposes when searching for stability issues, especially
#: if e_min is < 1 GeV.
dXmax = 1.0

#: Minimal CR nucleon energy in primary model. If (low energy)
#: hadronic interaction model doesn't properly implement interactions
#: or cross sections, nucleons can "drop through" without cascading

minimal_primary_energy = 3.0

#: Enable default tracking particles, such as pi_numu, pr_mu+, etc.
#: If only total fluxes are of interest, disable this feature to gain
#: performance since the eqution system becomes smaller and sparser
enable_default_tracking = True

#: Ionization and radiative losses according to stopping power tables (PDG)
enable_energy_loss = True

#: Apply stopping power to all charged hadrons (muon dEdX is used and is ~ok)
generic_losses_all_charged = True

#: Treat radiation (bremsstrahlung) as continuous loss, disable if explicit
#: electromagnetic cross sections available
enable_cont_rad_loss = True

#: Fall-back to air production matrices if medium not included in data file
fallback_to_air_cs = True

#: enable EM ionization loss for electrons and positrons
enable_em_ion = True

#: Improve (explicit solver) stability by averaging the continous loss
#: operator
average_loss_operator = False

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

#: Advanced settings (some options might be obsolete/not working)
adv_set = {
    #: Disable particle production by all hadrons, except nucleons
    "disable_interactions_of_unstable": False,
    #: Disable particle production by charm *projectiles* (interactions)
    "disable_charm_pprod": False,
    #: repair DPMJET-III K0S/L: due to a bug in matrix generation the
    # matrices are not properly populated. As an intermediate fix, K0S/L
    # are approximated from a mixture of K+,K- distributions. In future
    # DPMEJET versions, this bug will be resolved and the parameter is temporal.
    "fix_dpmjet_neutral_kaons": True,
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
    "disabled_particles": [20, 19, 18, 17, 97, 98, 99, 101, 102, 103],
    #: Disable leptons coming from prompt hadron decays at the vertex
    "disable_direct_leptons": False,
    #: Difficult to explain parameter
    "disable_leading_mesons": False,
    #: Do not apply mixing to these particles
    "exclude_from_mixing": [13],
    #: Switch off decays. E.g., disable muon decay with [13,-13]
    "disable_decays": [],
    #: Force particles to be treated as resonance
    "force_resonance": [],
    #: Disable mixing between resonance approx. and full propagation
    "no_mixing": False,
    #: Force the interaction cross sections to a specific model
    "forced_int_cs": None,
    #: Replace only the meson air cross sections with that from a different model
    "replace_meson_cross_sections_with": None,
}

#: Particles for compact mode
standard_particles = [11, 12, 13, 14, 16, 211, 321, 2212, 2112, 3122, 411, 421, 431]

#: Anti-particles
standard_particles += [-pid for pid in standard_particles]

#: Neutral & unflavored particles
#: append 221, 223, 333, if eta, omega and phi needed directly
standard_particles += [22, 111, 130, 310]  #: , 221, 223, 333]

#: This construct provides access to the attributes as in previous
#: versions, using `from mceq_config import config`. The future versions
#: will access the module attributes directly.

#: Autodetect best solver
#: determine shared library extension and MKL path
pf = platform.platform()
has_accelerate = False

prefix = pathlib.Path(sys.prefix)
if "Linux" in pf:
    mkl_path = prefix / "lib" / "libmkl_rt.so"
elif "macOS" in pf:
    mkl_path = prefix / "lib" / "libmkl_rt.dylib"
    has_accelerate = True
else:
    # Windows or unknown OS: search for mkl_rt*.dll in Library/bin and lib
    mkl_path = None
    mkl_dirs = [prefix / "Library" / "bin", prefix / "lib"]
    mkl_candidates = []
    for d in mkl_dirs:
        if d.exists():
            mkl_candidates.extend(d.glob("mkl_rt*.dll"))
    if mkl_candidates:
        mkl_path = mkl_candidates[0]
    else:
        # fallback to default path
        mkl_path = prefix / "Library" / "bin" / "mkl_rt.dll"

# mkl library handler
mkl = None

has_mkl = bool(mkl_path.is_file())

# Look for cupy module
has_cuda = importlib.util.find_spec("cupy") is not None


class FileIntegrityCheck:
    """
    A class to check a file integrity against provided checksum

    Attributes
    ----------
    filename : str
        path to the file
    checksum : str
        hex of sha256 checksum
    Methods
    -------
    succeeded():
        returns True if checksum and calculated checksum of the file are equal

    get_file_checksum():
        returns checksum of the file
    """

    import hashlib

    def __init__(self, filename, checksum=""):
        self.filename = filename
        self.checksum = checksum
        self.sha256_hash = self.hashlib.sha256()
        self.hash_is_calculated = False

    def _calculate_hash(self):
        if not self.hash_is_calculated:
            try:
                with open(self.filename, "rb") as file:
                    for byte_block in iter(lambda: file.read(4096), b""):
                        self.sha256_hash.update(byte_block)
                self.hash_is_calculated = True
            except OSError as ex:
                print(f"FileIntegrityCheck: {ex}")

    def succeeded(self):
        self._calculate_hash()
        return self.hash_is_calculated and self.sha256_hash.hexdigest() == self.checksum

    def get_file_checksum(self):
        self._calculate_hash()
        return self.sha256_hash.hexdigest()


def _download_file(url, outfile):
    """Downloads the MCEq database from github"""

    import math

    import requests
    from tqdm import tqdm

    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(r.headers.get("content-length", 0))
    block_size = 1024 * 1024
    wrote = 0
    with open(outfile, "wb") as f:
        for data in tqdm(
            r.iter_content(block_size),
            total=math.ceil(total_size // block_size),
            unit="MB",
            unit_scale=True,
        ):
            wrote = wrote + len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size:
        raise Exception("ERROR, something went wrong")


# Download database file from github
base_url = "https://github.com/afedynitch/MCEq/releases/download/"
release_tag = "builds_on_azure/"
url = base_url + release_tag + mceq_db_fname
# sha256 checksum of the file
# https://github.com/afedynitch/MCEq/releases/download/builds_on_azure/mceq_db_lext_dpm191_v12.h5
file_checksum = "6353f661605a0b85c3db32e8fd259f68433392b35baef05fd5f0949b46f9c484"

filepath_to_database = data_dir / mceq_db_fname
# if path.isfile(filepath_to_database):
#     is_file_complete = FileIntegrityCheck(
#         filepath_to_database, file_checksum
#     ).succeeded()
# else:
#     is_file_complete = False
is_file_complete = True
if not is_file_complete:
    print(f"Downloading for mceq database file {mceq_db_fname}.")
    if debug_level >= 2:
        print(url)
    _download_file(url, filepath_to_database)

# old_database = "mceq_db_lext_dpm191.h5"
# filepath_to_old_database = path.join(data_dir, old_database)

# if path.isfile(filepath_to_old_database):
#     import os

#     print("Removing previous database {0}.".format(old_database))
#     os.unlink(filepath_to_old_database)
