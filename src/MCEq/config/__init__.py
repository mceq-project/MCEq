import os
import pathlib
import platform
import sys
import warnings

from MCEq import base_path

from . import detect

#: Debug verbosity; level 0 still permits messages logged at level 0
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
data_dir = pathlib.Path(base_path) / "data"

#: File name of the MCEq database
mceq_db_fname = "mceq_base_air_1d_v2.h5"

#: File name of the MCEq database
em_db_fname = "mceq_db_EM_Tsai-Max_Z7.31.h5"

#: Data-release manifest (YAML) listing each database file with its energy
#: grid, checksum and the hadronic models it carries. Additional or updated
#: model files are registered here.
mceq_db_manifest = "mceq_db_manifest_v2.yaml"

# =================================================================
# Atmosphere and geometry settings
# =================================================================

#: The latest versions of MCEq work in kinetic energy not total energy
#: If you want the result to be compatible with the previous choose
#: 'total energy' else 'kinetic energy'
return_as = "kinetic energy"
#: Atmospheric model in the format: (model, (arguments))
#: CORSIKA BK_USStd is the default — generic, location-free, and dependency-free.
#: NRLMSIS 2.1 (whole-atmosphere refit with newer satellite data) is available
#: opt-in but requires the external 'nrlmsis' package
#: (pip install 'git+https://github.com/afedynitch/nrlmsis2.1').
density_model = ("CORSIKA", ("BK_USStd", None))
#: Alternatives:
#: density_model = ('MSIS21', ('SouthPole', 'January'))   # NRLMSIS 2.1 — needs 'nrlmsis' (opt-in)
#: density_model = ('MSIS00', ('SouthPole', 'January'))  # MSISE-00 (bundled C implementation)
#: density_model = ('MSIS00_IC', ('SouthPole', 'January'))
#: density_model = ('MSIS21_IC', ('SouthPole', 'January'))     # detector-centered NRLMSIS 2.1
#: density_model = ('MSIS21_KM3NeT', ('ORCA', 'January'))      # detector-centered NRLMSIS 2.1 at ORCA
#: density_model = ('GeneralizedTarget', None)

#: Definition of prompt (only for correct accounting). Leptons from parent particles
#: with ctau < prompt_ctau will be counted in the pr_[mu, numu, nue] category, whereas
#: everything else will be attributed to the "conventional" category
# cm (everything shorter-lived than K0s will be considered prompt)
prompt_ctau = 2.6842

#: Bin-averaged rates for databases with stencil-averaged tables: the decay and
#: interaction rates of the parents the database marks (mceq-maintenance-tools
#: ``bin_average``) become banded stencil operators instead of diagonals, so the
#: counted decays and interactions are bin integrals. No effect on databases
#: without the marks.
bin_average_rates = True

#: Approximate value for the maximum density expected. Used by the
#: atmosphere model. Default value: air at the surface.
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
r_E = 6371.315e3  # Earth radius in m, as in CORSIKA
h_obs = 0.0  # observation level in m
h_atm = 112.8e3  # top of the atmosphere in m
X_start = 0.0  # starting slant depth in g/cm^-2

#: Default parameters for GeneralizedTarget
#: Total length of the target [m]
len_target = 1000.0
#: density of default material in g/cm^3
env_density = 0.001225
env_name = "air"
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

#: Enable electromagnetic cascade with matrices from the EM database
#: (``em_db_fname``)
enable_em = False

#: Air density in g/cm³ used to select a density slice from a
#: ρ-stratified EM database (one produced by
#: ``mceq-maintenance-tools/database_generator/5_assemble_em_db.py``
#: with ``--air-density-grid``). The loader chooses the nearest stored
#: density in log10 space; ``None`` selects the legacy single-density
#: slice for back-compat with un-stratified EM databases.
em_air_density = None

#: ETD2RK kernel implementation. Choices:
#:   "auto"            — pick the best available ETD2 kernel (see below).
#:   "numpy_etd2"      — pure-numpy ETD2RK (always available).
#:   "accelerate_etd2" — Apple Accelerate-backed ETD2RK (macOS only).
#:   "mkl_etd2"        — Intel MKL-backed ETD2RK (Linux/Windows; faster
#:                       sparse SpMV than numpy on multi-core CPUs).
#:   "cuda_etd2"       — NVIDIA cuSPARSE-backed ETD2RK (requires cupy and
#:                       a CUDA-capable GPU; recommended for large state
#:                       vectors and many solve() calls).
_kernel_config_request = "auto"

#: Select CUDA device ID if you have multiple GPUs
cuda_gpu_id = 0

#: CUDA floating-point precision of the state (default 64 bits); ``solve()``
#: and the batch routes use it when the call names no ``dtype``.
cuda_fp_precision = 64

#: CUDA precision of the stored off-diagonal operator values: None keeps the
#: state precision; 32 under an fp64 state is mixed precision -- fp32 values,
#: fp64 state and accumulation -- which halves the bytes the sparse product
#: streams per step.
cuda_operator_precision = None

#: Floating point precision (is set automatically)
floatlen = None


def _available_cpus():
    """CPUs this process may run on (the affinity mask, else ``cpu_count``)."""
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return os.cpu_count() or 1


def _default_threads():
    """BLAS threads to use: an explicit thread count in the environment,
    else ``min(16, cpus)`` over the CPUs this process may run on.
    """
    for var in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        value = os.environ.get(var, "")
        if value.isdigit() and int(value) > 0:
            return int(value)
    return min(16, _available_cpus())


#: BLAS thread count; change with ``MCEq.config.set_mkl_threads(n)``.
mkl_threads = _default_threads()

# =========================================================================
# Advanced settings
# =========================================================================

#: Default parameters of the non-uniform ETD2 integration path
#: (:func:`MCEq.solvers.etd2_nonuniform_path`); each can be overridden per
#: call via ``MCEqRun.solve(..., eps=..., dX_max=...)``.
etd2_path = {
    #: Within-step log inverse-density variation. None selects .03 for
    #: 1D hadronic/lepton transport and .01 for 2D or electron transport.
    #: A numerical-accuracy guideline for the path controller; validate
    #: flux accuracy separately.
    "eps": None,
    #: Requested ceiling in g/cm^2: None selects 5 for 1D hadronic/lepton
    #: transport, 2 for 2D or electron transport. Matrix guards still apply.
    "dX_max": None,
    #: Floor on the step size. Prevents the controller from picking 0
    #: when |d ln rho_inv / dX| is very large (top of atmosphere).
    "dX_min": 0.01,
    #: Forward-FD probe span in g/cm^2 used to estimate
    #: |d ln rho_inv / dX|. Must be large enough to cross the
    #: `r_X2rho` spline saturation cap at the top of atmosphere
    #: (~1e-4 g/cm^2 for CORSIKA atmospheres) and small enough to
    #: resolve the local derivative in the bulk.
    "fd_span": 0.01,
}

#: Cap EM integration steps at ``em_step_safety / r_EM``, with ``r_EM`` the
#: spectral radius of the e+/-/gamma off-diagonal block of ``int_m``.
em_adaptive_step = False

#: Safety factor of the EM step cap ``dX = em_step_safety / r_EM`` (g/cm^2).
em_step_safety = 0.12

#: Largest EM block dimension for which ``r_EM`` uses dense ``eigvals``
#: (ARPACK above).
em_step_dense_eig_max = 4000

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

#: Apply stopping power to all charged hadrons with the generic hadron loss
#: table, interpolated in beta-gamma, so low-energy charged hadrons range out.
#: False: only muons (and e+-, see ``enable_em_ion``) carry continuous losses.
generic_losses_all_charged = True

#: Treat radiation (bremsstrahlung) as continuous loss, disable if explicit
#: electromagnetic cross sections available
enable_cont_rad_loss = True

#: Fall-back to air production matrices if medium not included in data file
fallback_to_air_cs = True

#: enable EM ionization loss for electrons and positrons
enable_em_ion = True

#: Average the continuous-loss operator through repeated finite steps of size
#: ``loss_step_for_average``. This changes the effective operator.
average_loss_operator = False

#: Step size (dX) for averaging
loss_step_for_average = 1e-1

#: Stencil of the continuous-loss operator ``d/du``, ``u = ln E``
#: (:mod:`MCEq.operators.loss_stencil`):
#:   "expfit_low_upwind2" -- expfit interior, second-order upwind rows in the
#:                 low-energy layer (``loss_stencil_low_upwind_rows``).
#:   "expfit_low_upwind" -- the same with first-order upwind rows.
#:   "expfit"   -- 7-point exponentially fitted stencil anchored at
#:                 ``loss_stencil_alpha0``.
#:   "centered" -- 6th-order centered FD.
#:   "biased"   -- 7-point biased stencil (v1 default).
#:   "upwind", "upwind2" -- first-/second-order upwind everywhere.
loss_stencil_method = "expfit_low_upwind2"

#: Number of low-energy rows replaced by the selected upwind stencil when
#: ``loss_stencil_method`` is ``"expfit_low_upwind"`` or
#: ``"expfit_low_upwind2"``. The layer is widened per species where the loss
#: stiffness requires it.
loss_stencil_low_upwind_rows = 8

#: Anchor exponent for the "expfit" stencil. The stencil is constructed to
#: be exact for f = exp(a u) at trial slopes a = -alpha0 + delta around
#: -alpha0; alpha0 ~ 3 covers the typical CR power-law range. The stencil
#: is robust to mis-specifications of order +/- 1.
loss_stencil_alpha0 = 3.0

#: Raise exception when requesting unknown particles from get_solution
excpt_on_missing_particle = False

#: When using modified particle production matrices use
#: isospin symmetries to determine the corresponding
#: modification for neutrons and K0L/K0S
use_isospin_sym = True

#: Helicity-dependent muon decays from analytical expressions. Keep on: the
#: ``decays/unpolarized`` DB dataset halves the K_mu2 channel. Forced off when
#: ``enable_em`` is set.
muon_helicity_dependence = True

#: Coulomb scattering of muons in the 2D transport (see
#: :mod:`MCEq.operators.scattering`). Disable only to compare against MC runs
#: with scattering switched off.
muon_multiple_scattering = True

#: Muon scattering model: "screened-coulomb" (screened Rutherford, with the
#: single-scattering tail; needs the medium composition in the database) or
#: "gaussian" (CORSIKA Gauss approximation, Gaussian core only).
muon_scattering_model = "screened-coulomb"

#: Screened Coulomb only: nuclear form factor and electron kinematic limit as
#: in Geant4's WentzelVI cross section (see :mod:`MCEq.operators.scattering`).
#: False restores point nuclei, whose Rutherford tail over-populates large
#: angles above ~1 GeV.
muon_scattering_finite_nuclei = True

#: sec(theta) path-elongation correction for the 2D transport. The
#: paraxial solver books all losses per unit axis-projected depth,
#: over-predicting the wide-angle density of species in local
#: production/loss equilibrium (sub-GeV hadrons and muons) by
#: sec(theta). Physics, operator construction and solver integration:
#: :mod:`MCEq.operators.secant`. All angles are relative to the shower axis (like
#: the angular modes), independent of the axis' zenith angle.
#: "auto" (default) and True apply it on every solve with a 2D database;
#: False: off. Ignored on 1D databases.
secant_theta_transport = "auto"
#: angle (deg) at which the sec(theta) elongation is clamped:
#: g(theta) = min(sec theta, sec cap). Raise toward 90 for more
#: elongation at wide angles at the price of a worse-conditioned
#: coupling operator; 0 switches the elongation off. Valid range [0, 90).
secant_theta_cap_deg = 75.0
#: Legendre 2D databases only: remove produced particles moving backward
#: (no albedo) with :func:`MCEq.operators.legendre.backward_removal`. Off
#: transports them as if they moved downward at the cap angle.
legendre_backward_removal = True
#: apply the coupling only to state columns with E_kin (GeV) below this
#: threshold, above which the correction is negligible. None applies it to
#: all energies.
secant_theta_e_max = 31.6
#: Assume nucleon, pion and kaon cross sections for interactions of
#: rare or exotic particles (mostly relevant for non-compact mode)
assume_nucleon_interactions_for_exotics = True

#: Optional run-time low-energy model blending in the HDF5 backend.  When
#: ``model`` is None (default), the selected interaction model is loaded
#: unchanged.  ``he_le_trwidth`` is the 10--90% sigmoid width in log10(E/GeV)
#: decades; zero selects a hard switch at ``he_le_transition``.
low_energy_extension = {
    "model": None,
    "he_le_transition": 80,  # GeV
    "he_le_trwidth": 0.3,  # decades (10--90% width)
    "use_unknown_cs": True,
}

#: Advanced settings (some options might be obsolete/not working)
adv_set = {
    #: Exclude proton, neutron, antiproton, and antineutron projectiles
    "disable_interactions_of_unstable": False,
    #: Disable particle production by charm *projectiles* (interactions)
    "disable_charm_pprod": False,
    #: Restrict projectiles to the listed PDG IDs; antiparticle IDs must be
    #: listed explicitly.
    #: Faster initialization,
    #: For inclusive lepton flux computations:
    #: precision loss ~ 1%, for SIBYLL2.3.X with charm 5% above 10^7 GeV
    #: Might be different for yields (set_single_primary_particle)
    #: For full precision or if in doubt, use []
    "allowed_projectiles": [],  # [2212, 2112, 211, 321, 130, 11, 22],
    #: Disable particle (production). Photons and electrons need not be
    #: listed: they are excluded automatically whenever ``enable_em`` is
    #: off, and included when it is on. The HDF5 reader checks absolute
    #: particle IDs against this list, so a positive ID excludes both signs
    #: there; later interaction-table filtering uses literal IDs.
    "disabled_particles": [],  # 20, 19, 18, 17, 97, 98, 99, 101, 102, 103
    #: Disable leptons coming from prompt hadron decays at the vertex
    "disable_direct_leptons": False,
    #: Switch off decays. E.g., disable muon decay with [13,-13]
    "disable_decays": [],
    #: Force particles (by absolute PDG id, excluding standard_particles) to
    #: be treated as resonances: folded into other particles' matrices, with
    #: no state vector entries of their own. Empty: propagate everything.
    "force_resonance": [],
    #: Force the interaction cross sections to a specific model
    "forced_int_cs": None,
    #: Replace only the meson air cross sections with that from a different model
    "replace_meson_cross_sections_with": None,
}

#: Particles for compact mode
standard_particles = [11, 12, 13, 14, 16, 211, 321, 2212, 2112, 3122, 411, 421, 431]

#: Anti-particles
standard_particles += [-pid for pid in standard_particles]

#: unflavored particles
#: append 221, 223, 333, if eta, omega and phi needed directly
standard_particles += [22, 111, 130, 310]  #: , 221, 223, 333]

#: Loaded ``libmkl_rt`` handle, or None until :func:`load_mkl` loads it.
mkl = None

# Platform probes resolve on first read through __getattr__ (see detect.py)
# and are then cached as module attributes.

_LAZY = {
    "has_mkl": lambda: detect.has_mkl(),
    "has_cuda": lambda: detect.has_cuda(),
    "has_accelerate": lambda: detect.has_accelerate(),
    "mkl_path": lambda: detect.mkl_library_path(),
    "kernel_config": lambda: detect.resolve_kernel(_kernel_config_request),
    "pf": lambda: platform.platform(),
    "prefix": lambda: pathlib.Path(sys.prefix),
}


def __getattr__(name):
    probe = _LAZY.get(name)
    if probe is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = probe()
    globals()[name] = value  # subsequent reads bypass this hook
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


def load_mkl():
    """Load ``libmkl_rt`` once if it is installed; return the handle or None."""
    global mkl
    if mkl is None and detect.has_mkl():
        from ctypes import cdll

        # The Windows loader on Python 3.10/3.11 does not accept a Path.
        mkl = cdll.LoadLibrary(os.fspath(detect.mkl_library_path()))
    return mkl


#: Environment variables every BLAS MCEq can reach reads when it loads.
_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _publish_thread_env(nthreads):
    """Set the thread environment variables read by BLAS libraries loaded later."""
    for var in _THREAD_ENV:
        os.environ[var] = str(nthreads)


#: Handle of the process-wide BLAS limit, kept alive by :func:`set_mkl_threads`.
_blas_limiter = None


def set_mkl_threads(nthreads):
    """Set the thread count of MKL and, with threadpoolctl, every other loaded BLAS pool.

    Also sets the thread environment variables for BLAS libraries loaded later.
    """
    global mkl_threads
    from ctypes import byref, c_int

    mkl_threads = nthreads
    if load_mkl() is not None:
        mkl.mkl_set_num_threads(byref(c_int(nthreads)))
    global _blas_limiter
    _publish_thread_env(nthreads)

    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        if debug_level >= 2:
            print(
                "threadpoolctl is not installed; only MKL's pool is limited, and "
                "the dense secant GEMMs may contend on a many-core host."
            )
    else:
        # Retain the limiter so a later call can unregister it. It applies to
        # already loaded libraries; the environment settings cover libraries
        # loaded later.
        if _blas_limiter is not None:
            _blas_limiter.unregister()
        _blas_limiter = threadpool_limits(limits=nthreads, user_api="blas")
    if debug_level >= 5:
        print(f"BLAS threads limited to {nthreads}")


# Only the environment is published at import: the dlopen and the threadpoolctl
# pass happen on the first explicit set_mkl_threads, or never.
_publish_thread_env(mkl_threads)


#: Use the EM database's energy grid and skip hadronic interaction and decay
#: channels. The backend still requires the hadronic database for other
#: initialization and loss data.
em_standalone_grid = False


# Compatibility layer for dictionary access to config attributes
# This is deprecated and will be removed in future


class MCEqConfigCompatibility(dict):
    """Deprecated compatibility object containing copied namespace attributes.

    Its dictionary entries are separate from the module attributes and do not
    update module settings. Use ``config.variable`` instead of
    ``config['variable']``.
    """

    def __init__(self, namespace):
        self.__dict__.update(namespace)
        if debug_level > 1:
            warn_str = (
                "Config dictionary is deprecated. "
                + "Use config.variable instead of config['variable']"
            )
            warnings.warn(warn_str, FutureWarning)

    def __setitem__(self, key, value):
        key = key.lower()
        if key not in self.__dict__:
            raise Exception("Unknown config key", key)
        return super(MCEqConfigCompatibility, self).__setitem__(key, value)


config = MCEqConfigCompatibility(globals())


def secant_theta_cap():
    """Validated float value of :data:`secant_theta_cap_deg`.

    The cap is defined relative to the shower axis (like the angular
    modes themselves) and must lie in [0, 90): sec(theta) diverges at
    90 deg (see :mod:`MCEq.operators.secant`).
    """
    cap = float(secant_theta_cap_deg)
    if not 0.0 <= cap < 90.0:
        raise ValueError(
            f"config.secant_theta_cap_deg = {cap:g} is outside the "
            "supported range [0, 90)."
        )
    return cap


def secant_enabled(is_2d):
    """True if the sec(theta) coupling applies: a 2D database and
    :data:`secant_theta_transport` set ("auto" or True)."""
    flag = secant_theta_transport
    auto = isinstance(flag, str) and flag.lower() == "auto"
    return bool(is_2d and (auto or flag))


def muon_helicity():
    """True if helicity-dependent muon decays are used: forced off with
    :data:`enable_em`, whose cascade the helicity rows destabilize."""
    return muon_helicity_dependence and not enable_em
