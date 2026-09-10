import os
import pathlib
import platform
import sys
import warnings

from MCEq import base_path
from MCEq import mkl_runtime as _mkl_runtime

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
mceq_db_fname = "mceq_db_lext_dpm193_v142.h5"

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

#: CUDA floating-point precision (default 64 bits)
cuda_fp_precision = 64

#: Floating point precision (is set automatically)
floatlen = None

#: BLAS thread limit, defaulting to ``min(16, os.cpu_count() or 1)``.
#: Irrelevant for GPU integrators, but can affect initialization speed if
#: numpy is linked to MKL. Sparse products and narrow dense products can be
#: limited by memory bandwidth or thread contention: MKL's sparse SpMV scales
#: near-linearly to ~16 threads, then plateaus or regresses on most servers,
#: and the dense mode-coupling GEMMs of the secant routes are skinny
#: ((n_P, n_k) @ (n_k, n_g*K)) so an all-cores fan-out on those shapes is pure
#: contention. :func:`set_mkl_threads` applies the limit to supported, loaded
#: BLAS libraries, OpenBLAS (what most numpy wheels link) included. Override
#: after import for full control: ``MCEq.config.set_mkl_threads(n)``.
mkl_threads = min(16, os.cpu_count() or 1)

# =========================================================================
# Advanced settings
# =========================================================================

#: Default parameters for the non-uniform integration path used by
#: ETD2 kernels (`numpy_etd2`, `accelerate_etd2`, ...). See
#: docs/mceq_v1.x_v2_diff.md and `MCEq.solvers.etd2_nonuniform_path`.
#: Each value can be overridden per-call via
#: `MCEqRun.solve(..., eps=..., dX_max=...)`.
etd2_path = {
    #: Bound on the within-step variation of log inverse density. Smaller
    #: values produce finer steps, especially in the upper atmosphere.
    "eps": 0.01,
    #: Accuracy ceiling in g/cm^2. The actual assembled loss stencil and
    #: secant coupling may require a smaller step, especially at low energy.
    "dX_max": 2.0,
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

#: Additionally cap EM integration steps at ``em_step_safety / r_EM``, where
#: ``r_EM`` estimates the stiffness of the EM off-diagonal interaction block
#: (the e+/-/gamma block of ``int_m``; the density-gradient schedule alone does
#: not refine on its own in homogeneous media or at a low-density shower start,
#: and the legacy 20 g/cm^2 step over-integrates the EM cascade and biases the
#: charged-shower X_max deep). EM multiplication rates can require finer steps
#: than the density-gradient criterion. Because r_EM tracks the operator, the
#: cap also tightens automatically as the energy grid is refined. Disabled by
#: default, which preserves the legacy schedule exactly.
em_adaptive_step = False

#: Dimensionless safety factor for the EM step cap: ``dX = em_step_safety /
#: r_EM`` in g/cm^2. Smaller values use finer steps; default 0.12 holds the
#: EM shower maximum to <0.05 g/cm^2 and the ground spectrum to <0.05% of the
#: zero-step limit while cutting step counts ~3x versus a value twice as
#: small; the largest value still inside a <0.1 g/cm^2 / <0.1% tolerance is
#: 0.16. Nmax is step-independent to <0.001% throughout.
em_step_safety = 0.12

#: Max EM off-diagonal block dimension for which ``r_EM`` (the EM step cap
#: scale) is computed with a dense ``np.linalg.eigvals``. The EM block is a
#: small sub-system (a few e+/-/gamma species x dim_e) and strongly
#: NON-NORMAL, on which sparse ``eigs(k=1)`` routinely fails to converge and
#: silently degrades to a matrix-norm over-estimate. Dense eigenvalues are
#: estimates, not exact values, but are deterministic and cheap at this size
#: (~1-2 s at dim 2000, computed once per matrix build and cached). Above this
#: guard, fall back to ARPACK then matrix-norm.
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

#: Apply stopping power to all charged hadrons using the generic hadron loss
#: table, interpolated in beta-gamma. This allows low-energy charged hadrons to
#: range out; without it, sub-4-GeV protons and charged hadrons never stop
#: (~2 MeV/g/cm^2 x 2000 g/cm^2 ~ 4 GeV of ionization loss across a slant
#: column) and pile up unphysically at the low-energy end of deep-atmosphere
#: spectra. The default stencil ``loss_stencil_method =
#: "expfit_low_upwind2"`` uses second-order upwind rows near the lower energy
#: boundary; with the pure ``"expfit"`` operator the extra hadronic loss rows
#: excite the low-energy boundary cliff and deep-slant solves diverge outright.
#: Set False for the v1.x behaviour where only muons (and e+-, see
#: ``enable_em_ion``) carried continuous losses.
generic_losses_all_charged = True

#: Treat radiation (bremsstrahlung) as continuous loss, disable if explicit
#: electromagnetic cross sections available
enable_cont_rad_loss = True

#: Fall-back to air production matrices if medium not included in data file
fallback_to_air_cs = True

#: enable EM ionization loss for electrons and positrons
enable_em_ion = True

#: Average the continuous-loss operator through repeated finite steps.
#: Disabled by default: averaging changes the effective operator and can shift
#: EM shower observables (inflating the EM number Xmax by a few g/cm^2 and Nmax
#: by several percent). Set True only to reproduce pre-2026-06 numbers.
average_loss_operator = False

#: Step size (dX) for averaging
loss_step_for_average = 1e-1

#: Stencil for the continuous-loss differential operator on the
#: log-uniform energy grid. Choices:
#:   "expfit_low_upwind2" (default) -- exponentially-fitted 7-point interior
#:                 stencil with the low-energy boundary layer
#:                 (``loss_stencil_low_upwind_rows`` rows) replaced by
#:                 second-order upwind rows in the low-energy boundary region.
#:                 This is the validated canonical configuration: it removes
#:                 the low-energy boundary cliff of the pure "expfit" operator
#:                 (essential for the 1 MeV EM grid, harmless on hadronic-only
#:                 grids where it touches only the lowest rows).
#:   "expfit_low_upwind" -- same with first-order upwind rows.
#:   "expfit"   -- pure 7-point exponentially-fitted stencil anchored at
#:                 ``loss_stencil_alpha0``. Designed to be near-exact for
#:                 power-law spectra E^{-alpha} with alpha ~ alpha0 on the
#:                 default 10 bins/decade grid; orders of magnitude smaller
#:                 truncation error than plain FD on steep spectra. Uses
#:                 one-sided polynomial-fit boundary rows that develop
#:                 large non-normal transients at a low-energy grid floor
#:                 (the "boundary cliff", see ``docs/mceq_v1.x_v2_diff.md``).
#:   "centered" -- symmetric 6th-order centered FD ([-3..3], [-1,9,-45,45,-9,1]/60).
#:   "biased"   -- legacy 7-point biased "6th-order" stencil (v1 default).
#: "expfit"/"centered"/"biased" share the same one-sided polynomial-fit
#: stencils on the boundary rows (0,1,2 and last-2,last-1,last).
loss_stencil_method = "expfit_low_upwind2"

#: Number of low-energy rows replaced by the selected upwind stencil when
#: ``loss_stencil_method`` is ``"expfit_low_upwind"`` or
#: ``"expfit_low_upwind2"`` (the default); default 8. Increasing this region
#: reduces exposure to the polynomial boundary rows (whose non-normal
#: transients grow at a low-energy grid floor) but shrinks the region using the
#: exponentially fitted stencil. The solver also widens this layer per species
#: when a particle's loss stiffness requires it; this value is the floor.
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

#: Helicity dependent muon decays from analytical expressions. Default True.
#: KEEP THIS ON for flux calculations: the alternative ``decays/unpolarized``
#: DB dataset carries a construction defect (the inclusive K± -> mu nu entry
#: was averaged with its 3-body duplicate, halving the dominant K_mu2
#: channel), which suppresses conventional nu_mu by up to ~40% at TeV
#: energies (all DBs up to and including v150; fix pending a DB rebuild).
#: The EM cascade is the exception: the helicity L/R variants add e±/mu
#: semi-Lagrangian rows without diagonal damping that blow up in the EM
#: system (the ``_EM_BLOWUP_CAVEAT``), so ``enable_em`` runs force this
#: flag off at MCEqRun construction (with a warning). EM shower-maximum
#: observables do not involve kaon neutrinos and are unaffected by the
#: dataset defect.
muon_helicity_dependence = True

#: Gaussian multiple Coulomb scattering of muons in the 2D transport,
#: applied as per-mode diagonal damping -kappa^2 theta_s^2(E)/4 with the
#: CORSIKA Gauss-approximation theta_s (E_s = 21 MeV, lambda_s = 37.7
#: g/cm2; Gaussian core only, no Moliere tail). Disable only to compare
#: against MC runs with scattering switched off.
muon_multiple_scattering = True

#: sec(theta) path-elongation correction for the 2D transport. The
#: paraxial solver books all losses per unit axis-projected depth,
#: over-predicting the wide-angle density of species in local
#: production/loss equilibrium (sub-GeV hadrons and muons) by
#: sec(theta). Physics, operator construction and solver integration:
#: :mod:`MCEq.operators.secant`. All angles are relative to the shower axis (like
#: the Hankel modes), independent of the axis' zenith angle.
#: "auto" (default) and True both apply it on every solve with a 2D
#: database — every entry point, every kernel, fp32 and fp64. False: off.
#: Ignored on 1D databases.
secant_theta_transport = "auto"
#: angle (deg) at which the sec(theta) elongation is clamped:
#: g(theta) = min(sec theta, sec cap). Raise toward 90 for more
#: elongation at wide angles at the price of a worse-conditioned
#: coupling operator; below ~50 the operator's eigenbasis is
#: numerically defective and the build raises. Valid range [50, 90).
secant_theta_cap_deg = 75.0
#: zero coupling-matrix rows with kappa > this. The correction has no
#: support at narrow angular scales (high kappa); raising the limit
#: only adds inversion ringing, lowering it truncates the operator.
secant_theta_row_kmax = 50.0
#: ridge strength of the operator fit, relative to the top singular
#: value of the Gram matrix. Increase only if the operator build
#: reports conditioning problems.
secant_theta_lam_rel = 1e-9
#: weight of the S @ 1 = 1 constraint (kappa-flat = collimated states
#: are not elongated).
secant_theta_w_flat = 1.0
#: apply the coupling only to state columns with E_kin (GeV) below this
#: threshold. The correction is 30-60% at 0.1 GeV, ~1% at 2-4 GeV and
#: <0.1% above 10 GeV, so the default excludes energies where it is
#: numerically irrelevant. None applies it to all energies.
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
    #: Disable particle (production)
    #: Default disables both e- (PDG 11) and e+ (PDG -11). Until a
    #: validated EM database is shipped, the ETD2 EM cascade can blow up
    #: at extreme zenith — see the "EM cascade caveat" in
    #: docs/mceq_v1.x_v2_diff.md. The HDF5 reader checks absolute particle
    #: IDs against this list, so a positive ID excludes both signs there;
    #: later interaction-table filtering uses literal IDs.
    "disabled_particles": [11, -11],  # 20, 19, 18, 17, 97, 98, 99, 101, 102, 103
    #: Disable leptons coming from prompt hadron decays at the vertex
    "disable_direct_leptons": False,
    #: Switch off decays. E.g., disable muon decay with [13,-13]
    "disable_decays": [],
    #: Force particles (by absolute PDG id, excluding standard_particles) to
    #: be treated as resonances — folded into other particles' matrices at
    #: build time and not propagated as their own state vector entries.
    #: Empty list = full propagation for everything (the default after the
    #: ETD2RK migration). Retained as an opt-in escape hatch.
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

#: Platform and backend detection resolve on first read, through the module
#: ``__getattr__`` below: importing MCEq must not dlopen a BLAS, probe a GPU or
#: decide which kernel will run. Reading any of ``has_mkl``, ``has_cuda``,
#: ``has_accelerate``, ``mkl_path`` or ``kernel_config`` caches the answer as a
#: real module attribute, so the cost is paid once and later reads are ordinary
#: attribute lookups.

#: ``libmkl_rt`` handle. :mod:`MCEq.mkl_runtime` owns the process-wide MKL
#: handle (one owner below config, shared by the MKL backend and
#: :func:`set_mkl_threads`); ``config.mkl`` exposes it through
#: ``__getattr__``; no local binding should shadow that view.
_mkl_runtime.set_resolver(lambda: detect.has_mkl(), detect.mkl_library_path)

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
    if name == "mkl":
        # Return the existing MKL handle without loading the library.
        return _mkl_runtime.lib()
    probe = _LAZY.get(name)
    if probe is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = probe()
    globals()[name] = value  # subsequent reads bypass this hook
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


def _load_mkl():
    """Lazily load ``libmkl_rt`` exactly once (compat entry point).

    Delegate to :func:`MCEq.mkl_runtime.load`, which owns the cached
    process-wide handle. The single-handle rule that motivated splitting the
    load from :func:`set_mkl_threads` — every ``MklSparseMatrix`` wrapper must
    see one symbol table for the process lifetime — moved with the memo. Kept
    as the name user code (and ``tests/test_config.py``) already reaches; the
    load-path test targets the new owner.
    """
    return _mkl_runtime.load()


#: Environment variables every BLAS MCEq can reach reads when it loads.
_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _publish_thread_env(nthreads):
    """Announce the thread count to BLAS libraries not yet in the process.

    A BLAS reads its thread count once, when it loads, and numpy loads its own
    lazily on the first product. Publishing costs microseconds and needs no
    library present, which is why it is the only part of the thread setting
    that runs at import.
    """
    for var in _THREAD_ENV:
        os.environ[var] = str(nthreads)


#: Handle of the process-wide BLAS limit, kept alive by :func:`set_mkl_threads`.
_blas_limiter = None


def set_mkl_threads(nthreads):
    """Set the thread count of every BLAS pool MCEq can reach.

    MKL through ``mkl_set_num_threads`` (loading ``libmkl_rt`` on the first
    call) and, when threadpoolctl is available, every other pool it finds —
    OpenBLAS above all, which is what numpy links against in most wheels.
    One process-wide setting, applied once: the solver never adjusts a pool
    around an individual step loop.

    Idempotent on the library side: reuse the loaded MKL handle. Each call
    updates the thread environment and reapplies limits to supported, loaded
    BLAS libraries. The cached cdll handle is preserved, so handles in
    ``MklSparseMatrix`` wrappers stay valid across thread-count changes.
    """
    global mkl_threads
    from ctypes import byref, c_int

    _mkl_runtime.load()
    mkl_threads = nthreads
    lib = _mkl_runtime.lib()
    if lib is not None:
        lib.mkl_set_num_threads(byref(c_int(nthreads)))
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

    The cap is defined relative to the shower axis (like the Hankel
    modes themselves) and must lie in [50, 90): sec(theta) diverges at
    90 deg, and below ~50 deg the coupling operator's eigenbasis is
    numerically defective so it cannot be built (see :mod:`MCEq.operators.secant`).
    """
    cap = float(secant_theta_cap_deg)
    if not 50.0 <= cap < 90.0:
        raise ValueError(
            f"config.secant_theta_cap_deg = {cap:g} is outside the "
            "supported range [50, 90)."
        )
    return cap


def secant_mode(is_2d):
    """Resolve :data:`secant_theta_transport` against the database
    dimensionality.

    Returns ``"off"`` (1D database, or the flag is False) or one of
    ``"auto"`` (the default) and ``"require"`` (flag is True), which the
    resolver treats alike: the coupling is available on every kernel at
    every precision, so no configuration is left to downgrade or refuse.
    """
    flag = secant_theta_transport
    if not is_2d:
        return "off"
    if isinstance(flag, str) and flag.lower() == "auto":
        return "auto"
    return "require" if flag else "off"


# Grouped views over the names above. They read and write the corresponding
# flat configuration attributes, so a component handed
# `config.grid` still sees a later `config.e_min = ...`.
from . import groups as _groups  # noqa: E402

globals().update(_groups.build())
GROUP_OF = _groups.FLAT_TO_GROUP
