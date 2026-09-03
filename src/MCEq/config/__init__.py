import os
import pathlib
import platform
import sys
import warnings

from MCEq import base_path

from . import detect

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
data_dir = pathlib.Path(base_path) / "data"

#: File name of the MCEq database
mceq_db_fname = "mceq_db_lext_dpm193_v140.h5"

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
#: density_model = ('MSIS00', ('SouthPole', 'January'))  # MSISE-00 (Fortran-via-C, fork-unsafe)
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
r_E = 6371.315e3  # Earth radius in m (as in CORSIKA; was 6391 km < v2.0, a typo)
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

#: Air-target EM matrices: select a specific density slice from a
#: ρ-stratified EM database (one produced by
#: ``mceq-maintenance-tools/database_generator/5_assemble_em_db.py``
#: with ``--air-density-grid``).  Value is air density in g/cm³.
#: ``None`` (default) loads the legacy single-density slice — back-compat
#: with un-stratified EM databases.  When set, the loader picks the
#: ρ subgroup whose stored density is closest in log10 to this value.
#: Used for LPM-realistic atmospheric cascades; see
#: ``wiki/methods/lpm-density-factorization.md`` in mceq-em-integration.
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

#: CUDA Floating point precision (default 32-bit 'float')
cuda_fp_precision = 64

#: Floating point precision (is set automatically)
floatlen = None

#: Number of MKL threads (for sparse matrix multiplication the performance
#: advantage from using more than a few threads is limited by memory bandwidth)
#: Irrelevant for GPU integrators, but can affect initialization speed if
#: numpy is linked to MKL. Default is ``min(16, os.cpu_count())``: MKL's
#: sparse SpMV scales near-linearly to ~16 threads on the SIBYLL21
#: matrices, then plateaus / regresses on most servers due to memory
#: bandwidth and NUMA effects. The same count is applied to every BLAS pool
#: MCEq can reach, OpenBLAS included, because the dense mode-coupling GEMMs of
#: the secant routes are skinny ((n_P, n_k) @ (n_k, n_g*K)) and an all-cores
#: fan-out on those shapes is pure contention — 66x slower than a capped pool
#: at K = 8 on a 48-core host. Override after import for full control:
#: ``MCEq.config.set_mkl_threads(n)``.
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
    #: Within-step | d ln rho_inv / dX | bound. Smaller -> finer steps in
    #: the upper atmosphere. 0.3 gives sub-percent muon-flux agreement
    #: across the spectrum at all zeniths; see the design doc for the
    #: tolerance/step-count tradeoff.
    "eps": 0.3,
    #: Cap on the step size in g/cm^2 — the off-diagonal stability cliff
    #: `h * spec(int_off) < 2`, with spec(int_off) ~ 0.094 for the
    #: standard MCEq matrix.
    "dX_max": 20.0,
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

#: EM-cascade adaptive step cap (cure B). The ETD2 ``dX_max`` of 20 g/cm^2
#: above is set by the off-diagonal stability/accuracy bound
#: ``h * spec(int_off) < 2`` with ``spec(int_off) ~ 0.094`` for the *hadronic*
#: matrix. The e+/-/gamma block of ``int_m`` is far stiffer (steep
#: bremsstrahlung/pair multiplication, near-singular soft tail), so the same
#: bound demands a much smaller step. When the density-gradient schedule does
#: not refine on its own (homogeneous media, low-density shower start), the
#: large legacy step over-integrates the EM cascade and biases the
#: charged-shower X_max deep by ~8-12 g/cm^2 (muon/hadron profiles are
#: unaffected: their off-diagonal block is ~200x less stiff). When True, the
#: effective step is additionally capped at ``em_step_safety / r_EM``, where
#: ``r_EM`` is the explicit-stepping stiffness scale of the e+/-/gamma block
#: of ``int_m`` (spectral radius of its off-diagonal part, 1/(g/cm^2)).
#: Default False preserves the legacy schedule exactly; enable for absolute
#: EM-X_max work. Because r_EM tracks the operator, the cap also tightens
#: automatically as the energy grid is refined.
em_adaptive_step = False

#: Dimensionless safety factor for ``em_adaptive_step``: effective cap
#: ``dX = em_step_safety / r_EM`` [g/cm^2]. This is the per-step off-diagonal
#: accuracy budget ``h * spec(int_off_EM)``; the EM-X_max bias is an accuracy
#: (not stability) effect, so this is far tighter than the explicit stability
#: cliff. Re-calibrated 2026-06 against the gamma@100TeV charged-X_max
#: convergence on the v13 1-MeV air EM grid AFTER the dense-r_EM fix
#: (true spec(int_off_EM) = 0.5155 1/(g/cm^2); the previous ARPACK/norm
#: estimate over-stated it and forced ~3x more steps). Convergence vs the
#: dX->0 limit (e+- >=1 MeV, parabolic peak):
#:   safety  cap[g/cm^2]  nsteps  dXmax[g/cm^2]  ground-spectrum max-rel
#:    0.04      0.078       13448     +0.003           0.005%
#:    0.12      0.233        4562     +0.030           0.046%   <- default
#:    0.16      0.310        3528     +0.053           0.081%   (spec ceiling)
#:    0.24      0.466        2290     +0.114           0.172%   (over tolerance)
#: 0.12 holds X_max to <0.05 g/cm^2 and the ground spectrum to <0.05% of the
#: dX->0 limit while cutting steps ~3x vs the old 0.04; 0.16 is the largest
#: value still inside the <0.1 g/cm^2 / <0.1% tolerance. Nmax is
#: step-independent to <0.001% throughout. (Legacy fixed 20 g/cm^2 steps bias
#: X_max up to +57 g/cm^2 in homogeneous media.) See the wiki lesson
#: ``mceq-loss-averaging-grid-fragility``.
em_step_safety = 0.12

#: Max EM off-diagonal block dimension for which ``r_EM`` (the cure-B step
#: scale) is computed with a dense ``np.linalg.eigvals``. The EM block is a
#: small sub-system (a few e+/-/gamma species x dim_e) and strongly
#: NON-NORMAL, on which sparse ``eigs(k=1)`` routinely fails to converge and
#: silently degrades to a matrix-norm over-estimate. Dense eigvals is exact,
#: deterministic and cheap at this size (~1-2 s at dim 2000, computed once per
#: matrix build and cached). Above this guard, fall back to ARPACK then norm.
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

#: Apply stopping power to all charged hadrons (the muon dEdX is used and is
#: ~ok). Default True: without it sub-4-GeV protons and charged hadrons never
#: range out (~2 MeV/g/cm^2 x 2000 g/cm^2 ~ 4 GeV of ionization loss across a
#: slant column) and pile up unphysically at the low-energy end of
#: deep-atmosphere spectra. Requires the monotone low-energy boundary layer of
#: ``loss_stencil_method = "expfit_low_upwind2"`` (the default): with the pure
#: ``"expfit"`` operator the extra hadronic loss rows excite the low-energy
#: boundary cliff and deep-slant solves diverge outright. Set False to
#: reproduce the v1.x behaviour where only muons (and e+-, see
#: ``enable_em_ion``) carried continuous losses.
generic_losses_all_charged = True

#: Treat radiation (bremsstrahlung) as continuous loss, disable if explicit
#: electromagnetic cross sections available
enable_cont_rad_loss = True

#: Fall-back to air production matrices if medium not included in data file
fallback_to_air_cs = True

#: enable EM ionization loss for electrons and positrons
enable_em_ion = True

#: Improve (explicit solver) stability by averaging the continous loss
#: operator. Default False: the canonical configuration is the raw
#: ``expfit_low_upwind2`` loss stencil (now the default, see
#: ``loss_stencil_method``) with NO averaging — averaging inflates the EM
#: number Xmax ~+2.7 g/cm^2 / Nmax ~5.6%. Set True only to reproduce
#: pre-2026-06 historical numbers.
average_loss_operator = False

#: Step size (dX) for averaging
loss_step_for_average = 1e-1

#: Stencil for the continuous-loss differential operator on the
#: log-uniform energy grid. Choices:
#:   "expfit_low_upwind2" (default) -- exponentially-fitted 7-point interior
#:                 stencil with the low-energy boundary layer
#:                 (``loss_stencil_low_upwind_rows`` rows) replaced by
#:                 monotone second-order upwind rows. This is the validated
#:                 canonical configuration: it removes the low-energy
#:                 boundary cliff of the pure "expfit" operator (essential
#:                 for the 1 MeV EM grid, harmless on hadronic-only grids
#:                 where it touches only the lowest rows).
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

#: Number of low-energy rows replaced when ``loss_stencil_method`` is
#: ``"expfit_low_upwind"`` or ``"expfit_low_upwind2"`` (the default). At
#: 10 bins/decade and a 1 MeV EM floor, the formal one-sided boundary rows
#: (0..2) are not enough: the raw operator still develops enormous
#: non-normal transients. Eight rows is the first stable setting in the
#: realistic-screening 1 MeV/no-averaging diagnostic and leaves the expfit
#: interior untouched above ~6 MeV.
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
#: :mod:`MCEq.secant`. All angles are relative to the shower axis (like
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
    #: Default disables both e- (PDG 11) and e+ (PDG -11). Until a
    #: validated EM database is shipped, the ETD2 EM cascade can blow up
    #: at extreme zenith — see the "EM cascade caveat" in
    #: docs/mceq_v1.x_v2_diff.md. Both signs must be listed: the
    #: disable list is matched literally, not by absolute PDG id.
    "disabled_particles": [11, -11],  # 20, 19, 18, 17, 97, 98, 99, 101, 102, 103
    #: Disable leptons coming from prompt hadron decays at the vertex
    "disable_direct_leptons": False,
    #: Difficult to explain parameter
    "disable_leading_mesons": False,
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

#: This construct provides access to the attributes as in previous
#: versions, using `from mceq_config import config`. The future versions
#: will access the module attributes directly.

#: Platform and backend detection resolve on first read, through the module
#: ``__getattr__`` below: importing MCEq must not dlopen a BLAS, probe a GPU or
#: decide which kernel will run. Reading any of ``has_mkl``, ``has_cuda``,
#: ``has_accelerate``, ``mkl_path`` or ``kernel_config`` caches the answer as a
#: real module attribute, so the cost is paid once and later reads are ordinary
#: attribute lookups.

#: ``libmkl_rt`` handle, populated by :func:`_load_mkl`.
mkl = None

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


def _load_mkl():
    """Lazily load ``libmkl_rt`` exactly once and cache it on ``mkl``.

    Splitting the load from :func:`set_mkl_threads` is important:
    ``MklSparseMatrix`` instances pin their own reference to the cdll
    handle, so re-loading the library on every thread-count change
    (the previous behaviour) would leave already-built wrappers tied
    to a stale ``cdll`` while the global ``mkl`` pointed at a fresh
    one — a subtle source of cross-handle bugs. By keeping the global
    pinned to a single cdll for the lifetime of the process we ensure
    every wrapper sees the same symbol table.
    """
    global mkl
    if mkl is not None or not detect.has_mkl():
        return
    from ctypes import cdll

    # ``os.fspath`` is not optional: ``CDLL.__init__`` only calls it itself
    # from CPython 3.12, and its Windows branch does ``'/' in name`` first,
    # which raises TypeError on a Path under 3.10/3.11.
    mkl = cdll.LoadLibrary(os.fspath(detect.mkl_library_path()))


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

    Idempotent on the library side: only ``mkl_set_num_threads`` is
    called on subsequent invocations. The cached cdll handle is
    preserved, so handles in ``MclSparseMatrix`` wrappers stay valid
    across thread-count changes.
    """
    global mkl_threads
    from ctypes import byref, c_int

    _load_mkl()
    mkl_threads = nthreads
    if mkl is not None:
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
        # threadpool_limits restores the previous limits when the object is
        # collected, so the handle is kept for the life of the process. It only
        # reaches libraries already loaded; the environment above covers the rest.
        if _blas_limiter is not None:
            _blas_limiter.unregister()
        _blas_limiter = threadpool_limits(limits=nthreads, user_api="blas")
    if debug_level >= 5:
        print(f"BLAS threads limited to {nthreads}")


# Only the environment is published at import: the dlopen and the threadpoolctl
# pass happen on the first explicit set_mkl_threads, or never.
_publish_thread_env(mkl_threads)


#: Take the energy grid from the EM database instead of the hadronic one.
#: Only meaningful for a standalone EM cascade, where there is no hadronic DB
#: to define it.
em_standalone_grid = False


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
    numerically defective so it cannot be built (see :mod:`MCEq.secant`).
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


# Grouped views over the names above, one per layer of the plan's section 2.2.
# They read and write through to this module, so a component handed
# `config.grid` still sees a later `config.e_min = ...`.
from . import groups as _groups  # noqa: E402

globals().update(_groups.build())
GROUP_OF = _groups.FLAT_TO_GROUP
