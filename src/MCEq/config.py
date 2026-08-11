import importlib
import os
import pathlib
import platform
import sys
import warnings

import numpy as np

from MCEq import base_path

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
#: Approximate value for the maximum density expected. Used by the
#: atmosphere model. Default value: air at the surface.
max_density = (0.001225,)
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

#: TK: energy grid defaults for the cross sections and continuous _cont_losses
#: ported over from the 1D database to the 2D database. Used to "cut" the
#: cross-section arrays defined on the entire 1D MCEq grid down to the
#: smaller 2D grid.
default_ebins = np.logspace(-2, 12, 14 * 10 + 1)
default_ecenters = 0.5 * (default_ebins[1:] + default_ebins[:-1])

#: Enable electromagnetic cascade with matrices from EmCA
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
kernel_config = "auto"

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
#: bandwidth and NUMA effects. Override after import for full control:
#: ``MCEq.config.set_mkl_threads(n)``.
mkl_threads = min(16, os.cpu_count() or 1)

#: Block size for the MKL ETD2 BSR off-diagonal storage. ``6`` is the
#: empirically-tuned default — ~1.5x faster than CSR on SIBYLL21 matrices
#: with MKL >= 2024 (see ``docs/mceq_v1.x_v2_diff.md`` §8.4). MKL appears
#: to specialise its BSR microkernel for ``b in [2, 7]``; ``b >= 8`` falls
#: into a generic path that's slower than CSR for these matrices. Set
#: ``None`` to fall back to CSR (useful for debugging or if a future MKL
#: regresses BSR perf).
mkl_bsr_blocksize = 6

#: Block size for the numpy ETD2 BSR off-diagonal storage. ``11`` is the
#: empirically-tuned default — ~2x faster than CSR on SIBYLL21 matrices
#: via scipy's BSR matvec. scipy's BSR kernel benefits from larger blocks
#: than MKL's (the C++ template's per-block overhead is amortised better),
#: and ``b = 11`` happens to tile the 121-energy-bin macro-blocks neatly
#: (121 = 11**2). Set ``None`` to fall back to CSR.
numpy_bsr_blocksize = 11

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

#: Muon multiple scattering from the CORSIKA-like Gauss approximation
#: (PR #48 / 2D path; folded into the 2D D matrix in Task 1.3).
muon_multiple_scattering = True

#: sec(theta) path-elongation correction for the 2D transport (see
#: MCEq/secant.py). The paraxial solver books all losses per unit
#: axis-projected depth; enabling this right-multiplies the per-mode
#: transport operator by the constant Hankel-space matrix S = I + T
#: representing multiplication by min(sec theta, sec cap), which charges
#: every particle its physical sec(theta) path on the parent side of the
#: yield kick (loss-free daughters — neutrinos — are preserved exactly).
#: Applied inside the ETD2RK kernels (numpy/mkl/cuda): the coupled
#: same-(species,E) block d_i*S_P is integrated exactly in the
#: eigenbasis of S_P, unconditionally stable at any stiffness.
#: Tri-state: "auto" (default) applies the correction to every
#: single-axis solve() on a 2D database and falls back to the paraxial
#: transport (with a warning) on the multi-RHS/carousel entry points
#: and kernels where the coupling is not implemented; True requires it
#: (unsupported paths raise); False disables it. 1D databases are
#: never affected.
secant_theta_transport = "auto"
#: cap angle in degrees for the sec(theta) growth (transport breaks down
#: at 90 deg), or the string "auto". For an axis inclined at zenith
#: theta_z the azimuthal ring at axis-angle theta first touches the
#: horizon at 90 - theta_z; beyond that the flat-atmosphere sec(theta)
#: law has no single meaningful value and the m=0 solver over-attenuates
#: (measured in the 2026-08-10 cap sweep: the >60 deg overshoot is the
#: cap, and it wants to go DOWN). "auto" therefore sets
#: cap = clip(90 - theta_zenith + 5, 30, 75), snapped to 5-degree steps
#: so the disk-cached operator set stays small. A float pins the cap
#: (75.0 was the original static default).
secant_theta_cap_deg = "auto"
#: zero T rows with kappa > this: the correction has no support at narrow
#: angular scales and high-kappa rows carry inversion ringing.
secant_theta_row_kmax = 50.0
#: ridge strength (relative to the Gram matrix top singular value) for
#: the operator fit.
secant_theta_lam_rel = 1e-9
#: weight of the flat-state (kappa-flat = collimated) damping term:
#: enforces S@1 = 1 so collimated states pass through untouched.
secant_theta_w_flat = 1.0
#: apply the coupling only to state columns with E_kin below this (GeV).
#: The effect is 30-60% at 0.1 GeV, ~1% at 2-4 GeV, <0.1% above 10 GeV.
#: None disables the gate.
secant_theta_e_gate = 31.6

#: 2D (Hankel-mode) databases are production-supported for the FLUKA
#: interaction model only, on the energy range the FLUKA angular cubes
#: cover (= the 2D database's own grid). Selecting a different model or
#: enabling runtime HE/LE blending on a 2D database raises — coupling a
#: 1D high-energy model to the 2D low-energy window is the (postponed)
#: hybrid kappa-window extension. The historical URQMD/PR#48 validation
#: databases carry other model labels; the regression tests disable
#: this restriction explicitly.
restrict_2d_to_fluka = True

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

#: Autodetect best solver
#: determine shared library extension and MKL path
pf = platform.platform()
has_accelerate = False

prefix = pathlib.Path(sys.prefix)
if "Linux" in pf:
    mkl_libs = list((prefix / "lib").glob("libmkl_rt*"))
    mkl_path = mkl_libs[0] if mkl_libs else prefix / "lib" / "libmkl_rt.so"
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

    mkl_path = os.fspath(mkl_path)

# mkl library handler
mkl = None

has_mkl = bool(pathlib.Path(mkl_path).is_file())

# Look for cupy module
has_cuda = importlib.util.find_spec("cupy") is not None

# Pick the fastest available ETD2RK kernel. CUDA is intentionally not
# auto-selected: spinning up a GPU context has nontrivial cost and a
# matching cupy install is not always present on machines that have a
# GPU. Apple Accelerate wins on macOS, Intel MKL wins on x86 Linux /
# Windows when present, otherwise we fall back to plain numpy.
if kernel_config == "auto":
    if has_accelerate:
        kernel_config = "accelerate_etd2"
    elif has_mkl:
        kernel_config = "mkl_etd2"
    else:
        kernel_config = "numpy_etd2"
else:
    kc = kernel_config.lower()
    if kc in ("cuda", "cuda_etd2") and not has_cuda:
        raise Exception("CUDA unavailable. Make sure cupy is installed.")
    elif kc in ("mkl", "mkl_etd2") and not has_mkl:
        raise Exception("MKL unavailable. Make sure Intel MKL is installed.")
    elif kc in ("accelerate", "accelerate_etd2") and not has_accelerate:
        raise Exception("Accelerate unavailable. Only on MacOS.")

if debug_level >= 2:
    print(f"Auto-detected {kernel_config} solver.")


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
    if mkl is not None or not has_mkl:
        return
    from ctypes import cdll

    mkl = cdll.LoadLibrary(mkl_path)


def set_mkl_threads(nthreads):
    """Set the MKL thread count (loads ``libmkl_rt`` on the first call).

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
        if debug_level >= 5:
            print(f"MKL threads limited to {nthreads}")


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
# sha256 checksum of the default database file
# https://github.com/afedynitch/MCEq/releases/download/builds_on_azure/mceq_db_lext_dpm191_v12.h5
file_checksum = "5da415e9bcf81926b1061d5792d75cb3aceb9de173beccb4695fd3909a0bfdd0"


def ensure_db_available():
    """Download the MCEq database if not already present.

    Called by MCEqRun.__init__ so that the download is deferred until the
    database is actually needed.  This allows tests (and other callers) to
    override ``config.mceq_db_fname`` before a download is attempted.

    The integrity check only applies to the default database; non-default
    files are accepted as-is if they exist.
    """
    import os

    _url = base_url + release_tag + mceq_db_fname
    filepath = data_dir / mceq_db_fname
    if filepath.exists():
        is_complete = (
            FileIntegrityCheck(filepath, file_checksum).succeeded()
            if mceq_db_fname == "mceq_db_lext_dpm193_v140.h5"
            else True
        )
    else:
        is_complete = False

    if not is_complete:
        print(f"Downloading MCEq database file {mceq_db_fname}.")
        if debug_level >= 2:
            print(_url)
        _download_file(_url, filepath)

    old_db = data_dir / "mceq_db_lext_dpm191.h5"
    if old_db.exists():
        print(f"Removing previous database {old_db.name}.")
        os.unlink(old_db)
