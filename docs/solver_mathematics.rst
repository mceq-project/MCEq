.. _solver-mathematics:

***********************************
The transport step: the mathematics
***********************************

.. currentmodule:: MCEq

This page states the mathematics of the ETD2RK transport step of MCEq 2
as implemented, and names the function that carries each formula. It
covers two things: the diagonal-exact splitting and the exponential
integrator, and the :math:`\sec\theta` path elongation carried in the
eigenbasis of the mode coupling. The code is slim on purpose; what a stage
computes is written here once.

Cascade equations in slant depth
================================

The differential fluxes :math:`\Phi_h(E, X)` of the species :math:`h`
obey, per unit slant depth :math:`X` along the shower axis,

.. math::

   \frac{\partial \Phi_h}{\partial X}
   = -\frac{\Phi_h}{\lambda_{\mathrm{int},h}}
     -\frac{\Phi_h}{\lambda_{\mathrm{dec},h}(E, X)}
     +\frac{\partial}{\partial E}\bigl[\mu_h(E)\,\Phi_h\bigr]
     +\sum_l \int \mathrm{d}E'\,
        \frac{c_{l\to h}(E', E)}{\lambda_{\mathrm{int},l}}\,\Phi_l(E')
     +\sum_l \int \mathrm{d}E'\,
        \frac{d_{l\to h}(E', E)}{\lambda_{\mathrm{dec},l}(E', X)}\,\Phi_l(E') ,

with :math:`\lambda_{\mathrm{dec}}(E, X) = \rho(X)\,E\,c\tau_h / m_h`. On
the energy grid the fluxes form a state vector :math:`\Phi \in
\mathbb{R}^N` and the system is linear,

.. math::
   :label: linear

   \frac{\mathrm{d}\Phi}{\mathrm{d}X} = \bigl[A + \rho^{-1}(X)\,B\bigr]\Phi .

:math:`A` (``int_m``) holds the interaction losses on the diagonal, the
production kernels off it and the energy-loss band; :math:`B` (``dec_m``)
holds the decay rates on the diagonal and the decay kernels off it. Both
are constant along the path; only :math:`\rho^{-1}(X)` changes. Every
short time scale sits on the diagonal.
:class:`~MCEq.operators.matrix_builder.MatrixBuilder` assembles the two
matrices.

Diagonal-exact splitting and the exponential integrator
=======================================================

Over one step :math:`[X_n, X_n + h]` the inverse density is replaced by
its step mean :math:`\rho^{-1}_n` and the operator is split into its
diagonal and its remainder,

.. math::
   :label: split

   A + \rho^{-1}_n B = \mathrm{diag}(D) + N, \qquad
   D = \mathrm{diag}(A) + \rho^{-1}_n \mathrm{diag}(B), \qquad
   N = A_{\mathrm{off}} + \rho^{-1}_n B_{\mathrm{off}} .

:func:`~MCEq.operators.compiled.split_diagonal` performs the split of the
matrices once; the step combines the pieces with the current
:math:`\rho^{-1}_n` in :func:`~MCEq.solvers.numerics.diagonal_factors`.
The variation of constants gives the exact local solution

.. math::

   \Phi(X_n + h) = e^{hD}\,\Phi_n
     + \int_0^h e^{(h - s)D}\, N\, \Phi(X_n + s)\,\mathrm{d}s ,

and the two-stage rule of Cox and Matthews approximates the integral with
the source :math:`F(\Phi) = N\Phi` at the start of the step and at a
predicted end state :math:`a`:

.. math::
   :label: etd2

   a &= e^{hD}\,\Phi_n + h\,\varphi_1(hD)\,F(\Phi_n),

   \Phi_{n+1} &= a + h\,\varphi_2(hD)\,\bigl[F(a) - F(\Phi_n)\bigr],

.. math::
   :label: phi

   \varphi_1(z) = \frac{e^z - 1}{z}, \qquad
   \varphi_2(z) = \frac{e^z - 1 - z}{z^2}, \qquad
   \varphi_1(0) = 1,\ \varphi_2(0) = \tfrac12 .

The three factors :math:`e^{hD}`, :math:`h\varphi_1(hD)`,
:math:`h\varphi_2(hD)` are vectors computed once per step by
:func:`~MCEq.solvers.numerics.phi_factors` and
:func:`~MCEq.solvers.numerics.diagonal_factors`. The step size is folded
into the :math:`\varphi` factors, so the predictor and the corrector are
products of three arrays, :data:`~MCEq.solvers.numerics.PREDICTOR_EXPR`
and :data:`~MCEq.solvers.numerics.CORRECTOR_EXPR`. Every backend lowers
these two strings: numpy (:func:`~MCEq.solvers.numerics.predictor`,
:func:`~MCEq.solvers.numerics.corrector`), C (the ``ETD2_PREDICT`` and
``ETD2_CORRECT`` macros of ``etd2_kernels.c``) and cupy (the kernel set of
the CUDA backend). The quotients in :eq:`phi` cancel as :math:`z \to 0`
and are replaced by Taylor series below :math:`|z| = 1.3\times10^{-4}`
(:math:`\varphi_1`) and :math:`6.3\times10^{-3}` (:math:`\varphi_2`).

Two limits determine the behaviour of the rule. For a strongly damped
component, :math:`hD_i \to -\infty`, the predictor returns the local
equilibrium :math:`a_i \to F_i(\Phi_n) / |D_i|`, so a short-lived species
leaves every step in equilibrium with its source, whatever the step. For
:math:`|hD_i| \ll 1` the rule is the trapezoidal rule for the source
term.

The step loop is :func:`~MCEq.solvers.etd2_driver`; it is the same
loop on every backend and for every route. A backend
(:class:`~MCEq.solvers.backends.host.HostBackend`,
:class:`~MCEq.solvers.backends.cuda.CudaBackend`) executes the stages: the
factor stage (``diag_factors``), the sparse product :math:`N x`
(``apply_off``), the predictor and the corrector. The diagonals and the
:math:`\varphi` factors are always formed in double precision, whatever the
precision of the state, see
:data:`~MCEq.solvers.backends.base._PRECISION_CONTRACT`.

In single precision the limit is the exponent range, not the mantissa. The
fluxes fall roughly as :math:`E^{-3}` over the twelve decades of the energy
grid and reach :math:`10^{-40}` near :math:`10^{10}` GeV at the surface,
below the smallest normal single-precision number; the products
:math:`N_{ij} x_j` inside the sparse product fall below it earlier, near
:math:`10^{7}` GeV. The equation is linear, so the driver integrates
:math:`s\Phi` with :math:`s` a power of two that puts :math:`\max|\Phi_0|`
at :math:`2^{76}` (:func:`~MCEq.solvers.etd2._fp32_scale`) and divides
:math:`s` out of the result; :math:`s` is exact and the headroom above,
about :math:`2^{52}`, covers the growth of the secondaries and the largest
operator entries. What remains is the rounding of the stored state, once
per stage. On the production 2D operator (0.05 GeV to :math:`10^{11}` GeV,
48 modes) the angle-integrated lepton fluxes then agree with double
precision below :math:`10^{10}` GeV to :math:`10^{-5}` at 0° (1100 steps),
:math:`2\times10^{-5}` at 60° (1600 steps) and :math:`3\times10^{-4}` at 90°
(18 800 steps); the error grows with the number of steps.

What restricts the step
-----------------------

The remainder :math:`N` is explicit, so a condition :math:`h\,\varrho(N)
\lesssim 2` remains. For the production kernels this allows steps of
hundreds of g/cm². The decay kernels carry large entries, but their rows
have correspondingly large negative diagonals and the integrating factor
damps them. In practice two other limits bind.

**The integration path.** Freezing :math:`\rho^{-1}` over a step is
accurate only if it changes little within it, so the step is taken from
the local logarithmic variation of the inverse density,

.. math::
   :label: step

   h_n = \min\!\left(h_{\max},\
     \frac{\epsilon}{|\mathrm{d}\ln\rho^{-1}/\mathrm{d}X|_{X_n}}\right),
   \qquad h_n \ge h_{\min},

with :math:`\rho^{-1}_n` the integral mean over the step
(:func:`~MCEq.solvers.etd2_nonuniform_path`,
``calculate_integration_path`` of :mod:`MCEq.driver.paths`). The defaults are
:math:`\epsilon = 0.03`, :math:`h_{\max} = 5` g/cm² for the
one-dimensional hadron and lepton transport and :math:`\epsilon = 0.01`,
:math:`h_{\max} = 2` g/cm² for the two-dimensional transport or retained
electron states, :math:`h_{\min} = 0.01` g/cm².

**The continuous-loss band.** The energy loss is an advection towards
lower energies, :math:`\partial_E[\mu_h \Phi_h] = E^{-1}\partial_u[\mu_h
\Phi_h]` in :math:`u = \ln E`, represented per species by the band
:math:`\mathrm{diag}(1/E)\, D_u\, \mathrm{diag}(\mu_h)` added to the
diagonal block of :math:`A`. Its diagonal joins :math:`D`, its off-diagonal
entries the explicit remainder. The interior rows of :math:`D_u` are
exponentially fitted (exact on power laws :math:`E^{-\alpha}` for
:math:`\alpha \in [1, 4]`, see :mod:`MCEq.operators.loss_stencil`); the
lowest rows are second-order upwind rows, the *closure*, whose number the
builder chooses per species so that every row whose explicit one-step
stiffness :math:`h\mu_h/(E\Delta u)` exceeds :math:`1/2` at a design
depth of 20 g/cm² is an upwind row
(:meth:`~MCEq.operators.matrix_builder.MatrixBuilder._upwind_rows_for`).
The largest off-diagonal row sum of the assembled bands, *excluding the
closure rows*, defines a rate :math:`r_{\mathrm{loss}}` and caps the step
at

.. math::
   :label: losscap

   h \le \frac{1/2}{r_{\mathrm{loss}}\,\max_j \lambda_j},

with :math:`\lambda_j` the eigenvalues of the :math:`\sec\theta` coupling
of the next section (one without it):
:func:`~MCEq.operators.stiffness.continuous_loss_rate`,
:data:`~MCEq.operators.stiffness.LOSS_STEP_SAFETY`,
:func:`~MCEq.driver.paths.continuous_loss_dx_cap`. The closure rows are
left out because they are sized for the design depth already and their
one-step map is bounded for any step under the diagonal-exact splitting;
their row sums fall as :math:`1/E` and measure accuracy, not stability.
This is a conservative policy, not a proof of stability of the full
non-normal operator.

Two-dimensional transport
=========================

In the two-dimensional version the state carries the angular distribution
with respect to the shower axis as :math:`n_k` Hankel modes
:math:`\kappa_k`; every mode obeys :eq:`linear` with its own kernels
:math:`A(\kappa_k)`, :math:`B(\kappa_k)`, and the modes are decoupled in
the paraxial approximation. The mode :math:`\kappa = 0` is the
angle-integrated flux. The only :math:`\kappa`-dependent diagonal entry is
the multiple-scattering damping of the muons,
:math:`-\kappa^2\theta_s^2(E)/4` (:mod:`MCEq.operators.scattering`), which
enters :math:`D` and is integrated exactly. The :math:`n_k` mode systems
are one block-diagonal system of dimension :math:`n_k N`, integrated as one
state array with the diagonal factors computed once.

The :math:`\sec\theta` path elongation in the eigenbasis
=========================================================

The two-dimensional equations book losses per unit of axis-projected
depth, while a particle at angle :math:`\theta` to the axis traverses
:math:`\sec\theta` times more matter. For populations in local
equilibrium (sub-GeV hadrons, muons) the paraxial equations overestimate
the wide-angle density by that factor. The correction multiplies the flux
entering the transport operator by :math:`g(\theta) = \min(\sec\theta,
\sec\theta_{\mathrm{cap}})`, a constant matrix :math:`S = I + T` on the
mode index in Hankel space:

.. math::
   :label: secant

   \frac{\mathrm{d}\tilde\Phi_k}{\mathrm{d}X}
   = \bigl[A(\kappa_k) + \rho^{-1}(X) B(\kappa_k)\bigr]
     \sum_{k'} S_{kk'}\,\tilde\Phi_{k'} .

:mod:`MCEq.operators.secant` builds :math:`T` by a regularised
least-squares fit and restricts it to the modes :math:`P` (:math:`\kappa
\le` ``row_kmax``) and to the state columns :math:`g` with :math:`E_{\rm
kin} <` ``config.secant_theta_e_max``. The elongation acts on the parent's
path length, not on the daughter's emission angle, so it must cover all
loss channels of a species or none; loss-free daughters such as neutrinos
are preserved.

Placing :math:`T` in the explicit remainder fails: in the stiff limit the
step for a fast species becomes a fixed-point iteration across modes with
iteration matrix :math:`D_S^{-1}(S - D_S)`, whose spectral radius exceeds
one at the production cap. The coupled loss term :math:`D_i S` has to be
integrated exactly.

The coupled block in the eigenbasis
-----------------------------------

Write the state as an :math:`n_k \times N` array and order the energy
columns so that the coupled columns :math:`g` come first
(:func:`~MCEq.operators.compiled.secant_layout`). The coupled modes
:math:`P` form a leading block of the mode axis, so the coupled part of
the state is the corner :math:`\mathcal{C}(\Phi) = \Phi_{P,g}`, a strided
view (:func:`~MCEq.operators.compiled.coupled_corner`). With :math:`S_P =
I + T_{PP}` the diagonal term of :eq:`secant` on the corner is

.. math::

   \frac{\mathrm{d}\Phi_{j,i}}{\mathrm{d}X}\Big|_{\mathrm{exact}}
   = D^0_i \sum_{j' \in P} (S_P)_{jj'}\,\Phi_{j',i}, \qquad j \in P,\ i \in g,

where :math:`D^0` is the diagonal of the :math:`\kappa = 0` operator,
common to all modes. :math:`S_P` acts on the mode index and :math:`D^0_i`
on the energy index, so they commute, and the eigendecomposition
:math:`S_P = V \mathrm{diag}(\lambda) V^{-1}` diagonalises the coupled
block. The corner of the state is carried in the eigenbasis,

.. math::

   \Psi = V^{-1}\,\mathcal{C}(\Phi),

where the exactly integrated operator is the diagonal

.. math::
   :label: cornerL

   L_{j,i} = \lambda_j D^0_i
           + \bigl[V^{-1}\mathrm{diag}(\Delta_{\cdot,i})\,V\mathrm{diag}(\lambda)\bigr]_{jj},
   \qquad \Delta_{j,i} = D^f_{j,i} - D^0_i,

with :math:`D^f` the full, :math:`\kappa`-dependent diagonal on the
coupled plane and :math:`\Delta` its :math:`\kappa`-dependent part, the
multiple-scattering damping of the muons
(:attr:`~MCEq.operators.compiled.CompiledOperator.exact_slot_diagonals`).
:math:`\Delta` does not commute with :math:`S_P`, so it cannot be
integrated exactly in this basis; its eigenbasis diagonal goes into
:math:`L` and only its mode mixing stays in the remainder. This matters
for stability. With all of :math:`\Delta` in the remainder, the one-step
map of the lowest muon bin of the production configuration (:math:`E =
45` MeV, :math:`\kappa_{\max} = 2000`, :math:`\sec\theta` capped at
3.86) has spectral radius 1.0 at :math:`h = 2` g/cm² and 1.36 at
:math:`h = 2.37`, where a horizontal solve at :math:`dX_{\max} = 3`
became nonfinite; with :eq:`cornerL` the radius is 0.50 at :math:`h = 2`
and 0.42 at :math:`h = 3`.
The corner therefore receives the same elementwise factors as the rest of
the state and the predictor and corrector :eq:`etd2` apply unchanged. Only
the remainder needs the coupling. The operand of the transport operator is
:math:`w = S\Phi`, which differs from :math:`\Phi` only on the corner,

.. math::

   \mathcal{C}(w) = V\mathrm{diag}(\lambda)\,\Psi + T_{P,P^c}\,\Phi_{P^c,g}
                  = W\,\mathcal{L}(\Phi), \qquad
   W = \bigl[\,V\mathrm{diag}(\lambda)\ \big|\ T_{P,P^c}\,\bigr],

with :math:`\mathcal{L}(\Phi) = \Phi_{\cdot,g}` the low-energy block over
every mode. :math:`V`,
:math:`V^{-1}`, :math:`\lambda` and :math:`W` are the ``coupling``
namespace of the compiled operator
(:func:`~MCEq.operators.compiled.secant_coupling`). The remainder on the
corner, in the eigenbasis, is everything that is not the exactly integrated
diagonal:

.. math::
   :label: cornerF

   \mathcal{C}(F) = V^{-1}\Bigl[(N w)_{P,g} + D^f_{P,g} \odot \mathcal{C}(w)\Bigr]
                    - L_{j,i}\,\Psi_{j,i},

with :math:`D^f` as in :eq:`cornerL`
(:attr:`~MCEq.operators.compiled.CompiledOperator.corner_diagonals`) and
:math:`L` the exact diagonal of :eq:`cornerL`.
Outside the corner :math:`F = Nw` as before. The step
(:func:`~MCEq.solvers.etd2_driver`, ``eval_F``) puts :math:`\mathcal{C}(w)`
in place of :math:`\Psi` for the sparse product, adds the diagonal term,
rotates by :math:`V^{-1}`, subtracts :math:`L\Psi` and restores
:math:`\Psi`. Per step this adds two dense products of the low-energy
block around each of the two sparse products. The corner is rotated into
the eigenbasis once before the loop and back once after it.

Several paths at once
=====================

The operator does not depend on the zenith angle; only the path
:math:`(h_n, \rho^{-1}_n)` does. The state therefore carries :math:`K`
columns, each following its own path, and all :math:`K` columns are
multiplied by the same sparse operator in one pass over its nonzeros. The
diagonal factors become :math:`(N, K)` arrays with
:math:`h` and :math:`\rho^{-1}` varying along the column axis. A column
with :math:`h = 0` has :math:`e^{hD} = 1`, :math:`h\varphi_{1,2} = 0` and is
left unchanged by the same arithmetic; finished
columns are read out and reloaded with the next path while the others
continue (:class:`~MCEq.solvers.schedule.CarouselSchedule`).

Summary of one step
===================

Stage by stage, with the backend method that executes it:

1. From the path take :math:`h_n` and :math:`\rho^{-1}_n`. Form
   :math:`D` over the full state (on the corner :math:`L_{j,i}` of
   :eq:`cornerL`) and the factors :math:`e^{hD}`, :math:`h\varphi_1`,
   :math:`h\varphi_2` (``diag_factors``).
2. Evaluate :math:`F(\Phi_n) = N\Phi_n` (``apply_off``). With the
   elongation, replace the corner by :math:`\mathcal{C}(S\Phi_n)` before
   the sparse product and finish the corner with :eq:`cornerF`.
3. Predictor :eq:`etd2`: :math:`a = e^{hD}\Phi_n + h\varphi_1 F(\Phi_n)`
   (``predictor``).
4. Evaluate :math:`F(a)` as in step 2.
5. Corrector :math:`\Phi_{n+1} = a + h\varphi_2[F(a) - F(\Phi_n)]`
   (``corrector``).
6. Record snapshots at requested depths, or exchange finished columns.

Accuracy
========

Three sources of step error remain once the diagonal is exact: the
:math:`h^2` terms of the explicit production and of the loss band, largest
for the low-energy muon spectrum; the frozen density, bounded by
:math:`\epsilon` in :eq:`step`; and the daughters of parents that decay
within a step, which see the parent only at the start and the predicted
end of the step. The last is a first-order error; in one dimension it
leaves a floor of about :math:`3\times10^{-3}` on the conventional and
prompt leptons that does not shrink with the step. The table gives the
worst relative step error :math:`|\Phi/\Phi_{\mathrm{ref}} - 1|` of
:math:`E^3\Phi` over the conventional and prompt muon and neutrino spectra
above 0.15, 1 and 10 GeV, in units of :math:`10^{-3}`. One dimension:
DPMJET-III 19.3, 60°, reference at :math:`h_{\max} = 0.25` g/cm². Two
dimensions: FLUKA, Hankel mode 0, :math:`\sec\theta` transport, 60°,
reference at 0.5 g/cm².

.. list-table::
   :header-rows: 1

   * - :math:`h_{\max}` (g/cm²)
     - one dimension: steps
     - ≥ 0.15 / 1 / 10 GeV
     - two dimensions: steps
     - ≥ 0.15 / 1 / 10 GeV
   * - 1
     - 2228
     - 3.75 / 3.18 / 3.18
     - 2546
     - 0.32 / 0.25 / 0.23
   * - 2
     - 1219
     - 4.34 / 3.23 / 3.23
     - 1584
     - 1.08 / 0.69 / 0.25
   * - 3
     -
     -
     - 1280
     - 1.85 / 1.07 / 0.36
   * - 5
     - 631
     - 5.36 / 3.12 / 1.78
     - 1056
     - 3.28 / 1.51 / 0.47
   * - 10
     - 448
     - 13.1 / 6.17 / 2.27
     - 908
     - 7.41 / 2.05 / 0.57

The default ceilings are 5 g/cm² (one dimension) and 2 g/cm² (two
dimensions).

References
==========

- S. M. Cox, P. C. Matthews, *Exponential time differencing for stiff
  systems*, J. Comput. Phys. 176 (2002) 430; M. Hochbruck, A. Ostermann,
  *Exponential integrators*, Acta Numerica 19 (2010) 209.
