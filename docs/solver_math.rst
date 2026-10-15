.. _solver-math:

*******************
Transport step math
*******************

.. currentmodule:: MCEq

The ETD2RK transport step of MCEq, formula by formula, with the function that
implements each.

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
holds the decay rates on the diagonal and the decay kernels off it. Both are
constant along the path, built by :class:`~MCEq.operators.matrix_builder.MatrixBuilder`; only
:math:`\rho^{-1}(X)` changes. Every short time scale sits on the diagonal.

Diagonal-exact splitting and the exponential integrator
=======================================================

Over one step :math:`[X_n, X_n + h]` the inverse density is replaced by
its step mean :math:`\rho^{-1}_n` and the operator is split,

.. math::
   :label: split

   A + \rho^{-1}_n B = \mathrm{diag}(D) + N, \qquad
   D = \mathrm{diag}(A) + \rho^{-1}_n \mathrm{diag}(B), \qquad
   N = A_{\mathrm{off}} + \rho^{-1}_n B_{\mathrm{off}}

(:func:`~MCEq.operators.compiled.split_diagonal` once,
:func:`~MCEq.solvers.numerics.diagonal_factors` per step). The variation of
constants gives the exact local solution

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
   \qquad \varphi_1(z) = \frac{e^z - 1}{z}, \quad
   \varphi_2(z) = \frac{e^z - 1 - z}{z^2}.

The vectors :math:`e^{hD}`, :math:`h\varphi_1(hD)`, :math:`h\varphi_2(hD)`
are computed once per step in double precision, whatever the state precision
(:func:`~MCEq.solvers.numerics.phi_factors`), with Taylor series below
:math:`|z| = 1.3\times10^{-4}` (:math:`\varphi_1`) and
:math:`6.31\times10^{-3}` (:math:`\varphi_2`). The predictor and the
corrector are then the elementwise expressions
:data:`~MCEq.solvers.numerics.PREDICTOR_EXPR` and
:data:`~MCEq.solvers.numerics.CORRECTOR_EXPR`, lowered by numpy
(:func:`~MCEq.solvers.numerics.predictor`,
:func:`~MCEq.solvers.numerics.corrector`), by the ``ETD2_PREDICT`` and
``ETD2_CORRECT`` macros of the C kernels and by the CUDA kernels.

For :math:`hD_i \to -\infty` the predictor returns the local equilibrium
:math:`a_i \to F_i(\Phi_n) / |D_i|`, whatever the step; for
:math:`|hD_i| \ll 1` the rule is the trapezoidal rule for the source.

In fp32 the high-energy fluxes would fall below the smallest normal number.
The equation is linear, so the driver integrates :math:`s\Phi`, :math:`s`
the power of two that puts :math:`\max|\Phi_0|` at :math:`2^{76}`
(``MCEq.solvers.etd2._fp32_scale``), and divides :math:`s` out exactly.

The step size
=============

The remainder :math:`N` is explicit, so :math:`h\,\varrho(N) \lesssim 2`
remains. The step is the smallest of three limits.

**The integration path.** :math:`\rho^{-1}` must change little within a
step:

.. math::
   :label: step

   h_n = \min\!\left(h_{\max},\
     \frac{\epsilon}{|\mathrm{d}\ln\rho^{-1}/\mathrm{d}X|_{X_n}}\right),
   \qquad h_n \ge h_{\min},

with :math:`\rho^{-1}_n` the integral mean over the step, and steps shortened
to land on the requested depths (:func:`~MCEq.solvers.etd2_nonuniform_path`).
The parameters are ``config.etd2_path`` (``eps``, ``dX_max``, ``dX_min``,
and ``fd_span`` for the derivative) or keywords of
:meth:`MCEq.core.MCEqRun.solve`. For ``None``,
``MCEqRun._calculate_integration_path`` selects :math:`\epsilon = 0.03`,
:math:`h_{\max} = 5` g/cm² in 1D hadron and lepton transport and
:math:`\epsilon = 0.01`, :math:`h_{\max} = 2` g/cm² with a 2D database or
electrons; :math:`h_{\min} = 0.01` g/cm². It lowers :math:`h_{\max}` to
the smaller of the two caps below.

**The continuous-loss band.** The energy loss is an advection towards lower
energies, :math:`\partial_E[\mu_h \Phi_h] = E^{-1}\partial_u[\mu_h \Phi_h]`
in :math:`u = \ln E`, the band :math:`\mathrm{diag}(1/E)\, D_u\,
\mathrm{diag}(\mu_h)` in the diagonal block of :math:`A`, with :math:`D_u`
from ``config.loss_stencil_method`` (:mod:`MCEq.operators.loss_stencil`).
With the default ``"expfit_low_upwind2"`` the interior rows are
exponentially fitted, exact on seven power laws :math:`E^{-\alpha}`,
:math:`\alpha \in [\alpha_0 - 2, \alpha_0 + 1]` (:math:`\alpha_0 =`
``config.loss_stencil_alpha0`` = 3), and
the lowest rows are second-order upwind rows, the *closure*. Per species,
the closure covers every row whose explicit one-step stiffness
:math:`h\mu_h/(E\Delta u)` at a design depth of 20 g/cm² reaches 1/2, and at
least ``UPWIND_ROWS_FLOOR`` = 8 rows (``MatrixBuilder._upwind_rows_for``);
closure rows exist only for the ``expfit_low_upwind*`` methods. The largest
off-diagonal row sum of the assembled bands, closure rows excluded, defines
a rate :math:`r_{\mathrm{loss}}` and caps the step at

.. math::
   :label: losscap

   h \le \frac{1/2}{r_{\mathrm{loss}}\,\max_j \lambda_j},

with :math:`\lambda_j` the eigenvalues of the :math:`\sec\theta` coupling
below (1 without it): :func:`~MCEq.solvers.path.continuous_loss_rate`,
:data:`~MCEq.solvers.path.LOSS_STEP_SAFETY`,
``MCEqRun._continuous_loss_dx_cap``. The closure rows are excluded because
their one-step map is bounded for any step under the diagonal-exact
splitting. The cap is a heuristic, not a stability proof.

**The EM cascade.** With ``config.em_adaptive_step`` (off by default) the
step is also capped at ``config.em_step_safety`` (0.12) divided by
:math:`r_{\mathrm{EM}}`, the spectral radius of the off-diagonal
:math:`e^\pm/\gamma` block of :math:`A` (mode :math:`l = 0` on a 2D
operator; dense ``eigvals`` up to ``config.em_step_dense_eig_max`` rows,
ARPACK above): :func:`~MCEq.solvers.path.em_step_scale`,
:func:`~MCEq.solvers.path.em_cascade_dx_cap`.

The step error comes from the :math:`h^2` terms of the explicit production
and loss band, the frozen density (:math:`\epsilon`) and the daughters of
parents that decay within a step, which see the parent only at the start
and the predicted end of it.

Two-dimensional transport
=========================

In the two-dimensional version the state carries the angular distribution
about the shower axis as :math:`n_k` Legendre moments :math:`a_l = \langle
P_l(\cos\theta)\rangle` on a grid of degrees :math:`l_k`, dense from 0 and
sparse above (:mod:`MCEq.operators.legendre`). A production, decay or
scattering kernel that depends only on the angle between parent and
secondary acts on the moments diagonally (Funk-Hecke theorem), so every
mode obeys :eq:`linear` with its own :math:`A(l_k)`, :math:`B(l_k)`; in the
paraxial approximation the modes are decoupled and form one block-diagonal
system of dimension :math:`n_k N`. The mode :math:`l = 0` is the
angle-integrated flux. The only :math:`l`-dependent diagonal entry is the
muon multiple-scattering rate :math:`-\Gamma(E, l)`
(:mod:`MCEq.operators.scattering`), integrated exactly as part of :math:`D`.

With ``config.legendre_backward_removal`` the off-diagonal production and
decay terms of the dense block of modes are multiplied by :math:`I - R`,
with :math:`R` the backward part of the Jackson-smoothed distribution
(:func:`~MCEq.operators.legendre.backward_removal`). The continuous-loss band
bypasses it, since energy loss keeps the direction.

The :math:`\sec\theta` path elongation
======================================

The two-dimensional equations book losses per unit of axis-projected depth,
while a particle at angle :math:`\theta` to the axis traverses
:math:`\sec\theta` times more matter; for populations in local equilibrium
(sub-GeV hadrons, muons) the paraxial equations overestimate the wide-angle
density by that factor. The correction multiplies the flux entering the
transport operator by :math:`g(\theta) = \min(\sec\theta,
\sec\theta_{\mathrm{cap}})`, :math:`\theta_{\mathrm{cap}} =`
``config.secant_theta_cap_deg`` (75°; 0 gives :math:`S = I`), a constant
matrix :math:`S = I + T` on the mode index, exact by Gauss-Legendre
quadrature on the two smooth pieces of :math:`g`
(:func:`~MCEq.operators.legendre.path_elongation`):

.. math::
   :label: secant

   S_{ll'} = \frac{2l' + 1}{2} \int_{-1}^{1} P_l(\mu)\, g(\mu)\,
   P_{l'}(\mu)\,\mathrm{d}\mu, \qquad
   \frac{\mathrm{d}\tilde\Phi_k}{\mathrm{d}X}
   = \bigl[A(l_k) + \rho^{-1}(X) B(l_k)\bigr]
     \sum_{k'} S_{kk'}\,\tilde\Phi_{k'} .

:mod:`MCEq.operators.secant` restricts :math:`T` to the rows :math:`P` of
the dense block of the :math:`l` grid, whose columns reach every mode
through the cubic interpolation in :math:`l`
(:func:`~MCEq.operators.legendre.l_interpolation`), and to the state columns
:math:`g` with :math:`E_{\rm kin} <` ``config.secant_theta_e_max``. The
elongation acts on the parent's path length, not on the daughter's emission
angle; it covers all loss channels of a species and preserves loss-free
daughters such as neutrinos. In the explicit remainder :math:`T` would make
the stiff limit a divergent fixed-point iteration across modes, so the
coupled loss term :math:`D_i S` is integrated exactly.

Write the state as an :math:`n_k \times N` array with the columns :math:`g`
first (:func:`~MCEq.operators.compiled.secant_layout`); the modes :math:`P`
lead the mode axis, so the coupled part of the state is the corner
:math:`\mathcal{C}(\Phi) = \Phi_{P,g}`, a strided view
(:func:`~MCEq.operators.compiled.coupled_corner`). With :math:`S_P = I +
T_{PP}` and :math:`D^0` the diagonal of the :math:`l = 0` operator, the
diagonal term on the corner is :math:`D^0_i \sum_{j'} (S_P)_{jj'}\Phi_{j',i}`.
:math:`S_P` acts on the mode index and :math:`D^0_i` on the energy index,
so a basis :math:`S_P \approx V\mathrm{diag}(\lambda)V^{-1}` diagonalises
the coupled block; the corner is carried as :math:`\Psi =
V^{-1}\mathcal{C}(\Phi)`.

``MCEq.operators.secant._canonical_basis`` builds :math:`V` from the
eigendecomposition of :math:`S_P`. Its eigenvalues must be real within
:math:`|\mathrm{Im}\,\lambda|/\mathrm{Re}\,\lambda \le 10^{-5}`. Directions
beyond the cap all see :math:`\sec\theta_\mathrm{cap}`, so :math:`S_P` has a
large near-degenerate cluster whose LAPACK basis depends on round-off.
Eigenvalues with consecutive relative gaps below :math:`3\times10^{-5}`
therefore form one cluster, spanned by the orthonormal invariant subspace of
an ordered real Schur form and rotated to diagonalise its projection of
:math:`\mathrm{diag}(l(l+1))`, the shape of the multiple-scattering damping.
:math:`\lambda = \mathrm{diag}(V^{-1}S_PV)`; the spread within a cluster
stays in the remainder. The exactly integrated operator is the diagonal

.. math::
   :label: cornerL

   L_{j,i} = \lambda_j D^0_i
           + \bigl[V^{-1}\mathrm{diag}(\Delta_{\cdot,i})\,V\mathrm{diag}(\lambda)\bigr]_{jj},
   \qquad \Delta_{j,i} = D^f_{j,i} - D^0_i,

with :math:`D^f` the full diagonal on the corner and :math:`\Delta` its
:math:`l`-dependent part, the muon multiple-scattering damping
(:attr:`~MCEq.operators.compiled.CompiledOperator.exact_slot_diagonals`).
:math:`\Delta` does not commute with :math:`S_P`; its diagonal in the basis
joins :math:`L`, which keeps the step stable, and only its mode mixing stays
in the remainder. A skewed basis with weights :math:`V^{-1}_{jl}V_{lj} < 0`
can make :math:`L_{j,i} > 0`, an exactly integrated growth.
:func:`~MCEq.operators.compiled.compile_operator` therefore switches to the
orthogonal real Schur basis :math:`S_P = QRQ^T` (:math:`V = Q`,
:math:`\lambda = \mathrm{diag}\,R`, the strictly upper part of :math:`R` in
the remainder) if the interaction part of :math:`L` has a positive maximum
in the eigenbasis and the Schur basis lowers that maximum
(:func:`~MCEq.operators.compiled.secant_coupling`).

The corner then receives elementwise factors built from :math:`L`, and the
predictor and corrector :eq:`etd2` apply unchanged; only the remainder needs
the coupling. The operand of the transport operator, :math:`w = S\Phi`,
differs from :math:`\Phi` only on the corner,

.. math::

   \mathcal{C}(w) = S_P V\,\Psi + T_{P,P^c}\,\Phi_{P^c,g}
                  = W\,\mathcal{L}(\Phi), \qquad
   W = \bigl[\,S_P V\ \big|\ T_{P,P^c}\,\bigr],

with :math:`\mathcal{L}(\Phi) = \Phi_{\cdot,g}` the low-energy block over
every mode. The remainder on the corner is everything except the exactly
integrated diagonal,

.. math::
   :label: cornerF

   \mathcal{C}(F) = V^{-1}\Bigl[(N w)_{P,g} + D^f_{P,g} \odot \mathcal{C}(w)\Bigr]
                    - L \odot \Psi ,

(:attr:`~MCEq.operators.compiled.CompiledOperator.corner_diagonals`), and
:math:`F = Nw` outside it. :math:`V`, :math:`V^{-1}`, :math:`\lambda` and
:math:`W` are the ``coupling`` namespace of the compiled operator. The
corner is rotated into the basis once before the step loop and back once
after it.

Several paths at once
=====================

The operator does not depend on the zenith angle, only the path
:math:`(h_n, \rho^{-1}_n)` does. The state carries :math:`K` columns, each
on its own path, multiplied by the same sparse operator in one pass; the
diagonal factors become :math:`(N, K)` arrays. A column with :math:`h = 0`
has :math:`e^{hD} = 1`, :math:`h\varphi_{1,2} = 0` and stays unchanged. In
the carousel, finished columns are read out and reloaded with the next path
while the others continue (:func:`~MCEq.solvers.schedule_lpt`,
:func:`~MCEq.solvers.compile_carousel_schedule`).

Summary of one step
===================

Stages of :func:`~MCEq.solvers.etd2_driver`, with the backend method
(:class:`~MCEq.solvers.backends.host.HostBackend`,
:class:`~MCEq.solvers.CudaBackend`):

1. Take :math:`h_n`, :math:`\rho^{-1}_n` from the path; form :math:`e^{hD}`,
   :math:`h\varphi_1`, :math:`h\varphi_2`, on the corner from :math:`L`
   (``diag_factors``).
2. :math:`F(\Phi_n) = N\Phi_n` (``apply_off``); with the elongation, the
   corner holds :math:`\mathcal{C}(S\Phi_n)` during the sparse product and is
   finished with :eq:`cornerF`.
3. :math:`a = e^{hD}\Phi_n + h\varphi_1 F(\Phi_n)` (``predictor``).
4. :math:`F(a)` as in stage 2.
5. :math:`\Phi_{n+1} = a + h\varphi_2[F(a) - F(\Phi_n)]` (``corrector``).
6. Record snapshots at requested depths, or exchange finished columns.

References
==========

- S. M. Cox, P. C. Matthews, *Exponential time differencing for stiff
  systems*, J. Comput. Phys. 176 (2002) 430; M. Hochbruck, A. Ostermann,
  *Exponential integrators*, Acta Numerica 19 (2010) 209.
