.. _solvers:
************************************************
solvers (:mod:`MCEq.solvers`)
************************************************
.. currentmodule:: MCEq.solvers


The module contains the integration kernels invoked by
:func:`MCEq.core.MCEqRun.solve()` and the path-builder used to choose
step sizes.

Numerical method
================

MCEq 2 uses an exponential-time-differencing Runge-Kutta scheme
(ETD2RK, Cox–Matthews) as its only integrator. The cascade equation

.. math::

   \frac{\mathrm{d}\Phi}{\mathrm{d}X} = \bigl[A + \rho^{-1}(X)\,B\bigr]\,\Phi

is split into a diagonal (stiff) part :math:`D = \mathrm{diag}(A) +
\rho^{-1}\,\mathrm{diag}(B)` treated exactly via an integrating
factor, and an off-diagonal (mild) part
:math:`N = A_{\text{off}} + \rho^{-1}\,B_{\text{off}}` advanced with
an explicit RK2 stage. Each step costs **4 sparse matrix-vector
products** (two evaluations of :math:`F(\Phi) = N\Phi`, one against
:math:`\Phi_n` and one against the intermediate stage :math:`a`)
plus a handful of element-wise vector ops on length-:math:`N` arrays.

The diagonal-exact treatment removes the explicit-stability bound
that forced ~10 :sup:`4` steps at high zenith in MCEq 1.x — the new
path-builder, :func:`etd2_nonuniform_path`, picks step sizes from
:math:`|\mathrm{d}\ln\rho^{-1}/\mathrm{d}X|` and ships the standard
atmosphere at θ = 89° in ~1300 steps. See
:doc:`/mceq_v1.x_v2_diff` for the full derivation, validation, and
the EM-cascade caveat.

Available kernels
=================

.. list-table::
   :header-rows: 1
   :widths: 20 18 12 50

   * - Kernel
     - Backend
     - Default
     - Notes
   * - :func:`solv_numpy_etd2`
     - scipy CSR / BSR
     - everywhere
     - Pure-Python fallback. Off-diagonals stored as BSR
       (:py:data:`config.numpy_bsr_blocksize` = 11) for ~2× faster
       SpMV than scipy CSR.
   * - :func:`solv_spacc_etd2`
     - Apple Accelerate (vecLib)
     - macOS
     - Picked by ``kernel_config = "auto"`` on macOS. Off-diagonals
       wrapped in :class:`~MCEq.spacc.SpaccMatrix`.
   * - :func:`solv_mkl_etd2`
     - Intel MKL sparse BLAS
     - Linux/Windows when ``libmkl_rt`` is found
     - Picked by ``kernel_config = "auto"`` when MKL is present.
       Off-diagonals stored as BSR
       (:py:data:`config.mkl_bsr_blocksize` = 6) — ~1.5× faster
       than MKL CSR; ~12× faster than ``numpy_etd2`` on the
       benchmark in §8.4.
   * - :func:`solv_cuda_etd2`
     - NVIDIA cuSPARSE via cupy
     - never auto-selected
     - Pick explicitly with ``config.kernel_config = "cuda_etd2"``.
       cuSPARSE BSR is not exposed by cupy 13, so this stays on
       CSR. Roughly on par with MKL for the SIBYLL21 system on a
       modern GPU; wins on much larger state vectors.

The auto-selection logic lives in :mod:`MCEq.config` (Apple
Accelerate → MKL → numpy in that order). Set
``config.kernel_config`` explicitly to override.

BSR off-diagonal storage
========================

The interaction matrix :math:`C` is **block-sparse with each non-empty
block (`n_E × n_E`) structurally upper-triangular** — kinematics
guarantees a particle of energy :math:`E` only produces secondaries
of energy :math:`E' \le E`. Storing the off-diagonals in BSR rather
than CSR lets scipy and MKL use vectorised dense-block microkernels
instead of the gather-scatter CSR loop. Empirically tuned defaults:

* ``numpy_bsr_blocksize = 11`` (scipy benefits from larger blocks;
  ``b = 11`` tiles the 121-bin macro-blocks neatly).
* ``mkl_bsr_blocksize = 6`` (MKL specialises its microkernel for
  ``b ∈ [2, 7]``; ``b ≥ 8`` falls into a generic path that's slower
  than CSR).

Both knobs accept ``None`` for a CSR fallback. Padding is handled
automatically (the matrix is zero-padded up to the next multiple of
``blocksize``; the kernel allocates working buffers at the padded
length).

Reference/API
=============
.. automodapi:: MCEq.solvers
  :inherited-members:
