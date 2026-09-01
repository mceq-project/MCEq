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

Architecture
============

The solver is layered like the matrix build:

1. :class:`MCEq.core.MatrixBuilder` produces the constant operator as two
   sparse matrices, ``int_m`` (:math:`A`) and ``dec_m`` (:math:`B`);
   :mod:`MCEq.secant` produces the constant sec(θ) mode-coupling operators.
2. :func:`MCEq.operator_assembly.compile_operator` assembles a
   :class:`~MCEq.operator_assembly.CompiledOperator` — the diagonal /
   off-diagonal split as CSR in the kernel's state layout (the low-E-first
   layout when the sec(θ) transport is on), the layout itself, and the
   coupling operators. Host-only and backend-agnostic; every backend sums
   the same products in the same order, so the cross-backend agreement of
   the kernels (≤ 1e-11 relative in fp64) is a property of this object.
3. A backend binds the compiled operator to its library: :func:`numpy_backend`
   (scipy CSR SpMM), :func:`mkl_backend` (MKL sparse BLAS handles,
   row-major SpMM over the ``(dim, K)`` state) or :func:`cuda_backend`
   (cuSPARSE via cupy, fp64 or fp32 with fp64 diagonal factors). The
   backend owns the handles / device buffers and executes the stages of the
   step on its array module — nothing else differs between backends.
4. :func:`etd2_driver` is the one step loop. It runs the single-axis solve
   (``K = 1``), the shared-path multi-RHS solve (``(dim, K)`` state, one
   path) and the LPT carousel (``(nsteps, K)`` per-lane paths with harvest /
   reset events, see :func:`schedule_lpt` and :func:`compile_carousel_schedule`),
   with or without the sec(θ) coupling. Its docstring lists the numbered
   stages of a step.

:class:`MCEq.core.MCEqRun` caches one compiled operator per (matrices,
coupling) and one backend per (``kernel_config``, precision, coupling);
``close()`` releases them. ``kernel_config`` selects the backend:
``numpy_etd2`` (always available), ``mkl_etd2`` (Linux/Windows when
``libmkl_rt`` is found; the ``"auto"`` choice), ``cuda_etd2`` (explicit
opt-in) and ``accelerate_etd2`` (macOS; the ``"auto"`` choice there). The
Accelerate kernels (:func:`solv_spacc_etd2` and siblings) and the MKL fp32
multi-RHS kernel are not on the driver yet and keep their own step loops.

:func:`~MCEq.solvers.solve_etd2` is the entry point: it compiles the
operator, binds the backend named by ``backend=`` and runs the driver,
releasing the handles it created. ``MCEqRun`` does not use it — it caches
its own operator and backend and calls :func:`~MCEq.solvers.etd2_driver`
directly.

Sparse storage
==============

The off-diagonals are CSR on every backend. Block (BSR) storage was
measured on 2026-08-30 against the 1D SIBYLL and the 2D FLUKA operators:
on the 2D operators it is slower than CSR at every K; on the 1D operators
it gains 15–20 % on the single SpMV alone and loses at K > 1, so it was
dropped from the driver. MKL's row-major SpMM over the C-ordered ``(dim, K)``
state runs 1.3–2× faster than the column-major tiled SpMM the former
multi-RHS kernels used.

Reference/API
=============
.. automodapi:: MCEq.solvers
  :inherited-members:
