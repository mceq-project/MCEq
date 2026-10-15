.. _solvers:

************************************************
solvers (:mod:`MCEq.solvers`)
************************************************
.. currentmodule:: MCEq.solvers


The module contains the integration kernels invoked by
:func:`MCEq.core.MCEqRun.solve()` and the path builder that chooses the
step sizes.

Numerical method
================

MCEq uses an exponential-time-differencing Runge-Kutta scheme (ETD2RK,
Cox–Matthews) as its only integrator. The cascade equation

.. math::

   \frac{\mathrm{d}\Phi}{\mathrm{d}X} = \bigl[A + \rho^{-1}(X)\,B\bigr]\,\Phi

is split into a diagonal (stiff) part :math:`D = \mathrm{diag}(A) +
\rho^{-1}\,\mathrm{diag}(B)`, integrated exactly through an integrating
factor, and an off-diagonal part
:math:`N = A_{\text{off}} + \rho^{-1}\,B_{\text{off}}`, advanced with an
explicit two-stage rule. A step evaluates :math:`F(\Phi) = N\Phi` twice,
against :math:`\Phi_n` and against the predicted state :math:`a`, each a
sparse product with :math:`A_{\text{off}}` and with :math:`B_{\text{off}}`,
plus elementwise operations on the state. The sec(θ) mode coupling is
carried in its eigenbasis. The formulas, the step-size rule and its caps
are on :ref:`solver-math`.

Since the diagonal is exact, the step size is not bound by the short
interaction and decay lengths. :func:`etd2_nonuniform_path` picks the steps
from :math:`|\mathrm{d}\ln\rho^{-1}/\mathrm{d}X|`;
``MCEqRun`` lowers its ceiling to the continuous-loss and EM caps of
:mod:`MCEq.solvers.path`.

Architecture
============

The solver is layered like the matrix build:

1. :class:`MCEq.core.MatrixBuilder` produces the constant operator as two
   sparse matrices, ``int_m`` (:math:`A`) and ``dec_m`` (:math:`B`);
   :mod:`MCEq.operators.secant` produces the constant sec(θ) mode-coupling
   operators.
2. :func:`MCEq.operators.compiled.compile_operator` assembles a
   :class:`~MCEq.operators.compiled.CompiledOperator`: the diagonal /
   off-diagonal split as CSR in the kernel's state layout (the low-E-first
   layout when the sec(θ) transport is on), the layout itself, and the
   coupling operators. It is host-only and backend-agnostic; every backend
   sums the same products in the same order.
3. :func:`make_backend` binds the compiled operator to the library of a
   kernel: :func:`numpy_backend` (scipy CSR SpMM), :func:`mkl_backend` (MKL
   sparse BLAS handles) and :func:`accelerate_backend` (Apple Accelerate
   sparse handles), the latter two row-major SpMM over the ``(dim, K)``
   state, or :class:`CudaBackend` (one CSR SpMM kernel for both operators,
   which keeps the sparsity pattern shared by the Legendre modes once). The
   backend owns the handles or device buffers and executes the stages of
   the step on its array module; nothing else differs between backends. On
   the host the sparse product is one ``apply_off`` binding per library:
   ``ScipyApplyOff``, and ``MklApplyOff`` and ``SpaccApplyOff`` on the shared
   ctypes binding ``CtypesApplyOff``.
4. :func:`etd2_driver` is the one step loop. It runs the single-axis solve
   (``K = 1``), the shared-path multi-RHS solve (``(dim, K)`` state, one
   path) and the LPT carousel (``(nsteps, K)`` per-lane paths with harvest /
   reset events, see :func:`schedule_lpt` and
   :func:`compile_carousel_schedule`), with or without the sec(θ) coupling.
   The stages of a step are listed on :ref:`solver-math`.

Precision
=========

The state is fp64 or fp32. The diagonals and the :math:`\varphi` factors
are always evaluated in fp64 and cast once to the state dtype; at fp32 the
driver integrates a power-of-two multiple of the state to keep the fluxes
clear of the subnormals. On CUDA, ``config.cuda_fp_precision`` sets the
state precision of ``solve()`` and the batch routes when the call names no
``dtype``, and ``config.cuda_operator_precision`` the precision of the stored
off-diagonal values (:class:`CudaBackend` ``operator_dtype``): ``None`` keeps
the state precision; 32 under an fp64 state stores the off-diagonals in fp32
while the state and the accumulation stay fp64, which halves the bytes the
sparse product streams per step. An operator precision above the state
precision raises ``ValueError``.

Caching and kernel selection
============================

:class:`MCEq.core.MCEqRun` caches one compiled operator per (matrices,
coupling) and one backend per (kernel, state dtype, coupling, operator
dtype); ``close()`` releases them. ``kernel_config`` selects the backend:
``numpy_etd2`` (always available), ``mkl_etd2`` (Linux/Windows when
``libmkl_rt`` is found), ``accelerate_etd2`` (macOS) and ``cuda_etd2``
(explicit opt-in only). ``"auto"`` tries Accelerate, MKL and numpy, in that
order. Every kernel runs :func:`etd2_driver`, so every route (single axis,
multi-RHS, the LPT carousel, the sec(θ) coupling) is available on all four,
in fp64 and fp32, in any combination.

:func:`solve_etd2` compiles the operator, binds the backend named by
``backend=`` and runs the driver, releasing the handles it created.
``MCEqRun`` caches its operator and backend instead; both get the backend
from :func:`make_backend`.

Sparse storage
==============

The off-diagonals are CSR on every backend. MKL and Accelerate run a
row-major SpMM over the C-ordered ``(dim, K)`` state. Accelerate is called
with ``ldb = ldc = K`` on the driver's own buffers; it accumulates without a
beta, so an overwritten output is zeroed first. At ``K = 1`` both libraries
run the SpMV.

Reference/API
=============
.. automodapi:: MCEq.solvers
  :inherited-members:

.. automodapi:: MCEq.solvers.path
  :include-all-objects:
  :no-inheritance-diagram:
  :skip: etd2_nonuniform_path, info

.. automodapi:: MCEq.solvers.numerics
  :include-all-objects:
  :no-inheritance-diagram:

.. automodapi:: MCEq.solvers.backends.host
  :include-all-objects:
  :no-inheritance-diagram:
