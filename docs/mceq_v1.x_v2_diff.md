# MCEq 2: Exponential Time-Differencing Solver and Continuous-Loss Stencil

This document describes the numerical changes between **MCEq 1.4.1** and
**MCEq 2** in the cascade-equation solver and the continuous-energy-loss
discretisation. It is written to be the source-of-truth reference for a
methods paper or extended scientific documentation: each section states
the underlying mathematics, the implementation choice in MCEq 2, the
reason behind the redesign, and the empirical validation that supports
it.

## Summary of changes

| Component                    | MCEq 1.4.1                                                                                      | MCEq 2                                                                                                |
| ---------------------------- | ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Time integrator              | Forward-Euler (explicit, 2 SpMV/step)                                                           | ETD2RK (Cox–Matthews exponential RK, 4 SpMV/step)                                                     |
| Stability bound              | `h · max(λ_int, ρ⁻¹·λ_dec) < margin` (diagonal-driven)                                          | `h · spec(N_off) < 2` (off-diagonal-only)                                                             |
| Step counts (θ=0/60/89°)     | ≈ 1.4·10³ / 2.9·10³ / 2.9·10⁴                                                                   | ≈ 8·10¹ / 1.3·10² / 1.3·10³                                                                           |
| Resonance approximation      | Energy-dependent mixing (`hybrid_crossover`, `mix_idx`, per-particle "hadronic ↔ resonance")    | Removed; every species propagated explicitly. ETD2's `exp(hD)` absorbs the stiffness automatically.   |
| Resonance opt-in             | n/a (always on)                                                                                 | `adv_set["force_resonance"]` (per-PDG escape hatch)                                                   |
| Path-step control            | Driven by leading-eigenvalue stability bound                                                    | Driven by `\|d ln ρ⁻¹/dX\|` (atmosphere-aware)                                                         |
| ρ⁻¹ sampling per step        | Point sample at `X_n`                                                                           | `scipy.integrate.quad`-averaged over `[X_n, X_{n+1}]`                                                  |
| Continuous-loss FD stencil   | 7-point banded; interior centred / one-sided biased polynomial                                  | 7-point banded; interior **exponentially fitted** (exact for `f = exp(α₀·u)`); boundaries unchanged    |
| Implementation (numpy)       | `solv_numpy` (Euler, in-place)                                                                  | `solv_numpy_etd2` (ETD2RK, preallocated scratch, in-place ufunc chain)                                 |
| Implementation (Accelerate)  | `solv_spacc_sparse` (Euler via Apple Accelerate)                                                | `solv_spacc_etd2` (ETD2RK; pre-split int_off / dec_off for SpMV reuse)                                 |
| Implementation (MKL / CUDA)  | `solv_MKL_sparse`, `solv_CUDA_sparse`                                                           | `solv_mkl_etd2` (Intel MKL sparse BLAS), `solv_cuda_etd2` (cuSPARSE via cupy)                          |
| Reduced state-vector default | full (all species)                                                                              | `disabled_particles = [11]` (drop e±) — see EM-cascade caveat in §7                                    |
| Headline wall-time speedup   | 1×                                                                                              | ≈ 9–11× across zenith range, sub-percent muon-flux agreement                                          |

The first half of this document (sections 1–3) develops the maths; the
second half (sections 4–6) presents the validation; sections 7–9
discuss limitations, implementation, and future work.

## 1. Cascade equation in slant depth

The MCEq state vector `Φ(X) ∈ ℝᴺ` indexes (particle species) × (kinetic
energy bin), with `N = n_species × n_E`. In slant depth `X` (g/cm²) the
cascade equation is

```
dΦ/dX = [ A + ρ⁻¹(X) · B ] Φ,                                       (1)
```

with constant sparse matrices `A = int_m` (interaction yields − cross-
section losses) and `B = dec_m` (decay yields − decay rates), and
`ρ⁻¹(X)` the depth-dependent inverse atmospheric density. Stiffness in
(1) lives almost entirely on the **diagonal**: per-energy interaction
loss `−σ_s/λ_int` from `A` and per-energy decay loss `−1/(γ_s τ_s ρ)`
from `B`. Off-diagonal entries (production / redistribution kernels)
are mild. This separation is what motivates the diagonal-exact ETD2RK
splitting of MCEq 2 (§3).

Throughout, we use the splitting

```
D(ρ⁻¹) = diag(A) + ρ⁻¹ · diag(B),         length-N vector
N(ρ⁻¹) = A_off  + ρ⁻¹ · B_off,            sparse, zero diagonal       (2)
```

so that `[A + ρ⁻¹·B] = diag(D) + N`.

## 2. MCEq 1.4.1 baseline

### 2.1 Forward-Euler integration

MCEq 1.4.1 advances (1) with explicit forward-Euler:

```
Φ_{n+1} = Φ_n + h_n · [A Φ_n + ρ⁻¹_n · B Φ_n]      (2 SpMVs / step)   (3)
```

Per-step cost: one sparse matrix–vector product against `A`, one
against `B`, one scalar multiply, one axpy. Implementation kernels in
1.4.1: `solv_numpy`, `solv_MKL_sparse`, `solv_CUDA_sparse`,
`solv_spacc_sparse`.

### 2.2 Stability bound and path control

The explicit stability region requires

```
h_n · max( λ_int(state), ρ⁻¹_n · λ_dec(state) ) < margin              (4)
```

where `λ_int = ‖diag(A)‖_∞` and `λ_dec = ‖diag(B)‖_∞`. In the upper
atmosphere (`X ≲ 10` g/cm²) the air density is small, so `ρ⁻¹` is
huge — at θ=89° it reaches ~10⁵ g⁻¹·cm³ — and the second term
dominates. The bound forces `h_n ∼ 10⁻³` g/cm² there. Total step
counts grow with `sec θ` and reach 28 800 at θ=89° on a typical
atmospheric path. `solve()` cost scales linearly with this count.

Path control was implemented in `_calculate_integration_path` and
governed by three configuration options (`leading_process`,
`stability_margin`, `dXmax`) that bounded `h_n` from above by a
fixed-margin estimate of (4). All three were removed in MCEq 2.

### 2.3 Resonance approximation

For short-lived species (charged π, K, charm), the decay term in (4)
forces `h_n` so small that the equilibrium

```
F(state) ≈ |D(state)| · state    ⇒    state ≈ F / |D|                 (5)
```

is reached well within one Euler step. Rather than resolving (5)
explicitly, MCEq 1.4.1 detected such species per energy bin and
*folded* their decay/interaction kernels into the matrices of their
parents at build time. The detector compared

```
λ_dec / λ_int  vs  config.hybrid_crossover     (default 0.5)
```

and split each particle's energy axis into a "resonance" range
(`residx = [0, mix_idx)`) where it was integrated out, and a "hadronic"
range (`hadridx = [mix_idx, n_E)`) where it propagated explicitly. The
folding lived in `MatrixBuilder._follow_chains` (recursive) and
`_fill_matrices` (top-level). Particles flagged `is_resonance = True`
had `mceqidx = -1` and never appeared in `Φ`.

The approximation traded resolved short-time dynamics for a smaller
state vector and per-step coupling. Empirical cost: a systematic
~0.3–0.7 % offset in the muon spectrum across all energies, traceable
to the discrete energy-bin boundary at `mix_idx`. The mixing logic
also drove ~80 LOC of bookkeeping spread across `particlemanager.py`,
`core.py`, and `config.py`.

### 2.4 Continuous energy loss

Continuous losses (ionisation, bremsstrahlung) for charged leptons and
hadrons are described by the Vlasov-like term

```
∂Φ_s/∂X = ⟨dE/dX⟩_s(E) · ∂Φ_s/∂E                                      (6)
```

per species `s`. On the log-energy grid `u = ln E`, this becomes
`∂Φ/∂X = (dE/dX / E) · ∂Φ/∂u`, discretised by a banded
finite-difference operator `D_u` for `∂/∂u`:

```
M_loss = diag(dE/dX(u_i) / E_i) · D_u                                 (7)
```

`M_loss` is added to `int_m` so that the same time-stepping handles
hadronic interaction, decay, and continuous loss uniformly.
`D_u` was a 7-point banded operator with **centred** or **biased**
polynomial-fit coefficients in 1.4.1. Order-of-accuracy on smooth
spectra is uniform across the interior; behaviour on boundary rows is
discussed in §5.3.

### 2.5 Limitations driving the redesign

1. **Step counts.** Forward-Euler at the diagonal-driven stability
   bound is bandwidth-bound on the cascade matrix and dominates
   `solve()` wall-time at all zeniths.
2. **Resonance-approximation systematic.** The 0.3–0.7 % offset is
   irreducible without tracking short-lived species explicitly, which
   the Euler bound makes prohibitively expensive.
3. **Stencil accuracy.** A polynomial-fit FD operator on `∂/∂u` is
   non-optimal for the locally exponential atmospheric flux
   `Φ ∼ E^{-α} = exp(-α·u)`. Targeting that exact functional form
   reduces interior error by O(10⁻³) (§5).
4. **Path control coupling.** Step-size selection driven by stability
   margins inside `MCEqRun` is opaque to the user and difficult to
   reason about; it should be an algorithmic property of the chosen
   integrator and atmosphere, not a hand-tuned `dXmax`.

## 3. MCEq 2: ETD2RK exponential integrator

### 3.1 Diagonal-exact splitting

Freeze `ρ⁻¹` over the step `[X_n, X_{n+1}]` and write (1) using (2)
with `D ≡ D(ρ⁻¹_n)`, `N ≡ N(ρ⁻¹_n)`:

```
dΦ/dX = D · Φ + N · Φ.                                                (8)
```

Variation of constants gives the *exact* local solution

```
Φ(X_n + h) = e^{h·D} Φ_n  +  ∫₀ʰ e^{(h−s)·D} N · Φ(X_n + s) ds.       (9)
```

Because `D` is diagonal, `e^{h·D}` is a length-N vector of scalar
exponentials. All stiffness from the diagonal of `A + ρ⁻¹·B` is
captured exactly by the integrating factor `e^{h·D}`; no scalar
explicit-stability bound applies along that axis.

### 3.2 ETD2RK update rule

We use the single-stage Cox–Matthews ETD2RK method
(Cox & Matthews 2002). Treating the off-diagonal source `F(state) =
N · state` with a 2-stage explicit Runge–Kutta scheme inside (9), the
update is

```
F(state) = N · state                            (= A_off·state + ρ⁻¹·B_off·state)
a        = e^{h·D} · Φ_n  +  h · φ₁(h·D) · F(Φ_n)                    (10)
Φ_{n+1}  = a            +  h · φ₂(h·D) · ( F(a) − F(Φ_n) )
```

with the *entire* functions

```
φ₁(z) = (e^z − 1) / z              (limit 1   as z → 0)              (11)
φ₂(z) = (e^z − 1 − z) / z²         (limit 1/2 as z → 0)
```

evaluated **elementwise** on the length-N vector `h·D`. The scheme is
locally O(h³) and globally O(h²) (Hochbruck & Ostermann 2010). Per-
step cost: 4 SpMVs (two `F` evaluations against `int_off` and
`dec_off`), one elementwise `exp`, two elementwise rationals
(`φ₁`, `φ₂`), and ~6 length-N axpy/scal ops.

### 3.3 Stiffness immunity along the diagonal

Two properties of the φ-functions are critical:

* For `Re z → −∞` (strongly-damped rows): `e^z → 0`, `φ₁(z) → −1/z`,
  `φ₂(z) → −1/z²`. Both factors *self-renormalise*. The stage `a`
  satisfies

  ```
  a[i] = e^{h·D[i]} Φ_n[i]  +  h · φ₁(h·D[i]) · F[i]
       → 0          +  h · ( -1 / (h·D[i]) ) · F[i]
       =  F[i] / |D[i]|.                                              (12)
  ```

  Equation (12) is exactly the equilibrium (5) that drove the
  resonance approximation in 1.4.1. ETD2RK reaches it *automatically*
  per step, no matter how large `h·|D|` becomes.

* For `|z| → 0` (weakly-damped rows): `φ₁(z) ≈ 1 + z/2 + z²/6`,
  `φ₂(z) ≈ 1/2 + z/6 + z²/24`. The scheme degenerates smoothly to a
  classical RK2 on the off-diagonal source.

### 3.4 Removal of the resonance approximation

The full resonance machinery (§2.3) was removed in MCEq 2 because (12)
makes it numerically redundant: feeding a short-lived species through
the explicit cascade gives exactly the equilibrium value (12) per step,
to machine precision.

Empirical confirmation: with `adv_set["no_mixing"] = True`
(the legacy switch that disabled mixing), every species was forced
into the cascade. `λ_dec` jumps from ~5·10⁻⁵ to ~2.4·10⁵ on the
diagonal — nine orders of magnitude — and ETD2 absorbs it cleanly.
Muon flux changes by 0.3–0.7 %, which is the built-in error of the
1.4.1 approximation, not a regression.

`adv_set["force_resonance"]` is retained as an explicit per-PDG opt-in
(empty list by default). PDGs listed there are folded the same way the
old global mechanism did, via a one-line `is_resonance` flag set in
`MCEqParticle._apply_force_resonance`. `_follow_chains` and
`_fill_matrices` retain the recursion branch that handles such
particles.

### 3.5 What sets the remaining stability bound

The off-diagonal block `N = A_off + ρ⁻¹·B_off` is treated explicitly,
so a residual stability bound remains:

```
h · spec(N) < 2.                                                      (13)
```

`spec(int_off) ≈ 0.094` (ρ⁻¹-independent), `spec(dec_off, no ρ⁻¹) ≈
10⁻³`. The interaction off-diagonal dominates, giving

```
h_max ≈ 2 / spec(int_off) ≈ 21 g/cm²                                  (14)
```

zenith-independent. The decay off-diagonals — which contain entries up
to 5·10⁴ from short-lived decays such as `π⁰ → γγ` — do *not* set
(14), because the rows carrying those entries have huge negative
diagonal damping in `D`, absorbed in `e^{h·D}` (§3.3).

**Source of the 21 g/cm² ceiling.** A right/left eigen-pair analysis
of `int_off` reveals that ~99.9 % of `‖λ_max‖` mass sits in the
within-particle muon-band sub-blocks of `M_loss` (§2.4), specifically
in the ±1-bin off-diagonals of the dE/dX FD stencil at the lowest E
bins (89 MeV — 141 MeV). Per-channel decomposition (SIBYLL23D, full
DB):

| Channel (parent → child, summed over E)              | % of λ_max |
| ---------------------------------------------------- | ---------- |
| `prres_mu⁻ → prres_mu⁻` (tracking-alias band)        | 36.6 %     |
| `mu⁻ → mu⁻`                                          | 23.2 %     |
| `mu⁺ → mu⁺`                                          | 23.2 %     |
| `prres_mu⁺ → prres_mu⁺`                              | 16.9 %     |
| All cross-species (true hadronic) channels combined  | <0.1 %     |

Toggling `enable_energy_loss`:

| `enable_energy_loss` | `spec(int_off)` | `h_max = 2 / spec(int_off)` |
| -------------------- | --------------- | --------------------------- |
| `True`  (default)    | 0.0934          | **21.4 g/cm²**              |
| `False`              | 0.0079          | **251.7 g/cm²**             |

The `~12×` ratio shows that the cliff is entirely an artifact of the
finite-difference discretisation of continuous energy loss, not of the
hadronic physics. The muon dE/dX FD stencil amplitude blows up at the
low-E edge because `dE/dX ≈ const` while the bin width in `ln E` is
fixed, so `dE/dX / E` (the FD coefficient after the `∂/∂u` change of
variable) is largest at the bottom of the grid. This motivates the
block-ETD proposal of §8.3.

## 4. Step-size control

### 4.1 Path control concept

Given (14), the stability ceiling on `h` is *not* the binding
constraint in practice for MCEq 2: the binding constraint is
**accuracy** of the per-step `e^{h·D}` snapshot, since `D` depends on
`ρ⁻¹(X)` which varies by ~7 orders of magnitude across the upper
atmosphere. Freezing `ρ⁻¹_n` at the start of the step is correct only
if `ρ⁻¹` varies little within `[X_n, X_{n+1}]`.

The MCEq 2 path builder picks `h_k` from an **atmosphere-aware**
tolerance:

```
h_k = clip( ε / |d ln ρ⁻¹/dX|_at_X_k ,  h_min ,  h_max )              (15)
```

with defaults `ε = 0.3`, `h_max = 20` g/cm² (just under the cliff
(14)), `h_min = 0.01` g/cm². Smaller `ε` ⇒ finer steps in the upper
atmosphere where `ρ⁻¹` varies fastest.

The local rate `|d ln ρ⁻¹/dX|` is estimated by a fixed-span forward
finite difference

```
rate(X) = | ln ρ⁻¹(X + δ) − ln ρ⁻¹(X) | / δ ,    δ = 0.01 g/cm²       (16)
```

The fixed `δ` is critical at `X = X_start = 0`, where the CORSIKA
density spline saturates to `ρ⁻¹ ≈ 10⁹` for `X ≲ 10⁻⁴` g/cm²; a
relative- or centred-FD reads zero variation there and would pick
`h = h_max`, missing the `ρ⁻¹` drop in the first 0.01 g/cm². The
fixed-`δ` FD always crosses the saturation cap and returns a finite
gradient.

### 4.2 ρ⁻¹ averaging convention

`ρ⁻¹_n` is the value frozen in (8). Three conventions were tested:

* **Start-of-step**: `ρ⁻¹_n = ρ⁻¹(X_n)`. Adequate for forward-Euler at
  `h ≈ 1` g/cm² (1.4.1 convention). At ETD2's coarser steps `h ≥ 5`
  g/cm² it produces high-E muon flux **+3700 %** over Euler@native at
  `h = 25` g/cm², because `ρ⁻¹` falls roughly exponentially across the
  step in the upper atmosphere (`α·h ≈ 9` at `h = 10` means
  `ρ⁻¹(start)` is ~6500× the value at end-of-step).

* **Mid-point**: `ρ⁻¹_n = ρ⁻¹(X_n + h/2)`. Geometric mean of endpoints
  for exponential profiles. High-E error drops to ~0.2 % at `h = 25`
  g/cm². Adequate for *uniform* stepping at `h ≤ 25` g/cm².

* **`scipy.integrate.quad`-averaged** (MCEq 2 default):

  ```
  ρ⁻¹_n = (1/h) · ∫_{X_n}^{X_n+h} r_X2rho(X′) dX′                     (17)
  ```

  Handles the saturation boundary in the very first step correctly
  (where `ρ⁻¹` drops 7 orders of magnitude across `δX ≈ 10⁻²` g/cm²)
  and runs in ~1 ms per step (~130 ms for a typical 130-step path —
  negligible against ~700 ms of `solve` time).

The path builder uses (17) by default. The implementation lives in
`MCEq.solvers.etd2_nonuniform_path`.

### 4.3 User-supplied snapshot grid

User-supplied `int_grid` (an array of X values where the solution
should be recorded) is interleaved with the natural schedule (15):
each requested snapshot point forces a step-boundary truncation, so
`len(int_grid) = 10⁴` produces `≥ 10⁴` steps with `grid_idcs`
recording each one. The path cache invalidates when any of
`X_start`, `eps`, `dX_max`, `dX_min`, `fd_span`, or `int_grid` change.

## 5. Exponentially-fitted continuous-loss stencil

### 5.1 Banded operator for ∂/∂u

`MatrixBuilder._construct_differential_operator` builds a 7-point
banded matrix `D_u` approximating `∂/∂u` on the log-uniform grid
`u_i = ln E_i` of constant spacing `Δu`. `cont_loss_operator` then
left-multiplies by `1/E` and right-multiplies by `dE/dX` to assemble
(7), with `M_loss` added into `int_m`.

The interior stencil has bandwidth 7. Three modes are available via
`config.loss_stencil_method`:

* `"centered"` — symmetric polynomial-fit, 6th-order interior
* `"biased"`   — one-sided polynomial-fit (the 1.4.1 default behaviour)
* `"expfit"`   — *exponentially fitted* (MCEq 2 default)

Boundary rows (0, 1, 2 and `last−2, last−1, last`) use one-sided
polynomial-fit stencils (4th-order at the corner, 5th–6th-order one
row in) that are identical across all three methods.

### 5.2 Why exponential fitting

The atmospheric flux is locally well-approximated by a power law in `E`,
which is a *pure exponential* in `u = ln E`:

```
Φ_s(u) ~ exp(−α_s · u)     ⇔    Φ_s(E) ~ E^{-α_s}.                    (18)
```

with species-dependent slope. A polynomial-fit FD stencil has its
maximum error precisely on functions of the form (18) — the leading
Taylor truncation `Φ⁽⁷⁾(u)` is non-zero. An *exponentially-fitted*
stencil chooses the seven row coefficients `c_{-3..+3}` to satisfy the
seven equations

```
∑_k c_k · exp(−α₀ · k · Δu)  =  −α₀                                   (19)
∑_k c_k · k^p             =  δ_{p,1}        (p = 0, …, 5)
```

i.e. exactness for `f = exp(−α₀ · u)` *and* polynomial exactness up to
order 5. With `α₀ ≈ 3` (the typical high-E muon-flux slope), the
interior `expfit` stencil achieves `5·10⁻¹⁶` relative error on `f
= exp(−3·u)` — round-off limited, as opposed to ~10⁻⁴ for `centered`
on the same trial function. The trade-off is that for spectra with
local slope `α ≠ α₀`, `expfit` is *less* accurate than `centered` on
the polynomial part, but the gain on the exponential part dominates
in atmospheric-flux applications.

Default `config.loss_stencil_alpha0 = 3`. The choice is justified by
slope measurements on production fluxes (GSF + DPMJET-III v193, θ=0°):

| species  | local α at E = 10² GeV | local α at E ≳ 10⁹ GeV |
| -------- | ---------------------- | ---------------------- |
| μ±        | 1.7                    | 3.4                    |
| ν_μ / ν̄_μ | 1.0                    | 3.1 / 3.2              |
| ν_e / ν̄_e | 0.55                   | 3.1 / 3.0              |

`expfit(α₀=3)` is exact for the high-E asymptote and degrades smoothly
toward the low-E end where `α` is small. The interior gain on full-
state atmospheric flux is ~10⁻³ relative against `biased`.

### 5.3 Boundary-row error

Boundary rows use one-sided polynomial-fit stencils that differ across
all three methods. On a synthetic steep test function `f = E⁻³`
(10 bpd grid):

| row         | E (GeV)  | rel. err on E⁻³        |
| ----------- | -------- | ---------------------- |
| 0           | 0.09     | −3.8 %                 |
| 1           | 0.13     | +0.7 %                 |
| 2           | 0.16     | −0.2 %                 |
| 3 .. last−3 | bulk     | 5·10⁻¹⁶ (`expfit`)     |
| last − 2    | 6.3·10¹⁰ | +0.4 %                 |
| last − 1    | 7.9·10¹⁰ | +2.2 %                 |
| last        | 1.0·10¹¹ | **+20 %**              |

The boundary stencil is exact for constants (row sums = 0) and gives
small residuals on flat spectra. Real atmospheric muon spectra at
0.1 GeV have local slope `α ≈ 0` (rising/flat — see table in §5.2),
so the bottom-row error is small in practice. The high-E cliff
(+20 % at the last bin on `E⁻³`) is on a bin with essentially zero
physical flux.

An `expfit_ghost` variant that applied the full 7-point stencil to
every row, folding out-of-grid columns onto the diagonal via the
analytic continuation `Φ_{i+k} ≈ Φ_i · exp(−α₀ · k · Δu)`, was
investigated and rejected: a single global `α₀` gives a one-sided
slope assumption that doesn't match real spectra at both edges. With
`α₀ = 3`, μ± at low E shifts by **−8.1 %** at `i = 0` and **+13.2 %**
at `i = 1` (oscillation pattern, classic stencil-misspecification
signature). A proper fix requires per-edge — and ideally per-particle
— `α₀`. Open issue tracked as the *boundary-row "cliff"* in §8.

## 6. Validation

All measurements below use SIBYLL-2.3D, GSF / Hillas–Gaisser primary,
the test database `mceq_db_v140reduced_compact.h5`, and the
session-fixture `mceq_sib21` (which keeps e± enabled).

### 6.1 Convergence study

ETD2RK is locally O(h³), globally O(h²). Both schemes are refined
together; the truth is forward-Euler oversampled 16× per native step
(h held effectively constant within each native segment). Order is
the observed log₂ of the error ratio between successive halvings of h.

| 1/h     | ‖Euler − truth‖ | ‖ETD2 − truth‖ | Euler order | ETD2 order    |
| ------- | --------------- | -------------- | ----------- | ------------- |
| native  | 2.32·10⁻⁴       | 3.96·10⁻⁵      | —           | —             |
| /2      | 1.08·10⁻⁴       | 2.10·10⁻⁵      | 1.10        | 0.92          |
| /4      | 4.64·10⁻⁵       | 1.68·10⁻⁵      | 1.22        | 0.32          |
| /8      | 1.55·10⁻⁵       | 1.58·10⁻⁵      | 1.58        | (truth-floor) |

ETD2's "order drop" at /4 and /8 is the truth-reference's own
discretisation floor — ETD2 has converged below `‖Euler@oversample=16
− exact ODE‖`. Full-state μ-flux error of Euler @ native vs truth is
**0.37 %**; ETD2 @ native is **0.025 %** — ETD2 is ~14× more accurate
at the same step grid.

A separate convergence test on a `GeneralizedTarget` water column
(constant density, no `ρ⁻¹` variation) confirms clean O(h²) under
uniform stepping: rel-L2 vs `h = 0.5` reference is 9.3·10⁻³ at
`h = 20`, dropping by 4× per halving of `h`.

### 6.2 Coarsening sweep (Euler vs ETD2 head-to-head)

`config.adv_set["disabled_particles"] = [11, -11]` (drop e±; see §7).
Euler at native step grid is the reference; "rel diff" is
scheme-vs-Euler@native, **not** vs converged truth (Euler@native
itself sits ~0.4 % off truth at vertical). `C` is the coarsening
factor (native step count divided by `C`).

θ = 0° (Euler @ native = 4140 ms reference, 0.37 % off truth):

| C  | Euler ms | Euler μ rel diff | ETD2 ms  | ETD2 μ rel diff |
| -- | -------- | ---------------- | -------- | --------------- |
| 1  | 4140     | baseline         | 8521     | 0.38 %          |
| 2  | 2076     | 0.19 %           | 4279     | 0.57 %          |
| 4  | 1057     | 💥 unstable      | 2128     | 0.97 %          |
| 8  | 524      | 💥               | **1057** | **1.02 %**      |
| 16 | 265      | 💥               | 532      | 1.22 %          |
| 32 | 130      | 💥               | 270      | 5.11 %          |

θ = 60° (Euler @ native = 8345 ms reference):

| C  | Euler ms | Euler μ rel diff | ETD2 ms  | ETD2 μ rel diff |
| -- | -------- | ---------------- | -------- | --------------- |
| 1  | 8345     | baseline         | 17 191   | 0.30 %          |
| 2  | 4205     | 0.29 %           | 8529     | 0.30 %          |
| 4  | 2086     | 💥               | 4272     | 0.32 %          |
| 8  | 1048     | 💥               | 2142     | 0.39 %          |
| 16 | 522      | 💥               | **1066** | **0.60 %**      |
| 32 | 263      | 💥               | 541      | 2.23 %          |

θ = 89° (Euler @ native = 83 s reference):

| C      | nsteps   | Euler ms | Euler μ rel diff | ETD2 ms      | ETD2 μ rel diff |
| ------ | -------- | -------- | ---------------- | ------------ | --------------- |
| 1      | 28872    | 86 849   | baseline         | 177 905      | 0.17 %          |
| 2      | 14436    | 43 259   | 0.17 %           | 87 787       | 0.17 %          |
| 4      | 7218     | 💥       | 💥               | 43 738       | 0.17 %          |
| 8      | 3609     | 💥       | 💥               | 21 784       | 0.22 %          |
| 16     | 1804     | 💥       | 💥               | 10 851       | 0.69 %          |
| **20** | **1443** | 💥       | 💥               | **8687**     | **1.13 %**      |
| 32     | 902      | 💥       | 💥               | 5446         | 💥              |

**Headline numbers**:

* ETD2 at native grid is ~14× more accurate than Euler at native
  (vs converged truth at θ=0°), at 2× the per-step cost.
* ETD2 @ C=2 ≈ Euler @ native in both wall-time and accuracy.
* 4–8× speedup at θ=0° (C=8) and 16× at θ=60° (C=16) while keeping
  μ-flux rel diff ≤ 1 %. Speedup grows with zenith.
* **9.6× speedup at θ=89°** with 1.13 % μ-flux rel diff (`C=20`,
  1443 steps — same step count as vertical).
* Hard cliff at `C ≈ 32`: explicit-stage stability bound (14) on the
  off-diagonal coupling.

### 6.3 Production validation: atmospheric muon flux

Validated muon-flux agreement vs Euler@native, sub-percent across
10 GeV – 100 TeV at all zenith angles, using the production
ρ-aware non-uniform path (15)–(17) with `dX_max = 20`:

| θ   | n_eul   | n_etd2 | wall-time speedup |
| --- | ------- | ------ | ----------------- |
| 0°  | 1450    | 81     | 9.5×              |
| 60° | 2862    | 133    | 11.3×             |
| 89° | 28 839  | 1325   | 11.4×             |

Speedup is roughly zenith-independent (Euler steps grow with `sec θ`,
ETD2 steps grow with the path length over a roughly constant local
scale-height-derived `h`). Step counts scale with the **path's total
log-density variation** rather than its stiffness. Numbers above were
captured during the MCEq 1.4.1 → MCEq 2 transition while both schemes
still coexisted on the development branch; a side-by-side comparison
notebook lived at `docs/examples/ETD2_solver_comparison.ipynb` and was
retired together with the forward-Euler kernels (recoverable from git
history before commit `cf563d9` if the head-to-head plots are needed).

### 6.4 Cross-backend equivalence

The Apple-Accelerate kernel `solv_spacc_etd2` is bit-equivalent to
`solv_numpy_etd2` to ~10⁻¹² relative L2 (machine precision) on real
SIBYLL matrices at θ=60°, uniform `h = 5` g/cm² mid-point sampled
path. Verified in `tests/test_solvers.py::test_solv_spacc_etd2_matches_numpy_etd2_real`.

## 7. EM cascade caveat

`solv_*_etd2` has no protective damping for state-vector rows whose
matrix-diagonal is small but whose off-diagonal source scales with
`ρ⁻¹`. In MCEq this affects the e± semi-Lagrangian L/R variants
(`e+_l, e+, e+_r, e-_l, e-, e-_r`) at very large `ρ⁻¹` (top-of-
atmosphere near horizon). With those rows present, ETD2 produces
`+inf`/`NaN` in the EM block at θ ≈ 90°. The contamination is
contained — e±/γ do not feed back into hadrons or leptons via
`int_m`/`dec_m` — so muon and neutrino fluxes are unaffected, but the
per-step ufunc machinery emits overflow / invalid-value warnings.

The kernels suppress those warnings via `np.errstate(over="ignore",
invalid="ignore")` around the per-step loop. The MCEq 2 default
configuration sets `adv_set["disabled_particles"] = [11]` (which
also disables −11 via the data-backend's symmetric filter), so by
default the e± rows are simply absent from the state vector. Photons
remain (π⁰ → γγ depends on them); they have no L/R variants and are
unaffected.

A future *block-ETD* extension treating each `(species, E_bin) →
{main, _l, _r}` triple as a 3×3 block exp would lift this restriction
without material per-step cost (§8.3).

## 8. Limitations and future work

### 8.1 Boundary-row "cliff" in `M_loss`

(§5.3.) Open issue. A proper fix requires per-edge — and ideally
per-particle — `α₀_low ≈ 0` and `α₀_high ≈ 3`, or an iterate-from-
previous-`Φ` adaptive scheme that reads the local slope at each edge.
The polynomial-fit boundary used now is acceptable in practice
because real low-E muon spectra have `α ≈ 0` (the polynomial fit is
exact for constants, row sums = 0).

### 8.2 EM cascade L/R variants

(§7.) The 3×3 block at each `(species, E_bin)` admits an *exact*
`exp(h·K_3×3)` factorisation; treating it as one `D_block` element
would absorb the missing diagonal damping and remove the
high-zenith e± instability. The block-ETD §8.3 framework is the same.

### 8.3 Block-ETD for the muon dE/dX cliff

(§3.5.) Lift the muon dE/dX FD stencil into the "exact" half of the
splitting:

```
M(X) = D_block + N_resid

D_block = block-diagonal:
    • banded n_E × n_E block per muon-like species
      (diag(d_int + ρ⁻¹·d_dec)[slc] + M_band_k)
    • scalar diagonal for every other row
N_resid = M − D_block                                                 (20)
```

The ETD2RK update (10) is unchanged in form — just with
matrix-valued φ-functions on each muon block. Per-block primitive
(via eigendecomposition on small `n_E × n_E ≈ 31×31`):

```python
def _block_factors(K, h, PHI1_SMALL=1e-6, PHI2_SMALL=1e-3):
    """Return matrices exp(hK), φ₁(hK), φ₂(hK) via eigendecomposition.
    One eig() per block per step; ~30 µs each, negligible vs SpMV cost."""
    Lam, U = np.linalg.eig(K)
    Uinv = np.linalg.inv(U)
    hLam = h * Lam
    eL = np.exp(hLam)
    phi1_L = np.where(np.abs(hLam) > PHI1_SMALL,
                      (eL - 1.0) / np.where(hLam != 0, hLam, 1.0),
                      1.0 + 0.5*hLam + hLam*hLam/6.0)
    phi2_L = np.where(np.abs(hLam) > PHI2_SMALL,
                      (eL - 1.0 - hLam) / np.where(hLam != 0, hLam*hLam, 1.0),
                      0.5 + hLam/6.0 + hLam*hLam/24.0)
    return (U * eL) @ Uinv, (U * phi1_L) @ Uinv, (U * phi2_L) @ Uinv
```

Cost / benefit estimate: 0.6 ms/step extra (20 small `eig`'s), step
count drops ~4× as `dX_max` rises from 20 to 200 g/cm². ~3× total
wall-time win on top of the in-place fix already shipped, with bigger
payoff at high zenith. Suggested rollout (no behaviour change until
step 4):

1. Land `dEdx_band` as a separate attribute on `MatrixBuilder`,
   keeping the `int_m = int_m_hadr + dEdx_band` invariant.
2. Add `_etd_split_block_cache(int_m, dec_m, dEdx_band)` returning
   `(d_int, d_dec, int_off_resid, dec_off, blocks)`.
3. Add `solv_numpy_etd2_block` next to `solv_numpy_etd2`. Both
   produce bit-identical output when `dEdx_band == 0`.
4. Wire behind `kernel_config = "numpy_etd2_block"`. Run zenith
   sweep; confirm `dX_max = 100` works cleanly.
5. Promote `numpy_etd2_block` to be the default `numpy_etd2`. Same
   for spacc.

Same machinery handles the 3×3 EM L/R blocks (§8.2): only the
`_extract_blocks` predicate differs.

### 8.4 MKL and CUDA kernels

`solv_mkl_etd2` and `solv_cuda_etd2` mirror `solv_spacc_etd2` (the
worked ctypes/native-BLAS example) and `solv_numpy_etd2` (the
mathematically clearest reference). Both:

* take pre-built backend handles for `int_off` and `dec_off` (the
  caller, `MCEqRun._build_kernel_dispatch`, materialises and caches
  them once per matrix rebuild),
* perform 4 SpMVs per step against those handles, sharing the
  diagonal-factor compute (`_etd_compute_diag_factors`) with the
  numpy kernel on host or the GPU equivalent
  (`_cuda_compute_diag_factors`) on device,
* maintain ctypes / cupy-array buffers in place across the loop —
  rebinding `phc` / `F_phi` / `F_a` / `a` would silently break the
  pointer/handle contract.

All three CPU kernels (numpy, MKL, spacc) store the off-diagonals as
**BSR rather than CSR** by default, with backend-specific block sizes
tuned empirically:

* **numpy** uses `blocksize=11` (`config.numpy_bsr_blocksize`). scipy's
  BSR matvec is a generic C++ template; it benefits from larger blocks
  than MKL's because the per-block overhead amortises better, and
  `b=11` happens to tile the 121-energy-bin macro-blocks neatly
  (`121 = 11²`). Real-world wall-clock improvement on SIBYLL21:
  **~2.0× over scipy CSR**.
* **MKL** uses `blocksize=6` (`config.mkl_bsr_blocksize`). MKL appears
  to specialise its BSR microkernel for `b ∈ [2, 7]`; `b ≥ 8` falls
  into a generic path that's slower than CSR for these matrices.
  Real-world improvement: ~1.5× over MKL CSR for the kernel itself.

The numpy kernel memoises its BSR conversion as a private attribute on
`int_m` (auto-clears when the matrix is GC'd, invalidates when
`dec_m` identity or blocksize changes); the MKL dispatch caches the
optimised handle in `MCEqRun._mkl_etd2_cache`, keyed by matrix
identity. Set the corresponding `*_bsr_blocksize` config to `None` to
fall back to CSR (useful for debugging or if a future scipy / MKL
regresses BSR perf). The matrix dimension is auto-padded with zero
rows / cols up to a multiple of `blocksize`, and the kernel allocates
working buffers at the padded length — the trailing slots stay zero
throughout because the matrix has zero rows / cols there.

MKL per-step SpMVs go through `mkl_sparse_d_mv` with
`y = α·A·x + β·y` (β=0 to zero `F_*`, β=1 to accumulate the
`ρ⁻¹·dec_off` term).

The CUDA kernel uploads CSR matrices and diagonals to device memory
once and keeps the per-step state on the GPU; only the boundary
state and final snapshots cross the host/GPU boundary. cuPy 13 does
not auto-discover the `nvidia-*` pip packages, so `CudaEtd2Context`
dlopens them defensively before the first kernel JIT. cuSPARSE BSR
SpMV is *not* exposed via cupy 13's sparse module, so the GPU
backend stays on CSR — cuSPARSE CSR is already near-bandwidth-optimal
on the A30, so the format is well-matched to the hardware.

Sanity-checked invariants (`tests/test_solvers.py`):

1. **MKL vs numpy** on real SIBYLL21 matrices at θ = 60°, uniform
   path: full-state rel-L2 ≤ 10⁻¹² (fp64). The 4 SpMVs/step are the
   same arithmetic on both backends, so equality is essentially
   round-off.
2. **CUDA vs numpy** on the same problem: rel-L2 ≤ 10⁻⁹ — cuSPARSE
   reorders partial sums vs scipy CSR, so we tolerate one extra
   order of magnitude beyond round-off.
3. **High-zenith stability** (θ = 89°, e±/γ disabled per §7): both
   kernels stay finite end-to-end.

## 9. Implementation reference

### 9.1 Module layout

| File                              | Contents                                                                                |
| --------------------------------- | --------------------------------------------------------------------------------------- |
| `src/MCEq/solvers.py`             | `solv_numpy_etd2`, `solv_spacc_etd2`, `_etd_split_cache`, `_etd_compute_diag_factors`, `_etd_step_buffers`, `etd2_nonuniform_path` |
| `src/MCEq/core.py`                | `MCEqRun.solve`, `_build_kernel_dispatch`, `_calculate_integration_path`, `MatrixBuilder._construct_differential_operator`, `_follow_chains`, `_fill_matrices` |
| `src/MCEq/particlemanager.py`     | `MCEqParticle`, `MCEqParticle._apply_force_resonance`                                   |
| `src/MCEq/config.py`              | `kernel_config`, `etd2_path` dict, `loss_stencil_method`, `loss_stencil_alpha0`, `adv_set["force_resonance"]`, `adv_set["disabled_particles"]` |

### 9.2 Per-step ufunc chain

`solv_numpy_etd2` (and the Apple Accelerate analogue
`solv_spacc_etd2`) both use a fully in-place, preallocated-scratch
loop. The per-step buffers (`D`, `hD`, `eD`, `phi1`, `phi2`, `mask1`,
`mask2`, `abs_hD`, general `scratch`) are allocated once via
`_etd_step_buffers(dim)` and reused across all `nsteps` iterations.

The φ-function evaluation deserves attention. The naive forms (11)
cancel catastrophically near zero; we use a Horner Taylor patched
into the analytic form via `np.copyto(..., where=mask)`:

```python
PHI1_SMALL = 1e-6
PHI2_SMALL = 1e-3   # phi2 cancels at a wider radius than phi1

# phi1 = (e^z - 1) / z on |z| > PHI1_SMALL, Taylor 1 + z/2 + z²/6 elsewhere
np.subtract(eD, 1.0, out=phi1)
np.abs(hD, out=abs_hD); np.greater(abs_hD, PHI1_SMALL, out=mask1)
np.divide(phi1, hD, out=phi1, where=mask1)
# Horner Taylor: ((1/6)·z + 1/2)·z + 1
np.multiply(hD, 1.0/6.0, out=scratch)
np.add(scratch, 0.5, out=scratch)
np.multiply(scratch, hD, out=scratch); np.add(scratch, 1.0, out=scratch)
np.invert(mask1, out=mask1)
np.copyto(phi1, scratch, where=mask1)
# (analogous for phi2 with PHI2_SMALL, hD² in denominator, Taylor 1/2 + z/6 + z²/24)
```

This avoids `np.where`'s eager double-evaluation and stays alloc-free
inside the loop. On the spacc kernel the per-step time dropped from
2.10 ms to 0.23 ms (~9×) on the reduced DB after this rewrite.

### 9.3 Configuration surface

Defaults in `src/MCEq/config.py`:

```python
kernel_config = "auto"       # auto → "accelerate_etd2" on macOS, else "numpy_etd2"

etd2_path = {
    "eps":     0.3,    # within-step | d ln ρ⁻¹/dX | tolerance
    "dX_max":  20.0,   # cap (just below the off-diagonal stability cliff)
    "dX_min":  0.01,   # floor (avoids 0-step at top of atmosphere)
    "fd_span": 0.01,   # forward-FD probe span for | d ln ρ⁻¹/dX |
}

loss_stencil_method  = "expfit"   # alternatives: "centered", "biased"
loss_stencil_alpha0  = 3.0        # exp-fit trial slope (α in exp(α·u))

adv_set["force_resonance"]   = []    # opt-in per-PDG resonance approximation
adv_set["disabled_particles"] = [11] # drop e± by default (EM cascade caveat)
```

Each `etd2_path` value can be overridden per call:

```python
mceq = MCEqRun(...)
mceq.solve()                                  # uses defaults
mceq.solve(eps=0.5, dX_max=15.0)              # coarser
mceq.solve(int_grid=np.linspace(50, 2000, 10000))  # >=10k snapshots
```

### 9.4 Removed configuration (1.4.1 → 2)

| Removed                                     | Replacement / migration                                          |
| ------------------------------------------- | ---------------------------------------------------------------- |
| `integrator`, `ode_params`                  | none (MCEq 2 ships only ETD2RK)                                  |
| `leading_process`, `stability_margin`, `dXmax` | `etd2_path["eps"]`, `etd2_path["dX_max"]`, `etd2_path["dX_min"]` |
| `hybrid_crossover`                          | none (resonance approximation removed)                           |
| `adv_set["no_mixing"]`                      | now the default behaviour                                        |
| `adv_set["exclude_from_mixing"]`            | none (no mixing logic)                                           |
| `kernel_config = "numpy" / "MKL" / "CUDA"`  | `"numpy_etd2"` / `"mkl_etd2"` / `"cuda_etd2"`                    |
| `MCEqParticle.is_mixed`, `mix_idx`, `E_mix`, `hadridx`, `residx` | none (folded into `is_resonance` binary)                                                          |
| `_assign_hadr_dist_idx`, `_assign_decay_idx`| `_assign_hadr_dist`, `_assign_decay_dist` (full-range, no slice) |
| `inverse_decay_length(cut=True)`            | `inverse_decay_length()` (full range)                            |

## 10. Methods evaluated and discarded

* **ETD1 / Lawson scheme** — first-order single-stage exponential
  integrator. Validated correct (clean order 1 in convergence study)
  but error constant is ~16× worse than Euler at the same grid; only
  useful as a stability fallback at extreme `C`. Removed.
* **Strang splitting** `e^{h/2 D} (I + h N) e^{h/2 D}` — splitting
  error grows linearly with stiffness via `O(h² [D, N])`; rel-L2
  worsens 3 % → 18 % over a 16× ρ⁻¹ sweep. Inadequate.
* **Krylov / `scipy.sparse.linalg.expm_multiply`** — accurate but
  ~300× slower per step due to Krylov subspace construction and
  norm-est overhead. Not viable as written.
* **Heun / RK2** on the full `[A + ρ⁻¹·B]` — explicit-Euler stability
  bound unchanged; doubles per-step cost without enabling
  coarsening. No advantage.
* **Adams–Bashforth-style ETD2** (multistep) — drops to first-order
  across variable-step boundaries between native segments. Replaced
  by ETD2RK (single-stage, robust to variable `h`).
* **`expfit_ghost`** boundary stencil with single global `α₀` —
  gives a one-sided slope assumption that doesn't match real spectra
  at both edges. Causes oscillation at low-E muon (−8.1 % / +13.2 %
  at the bottom two bins). Rejected (§5.3).

## References

* Cox D. A., Matthews P. C., 2002. *Exponential time differencing for
  stiff systems.* J. Comput. Phys. **176**, 430–455. ETD2RK update
  rule (10) and the φ-functions (11).
* Hochbruck M., Ostermann A., 2010. *Exponential integrators.* Acta
  Numerica **19**, 209–286. Order analysis of (10), survey of
  φ-function evaluation strategies.
* Fedynitch A., Engel R., Gaisser T. K., Riehn F., Stanev T., 2015.
  *Calculation of conventional and prompt lepton fluxes at very high
  energy.* EPJ Web of Conferences **99**, 08001. MCEq 1.4 baseline:
  the cascade equation (1) in slant depth, the resonance
  approximation (§2.3), the Forward-Euler iteration (§2.1).
* Fedynitch A., 2018. *MCEq numerical hadronic interaction model
  cascade equation solver.* AIP Conf. Proc. **1968**, 080001. Code
  architecture and the energy-loss FD operator (§2.4).
* Liechty C., Skeel R. D., 2009. *Exponentially fitted finite
  differences.* Numerical methods background for the `expfit` stencil
  family (§5).
