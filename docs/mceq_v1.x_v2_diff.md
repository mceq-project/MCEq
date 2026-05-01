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

## 11. Integrating 2D MCEq (PR #48) into v2

PR #48 ([branch `origin/2d`](https://github.com/mceq-project/MCEq/pull/48),
Kozynets / Fröse, based on Kozynets, Fedynitch, Becker Tjus,
arXiv:2306.15263) extends MCEq from a slant-depth-only cascade to a
joint *(slant-depth, angle-from-shower-axis)* problem solved in Hankel
frequency space. This section documents (a) the math the PR is solving,
(b) what its 1.4-era implementation does, (c) the v2 reformulation —
which is structurally cleaner and substantially faster than the PR's
sequential per-mode loop — and (d) what is still open.

### 11.1 Hankel-space cascade equation

The 2D state is a particle density `η_h(E, X, θ)` parameterised by
energy `E`, slant depth `X`, and the polar angle `θ` between the
particle and the shower axis. Under axisymmetry, the zeroth-order
Hankel transform

```
η̃_h(E, X, κ) = ∫₀^∞ η_h(E, X, θ) · J₀(κ·θ) · θ dθ                    (21)
```

turns the angular convolution against the production kernel into an
elementwise product over `κ` (Hankel convolution theorem). Each `κ` mode
then satisfies an independent cascade equation of the same shape as (1):

```
dη̃_κ/dX = [ A(κ) + ρ⁻¹(X) · B(κ) ] η̃_κ ,            for κ ∈ k_grid    (22)
```

Crucially, **k-modes are exactly decoupled** in (22): the off-block
entries between modes are zero by the convolution theorem, and the
authors confirm this in §III of the paper. We verified the structure
empirically against `mceq_db_URQMD_150GeV_2D.h5`: the per-channel
matrices stored as a `(n_k·n_E, n_k·n_E)` block-diagonal have
*exactly* zero off-block entries (max-abs `0.0`) for both the hadronic
and decay groups, and the per-block values fall off smoothly with `κ`
(SIBYLL-2.3D Λ̄→Λ̄ block-diagonal entries: 4.50 at `κ=0` to 0.19 at
`κ=2000`).

### 11.2 What is and isn't k-dependent

| Quantity                                             | k-dependent? |
| ---------------------------------------------------- | ------------ |
| Hadronic off-diagonal `A_off(κ) = int_off(κ)`        | **yes** (Hankel of angular kernel) |
| Decay off-diagonal `B_off(κ) = dec_off(κ)`           | **yes** (parent→daughter angle from kinematics) |
| Cross-section diagonal `−σ_inel/λ_int`                | no  |
| Decay-rate diagonal `−1/(γ τ ρ)`                     | no  |
| Continuous-loss FD stencil `M_loss` (acts on E only) | no  |
| Multiple Coulomb scattering damping (muon rows)      | **yes**, diagonal: `−κ²·θ_s²(E)/4` per step |

So `D = diag(A) + ρ⁻¹·diag(B) + δ_MS(κ)` has only one κ-dependent
piece — the multiple-scattering damping on muon rows — and that piece
is purely diagonal. This is the structural insight that makes ETD2RK
batch cleanly across modes (§11.4).

### 11.3 The PR's 1.4-era implementation

PR #48 keeps the forward-Euler kernel and, in 2D mode, runs it
sequentially across modes (`solv_numpy`, commit `584eea3`):

```python
for step in tqdm(range(nsteps)):
    int_deriv_k = [imc[k].dot(phc[k]) for k in range(len(config.k_grid))]
    dec_deriv_k = [dmc[k].dot(phc[k]) * ric[step]
                   for k in range(len(config.k_grid))]
    full_deriv_k = [int_deriv_k[k] + dec_deriv_k[k]
                    for k in range(len(config.k_grid))]
    phc += np.array(full_deriv_k) * dxc[step]
```

Pain points carried over from 1.4 + new ones from the layout:

1. **Forward-Euler at native step grid** — same diagonal-driven step
   ceiling as 1.4 (§2.2), with step counts that explode at high zenith.
   2D inherits this fully; 28 800 steps per mode at θ=89°.
2. **Sequential mode loop in Python** — list comprehensions inside the
   per-step loop allocate `n_k` lists and do `n_k` separate scipy SpMVs
   per step; the `np.array(full_deriv_k)` line then re-materialises
   the result. Even with the same arithmetic, dispatch overhead is
   `n_k` ×.
3. **Resonance approximation** (§2.3) — still on, with the same
   ~0.3–0.7 % systematic, now applied identically to every mode.
4. **Decay matrix stored 24× redundantly** — the database has
   `dec_m[κ]` at 24 different κ. They are *not* identical (kinematic
   angular spread is real), but they share more than 90 % of their
   structural sparsity, so this is leaving a lot of compression on
   the table (§11.5).
5. **Database energy grid is 60-bin (10 MeV–10 TeV)**, not the 121-bin
   1D grid; cross sections / continuous losses still come from the 140-
   bin 1D arrays and are sliced down via `config.default_ecenters`
   (`HDF5Backend.cs_cuts`). The current cs-cut path was rewritten in
   `ba916fa` to call the module-level `_eval_energy_cuts(...)` directly
   instead of the instance method; this needs to be kept.
6. **`config.enable_2D = True` by default** in PR #48's config, with
   `mceq_db_fname = "mceq_db_lext_dpm191_v131.h5"` (the 1D database).
   In v2 `enable_2D` should default to `False` and only flip when the
   2D database is opened.
7. **Multiple scattering** is folded as a per-step elementwise multiply
   on the muon rows (`phc[k][muon_rows] *= exp(−κ²·dX·θ_s²/4)`); this
   is operator splitting on top of forward-Euler and has the same
   `O(h)` splitting error as Strang on a stiff diagonal. With ETD2 the
   damping belongs in `D` directly (§11.4).
8. **Hankel→θ reconstruction** (`MCEqRun.convert_to_theta_space`,
   commit `e7abed3`) uses cubic interpolation onto a 5×-oversampled κ
   grid plus `np.trapz` against `J₀(κθ)·κ`. Adequate but ad-hoc; the
   structured **discrete Hankel transform** of Guizar-Sicairos &
   Gutiérrez-Vega (2004, JOSA A 21, 53) samples at zeros of `J₀` and
   gives near-exact transform pairs at the same cost. Worth swapping
   in but not load-bearing.

### 11.4 v2 reformulation: stacked-state ETD2RK

Because `D` is k-independent except for the diagonal multiple-scattering
piece, all k-modes share the integrating factor `e^{h·D}` and the
φ-functions. With state stacked as `Φ ∈ ℝ^{N×n_k}` (one column per
mode), the ETD2RK update (10) becomes batched along the mode axis:

```
F(Φ)[:, k] = int_off(κ_k) · Φ[:, k]  +  ρ⁻¹ · dec_off(κ_k) · Φ[:, k]
D(κ_k)     = d_int + ρ⁻¹ · d_dec  +  δ_MS(κ_k)               # length-N
a[:, k]    = exp(h·D(κ_k)) ⊙ Φ[:, k]  +  h · φ₁(h·D(κ_k)) ⊙ F(Φ)[:, k]
Φ[:, k]   <- a[:, k] + h · φ₂(h·D(κ_k)) ⊙ ( F(a)[:, k] − F(Φ)[:, k] )  (23)
```

Per-step structure:

* **Diagonal factors are computed once** if `δ_MS = 0` (no muon
  scattering). With `δ_MS(κ) ≠ 0` only on the muon rows, the cheapest
  layout is `D` of shape `(N, n_k)` where `D[i, k] = d_int[i] +
  ρ⁻¹·d_dec[i] + δ_MS[i, k]` (only the muon rows differ across `k`),
  and `_etd_compute_diag_factors` runs unchanged on the flattened
  `(N·n_k,)` view. Cost: identical to one 1D step's diagonal compute,
  amortised across all modes.
* **Off-diagonal SpMV is batched, not SpMM.** Each κ has its own
  operator `N(κ_k)` — same sparsity pattern, *different* values — so
  what looks like a `(N, n_k)` × sparse multiplication is actually
  `n_k` independent SpMVs. Classical SpMM (one sparse matrix × dense
  right-hand-side block, e.g. `cusparseSpMM`, `mkl_sparse_d_mm`) does
  not apply: the operator changes per column. The state stays a
  vector — per mode — even though we stack it for layout. Three
  implementation tiers:

  1. *Tier A (drop-in)* — call `solv_numpy_etd2` `n_k` times in a
     Python `for k in range(n_k):` loop. Each call inherits all of
     v2's path control, BSR conversion, and step-count reduction.
     Expected speedup over PR #48 forward-Euler: **9–11×** at fixed
     accuracy, just as in 1D (§6.3). This is the right first
     milestone — minimal new code, big win.
  2. *Tier B (custom batched-SpMV kernel)* — store the off-diagonals
     as a single value array `int_off_vals` of shape `(nnz, n_k)`
     sharing one `(indptr, indices)` (the union of per-mode supports;
     pattern density tracks the densest mode, κ=0). One step walks
     the sparsity pattern *once* and does `n_k` multiply-accumulates
     per nonzero into `out[:, k]`. Removes the Python per-mode
     dispatch overhead and amortises the cache traffic of the
     pattern walk across all modes. For BSR, the same with values
     `(n_blocks, b, b, n_k)`. Expected additional **~1.5–2×** on the
     SpMV portion. (An equivalent form: stitch the `n_k` operators
     into one block-diagonal sparse matrix of shape `(N·n_k, N·n_k)`
     and do *one* SpMV against the flattened length-`N·n_k` state.
     Same arithmetic, simpler kernel, no shared-pattern reuse — a
     fallback if the custom kernel is undesirable.)
  3. *Tier C (accelerator-batched)* — cuSPARSE since CUDA 11.3
     exposes `cusparseSpMVBatched` with one sparsity descriptor and
     per-batch value arrays — the exact primitive Tier B describes,
     executed on-device. With `n_k = 24` this is well inside the
     regime where the batched call beats `n_k` separate SpMVs by
     an order of magnitude.

     **MKL has no direct batched-SpMV equivalent.** Intel's sparse
     BLAS exposes `mkl_sparse_d_mv` (one matrix × one vector) and
     `mkl_sparse_d_mm` (one matrix × dense RHS block — *real* SpMM,
     does not apply since our operator differs per mode); the
     `*_batch` family is dense-only (`cblas_dgemm_batch`,
     `cblas_dgemv_batch`). On the MKL backend, the practical paths are,
     in order of preference:

     a. **Block-diagonal stitch.** Build one sparse matrix of shape
        `(N·n_k, N·n_k)` with the per-mode operators on the block
        diagonal, run one `mkl_sparse_d_mv` per ETD2 stage against
        the flattened length-`N·n_k` state. Same arithmetic as
        `n_k` separate SpMVs but a single inspector-executor handle
        with one cost-analysis pass; slots cleanly into
        `solv_mkl_etd2`'s `MCEqRun._mkl_etd2_cache` machinery. nnz
        grows `n_k×` versus a shared-pattern store (no compression),
        which is fine at MCEq sizes.
     b. **`n_k` handles sharing `(indptr, indices)`.** Distinct value
        buffers per mode, `n_k` calls per step in a tight C loop.
        `mkl_sparse_optimize` still amortises cost analysis since the
        pattern is identical across handles, but cache reuse across
        modes is lost.
     c. **Custom AVX-512 batched-SpMV kernel.** Walk `(indptr,
        indices)` once, do `n_k` FMAs per nonzero against
        `vals[nnz, n_k]`. `n_k = 24` doubles fits one AVX-512 register
        lane neatly. ~1.5–2× over (b) at the cost of a hand-written
        kernel.

     Option (a) is the right starting point for the MKL backend.

     **Benchmark on Apple Accelerate** (the closest multi-threaded
     CPU-sparse stack we ship today) confirms the ranking. SIBYLL-2.3D
     hadronic operator from `mceq_db_URQMD_150GeV_2D.h5` summed across
     channels gives a per-mode `(1500, 1500)` matrix at 16 % density
     (363k nnz/mode, 24 modes); the block-diagonal stitch is
     `(36000, 36000)` at 8.71M nnz. Mean SpMV time per *full step's
     worth of modes* (i.e., for all 24 modes combined; M-series, 12
     physical cores):

     | method                              | µs/call (24 modes) | µs/mode | rel. to D |
     | ----------------------------------- | ------------------:| -------:| ---------:|
     | A scipy CSR `n_k` SpMVs (loop)      |        7376        |   307   |     0.20× |
     | B scipy CSR block-diagonal SpMV     |        7270        |   303   |     0.20× |
     | scipy BSR(6) `n_k` SpMVs (loop)     |        4108        |   171   |     0.35× |
     | scipy BSR(6) block-diagonal SpMV    |        4325        |   180   |     0.33× |
     | C Accelerate CSR `n_k` SpMVs (loop) |        2508        |   104   |     0.57× |
     | D Accelerate CSR block-diagonal SpMV|        1441        |    60   |     1.00× |

     Three orthogonal axes are visible:

     * *Threading.* Forcing single-thread (`VECLIB_MAXIMUM_THREADS=1`)
       collapses C and D to scipy's CSR level (~7 ms) — the entire
       Accelerate win is threading. scipy CSR/BSR SpMV is
       single-threaded, so A and B are identical, and so are the BSR
       loop and BSR block-diagonal variants.
     * *Block-diagonal stitch vs per-mode loop.* For *threaded*
       backends, **D beats C by 1.8×** on the same arithmetic: the
       24-call loop spawns/joins worker threads 24 times on jobs too
       small (363k nnz each) to amortise it, whereas the single
       8.7M-nnz call dispatches once. For single-threaded backends
       the stitch is neutral.
     * *BSR vs CSR.* BSR speeds scipy up by ~1.8× independent of
       loop-vs-stitch (BSR(6) loop ≈ BSR(6) block-diag ≈ 4.1–4.3
       ms). The optimal `blocksize=6` here matches `60 = 6·10`; the
       v2 1D config uses `blocksize=11` because `n_E=121=11²`. The 2D
       database's `n_E=60` factors as `6·10` / `5·12` / `4·15`, and a
       blocksize sweep confirms `b ∈ {5, 6, 10}` are the winners,
       degrading at `b ≥ 12`.

     **The orthogonality matters**: BSR helps scipy but does *not*
     stack on top of the threaded-stitch win, because the threaded
     stitch is already 3× faster than the best scipy BSR variant.

     We further tested BSR *on Accelerate* directly, by exposing
     `sparse_matrix_block_create_double` / `sparse_insert_block_double`
     through a small ctypes shim (the existing `spacc.c` only wraps
     the point format). At `blocksize=6` (matching the 2D database's
     `n_E=60` factoring), measured times are:

     | variant                          | µs/call | µs/mode |
     | -------------------------------- | -------:| -------:|
     | Accelerate CSR block-diag (D)    |  **1441** | **60** |
     | Accelerate BSR(6) block-diag     |    2804 |    117 |
     | Accelerate CSR per-mode loop (C) |    2508 |    104 |
     | Accelerate BSR(6) per-mode loop  |    8364 |    349 |

     **BSR-on-Accelerate is ~2× slower than CSR-on-Accelerate** on
     both the block-diag and per-mode forms. The point-format
     threaded kernel in Apple's Sparse BLAS appears more tightly
     tuned than the block-format kernel at our matrix sizes
     (≥ 1500-dim, 16 % density, 6×6 sub-blocks that aren't fully
     dense). Block-diag still amortises threading by ~3× whether
     we're on CSR or BSR — so the stitch wins regardless of format
     — but the format choice on Accelerate is plainly **CSR**.

     For the **MKL** backend the picture is different and we
     haven't directly measured: MKL's threaded BSR microkernel is
     designed for `b ∈ [2, 7]` (the regime that gave v2's 1D 1.5×
     over MKL CSR, §8.4) and its threading model also amortises
     per-call dispatch differently than Accelerate's. Best guess:
     MKL BSR(6) block-diagonal stitch will land between Accelerate
     CSR-stitch and scipy-BSR-stitch. v2's existing
     `mkl_bsr_blocksize=6` already matches the 2D-database-optimal
     blocksize, so no new tuning is needed for that backend — but
     the BSR-vs-CSR decision should be re-measured once the
     MKL-2D path is wired up rather than assumed from 1D.

     Implication for MKL: option (a) — block-diagonal stitch — is
     not just easier, it is also the faster MKL path because
     `mkl_sparse_d_mv`'s thread dispatch follows the same logic as
     Accelerate's. Confirming this on MKL is a one-call experiment
     once the MKL-2D backend is wired up; we expect option (a) to
     beat option (b) by a similar 1.5–2× margin.

     Bench script: `/tmp/bench_2d_spmv.py` (kept as a reference under
     `docs/scripts/` if revived).

* **Stability bound.** `spec(N(κ))` is dominated by the `M_loss`
  band (§3.5), which is k-independent — so `h_max ≈ 21 g/cm²` from
  (14) is the same for every mode. The hadronic Hankel kernel
  off-diagonal *decreases* with `κ` (verified above), so high-κ modes
  are at most as stiff as `κ=0`. Path control (§4) is unchanged: one
  schedule serves all modes.

* **Multiple scattering.** With ETD2RK, the muon scattering damping
  `−κ²·θ_s²(E)/4` is just a contribution to `D` on muon rows. The
  integrating factor `e^{h·D}` then absorbs it exactly — no operator
  splitting, no `O(h)` extra error. This removes the standalone
  `phc[k][muon] *= exp(...)` factor in the PR's loop. (Sanity check:
  for k=0, `θ_s = 0` and the term vanishes, recovering the 1D limit
  exactly.)

**Headline expectation**: Tier A alone replaces the PR's
`28 872 × 24 = 692 928` per-mode forward-Euler steps at θ=89° with
≈ `1325 × 24 = 31 800` ETD2 steps, at the same per-step arithmetic
cost as v2's 1D ETD2 plus a factor of `n_k = 24` on the SpMV. Tier B
amortises one walk-of-sparsity per step across all modes, removing
roughly half of the residual Python overhead. End-to-end this should
land somewhere between **10× and 25×** faster than the PR as written,
zenith-independent.

### 11.5 Database changes for the integrated v2-2D path

* `enable_2D` should be detected from the database (`'k_dim' in
  common.attrs`), not from a config flag. The flag should be removed
  in favour of "the database is 2D-shaped or it isn't" — same pattern
  as `muon_helicity_dependence` reading from the decay group.
* The current 2D database stores 24 per-mode matrices side-by-side as
  a block-diagonal `(n_k·n_E, n_k·n_E)` CSR; the loader reshapes to
  `(n_k, n_E, n_E)` per (parent, child) pair. v2's matrix builder
  outputs `int_m, dec_m` of *the same* `(n_k, dim_states, dim_states)`
  tensor shape today (commit `c597dff`); switching the storage to a
  `(nnz_pattern, n_k)` value array against a single shared
  `(indptr, indices)` is a one-time loader change and is what enables
  tier B. The shared sparsity pattern is the union of nonzero supports
  across modes; pattern density scales as the densest mode (`κ=0`),
  so this is approximately free in storage.
* Cross sections and continuous losses live on the 1D 140-bin grid
  and are sliced down to the 60-bin 2D grid via
  `_eval_energy_cuts(config.default_ecenters, e_min, e_max)` (the
  `ba916fa` fix). v2 should keep this — there is no reason to
  re-tabulate cross sections on the 2D grid.

### 11.6 Open issues for 2D-on-v2

1. **EM cascade (§7) becomes load-bearing.** Lateral spread of EAS at
   small angles is dominated by the EM component (γ, e±). Disabling e±
   by default — the current v2 production stance — is acceptable for
   inclusive *muon* / *neutrino* fluxes but **not** for the angular
   distributions PR #48 is targeting. The 2D integration depends on
   shipping the block-ETD §8.2 fix that handles the L/R semi-Lagrangian
   triplets, *or* on a 2D database that omits the L/R variants
   altogether. Recommend: bring §8.2 forward in priority once 2D work
   resumes.
2. **Hankel reconstruction quality (investigated, legacy retained).**
   We investigated replacing the legacy cubic-interp + `np.trapz`
   inverse Hankel (`MCEqRun.convert_to_theta_space`, ported to
   `MCEq.hankel.inverse_hankel_legacy`) with a Filon-J₀ quadrature
   that integrates the J₀ oscillation exactly per segment via the
   closed form `∫ k J₀(αk) dk = (k/α) J₁(αk)`. Two variants tested:

   * **Segment-mean F per `[k_i, k_{i+1}]`** (no F interpolation).
   * **Linear F per `[k_i, k_{i+1}]`** (chord between endpoints; needs
     `∫ k² J₀(αk) dk` via the Struve identity from DLMF 10.22.5).

   Both implementations were verified against `scipy.integrate.quad`
   to machine precision on synthetic problems. On the production
   K_GRID `[0, 1, 2, 3, 4, 6, 9, 12, 17, 23, 32, 44, 61, 84, 115,
   158, 217, 299, 410, 563, 773, 1061, 1457, 2000]`, neither variant
   beat legacy on a Gaussian round-trip:

   | sigma  | legacy rel-err | linear-F Filon | ratio (Filon/legacy) |
   | -----: | -------------: | --------------: | -------------------: |
   | 0.001  |        1.4·10⁻¹|      1.2·10⁻¹  | 0.9× (truncation-bound, see below) |
   | 0.005  |        3.9·10⁻³|      5.1·10⁻²  | 13× **worse**       |
   | 0.01   |        2.4·10⁻³|      5.2·10⁻²  | 22× worse           |
   | 0.05   |        2.7·10⁻³|      5.3·10⁻²  | 20× worse           |
   | 0.1    |        2.0·10⁻³|      5.3·10⁻²  | 26× worse           |

   **Diagnosis:** the error budget is dominated by F-curvature within
   wide segments of the geometric K_GRID — *not* by J₀ oscillation.
   Concretely, in the segment `[44, 61]` for σ=0.05, F drops from
   `2.4·10⁻⁴` to `5·10⁻⁷` (factor ~480 within the segment); a linear
   chord overshoots the concave-up Gaussian by ~5 % at θ=0, which is
   the worst-θ case (J₀ does not oscillate, so the Filon advantage
   collapses). The legacy cubic spline catches this curvature; linear-F
   cannot. At high θ where J₀ oscillates within a segment, Filon does
   win — RMS error over θ ∈ [0, π/2] for σ=0.05 is 0.7 % (Filon) vs
   0.3 % (legacy) — but the `max` is dominated by θ=0.

   **Truncation floor for narrow Gaussians:** at σ=0.001 we have
   `∫_{2000}^∞ σ²·exp(−σ²k²/2)·k dk = exp(−2) ≈ 0.135`, so any
   quadrature on `[0, k_max]` is bounded above by 86.5 % recovery of
   `f(0) = 1`. Legacy's 13.7 % at σ=0.001 happens to come in slightly
   under that bound because `scipy.interpolate.interp1d(kind="cubic")`
   accidentally extrapolates past `k_max`; that's a coincidence, not
   a method advantage.

   **Conclusion: the database K_GRID density, not the J₀-quadrature
   strategy, is the binding accuracy constraint** for any low-order F
   approximation. The principled fix would be cubic-F Filon (cubic
   spline of F + closed-form `∫ k^n J₀(αk) dk` for `n = 0..3` per
   segment via Struve recursion) — this matches legacy at θ=0 and
   should beat it at high θ where J₀ oscillates within a segment.
   ~80–150 LOC of careful work; deferred until a downstream physics
   problem demonstrably needs better accuracy.

   `MCEq.hankel.inverse_hankel_legacy` is retained as the standalone,
   testable extraction (`tests/test_hankel.py` pins its accuracy on
   Gaussian round-trip). `MCEqRun.convert_to_theta_space` is left
   unchanged in v2 — same algorithm as PR #48, verified against the
   PR baseline (Task 1.6, ν_μ angular flux at 2.49 % rel-L2).

   Bench script preserved as `/tmp/bench_2d_spmv.py` and the
   investigation history is in branches `2d-on-v2` working tree
   (uncommitted Filon attempts) — recoverable from agent transcripts
   if cubic-F Filon is later picked up.
3. **The `ba916fa` "Do we actually need that?"** comment on the
   cross-section cut: yes, because the 2D HDF5 stores `cross_sections`
   on the 1D grid (`shape=(140, ...)`) and the 2D state vector lives on
   the 60-bin grid. The cut is needed; the `ba916fa` rewrite to use
   the module-level `_eval_energy_cuts` is correct.
4. **Force-resonance opt-in (§3.4).** With ETD2RK the resonance
   approximation is removed. The 2D database (`URQMD 3.4` low-E) was
   produced under the 1D resonance approximation though, so all
   short-lived species in the 2D matrices are already folded into
   parents at build time. Either re-tabulate without the approximation,
   or set `adv_set["force_resonance"]` to the same PDG list the 2D
   database was built with — see `URQMD34` equivalences map in
   `data.py` (`3122 → 2112`, `−3122 → −2112` for cross sections).
5. **Muon scattering parameters.** `lambda_s = 37.7 g/cm²`,
   `E_s = 0.021 GeV` are CORSIKA's Gauss-approximation constants
   (Heck & Pierog handbook p.12). Reasonable, but worth flagging
   that this is a model choice that lives outside the database — keep
   it in `config` and document the source.

### 11.7 Suggested rollout

1. ~~*On a `2d-v2` branch off `v2`*: port the 2D matrix-builder pieces
   (`_zero_mat`, `_csr_from_blocks`, `_follow_chains`, `_fill_matrices`
   2D branches in commits `c597dff`, `77fb4c7`) onto v2's
   `MatrixBuilder`. No solver changes yet; verify that
   `int_m.shape == (n_k, dim_states, dim_states)` rebuilds correctly
   against `mceq_db_URQMD_150GeV_2D.h5` and that the resonance
   approximation still folds short-lived species at this stage.~~
   ✅ landed in commits `23e5663`, `179cbce` (Tasks 1.1–1.2);
   rebuilds correctly against `mceq_db_URQMD_150GeV_2D.h5`. See
   `tests/test_2d_matrices.py`.
2. ~~*Tier A solver*: add `solv_numpy_etd2_2d(...)` that calls
   `_etd_get_split_for_numpy` per κ and reuses `_etd_compute_diag_factors`
   /step `n_k` times. Path schedule (§4) is shared. This is the
   minimum-viable v2-2D and should already match the PR's results
   (sub-percent on the κ=0 mode against 1D MCEq native step).~~
   ✅ resolved differently — Tasks 1.4 + 1.5 showed v2's existing
   `solv_numpy_etd2` is dimension-agnostic; the stitched
   `(n_k·N, n_k·N)` matrix flows through unchanged across all four
   backends (numpy / Accelerate / MKL-deferred / CUDA-deferred).
   Cross-backend equivalence test in `tests/test_solvers_2d.py`
   (commits `9eb6f31`, `2a8fd06`).
3. ~~*Multiple scattering folded into `D`*: add the muon-row κ²-damping
   into `_etd_compute_diag_factors` via a `delta_MS` length-`N`
   summand (per-κ-call; trivial). Verify against a CORSIKA-level
   muon angular profile at a single energy decade.~~
   ✅ landed in commit `06af608` (Task 1.3). Tested in
   `tests/test_2d_muon_scattering.py`. ETD2RK absorbs the
   κ-dependent muon-row damping exactly via `e^{h·D}`; PR's
   per-step elementwise multiplier removed.
4. ~~*Tier B stacked SpMM*: introduce a `BatchedOffdiag` class wrapping
   `(indptr, indices, vals[nnz, n_k])` and replace the per-κ-loop
   with a single fused walk. Bench against tier A; promote when
   ≥1.5× faster on SIBYLL23D θ=60°.~~ — deferred. Block-diagonal
   stitch is the structurally cleanest first form and was sufficient
   for the headline 23× wall-time win (§11.8); a custom batched-SpMV
   kernel is reserved for the moment a downstream physics problem
   demands it.
5. ~~*Inverse Hankel quality*: legacy cubic-interp + `np.trapz` is
   retained for v2 (see §11.6 item 2). Filon-J₀ on the existing
   K_GRID was investigated and rejected — neither a Guizar-Sicairos
   DHT nor a cubic-F Filon is a "drop-in"; both require K_GRID
   redesign or significant new quadrature code. Defer until a
   downstream physics problem demonstrably needs better than the
   2–4·10⁻³ rel-err legacy delivers for σ ≥ 0.005.~~ — investigated,
   not adopted. See §11.6 item 2 — Filon-J₀ on the geometric K_GRID
   is bottlenecked by F-curvature within wide segments, not by J₀
   oscillation. Legacy `MCEqRun.convert_to_theta_space` (cubic
   interp + np.trapz) retained; the cubic-F + closed-form
   J₀-moments path is the principled future improvement when
   accuracy actually bites.
6. *EM cascade (§8.2)*: still required for shower physics where the
   EM component matters. Out of scope for the muon/neutrino path
   landed here.

### 11.8 Validation summary

Measured at the close of the 2D-on-v2 work (commits
`9834951`..`db6ebf8`).

**Cross-backend correctness (Task 1.5).**
- `solv_numpy_etd2` ≡ `solv_spacc_etd2` on the stitched
  `(36000, 36000)` 2D operator: rel-L2 ≤ `1·10⁻¹⁰`, no tolerance
  widening needed. Test:
  `tests/test_solvers_2d.py::test_2d_accelerate_matches_numpy`. MKL
  and CUDA placeholders auto-skip on this hardware.

**Per-mode Hankel-space agreement vs PR #48 forward-Euler baseline
(Task 1.6).**
On the lepton block (510 entries; the hadron block has a
particle-list ordering difference unrelated to the solver — see
test docstring) at the deepest snapshot, EPOSLHC, θ=30°, 100 GeV
proton primary:

| metric | rel-L2 |
| ------ | -----: |
| min over k-modes        | 0.124 % |
| median over k-modes     | 0.488 % |
| max  over k-modes (k=18, κ=410) | 1.71 % |

The 1.71 % max is the *irreducible* PR-Euler-vs-v2-ETD2RK gap on
this problem (PR uses `1 + h·D`, v2 absorbs the diagonal exactly
via `e^{h·D}`). Convergence sweep (`eps=0.05/dX_max=2` →
`eps=0.01/dX_max=0.5`) shows v2-ETD2RK converged to <0.1 % within
itself; the gap is solver-method, not insufficient step refinement.

**Angular flux agreement vs baseline (Task 1.6).**
ν_μ + ν̄_μ summed over helicity, summed over energy, deepest
snapshot, via the legacy `convert_to_theta_space`: **2.49 % rel-L2**
vs the PR baseline. Bound: 5 %.

**Filon-J₀ inverse Hankel investigation (Tasks 2.1–2.2 / §11.6).**
Negative result. Legacy retained.

**Wall-time & step counts (Task 3.1, notebook config).**
On `examples/Angular_shower_development.ipynb` (EPOSLHC, θ=30°,
single 100 GeV proton primary, save_depths from
`dm.h2X([15, 5, 0.2, 0] km)`):

| solver | wall time | steps |
| ------ | --------: | ----: |
| PR #48 forward-Euler        | ≈ 22 s | 2560 |
| v2 ETD2RK (this work)       | **0.96 s** | **92** |

≈ 23× wall-time reduction at unchanged accuracy. Step count drops
28× because v2's ρ⁻¹-aware path control (§4) sets `h_n` from the
atmosphere's local scale-height, while PR's forward-Euler step
count was driven by the diagonal stability bound (§2.2).

### 11.9 Minor follow-ups (cleanup, not blocking)

Two small UX warts surfaced during Task 3.1:

1. **`grid_sol` layout.** v2 stores 2D snapshots as flat
   `(n_grid, n_k·N)` arrays (consistent with the stitched-state
   ETD2RK kernel). The legacy `convert_to_theta_space` (PR #48
   verbatim) expects per-snapshot `(n_k, N)` slabs, so the notebook
   does an explicit
   `[snap.reshape(n_k, N) for snap in mceq_2d.grid_sol]` reshape
   before reconstruction. A future cleanup: have
   `convert_to_theta_space` accept the v2 layout natively, or
   expose a `grid_sol_per_mode` view.

2. **`dm.h2X(0) > dm.max_X` floating-point edge.** The CORSIKA
   atmosphere's `h2X(0)` returns a value ~`1·10⁻¹²` above
   `dm.max_X`, which v2's path-builder rejects (loop
   `while X < max_X` exits before the snapshot fires). The notebook
   clamps `save_depths = np.minimum(save_depths, dm.max_X * (1 - 1e-9))`
   as a workaround. Cleaner fix: either `h2X` clamps at `max_X`,
   or `_calculate_integration_path` tolerates
   `int_grid[-1] >= max_X` within a small ε.

Both are isolated to user-code workarounds today; neither blocks
downstream work.

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
* Kozynets T., Fedynitch A., Becker Tjus J., 2023.
  *Atmospheric lepton fluxes via two-dimensional matrix cascade
  equations.* arXiv:2306.15263. The 2D (Hankel-mode) extension of
  MCEq (§11): equation (15) in the paper is (22) here, the inverse
  Hankel transform is (21).
* Guizar-Sicairos M., Gutiérrez-Vega J. C., 2004. *Computation of
  quasi-discrete Hankel transforms of integer order for propagating
  optical wave fields.* JOSA A **21**, 53–58. Reference DHT for a
  future `convert_to_theta_space` replacement; sampled at zeros of
  `J₀`, would require redesigning the database K_GRID (see §11.6).
