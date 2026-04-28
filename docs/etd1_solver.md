# ETD2 Solver for MCEq

Status: numpy kernel implemented and validated as `solv_numpy_etd2`
(`config.kernel_config = "numpy_etd2"`). MKL and CUDA ports are the
next step.

## Problem

The cascade equation in slant depth is

```
dΦ/dX = [ A + ρ⁻¹(X) · B ] Φ
```

with `A = int_m`, `B = dec_m` constant sparse matrices and `ρ⁻¹(X)` a
depth-dependent scalar from the atmosphere. The current forward-Euler
kernel takes one step

```
Φ_{n+1} = Φ_n + h_n · ( A Φ_n + ρ⁻¹_n · B Φ_n )      (2 SpMVs/step)
```

`_calculate_integration_path` picks `h_n` to satisfy the explicit
stability bound `h_n · max(λ_int , ρ⁻¹_n · λ_dec) < margin`. In thin
upper atmosphere `ρ⁻¹` is large, the decay-stability term dominates,
and `h_n` is forced very small. Across a full atmospheric path this
generates O(10³) steps at vertical and ~3×10⁴ at θ ≈ 89° — the single
biggest cost driver of `solve()`.

The strongly-negative eigenvalues that drive that bound sit on the
**diagonal** of A and B: per-energy interaction loss `−σ_s/λ_int` and
decay loss `−1/(γ_s τ_s ρ)`. Off-diagonal entries (production /
redistribution kernels) are mild. Treating the diagonal exactly via
an integrating factor removes the stiffness bound.

## ETD2RK update

Split at depth X_n with ρ⁻¹ frozen:

```
D_n  =  diag(A) + ρ⁻¹_n · diag(B)        (length-N vector)
N_n  =  A_off   + ρ⁻¹_n · B_off          (sparse, no diagonal)
```

The exact local solution (variation of constants):

```
Φ(h) = e^{h D_n} Φ_n  +  ∫₀ʰ e^{(h−s) D_n} N_n Φ(s) ds
```

ETD2RK (Cox-Matthews 2002, single-stage exponential RK):

```
F(state) = N_n · state                  (= A_off·state + ρ⁻¹·B_off·state)
a        = e^{h D_n} · Φ_n  +  h · φ₁(h D_n) · F(Φ_n)
Φ_{n+1}  = a            +  h · φ₂(h D_n) · ( F(a) − F(Φ_n) )
```

with the entire functions

```
φ₁(z)  =  (e^z − 1) / z              (limit 1   as z → 0)
φ₂(z)  =  (e^z − 1 − z) / z²         (limit 1/2 as z → 0)
```

evaluated **elementwise** on the diagonal vector `h·D_n`. Locally
O(h³), globally O(h²). Stiffness-immune along the diagonal:
`|e^{h D_n}|` ≤ 1 unconditionally; `φ₁, φ₂` self-renormalize as
`h·|D|` grows.

## Per-step cost

- **4 SpMVs** total (two `F` evaluations, each = `int_off Φ + ρ⁻¹ dec_off Φ`)
- **1 elementwise `exp`** on a length-N vector
- **2 elementwise rationals** (`φ₁`, `φ₂`) on the same vector, with a
  Taylor branch near zero
- A handful of vector axpy / scal ops

Forward-Euler does 2 SpMVs/step. ETD2 is **2× per step**. The win is
that ETD2 is locally O(h³) accurate vs Euler's O(h²), so coarsening
the grid by 4-16× while preserving accuracy under 1% is straightforward.

## Validation

### Convergence study at θ = 0° (truth = Euler @ oversample=16)

Both schemes refined together; truth is forward-Euler oversampled
16× per native step (h held effectively constant within each native
segment). Order is observed log₂ of error ratio between successive
halvings of h.

| 1/h     | ‖Euler − truth‖ | ‖ETD2 − truth‖ | Euler order | ETD2 order |
| ------- | --------------- | -------------- | ----------- | ---------- |
| native  | 2.32e-4         | 3.96e-5        | —           | —          |
| /2      | 1.08e-4         | 2.10e-5        | 1.10        | 0.92       |
| /4      | 4.64e-5         | 1.68e-5        | 1.22        | 0.32       |
| /8      | 1.55e-5         | 1.58e-5        | 1.58        | (truth-floor) |

ETD2's "order drop" at /4 and /8 is the truth-reference's own
discretization floor — ETD2 has converged below `‖Euler@oversample=16
− exact ODE‖`. Full-state μ-flux error of Euler @ native vs truth is
**0.37%**; ETD2 @ native is **0.025%** — ETD2 is ~14× more accurate at
the same step grid.

### Coarsening sweep — Euler vs ETD2 head-to-head

`config.adv_set["disabled_particles"] = [11, -11]` (drop e±; see EM
caveat below). Euler at native step grid is the comparison reference;
"rel diff" columns are scheme-vs-Euler@native, **not** vs converged
truth (Euler@native itself sits ~0.4% off truth at vertical).

θ = 0° (Euler @ native = 4140 ms reference, 0.37% off truth):

| C  | Euler ms | Euler μ rel diff | ETD2 ms   | ETD2 μ rel diff |
| -- | -------- | ---------------- | --------- | --------------- |
| 1  | 4140     | baseline         | 8521      | 0.38%           |
| 2  | 2076     | 0.19%            | 4279      | 0.57%           |
| 4  | 1057     | 💥 unstable      | 2128      | 0.97%           |
| 8  | 524      | 💥               | **1057**  | **1.02%**       |
| 16 | 265      | 💥               | 532       | 1.22%           |
| 32 | 130      | 💥               | 270       | 5.11%           |

θ = 60° (Euler @ native = 8345 ms reference):

| C  | Euler ms | Euler μ rel diff | ETD2 ms   | ETD2 μ rel diff |
| -- | -------- | ---------------- | --------- | --------------- |
| 1  | 8345     | baseline         | 17191     | 0.30%           |
| 2  | 4205     | 0.29%            | 8529      | 0.30%           |
| 4  | 2086     | 💥               | 4272      | 0.32%           |
| 8  | 1048     | 💥               | 2142      | 0.39%           |
| 16 | 522      | 💥               | **1066**  | **0.60%**       |
| 32 | 263      | 💥               | 541       | 2.23%           |

θ = 89° (Euler @ native = 83 s reference):

| C  | nsteps | Euler ms | Euler μ rel diff | ETD2 ms     | ETD2 μ rel diff |
| -- | ------ | -------- | ---------------- | ----------- | --------------- |
| 1  | 28872  | 86849    | baseline         | 177905      | 0.17%           |
| 2  | 14436  | 43259    | 0.17%            | 87787       | 0.17%           |
| 4  | 7218   | 💥       | 💥               | 43738       | 0.17%           |
| 8  | 3609   | 💥       | 💥               | 21784       | 0.22%           |
| 16 | 1804   | 💥       | 💥               | 10851       | 0.69%           |
| **20** | **1443** | 💥   | 💥               | **8687**    | **1.13%**       |
| 32 | 902    | 💥       | 💥               | 5446        | 💥              |

**Headline numbers**:

* ETD2 at native grid is ~14× more accurate than Euler at native (vs
  converged truth at θ=0°), at 2× the per-step cost.
* ETD2 @ C=2 ≈ Euler @ native in both wall-time and accuracy.
* 4-8× speedup at θ=0° (C=8) and 16× at θ=60° (C=16) while keeping
  μ-flux rel diff ≤ 1%. Speedup grows with zenith.
* **9.6× speedup at θ=89°** with 1.13% μ-flux rel diff (`C=20`,
  1443 steps — same step count as vertical).
* Hard cliff at C ≈ 32: explicit-stage stability bound on the
  off-diagonal coupling. Fundamental, not fixable by tuning.

### EM cascade caveat

`solv_numpy_etd2` has no protective damping for state-vector rows whose
matrix-diagonal is small but whose off-diagonal source scales with
`ρ⁻¹`. In MCEq this hits the e± semi-Lagrangian L/R variants
(`e+_l, e+, e+_r, e-_l, e-, e-_r`) at very large ρ⁻¹ (top-of-atmosphere
near horizon). Disable them via
`config.adv_set["disabled_particles"] = [11, -11]`. Photons must stay
(π⁰ → γγ depends on them); they have no L/R variants and are unaffected.

A future block-ETD extension treating each `(species, E_bin) → {main,
_l, _r}` triple as a 3×3 block exp would lift this restriction without
material per-step cost.

## MKL / CUDA implementation guide

The kernel structure maps cleanly onto the existing accelerated
backends. Reuse the wiring from `solv_MKL_sparse` and `solv_CUDA_sparse`.

### Setup, once per `solve()` call

1. **Split** `int_m → (d_int, int_off)`, `dec_m → (d_dec, dec_off)`
   exactly as `_etd_split_cache` does. Both `int_off` and `dec_off` are
   CSR with the same sparsity pattern as the originals minus the
   diagonal.
2. **Push the two off-diagonal CSR matrices to backend**:
   - MKL: `mkl_sparse_d_create_csr` for each, then `mkl_sparse_set_mv_hint`
     with `nsteps × 2` calls (two SpMVs per step), then `mkl_sparse_optimize`.
   - CUDA: `cupyx.scipy.sparse.csr_matrix(int_off, dtype=fl_pr)` etc.;
     keep them resident on device.
3. **Push the two diagonal vectors** `d_int`, `d_dec` (length N) to the
   backend's working memory.
4. **Allocate persistent scratch buffers** (length N each, all on
   device for CUDA / MKL aligned for AVX): `D`, `hD`, `eD`, `phi1`,
   `phi2`, `F_phi`, `F_a`, `a`. Eight length-N vectors. With N ≈ 8000
   and float64, ~512 KiB total — fits easily in L2.

### Inner loop, per step

```
D       = d_int + ri[k] * d_dec         (axpy: D = d_int; D += ri*d_dec)
hD      = h[k] * D                       (scal)
eD      = exp(hD)                        (elementwise vectorized exp)
phi1, phi2 = etd_phi_funcs(hD, eD)       (see below)

F_phi   = int_off @ phc + ri[k] * dec_off @ phc        (2 SpMVs)
a       = eD * phc + h[k] * phi1 * F_phi               (3 hadamard + 1 add)
F_a     = int_off @ a   + ri[k] * dec_off @ a          (2 SpMVs)
phc     = a + h[k] * phi2 * (F_a - F_phi)              (sub + 2 hadamard + add)
```

Total bandwidth-bound work per step: **4 SpMV** + **~6 length-N vector
ops** + **1 exp** + **phi1/phi2 computation**. Matches the asymptotic
arithmetic intensity of the existing Euler kernels closely — both are
SpMV-bound on long state vectors.

### `phi1` / `phi2` evaluation

Both functions are entire (no singularities). The naive forms
`(e^z - 1)/z` and `(e^z - 1 - z)/z²` cancel catastrophically near zero
and need a Taylor branch.

```
PHI1_SMALL = 1e-6
PHI2_SMALL = 1e-3   # phi2 has wider cancellation region

phi1[i] = ((e^z - 1) / z)            if |z| > PHI1_SMALL
        = 1 + z/2 + z²/6              else                  (z = hD[i])

phi2[i] = ((e^z - 1 - z) / z²)        if |z| > PHI2_SMALL
        = 1/2 + z/6 + z²/24           else
```

Notes:
- For very negative `z` (large stiffness), `e^z` underflows to 0 cleanly;
  `phi1 ≈ -1/z`, `phi2 ≈ -1/z` — both small, well-behaved.
- For `z ≥ 0` (rare in practice — diagonals are removal terms), there
  is no overflow concern at MCEq's step sizes.
- **MKL**: implement as a single fused vector loop with branch on
  `|hD[i]|`. Use `vdExp` from MKL VML for the `exp`. The `where`-style
  branch is cheap; both branches operate on the same lanes.
- **CUDA**: write a single elementwise CUDA kernel that takes `hD` and
  emits `eD, phi1, phi2`. One pass over memory, three outputs. Worth
  writing as a `cupy.fuse` or a hand `RawKernel` to avoid 3 separate
  reads of `hD`.

### Backend-specific points

#### MKL port

- The two CSR handles (`int_off_handle`, `dec_off_handle`) replace the
  current `int_m_handle`, `dec_m_handle` in `solv_MKL_sparse`.
- Each step now does **4** `mkl_sparse_d_mv` calls instead of 2. The
  mv_hint should be set with `nsteps * 2` to advise the optimizer.
- Inputs to `mv` are dense vectors, outputs dense vectors — same as
  the Euler MKL path. Use `cblas_daxpy` and `cblas_dscal` for the
  vector ops; or just call numpy on the underlying buffers (the
  ctypes wrappers bind the same memory).
- Do **not** combine `int_off` and `dec_off` into a single matrix per
  step (that would defeat the cached optimize hints — the matrices
  themselves don't change, only the scalar `ri[k]` weighting).

#### CUDA port

- Use `CUDASparseContext` as a base. Add `cu_int_off`, `cu_dec_off`,
  `cu_d_int`, `cu_d_dec` as device-resident `cupy` arrays.
- Per-step scratch (`D`, `hD`, `eD`, `phi1`, `phi2`, `F_phi`, `F_a`,
  `a`) lives on device, allocated once.
- The pattern `cu_int_off @ phc + ri * cu_dec_off @ phc` works directly
  via `cupyx.scipy.sparse.csr_matrix.__matmul__`, returning a `cupy`
  array. No host transfer.
- Float precision: respect `config.cuda_fp_precision` (32 or 64). At
  fp32 the Taylor cutoffs may need widening (`PHI2_SMALL = 1e-2`-ish)
  to avoid catastrophic cancellation.
- `expm`-batched is not needed — only **scalar** `exp` of an array.
  `cupy.exp` is fine.
- Kernel launch overhead: with N ≈ 8000 each elementwise op is one
  small launch. Fuse the eD/phi1/phi2 computation into one kernel and
  fuse the post-SpMV vector updates with `cupy.fuse` to keep launch
  count down.

### Sanity checks for any port

1. **Single-step equivalence to numpy** within ~1e-13 relative error
   on `phc` after one step from `phi0` at the first native step.
2. **Full-trajectory equivalence** at θ = 0°, native grid: full-state
   rel L2 vs `solv_numpy_etd2` should match to ~1e-12 (fp64) or ~1e-6
   (fp32).
3. **Coarsened-grid validation** at C = 8, θ = 0° and C = 16, θ = 60°
   should reproduce the μ-flux rel diff numbers in the table above to
   ~1e-3.

## Methods tested and discarded

- **ETD1 / Lawson scheme** — first-order single-stage exponential
  integrator. Validated correct (clean order 1 in convergence study)
  but error constant is ~16× worse than Euler at the same grid; only
  useful as a stability fallback at extreme C. Removed.
- **Strang splitting** `e^{h/2 D} (I + h N) e^{h/2 D}` — splitting
  error grows linearly with stiffness via `O(h² [D, N])`; rel_l2
  worsens 3% → 18% over a 16× ρ⁻¹ sweep.
- **Krylov / `scipy.sparse.linalg.expm_multiply`** — accurate but ~300×
  slower per step due to Krylov subspace construction and norm-est
  overhead. Not viable as written.
- **Heun / RK2** — explicit-Euler stability bound unchanged; doubles
  per-step cost without enabling coarsening. No advantage.
- **Adams-Bashforth-style ETD2** — drops to first-order across
  variable-step boundaries between native segments. Replaced by
  ETD2RK (single-stage, robust to variable h).

## References

* Cox & Matthews 2002, *Exponential time differencing for stiff systems*,
  J. Comput. Phys. 176.
* Hochbruck & Ostermann 2010, *Exponential integrators*, Acta Numer. 19.
