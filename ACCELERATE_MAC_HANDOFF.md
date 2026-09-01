# Hand-off: finish the Accelerate backend on macOS

**Temporary file. Delete it in the commit that finishes the work** — `AGENTS.md`
asks that planning notes stay out of this repo, and this one exists only because
the work below cannot be done on the machine that started it.

Branch `refactor/00-golden`, commit `2139899` "refactor(solvers): Accelerate onto
the driver". Nothing is pushed.

## What happened, and why it needs you

Accelerate was the last kernel family with its own step loops. `2139899` turned
it into a third `apply_off` binding of `HostBackend` — `SpaccApplyOff` beside
`ScipyApplyOff` and `MklApplyOff` — and deleted
`solv_spacc_etd2{,_multirhs,_multirhs_f32,_carousel}` (647 lines),
`SpaccMatrixF32`, `MCEqRun._legacy_handles` and the unreachable `"spacc_etd2"`
alias. `SpaccMatrix` now takes `dtype=` and selects the `*_double` / `*_float`
entry points from one table.

It was written and reviewed on Linux, where `config.has_accelerate` is `False`
and `import MCEq.spacc` raises (no `libspacc` to load). **Not one line of the
real Accelerate path has ever executed.** `spacc.c` is deliberately untouched, so
you can check out this commit and run against an existing build without a
rebuild.

What *was* verified there: `tests/test_spacc_apply_off.py` drives `SpaccApplyOff`
through a fake handle that accepts only ctypes pointers and honours `ldb`/`ldc`
as column strides. All 12 cases (fp64 and fp32 × K ∈ {1, 3, 70} × scalar and
lane `ri`) are bitwise equal to the scipy binding, and 17 deliberate layout,
tiling and `ri` mutations are all caught. So the pointer bookkeeping is sound;
what is unproven is everything involving the actual library.

## 1. Run the tests that have never run

```bash
pip install -e ".[test]"
pytest tests/test_spacc_apply_off.py -q          # must pass anywhere; run it first
pytest tests/ -q -k "spacc or accelerate" -p no:randomly
pytest tests/ -q -p no:randomly                  # then the whole suite
```

19 ids skip on Linux and will execute for the first time on your machine:

- `tests/test_solvers.py`: `test_spacc_matrix_creation[float64|float32]`,
  `test_spacc_gemv_matches_scipy[float64-1e-12|float32-1e-05]`,
  `test_spacc_double_del_is_safe`, `test_spacc_del_with_none_store_id`,
  `test_spacc_matrix_store_full`, `test_solve_multirhs_dtype_float32`,
  `test_solve_etd2_accelerate_multirhs_matches_numpy_multirhs_toy[1|4|16|70]`,
  `test_solve_etd2_accelerate_matches_numpy_etd2_{toy,real}`
- `tests/test_exit.py`: the four Accelerate / mstore subprocess tests
- `tests/test_solvers_2d.py::test_2d_accelerate_matches_numpy` (needs
  `mceq_db_v2_fluka2d_rc7.h5`)

Two of these carry a specific risk:

- **`K = 70` is new** and is the only case that crosses the 64-column tile
  boundary. Tile widths are asserted on Linux against the fake; what is unproven
  is that Accelerate accepts a tile base pointer with `ldb = ldc = dim`. The
  deleted loops passed the identical address, computed as
  `c_double.from_address(base + c0*dim*8)` rather than a numpy view's
  `data_as`, so this should hold — but a wrong `ld` reads out of bounds and
  faults rather than failing an assertion.
- **`test_spacc_matrix_explicit_del_exit` now allocates 4 fp64 + 1 fp32
  `SpaccMatrix`** where it used to allocate 5 fp64. That is the first script to
  mix typed slots in one `mstore` pool. `free_mstore_at` calls the type-agnostic
  `sparse_matrix_destroy` for both, and `SpaccMatrixF32` already used that same
  free path, so it should be fine — confirm it, because the failure mode is a
  crash at interpreter exit.

## 2. The A/B benchmark — the one number nobody has

`SpaccApplyOff` stages the driver's row-major `(dim, K)` state through
Fortran-ordered scratch, because Accelerate has no row-major SpMM. At `K > 1`
that is **four transposing `(dim, K)` copies per step** — `etd2_driver` calls
`apply_off` twice, each copying in and out — where the deleted kernels kept the
state column-major end to end and did none. At `dim = 171360, K = 64, fp64` each
copy touches ~87 MB of strided traffic. The tile size is unchanged, so the SpMM
operating point is identical and **the copies are the entire delta**. At `K = 1`
there is no staging at all.

Compare `2139899` against its parent `46f7cb3` (which still has the old loops),
same database, same paths, on: single axis (`K = 1`), multi-RHS at `K = 8` and
`K = 64`, and the LPT carousel. `scripts/bench_etd2_steps.py` in the
`mceq-em-integration` project repo does exactly this shape — one init, truncated
paths at 50/200 steps, ms/step read off as the slope — but it is not in this
repo, so either copy it over or time `MCEqRun.solve()` directly with a fixed
step count.

Expected outcome: within a few percent at `K = 1`, some loss at large `K`. If the
loss at `K = 64` is worse than roughly 10%, take the escape hatch below.

## 3. Escape hatch if the copies cost too much

Accelerate's `sparse_matrix_product_dense_double` takes a `CBLAS_ORDER` as its
first argument. `spacc.c:71` and `spacc.c:111` hardcode `CblasColMajor`. Passing
`CblasRowMajor` instead — plumbed through as an argument, the way MKL's
`layout=101/102` already is in `MklApplyOff` — removes the staging entirely and
makes `SpaccApplyOff` a near-clone of `MklApplyOff` with no copies. It was not
done on Linux precisely because C cannot be compiled or benchmarked there, and
changing it would have forced you to rebuild before you could measure anything.

If you take this route, `ldb`/`ldc` become `K` rather than `dim`, the tiling
loop changes meaning (row-major tiles are not contiguous column blocks — most
likely drop the tiling and re-measure), and `tests/test_spacc_apply_off.py`'s
fake must be updated to honour the order argument. Do not skip that last part:
the fake is the only thing standing between a layout bug and a silent wrong
answer on the one platform CI never sees.

## 4. Dead code you can now delete

These lost their last caller when the four loops went. They were left in place
because neither half can be compiled or re-benchmarked on Linux, and deleting
only the Python half would have desynchronised the two.

- `src/MCEq/spacc/spacc.c`: `daxpy` (169–172) and the eight ETD2 post-apply
  kernels (174–355) — `etd2_post_apply{1,2}_{multirhs,multipath}` and their four
  `_f32` siblings. `HostBackend` uses the row-major fused kernels from
  `MCEq.etd2_kernels` instead. `test()` (417–444) is also unreferenced.
- `src/MCEq/spacc/__init__.py`: the matching `argtypes` blocks and module-level
  aliases for those symbols, plus `daxpy` and `spacc.test.restype`. They carry a
  comment marking them dead.

Confirm with `grep -rn "etd2_post_apply\|daxpy" src/ tests/` — the only survivors
should be the `_rowmajor` names from `MCEq.etd2_kernels` and the CUDA kernel
names in `solvers/backends/cuda.py`. Rebuild and re-run the suite after removing
them; that is the whole point of doing it on a Mac.

## 5. Two smaller things to judge

- **mstore pressure.** Accelerate handles moved from the flat `_legacy_handles`
  keys into `MCEqRun._backend_cache`, keyed `(kernel, precision, coupled)`. A run
  that toggles both the sec(θ) coupling and precision can now hold four backends
  = 8 slots against `#define SIZE_MSTORE 10`. Not reachable in the normal
  single-database flow, but `core.py:867` records this pool overflowing before
  (PR #163), and the failure is a hard `"Matrix creation failed."`. Consider
  raising `SIZE_MSTORE` while you have a compiler.
- **The coupled route on Accelerate is newly reachable.** `_resolve_secant` used
  to list Accelerate as a configuration without a secant route and silently fall
  back to the paraxial transport; that blocker is gone, since the coupled route
  is driver code plus `apply_off`. It was verified on Linux against a fake handle
  (bit-identical to numpy on the 2D FLUKA rc7 operator), never against the real
  SpMM. `test_2d_accelerate_matches_numpy` is paraxial, so add a coupled 2D
  comparison — `config.secant_theta_transport = "require"`, Accelerate vs numpy —
  before trusting it. fp32 *with* coupling remains CUDA-only by design.
