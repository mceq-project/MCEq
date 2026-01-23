# MCEq CUDA Solver Modernization Summary

## Issues Fixed

1. **CuPy Interface Deprecation**: Updated CUDA solver to use modern `cupyx.scipy.sparse` interface instead of deprecated `cupy.cusparse.csrmv` which was removed in newer CuPy versions.

2. **NumPy Solver Input Mutation Bug**: Fixed critical bug where NumPy solver modified input arrays in place, causing subsequent solver calls to use wrong initial conditions.

3. **MKL Library Detection**: Fixed IndexError when no MKL libraries are found in the environment.

## Code Changes

### /src/MCEq/solvers.py
- **Line 28**: Added `.copy()` to prevent input array modification: `phc = phi.copy()`
- **Lines 70-78**: Updated imports and error message for CuPy
- **Lines 115-123**: Modernized CUDA `solve_step` method to use `cupyx.scipy.sparse` and `@` operator

### /src/MCEq/config.py
- **Lines 242-243**: Added safe MKL library detection with fallback to prevent IndexError

### /pyproject.toml  
- **Lines 33-34**: Added optional CUDA dependency: `cuda = ["cupy-cuda12x>=12.0.0"]`

### /tests/test_solver_regression.py (New File)
- Added regression tests to prevent future occurrences of these bugs
- `test_solv_numpy_does_not_modify_input_phi`: Ensures input arrays aren't modified
- `test_cuda_numpy_solver_consistency`: Validates CUDA/NumPy solver agreement

## Installation

### For CPU-only usage:
```bash
pip install MCEq
```

### For CUDA support:
```bash
pip install MCEq[cuda]
# or manually: pip install cupy-cuda12x>=12.0.0
```

## Version Requirements

- **CuPy >= 12.0.0** required for CUDA functionality
- Older CuPy versions are not supported due to deprecated interface removal

## Test Results

All solver tests now pass:
- ✅ `test_solv_numpy_runs`
- ✅ `test_solv_CUDA_sparse_matches_numpy` 
- ✅ `test_solv_MKL_sparse_matches_numpy`
- ✅ `test_solv_numpy_does_not_modify_input_phi` (new)
- ✅ `test_cuda_numpy_solver_consistency` (new)

The CUDA and NumPy solvers now produce numerically matching results (differences only in 1e-8 range due to float32 vs float64 precision).
