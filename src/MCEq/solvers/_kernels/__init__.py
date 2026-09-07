"""Compiled kernels owned by the solvers layer.

Each subpackage wraps one compiled extension: :mod:`._kernels.etd2` the
fused ETD2 step stages, :mod:`._kernels.spacc` the Accelerate SpMM
(macOS only, moved at M12). The extensions are loaded lazily by the
backends, never at ``MCEq.solvers`` import time.
"""
