The compiled ETD2 step stages moved from `MCEq.etd2_kernels` to
`MCEq.solvers._kernels.etd2`, where the kernels sit in the layer that
owns them. No flat shim: the module is private (user census 0), and
its ctypes wrapper and CMake install path moved with it. `spacc`
follows at the macOS half of the move.
