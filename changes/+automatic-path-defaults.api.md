Select ETD2 defaults from the assembled system: `eps=.03`, `dX_max=5` g/cm²
for 1D hadronic/lepton transport, and `.01`, `2` for 2D or retained electron
states. Configuration values of `None` select these defaults; explicit
values and matrix-derived step caps are preserved. Warn without overriding
low-energy CUDA FP32 2D secant solves because angular precision errors can
exceed one percent. Assemble 1D operators directly as sparse matrices and
reuse HDF5 read handles within loading scopes.
