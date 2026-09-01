``MCEq.solvers`` is a package: ``numerics``, ``etd2``, ``path``, ``schedule``
and ``backends/{base,host,mkl,cuda,accelerate}``. Every name the module
exported as public surface is still importable from ``MCEq.solvers``; the
private helpers now live in the module that owns them, so
``MCEq.solvers._PRECISION_CONTRACT`` is
``MCEq.solvers.backends.base._PRECISION_CONTRACT`` and
``MCEq.solvers._cuda_etd2_kernels`` is
``MCEq.solvers.backends.cuda._cuda_etd2_kernels``.

Importing ``MCEq.solvers`` stays free of platform work: no backend module
dlopens a library, imports cupy or probes for a device at import time.
