`MCEq.config` is a package. Platform, MKL, CUDA and Accelerate detection moved
to `MCEq.config.detect` and resolves on first read instead of at import, so
importing MCEq no longer dlopens a BLAS, probes a GPU or decides which kernel
will run; `import MCEq.config` costs 0.07 s instead of 0.43 s. The database
download helpers moved to `MCEq.download`. Every import style of `config` keeps
resolving to the same mutable module.
