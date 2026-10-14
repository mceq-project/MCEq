"""Compiled kernels for the environment layer.

``nrlmsise00`` (NRLMSISE-00) and ``corsikaatm`` (CORSIKA density and overburden)
are C extensions built by CMake; each loader finds its shared object beside its
``__init__.py``.
"""
