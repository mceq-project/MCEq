"""Compiled kernels for the environment layer.

``nrlmsise00`` (the NRLMSISE-00 model) and ``corsikaatm`` (CORSIKA's density
and overburden helpers) are C extensions built by CMake. Both loaders locate
their shared object relative to ``__file__``, so the ``.so`` has to sit beside
the ``__init__.py`` that loads it -- ``CMakeLists.txt``'s install DESTINATION
is the one place the layout is written down.
"""
