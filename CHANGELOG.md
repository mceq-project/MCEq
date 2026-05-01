

<!-- towncrier release notes start -->

# MCEq 1.4.1 (2026-03-16)

## Bug Fixes

- Fixed segfault at interpreter exit on macOS when the Accelerate (spacc) solver is used. The `atexit` handler and `SpaccMatrix.__del__` both tried to free the same sparse matrix handles, causing a double-free via `sparse_matrix_destroy(NULL)`. Removed the redundant `atexit` handler (cleanup is handled solely by `__del__`) and added a NULL guard in the C-level `free_mstore_at` as defence-in-depth.

  Bump version to 1.4.1 for a new bugfix release. ([#150](https://github.com/mceq-project/MCEq/pull/150))
- Fixed an ordering bug in tests that could result in uncontrolled model settings in integration_path tests when executed with `pytest-xdist`. ([#151](https://github.com/mceq-project/MCEq/pull/151))

## Maintencance

- Defer the MCEq database download from import time to the first ``MCEqRun``
  instantiation via ``config.ensure_db_available()``. This prevents the large
  default database from being downloaded when ``MCEq.config`` is merely imported
  (e.g. during test collection). CI workflow now uses the ``cache-hit`` output of
  ``actions/cache`` to reliably skip redundant downloads. ([#147](https://github.com/mceq-project/MCEq/pull/147))


# MCEq 1.4.0 (2026-03-04)

## Bug Fixes

- Fixes `numpy.trapz` import for numpy>=2.4.
  Fixes `tests/test_core.py::test_solve_skip_integration_path`.
  Updates pre-commit to use ruff. ([#127](https://github.com/mceq-project/MCEq/pull/127))
- Adding or removing particles in MCEq after initialisation of a `MCEqRun` instance required a call of the 
  `regenerate_matrices` method in order to propagate these changes to the calculation of the flux.
  This method did not automatically resized the flux vector, which lead to a shape mismatch when calling `solve()`.
  We fix this by adding `_resize_vectors_and_restore` internally to the call to `regenerate_matrices`. ([#132](https://github.com/mceq-project/MCEq/pull/132))
- Changes optional argument `pdg_id` in `set_initial_spectrum` to be a required argument. ([#135](https://github.com/mceq-project/MCEq/pull/135))
- Fixed a bug where MCEq failed to load when running mkl with Python < 3.12 on windows. ([#138](https://github.com/mceq-project/MCEq/pull/138))
- Auto detection of leading eigenvalues setting.
  Changing `config.leading_process` is now effective.
- Deprecation warning forced for config access via dictionary (instead of module)
- Running MCEq with reduced energy did not account for correct mixing if the mixing energy lies outside the MCEq maximum energy.
  This is fixed now by checking for threshold < hybrid_crossover.

## Maintencance

- Updated the Docstring of `MCEqRun.get_solution`. ([#133](https://github.com/mceq-project/MCEq/pull/133))
- Removed an unnecessary `else` in `set_density_model` that was already catched. ([#136](https://github.com/mceq-project/MCEq/pull/136))
- Update the README ([#137](https://github.com/mceq-project/MCEq/pull/137))
- Updated the CI to run on macos-latest and macos-26.
  Updated the CI to run on all python versions from 3.9 to 3.14 ([#138](https://github.com/mceq-project/MCEq/pull/138))
- Geometry/atmosphere interfaces moved to package level from redundant python source files.

## Documentation Updates

- Updated the citation list to the new hadronic interaction models.
  Merged both citation pages in the docs to one page. ([#134](https://github.com/mceq-project/MCEq/pull/134))

## New Features

- This is large PR merging changes towards v1.4 over the years.

  Most significant we update MCEq to provide a new Database with updated models.
  With this database cross-sections, hadronic yields, and decay yields get updated.

  For a more in detail description of v1.4 read the updated [Documentation](https://mceq.readthedocs.io/en/latest/) of MCEq. ([#128](https://github.com/mceq-project/MCEq/pull/128))
- Additional functions to help returning total momentum or energy spectra (instead of kinetic).
- Choice for different media with option `interaction_medium = 'air | water | rock | ice | co2 | hydrogen | iron`.
  Not all are yet available in this release.

  `A_target = 'auto'` will pic correct mass number for the selected medium.

  Medium can be selected by passing a keyword argument to `MCEqRun(...,medium='water',...)`.
- Config defaults "auto" setting for `kernel_config` and respects other custom settings.
  Accelerated backends are preferred over numpy.
- Continuous losses taken into account for all charged particles, muons (PDG), electrons (ESTAR) and protons (PSTAR) have accurate tables. Generic "rescaled proton dEdX" for other charged particles if option `generic_losses_all_charged = True`.
  Default is `False`.
- Fall back option `fallback_to_air_cs` in case hadronic interaction matrices for selected medium not available
- Some advanced options to mix and match yields and inelastic cross sections from different models
- The Data Driven Model (DDM) is now available.
- The config flag `enable_cont_rad_loss = True` controls if radiative losses (bremsstrahlung) are included in the continuous loss terms or handled by an EM model. This is the new default!
- `Accelerate` backend is now available on macOS.
- `MCEqRun.density_model.set_h_obs` can be used to change observation level altitude


# MCEq 1.3.1 (2025-11-05)

## Bug Fixes

- Fixes `z_factor` calculation if minimal MCEq energy is above 2 GeV. Just failed before. ([#81](https://github.com/mceq-project/MCEq/pull/81))
- Fixes/Updates backends for MCEq:
  1. CUDA: Update from `cupy.cusparse.csrmv` to newer `cupyx.scipy.sparse`
  2. Numpy: Fixed bug where numpy solver mutates the input arrays, resulting in wrong solutions for subsequent calls of `MCEqRun.solve()`
  3. MKL: Fixed error due do not being able to find MKL library ([#83](https://github.com/mceq-project/MCEq/pull/83))
- Threshold cross-over between resonance approximation and mixing was off-by-one.
  1. Treats particles that are not in the standard_particle list as resonances.
  2. Fixes condition for finding the mixing energy from `threshold>cross_over` to `threshold>=cross_over`. ([#100](https://github.com/mceq-project/MCEq/pull/100))
- Fixes correctly setting the user-provided interaction model in the `init` of `MCEqRun`. Initialization failed if `SIBYLL2.3C` was not included in the database. ([#106](https://github.com/mceq-project/MCEq/pull/106))

## Maintencance

- Update the MKL backend of MCEq to use more modern `oneMKL` library. ([#63](https://github.com/mceq-project/MCEq/pull/63))
- Cleaning up tests of MCEq. ([#64](https://github.com/mceq-project/MCEq/pull/64))
- Adding new tests to `MCEq.core`. Enhances test coverage! ([#67](https://github.com/mceq-project/MCEq/pull/67))
- Adding new tests to `MCEq.solvers`. Enhances test coverage! ([#74](https://github.com/mceq-project/MCEq/pull/74))
- Adding new tests to `MCEq.geometry`. Covers densities and atmospheres. Enhances test coverage! ([#77](https://github.com/mceq-project/MCEq/pull/77))
- Removes the `WHR_charm` class! Adds tests for `MCEq.charm_models`. Enhances test coverage! ([#80](https://github.com/mceq-project/MCEq/pull/80))
- Adding `towncrier` to the project. This enables simple and meaningful generation of future changelogs. ([#109](https://github.com/mceq-project/MCEq/pull/109))

## Documentation Updates

- Completely updates the MCEq documentation to new `pydata_sphinx_theme`. Enjoy it on [mceq.readthedocs.io](https://mceq.readthedocs.io)! ([#82](https://github.com/mceq-project/MCEq/pull/82))
- Adds `intersphinx` mappings to the Documentation. Enables cross-referencing to other external documentations. ([#91](https://github.com/mceq-project/MCEq/pull/91))
- Fixes the ReadTheDocs build by changing the install method of the `docs` dependencies in `.readthedocs.yml`. ([#93](https://github.com/mceq-project/MCEq/pull/93))


Version 1.3.0:

- New buildsystem. Uses [scikit-build.core](https://github.com/scikit-build/scikit-build-core).
- `mceq_config` deprecated, use `import MCEq.config`
- Updated tests
- New atmosphere models for the South Pole
- Some linting


Version 1.2.6:

- Fixed a bug where alternating changes between up- and downgoing zenith angles are not detected correctly and the same results returned

Version 1.2.5:

- Migrated to github actions for CI. Thx @jncots

Version 1.2.3:


- Binary wheel for aarch64 on linux. Thanks to @obidev

Version 1.2.2:


- Added wheels for Python 3.9 on 64 bit systems
- Removed binary wheels for 32bit due to lack of h5py wheels and mainstream has transitioned to 64bit. 32bit users can build MCEq from source.

Version 1.2.1:

- Some cleanup and new convenience functions on MCEqParticle
- Auto detection of leading eigenvalues setting
- Mixing energy more robustly calculated
- Stopping power for all charged hadrons enabled by option 'generic_losses_all_charged' (req. new data file)
- Default minimal energy increased to 1 GeV because it's safe under all conditions (no swing)
- get_AZN function fixed to return integers only (thx Max)

Version 1.2.0:

- New data tables: physics will be affected mostly low energies < 30 GeV and minor
corrections can be visible for particle ratios at higher energies.
[See dedicated doc page](http://mceq.readthedocs.org/en/latest/v12v11_diff.html).
- tests have been updated to the new version and will fail if used with the old database file
- SIBYLL23C release is updated to patch level 04 instead of the previous 01. The results are very similar and changes are smaller than in CORSIKA because MCEq uses the air target and not the Nitrogen/Oxygen mix.
- QGSJET tables had bugs and there are more pronounced changes
- Projectile equivalence tables updated (thx to [CORSIKA8 team](https://www.ikp.kit.edu/corsika/88.php))
- Documentation badge
- Minor (cosmetic and technical) updates of crflux and particletools packages
- crflux includes the spline for the GlobalSplineFitBeta class and will be updated during install.  
- set_density_profile accepts a density object as parameter in parallel to the previous definition
- some config values that can produce inconsistent results (A_target for ex.) are saved in the objects that can trigger such inconsistencies. Changing config values in runtime should be more safe, but not free of failures. It is still not recommended to change config values in runtime if this can be avoided.

Version 1.1.3:

- Added atmospheres for KM3NeT by @Kakiczi (<https://github.com/Kakiczi>)
- new keyword for MCEqRun "build_matrices"=False (to prevent matrix building on init)
- Equivalent projectile mappings separated for SIBYLL 2.1 and 2.3  

Version 1.1.2:

- Hotfix for MKL library handler

Version 1.1.0:

Minor version bump because of an interface change in some convenience functions, for example
dNdxlab in particledata. Those now consistently accept kinetic energy arguments. In the
previous versions some of these functions required laboratory energies others kinetic, that
may have generated some confusion. Other changes include:

- multiple calls to `set_single_particle` can define an initial state by using an the `append` flag
- updated to crflux 1.0.3 (Windows compatibility)
- Flux and result array are re-created when interaction model changes (not resized)
- Fixed ctypes bug in NRLMSISE
- Long description fixed in setup.py
- added license to scikit-hep project's azure-build-helpers by Henryii@github
- build includes Python 3.8 binaries (except for 32bit Linux)
- tests moved into MCEq package
- improved ctypes library finding
- new convenience function MCEqRun.closest_energy to obtain the closest grid point
- new convenience functions mceq_config.set_mkl_threads allows setting thread count in run time
- new CorsikaAtmosphere location "ANTARES/KM3NeT-ORCA"
- tests for atmospheres

Version 1.0.9:

- disable_decays flag in advanced options fixed
- threshold energy not used in n_mu, n_e
- new generic function 'n_particles' for arbitrary particle types
- new config option dedx_material

Version 1.0.8:

- Fixed a Python3 compatibility issue in density profiles
- Cross checked and corrected the functionality of "disabled particles" in config file
- Version tagged for paper submission

Version 1.0.6 and 1.0.7:

- A few typos corrected

Version 1.0.5:

- Check added to make sure depth grids are strictly increasing
- Tutorial updated to reflect this fact
- New advanced variable in config "stability_margin"
- New method in MCEqRun to set the entire spectrum for combinations
    of particles as initial condition

Version 1.0.4:
    First official version distributed over PyPi

Version 1.0.0:

General remark::
    This is a major rewrite of the MCEq core code. Mostly obsolete stuff is removed.
    The interfaces to the various data sources are handled in a different way via the
    container 'ParticleManager'. Each MCEqParticle object knows its properties, for
    instance if it can decay or interact, and if requires to be included in continuous
    losses. The matrix generation could be simplified a lot and the class MCEqRun became
    what it was meant to be - a user interface. Several experimental and less successful
    solver implementations have been removed. The splitting between energy and time solvers
    is not necessary anymore since the only successful energy (derivative) solver is the
    one based on DifferentialOperators that are not part of the "C Matrix". With the new
    structure it became simpler to exchange physical models or track particle decays of
    any kind.
