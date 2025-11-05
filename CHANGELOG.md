# Changes in MCEq since moving from MCEq_classic to the 1.X.X versions:

<!-- towncrier release notes start -->

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
