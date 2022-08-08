# Changes in MCEq since moving from MCEq_classic to the 1.X.X versions

Version 1.3.8:

- New default data file in preparation for 1.4 update
- Recomputed decay database and separated 3-body (3b) decays of Kaons
- Unpolarized 3b decay of kaons full accounted for helicity dependent calculations

Version 1.3.7:

- New interface to Apple Accelerate/vecLib library for accelerated computation on Apple Silicon macs
- Geometry/atmosphere interfaces moved to package level from redundant python source files
- Merged AArch64 on linux support from master
- Some advanced options to mix and match yields and inelastic cross sections from different models
- Different Z-Factor function in DDM

Version 1.3.5:

- Code formatter changed to black
- Additional functions to help returning total momentum or energy spectra (instead of kinetic)
- changing observation level `mceq.density_profile.set_h_obs()` triggers density spline recomputation

Version 1.3.4:

- DPMJET-III K0 bug discovered and worked around. K0S/L matrices were not generated properly. The workaround is to construct K0 distributions from a sum of K+ and K- with proportions determined from a fit of the Zfactors to the true K0S/L distributions. K0S is equal to K0L by definition in all of the models.
- Source dist fix. Package should now compile under Python 3.9 or other custom platforms easily via pip.

Version 1.3.3:

- Initialization moved almost entirely to GPU if available, matrix construction may be x2-x3 faster before
- Config defaults "auto" setting for `kernel_config` and respects other custom settings
- GPU sparse solver simplified
- Floating point precision defaults always to fp32 (see `config.floatlen`). MKL doesn't work with fp32 for some reason.

Version 1.3.1:

- Choice for different media with option `interaction_medium = 'air | water | rock | ice | co2 | hydrogen | iron`
- Medium can be selected by passing a keyword argument to `MCEqRun(...,medium='water',...)`
- Update to air interaction cross section, which is now consistently computed for mixture of N, O and Ar
- The pre-averaged cross section for air in SIBYLL2.3 models can be selected with medium='air-legacy'
- `A_target = 'auto'` will pic correct mass number for the selected medium
- Continuous losses taken into account for all charged particles, muons (PDG), electrons (ESTAR) and protons (PSTAR) have accurate tables. Generic "rescaled proton dEdX" for other charged particles if option `generic_losses_all_charged = True`
- The config flag `enable_cont_rad_loss = True` controls if radiative losses (bremsstrahlung) are included in the continuous loss terms or handled by an EM model
- Fall back option `fallback_to_air_cs` in case hadronic interaction matrices for selected medium not available
- MCEqRun.density_model.set_h_obs can be used to change observation level altitude
- Zenith angles > 90 accepted for h_obs > 0 since up-going cascades can develop from below the horizon (different to IceCube centered)
- deprecation warning forced for config access via dictionary (instead of module)

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
