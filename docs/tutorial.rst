.. _tutorial:

Tutorial
--------

The main user interface is the class :class:`MCEq.core.MCEqRun` that requires
a reference to a cosmic ray model for the initialization. Any cosmic ray flux model
from `the crflux package <http://crfluxmodels.readthedocs.org/en/latest/index.html#>`_
can be selected::

    import numpy as np
    import crflux.models as pm
    from MCEq.core import MCEqRun

    # Initalize MCEq by creating the user interface object MCEqRun
    mceq = MCEqRun(

        # High-energy hadronic interaction model
        interaction_model='SIBYLL23E',

        # Low-energy hadronic interaction model below 80 GeV (the default;
        # None switches the blending off)
        low_energy_model='FLUKA20251',

        # cosmic ray flux at the top of the atmosphere
        primary_model=(pm.HillasGaisser2012, 'H3a'),

        # zenith angle
        theta_deg=0,
    )

The code will raise an exception if a non-existent hadronic interaction
model is selected and will list the currently available models. All models
can be changed between calls to the solver.

Solving cascade equations
.........................

The solver is launched for the current set of parameters by::

    mceq.solve()

The integrator is an exponential time-differencing scheme of second order
(ETD2). Its implementation is selected with ``config.kernel_config``:
``"numpy_etd2"``, ``"mkl_etd2"``, ``"accelerate_etd2"`` (macOS) or
``"cuda_etd2"`` (NVIDIA GPU through cupy); the short names ``"numpy"``,
``"mkl"``, ``"accelerate"`` and ``"cuda"`` are accepted. The default ``"auto"``
takes the first available of Accelerate, MKL and numpy. CUDA is used only
when selected explicitly::

    from MCEq import config
    config.kernel_config = "cuda"

The spectrum of each particle species at the surface can be retrieved as numpy array with ::

    mceq.get_solution('mu+')

List available particle species managed by :mod:`MCEq.particlemanager`::

    mceq.pman.print_particle_tables(0)

To multiply the solution automatically with :math:`E^{\rm mag}` use ::

    mceq.get_solution('mu+', mag=3) # for E^3 * flux

To obtain a solution along the cascade trajectory in depth :math:`X`, create a
grid and pass it to the solver ::

    # A linearly spaced set of points from 0.1 up to the X value corresponding
    # to the depth at the surface `max_X` (for the selected zenith angle and atmospheric model/season)
    n_pts = 100
    X_grid = np.linspace(0.1, mceq.density_model.max_X, n_pts)

    mceq.solve(int_grid=X_grid)

To obtain particle spectra at each depth point::

    longitudinal_spectrum = []
    for idx in range(n_pts):
        print('Reading solution at X = {0:5.2f} g/cm2'.format(X_grid[idx]))
        longitudinal_spectrum.append(mceq.get_solution('mu+', grid_idx=idx))

To obtain the solutions at equivalent altitudes one needs to simply map the
the values of :math:`X` to the corresponding altitude for the **current** zenith
angle and atmospheric model::

    h_grid = mceq.density_model.X2h(X_grid)

To define a strictly increasing grid in X (strictly decreasing in altitude), use
the converter function between height and depth::

    h_grid = np.linspace(50 * 1e3 * 1e2, 0) # altitudes from 50 to 0 km (in cm)
    X_grid = mceq.density_model.h2X(h_grid)

    mceq.solve(int_grid=X_grid)

Particle numbers can be obtained by using predefined functions or by integrating
the spectrum. These functions support `grid_idx` (as shown above) and a minimal
energy cutoff (larger than the minimal grid energy :data:`MCEq.config.e_min`)::

    # Number of muons
    n_mu = mceq.n_mu(grid_idx=None, min_energy_cutoff=1e-1)

    # Number of electrons
    n_e = mceq.n_e(grid_idx=None, min_energy_cutoff=86e-3)

    # Number of protons above minimal grid energy
    n_p = np.sum(mceq.get_solution('p+', integrate=True))

All particles listed by
:meth:`MCEq.particlemanager.ParticleManager.print_particle_tables` are
available to :meth:`MCEq.core.MCEqRun.get_solution`.

Changing geometrical and atmospheric parameters
...............................................

To change the zenith angle ::

    mceq.set_zenith_azimuth(<zenith_angle_in_degrees>)

Most geometries support angles between 0 (vertical) and 90 degrees.

To change the density profile ::

    mceq.set_density_model(('MSIS00', ('SoudanMine', 'June')))

The default is ``('CORSIKA', ('BK_USStd', None))``. The model names are the
keys of :data:`MCEq.core.DENSITY_MODELS`:

- 'CORSIKA' - Linsley parameterizations from the CORSIKA air-shower MC
  (see :class:`MCEq.geometry.density_profiles.CorsikaAtmosphere`).
- 'MSIS00', 'MSIS00_IC', 'MSIS00_KM3NeT' - NRLMSISE-00 (Picone et al. 2002),
  bundled as a C extension.
- 'MSIS21', 'MSIS21_IC', 'MSIS21_KM3NeT' - NRLMSIS 2.1 (Emmert et al. 2021);
  needs the separately installed ``nrlmsis`` package
  (``pip install 'git+https://github.com/afedynitch/nrlmsis2.1'``).
- 'AIRS' - tabulated AIRS satellite data at the South Pole, extended above the
  tables with NRLMSISE-00. The tables are not provided; pass their directory
  as ``table_path``: ``('AIRS', ('SouthPole', 'January', True, '/path/to/airs'))``.
- 'Isothermal' - simple isothermal model with scale height 6.3 km.
- 'GeneralizedTarget' - piece-wise homogeneous density.

The two MSIS families share one interface. The configuration is
``(location, month)`` with ``location`` a key of
:data:`MCEq.geometry.atmosphere_parameters.LOCATIONS`, and the atmosphere is
evaluated at local noon of the first day of that month. The ``_IC`` variants
are centred on IceCube at the South Pole, the ``_KM3NeT`` variants take
``'ORCA'`` or ``'ARCA'`` as location; both accept zenith angles up to 180°
(upgoing).

Columns end at the observation level ``config.h_obs`` (in m). The default
``None`` puts it at the ground of the atmosphere model: sea level for most
models, the ice surface at 2834 m for the CORSIKA South Pole sets
(``'SouthPole'``, ``'PL_SouthPole'``, ``'SDR_SouthPole'``) and for AIRS,
which are fits above it. A level below the ground raises ``ValueError``; a
higher one, e.g. a mountain site, is set with ``config.h_obs`` before the
model is built or with ``mceq.density_model.set_h_obs(h_cm)``. The
detector-centred ``_IC`` and ``_KM3NeT`` models place the level themselves.

Refer for more info to :ref:`densities`.

After changing the models, the spectra can be recomputed with :meth:`MCEq.core.MCEqRun.solve`.

Batched solves: many angles, days, or primaries in one go
.........................................................

A loop of ``solve()`` calls with the same interaction model runs faster as one
batched call, because the batch members share one transport operator. Pick the
entry point by what varies:

===============================================  =============================================
What varies across the batch                     Use
===============================================  =============================================
Only the initial spectrum (``phi0`` columns)     :meth:`MCEq.core.MCEqRun.solve_batch` with
                                                 ``conditions=None``
A (zenith × azimuth) sky grid                    :meth:`MCEq.core.MCEqRun.solve_fullsky`
Directions, days/atmospheres, or a mix           :meth:`MCEq.core.MCEqRun.solve_batch` with
                                                 ``conditions=[...]``
===============================================  =============================================

All variants work with every kernel and return an
:class:`MCEq.batch.MCEqBatchResult`, from which the spectra of each member are
read by name::

    # 1) a scan over single-primary energies (e.g. a response matrix)
    columns = []
    for E in mceq.e_grid[90::10]:
        mceq.set_single_primary_particle(E, pdg_id=2212)
        columns.append(mceq.get_initial_state())
    res = mceq.solve_batch(np.stack(columns, axis=1))
    res.get_solution("total_mu+", k=3, mag=3)

    # 2) one spectrum through 4 months x 3 zenith angles
    months = ["January", "April", "July", "October"]
    conditions = [
        {"zenith_deg": z, "density_model": ("MSIS00", ("SouthPole", month))}
        for month in months for z in (0.0, 30.0, 60.0)]
    res = mceq.solve_batch(conditions=conditions)
    res.get_solution("conv_numu", k=5)

    # 3) a full sky grid (adds the geomagnetic cutoff + skymap helpers)
    zenith_grid = np.arange(0.0, 90.0, 10.0)
    azimuth_grid = np.arange(0.0, 360.0, 30.0)
    res = mceq.solve_fullsky(zenith_grid, azimuth_grid)
    res.get_solution("conv_numu", zenith=60.0, azimuth=90.0)
    res.skymap("conv_numu", 100.0)      # (n_zen, n_az) map at 100 GeV

Initial-state columns are built with the setters of the run
(:meth:`~MCEq.core.MCEqRun.set_single_primary_particle`,
:meth:`~MCEq.core.MCEqRun.set_initial_spectrum`,
:meth:`~MCEq.core.MCEqRun.set_primary_model`) and copied with
:meth:`~MCEq.core.MCEqRun.get_initial_state`. A 1-D ``phi0`` (or
``phi0=None`` for the current initial state) is broadcast to all batch
members; a 2-D ``(dim_states, K)`` array carries one column per member.

``int_grid`` snapshots are available only for ``conditions=None``.
``path_workers=N`` builds the integration paths of a zenith/azimuth batch in a
pool of ``N`` forked processes; it is not available for batches whose
conditions set ``density_model``.

For sky grids, ``solve_fullsky`` applies the geomagnetic rigidity cutoff when
``geomagnetic_cutoff=True`` is passed to it or to the ``MCEqRun`` constructor;
it is off by default, so ``solve_fullsky`` agrees with
:meth:`~MCEq.core.MCEqRun.solve`. It needs an
atmosphere with a location: ``USStd``, ``BK_USStd`` and the averaged
``KM3NeT`` set have none. Atmospheres of a named site use cutoff maps
shipped with MCEq (downgoing directions on a 5° grid, IGRF at
:data:`MCEq.geometry.cutoff.CUTOFF_DATE`); detector-centred MSIS atmospheres
trace the full sky with gtracr on first use and cache the map under
``<config.data_dir>/gtracr_cutoffs/``. Without an ``azimuth_grid`` the
transmitted primary flux is averaged over azimuth. Passing a 2-D ``phi0``
puts the caller in charge of the per-pixel primary — the cutoff is then
*not* applied on top, with a warning.

The ``Full_sky_carousel`` example notebook walks through the sky-grid
API: serial scan baseline, one-shot carousel, per-pixel ``phi0``, and
the cutoff at a mid-latitude detector.

Changing hadronic interaction models
....................................

To change the hadronic interaction model ::

    mceq.set_interaction_model('EPOS-LHC')

The models in the v2 data release are:

- FLUKA20251 (low-energy model, used together with a high-energy model)
- SIBYLL23E
- SIBYLL23ESTARBAR, SIBYLL23ESTARMIXED, SIBYLL23ESTARRHO,
  SIBYLL23ESTARSTRANGE (SIBYLL★ variants)
- SIBYLL23D
- SIBYLL21
- EPOSLHCR
- EPOSLHC
- QGSJETIII
- QGSJETII04
- DPMJETIII193
- PYTHIA8CASCADE
- PYTHIA8ANGANTYR

The default database ``mceq_base_air_1d_v2.h5`` holds FLUKA20251 and
SIBYLL23E; the other models are downloaded with the extra-models package on
first use (see :doc:`data_installation`). The model name is converted to
upper case and dashes and dots are removed, so ``'EPOS-LHC'`` and
``'EPOSLHC'`` select the same model.

MCEq will take care of updating all data structures regenerating the matrices. This call
takes some time since data memory needs to be allocated and some numbers crunched. If you
use this function in a loop for multiple computations, put it further out.

Changing cosmic ray flux model
..............................

The flux of cosmic ray nucleons at the top of the atmosphere (primary flux) is the initial condition. The
module :mod:`crflux.models` contains a contemporary selection of flux models. Refer to the
`crflux documentation <https://crfluxmodels.readthedocs.io/en/latest/>`_ or
`the source code <https://github.com/afedynitch/crflux>`_.

To change the primary flux use :meth:`MCEq.core.MCEqRun.set_primary_model` ::

    import crflux.models as pm

    mceq.set_primary_model(pm.HillasGaisser2012, 'H3a')

Using MCEq for air-showers
..........................

MCEq currently provides solutions of the one-dimensional (longitudinal) cascade equations in
the variable X (depth). Therefore, full air-shower calculations including the lateral (transverse)
extension of particle densities are not possible. What is possible is the computation of longitudinal
profiles of particle numbers or depth dependence of spectra. The only difference between "air-shower mode"
and the standard "inclusive flux modes" is the initial condition. For air-showers the initial condition
is a single particle of a certain type and fixed energy, instead of an entire spectrum of cosmic
ray nucleons as described above. To launch a cascade from a single particle use
:meth:`MCEq.core.MCEqRun.set_single_primary_particle` ::

    # For a 1 EeV proton
    mceq.set_single_primary_particle(1e9, pdg_id=2212)

    # Or for a 1 EeV iron nucleus
    mceq.set_single_primary_particle(1e9, corsika_id=5626)

The zenith angle has to be set as shown above with :meth:`MCEq.core.MCEqRun.set_zenith_azimuth`.
