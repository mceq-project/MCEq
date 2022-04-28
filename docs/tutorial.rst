.. _tutorial:

Tutorial
--------

The main user interface is the class :class:`MCEq.core.MCEqRun` that requires
a reference to a cosmic ray model for the initialization. Any cosmic ray flux model
from `the crflux package <http://crfluxmodels.readthedocs.org/en/latest/index.html#>`_.
can be selected::

    from MCEq.core import MCEqRun
    import crflux.models as crf

    # Initalize MCEq by creating the user interface object MCEqRun
    mceq = MCEqRun(

        # High-energy hadronic interaction model
        interaction_model='SIBYLL23C',

        # cosmic ray flux at the top of the atmosphere
        primary_model = (crf.HillasGaisser2012, 'H3a'), 
        
        # zenith angle
        theta_deg = 0. 
    )

The code will raise an exception of a non-existent hadronic interaction
model is selected and will list the currently available models. All models
can be changed between calls to the solver.

Solving cascade equations
.........................

The solver is launched for the current set of parameters by::

    mceq.solve()

By default MCEq will pick the numpy, MKL or the CUDA solver, depending on the
the installed packages. Currently only 'forward-euler' solvers are available,
which are fast and stable enough.

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
        print('Reading solution at X = {0:5.2f} g/cm2'.format(x_grid[idx]))
        longitudinal_spectrum.append(mceq.get_solution('mu+', grid_idx=idx))

To obtain the solutions at equivalent altitudes one needs to simply map the
the values of :math:`X` to the corresponding altitude for the **current** zenith
angle and atmospheric model::

    h_grid = mceq.density_model.X2h(X_grid)

To define a strictly increasing grid in X (=stricktly decreasing in altitude), using the converter function between height and depth::

    h_grid = np.linspace(50 * 1e3 * 1e2, 0) # altitudes from 50 to 0 km (in cm)
    X_grid = mceq.density_model.h2X(h_grid)

    mceq.solve(int_grid=X_grid)

Particle numbers can be obtained by using predefined functions or by integrating
the spectrum. These functions support `grid_idx` (as shown above) and a minimal
energy cutoff (larger than the minimal grid energy :attr:`mceq_config.e_min`)::

    # Number of muons
    n_mu = mceq.n_mu(grid_idx=None, min_energy_cutoff=1e-1)

    # Number of electrons
    n_e = mceq.n_e(grid_idx=None, min_energy_cutoff=86e-3)

    # Number of protons above minimal grid energy
    n_p = np.sum(mceq.get_solution('p+', integrate=True))

All particles listed by :func:`MCEq.ParticleManager.print_particle_tables(0)` are
available to :func:`MCEq.core.get_solution`.

Changing geometrical and atmospheric parameters
...............................................

To change the zenith angle ::

    mceq.set_theta_deg(<zenith_angle_in_degrees>)

Most geometries support angles between 0 (vertical) and 90 degrees.

To change the density profile ::

    mceq.set_density_model(('MSIS00', ('Sudbury', 'June')))

Available models are:

- 'CORSIKA' - Linsley-parameterizations from the CORSIKA air-shower MC (see :func:`MCEq.geometry.density_models.CorsikaAtmosphere.init_parameters`)
- 'MSIS00' and 'MSIS00_IC' - NRLMSISE-00 global static atmospheric model by NASA (_IC = centered on IceCube at the South Pole, where zenith angles > 90 degrees are up-going)
- 'AIRS' - an interface to tabulated satellite data (not provided), extrapolated with MSIS00 at altitudes above 50km
- 'Isothermal' - a simple isothermal model with scale height at 6.3 km
- 'GeneralizedTarget' - a piece-wise homogeneous density (not exponential like the atmosphere)

Refer for more info to :ref:`densities`.

After changing the models, the spectra can be recomputed with a :func:`MCEq.core.MCEqRun.solve()`.

Changing hadronic interaction models
....................................

To change the hadronic interaction model ::

    mceq.set_interaction_model('EPOS-LHC')

Currently available models are:

- SIBYLL-2.3c
- SIBYLL-2.3
- SIBYLL-2.1
- EPOS-LHC
- QGSJet-II-04
- QGSJet-II-03
- QGSJet-01c
- DPMJET-III-3.0.6
- DPMJET-III-19.1
- SIBYLL-2.3c_pp (for proton-proton collisions)

More models planned. Note that internally the model name string is
transformed to upper case, and dashes and points are removed.

MCEq will take care of updating all data structures regenerating the matrices. This call
takes some time since data memory needs to be allocated and some numbers crunched. If you
use this function in a loop for multiple computations, put it further out.

Changing cosmic ray flux model
..............................

The flux of cosmic ray nucleons at the top of the atmosphere (primary flux) is the initial condition. The
module :mod:`crflux.models` contains a contemporary selection of flux models. Refer to the
`crflux documentation <https://crfluxmodels.readthedocs.io/en/latest/>`_ or 
`the source code <https://github.com/afedynitch/crflux>`_.

To change the primary flux use :func:`MCEq.core.MCEqRun.set_primary_model` ::

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
:func:`MCEq.core.MCEqRun.set_single_primary_particle` ::

    # For a 1 EeV proton
    mceq.set_single_primary_particle(1e9, pdg_id=2212)

    # Or for a 1 EeV iron nucleus
    mceq.set_single_primary_particle(1e9, corsika_id=5626)

The zenith angle has to be set as shown above with :func:`MCEq.core.MCEqRun.set_zenith_deg`.
