.. tutorial:

Tutorial
--------

The main user interface is the :class:`MCEqRun`. For the initialization
a reference to a primary model class is required. Any cosmic ray flux model
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
model is selected and will list the currently available models. Models
can be changed between calls to the solver.

Solving cascade equations
.........................

The solver is launched for the current set of parameters by::

    mceq.solve()

By default MCEq will pick the numpy, MKL or the CUDA solver, depending on the
the installed packages. Currently only 'forward-euler' solvers are available,
which are fast and stable enough.

To obtain a solution along various points in :math:`X`, create a
grid and then supply it to the solver::

    # A linearly spaced set of points from 0.1 up to an X value
    # corresponding to the depth at the surface for the selected zenith
    # angle and atmospheric model/season
    n_pts = 100
    X_grid = np.linspace(0.1, mceq.density_profile.maxX, n_pts)
    
    mceq.solve(int_grid=X_grid)

To obtain particle spectra at each depth point::

    for idx in range(n_pts):
        print('Obtaining solution at X = {0:5.2f} g/cm2'.format(x_grid[idx]))
        some_list.append(mceq.get_solution('mu+',grid_idx=idx))

To obtain the solutions at equivalent altitudes one needs to simply map the
the values of :math:`X` to the corresponding altitude for the **current** zenith
angle and atmospheric model::

    h_grid = mceq.density_profile.X2h(X_grid)

To define a grid in X that corresponds to certain altitudes, use the inverse
function::

    h_grid = np.linspace(0,50*1e3*1e2) # altitudes from 0 to 50 km (in cm)
    X_grid = mceq.density_profile.h2X(h_grid)

    mceq.solve(int_grid=X_grid)

Changing geometrical and atmospheric parameters
...............................................

To change the zenith angle ::

    mceq.set_zenith_deg(<zenith_angle_in_degrees>)

Most geometries support angles between 0 (vertical) and 90 degrees.

To change the density profile ::

    mceq.set_density_model(('MSIS00', 'Sudbury', 'June'))

Available models are:

- 'CORSIKA' - Linsley-parameterizations from the CORSIKA air-shower MC (see :func:`MCEq.geometry.density_profiles.CorsikaAtmosphere.init_parameters`)
- 'MSIS00' and 'MSIS00_IC' - NRLMSISE-00 global static atmospheric model by NASA (_IC = centered on IceCube at the South Pole, where zenith angles > 90 degrees are up-going)
- 'AIRS' - an interface to tabulated satellite data (not provided), extrapolated with MSIS00 at altitudes above 50km
- 'Isothermal' - a simple isothermal model with scale height at 6.3 km
- 'GeneralizedTarget' - a piece-wise homogeneous density (not exponential like the atmosphere)

Refer for more info to :ref:`densities`.

After changing the models, the spectra can be recomputed with a simple `mceq.solve()`.

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

