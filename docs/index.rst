.. Matrix Cascade Equation (MCEq) documentation master file, created by
   sphinx-quickstart on Fri Nov 21 10:13:38 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Matrix Cascade Equations (MCEq) documentation
===============================================

Purpose of the code:

    This program is a toolkit for calculations of particle cascades in the
    Earth's atmosphere or other target media. Particles are represented by
    average densities within discrete energy ranges (bins). The results of
    calculations are, therefore, differential spectra.

Installation:
.............

The installation via PyPi is the simplest method::

    pip install MCEq

Optionally, one can (and is encouraged to) accelerate calculations with BLAS
libraries. For Intel CPUs the Math Kernel Library (MKL) provides quite some
speed-up compared to plain numpy. MCEq will auto-detect MKL if it is already
installed. To enable MKL by default use::

    pip install MCEq[MKL]

More speed-up can be achieved by using the cuSPARSE library from nVidia's
CUDA toolkit. This requires the cupy library. If cupy is detected, MCEq
will try to use cuSPARSE as solver. To install MCEq with CUDA 10.1 support::

    pip install MCEq[CUDA]

Alternatively, install cupy by yourself (see [cupy homepage](https://cupy.chainer.org)).


How to use the code:
....................

Open an new python file or jupyter notebook::

    from MCEq.core import config, MCEqRun
    import crflux.models as crf
    # matplotlib used plotting. Not required to run the code.
    import matplotlib.pyplot as plt


    # Initalize MCEq by creating the user interface :class:`MCEqRun`
    mceq = MCEqRun(
        # High-energy hadronic interaction model
        interaction_model='SIBYLL23C',
        # cosmic ray flux at the top of the atmosphere
        primary_model = (crf.HillasGaisser2012, 'H3a'), 
        # zenith angle
        theta_deg = 0. 
        )
    
    # Solve the equation system
    mceq.solve()

    # Multiply fluxes be E**mag to better see the features
    mag = 3
    muon_flux = (mceq.get_solution('mu+', mag) + 
                 mceq.get_solution('mu-', mag))
    numu_flux = (mceq.get_solution('numu', mag) + 
                 mceq.get_solution('antinumu', mag))
    nue_flux = (mceq.get_solution('nue', mag) +
                mceq.get_solution('antinue', mag))

    plt.loglog(mceq.e_grid, muon_flux, label='muons')
    plt.loglog(mceq.e_grid, numu_flux, label='muon neutrinos')
    plt.loglog(mceq.e_grid, nue_flux, label='electron neutrinos')

    plt.xlim(1., 1e9)
    plt.xlabel('Kinetic energy (GeV))
    plt.ylim(1e-6, 1.)
    plt.ylabel('r"$(E/\text{GeV})^3\,\Phi$ (GeV cm$^{-2}$\,$s$^{-1}\,$sr$^{-1}$)" (GeV))
    plt.legend()
    plt.show()







Citations:
..........

If you use MCEq in your scientific publication, please cite
the code **AND** the physical model publications properly.

The current citation for the code is:

   | *Calculation of conventional and prompt lepton fluxes at very high energy*
   | A. Fedynitch, R. Engel, T. K. Gaisser, F. Riehn, T. Stanev,
   | EPJ Web Conf. 99 (2015) 08001
   | `arXiv:1503.00544 <http://arxiv.org/abs/1503.00544>`_

See :ref:`citations` for the papers about the physical models.


Contents:

.. toctree::
   :maxdepth: 3

   intro
   citations
   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

