.. _quickstart:
:html_theme.sidebar_primary.remove: true

**********
Quickstart
**********

Installation
............

The installation via PyPi is the simplest method

.. code-block::

   pip install MCEq

Optionally, one can (and is encouraged to) accelerate calculations with BLAS
libraries. For Intel CPUs the Math Kernel Library (MKL) provides quite some
speed-up compared to plain numpy. MCEq will auto-detect MKL if it is already
installed. To enable MKL by default use

.. code-block::

    pip install MCEq[MKL]

More speed-up can be achieved by using the cuSPARSE library from nVidia's
CUDA toolkit. This requires the cupy library. If cupy is detected, MCEq
will try to use cuSPARSE as solver. To install MCEq with CUDA 10.1 support

.. code-block::

    pip install MCEq[CUDA]

Alternatively, install cupy by yourself (see `cupy homepage <https://cupy.chainer.org>`__).

Supported architectures:

- Linux 32- and 64-bit (x86_64 and AArch64)
- Mac OS X
- Windows

.. note:: 
   pip installations of scipy on Windows may be faulty. If scipy throws errors
   on import, use `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_

Upgrading
.........

For installations with pip, upgrading the code and data tables can be done with

.. code-block::

    pip install MCEq --upgrade

In case of major updates the database file will be updated on first import and the old
one will be removed. For installations from source, pull the latest release or master branch. The database file will be updated automatically as well.

Building MCEq from source
.........................

To modify the code and contribute, the code needs to be installed from the github source

.. code-block::

    git clone https://github.com/afedynitch/MCEq.git
    cd MCEq
    pip install -e .

This will build and install MCEq in editable mode, and changes to this source directory will
be immediately reflected in any code that imports MCEq in the current python enviroment.

Running MCEq
...........

Open an new python file or jupyter notebook/lab

.. code-block::

    from MCEq.core import config, MCEqRun
    import crflux.models as crf
    # matplotlib used plotting. Not required to run the code.
    import matplotlib.pyplot as plt


    # Initalize MCEq by creating the user interface object MCEqRun
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

    # Obtain the result
    # Multiply fluxes be E**mag to resolve the features of the steep spectrum
    mag = 3
    muon_flux = (mceq.get_solution('mu+', mag) + 
                 mceq.get_solution('mu-', mag))
    numu_flux = (mceq.get_solution('numu', mag) + 
                 mceq.get_solution('antinumu', mag))
    nue_flux = (mceq.get_solution('nue', mag) +
                mceq.get_solution('antinue', mag))

    # The lines below are for plotting with matplotlib 
    plt.loglog(mceq.e_grid, muon_flux, label='muons')
    plt.loglog(mceq.e_grid, numu_flux, label='muon neutrinos')
    plt.loglog(mceq.e_grid, nue_flux, label='electron neutrinos')

    plt.xlim(1., 1e9)
    plt.xlabel('Kinetic energy (GeV)')
    plt.ylim(1e-6, 1.)
    plt.ylabel(r'$(E/\text{GeV})^3\,\Phi$ (GeV cm$^{-2}$\,$s$^{-1}\,$sr$^{-1}$) (GeV)')
    plt.legend()
    plt.show()


.. warning::

  Due to a bug found in the decay table generation for the "new" MCEq versions
  there are some larger changes for the lowest energies at tens of GeV. To estimate
  if your computations may be affected check out :ref:`v12v11_diff`.

