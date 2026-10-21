:html_theme.sidebar_primary.remove: true

.. _quickstart:

**********
Quickstart
**********

Installation
............

The installation via PyPi is the simplest method

.. code-block::

   pip install MCEq

Calculations can be accelerated with BLAS libraries. On macOS MCEq uses
Apple Accelerate. On Linux and Windows MCEq uses Intel MKL if its runtime
library ``libmkl_rt`` is installed in the Python environment, for example by

.. code-block::

    pip install mkl

More speed-up can be achieved on an NVIDIA GPU through the cupy library,
selected with ``config.kernel_config = "cuda_etd2"`` (never chosen by ``"auto"``).
To install MCEq with cupy for CUDA 12 (any driver from 525 on) or for a
CUDA 13 driver and toolkit

.. code-block::

    pip install MCEq[cuda]
    pip install MCEq[cuda13]

Alternatively, install cupy by yourself (see `cupy homepage <https://cupy.dev>`__).

Supported platforms (Python 3.10 or later):

- Linux 64-bit (x86_64 and AArch64)
- macOS (Apple silicon and Intel)
- Windows 64-bit (x86_64 and ARM64)

Upgrading
.........

For installations with pip, upgrading the code and data tables can be done with

.. code-block::

    pip install MCEq --upgrade

For installations from source, pull the latest release or the main branch.
Database files that are missing are downloaded into ``config.data_dir`` when
the first :class:`MCEq.core.MCEqRun` is created.

.. note::
   The data are distributed as several files described by a YAML manifest
   (a 1D base file with FLUKA20251 and SIBYLL23E, an extra-models file and 2D
   files), so an install carries only the models it runs; a model missing
   from the selected database is fetched from the data release on first use.
   See :doc:`../data_installation` for the file list, the cluster pre-fetch
   CLI and how to register a database of your own.

Building MCEq from source
.........................

To modify the code and contribute, the code needs to be installed from the github source

.. code-block::

    git clone https://github.com/mceq-project/MCEq.git
    cd MCEq
    pip install -e .

This will build and install MCEq in editable mode, and changes to this source directory will
be immediately reflected in any code that imports MCEq in the current python enviroment.

Running MCEq
............

Open an new python file or jupyter notebook/lab

.. code-block::

    from MCEq.core import MCEqRun
    import crflux.models as crf
    # matplotlib used plotting. Not required to run the code.
    import matplotlib.pyplot as plt


    # Initalize MCEq by creating the user interface object MCEqRun
    mceq = MCEqRun(

        # High-energy hadronic interaction model
        interaction_model='SIBYLL23E',

        # Low-energy hadronic interaction model below 80 GeV (the default;
        # None switches the blending off)
        low_energy_model='FLUKA20251',

        # cosmic ray flux at the top of the atmosphere
        primary_model=(crf.HillasGaisser2012, 'H3a'),

        # zenith angle
        theta_deg=0.
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
    plt.ylabel(r'$E^3\,\Phi$ (GeV$^2$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$)')
    plt.legend()
    plt.show()
