MCEq - Matrix cascade equation (Release Candidate 1)
====================================================

This version was previously known as 'dev' branch.

This scientific package might be useful fo all who deal with high-energy inclusive atmospheric fluxes of muons and neutrinos. 
In particular it might be useful for astroparticle physics experiments, for example  `IceCube <https://icecube.wisc.edu>`_ or 
`MINOS <http://www-numi.fnal.gov/PublicInfo/index.html>`_, for calculations of systematic uncertainties and atmospheric backgrounds.

Status (Updated)
------

This is release candiate of the final version 1.0. It has several new features including:
- extended energy range (1 GeV - 10^11 GeV)
- new interaction models, SIBYLL 2.3 + 2.3c, EPOS-LHC and DPMJET-III 17.1
- compact (=very fast) mode
- low-energy extension (with DPMJET-III) of high-energy interaction models
- computation of hadron and lepton yields along an air-shower trajectory (average air-shower)
- energy loss for muons
- a generalized target mode, with arbitrary density profiles of target material (experimental and physics is not yet accurate)

`Documentation (updated) <http://mceq.readthedocs.org/en/latest/>`_
--------------------------------------------------------------------

The latest version of the documentation can be found `here <http://mceq.readthedocs.org/en/latest/>`_.

Please cite our work
--------------------

If you are using this code in your scientific work, please cite the code **AND** the
physical models. A complete list of references can be found in the 
`Citations section of the docs <http://mceq.readthedocs.org/en/latest/citations.html>`_.

System requirements
-------------------

- Some kind of modern CPU with FPU unit
- 2GB (8GB of RAM is recommended)
- ~2GB of disk space
- OS: Linux, Mac or Windows 10

Software requirements
---------------------

The majority of the code is pure Python. Some functions are accelerated through Just-In-Time (JIT) compilation 
using `numba <http://numba.pydata.org>`_, which requires the `llvmlite` package.

Dependencies:

* python-2.7
* numpy
* scipy
* numba
* matplotlib
* jupyter notebook (optional, but needed for examples)
* progressbar

Additional dependencies are required for the C implementation of the NRLMSISE-00 atmosphere:

* a C compiler (GNU gcc, for example)
* make
* ctypes

Installation
------------
The installation simplest method relies on the Python package manager `Anaconda/Miniconda <https://store.continuum.io/cshop/anaconda/>`_ 
by `Continuum Analytics <http://www.continuum.io>`_. It doesn't just improve your life, but also provides most of the scientific computing 
packages by default. It also distributes a numpy version integrated with `Intel's Math Kernel Library <https://software.intel.com/en-us/intel-mkl>`_ (MKL).
It will not spoil your system Python paths and will install itself into a specified directory. The only action which is needed for activation, 
is to add this directory to your system `$PATH` variable. To uninstall just delete this directory.

#. Download one of the installers for your system architecure from here:

	* `Anaconda <http://continuum.io/downloads>`_ - larger download, already containing most of the scientific packages and the package manager `conda` itself
	* `Miniconda <http://conda.pydata.org/miniconda.html>`_ - minimal download, which contains the minimum requirements for the package manager `conda`.

#. Run the installer and follow the instructions:

	.. code-block:: bash

	   $ bash your-chosen-conda-distribution.sh

	Open a new terminal window to reload your new `$PATH` variable.


#. `Cd` to you desired working directory. And clone (*note the `--recursive`*) this project including submodules:

	.. code-block:: bash

	   $ git clone --recursive https://github.com/afedynitch/MCEq.git

	It will clone this github repository into a folder called `MCEq` and download all files.
	Enter this directory.

#. To install all dependencies into you new conda environment

	.. code-block:: bash

	   $ conda install --file conda_req.txt

	This will ask conda to download and install all needed packages into its default environment.

#. (**Optional**) If you know what a `virtualenv` is, the corresponding commands to download and install all packages in a newly created environment `mceq_env` are

	.. code-block:: bash

	   $ conda create -n mceq_env --file conda_req.txt
	   $ source activate mceq_env

	To quit this environment just

	.. code-block:: bash

	   $ deactivate

#. Run some example

	.. code-block:: bash

	   $ jupyter notebook

	click on the examples directory and select `basic_flux.ipynb`. Click through the blocks and see what happens.

Troubleshooting
---------------
You might run into `problems with Anaconda <https://github.com/conda/conda/issues/394>`_  if you have previous Python installations. A workaround is to set the environement variable

	.. code-block:: bash

	   $ export PYTHONNOUSERSITE=1

Thanks to F.C. Penha for pointing this out.

Contributers
------------

*Anatoli Fedynitch*

Copyright and license
---------------------
Code and documentation copyright 2014-2017 Anatoli Fedynitch. Code released under `the MIT license <https://github.com/afedynitch/MCEq/blob/master/LICENSE>`_.
