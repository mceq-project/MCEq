MCEq - Matrix cascade equation
==============================

This scientific package might be useful fo all who deal with high-energy inclusive atmospheric fluxes of muons and neutrinos. In particular it might be useful for experiments, for example  `IceCube <https://icecube.wisc.edu>`_ or `MINOS <http://www-numi.fnal.gov/PublicInfo/index.html>`_, for calculations of systematic uncertainties and atmospheric backgrounds.

Status
------

The current development status is **alpha**. Although the numerical part of the program is rather stable, the parts related to user interaction, installation etc. are not finished, yet. Check `the wiki <https://github.com/afedynitch/MCEq/wiki>`_ for further items on the *ToDo-list*. Also feel free to open issues.

`Documentation <http://mceq.readthedocs.org/en/latest/>`_
---------------------------------------------------------

As mentioned above the project is development. The current state of the documentation is more suited for developers rather than end-users. Mostly it is auogenerated `sphinx`-docs. Check it out before touching the code.  The latest version of the documentation can be found `here <http://mceq.readthedocs.org/en/latest/>`_.

System requirements
-------------------

- Some kind of modern CPU (Core2Duo++)
- 4GB (currently 8GB of RAM is stongly recommended. The solver is not optimzed for memory usage, however there's lots of room for improvement)
- ~1GB of disk space
- a recent Linux or Mac OS X operating system. Windows might be suitable, but was not checked.

Software requirements
---------------------

The majority of the code consists of pure Python modules. Some functions are accelerated through Just-In-Time (JIT) compilation using `numba <http://numba.pydata.org>`_, which requires the `llvmlite` package.

Dependencies:

* python-2.7 (Python 3 not compatible yet)
* numpy
* scipy
* matplotlib
* ipython + notebook (optional, but needed for examples)
* numba
* progressbar


Installation
------------
The installation simplest method relies on the Python package manager `Anaconda/Miniconda <https://store.continuum.io/cshop/anaconda/>`_ by `Continuum Analytics <http://www.continuum.io>`_. It doesn't just improve your life, but also provides most of the scientific computing packages by default. It will not spoil your system Python paths and will install itself into a specified directory. The only action which is needed for activation, is to add this directory to your system `$PATH` variable. To uninstall just delete this directory.

#. Download one of the installers for your system architecure from here:

	* `Anaconda <http://continuum.io/downloads>`_ - larger download, already containing most of the scientific packages and the package manager `conda` itself
	* `Miniconda <http://conda.pydata.org/miniconda.html>`_ - minimal download, which contains the minimum requirements for the package manager `conda`.

#. Run the installer and follow the instructions:

	.. code-block:: bash

	   $ bash your-chosen-conda-distribution.sh

	Open a new terminal window to reload your new `$PATH` variable.


#. `Cd` to you desired working directory. And clone this project including submodules:

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

#. (**Optional**) Acceleration of the integration routines can be achieved using `Intel Math Kernel Library <https://software.intel.com/en-us/intel-mkl>`_ (MKL). Anaconda offers MKL-linked numpy binaries free for academic use. It is necessary to register using your *.edu* mail adress to receive a license. The demo period is 30 days. If you want to give it a try

	.. code-block:: bash

		   $ conda install mkl

	Change in `mceq_config.py` the `kernel` entry to 'MKL'.

#. Run some example

	.. code-block:: bash

	   $ ipython notebook

	click on the examples directory and select `basic_flux.ipynb`. Click through the blocks and see what happens.

Troubleshoting
--------------
You might run into `problems with Anaconda <https://github.com/conda/conda/issues/394>`_  if you have previous 
Python installations. A workaround is to set the environement variable
	.. code-block:: bash

	   $ export PYTHONNOUSERSITE=1
	   
Thanks to F.C. Penha for pointing this out.

Citation
--------
If you are using this code in your scientific work, please cite 

   | *Calculation of conventional and prompt lepton fluxes at very high energy*
   | A. Fedynitch, R. Engel, T. K. Gaisser, F. Riehn, T. Stanev,
   | `arXiv:1503.00544 <http://arxiv.org/abs/1503.00544>`_

Please, also cite or footnote this Github site and revisit this page from time to time, 
to get the most up2date information.

The models inside this code need to be cited separately. Please
`see the documentation <http://mceq.readthedocs.org/en/latest/citations.html>`_ for the complete list of references.


Contributers
------------

*Anatoli Fedynitch*

Copyright and license
---------------------
Code and documentation copyright 2014-2015 Anatoli Fedynitch. Code released under `the MIT license <https://github.com/afedynitch/MCEq/blob/master/LICENSE>`_.
