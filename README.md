![PyPI](https://img.shields.io/pypi/v/MCEq)
[![Build Status](https://dev.azure.com/afedynitch/MCEq/_apis/build/status/afedynitch.MCEq?branchName=master)](https://dev.azure.com/afedynitch/MCEq/_build/latest?definitionId=1&branchName=master)
![Azure DevOps releases](https://img.shields.io/azure-devops/release/afedynitch/e02bcbf5-db8e-4417-ad07-cc2547ea47e0/6/6)

# MCEq - Matrix cascade equations

MCEq is a tool to numerically solve cascade equations that describe the evolution
of particle densities as they propagate through a gaseous or dense medium.
The main application are particle cascades in the Earth's atmosphere.
Particles are represented by average densities on discrete energy bins.
The results are differential energy spectra or total particle numbers.
Various models/parameterizations for particle interactions and atmospheric
density profiles are packaged with the code.  

This is a new version of the code and may break compatibility with the previous versions. 
The old versions known as 'master' and 'development' branch are deprecated and located in the 
[MCEq_classic repository](https://github.com/afedynitch/MCEq_classic).

## [Documentation](http://mceq.readthedocs.org/en/latest/)

[The documentation](http://mceq.readthedocs.org/en/latest/) contains installation instructions, a tutorial and more.

## Please cite our work

If you are using this code in your scientific work, please cite the code **AND** the
physical models. A complete list of references can be found in the 
[Citations section of the docs](http://mceq.readthedocs.org/en/latest/citations.html).

### Authors:

*Anatoli Fedynitch*

### Contributers

*[Hans Dembinski](https://github.com/HDembinski)*

## Copyright and license

Code released under [the BSD 3-clause license (see LICENSE)](LICENSE).
