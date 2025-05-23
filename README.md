![PyPI](https://img.shields.io/pypi/v/MCEq)
[![Documentation](https://readthedocs.org/projects/mceq/badge/?version=latest)](https://mceq.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://dev.azure.com/afedynitch/MCEq/_apis/build/status/afedynitch.MCEq?branchName=master)](https://dev.azure.com/afedynitch/MCEq/_build/latest?definitionId=1&branchName=master)
![Azure DevOps releases](https://img.shields.io/azure-devops/release/afedynitch/e02bcbf5-db8e-4417-ad07-cc2547ea47e0/6/6)

# MCEq - Matrix cascade equations

MCEq is a numerical tool for solving cascade equations that model the evolution of particle densities as they traverse gaseous or dense media. Its primary application is simulating particle cascades in the Earth's atmosphere, where particles are tracked as average densities across discrete energy bins. MCEq outputs differential energy spectra and total particle counts, supporting a range of models and parameterizations for particle interactions and atmospheric density profiles.

Very early releases, previously maintained under the 'master' and 'development' branches, are archived and available in the [MCEq_classic repository](https://github.com/afedynitch/MCEq_classic).

## [Documentation](http://mceq.readthedocs.org/en/latest/)

[The documentation](http://mceq.readthedocs.org/en/latest/) contains installation instructions, a tutorial and more.

### Version 1.3
This version is physically identical to the 1.2.X release, but features a modernized build system and now provides wheels up to Python 3.13. While this is primarily a technical update, core development has resumed and new features will be merged more frequently.

Please consult the [CHANGELOG](CHANGELOG) and the dedicated [doc page](http://mceq.readthedocs.org/en/latest/v12v11_diff.html).

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
