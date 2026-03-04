![PyPI](https://img.shields.io/pypi/v/MCEq)
[![Documentation](https://readthedocs.org/projects/mceq/badge/?version=latest)](https://mceq.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://dev.azure.com/afedynitch/MCEq/_apis/build/status/afedynitch.MCEq?branchName=master)](https://dev.azure.com/afedynitch/MCEq/_build/latest?definitionId=1&branchName=master)

# MCEq - Matrix cascade equations

MCEq is a numerical tool for solving cascade equations that model the evolution of particle densities as they traverse gaseous or dense media. Its primary application is simulating particle cascades in the Earth's atmosphere, where particles are tracked as average densities across discrete energy bins. MCEq outputs differential energy spectra and total particle counts, supporting a range of models and parameterizations for particle interactions and atmospheric density profiles.

Very early releases, previously maintained under the 'master' and 'development' branches, are archived and available in the [MCEq_classic repository](https://github.com/afedynitch/MCEq_classic).

## [Documentation](http://mceq.readthedocs.org/en/latest/)

[The documentation](http://mceq.readthedocs.org/en/latest/) contains installation instructions, a tutorial and more.

### Version 1.4
With this version we update the database to contain new hadronic interaction models alongside some bug fixes and new 
features, such as DDM.

Please consult the [CHANGELOG](CHANGELOG.md) and the dedicated [doc page](https://mceq.readthedocs.io/en/latest/v14v13_diff.html).

## Please cite our work

If you are using this code in your scientific work, please cite the code **AND** the
physical models!
We provide citation resources in our [Docs](https://mceq.readthedocs.io/en/latest/citeus.html)!

### Authors:

*Anatoli Fedynitch*
*Stefan Fröse*

### Contributers

See [here](https://github.com/mceq-project/MCEq/graphs/contributors) and [here](https://github.com/afedynitch/MCEq_classic/graphs/contributors) for MCEq_classic.

## Copyright and license

Code released under [the BSD 3-clause license (see LICENSE)](LICENSE).
