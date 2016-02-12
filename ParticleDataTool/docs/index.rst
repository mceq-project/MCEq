.. ParticleDataTool documentation master file, created by
   sphinx-quickstart on Fri Nov 28 12:04:38 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation for the :mod:`ParticleDataTool`
=============================================

.. toctree::
   :maxdepth: 2

````````````
Introduction
````````````
The module :mod:`ParticleDataTool` was developed as a convinience
library for translating particle codes from one high energy hadronic
interaction model into the more general naming convention of the
Particle Data Group `PDG <http://pdg.lbl.gov>`_. Soon it became clear,
that obtaining particle properties, such as mass, charge or life-time
is also a frequent task. The module was then extended with an interface 
to an XML database of particle properties, which is shipped as part of the
`PYTHIA 8 <http://home.thep.lu.se/~torbjorn/pythia81html/Welcome.html>`_ 
Monte Carlo. It is licensed under the GPL V2.
The largest part of the information contained in this XML is ignored 
and could be subject for future extensions if needed. According to the 
PYTHIA 8 documentation, this database is based on 2006 data obtained 
from the PDG.

```````
Caching
```````
When an instance of the :class:`ParticleDataTool.PYTHIAParticleData` 
is created, the result from parsing the XML and creating the tables 
is saved to a file in the working directory. Subsequent instatiations 
of this class will always load the tables from this file. To force 
parsing the XML file again just delete the file "ParticleData.ppl".

````````````````````
Module documentation
````````````````````

.. automodule:: ParticleDataTool
   :members:



Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

