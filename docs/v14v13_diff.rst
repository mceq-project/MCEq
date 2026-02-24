.. _v14v13_diff:

MCEq v1.4
#########

Welcome to MCEq v1.4, a long foreseen release!
| With this versions we updated major parts of the MCEq database.
| We introduce **new Hadronic Interaction Models**, **updated decay yields**, and **updated cross sections**.
| The **Data Driven Model (DDM)** is additionally available with this release.


What has changed?
=================

For a more detailed list please refer to the latest Changelog_.

The significant changes are updates to the MCEq database!

1. Hadronic Interaction Models
   | The new database is composed of a set of **baseline models** (Sibyll-2.1, QGSjetII04, Epos-LHC),
   | to provide a comparison between MCEq v1.3 and v1.4, as well as a set of **new models** (Sibyll-2.3d, Sibyll-2.3e, QGSjetIII, DPMJetIII-19.3, Epos-LHC-R).
2. Decay Yields
   | The calculation of decay yields moved completly to the *Pythia* interface of chromo_.
   | Decays are now computed in the rest-frame of the particle of interest.
3. Cross Sections
   | All cross sections are updated to **production** cross sections of the particle of interest.



.. _Changelog: https://github.com/mceq-project/MCEq/blob/main/CHANGELOG.md
.. _chromo: https://github.com/impy-project/chromo


Baseline Comparison
-------------------

In the following we compare the aforementioned baseline models (Sibyll-2.1, QGSjetII04, Epos-LHC) in v1.4 against v1.3.

Lepton Fluxes
^^^^^^^^^^^^^

.. figure:: _static/graphics/comparison14/muon_flux_ratio.png
    :width: 1024 px
    :align: left

.. figure:: _static/graphics/comparison14/muon_neutrino_flux_ratio.png
    :width: 1024 px
    :align: left

.. figure:: _static/graphics/comparison14/electron_neutrino_flux_ratio.png
    :width: 1024 px
    :align: left


Contact
=======

We track various changes in much more detail. 
If you encounter any problems, please contact Stefan_.



.. _Stefan: mailto:stefanfroese@as.edu.tw
