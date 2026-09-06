"""Species-layer constants: the shared table singleton and unit constants.

``_pdata`` stays an import-time instance (the plan's lazification wish would
hide it from ``vars()``, which the pinned surface contract forbids — see the
test note in ``tests/test_phase4_surface.py``). ``_pname`` and the neutrino
name map moved here verbatim from the flat ``particlemanager`` module.
"""

from particletools.tables import PYTHIAParticleData

from MCEq.misc import info

info(5, "Initialization of PYTHIAParticleData object")
_pdata = PYTHIAParticleData()

# Exact atomic mass constant in grams. Cross sections are converted to inverse
# interaction lengths in g^-1 cm^2 through sigma / (A_target * m_u).
ATOMIC_MASS_UNIT_G = 1.0 / 6.02214076e23


backward_compatible_namestr = {
    "nu_mu": "numu",
    "nu_mubar": "antinumu",
    "nu_e": "nue",
    "nu_ebar": "antinue",
    "nu_tau": "nutau",
    "nu_taubar": "antinutau",
}


# Replace particle names for neutrinos with those used
# in previous MCEq versions
def _pname(pdg_id_or_name):
    """Replace some particle names from pythia database with those from previous
    MCEq versions for backward compatibility."""

    pythia_name = _pdata.name(pdg_id_or_name)
    if pythia_name in backward_compatible_namestr:
        return backward_compatible_namestr[pythia_name]
    return pythia_name
