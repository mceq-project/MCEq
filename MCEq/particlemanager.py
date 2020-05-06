
import six
from math import copysign
import numpy as np
import mceq_config as config
from MCEq.misc import info, print_in_rows, getAZN

from particletools.tables import PYTHIAParticleData
info(5, 'Initialization of PYTHIAParticleData object')
_pdata = PYTHIAParticleData()

backward_compatible_namestr = {
    'nu_mu': 'numu',
    'nu_mubar': 'antinumu',
    'nu_e': 'nue',
    'nu_ebar': 'antinue',
    'nu_tau': 'nutau',
    'nu_taubar': 'antinutau'
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


class MCEqParticle(object):
    """Bundles different particle properties for simplified
    availability of particle properties in :class:`MCEq.core.MCEqRun`.

    Args:
      pdg_id (int): PDG ID of the particle
      egrid (np.array, optional): energy grid (centers)
      cs_db (object, optional): reference to an instance of
                                :class:`InteractionYields`
    """
    def __init__(self,
                 pdg_id,
                 helicity,
                 energy_grid=None,
                 cs_db=None,
                 init_pdata_defaults=True):

        #: (bool) if it's an electromagnetic particle
        self.is_em = abs(pdg_id) == 11 or pdg_id == 22
        #: (int) helicity -1, 0, 1 (0 means undefined or average)
        self.helicity = helicity
        #: (bool) particle is a nucleus (not yet implemented)
        self.is_nucleus = False
        #: (bool) particle is a hadron
        self.is_hadron = False
        #: (bool) particle is a lepton
        self.is_lepton = False
        #: (float) ctau in cm
        self.ctau = None
        #: (float) mass in GeV
        self.mass = None
        #: (str) species name in string representation
        self.name = None
        #: Mass, charge, neutron number
        self.A, self.Z, self.N = getAZN(pdg_id)
        #: (bool) particle has both, hadron and resonance properties
        self.is_mixed = False
        #: (bool) if particle has just resonance behavior
        self.is_resonance = False
        #: (bool) particle is interacting projectile
        self.is_projectile = False
        #: (bool) particle is stable
        self.is_stable = False or pdg_id in config.adv_set['disable_decays']
        #: (bool) can_interact
        self.can_interact = False
        #: (bool) has continuous losses dE/dX defined
        self.has_contloss = False
        #: (np.array) continuous losses in GeV/(g/cm2)
        self.dEdX = None
        #: (bool) is a tracking particle
        self.is_tracking = False
        #: decay channels if any
        self.decay_dists = {}
        #: (int) Particle Data Group Monte Carlo particle ID
        self.pdg_id = (pdg_id, helicity)
        #: (int) Unique PDG ID that is different for tracking particles
        self.unique_pdg_id = (pdg_id, helicity)
        #: (int) MCEq ID
        self.mceqidx = -1

        #: (float) mixing energy, transition between hadron and
        # resonance behavior
        self.E_mix = 0
        #: (int) energy grid index, where transition between
        # hadron and resonance occurs
        self.mix_idx = 0
        #: (float) critical energy in air at the surface
        self.E_crit = 0
        # Energy and cross section dependent inits
        self.current_cross_sections = None
        self._energy_grid = energy_grid

        # Variables for hadronic interaction
        self.current_hadronic_model = None
        self.hadr_secondaries = []
        self.hadr_yields = {}

        # Variables for decays
        self.children = []
        self.decay_dists = {}

        # A_target
        self.A_target = config.A_target

        if init_pdata_defaults:
            self._init_defaults_from_pythia_database()

        if self._energy_grid is not None and cs_db is not None:
            #: interaction cross section in 1/cm2
            self.set_cs(cs_db)

    def _init_defaults_from_pythia_database(self):
        """Init some particle properties from :mod:`particletools.tables`."""
        #: (bool) particle is a nucleus (not yet implemented)
        self.is_nucleus = _pdata.is_nucleus(self.pdg_id[0])
        #: (bool) particle is a hadron
        self.is_hadron = _pdata.is_hadron(self.pdg_id[0])
        #: (bool) particle is a hadron
        self.is_lepton = _pdata.is_lepton(self.pdg_id[0])
        #: Mass, charge, neutron number
        self.A, self.Z, self.N = getAZN(self.pdg_id[0])
        #: (float) ctau in cm
        self.ctau = _pdata.ctau(self.pdg_id[0])
        #: (float) mass in GeV
        self.mass = _pdata.mass(self.pdg_id[0])
        #: (str) species name in string representation
        name = _pname(self.pdg_id[0]) if self.name is None else self.name
        if self.helicity == -1:
            name += '_l'
        elif self.helicity == +1:
            name += '_r'
        self.name = name
        #: (bool) particle is stable
        #: TODO the exclusion of neutron decays is a hotfix
        self.is_stable = (not self.ctau < np.inf or
                          self.pdg_id[0] in config.adv_set['disable_decays'])

    def init_custom_particle_data(self, name, pdg_id, helicity, ctau, mass,
                                  **kwargs):
        """Add custom particle type. (Incomplete and not debugged)"""
        #: (int) Particle Data Group Monte Carlo particle ID
        self.pdg_id = (pdg_id, helicity)
        #: (bool) if it's an electromagnetic particle
        self.is_em = kwargs.pop('is_em', abs(pdg_id) == 11 or pdg_id == 22)
        #: (bool) particle is a nucleus (not yet implemented)
        self.is_nucleus = kwargs.pop('is_nucleus',
                                     _pdata.is_nucleus(self.pdg_id[0]))
        #: (bool) particle is a hadron
        self.is_hadron = kwargs.pop('is_hadron',
                                    _pdata.is_hadron(self.pdg_id[0]))
        #: (bool) particle is a hadron
        self.is_lepton = kwargs.pop('is_lepton',
                                    _pdata.is_lepton(self.pdg_id[0]))
        #: Mass, charge, neutron number
        self.A, self.Z, self.N = getAZN(self.pdg_id[0])
        #: (float) ctau in cm
        self.ctau = ctau
        #: (float) mass in GeV
        self.mass = mass
        #: (str) species name in string representation
        self.name = name
        #: (bool) particle is stable
        self.is_stable = not self.ctau < np.inf

    def set_cs(self, cs_db):
        """Set cross section adn recalculate the dependent variables"""
        info(11, 'Obtaining cross sections for', self.pdg_id)
        self.current_cross_sections = cs_db.iam
        self.cs = cs_db[self.pdg_id[0]]
        if sum(self.cs) > 0:
            self.can_interact = True
        else:
            self.can_interact = False
        self._critical_energy()
        self._calculate_mixing_energy()

    def set_hadronic_channels(self, hadronic_db, pmanager):
        """Changes the hadronic interaction model.

        Replaces indexing of the yield dictionary from PDG IDs
        with references from partilcle manager.
        """

        self.current_hadronic_model = hadronic_db.iam
        # Collect MCEqParticle references to children
        # instead of PDG ID as index
        if self.pdg_id in hadronic_db.parents and not self.is_tracking:
            self.is_projectile = True
            self.hadr_secondaries = [
                pmanager.pdg2pref[pid]
                for pid in hadronic_db.relations[self.pdg_id]
            ]
            self.hadr_yields = {}
            for s in self.hadr_secondaries:
                self.hadr_yields[s] = hadronic_db.get_matrix(
                    self.pdg_id, s.pdg_id)
        else:
            self.is_projectile = False
            self.hadr_secondaries = []
            self.hadr_yields = {}

    def add_hadronic_production_channel(self, child, int_matrix):
        """Add a new particle that is produced in hadronic interactions.

        The int_matrix is expected to be in the correct shape and scale
        as the other interaction (dN/dE(i,j)) matrices. Energy conservation
        is not checked.
        """

        if not self.is_projectile:
            raise Exception('The particle should be a projectile.')

        if child in self.hadr_secondaries:
            info(1, 'Child {0} has been already added.'.format(child.name))
            return
        
        self.hadr_secondaries.append(child)
        self.hadr_yields[child] = int_matrix
    
    def add_decay_channel(self, child, dec_matrix, force=False):
        """Add a decay channel.
        
        The branching ratios are not renormalized and one needs to take care
        of this externally.
        """
        if self.is_stable:
            raise Exception('Cannot add decay channel to stable particle.')
        
        if child in self.children and not force:
            info(1, 'Child {0} has been already added.'.format(child.name))
            return
        elif child in self.children and force:
            info(1, 'Overwriting decay matrix of child {0}.'.format(child.name))
            self.decay_dists[child] = dec_matrix
            return

        self.children.append(child)
        self.decay_dists[child] = dec_matrix

    def set_decay_channels(self, decay_db, pmanager):
        """Populates decay channel and energy distributions"""

        if self.is_stable or self.is_tracking:
            # Variables for decays
            self.children = []
            self.decay_dists = {}
            return

        if self.pdg_id not in decay_db.parents:
            raise Exception('Unstable particle without decay distribution:',
                            self.pdg_id, self.name)

        self.children = []
        self.children = [pmanager[d] for d in decay_db.children(self.pdg_id)]
        self.decay_dists = {}
        for c in self.children:
            self.decay_dists[c] = decay_db.get_matrix(self.pdg_id, c.pdg_id)

    def track_decays(self, tracking_particle):
        children_d = {}
        for c in self.children:
            children_d[c.pdg_id] = c
        if tracking_particle.pdg_id not in list(children_d):
            info(
                17, 'Parent particle {0} does not decay into {1}'.format(
                    self.name, tracking_particle.name))
            return False
        # Copy the decay distribution from original PDG
        self.children.append(tracking_particle)
        self.decay_dists[tracking_particle] = self.decay_dists[children_d[
            tracking_particle.pdg_id]]
        return True

    def track_interactions(self, tracking_particle):
        secondaries_d = {}
        for s in self.hadr_secondaries:
            secondaries_d[s.pdg_id] = s
        if tracking_particle.pdg_id not in list(secondaries_d):
            info(
                17, 'Parent particle {0} does not produce {1} at the vertex'.
                format(self.name, tracking_particle.name))
            return False
        # Copy the decay distribution from original PDG
        self.hadr_secondaries.append(tracking_particle)
        self.hadr_yields[tracking_particle] = self.hadr_yields[secondaries_d[
            tracking_particle.pdg_id]]
        return True

    def is_secondary(self, particle_ref):
        """`True` if this projectile and produces particle `particle_ref`."""
        if not isinstance(particle_ref, self.__class__):
            raise Exception('Argument not of MCEqParticle type.')
        return particle_ref in self.hadr_secondaries

    def is_child(self, particle_ref):
        """`True` if this particle decays into `particle_ref`."""
        if not isinstance(particle_ref, self.__class__):
            raise Exception('Argument not of MCEqParticle type.')
        return particle_ref in self.children

    @property
    def hadridx(self):
        """Returns index range where particle behaves as hadron.

        Returns:
          :func:`tuple` (int,int): range on energy grid
        """
        return (self.mix_idx, self._energy_grid.d)

    @property
    def residx(self):
        """Returns index range where particle behaves as resonance.

        Returns:
          :func:`tuple` (int,int): range on energy grid
        """
        return (0, self.mix_idx)

    @property
    def lidx(self):
        """Returns lower index of particle range in state vector.

        Returns:
          (int): lower index in state vector :attr:`MCEqRun.phi`
        """
        return self.mceqidx * self._energy_grid.d

    @property
    def uidx(self):
        """Returns upper index of particle range in state vector.

        Returns:
          (int): upper index in state vector :attr:`MCEqRun.phi`
        """
        return (self.mceqidx + 1) * self._energy_grid.d

    def inverse_decay_length(self, cut=True):
        r"""Returns inverse decay length (or infinity (np.inf), if
        particle is stable), where the air density :math:`\rho` is
        factorized out.

        Args:
          E (float) : energy in laboratory system in GeV
          cut (bool): set to zero in 'resonance' regime
        Returns:
          (float): :math:`\frac{\rho}{\lambda_{dec}}` in 1/cm
        """
        try:
            dlen = self.mass / self.ctau / (self._energy_grid.c + self.mass)
            if cut:
                dlen[0:self.mix_idx] = 0.
            # Correction for bin average, since dec. length is a steep falling
            # function. This factor averages the value over bin length for
            # 10 bins per decade.
            # return 0.989 * dlen
            return dlen
        except ZeroDivisionError:
            return np.ones_like(self._energy_grid.d) * np.inf

    def inel_cross_section(self, mbarn=False):
        """Returns inelastic cross section.

        Args:
          mbarn (bool) : if True cross section in mb otherwise in cm**2
        Returns:
          (float): :math:`\\sigma_{\\rm inel}` in mb or cm**2
        """
        #: unit - :math:`\text{GeV} \cdot \text{fm}`
        GeVfm = 0.19732696312541853
        #: unit - :math:`\text{GeV} \cdot \text{cm}`
        GeVcm = GeVfm * 1e-13
        #: unit - :math:`\text{GeV}^2 \cdot \text{mbarn}`
        GeV2mbarn = 10.0 * GeVfm**2
        #: unit conversion - :math:`\text{mbarn} \to \text{cm}^2`
        mbarn2cm2 = GeV2mbarn / GeVcm**2
        if mbarn:
            return mbarn2cm2 * self.cs

        return self.cs

    def inverse_interaction_length(self):
        """Returns inverse interaction length for A_target given by config.

        Returns:
          (float): :math:`\\frac{1}{\\lambda_{int}}` in cm**2/g
        """

        m_target = self.A_target * 1.672621 * 1e-24  # <A> * m_proton [g]
        return self.cs / m_target

    def _assign_hadr_dist_idx(self, child, projidx, chidx, cmat):
        """Copies a subset, defined between indices ``projidx`` and ``chiidx``
        from the ``hadr_yields`` into ``cmat``

        Args:
          child (int): PDG ID of final state child/secondary particle
          projidx (int,int): tuple containing index range relative
                             to the projectile's energy grid
          dtridx (int,int): tuple containing index range relative
                            to the child's energy grid
          cmat (numpy.array): array reference to the interaction matrix
        """
        cmat[chidx[0]:chidx[1], projidx[0]:projidx[1]] = self.hadr_yields[
            child][chidx[0]:chidx[1], projidx[0]:projidx[1]]

    def _assign_decay_idx(self, child, projidx, chidx, cmat):
        """Copies a subset, defined between indices ``projidx`` and ``chiidx``
        from the ``hadr_yields`` into ``cmat``

        Args:
          child (int): PDG ID of final state child/secondary particle
          projidx (int,int): tuple containing index range relative
                             to the projectile's energy grid
          dtridx (int,int): tuple containing index range relative
                            to the child's energy grid
          cmat (numpy.array): array reference to the interaction matrix
        """
        cmat[chidx[0]:chidx[1], projidx[0]:projidx[1]] = self.decay_dists[
            child][chidx[0]:chidx[1], projidx[0]:projidx[1]]

    def dN_dxlab(self, kin_energy, sec_pdg, verbose=True, **kwargs):
        r"""Returns :math:`dN/dx_{\rm Lab}` for interaction energy close
        to ``kin_energy`` for hadron-air collisions.

        The function respects modifications applied via :func:`_set_mod_pprod`.

        Args:
            kin_energy (float): approximate interaction kin_energy
            prim_pdg (int): PDG ID of projectile
            sec_pdg (int): PDG ID of secondary particle
            verbose (bool): print out the closest enerkin_energygy
        Returns:
            (numpy.array, numpy.array): :math:`x_{\rm Lab}`, :math:`dN/dx_{\rm Lab}`
        """

        eidx = (np.abs(self._energy_grid.c - kin_energy)).argmin()
        en = self._energy_grid.c[eidx]
        info(10, 'Nearest energy, index: ', en, eidx, condition=verbose)

        m = self.hadr_yields[sec_pdg]
        xl_grid = (self._energy_grid.c[:eidx + 1]) / en
        xl_dist = en * xl_grid * m[:eidx +
                                   1, eidx] / self._energy_grid.w[:eidx + 1]

        return xl_grid, xl_dist

    def dNdec_dxlab(self, kin_energy, sec_pdg, verbose=True, **kwargs):
        r"""Returns :math:`dN/dx_{\rm Lab}` for interaction energy close
        to ``kin_energy`` for hadron-air collisions.

        The function respects modifications applied via :func:`_set_mod_pprod`.

        Args:
            kin_energy (float): approximate interaction energy
            prim_pdg (int): PDG ID of projectile
            sec_pdg (int): PDG ID of secondary particle
            verbose (bool): print out the closest energy
        Returns:
            (numpy.array, numpy.array): :math:`x_{\rm Lab}`, :math:`dN/dx_{\rm Lab}`
        """

        eidx = (np.abs(self._energy_grid.c - kin_energy)).argmin()
        en = self._energy_grid.c[eidx]
        info(10, 'Nearest energy, index: ', en, eidx, condition=verbose)

        m = self.decay_dists[sec_pdg]
        xl_grid = (self._energy_grid.c[:eidx + 1]) / en
        xl_dist = en * xl_grid * m[:eidx +
                                   1, eidx] / self._energy_grid.w[:eidx + 1]

        return xl_grid, xl_dist

    def dN_dEkin(self, kin_energy, sec_pdg, verbose=True, **kwargs):
        r"""Returns :math:`dN/dE_{\rm Kin}` in lab frame for an interaction energy
        close to ``kin_energy`` (total) for hadron-air collisions.

        The function respects modifications applied via :func:`_set_mod_pprod`.

        Args:
            kin_energy (float): approximate interaction energy
            prim_pdg (int): PDG ID of projectile
            sec_pdg (int): PDG ID of secondary particle
            verbose (bool): print out the closest energy
        Returns:
            (numpy.array, numpy.array): :math:`x_{\rm Lab}`, :math:`dN/dx_{\rm Lab}`
        """

        eidx = (np.abs(self._energy_grid.c - kin_energy)).argmin()
        en = self._energy_grid.c[eidx]
        info(10, 'Nearest energy, index: ', en, eidx, condition=verbose)

        m = self.hadr_yields[sec_pdg]
        ekin_grid = self._energy_grid.c
        elab_dist = m[:eidx + 1, eidx]  / self._energy_grid.w[eidx]

        return ekin_grid[:eidx + 1], elab_dist

    def dN_dxf(self,
               energy,
               prim_pdg,
               sec_pdg,
               pos_only=True,
               verbose=True,
               **kwargs):
        r"""Returns :math:`dN/dx_{\rm F}` in c.m. for interaction energy close
        to ``energy`` (lab. not kinetic) for hadron-air collisions.

        The function respects modifications applied via :func:`_set_mod_pprod`.

        Args:
            energy (float): approximate interaction lab. energy
            prim_pdg (int): PDG ID of projectile
            sec_pdg (int): PDG ID of secondary particle
            verbose (bool): print out the closest energy

        Returns:
            (numpy.array, numpy.array): :math:`x_{\rm F}`, :math:`dN/dx_{\rm F}`
        """
        if not hasattr(self, '_ptav_sib23c'):
            # Load spline of average pt distribution as a funtion of log(E_lab) from sib23c
            import pickle
            from os.path import join
            self._ptav_sib23c = pickle.load(
                open(join(config.data_dir, 'sibyll23c_aux.ppd'), 'rb'))[0]

        def xF(xL, Elab, ppdg):

            m = {2212: 0.938, 211: 0.139, 321: 0.493}
            mp = m[2212]

            Ecm = np.sqrt(2 * Elab * mp + 2 * mp**2)
            Esec = xL * Elab
            betacm = np.sqrt((Elab - mp) / (Elab + mp))
            gammacm = (Elab + mp) / Ecm
            avpt = self._ptav_sib23c[ppdg](
                np.log(np.sqrt(Elab**2) - m[np.abs(ppdg)]**2))

            xf = 2 * (-betacm * gammacm * Esec + gammacm *
                      np.sqrt(Esec**2 - m[np.abs(ppdg)]**2 - avpt**2)) / Ecm
            dxl_dxf = 1. / (
                2 *
                (-betacm * gammacm * Elab + xL * Elab**2 * gammacm / np.sqrt(
                    (xL * Elab)**2 - m[np.abs(ppdg)]**2 - avpt**2)) / Ecm)

            return xf, dxl_dxf

        eidx = (np.abs(self._energy_grid.c + self.mass - energy)).argmin()
        en = self._energy_grid.c[eidx] + self.mass
        info(2, 'Nearest energy, index: ', en, eidx, condition=verbose)
        m = self.hadr_yields[sec_pdg]
        xl_grid = (self._energy_grid.c[:eidx + 1] + self.mass) / en
        xl_dist = xl_grid * en * m[:eidx + 1, eidx] / np.diag(
            self._energy_grid.w)[:eidx + 1]
        xf_grid, dxl_dxf = xF(xl_grid, en, sec_pdg)
        xf_dist = xl_dist * dxl_dxf

        if pos_only:
            xf_dist = xf_dist[xf_grid >= 0]
            xf_grid = xf_grid[xf_grid >= 0]
            return xf_grid, xf_dist

        return xf_grid, xf_dist

    def _critical_energy(self):
        """Returns critical energy where decay and interaction
        are balanced.

        Approximate value in Air.

        Returns:
          (float): :math:`\\frac{m\\ 6.4 \\text{km}}{c\\tau}` in GeV
        """
        if self.is_stable or self.ctau <= 0.:
            self.E_crit = np.inf
        else:
            self.E_crit = self.mass * 6.4e5 / self.ctau

    def _calculate_mixing_energy(self):
        """Calculates interaction/decay length in Air and decides if
        the particle has resonance and/or hadron behavior.

        Class attributes :attr:`is_mixed`, :attr:`E_mix`, :attr:`mix_idx`,
        :attr:`is_resonance` are calculated here. If the option `no_mixing`
        is set in config.adv_config particle is forced to be a resonance
        or hadron behavior.

        Args:
          e_grid (numpy.array): energy grid of size :attr:`d`
          max_density (float): maximum density on the integration path (largest
                               decay length)
        """

        cross_over = config.hybrid_crossover
        max_density = config.max_density

        d = self._energy_grid.d
        inv_intlen = self.inverse_interaction_length()

        inv_declen = self.inverse_decay_length()
        # If particle is stable, no "mixing" necessary
        if (not np.any(np.nan_to_num(inv_declen) > 0.)
                or abs(self.pdg_id[0]) in config.adv_set["exclude_from_mixing"]
                or config.adv_set['no_mixing']
                or self.pdg_id[0] in config.adv_set['disable_decays']):
            self.mix_idx = 0
            self.is_mixed = False
            self.is_resonance = False
            return
            
        # If particle is forced to be a "resonance" 
        if (np.abs(self.pdg_id[0]) in config.adv_set["force_resonance"]):
            self.mix_idx = d - 1
            self.E_mix = self._energy_grid.c[self.mix_idx]
            self.is_mixed = False
            self.is_resonance = True
        # Particle can interact and decay
        elif self.can_interact and not self.is_stable:
            # This is lambda_dec / lambda_int
            threshold = np.zeros_like(inv_intlen)
            mask = inv_declen != 0.
            threshold[mask] = inv_intlen[mask] * max_density / inv_declen[mask] 
            del mask
            self.mix_idx = np.where(threshold > cross_over)[0][0]
            self.E_mix = self._energy_grid.c[self.mix_idx]
            self.is_mixed = True
            self.is_resonance = False
        # These particles don't interact but can decay (e.g. tau leptons)
        elif not self.can_interact and not self.is_stable:
            mask = inv_declen != 0.
            self.mix_idx = np.where(
                max_density / inv_declen > config.dXmax)[0][0]
            self.E_mix = self._energy_grid.c[self.mix_idx]
            self.is_mixed = True
            self.is_resonance = False
        # Particle is stable but that should be handled above
        else:
            print(self.name, "This case shouldn't occur.")
            threshold = np.inf
            self.mix_idx = 0
            self.is_mixed = False
            self.is_resonance = False


    def __eq__(self, other):
        """Checks name for equality"""
        if isinstance(other, MCEqParticle):
            return self.name == other.name
        else:
            return NotImplemented

    def __neq__(self, other):
        """Checks name for equality"""
        if isinstance(other, MCEqParticle):
            return self.name != other.name
        else:
            return NotImplemented

    def __hash__(self):
        """Instruction for comuting the hash"""
        return hash(self.name)

    def __repr__(self):
        a_string = ("""
        {0}:
        is_hadron     : {1}
        is_lepton     : {2}
        is_nucleus    : {3}
        is_stable     : {4}
        is_mixed      : {5}
        is_resonance  : {6}
        is_tracking   : {7}
        is_projectile : {8}
        mceqidx       : {9}
        E_mix         : {10:2.1e} GeV\n""").format(
            self.name, self.is_hadron, self.is_lepton, self.is_nucleus,
            self.is_stable, self.is_mixed, self.is_resonance, self.is_tracking,
            self.is_projectile, self.mceqidx, self.E_mix)
        return a_string


class ParticleManager(object):
    """Database for objects of :class:`MCEqParticle`.

    Authors:
        Anatoli Fedynitch (DESY)
        Jonas Heinze (DESY)
    """
    def __init__(self, pdg_id_list, energy_grid, cs_db, mod_table=None):
        # (dict) Dimension of primary grid
        self._energy_grid = energy_grid
        # Particle index shortcuts
        #: (dict) Converts PDG ID to index in state vector
        self.pdg2mceqidx = {}
        #: (dict) Converts particle name to index in state vector
        self.pname2mceqidx = {}
        #: (dict) Converts PDG ID to reference of
        # :class:`particlemanager.MCEqParticle`
        self.pdg2pref = {}
        #: (dict) Converts particle name to reference of
        #: class:`particlemanager.MCEqParticle`
        self.pname2pref = {}
        #: (dict) Converts MCEq index to reference of
        #: class:`particlemanager.MCEqParticle`
        self.mceqidx2pref = {}
        #: (dict) Converts index in state vector to PDG ID
        self.mceqidx2pdg = {}
        #: (dict) Converts index in state vector to reference
        # of :class:`particlemanager.MCEqParticle`
        self.mceqidx2pname = {}
        # Save setup of tracked particles to reapply the relations
        # when models change
        self.tracking_relations = {}
        #: (int) Total number of species
        self.nspec = 0
        # save currently applied cross section model
        self.current_cross_sections = None
        # save currently applied hadronic model
        self.current_hadronic_model = None
        # Cross section database
        self._cs_db = cs_db
        # Dictionary to save te tracking particle config
        self.tracking_relations = []
        # Save the tracking relations requested by default tracking
        self._tracking_requested = []

        self._init_categories(particle_pdg_list=pdg_id_list)

        self.print_particle_tables(10)

    def set_cross_sections_db(self, cs_db):
        """Sets the inelastic cross section to each interacting particle.

        This applies to most of the hadrons and does not imply that the
        particle becomes a projectile. parents need in addition defined
        hadronic channels.
        """

        info(5, 'Setting cross section particle variables.')
        if self.current_cross_sections == cs_db.iam:
            info(10, 'Same cross section model not applied to particles.')
            return

        for p in self.cascade_particles:
            p.set_cs(cs_db)
        self.current_cross_sections = cs_db.iam
        self._update_particle_tables()

    def set_decay_channels(self, decay_db):
        """Attaches the references to the decay yield tables to
        each unstable particle"""

        info(5, 'Setting decay info for particles.')
        for p in self.all_particles:
            p.set_decay_channels(decay_db, self)

        self._restore_tracking_setup()
        self._update_particle_tables()

    def set_interaction_model(self,
                              cs_db,
                              hadronic_db,
                              updated_parent_list=None,
                              force=False):
        """Attaches the references to the hadronic yield tables to
        each projectile particle"""

        info(5, 'Setting hadronic secondaries for particles.')
        if (self.current_hadronic_model == hadronic_db.iam and 
            not force and updated_parent_list is None):
            info(10, 'Same hadronic model not applied to particles.')
            return
        if updated_parent_list is not None:
            self._init_categories(updated_parent_list)

        for p in self.cascade_particles:
            p.set_cs(cs_db)
            p.set_hadronic_channels(hadronic_db, self)

        self.current_hadronic_model = hadronic_db.iam
        self._update_particle_tables()

    def set_continuous_losses(self, contloss_db):
        """Set continuous losses terms to particles with ionization
        and radiation losses."""

        for p in self.cascade_particles:
            if p.pdg_id in contloss_db:
                p.has_contloss = True
                p.dEdX = contloss_db[p.pdg_id]

    def add_tracking_particle(self,
                              parent_list,
                              child_pdg,
                              alias_name,
                              from_interactions=False):
        """Allows tracking decay and particle production chains.

        Replaces previous ``obs_particle`` function that allowed to track
        only leptons from decays certain particles. This present feature
        removes the special PDG IDs 71XX, 72XX, etc and allows to define
        any channel like::

            $ particleManagerInstance.add_tracking_particle([211], 14, 'pi_numu')

        This will store muon neutrinos from pion decays under the alias 'pi_numu'.
        Multiple parents are allowed::

            $ particleManagerInstance.add_tracking_particle(
                [411, 421, 431], 14, 'D_numu')

        Args:

            alias (str): Name alias under which the result is accessible in get_solution
            parents (list): list of parent particle PDG ID's
            child (int): Child particle
            from_interactions (bool): track particles from interactions
        """
        from copy import copy

        info(10, 'requested for', parent_list, child_pdg, alias_name)

        for p in parent_list:
            if (p, child_pdg, alias_name,
                    from_interactions) in self._tracking_requested:
                continue
            self._tracking_requested.append(
                (p, child_pdg, alias_name, from_interactions))

        # Check if tracking particle with the alias not yet defined
        # and create new one of necessary
        if alias_name in self.pname2pref:
            info(15, 'Re-using tracking particle', alias_name)
            tracking_particle = self.pname2pref[alias_name]
        elif child_pdg not in self.pdg2pref:
            info(15, 'Tracking child not a available',
                 'for this interaction model, skipping.')
            return
        else:
            info(10, 'Creating new tracking particle')
            # Copy all preferences of the original particle
            tracking_particle = copy(self.pdg2pref[child_pdg])
            tracking_particle.is_tracking = True
            tracking_particle.name = alias_name
            # Find a unique PDG ID for the new tracking particle
            # print child_pdg[0], int(copysign(1000000, child_pdg[0]))
            unique_child_pdg = (child_pdg[0] +
                                int(copysign(1000000, child_pdg[0])),
                                tracking_particle.helicity)

            for i in range(100):
                if unique_child_pdg not in list(self.pdg2pref):
                    break
                info(
                    20, '{0}: trying to find unique_pdg ({1}) for {2}'.format(
                        i, tracking_particle.name, unique_child_pdg))
                unique_child_pdg = (unique_child_pdg[0] +
                                    int(copysign(10000, child_pdg[0])),
                                    tracking_particle.helicity)
            tracking_particle.unique_pdg_id = unique_child_pdg

        # Track if attempt to add the tracking particle succeeded at least once
        track_success = False
        # Include antiparticle

        for parent_pdg in list(
                set(parent_list + [(-p, h) for (p, h) in parent_list])):
            if parent_pdg not in self.pdg2pref:
                info(15,
                     'Parent particle {0} does not exist.'.format(parent_pdg))
                continue
            if (parent_pdg, child_pdg, alias_name,
                    from_interactions) in self.tracking_relations:
                info(
                    20, 'Tracking of {0} from {1} already activated.'.format(
                        tracking_particle.name,
                        self.pdg2pref[parent_pdg].name))
                continue

            if not from_interactions:
                track_method = self.pdg2pref[parent_pdg].track_decays
            else:
                track_method = self.pdg2pref[parent_pdg].track_interactions

            # Check if the tracking is successful. If not the particle is not
            # a child of the parent particle
            if track_method(tracking_particle):
                info(
                    15, 'Parent particle {0} tracking scheduled.'.format(
                        parent_pdg))
                self.tracking_relations.append(
                    (parent_pdg, child_pdg, alias_name, from_interactions))
                track_success = True
        if track_success and tracking_particle.name not in list(
                self.pname2pref):
            tracking_particle.mceqidx = np.max(list(self.mceqidx2pref)) + 1
            self.all_particles.append(tracking_particle)
            self.cascade_particles.append(tracking_particle)
            self._update_particle_tables()
            info(
                10, 'tracking particle {0} successfully added.'.format(
                    tracking_particle.name))

    def track_leptons_from(self,
                           parent_pdg_list,
                           prefix,
                           exclude_em=True,
                           from_interactions=False,
                           use_helicities=False):
        """Adds tracking particles for all leptons coming from decays of parents
        in `parent_pdg_list`.
        """

        leptons = [
            p for p in self.all_particles if p.is_lepton
            and not (p.is_em == exclude_em) and not p.is_tracking
        ]

        for lepton in leptons:
            if not use_helicities and lepton.pdg_id[1] != 0:
                continue
            self.add_tracking_particle(parent_pdg_list, lepton.pdg_id,
                                       prefix + lepton.name, from_interactions)

    def _init_categories(self, particle_pdg_list):
        """Determines the list of particles for calculation and
        returns lists of instances of :class:`data.MCEqParticle` .

        The particles which enter this list are those, which have a
        defined index in the SIBYLL 2.3 interaction model. Included are
        most relevant baryons and mesons and some of their high mass states.
        More details about the particles which enter the calculation can
        be found in :mod:`particletools`.

        Returns:
          (tuple of lists of :class:`data.MCEqParticle`): (all particles,
          cascade particles, resonances)
        """
        from MCEq.particlemanager import MCEqParticle

        info(5, "Generating particle list.")

        if particle_pdg_list is not None:
            particles = particle_pdg_list
        else:
            from particletools.tables import SibyllParticleTable
            modtab = SibyllParticleTable()
            particles = modtab.baryons + modtab.mesons + modtab.leptons

        # Remove duplicates
        particles = sorted(list(set(particles)))

        # Initialize particle objects
        particle_list = [
            MCEqParticle(pdg, hel, self._energy_grid, self._cs_db)
            for pdg, hel in particles
        ]

        # Sort by critical energy (= int_len ~== dec_length ~ int_cs/tau)
        particle_list.sort(key=lambda x: x.E_crit, reverse=False)

        # Cascade particles will "live" on the grid and have an mceqidx assigned
        self.cascade_particles = [
            p for p in particle_list if not p.is_resonance
        ]

        self.cascade_particles = sorted(self.cascade_particles,
                                        key=lambda p: abs(p.pdg_id[0]))

        # These particles will only exist implicitely and integated out
        self.resonances = [p for p in particle_list if p.is_resonance]

        # Assign an mceqidx (position in state vector) to each explicit particle
        # Resonances will kepp the default mceqidx = -1
        for mceqidx, h in enumerate(self.cascade_particles):
            h.mceqidx = mceqidx

        self.all_particles = self.cascade_particles + self.resonances
        self._update_particle_tables()

    def add_new_particle(self, new_mceq_particle):
        
        if new_mceq_particle in self.all_particles:
            info(0, 'Particle {0}/{1} has already been added. Use it.'.format(
                new_mceq_particle.name, new_mceq_particle.pdg_id
            ))
            return

        if not new_mceq_particle.is_resonance:
            info(2, 'New particle {0}/{1} is not a resonance.'.format(
                new_mceq_particle.name, new_mceq_particle.pdg_id
            ))
            new_mceq_particle.mceqidx = len(self.cascade_particles)
            self.cascade_particles.append(new_mceq_particle)
        else:
            info(2, 'New particle {0}/{1} is a resonance.'.format(
                new_mceq_particle.name, new_mceq_particle.pdg_id
            ))
            self.resonances.append(new_mceq_particle)

        self.all_particles = self.cascade_particles + self.resonances
        self._update_particle_tables()

    def _update_particle_tables(self):
        """Update internal mapping tables after changes to the particle
        list occur."""

        self.n_cparticles = len(self.cascade_particles)
        self.dim = self._energy_grid.d
        self.dim_states = self._energy_grid.d * self.n_cparticles

        # Clean all dictionaries
        [
            d.clear() for d in [
                self.pdg2mceqidx, self.pname2mceqidx, self.mceqidx2pdg,
                self.mceqidx2pname, self.mceqidx2pref, self.pdg2pref,
                self.pname2pref
            ]
        ]

        for p in self.all_particles:
            self.pdg2mceqidx[p.unique_pdg_id] = p.mceqidx
            self.pname2mceqidx[p.name] = p.mceqidx
            self.mceqidx2pdg[p.mceqidx] = p.unique_pdg_id
            self.mceqidx2pname[p.mceqidx] = p.name
            self.mceqidx2pref[p.mceqidx] = p
            self.pdg2pref[p.unique_pdg_id] = p
            self.pname2pref[p.name] = p

        self.print_particle_tables(20)

    def _restore_tracking_setup(self):
        """Restores the setup of tracking particles after model changes."""

        info(10, 'Restoring tracking particle setup')

        if not self.tracking_relations and config.enable_default_tracking:
            self._init_default_tracking()
            return

        # Clear tracking_relations for this initialization
        self.tracking_relations = []

        for pid, cid, alias, int_dec in self._tracking_requested:
            if pid not in self.pdg2pref:
                info(15, 'Can not restore {0}, since not in particle list.')
                continue
            self.add_tracking_particle([pid], cid, alias, int_dec)

    def _init_default_tracking(self):
        """Add default tracking particles for leptons from pi, K, and mu"""
        # Init default tracking particles
        info(1, 'Initializing default tracking categories (pi, K, mu)')
        self._tracking_requested_by_default = []
        for parents, prefix, with_helicity in [([(211, 0)], 'pi_', True),
                                               ([(321, 0)], 'k_', True),
                                               ([(13, -1),
                                                 (13, 1)], 'mulr_', False),
                                               ([(13, 0)], 'mu_h0_', False),
                                               ([(13, -1), (13, 0),
                                                 (13, 1)], 'mu_', False),
                                               ([(310, 0),
                                                 (130, 0)], 'K0_', False)]:
            self.track_leptons_from(parents,
                                    prefix,
                                    exclude_em=True,
                                    use_helicities=with_helicity)

        # Track prompt leptons
        self.track_leptons_from([
            p.pdg_id for p in self.all_particles if p.ctau < config.prompt_ctau
        ],
                                'prcas_',
                                exclude_em=True,
                                use_helicities=False)
        # Track leptons from interaction vertices (also prompt)
        self.track_leptons_from(
            [p.pdg_id for p in self.all_particles if p.is_projectile],
            'prres_',
            exclude_em=True,
            from_interactions=True,
            use_helicities=False)

        self.track_leptons_from(
            [p.pdg_id for p in self.all_particles if p.is_em],
            'em_',
            exclude_em=True,
            from_interactions=True,
            use_helicities=False)

    def __contains__(self, pdg_id_or_name):
        """Defines the `in` operator to look for particles"""
        if isinstance(pdg_id_or_name, six.integer_types):
            pdg_id_or_name = (pdg_id_or_name, 0)
        elif isinstance(pdg_id_or_name, six.string_types):
            pdg_id_or_name = (_pdata.pdg_id(pdg_id_or_name), 0)
        return pdg_id_or_name in list(self.pdg2pref)

    def __getitem__(self, pdg_id_or_name):
        """Returns reference to particle object."""
        if isinstance(pdg_id_or_name, tuple):
            return self.pdg2pref[pdg_id_or_name]
        elif isinstance(pdg_id_or_name, six.integer_types):
            return self.pdg2pref[(pdg_id_or_name, 0)]
        else:
            return self.pdg2pref[(_pdata.pdg_id(pdg_id_or_name), 0)]

    def keys(self):
        """Returns pdg_ids of all particles"""
        return [p.pdg_id for p in self.all_particles]

    def __repr__(self):
        str_out = ""
        ident = 3 * ' '
        for s in self.all_particles:
            str_out += s.name + '\n' + ident
            str_out += 'PDG id : ' + str(s.pdg_id) + '\n' + ident
            str_out += 'MCEq idx : ' + str(s.mceqidx) + '\n\n'

        return str_out

    def print_particle_tables(self, min_dbg_lev=2):

        info(min_dbg_lev, "Hadrons and stable particles:", no_caller=True)
        print_in_rows(min_dbg_lev, [
            p.name for p in self.all_particles
            if p.is_hadron and not p.is_resonance and not p.is_mixed
        ])

        info(min_dbg_lev, "\nMixed:", no_caller=True)
        print_in_rows(min_dbg_lev,
                      [p.name for p in self.all_particles if p.is_mixed])

        info(min_dbg_lev, "\nResonances:", no_caller=True)
        print_in_rows(min_dbg_lev,
                      [p.name for p in self.all_particles if p.is_resonance])

        info(min_dbg_lev, "\nLeptons:", no_caller=True)
        print_in_rows(min_dbg_lev, [
            p.name
            for p in self.all_particles if p.is_lepton and not p.is_tracking
        ])
        info(min_dbg_lev, "\nTracking:", no_caller=True)
        print_in_rows(min_dbg_lev,
                      [p.name for p in self.all_particles if p.is_tracking])

        info(min_dbg_lev,
             "\nTotal number of species:",
             self.n_cparticles,
             no_caller=True)

        # list particle indices
        if False:
            info(10, "Particle matrix indices:", no_caller=True)
            some_index = 0
            for p in self.cascade_particles:
                for i in range(self._energy_grid.d):
                    info(10, p.name + '_' + str(i), some_index, no_caller=True)
                    some_index += 1
