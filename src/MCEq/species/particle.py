"""``MCEqParticle``: one particle's properties, channels, and rates.

Moved verbatim from the flat ``MCEq.particlemanager`` (§13.2 item 6). The class
reads the shared table through :mod:`MCEq.species.constants` and its settings
from the injected physics group; the function-local ``config`` fallback in
``__init__`` is unchanged and carries the C5 ledger line.
"""

import numpy as np

from MCEq.misc import average_A_target, getAZN, info
from MCEq.species.constants import ATOMIC_MASS_UNIT_G, _pdata, _pname


class MCEqParticle:
    """Bundles different particle properties for simplified
    availability of particle properties in :class:`MCEq.core.MCEqRun`.

    Args:
      pdg_id (int): PDG ID of the particle
      egrid (np.array, optional): energy grid (centers)
      cs_db (object, optional): reference to an instance of
                                :class:`InteractionYields`
      physics: settings group (``config.physics``) holding the decay and
               resonance filters. :class:`ParticleManager` passes its own.
    """

    def __init__(
        self,
        pdg_id,
        helicity,
        energy_grid=None,
        cs_db=None,
        init_pdata_defaults=True,
        A_target=None,
        *,
        physics,
    ):
        #: settings group this particle reads its filters from
        self._physics = physics
        if A_target is None:
            A_target = average_A_target(self._physics.interaction_medium)
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
        #: (bool) particle is charged
        self.is_charged = False
        #: (float) ctau in cm
        self.ctau = None
        #: (float) mass in GeV
        self.mass = None
        #: (int) Charge
        self.charge = None
        #: (str) species name in string representation
        self.name = None
        #: Mass, charge, neutron number
        self.A, self.Z, self.N = getAZN(pdg_id)
        #: (bool) if True, this particle is folded into other particles'
        #: matrices at build time and does not get its own state-vector slot.
        #: Set by ``adv_set["force_resonance"]``; default ``False`` (full
        #: propagation for everything).
        self.is_resonance = False
        #: (bool) particle is interacting projectile
        self.is_projectile = False
        #: (bool) particle is stable
        self.is_stable = pdg_id in physics.filters["disable_decays"]
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

        #: (float) interaction threshold (idx where cs>0)
        self.int_idx = 0
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
        self.A_target = A_target

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
        if self.is_nucleus:
            self.charge = self.Z
        else:
            self.charge = int(_pdata.charge(self.pdg_id[0]))
        self.is_charged = bool(abs(self.charge))
        #: (float) ctau in cm
        self.ctau = _pdata.ctau(self.pdg_id[0])
        #: (float) mass in GeV
        self.mass = _pdata.mass(self.pdg_id[0])
        #: (str) species name in string representation
        name = _pname(self.pdg_id[0]) if self.name is None else self.name
        if self.helicity == -1:
            name += "_l"
        elif self.helicity == +1:
            name += "_r"
        self.name = name
        #: (bool) particle is stable
        #: TODO the exclusion of neutron decays is a hotfix
        self.is_stable = (
            not self.ctau < np.inf
            or self.pdg_id[0] in self._physics.filters["disable_decays"]
        )

    def init_custom_particle_data(self, name, pdg_id, helicity, ctau, mass, **kwargs):
        """Add custom particle type. (Incomplete and not debugged)"""
        #: (int) Particle Data Group Monte Carlo particle ID
        self.pdg_id = (pdg_id, helicity)
        #: (bool) if it's an electromagnetic particle
        self.is_em = kwargs.pop("is_em", abs(pdg_id) == 11 or pdg_id == 22)
        #: (bool) particle is a nucleus (not yet implemented)
        self.is_nucleus = kwargs.pop("is_nucleus", _pdata.is_nucleus(self.pdg_id[0]))
        #: (bool) particle is a hadron
        self.is_hadron = kwargs.pop("is_hadron", _pdata.is_hadron(self.pdg_id[0]))
        #: (bool) particle is a hadron
        self.is_lepton = kwargs.pop("is_lepton", _pdata.is_lepton(self.pdg_id[0]))
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
        info(11, "Obtaining cross sections for", self.pdg_id)
        self.current_cross_sections = cs_db.iam
        self.cs = cs_db[self.pdg_id[0]]
        if sum(self.cs) > 0:
            self.can_interact = True
        else:
            self.can_interact = False

        self._critical_energy()
        self._apply_force_resonance()

    def _set_channels(self, channel_table, pmanager, child_keys, keep_tracking=False):
        """One wiring routine for every channel table (plan §6.1, R2-A).

        ``channel_table`` is a ``MCEq.data.protocols.ChannelTable``:
        the child keys of this particle are remapped to particle-manager
        references and the matrices are fetched through the table's
        ``get_matrix``. Domain guards stay in the callers: the
        hadronic/decay wiring differs in how a missing parent is treated
        (soft skip vs hard error) and only hadronic carries tracking
        particles, so collapsing the guards would change behaviour —
        the shared part is this remap-and-fetch body.

        Returns ``(refs, yields)`` so the callers rebind their own
        secondaries/children list and yield dict (rebinding, not
        clearing: tracking-particle copies share these dict objects
        until the first re-wire).
        """
        refs = [pmanager[key] for key in child_keys]
        if keep_tracking:
            refs += [ref for ref in self.hadr_secondaries if ref.is_tracking]
        yields = {}
        for ref in refs:
            yields[ref] = channel_table.get_matrix(self.pdg_id, ref.pdg_id)
        return refs, yields

    def set_hadronic_channels(self, hadronic_db, pmanager):
        """Changes the hadronic interaction model.

        Replaces indexing of the yield dictionary from PDG IDs
        with references from partilcle manager.
        """

        self.current_hadronic_model = hadronic_db.iam
        if self.pdg_id in hadronic_db.parents and not self.is_tracking:
            self.is_projectile = True
            self.hadr_secondaries, self.hadr_yields = self._set_channels(
                hadronic_db,
                pmanager,
                hadronic_db.relations[self.pdg_id],
                keep_tracking=True,
            )
        else:
            self.is_projectile = False
            self.hadr_secondaries = []
            self.hadr_yields = {}

        if self.is_tracking:
            # Copy properties of the original particle for tracking
            orig_part = pmanager.pdg2pref[self.pdg_id]
            self.is_projectile = orig_part.is_projectile
            if orig_part in orig_part.hadr_secondaries:
                self.hadr_secondaries.append(self)
                self.hadr_yields[self] = orig_part.hadr_yields[orig_part]

    def add_hadronic_production_channel(self, child, int_matrix):
        """Add a new particle that is produced in hadronic interactions.

        The int_matrix is expected to be in the correct shape and scale
        as the other interaction (dN/dE(i,j)) matrices. Energy conservation
        is not checked.
        """

        if not self.is_projectile:
            raise Exception("The particle should be a projectile.")

        if child in self.hadr_secondaries:
            info(1, f"Child {child.name} has been already added.")
            return

        self.hadr_secondaries.append(child)
        self.hadr_yields[child] = int_matrix

    def add_decay_channel(self, child, dec_matrix, force=False):
        """Add a decay channel.

        The branching ratios are not renormalized and one needs to take care
        of this externally.
        """
        if self.is_stable:
            raise Exception("Cannot add decay channel to stable particle.")

        if child in self.children and not force:
            info(1, f"Child {child.name} has been already added.")
            return
        if child in self.children and force:
            info(1, f"Overwriting decay matrix of child {child.name}.")
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
            raise Exception(
                "Unstable particle without decay distribution:", self.pdg_id, self.name
            )

        self.children, self.decay_dists = self._set_channels(
            decay_db, pmanager, decay_db.children(self.pdg_id)
        )

    def track_decays(self, tracking_particle):
        children_d = {}
        for c in self.children:
            children_d[c.pdg_id] = c
        if tracking_particle.pdg_id not in list(children_d):
            _n = self.name
            _tn = tracking_particle.name
            info(
                17,
                f"Parent particle {_n} does not decay into {_tn}",
            )
            return False
        # Copy the decay distribution from original PDG
        self.children.append(tracking_particle)
        self.decay_dists[tracking_particle] = self.decay_dists[
            children_d[tracking_particle.pdg_id]
        ]
        return True

    def track_interactions(self, tracking_particle):
        secondaries_d = {}
        for s in self.hadr_secondaries:
            secondaries_d[s.pdg_id] = s
        if tracking_particle.pdg_id not in list(secondaries_d):
            _n = self.name
            _tn = tracking_particle.name
            info(
                17,
                f"Parent particle {_n} does not produce {_tn} at the vertex",
            )
            return False
        # Copy the interaction matrix from original PDG
        self.hadr_secondaries.append(tracking_particle)
        self.hadr_yields[tracking_particle] = self.hadr_yields[
            secondaries_d[tracking_particle.pdg_id]
        ]
        return True

    def is_secondary(self, particle_ref):
        """`True` if this projectile and produces particle `particle_ref`."""
        if not isinstance(particle_ref, self.__class__):
            raise Exception("Argument not of MCEqParticle type.")
        return particle_ref in self.hadr_secondaries

    def is_child(self, particle_ref):
        """`True` if this particle decays into `particle_ref`."""
        if not isinstance(particle_ref, self.__class__):
            raise Exception("Argument not of MCEqParticle type.")
        return particle_ref in self.children

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

    def inverse_decay_length(self):
        r"""Returns inverse decay length, where the air density
        :math:`\rho` is factorized out.

        The physical decay rate per unit path length is
        :math:`1/\lambda_{dec} = m/(c\tau\,p)`, with :math:`p` the lab
        momentum (:math:`p = \beta\gamma m c`). Using the total energy
        :math:`E = E_{kin} + m` in place of :math:`p` drops the factor
        :math:`1/\beta` and understates decay of slow (:math:`\beta<1`)
        particles; the momentum form is exact at all energies.

        A stable particle (``ctau = inf``) returns zeros -- the physical
        inverse length -- not the ``inf`` the pre-fix docstring promised;
        ``inf`` is what the ``ctau == 0`` path returns, now as a length-``d``
        vector (B9, R3).

        Returns:
          (numpy.array): :math:`\frac{\rho}{\lambda_{dec}}` in 1/cm,
          shape ``(d,)`` on every path
        """
        try:
            e_tot = self._energy_grid.c + self.mass
            p_lab = np.sqrt(e_tot**2 - self.mass**2)
            return self.mass / self.ctau / p_lab
        except ZeroDivisionError:
            return np.ones_like(self._energy_grid.c) * np.inf

    def prod_cross_section(self, mbarn=False):
        """Returns production cross section.

        Args:
          mbarn (bool) : if True cross section in mb otherwise in cm**2
        Returns:
          (float): :math:`\\sigma_{\\rm prod}` in mb or cm**2
        """
        from MCEq.data.cross_sections import InteractionCrossSections

        if mbarn:
            return self.cs / InteractionCrossSections.mbarn2cm2

        return self.cs

    def inverse_interaction_length(self):
        """Returns inverse interaction length for this particle's ``A_target``.

        Returns:
          (float): :math:`\\frac{1}{\\lambda_{int}}` in cm**2/g
        """

        m_target = self.A_target * ATOMIC_MASS_UNIT_G  # <A> * m_u [g]
        return self.cs / m_target

    def _assign_hadr_dist(self, child, cmat):
        """Copies the full ``hadr_yields[child]`` matrix into ``cmat``.

        Args:
          child (MCEqParticle): final-state secondary particle.
          cmat (np.ndarray): destination interaction-matrix block.
        """
        cmat[:, :] = self.hadr_yields[child][:, :]

    def _assign_decay_dist(self, child, cmat):
        """Copies the full ``decay_dists[child]`` matrix into ``cmat``.

        Args:
          child (MCEqParticle): decay product.
          cmat (np.ndarray): destination decay-matrix block.
        """
        cmat[:, :] = self.decay_dists[child][:, :]

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
            (:func:`numpy.array`, :func:`numpy.array`): :math:`x_{\rm Lab}`, :math:`dN/dx_{\rm Lab}`
        """

        eidx = (np.abs(self._energy_grid.c - kin_energy)).argmin()
        en = self._energy_grid.c[eidx]
        info(10, "Nearest energy, index: ", en, eidx, condition=verbose)

        m = self.hadr_yields[sec_pdg]
        xl_grid = (self._energy_grid.c[: eidx + 1]) / en

        xl_dist = en * xl_grid * m[: eidx + 1, eidx] / self._energy_grid.w[: eidx + 1]

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
            (:func:`numpy.array`, :func:`numpy.array`): :math:`x_{\rm Lab}`, :math:`dN/dx_{\rm Lab}`
        """

        eidx = (np.abs(self._energy_grid.c - kin_energy)).argmin()
        en = self._energy_grid.c[eidx]
        info(10, "Nearest energy, index: ", en, eidx, condition=verbose)

        m = self.decay_dists[sec_pdg]
        xl_grid = (self._energy_grid.c[: eidx + 1]) / en

        xl_dist = en * xl_grid * m[: eidx + 1, eidx] / self._energy_grid.w[: eidx + 1]

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
            (:func:`numpy.array`, :func:`numpy.array`): :math:`x_{\rm Lab}`, :math:`dN/dx_{\rm Lab}`
        """

        eidx = (np.abs(self._energy_grid.c - kin_energy)).argmin()
        en = self._energy_grid.c[eidx]
        info(10, "Nearest energy, index: ", en, eidx, condition=verbose)

        m = self.hadr_yields[sec_pdg]
        ekin_grid = self._energy_grid.c
        elab_dist = m[: eidx + 1, eidx] / self._energy_grid.w[eidx]
        return ekin_grid[: eidx + 1], elab_dist

    def _critical_energy(self):
        """Returns critical energy where decay and interaction
        are balanced.

        Approximate value in Air.
        """
        if self.is_stable or self.ctau <= 0.0:
            self.E_crit = np.inf
        else:
            self.E_crit = self.mass * 6.4e5 / self.ctau

    def _apply_force_resonance(self):
        """Flag this particle as a resonance if it appears in
        ``physics.filters["force_resonance"]`` and is not in
        ``physics.standard_particles``.

        A resonance particle is folded into other particles' interaction /
        decay matrices at build time (via ``MatrixBuilder._follow_chains``)
        and never gets its own state-vector slot. Default
        ``force_resonance = []`` means full propagation for everything.
        """
        pid = abs(self.pdg_id[0])
        self.is_resonance = (
            pid in self._physics.filters["force_resonance"]
            and pid not in self._physics.standard_particles
        )

    def __eq__(self, other):
        """Checks name for equality"""
        if isinstance(other, MCEqParticle):
            return self.name == other.name
        return NotImplemented

    def __neq__(self, other):
        """Checks name for equality"""
        if isinstance(other, MCEqParticle):
            return self.name != other.name
        return NotImplemented

    def __hash__(self):
        """Instruction for comuting the hash"""
        return hash(self.name)

    def __repr__(self):
        a_string = f"""
        {self.name}:
        is_hadron     : {self.is_hadron}
        is_lepton     : {self.is_lepton}
        is_nucleus    : {self.is_nucleus}
        is_stable     : {self.is_stable}
        is_resonance  : {self.is_resonance}
        is_tracking   : {self.is_tracking}
        is_projectile : {self.is_projectile}
        mceqidx       : {self.mceqidx}\n"""
        return a_string
