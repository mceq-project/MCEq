"""MCEqParticle: properties, decay and interaction channels, and rates of a species."""

import numpy as np

from MCEq import config
from MCEq.data.cross_sections import InteractionCrossSections
from MCEq.misc import average_A_target, getAZN, info
from MCEq.species import bin_average
from MCEq.species.constants import ATOMIC_MASS_UNIT_G, _pdata, _pname


class MCEqParticle:
    """Properties, channels and rates of one particle species in
    :class:`MCEq.core.MCEqRun`.

    Args:
      pdg_id (int): PDG ID of the particle
      helicity (int): helicity -1, 0, or 1 (0 means undefined or average)
      energy_grid: energy-grid object (centers, edges, widths, bin count)
      cs_db: cross-section provider, i.e.
             :class:`MCEq.data.InteractionCrossSections`
      init_pdata_defaults (bool): fill name, mass, ctau and charge from the
             PYTHIA particle table
      A_target (float): average target mass number; default from ``config``
    """

    def __init__(
        self,
        pdg_id,
        helicity,
        energy_grid=None,
        cs_db=None,
        init_pdata_defaults=True,
        A_target=None,
    ):  # anfe: the init looks really boilerplate. Isn't there a better way to design this class?
        if A_target is None:
            A_target = average_A_target()
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
        #: (bool) folded into other particles' matrices, no state-vector slot
        self.is_resonance = False
        #: (bool) particle is interacting projectile
        self.is_projectile = False
        #: (bool) particle is stable
        self.is_stable = pdg_id in config.adv_set["disable_decays"]
        #: (bool) can_interact
        self.can_interact = False
        #: (bool) has continuous losses dE/dX defined
        self.has_contloss = False
        #: (np.array) continuous losses in GeV/(g/cm2)
        self.dEdX = None
        #: (bool) is a tracking particle
        self.is_tracking = False
        #: `(PDG ID, helicity)` particle identifier
        self.pdg_id = (pdg_id, helicity)
        #: `(PDG ID, helicity)` identifier, distinct for tracking particles
        self.unique_pdg_id = (pdg_id, helicity)
        #: (int) particle index in the state layout
        self.mceqidx = -1

        #: (int) interaction threshold (idx where cs>0)
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
            #: interaction cross section in cm²
            self.set_cs(cs_db)

    def _init_defaults_from_pythia_database(
        self,
    ):  # anfe: also boiler plate. Looks to me like the class' design is missing something (as commented in __init__ )
        """Init some particle properties from :mod:`particletools.tables`."""
        #: (bool) particle is a nucleus (not yet implemented)
        self.is_nucleus = _pdata.is_nucleus(self.pdg_id[0])
        #: (bool) particle is a hadron
        self.is_hadron = _pdata.is_hadron(self.pdg_id[0])
        #: (bool) particle is a lepton
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
        self.is_stable = (
            not self.ctau < np.inf or self.pdg_id[0] in config.adv_set["disable_decays"]
        )

    def init_custom_particle_data(
        self, name, pdg_id, helicity, ctau, mass, **kwargs
    ):  # anfe: same here also looks quite boilerplate
        """Add custom particle type. (Incomplete and not debugged)"""
        #: `(PDG ID, helicity)` particle identifier
        self.pdg_id = (pdg_id, helicity)
        #: (bool) if it's an electromagnetic particle
        self.is_em = kwargs.pop("is_em", abs(pdg_id) == 11 or pdg_id == 22)
        #: (bool) particle is a nucleus (not yet implemented)
        self.is_nucleus = kwargs.pop("is_nucleus", _pdata.is_nucleus(self.pdg_id[0]))
        #: (bool) particle is a hadron
        self.is_hadron = kwargs.pop("is_hadron", _pdata.is_hadron(self.pdg_id[0]))
        #: (bool) particle is a lepton
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

    def set_cs(self, cs_db):  # anfe: shouldn't be a public function
        """Set cross section and recalculate the dependent variables"""
        info(11, "Obtaining cross sections for", self.pdg_id)
        self.current_cross_sections = cs_db.iam
        self.cs = cs_db[self.pdg_id[0]]
        if sum(self.cs) > 0:
            self.can_interact = True
        else:
            self.can_interact = False

        self._critical_energy()
        self._apply_force_resonance()

    def _set_channels(
        self, channel_table, pmanager, child_keys, keep_tracking=False
    ):  # anfe: why do we need this function exactly? It doesn't do anything special. and it is simply used below? No need.
        """Child references and their matrices from ``channel_table``; children
        the manager does not carry are skipped."""
        refs = []
        for key in child_keys:
            try:
                refs.append(pmanager[key])
            except KeyError:
                continue  # a child this system does not propagate
        if keep_tracking:
            refs += [ref for ref in self.hadr_secondaries if ref.is_tracking]
        yields = {}
        for ref in refs:
            yields[ref] = channel_table.get_matrix(self.pdg_id, ref.pdg_id)
        return refs, yields

    def set_hadronic_channels(self, hadronic_db, pmanager):
        """Changes the hadronic interaction model.

        Replaces indexing of the yield dictionary from PDG IDs
        with references from particle manager.
        """

        self.current_hadronic_model = hadronic_db.iam
        self._int_stencil = self._stencil_radius(
            getattr(hadronic_db, "bin_average", None)
        )
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
            self._int_stencil = orig_part._int_stencil
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

        self._dec_stencil = self._stencil_radius(getattr(decay_db, "bin_average", None))
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

    def track_decays(
        self, tracking_particle
    ):  # anfe: can we unify the tracking decays or interactions, the codes are almost the same.
        children_d = {}
        for c in self.children:
            children_d[c.pdg_id] = c
        if tracking_particle.pdg_id not in children_d:
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

    def track_interactions(self, tracking_particle):  # anfe: see above
        secondaries_d = {}
        for s in self.hadr_secondaries:
            secondaries_d[s.pdg_id] = s
        if tracking_particle.pdg_id not in secondaries_d:
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

    def is_secondary(
        self, particle_ref
    ):  # anfe: is_secondary and is_child are used only in a very few places. We should rather overload the operator to check for "sec in proj.decay_children" and "proj.prod_children" or so
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
        """Lower offset of this particle's energy bins in a state vector.

        Returns:
          (int): index of this particle's first energy bin
        """
        return self.mceqidx * self._energy_grid.d

    @property
    def uidx(self):
        """Exclusive upper offset of this particle's energy bins in a state vector.

        Returns:
          (int): index just past this particle's last energy bin
        """
        return (self.mceqidx + 1) * self._energy_grid.d

    def inverse_decay_length(self):
        r"""Returns inverse decay length, where the air density
        :math:`\rho` is factorized out.

        :math:`1/\lambda_{dec} = m/(c\tau\,p)`, with :math:`p` the lab
        momentum. A stable particle (``ctau = inf``) returns zeros;
        ``inf`` is what the ``ctau == 0`` path returns.

        Returns:
          (numpy.array): :math:`\frac{\rho}{\lambda_{dec}}` in cm⁻¹,
          shape ``(d,)`` on every path
        """
        try:
            e_tot = self._energy_grid.c + self.mass
            p_lab = np.sqrt(e_tot**2 - self.mass**2)
            return self.mass / self.ctau / p_lab
        except ZeroDivisionError:
            return np.ones_like(self._energy_grid.c) * np.inf

    def _stencil_radius(self, marks):
        """Stencil radius if ``marks`` (``(radius, {pdg})``) cover this particle, else 0."""
        if not marks or self.pdg_id[0] not in marks[1]:
            return 0
        return int(marks[0])

    def decay_rate_operator(self):
        """Banded bin-averaged decay rate (see :mod:`MCEq.species.bin_average`),
        or None when the decay tables of this particle are not stencil-averaged.

        Used in place of ``diag(inverse_decay_length())``; same units.
        """
        r = getattr(self, "_dec_stencil", 0)
        if not r or self.is_stable or not self.ctau > 0:
            return None

        def rate(e_kin):
            e_tot = e_kin + self.mass
            return self.mass / self.ctau / np.sqrt(e_tot**2 - self.mass**2)

        return bin_average.rate_operator(self._energy_grid.b, rate, r)

    def interaction_rate_operator(self):
        """Banded bin-averaged interaction rate, or None when the yield tables of
        this particle are not stencil-averaged.

        Used in place of ``diag(inverse_interaction_length())``; the cross section is
        interpolated between the bin centres linearly in log-log (linearly in
        log E where it vanishes somewhere on the grid).
        """
        r = getattr(self, "_int_stencil", 0)
        if not r:
            return None
        lam = np.asarray(self.inverse_interaction_length(), dtype=np.float64)
        log_c = np.log(np.asarray(self._energy_grid.c, dtype=np.float64))
        if np.all(lam > 0):
            log_l = np.log(lam)

            def rate(e_kin):
                return np.exp(np.interp(np.log(e_kin), log_c, log_l))

        else:

            def rate(e_kin):
                return np.interp(np.log(e_kin), log_c, lam)

        return bin_average.rate_operator(self._energy_grid.b, rate, r)

    def prod_cross_section(self, mbarn=False):
        """Returns production cross section.

        Args:
          mbarn (bool) : if True cross section in mb otherwise in cm**2
        Returns:
          (numpy.array): :math:`\\sigma_{\\rm prod}`, one value per energy bin,
          in mb or cm**2
        """
        if mbarn:
            return self.cs / InteractionCrossSections.mbarn2cm2

        return self.cs

    def inverse_interaction_length(self):
        """Returns inverse interaction length for this particle's ``A_target``.

        Returns:
          (numpy.array): :math:`\\frac{1}{\\lambda_{int}}`, one value per
          energy bin, in cm**2/g
        """

        m_target = self.A_target * ATOMIC_MASS_UNIT_G  # <A> * m_u [g]
        return self.cs / m_target

    def _assign_hadr_dist(
        self, child, cmat
    ):  # anfe: do we actually need this function and the one below?
        """Copy ``hadr_yields[child]`` into the matrix block ``cmat``."""
        cmat[:, :] = self.hadr_yields[child][:, :]

    def _assign_decay_dist(self, child, cmat):
        """Copy ``decay_dists[child]`` into the matrix block ``cmat``."""
        cmat[:, :] = self.decay_dists[child][:, :]

    def parent_column(self, mat, eidx, kind="int"):
        """Secondaries of this particle at the centre of bin ``eidx``.

        ``kind`` is ``"int"`` for a yield table, ``"dec"`` for a decay table.

        Column ``eidx`` of ``mat`` (a yield or decay table on the run grid) for
        bin-centre tables. Stencil-averaged tables (:mod:`MCEq.species.bin_average`)
        read neighbouring cell contents as a smooth density, so the table is
        applied to the delta deposition at the bin centre, which reproduces the
        spectrum of one parent energy to the stencil order. Bins within the
        stencil radius of the grid edge return the column.
        """
        r = getattr(self, "_dec_stencil" if kind == "dec" else "_int_stencil", 0)
        mat = np.asarray(mat)
        if not r:
            return mat[:, eidx]
        grid = self._energy_grid
        try:
            first, contents = bin_average.delta_deposition(grid.b, grid.c[eidx], r)
        except ValueError:
            return mat[:, eidx]
        sl = slice(first, first + len(contents))
        # same parent count as the column (phi_eidx = 1)
        return mat[:, sl] @ (contents * grid.w[eidx] / grid.w[sl])

    def dN_dxlab(self, kin_energy, sec_pdg, verbose=True, **kwargs):
        r"""Returns :math:`dN/dx_{\rm Lab}` for interaction energy close
        to ``kin_energy`` for hadron-air collisions on this projectile.

        Includes modifications applied via :meth:`MCEq.core.MCEqRun.set_mod_pprod`.

        Args:
            kin_energy (float): approximate interaction kinetic energy
            sec_pdg: `(PDG ID, helicity)` key of the secondary particle
            verbose (bool): print out the closest energy
        Returns:
            (:func:`numpy.array`, :func:`numpy.array`): :math:`x_{\rm Lab}`, :math:`dN/dx_{\rm Lab}`
        """

        eidx = (np.abs(self._energy_grid.c - kin_energy)).argmin()
        en = self._energy_grid.c[eidx]
        info(10, "Nearest energy, index: ", en, eidx, condition=verbose)

        col = self.parent_column(self.hadr_yields[sec_pdg], eidx)
        xl_grid = (self._energy_grid.c[: eidx + 1]) / en

        xl_dist = en * xl_grid * col[: eidx + 1] / self._energy_grid.w[: eidx + 1]

        return xl_grid, xl_dist

    def dNdec_dxlab(self, kin_energy, sec_pdg, verbose=True, **kwargs):
        r"""Returns the decay spectrum :math:`dN/dx_{\rm Lab}` of this parent
        at the kinetic-energy bin nearest to ``kin_energy``.

        Args:
            kin_energy (float): approximate parent kinetic energy
            sec_pdg: `(PDG ID, helicity)` key of the decay child
            verbose (bool): print out the closest energy
        Returns:
            (:func:`numpy.array`, :func:`numpy.array`): :math:`x_{\rm Lab}`, :math:`dN/dx_{\rm Lab}`
        """

        eidx = (np.abs(self._energy_grid.c - kin_energy)).argmin()
        en = self._energy_grid.c[eidx]
        info(10, "Nearest energy, index: ", en, eidx, condition=verbose)

        col = self.parent_column(self.decay_dists[sec_pdg], eidx, kind="dec")
        xl_grid = (self._energy_grid.c[: eidx + 1]) / en

        xl_dist = en * xl_grid * col[: eidx + 1] / self._energy_grid.w[: eidx + 1]

        return xl_grid, xl_dist

    def dN_dEkin(self, kin_energy, sec_pdg, verbose=True, **kwargs):
        r"""Returns :math:`dN/dE_{\rm Kin}` in lab frame for an interaction
        kinetic energy close to ``kin_energy``.

        Includes modifications applied via :meth:`MCEq.core.MCEqRun.set_mod_pprod`.

        Args:
            kin_energy (float): approximate interaction kinetic energy
            sec_pdg: `(PDG ID, helicity)` key of the secondary particle
            verbose (bool): print out the closest energy
        Returns:
            (:func:`numpy.array`, :func:`numpy.array`): the secondary
            kinetic-energy grid and the yield column divided by the width of
            the parent bin
        """

        eidx = (np.abs(self._energy_grid.c - kin_energy)).argmin()
        en = self._energy_grid.c[eidx]
        info(10, "Nearest energy, index: ", en, eidx, condition=verbose)

        col = self.parent_column(self.hadr_yields[sec_pdg], eidx)
        ekin_grid = self._energy_grid.c
        elab_dist = col[: eidx + 1] / self._energy_grid.w[eidx]
        return ekin_grid[: eidx + 1], elab_dist

    # anfe: For the dN_d.... functions above, it should be sufficient to just use one provate function with arguments specifying the variables, and three short functions just call this private function with correct argument to avoid copy/paste in multiple places.

    def _critical_energy(self):
        """Set the approximate atmospheric critical energy ``E_crit``.

        The critical energy is where decay and interaction balance. Stable
        particles and nonpositive lifetimes receive infinity.
        """
        if self.is_stable or self.ctau <= 0.0:
            self.E_crit = np.inf
        else:
            self.E_crit = self.mass * 6.4e5 / self.ctau

    def _apply_force_resonance(
        self,
    ):  # anfe: why do we need a function for that? Isn't it just somewhere in the init clear that it is a resonance?
        """Flag as resonance if in ``config.adv_set["force_resonance"]`` and
        not in ``config.standard_particles``."""
        pid = abs(self.pdg_id[0])
        self.is_resonance = (
            pid in config.adv_set["force_resonance"]
            and pid not in config.standard_particles
        )

    def __eq__(
        self, other
    ):  # anfe: danger. We shouldn't use that. same for __neq__ and __hash__ only by name is not good. avoid in downstream.
        """Checks name for equality"""
        if isinstance(other, MCEqParticle):
            return self.name == other.name
        return NotImplemented

    def __hash__(self):
        """Hash of the name."""
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
