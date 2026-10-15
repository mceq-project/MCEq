"""Particle species of a cascade: :class:`MCEqParticle` and the :class:`ParticleManager` registry."""

from copy import copy
from math import copysign

import numpy as np
from particletools.tables import PYTHIAParticleData

from MCEq import bin_average, config
from MCEq.data import InteractionCrossSections
from MCEq.misc import average_A_target, getAZN, info, print_in_rows

_pdata = PYTHIAParticleData()

#: Atomic mass constant in grams; sigma / (A_target * m_u) is the inverse
#: interaction length in cm^2/g.
ATOMIC_MASS_UNIT_G = 1.0 / 6.02214076e23

#: MCEq names of the neutrinos, where they differ from the PYTHIA names.
backward_compatible_namestr = {
    "nu_mu": "numu",
    "nu_mubar": "antinumu",
    "nu_e": "nue",
    "nu_ebar": "antinue",
    "nu_tau": "nutau",
    "nu_taubar": "antinutau",
}

#: Species that exist only in the electromagnetic cascade (pi0 decays into
#: photons only); they are propagated only with ``config.enable_em``.
EM_PDG_IDS = (22, 11, -11, 111)


def _pname(pdg_id_or_name):
    """MCEq name of a particle: the PYTHIA name, with MCEq's neutrino names."""
    pythia_name = _pdata.name(pdg_id_or_name)
    return backward_compatible_namestr.get(pythia_name, pythia_name)


class MCEqParticle:
    """One particle species: properties, production and decay channels, rates.

    Args:
      pdg_id (int): PDG ID
      helicity (int): -1, 0 or 1 (0 means undefined or averaged)
      energy_grid: energy grid of the run (centers, edges, widths, bin count)
      cs_db: :class:`MCEq.data.InteractionCrossSections`, sets the cross section
      init_pdata_defaults (bool): read name, mass, lifetime and charge from the
        PYTHIA particle table
      A_target (float): mean target mass number; default from ``config``
    """

    def __init__(
        self,
        pdg_id,
        helicity,
        energy_grid=None,
        cs_db=None,
        init_pdata_defaults=True,
        A_target=None,
    ):
        #: ``(PDG ID, helicity)`` identifier
        self.pdg_id = (pdg_id, helicity)
        #: key in the manager's maps, distinct for tracking particles
        self.unique_pdg_id = self.pdg_id
        self.helicity = helicity
        self.A_target = average_A_target() if A_target is None else A_target
        self._energy_grid = energy_grid
        #: index in the state vector; -1 for resonances
        self.mceqidx = -1
        self.A, self.Z, self.N = getAZN(pdg_id)

        self.name = self.mass = self.ctau = self.charge = None
        self.is_nucleus = self.is_hadron = self.is_lepton = False
        self.is_stable = pdg_id in config.adv_set["disable_decays"]
        if init_pdata_defaults:
            self.is_nucleus = _pdata.is_nucleus(pdg_id)
            self.is_hadron = _pdata.is_hadron(pdg_id)
            self.is_lepton = _pdata.is_lepton(pdg_id)
            self.charge = self.Z if self.is_nucleus else int(_pdata.charge(pdg_id))
            #: (float) ctau in cm
            self.ctau = _pdata.ctau(pdg_id)
            #: (float) mass in GeV
            self.mass = _pdata.mass(pdg_id)
            self.name = _pname(pdg_id) + {-1: "_l", 1: "_r"}.get(helicity, "")
            self.is_stable = (
                not self.ctau < np.inf or pdg_id in config.adv_set["disable_decays"]
            )
        #: folded into the matrices of its parents, no state-vector slot
        self.is_resonance = (
            abs(pdg_id) in config.adv_set["force_resonance"]
            and abs(pdg_id) not in config.standard_particles
        )

        #: copy of another species that accumulates one origin of it
        self.is_tracking = False
        #: interacts (a parent of the interaction tables)
        self.is_projectile = False
        self.has_contloss = False
        #: continuous losses in GeV/(g/cm2)
        self.dEdX = None
        #: ``{secondary: yield matrix}`` of the interactions
        self.hadr_yields = {}
        #: ``{child: decay matrix}``
        self.decay_dists = {}
        #: interaction cross section in cm²
        self.cs = None
        self.current_cross_sections = None
        self.current_hadronic_model = None
        # stencil radii of bin-averaged interaction and decay tables
        self._int_stencil = 0
        self._dec_stencil = 0

        if energy_grid is not None and cs_db is not None:
            self.set_cs(cs_db)

    def set_cs(self, cs_db):
        """Take the cross section of this species from ``cs_db``."""
        info(11, "Obtaining cross sections for", self.pdg_id)
        self.current_cross_sections = cs_db.iam
        self.cs = cs_db[self.pdg_id[0]]

    @property
    def is_em(self):
        return abs(self.pdg_id[0]) == 11 or self.pdg_id[0] == 22

    @property
    def is_charged(self):
        return bool(self.charge)

    @property
    def can_interact(self):
        return self.cs is not None and bool(np.sum(self.cs) > 0)

    @property
    def hadr_secondaries(self):
        return list(self.hadr_yields)

    @property
    def children(self):
        return list(self.decay_dists)

    @property
    def E_crit(self):
        """Approximate critical energy in air, where decay and interaction balance."""
        if self.is_stable or self.ctau <= 0.0:
            return np.inf
        return self.mass * 6.4e5 / self.ctau

    def add_hadronic_production_channel(self, child, int_matrix):
        """Add ``child`` with the yield matrix ``int_matrix`` (dN/dE(i, j), not
        checked for energy conservation)."""
        if not self.is_projectile:
            raise Exception("The particle should be a projectile.")
        if child in self.hadr_yields:
            info(1, f"Child {child.name} has been already added.")
            return
        self.hadr_yields[child] = int_matrix

    def add_decay_channel(self, child, dec_matrix, force=False):
        """Add a decay into ``child``; branching ratios are not renormalised.
        ``force`` replaces an existing channel."""
        if self.is_stable:
            raise Exception("Cannot add decay channel to stable particle.")
        if child in self.decay_dists:
            if not force:
                info(1, f"Child {child.name} has been already added.")
                return
            info(1, f"Overwriting decay matrix of child {child.name}.")
        self.decay_dists[child] = dec_matrix

    @property
    def lidx(self):
        """Index of this particle's first energy bin in the state vector."""
        return self.mceqidx * self._energy_grid.d

    @property
    def uidx(self):
        """Index just past this particle's last energy bin in the state vector."""
        return (self.mceqidx + 1) * self._energy_grid.d

    def inverse_decay_length(self):
        r"""Inverse decay length :math:`m/(c\tau\,p)` in cm⁻¹ with the density
        factored out; zeros for stable particles, ``inf`` for ``ctau = 0``."""
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
        """Banded bin-averaged decay rate (:mod:`MCEq.bin_average`) in place of
        ``diag(inverse_decay_length())``, or None for bin-centre decay tables."""
        r = self._dec_stencil
        if not r or self.is_stable or not self.ctau > 0:
            return None

        def rate(e_kin):
            e_tot = e_kin + self.mass
            return self.mass / self.ctau / np.sqrt(e_tot**2 - self.mass**2)

        return bin_average.rate_operator(self._energy_grid.b, rate, r)

    def interaction_rate_operator(self):
        """Banded bin-averaged interaction rate in place of
        ``diag(inverse_interaction_length())``, or None for bin-centre yield tables.

        The cross section is interpolated linearly in log-log, or in log E where
        it vanishes somewhere on the grid.
        """
        r = self._int_stencil
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
        """Production cross section per energy bin, in mb or cm²."""
        if mbarn:
            return self.cs / InteractionCrossSections.mbarn2cm2
        return self.cs

    #: v1.4.2 name of :meth:`prod_cross_section`
    inel_cross_section = prod_cross_section

    def inverse_interaction_length(self):
        """Inverse interaction length in cm²/g on a target of mass number ``A_target``."""
        m_target = self.A_target * ATOMIC_MASS_UNIT_G  # <A> * m_u [g]
        return self.cs / m_target

    def parent_column(self, mat, eidx, kind="int"):
        """Secondaries of this particle at the centre of bin ``eidx``.

        Column ``eidx`` of the yield (``kind="int"``) or decay (``"dec"``) table
        ``mat``. A stencil-averaged table is applied to the delta deposition at
        the bin centre instead, except within the stencil radius of the grid edge.
        """
        r = self._dec_stencil if kind == "dec" else self._int_stencil
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

    def _column_at(self, kin_energy, mat, kind, verbose):
        """``(eidx, E, column)`` of ``mat`` at the bin nearest to ``kin_energy``."""
        eidx = (np.abs(self._energy_grid.c - kin_energy)).argmin()
        en = self._energy_grid.c[eidx]
        info(10, "Nearest energy, index: ", en, eidx, condition=verbose)
        return eidx, en, self.parent_column(mat, eidx, kind)

    def _dN_dxlab(self, kin_energy, mat, kind, verbose):
        eidx, en, col = self._column_at(kin_energy, mat, kind, verbose)
        xl_grid = self._energy_grid.c[: eidx + 1] / en
        return xl_grid, en * xl_grid * col[: eidx + 1] / self._energy_grid.w[: eidx + 1]

    def dN_dxlab(self, kin_energy, sec_pdg, verbose=True, **kwargs):
        r"""``(x_lab, dN/dx_lab)`` of secondary ``sec_pdg`` in interactions at the
        bin nearest to ``kin_energy``, including production modifications."""
        return self._dN_dxlab(kin_energy, self.hadr_yields[sec_pdg], "int", verbose)

    def dNdec_dxlab(self, kin_energy, sec_pdg, verbose=True, **kwargs):
        r"""``(x_lab, dN/dx_lab)`` of decay child ``sec_pdg`` at the bin nearest
        to ``kin_energy``."""
        return self._dN_dxlab(kin_energy, self.decay_dists[sec_pdg], "dec", verbose)

    def dN_dEkin(self, kin_energy, sec_pdg, verbose=True, **kwargs):
        r"""``(E_kin, dN/dE_kin)`` of secondary ``sec_pdg`` in interactions at the
        bin nearest to ``kin_energy``, including production modifications."""
        eidx, _, col = self._column_at(
            kin_energy, self.hadr_yields[sec_pdg], "int", verbose
        )
        return (
            self._energy_grid.c[: eidx + 1],
            col[: eidx + 1] / self._energy_grid.w[eidx],
        )

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


class ParticleManager:
    """Registry of the species of a run, their channels and state-vector indices.

    Args:
      pdg_id_list (list): ``(PDG ID, helicity)`` keys of the species; the EM
        species (:data:`EM_PDG_IDS`) are left out unless ``config.enable_em``
      energy_grid: energy grid of the run
      cs_db: :class:`MCEq.data.InteractionCrossSections`
      medium (str): interaction medium; default ``config.interaction_medium``
    """

    def __init__(self, pdg_id_list, energy_grid, cs_db, medium=None):
        self._energy_grid = energy_grid
        self._cs_db = cs_db
        self._medium = config.interaction_medium if medium is None else medium
        #: PDG ID -> state-vector index
        self.pdg2mceqidx = {}
        #: name -> state-vector index
        self.pname2mceqidx = {}
        #: PDG ID -> :class:`MCEqParticle`
        self.pdg2pref = {}
        #: name -> :class:`MCEqParticle`
        self.pname2pref = {}
        #: state-vector index -> :class:`MCEqParticle`
        self.mceqidx2pref = {}
        #: state-vector index -> PDG ID
        self.mceqidx2pdg = {}
        #: state-vector index -> name
        self.mceqidx2pname = {}
        self.current_hadronic_model = None
        #: applied tracking, ``(parent, child, alias, from_interactions)``
        self.tracking_relations = []
        # tracking requests, replayed after model changes
        self._tracking_requested = []

        self._init_categories(pdg_id_list)
        self.print_particle_tables(10)

    def set_interaction_model(
        self, cs_db, hadronic_db, updated_parent_list=None, force=False
    ):
        """Take cross sections and interaction channels from ``cs_db`` and
        ``hadronic_db``; ``updated_parent_list`` rebuilds the species first."""
        info(5, "Setting hadronic secondaries for particles.")
        if (
            self.current_hadronic_model == hadronic_db.iam
            and not force
            and updated_parent_list is None
        ):
            info(10, "Same hadronic model not applied to particles.")
            return
        if updated_parent_list is not None:
            self._init_categories(updated_parent_list)

        for p in self.cascade_particles:
            p.set_cs(cs_db)
            self._set_interactions(p, hadronic_db)
        for parent, _, alias, from_interactions in self.tracking_relations:
            if from_interactions:
                self._track(self.pdg2pref[parent], self.pname2pref[alias], True)

        self.current_hadronic_model = hadronic_db.iam
        self._update_particle_tables()

    def _set_interactions(self, p, hadronic_db):
        """Interaction channels of ``p``; a tracking particle produces only itself."""
        p.current_hadronic_model = hadronic_db.iam
        p._int_stencil = p._stencil_radius(hadronic_db.bin_average)
        p.hadr_yields = {}
        if p.is_tracking:
            species = self.pdg2pref[p.pdg_id]
            p.is_projectile = species.is_projectile
            p._int_stencil = species._int_stencil
            if species in species.hadr_yields:
                p.hadr_yields[p] = species.hadr_yields[species]
            return
        p.is_projectile = p.pdg_id in hadronic_db.parents
        if not p.is_projectile:
            return
        for key in hadronic_db.relations[p.pdg_id]:
            if key in self.pdg2pref:  # else a secondary this system does not propagate
                child = self.pdg2pref[key]
                p.hadr_yields[child] = hadronic_db.get_matrix(p.pdg_id, child.pdg_id)

    def set_decay_channels(self, decay_db):
        """Take the decay channels of the unstable species from ``decay_db`` and
        restore the tracking."""
        info(5, "Setting decay info for particles.")
        for p in self.all_particles:
            p._dec_stencil = p._stencil_radius(decay_db.bin_average)
            p.decay_dists = {}
            if p.is_stable or p.is_tracking:
                continue
            if p.pdg_id not in decay_db.parents:
                raise Exception(
                    "Unstable particle without decay distribution:", p.pdg_id, p.name
                )
            for key in decay_db.children(p.pdg_id):
                if key in self.pdg2pref:
                    child = self.pdg2pref[key]
                    p.decay_dists[child] = decay_db.get_matrix(p.pdg_id, child.pdg_id)

        self._restore_tracking_setup()
        self._update_particle_tables()

    def set_continuous_losses(self, contloss_db):
        """Set continuous losses terms to particles with ionization
        and radiation losses."""
        generic_table = contloss_db.generic_table
        for p in self.cascade_particles:
            if p.pdg_id in contloss_db:
                p.has_contloss = True
                p.dEdX = contloss_db[p.pdg_id]
            elif config.generic_losses_all_charged and p.is_charged and not p.is_em:
                # The stopping power of all charged particles is close to the
                # proton one at the same beta*gamma; clip to the tabulated range.
                betagamma_p = (
                    np.sqrt((self._energy_grid.c + p.mass) ** 2 - p.mass**2) / p.mass
                )
                ln_boost = np.clip(
                    np.log(betagamma_p),
                    np.log(generic_table[0][0]),
                    np.log(generic_table[0][-1]),
                )
                p.dEdX = -np.exp(contloss_db.generic_spl(ln_boost))
                p.has_contloss = True

    def _track(self, parent, tracking_particle, from_interactions):
        """Give ``tracking_particle`` the channel ``parent -> its species``, if it exists."""
        channels = parent.hadr_yields if from_interactions else parent.decay_dists
        species = self.pdg2pref[tracking_particle.pdg_id]
        if species not in channels:
            info(17, f"{parent.name} has no channel to {tracking_particle.name}")
            return False
        channels[tracking_particle] = channels[species]
        return True

    def add_tracking_particle(
        self,
        parent_list,
        child_pdg,
        alias_name,
        from_interactions=False,
        include_antiparticle=False,
    ):
        """Track a parent-to-child decay or production channel under an alias.

        Parent and child keys are ``(PDG ID, helicity)`` tuples::

            $ particleManagerInstance.add_tracking_particle([(211, 0)], (14, 0), 'pi_numu')

        This stores muon neutrinos from pion decays under the alias 'pi_numu'.
        Multiple parents are allowed::

            $ particleManagerInstance.add_tracking_particle(
                [(411, 0), (421, 0), (431, 0)], (14, 0), 'D_numu')

        Args:

            parent_list (list): list of ``(PDG ID, helicity)`` parent keys
            child_pdg (tuple): ``(PDG ID, helicity)`` child key
            alias_name (str): name under which the result is accessible in
                get_solution
            from_interactions (bool): track particles from interactions
            include_antiparticle (bool): also track the channels of the
                antiparticles of ``parent_list``
        """
        info(10, "requested for", parent_list, child_pdg, alias_name)

        for p in parent_list:
            request = (
                p,
                child_pdg,
                alias_name,
                from_interactions,
                include_antiparticle,
            )
            if request not in self._tracking_requested:
                self._tracking_requested.append(request)

        if alias_name in self.pname2pref:
            info(15, "Re-using tracking particle", alias_name)
            tracking_particle = self.pname2pref[alias_name]
        elif child_pdg not in self.pdg2pref:
            info(15, "Tracking child not available for this interaction model.")
            return
        else:
            species = self.pdg2pref[child_pdg]
            tracking_particle = copy(species)
            tracking_particle.is_tracking = True
            tracking_particle.name = alias_name
            # Transported like its species, but produces nothing but itself.
            tracking_particle.hadr_yields = {}
            tracking_particle.decay_dists = {}
            if species in species.hadr_yields:
                tracking_particle.hadr_yields[tracking_particle] = species.hadr_yields[
                    species
                ]
            step = int(copysign(10000, child_pdg[0]))
            uid = child_pdg[0] + 100 * step
            while (uid, tracking_particle.helicity) in self.pdg2pref:
                uid += step
            tracking_particle.unique_pdg_id = (uid, tracking_particle.helicity)
            info(
                10,
                f"Creating new tracking particle {alias_name} with unique ID "
                f"{tracking_particle.unique_pdg_id}",
            )

        track_success = False
        if include_antiparticle:
            par_list = list(set(parent_list + [(-p, h) for (p, h) in parent_list]))
        else:
            par_list = list(set(parent_list))

        for parent_pdg in par_list:
            if parent_pdg not in self.pdg2pref:
                info(15, f"Parent particle {parent_pdg} does not exist.")
                continue
            relation = (parent_pdg, child_pdg, alias_name, from_interactions)
            if relation in self.tracking_relations:
                info(20, f"Tracking of {alias_name} from {parent_pdg} already active.")
                continue
            if self._track(
                self.pdg2pref[parent_pdg], tracking_particle, from_interactions
            ):
                info(15, f"Parent particle {parent_pdg} tracking scheduled.")
                self.tracking_relations.append(relation)
                track_success = True

        if track_success and tracking_particle.name not in self.pname2pref:
            tracking_particle.mceqidx = max(self.mceqidx2pref) + 1
            self.all_particles.append(tracking_particle)
            self.cascade_particles.append(tracking_particle)
            self._update_particle_tables()
            info(10, f"tracking particle {alias_name} successfully added.")

    def track_leptons_from(
        self,
        parent_pdg_list,
        prefix,
        exclude_em=True,
        from_interactions=False,
        use_helicities=False,
        include_antiparticle=False,
    ):
        """Adds tracking particles for all leptons coming from decays of parents
        in `parent_pdg_list`.

        Args:

            parent_pdg_list (list): PDG IDs of parent particles
                                    that leptons originate from
            prefix (list): prefix for each lepton name that will be
                           identified as coming from these parent particles
            exclude_em (bool, optional): exclude electromagnetic (e+-, gamma)
                from leptons
            from_interactions (bool, optional): track particles coming from
                interactions of parents instead of decays
            use_helicities (bool, optional): include nonzero-helicity leptons
                when true; otherwise select helicity zero only
            include_antiparticle (bool, optional): ex. [211, 321] will be
                automatically converted to [211,-211,321,-321]
        """
        leptons = [
            p
            for p in self.all_particles
            if p.is_lepton and not (exclude_em and p.is_em) and not p.is_tracking
        ]

        for lepton in leptons:
            if not use_helicities and lepton.pdg_id[1] != 0:
                continue
            self.add_tracking_particle(
                parent_pdg_list,
                lepton.pdg_id,
                prefix + lepton.name,
                from_interactions,
                include_antiparticle,
            )

    def _init_categories(self, particle_pdg_list):
        """Build the species; the cascade particles are ordered by |PDG ID|."""
        info(5, "Generating particle list.")
        particles = sorted(set(particle_pdg_list))
        if not config.enable_em:
            # Without EM transport photons and electrons have nowhere to go.
            particles = [p for p in particles if p[0] not in EM_PDG_IDS]
        #: ``(PDG ID, helicity)`` keys of the species
        self.pdg_id_list = particles

        particle_list = [
            MCEqParticle(
                pdg,
                hel,
                self._energy_grid,
                self._cs_db,
                A_target=average_A_target(self._medium),
            )
            for pdg, hel in particles
        ]
        self.cascade_particles = sorted(
            (p for p in particle_list if not p.is_resonance),
            key=lambda p: abs(p.pdg_id[0]),
        )
        self.resonances = [p for p in particle_list if p.is_resonance]
        for mceqidx, p in enumerate(self.cascade_particles):
            p.mceqidx = mceqidx

        self.all_particles = self.cascade_particles + self.resonances
        self.tracking_relations = []
        self._update_particle_tables()

    def add_new_particle(self, new_mceq_particle):
        """Add a species; a resonance gets no state-vector slot."""
        p = new_mceq_particle
        if p.name in self.pname2pref:
            info(0, f"Particle {p.name}/{p.pdg_id} has already been added. Use it.")
            return
        if p.is_resonance:
            info(2, f"New particle {p.name}/{p.pdg_id} is a resonance.")
            self.resonances.append(p)
        else:
            info(2, f"New particle {p.name}/{p.pdg_id} is not a resonance.")
            p.mceqidx = len(self.cascade_particles)
            self.cascade_particles.append(p)

        self.all_particles = self.cascade_particles + self.resonances
        self._update_particle_tables()

    def _update_particle_tables(self):
        """Rebuild the index maps after the species list changed."""
        self.n_cparticles = len(self.cascade_particles)
        self.dim = self._energy_grid.d
        self.dim_states = self._energy_grid.d * self.n_cparticles

        for d in [
            self.pdg2mceqidx,
            self.pname2mceqidx,
            self.mceqidx2pdg,
            self.mceqidx2pname,
            self.mceqidx2pref,
            self.pdg2pref,
            self.pname2pref,
        ]:
            d.clear()

        for p in self.all_particles:
            self.pdg2mceqidx[p.unique_pdg_id] = p.mceqidx
            self.pname2mceqidx[p.name] = p.mceqidx
            self.pdg2pref[p.unique_pdg_id] = p
            self.pname2pref[p.name] = p
        # resonances have no state-vector index (mceqidx = -1)
        for p in self.cascade_particles:
            self.mceqidx2pdg[p.mceqidx] = p.unique_pdg_id
            self.mceqidx2pname[p.mceqidx] = p.name
            self.mceqidx2pref[p.mceqidx] = p

    def _restore_tracking_setup(self):
        """Replay the tracking requests, or set up the default tracking on the
        first call."""
        info(10, "Restoring tracking particle setup")
        if not self._tracking_requested:
            if config.enable_default_tracking:
                self._init_default_tracking()
            return

        self.tracking_relations = []
        for pid, cid, alias, int_dec, incl_anti in self._tracking_requested:
            if pid not in self.pdg2pref:
                info(
                    15,
                    f"Can not restore {alias}: parent {pid} is not in the particle list.",
                )
                continue
            self.add_tracking_particle([pid], cid, alias, int_dec, incl_anti)

    def _init_default_tracking(self):
        """Add default tracking particles for leptons from pi, K, and mu"""
        info(1, "Initializing default tracking categories (pi, K, mu)")
        for parents, prefix, with_helicity in [
            ([(211, 0)], "pi_", True),
            ([(321, 0)], "k_", True),
            ([(13, -1), (13, 1)], "mulr_", False),
            ([(13, 0)], "mu_h0_", False),
            ([(13, -1), (13, 0), (13, 1)], "mu_", False),
            ([(310, 0), (130, 0)], "K0_", False),
        ]:
            self.track_leptons_from(
                parents,
                prefix,
                exclude_em=True,
                use_helicities=with_helicity,
                include_antiparticle=True,
            )

        # Prompt leptons from short-lived decays and from interaction vertices
        prompt_ctau = config.prompt_ctau
        self.track_leptons_from(
            [p.pdg_id for p in self.all_particles if p.ctau < prompt_ctau],
            "prcas_",
            exclude_em=True,
            use_helicities=False,
        )
        self.track_leptons_from(
            [p.pdg_id for p in self.all_particles if p.is_projectile],
            "prres_",
            exclude_em=True,
            from_interactions=True,
            use_helicities=False,
        )
        self.track_leptons_from(
            [p.pdg_id for p in self.all_particles if p.is_em],
            "em_",
            exclude_em=True,
            from_interactions=True,
            use_helicities=False,
        )

    def __contains__(self, pdg_id_or_name):
        """True if :meth:`__getitem__` finds the particle."""
        try:
            self[pdg_id_or_name]
        except KeyError:
            return False
        return True

    def __getitem__(self, pdg_id_or_name):
        """The particle with key ``(PDG ID, helicity)``, PDG ID or name."""
        if isinstance(pdg_id_or_name, tuple):
            return self.pdg2pref[pdg_id_or_name]
        if isinstance(pdg_id_or_name, int):
            return self.pdg2pref[(pdg_id_or_name, 0)]
        try:
            return self.pdg2pref[(_pdata.pdg_id(pdg_id_or_name), 0)]
        except KeyError:
            return self.pname2pref[pdg_id_or_name]

    def __repr__(self):
        str_out = ""
        ident = 3 * " "
        for s in self.all_particles:
            str_out += s.name + "\n" + ident
            str_out += "PDG id : " + str(s.pdg_id) + "\n" + ident
            str_out += "MCEq idx : " + str(s.mceqidx) + "\n\n"
        return str_out

    def print_particle_tables(self, min_dbg_lev=2):
        info(min_dbg_lev, "Hadrons and stable particles:", no_caller=True)
        print_in_rows(
            min_dbg_lev,
            [p.name for p in self.all_particles if p.is_hadron and not p.is_resonance],
        )

        info(min_dbg_lev, "\nResonances:", no_caller=True)
        print_in_rows(
            min_dbg_lev, [p.name for p in self.all_particles if p.is_resonance]
        )

        info(min_dbg_lev, "\nLeptons:", no_caller=True)
        print_in_rows(
            min_dbg_lev,
            [p.name for p in self.all_particles if p.is_lepton and not p.is_tracking],
        )
        info(min_dbg_lev, "\nTracking:", no_caller=True)
        print_in_rows(
            min_dbg_lev, [p.name for p in self.all_particles if p.is_tracking]
        )

        info(
            min_dbg_lev, "\nTotal number of species:", self.n_cparticles, no_caller=True
        )
