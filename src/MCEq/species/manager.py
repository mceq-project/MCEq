"""``ParticleManager``: the registry of tracked particles and index tables.

Moved verbatim from the flat ``MCEq.particlemanager`` (§13.2 item 6). The
``if False:`` index dump at the end of ``print_particle_tables`` is deleted
(plan section 5: the dead tables go with the move; it has been dead since the
matrix indices stopped being printed). The deferred
``from MCEq.particlemanager import MCEqParticle`` inside ``_init_categories``
now names the new home directly.
"""

from math import copysign

import numpy as np

from MCEq.misc import average_A_target, info, print_in_rows
from MCEq.species.constants import _pdata
from MCEq.species.particle import MCEqParticle


class ParticleManager:
    """Database for objects of :class:`MCEqParticle`.

    ``physics`` is the settings group (``config.physics``) that this manager and
    every particle it builds read their filters and switches from.

    Authors:
        Anatoli Fedynitch (DESY)
        Jonas Heinze (DESY)
    """

    def __init__(
        self,
        pdg_id_list,
        energy_grid,
        cs_db,
        medium=None,
        *,
        physics=None,
    ):
        from MCEq import config

        physics = config.physics if physics is None else physics
        # Resolved here rather than as a signature default: a default binds when
        # this module is first imported, which happens lazily from MCEqRun, so
        # whether a caller's `config.interaction_medium` was seen depended on
        # whether it was written before that import.
        medium = physics.interaction_medium if medium is None else medium
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
        # Medium
        self._medium = medium
        # Settings group, handed on to every MCEqParticle built here
        self._physics = physics
        # Dictionary to save te tracking particle config
        self.tracking_relations = []
        # Save the tracking relations requested by default tracking
        self._tracking_requested = []

        self._init_categories(particle_pdg_list=pdg_id_list)

        self.print_particle_tables(10)

    def set_cross_sections_db(self, cs_db):
        """Sets the production cross section to each interacting particle.

        This applies to most of the hadrons and does not imply that the
        particle becomes a projectile. parents need in addition defined
        hadronic channels.
        """

        info(5, "Setting cross section particle variables.")
        if self.current_cross_sections == cs_db.iam:
            info(10, "Same cross section model not applied to particles.")
            return

        for p in self.cascade_particles:
            p.set_cs(cs_db)
        self.current_cross_sections = cs_db.iam
        self._update_particle_tables()

    def set_decay_channels(self, decay_db):
        """Attaches the references to the decay yield tables to
        each unstable particle"""

        info(5, "Setting decay info for particles.")
        for p in self.all_particles:
            p.set_decay_channels(decay_db, self)

        self._restore_tracking_setup()
        self._update_particle_tables()

    def set_interaction_model(
        self, cs_db, hadronic_db, updated_parent_list=None, force=False
    ):
        """Attaches the references to the hadronic yield tables to
        each projectile particle"""

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
            elif (
                self._physics.generic_losses_all_charged
                and p.is_charged
                and not p.is_em
            ):
                # Stopping power dEdX is almost the same for all charged particles.
                # What changes is gamm*beta. We interpolate the dEdX tables
                # stored for protons to different energy grids.
                # Compute beta*gamma from kinetic energy
                betagamma_p = (
                    np.sqrt((self._energy_grid.c + p.mass) ** 2 - p.mass**2) / p.mass
                )
                p.dEdX = -np.exp(contloss_db.generic_spl(np.log(betagamma_p)))
                p.has_contloss = True

    def add_tracking_particle(
        self,
        parent_list,
        child_pdg,
        alias_name,
        from_interactions=False,
        include_antiparticle=False,
    ):
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

        info(10, "requested for", parent_list, child_pdg, alias_name)

        for p in parent_list:
            if (
                p,
                child_pdg,
                alias_name,
                from_interactions,
                include_antiparticle,
            ) in self._tracking_requested:
                continue
            self._tracking_requested.append(
                (p, child_pdg, alias_name, from_interactions, include_antiparticle)
            )

        # Check if tracking particle with the alias not yet defined
        # and create new one of necessary
        if alias_name in self.pname2pref:
            info(15, "Re-using tracking particle", alias_name)
            tracking_particle = self.pname2pref[alias_name]
        elif child_pdg not in self.pdg2pref:
            info(
                15,
                "Tracking child not a available",
                "for this interaction model, skipping.",
            )
            return
        else:
            info(10, "Creating new tracking particle")
            # Copy all preferences of the original particle
            tracking_particle = copy(self.pdg2pref[child_pdg])
            tracking_particle.is_tracking = True
            tracking_particle.name = alias_name
            # Find a unique PDG ID for the new tracking particle
            # print child_pdg[0], int(copysign(1000000, child_pdg[0]))
            unique_child_pdg = (
                child_pdg[0] + int(copysign(1000000, child_pdg[0])),
                tracking_particle.helicity,
            )

            for i in range(100):
                if unique_child_pdg not in list(self.pdg2pref):
                    break
                _tn = tracking_particle.name
                _u = unique_child_pdg
                info(
                    20,
                    f"{i}: trying to find unique_pdg ({_tn}) for {_u}",
                )
                unique_child_pdg = (
                    unique_child_pdg[0] + int(copysign(10000, child_pdg[0])),
                    tracking_particle.helicity,
                )
            tracking_particle.unique_pdg_id = unique_child_pdg
            info(
                10,
                "Creating new tracking particle {0} with unique ID {1}".format(
                    tracking_particle.name, unique_child_pdg
                ),
            )

        # Track if attempt to add the tracking particle succeeded at least once
        track_success = False
        # Include antiparticle if requested
        if include_antiparticle:
            par_list = list(set(parent_list + [(-p, h) for (p, h) in parent_list]))
        else:
            par_list = list(set(parent_list))

        for parent_pdg in par_list:
            if parent_pdg not in self.pdg2pref:
                info(15, f"Parent particle {parent_pdg} does not exist.")
                continue
            if (
                parent_pdg,
                child_pdg,
                alias_name,
                from_interactions,
            ) in self.tracking_relations:
                _tn = tracking_particle.name
                _n = self.pdg2pref[parent_pdg].name
                info(
                    20,
                    f"Tracking of {_tn} from {_n} already activated.",
                )
                continue

            if not from_interactions:
                info(15, "Child {0} tracks decays".format(parent_pdg))
                track_method = self.pdg2pref[parent_pdg].track_decays
            else:
                info(15, "Child {0} tracks interactions".format(parent_pdg))
                track_method = self.pdg2pref[parent_pdg].track_interactions

            # Check if the tracking is successful. If not the particle is not
            # a child of the parent particle
            if track_method(tracking_particle):
                info(15, f"Parent particle {parent_pdg} tracking scheduled.")
                self.tracking_relations.append(
                    (parent_pdg, child_pdg, alias_name, from_interactions)
                )
                track_success = True
        if track_success and tracking_particle.name not in list(self.pname2pref):
            tracking_particle.mceqidx = np.max(list(self.mceqidx2pref)) + 1
            self.all_particles.append(tracking_particle)
            self.cascade_particles.append(tracking_particle)
            self._update_particle_tables()
            info(
                10,
                f"tracking particle {tracking_particle.name} successfully added.",
            )

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
            use_helicities (bool, optional): ignore leptons with
                non-zero/defined helicity
            include_antiparticle (bool, optional): ex. [211, 321] will be
                automatically converted to [211,-211,321,-321]
        """

        leptons = [
            p
            for p in self.all_particles
            if p.is_lepton and not (p.is_em == exclude_em) and not p.is_tracking
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
            MCEqParticle(
                pdg,
                hel,
                self._energy_grid,
                self._cs_db,
                A_target=average_A_target(self._medium),
                physics=self._physics,
            )
            for pdg, hel in particles
        ]

        # Sort by critical energy (= int_len ~== dec_length ~ int_cs/tau)
        particle_list.sort(key=lambda x: x.E_crit, reverse=False)

        # Cascade particles will "live" on the grid and have an mceqidx assigned
        self.cascade_particles = [p for p in particle_list if not p.is_resonance]

        self.cascade_particles = sorted(
            self.cascade_particles, key=lambda p: abs(p.pdg_id[0])
        )

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
            _n = new_mceq_particle.name
            _p = new_mceq_particle.pdg_id
            info(
                0,
                f"Particle {_n}/{_p} has already been added. Use it.",
            )
            return

        if not new_mceq_particle.is_resonance:
            _n = new_mceq_particle.name
            _p = new_mceq_particle.pdg_id
            info(
                2,
                f"New particle {_n}/{_p} is not a resonance.",
            )
            new_mceq_particle.mceqidx = len(self.cascade_particles)
            self.cascade_particles.append(new_mceq_particle)
        else:
            _n = new_mceq_particle.name
            _p = new_mceq_particle.pdg_id
            info(
                2,
                f"New particle {_n}/{_p} is a resonance.",
            )
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
            d.clear()
            for d in [
                self.pdg2mceqidx,
                self.pname2mceqidx,
                self.mceqidx2pdg,
                self.mceqidx2pname,
                self.mceqidx2pref,
                self.pdg2pref,
                self.pname2pref,
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

        info(10, "Restoring tracking particle setup")

        if not self.tracking_relations and self._physics.enable_default_tracking:
            self._init_default_tracking()
            return

        # Clear tracking_relations for this initialization
        self.tracking_relations = []

        for pid, cid, alias, int_dec, incl_anti in self._tracking_requested:
            if pid not in self.pdg2pref:
                info(15, "Can not restore {0}, since not in particle list.")
                continue
            self.add_tracking_particle([pid], cid, alias, int_dec, incl_anti)

    def _init_default_tracking(self):
        """Add default tracking particles for leptons from pi, K, and mu"""
        # Init default tracking particles
        info(1, "Initializing default tracking categories (pi, K, mu)")
        self._tracking_requested_by_default = []
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

        # Track prompt leptons
        prompt_ctau = self._physics.prompt_ctau
        self.track_leptons_from(
            [p.pdg_id for p in self.all_particles if p.ctau < prompt_ctau],
            "prcas_",
            exclude_em=True,
            use_helicities=False,
        )
        # Track leptons from interaction vertices (also prompt)
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
        """Defines the `in` operator to look for particles"""
        if isinstance(pdg_id_or_name, (int,)):
            pdg_id_or_name = (pdg_id_or_name, 0)
        elif isinstance(pdg_id_or_name, (str,)):
            pdg_id_or_name = (_pdata.pdg_id(pdg_id_or_name), 0)
        return pdg_id_or_name in list(self.pdg2pref)

    def __getitem__(self, pdg_id_or_name):
        """Returns reference to particle object."""
        if isinstance(pdg_id_or_name, tuple):
            return self.pdg2pref[pdg_id_or_name]
        if isinstance(pdg_id_or_name, (int,)):
            return self.pdg2pref[(pdg_id_or_name, 0)]
        else:
            try:
                return self.pdg2pref[(_pdata.pdg_id(pdg_id_or_name), 0)]
            except KeyError:
                return self.pname2pref[pdg_id_or_name]

    def keys(self):
        """Returns pdg_ids of all particles"""
        return [p.pdg_id for p in self.all_particles]

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
