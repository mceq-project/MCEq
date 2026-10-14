"""Interaction, decay, cross-section and continuous-loss tables read through :class:`HDF5Backend`."""

from collections import defaultdict

import numpy as np

from MCEq import config
from MCEq.misc import gen_xmat, info, normalize_hadronic_model_name


def _prune_relations(relations, parents):
    """Drop the relations of non-parents in place; return the remaining particles."""
    particles = set()
    for p in list(relations):
        if p not in parents:
            del relations[p]
            continue
        particles.add(p)
        particles.update(relations[p])
    return sorted(particles)


class Interactions:
    """Class for managing the dictionary of interaction yield matrices.

    Args:
    mceq_hdf_db (object): instance of :class:`MCEq.data.HDF5Backend`
    """

    def __init__(self, mceq_hdf_db):
        #: MCEq HDF5Backend reference
        self.mceq_db = mceq_hdf_db
        #: reference to energy grid
        self.energy_grid = mceq_hdf_db.energy_grid
        #: List of active parents
        self.parents = None
        #: List of all known particles
        self.particles = None
        #: Dictionary parent/child relations
        self.relations = None
        #: Dictionary containing the distribuiton matrices
        self.index_d = None
        #: String containing the desciption of the model
        self.description = None

        #: (str) Interaction Model name
        self.iam = None
        #: (tuple) modified particle combination for error prop.
        self.mod_pprod = defaultdict(dict)
        #: (dict) absolute channel overrides ``{(parent, child): matrix}``,
        #: kept across ``load``; see ``set_channel_override``.
        self.channel_overrides = {}

    def set_channel_override(self, parent, child, matrix):
        """Replace the channel ``parent -> child`` by ``matrix`` in ``get_matrix``.

        Production modifications (``mod_pprod``) still multiply on top. Bare PDG
        ints are normalised to helicity-0 keys. The override does not add a
        relation, so a channel absent from the loaded model is never used.
        """
        parent = (parent, 0) if isinstance(parent, int) else tuple(parent)
        child = (child, 0) if isinstance(child, int) else tuple(child)
        self.channel_overrides[(parent, child)] = np.asarray(matrix)
        info(5, f"channel override set for {parent} -> {child}")
        return True

    def clear_channel_overrides(self):
        """Drop every ``set_channel_override`` injection.

        The table reverts to the file immediately; particle channels and
        assembled matrices must then be refreshed by the caller
        (``MCEqRun.regenerate_matrices``).
        """
        n = len(self.channel_overrides)
        self.channel_overrides = {}
        info(1, f"{n} channel overrides cleared.")
        return n

    def load(self, interaction_model, parent_list=None):
        self.iam = normalize_hadronic_model_name(interaction_model)
        # Load tables and index from file
        index = self.mceq_db.interaction_db(self.iam)
        filters = config.adv_set
        disabled_particles = filters["disabled_particles"]
        self.parents = [p for p in index["parents"] if p[0] not in disabled_particles]
        self.relations = index["relations"]
        self.index_d = index["index_d"]
        self.description = index["description"]
        #: ``(radius, {pdg})`` of stencil-averaged yield tables, or None
        self.bin_average = index.get("bin_average")

        # Advanced options

        if parent_list is not None:
            self.parents = [p for p in self.parents if p in parent_list]
        if filters["disable_charm_pprod"]:
            self.parents = [
                p
                for p in self.parents
                if not (400 < abs(p[0]) < 500 or 4000 < abs(p[0]) < 5000)
            ]
        if filters["disable_interactions_of_unstable"]:
            self.parents = [
                p for p in self.parents if p[0] not in [2212, 2112, -2212, -2112]
            ]
        if filters["allowed_projectiles"]:
            self.parents = [
                p for p in self.parents if p[0] in filters["allowed_projectiles"]
            ]

        self.particles = _prune_relations(self.relations, self.parents)
        if filters["disable_direct_leptons"]:
            for p in list(self.relations):
                self.relations[p] = [
                    c for c in self.relations[p] if not 10 < abs(c[0]) < 20
                ]

        if disabled_particles:
            for p in self.relations:
                self.relations[p] = [
                    c for c in self.relations[p] if c[0] not in disabled_particles
                ]
        if not self.particles:
            info(
                2, "None of the parent_list particles interact. Returning custom list."
            )
            self.particles = parent_list

    def __getitem__(self, key):
        return self.get_matrix(*key)

    def __contains__(self, key):
        """Defines the `in` operator to look for particles"""
        return key in self.parents

    def _gen_mod_matrix(self, x_func, *args):
        """Creates modification matrix using an (x,E)-dependent function.

        :math:`x = \\frac{E_{\\rm secondary}}{E_{\\rm primary}}` is the
        secondary energy divided by the projectile energy. ``x_func`` can be an
        arbitrary function modifying the :math:`x_\\text{lab}` distribution.
        Run this method each time you change ``x_func``, or its parameters,
        not each time you change modified particle.
        The ``args`` are passed to the function.

        Args:
          x_func (object): reference to function
          args (tuple): arguments of `x_func`

        Returns:
          (numpy.array): modification matrix
        """
        info(2, "Generating modification matrix for", x_func.__name__, args)

        xmat = gen_xmat(self.energy_grid)

        # select the relevant slice of interaction matrix
        modmat = x_func(xmat, self.energy_grid.c, *args)
        # Set lower triangular indices to 0. (should be not necessary)
        modmat[np.tril_indices(self.energy_grid.d, -1)] = 0.0

        return np.asarray(modmat)

    def _set_mod_pprod(self, prim_pdg, sec_pdg, x_func, args):
        """Sets combination of parent/secondary for error propagation.

        The production spectrum of ``sec_pdg`` in interactions of
        ``prim_pdg`` is modified according to the function passed to
        :meth:`Interactions._gen_mod_matrix`

        Args:
          prim_pdg (int): interacting (primary) particle PDG ID
          sec_pdg (int): secondary particle PDG ID
        """

        # Short cut for the pprod list
        mpli = self.mod_pprod
        pstup = (prim_pdg, sec_pdg)

        if config.use_isospin_sym and prim_pdg not in [2212, 2112]:
            raise Exception("Unsupported primary for isospin symmetries.")

        if (x_func.__name__, args) in mpli[(pstup)]:
            info(
                5,
                " no changes to particle production"
                + f" modification matrix of {prim_pdg}/{sec_pdg} for {x_func.__name__},{args}",
            )
            return False

        # Check function with same mode but different parameter is supplied
        for xf_name, fargs in list(mpli[pstup]):
            if (xf_name == x_func.__name__) and (fargs[0] == args[0]):
                info(
                    1,
                    "Warning. If you modify only the value of a function,",
                    "unset and re-apply all changes",
                )
                return False

        info(
            2,
            "modifying modify particle production"
            + f" matrix of {prim_pdg}/{sec_pdg} for {x_func.__name__},{args}",
        )

        kmat = self._gen_mod_matrix(x_func, *args)
        mpli[pstup][(x_func.__name__, args)] = kmat

        info(5, 'modification "strength"', np.sum(kmat) / np.count_nonzero(kmat))

        if not config.use_isospin_sym:
            return True

        # the other nucleon
        if prim_pdg == 2112:
            prim_pdg, symm_pdg = 2112, 2212
        else:
            prim_pdg, symm_pdg = 2212, 2112

        # p->pi+ = n-> pi-, p->pi- = n-> pi+
        if abs(sec_pdg) == 211:
            # Add the same mod to the isospin symmetric particle combination
            mpli[(symm_pdg, -sec_pdg)][("isospin", args)] = kmat
        # Charged and neutral kaons
        elif abs(sec_pdg) == 321:
            # approx.: p->K+ ~ n-> K+, p->K- ~ n-> K-
            mpli[(symm_pdg, sec_pdg)][("isospin", args)] = kmat
            k0_arg = (args[0], 0.5 * args[1])
            if (prim_pdg, -sec_pdg) in mpli:
                # Compute average of K+ and K- modification matrices
                # Save the 'average' argument (just for meaningful printout)
                for arg_name, arg_val in mpli[(prim_pdg, -sec_pdg)]:
                    if arg_name == args[0]:
                        k0_arg = (args[0], 0.5 * (args[1] + arg_val))

            k0mat = self._gen_mod_matrix(x_func, *k0_arg)

            # modify K0L/S
            for t in [
                (prim_pdg, 310),
                (prim_pdg, 130),
                (symm_pdg, 310),
                (symm_pdg, 130),
            ]:
                mpli[t][("isospin", k0_arg)] = k0mat

        elif abs(sec_pdg) == 411:
            ssec = np.sign(sec_pdg)
            mpli[(prim_pdg, ssec * 421)][("isospin", args)] = kmat
            mpli[(prim_pdg, ssec * 431)][("isospin", args)] = kmat
            mpli[(symm_pdg, sec_pdg)][("isospin", args)] = kmat
            mpli[(symm_pdg, ssec * 421)][("isospin", args)] = kmat
            mpli[(symm_pdg, ssec * 431)][("isospin", args)] = kmat

        # Leading particles
        elif abs(sec_pdg) == prim_pdg:
            mpli[(symm_pdg, symm_pdg)][("isospin", args)] = kmat
        elif abs(sec_pdg) == symm_pdg:
            mpli[(symm_pdg, prim_pdg)][("isospin", args)] = kmat
        else:
            info(0, " Warning: No isospin relation found for secondary" + str(sec_pdg))

        # Tell MCEqRun to regenerate the matrices if something has changed
        return True

    def print_mod_pprod(self):
        """Prints the active particle production modification."""

        for i, (prim_pdg, sec_pdg) in enumerate(sorted(self.mod_pprod)):
            for j, (argname, argv) in enumerate(self.mod_pprod[(prim_pdg, sec_pdg)]):
                info(
                    2,
                    f"{i + j}: {prim_pdg} -> {sec_pdg}, func: {argname}, arg: {argv}",
                    no_caller=True,
                )

    def get_matrix(self, parent, child):
        """Return the interaction channel for parent and child ``(PDG ID, helicity)`` keys.

        Args:
          parent: ``(PDG ID, helicity)`` key of the parent particle
          child: ``(PDG ID, helicity)`` key of the final-state child
        Returns:
          numpy.array: channel array, shaped ``(d, d)`` or ``(n_k, d, d)``
        """
        info(10, "Called for", parent, child)
        if child not in self.relations[parent]:
            raise Exception(f"trying to get empty matrix {parent} -> {child}")

        # An override replaces the file matrix; production modifications
        # still multiply on top below.
        if (parent, child) in self.channel_overrides:
            m = self.channel_overrides[(parent, child)]
        else:
            m = self.index_d[(parent, child)]

        if (parent[0], child[0]) in self.mod_pprod:
            info(
                5,
                f"using modified particle production for {parent[0]}/{child[0]}",
            )
            i = 0
            m = np.copy(m)
            for args, mmat in iter(self.mod_pprod[(parent[0], child[0])].items()):
                info(10, i, (parent[0], child[0]), args, np.sum(mmat), np.sum(m))
                i += 1
                m *= mmat

        return m


class Decays:
    """Class for managing the dictionary of decay yield matrices.

    Args:
      mceq_hdf_db (object): instance of :class:`MCEq.data.HDF5Backend`
    """

    def __init__(self, mceq_hdf_db, override_decay_db_name=None):
        #: MCEq HDF5Backend reference
        self.mceq_db = mceq_hdf_db
        self._default_decay_dset = override_decay_db_name

        if self._default_decay_dset is None:
            if config.muon_helicity():
                self._default_decay_dset = "polarized"
            else:
                self._default_decay_dset = "unpolarized"

    def load(self, parent_list=None, decay_dset=None):
        # Load tables and index from file
        if decay_dset is None:
            decay_dset = self._default_decay_dset

        index = self.mceq_db.decay_db(decay_dset)

        self.parents = index["parents"]
        self.particles = index["particles"]
        self.relations = index["relations"]
        self.index_d = index["index_d"]
        self.description = index["description"]
        #: ``(radius, {pdg})`` of stencil-averaged decay tables, or None
        self.bin_average = index.get("bin_average")
        if parent_list:
            # The listed parents and every decay product that decays further.
            def _follow_decay_chain(p, plist):
                if p in self.relations:
                    plist.append(p)
                    for d in self.relations[p]:
                        _follow_decay_chain(d, plist)

            plist = []
            for p in parent_list:
                _follow_decay_chain(p, plist)

            self.parents = sorted(set(plist))
            self.particles = _prune_relations(self.relations, self.parents)

    def __getitem__(self, key):
        return self.get_matrix(*key)

    def __contains__(self, key):
        """Defines the `in` operator to look for particles"""
        return key in self.parents

    def children(self, parent_pdg):
        if parent_pdg not in self.relations:
            raise Exception(f"Parent {parent_pdg} not in decay database.")

        return self.relations[parent_pdg]

    def get_matrix(self, parent, child):
        """Return the channel array for parent and child ``(PDG ID, helicity)`` keys.

        Args:
          parent: ``(PDG ID, helicity)`` key of the parent particle
          child: ``(PDG ID, helicity)`` key of the final-state child
        Returns:
          numpy.array: channel array, shaped ``(d, d)`` or ``(n_k, d, d)``
          depending on the stored channel
        """
        info(20, "entering with", parent, child)
        if child not in self.relations[parent]:
            raise Exception(f"trying to get empty matrix {parent} -> {child}")

        return self.index_d[(parent, child)]


class InteractionCrossSections:
    """Class for managing the dictionary of hadron-air cross-sections.

    Args:
      mceq_hdf_db (object): instance of :class:`MCEq.data.HDF5Backend`
      interaction_model (str): name of the interaction model
    """

    #: unit - :math:`\text{GeV} \cdot \text{fm}`
    GeVfm = 0.19732696312541853
    #: unit - :math:`\text{GeV} \cdot \text{cm}`
    GeVcm = GeVfm * 1e-13
    #: unit - :math:`\text{GeV}^2 \cdot \text{mbarn}`
    GeV2mbarn = 10.0 * GeVfm**2
    #: unit conversion - :math:`\text{mbarn} \to \text{cm}^2`
    mbarn2cm2 = GeVcm**2 / GeV2mbarn

    def __init__(self, mceq_hdf_db, interaction_model="DPMJETIII193"):
        #: MCEq HDF5Backend reference
        self.mceq_db = mceq_hdf_db
        #: reference to energy grid
        self.energy_grid = mceq_hdf_db.energy_grid
        #: List of active parents
        self.parents = None
        #: Cross-section arrays indexed by projectile
        self.index_d = None
        #: (str) Interaction Model name
        self.iam = normalize_hadronic_model_name(interaction_model)

    def __getitem__(self, parent):
        """Return the cross section in :math:`\\text{cm}^2` as a dictionary
        lookup."""
        return self.get_cs(parent)

    def __contains__(self, key):
        """Defines the `in` operator to look for particles"""
        return key in self.parents

    def load(self, interaction_model):
        #: (str) Interaction Model name
        self.iam = normalize_hadronic_model_name(interaction_model)
        # Load tables and index from file
        index = self.mceq_db.cs_db(self.iam)

        self.parents = index["parents"]
        self.index_d = index["index_d"]

    def get_cs(self, parent, mbarn=False):
        """Returns production ``parent``-air cross-section
        :math:`\\sigma_{prod}^{proj-Air}(E)` as vector spanned over
        the energy grid.

        Args:
          parent (int): PDG ID of parent particle
          mbarn (bool,optional): if ``True``, the units of the cross-section
                                 will be :math:`mbarn`,
                                 else :math:`\\text{cm}^2`

        Returns:
          numpy.array: cross-section in :math:`mbarn` or :math:`\\text{cm}^2`
        """

        message_templ = "replacing {0} with {1} cross section"
        if isinstance(parent, tuple):
            parent = parent[0]

        def _family_cs(base_pdg, label):
            # A negative parent uses the antiparticle column if the model has
            # one (annihilation channels for antibaryons), else the particle's.
            if parent < 0 and -base_pdg in self.index_d:
                info(15, message_templ.format(parent, f"anti-{label}"))
                return self.index_d[-base_pdg]
            info(15, message_templ.format(parent, label))
            return self.index_d[base_pdg]

        if parent in self.index_d:
            cs = self.index_d[parent]
        elif abs(parent) in self.index_d:
            cs = self.index_d[abs(parent)]
        elif 100 < abs(parent) < 300 and abs(parent) != 130:
            cs = _family_cs(211, "pi+-")
        elif 300 < abs(parent) < 1000 or abs(parent) == 130:
            cs = _family_cs(321, "K+-")
        elif abs(parent) > 1000 and abs(parent) < 5000:
            cs = _family_cs(2212, "nucleon")
        elif 5 < abs(parent) < 23:
            info(15, "returning 0 cross-section for lepton", parent)
            return np.zeros(self.energy_grid.d)
        else:
            info(1, "Strange case for parent, using zero cross section.")
            cs = np.zeros(self.energy_grid.d)

        if not mbarn:
            return self.mbarn2cm2 * cs
        return cs


class ContinuousLosses:
    """Manage stopping-power arrays loaded from ``mceq_hdf_db``.

    Args:
      mceq_hdf_db (object): instance of :class:`MCEq.data.HDF5Backend`
    """

    def __init__(self, mceq_hdf_db):
        #: MCEq HDF5Backend reference
        self.mceq_db = mceq_hdf_db
        #: Energy grid of the backend
        self.energy_grid = mceq_hdf_db.energy_grid
        #: List of active parents
        self.parents = None
        #: Stopping-power arrays indexed by particle
        self.index_d = None
        #: Raw ``(boost, dEdX)`` table behind :attr:`generic_spl`, used to
        #: clamp arguments to the tabulated range.
        self.generic_table = None
        # Load defaults
        self.load_db()

    def __getitem__(self, parent):
        """Return the particle's stopping-power array in GeV/(g/cm²) as
        a dictionary lookup."""
        return self.index_d[parent]

    def __contains__(self, key):
        """Defines the `in` operator to look for particles"""
        return key in self.parents

    def load_db(self):
        from scipy.interpolate import InterpolatedUnivariateSpline

        # Load tables and index from file
        index = self.mceq_db.continuous_loss_db()
        self.parents = index["parents"]
        self.index_d = index["index_d"]
        if "generic" not in index:
            raise Exception("New data file needed to support generic losses.")
        generic = index["generic"]
        self.generic_table = np.asarray(generic, dtype=np.float64)
        self.generic_spl = InterpolatedUnivariateSpline(
            np.log(generic[0]), np.log(generic[1]), k=1
        )
