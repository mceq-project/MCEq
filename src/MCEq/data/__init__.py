from os.path import (  # noqa: F401  (recorded surface; used by hdf5_backend now)
    isfile,
    join,
)

import numpy as np

# Since Phase 4 the backend class lives in `MCEq.data.hdf5_backend` and the
# blending/energy-grid helpers are reached by their consumers' own imports.
# The bindings below stay because they are the package's recorded public
# surface (tests/test_phase4_surface.py, both bounds); `HDF5Backend` is the
# name `MCEq.core` constructs through the dotted path.
from MCEq.data.blending import (  # noqa: F401
    blend_cross_sections,
    blend_yields,
    he_le_weight,
)
from MCEq.data.energy_grid import EnergyGrid, _eval_energy_cuts  # noqa: F401

# `equivalences` is both a submodule of this package and the dict that submodule
# defines. The import machinery binds the *module* onto `MCEq.data` while loading
# it; the `from ... import equivalences` below runs afterwards and rebinds the
# name to the dict, which is the object `MCEq.data.equivalences` has always been
# (pinned by tests/test_phase4_surface.py). Reach the module itself through
# `sys.modules`, never through the package attribute.
from MCEq.data.equivalences import (  # noqa: F401  (the dict is the surface; users moved)
    apply_equivalences,
    equivalences,
    reverse_equivalences,
)
from MCEq.data.hdf5_backend import HDF5Backend as HDF5Backend
from MCEq.data.model_names import (  # noqa: F401  (family_of is surface only)
    family_of,
    normalize_hadronic_model_name,
)
from MCEq.misc import info


class Interactions:
    """Class for managing the dictionary of interaction yield matrices.

    Args:
    mceq_hdf_db (object): instance of :class:`MCEq.data.HDF5Backend`
    """

    def __init__(self, mceq_hdf_db, physics=None):
        from collections import defaultdict

        from MCEq import config

        self._physics = config.physics if physics is None else physics
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
        # #: (tuple) selection of a band of coeffictients (in xf)
        # self.band = None
        #: (tuple) modified particle combination for error prop.
        self.mod_pprod = defaultdict(dict)

    def load(self, interaction_model, parent_list=None):
        from MCEq.misc import is_charm_pdgid

        self.iam = normalize_hadronic_model_name(interaction_model)
        # Load tables and index from file
        index = self.mceq_db.interaction_db(self.iam)
        filters = self._physics.filters
        disabled_particles = filters["disabled_particles"]
        self.parents = [p for p in index["parents"] if p[0] not in disabled_particles]
        self.relations = index["relations"]
        self.index_d = index["index_d"]
        self.description = index["description"]

        # Advanced options

        if parent_list is not None:
            self.parents = [
                p
                for p in self.parents
                if p in parent_list and p[0] not in disabled_particles
            ]
        if filters["disable_charm_pprod"]:
            self.parents = [p for p in self.parents if not is_charm_pdgid(p[0])]
        if filters["disable_interactions_of_unstable"]:
            self.parents = [
                p for p in self.parents if p[0] not in [2212, 2112, -2212, -2112]
            ]
        if filters["allowed_projectiles"]:
            self.parents = [
                p for p in self.parents if p[0] in filters["allowed_projectiles"]
            ]

        self.particles = []
        for p in list(self.relations):
            if p not in self.parents:
                _ = self.relations.pop(p, None)
                continue
            self.particles.append(p)
            self.particles += [
                d for d in self.relations[p] if d not in disabled_particles
            ]
        self.particles = sorted(list(set(self.particles)))
        if filters["disable_direct_leptons"]:
            for p in list(self.relations):
                self.relations[p] = [
                    c for c in self.relations[p] if not 10 < abs(c[0]) < 20
                ]

        if len(disabled_particles) > 0:
            for p in list(self.relations):
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

        :math:`x = \\frac{E_{\\rm primary}}{E_{\\rm secondary}}` is the
        fraction of secondary particle energy. ``x_func`` can be an
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
        from MCEq.data.energy_grid import gen_xmat

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
        :func:`InteractionYields.init_mod_matrix`

        Args:
          prim_pdg (int): interacting (primary) particle PDG ID
          sec_pdg (int): secondary particle PDG ID
        """

        # Short cut for the pprod list
        mpli = self.mod_pprod
        pstup = (prim_pdg, sec_pdg)

        if self._physics.use_isospin_sym and prim_pdg not in [2212, 2112]:
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

        if not self._physics.use_isospin_sym:
            return True

        prim_pdg, symm_pdg = 2212, 2112
        if prim_pdg == 2112:
            prim_pdg = 2112
            symm_pdg = 2212

        # p->pi+ = n-> pi-, p->pi- = n-> pi+
        if abs(sec_pdg) == 211:
            # Add the same mod to the isospin symmetric particle combination
            mpli[(symm_pdg, -sec_pdg)][("isospin", args)] = kmat

            # Assumption: Unflavored production coupled to the average
            # of pi+ and pi- production

            if np.any([p in self.parents for p in [221, 223, 333]]):
                unflv_arg = None
                if (prim_pdg, -sec_pdg) not in mpli:
                    # Only pi+ or pi- (not both) have been modified
                    unflv_arg = (args[0], 0.5 * args[1])

                if (prim_pdg, -sec_pdg) in mpli:
                    # Compute average of pi+ and pi- modification matrices
                    # Save the 'average' argument (just for meaningful output)
                    for arg_name, arg_val in mpli[(prim_pdg, -sec_pdg)]:
                        if arg_name == args[0]:
                            unflv_arg = (args[0], 0.5 * (args[1] + arg_val))

                unflmat = self._gen_mod_matrix(x_func, *unflv_arg)

                # modify eta, omega, phi, 221, 223, 333
                for t in [
                    (prim_pdg, 221),
                    (prim_pdg, 223),
                    (prim_pdg, 333),
                    (symm_pdg, 221),
                    (symm_pdg, 223),
                    (symm_pdg, 333),
                ]:
                    mpli[t][("isospin", unflv_arg)] = unflmat

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
        """Returns a ``DIM x DIM`` yield matrix.

        Args:
          parent (int): PDG ID of parent particle
          child (int): PDG ID of final state child/secondary particle
        Returns:
          numpy.array: yield matrix
        """
        info(10, "Called for", parent, child)
        if child not in self.relations[parent]:
            raise Exception(f"trying to get empty matrix {parent} -> {child}")

        m = self.index_d[(parent, child)]

        if (
            self._physics.filters["disable_leading_mesons"]
            and abs(child) < 2000
            and (parent, -child) in list(self.index_d)
        ):
            m_anti = self.index_d[(parent, -child)]
            ie = 50
            info(
                5,
                "sum in disable_leading_mesons",
                (np.sum(m[:, ie - 30 : ie]) - np.sum(m_anti[:, ie - 30 : ie])),
            )

            if (np.sum(m[:, ie - 30 : ie]) - np.sum(m_anti[:, ie - 30 : ie])) > 0:
                info(
                    5,
                    "inverting meson due to leading particle veto.",
                    child,
                    "->",
                    -child,
                )
                m = m_anti
            else:
                info(5, "no inversion since child not leading", child)
        else:
            info(20, "no meson inversion in leading particle veto.", parent, child)
        if (parent[0], child[0]) in list(self.mod_pprod):
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

    def __init__(self, mceq_hdf_db, override_decay_db_name=None, physics=None):
        from MCEq import config

        physics = config.physics if physics is None else physics
        #: MCEq HDF5Backend reference
        self.mceq_db = mceq_hdf_db
        #: (list) List of particles in the decay matrices
        self.parent_list = []
        self._default_decay_dset = override_decay_db_name

        if self._default_decay_dset is None:
            if physics.muon_helicity_dependence:
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
        # Advanced options
        regenerate_index = False
        if parent_list:
            # Take only the parents provided by the list
            # Add the decay products, which can become new parents
            def _follow_decay_chain(p, plist):
                if p in self.relations:
                    plist.append(p)
                    for d in self.relations[p]:
                        _follow_decay_chain(d, plist)
                else:
                    return plist

            plist = []
            for p in parent_list:
                _follow_decay_chain(p, plist)

            self.parents = sorted(list(set(plist)))
            regenerate_index = True

        if regenerate_index:
            self.particles = []
            for p in list(self.relations):
                if p not in self.parents:
                    _ = self.relations.pop(p, None)
                    continue
                self.particles.append(p)
                self.particles += self.relations[p]
            self.particles = sorted(list(set(self.particles)))

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
        """Returns a ``DIM x DIM`` decay matrix.

        Args:
          parent (int): PDG ID of parent particle
          child (int): PDG ID of final state child particle
        Returns:
          numpy.array: decay matrix
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
        #: Dictionary containing the distribuiton matrices
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
            """Family representative, sign-preserving: a negative parent
            uses the antiparticle column when the model provides one
            (annihilation channels for antibaryons; dedicated K-/pi-
            columns) and falls back to the particle column otherwise."""
            if parent < 0 and -base_pdg in self.index_d:
                info(15, message_templ.format(parent, f"anti-{label}"))
                return self.index_d[-base_pdg]
            info(15, message_templ.format(parent, label))
            return self.index_d[base_pdg]

        if parent in list(self.index_d):
            cs = self.index_d[parent]
        elif abs(parent) in list(self.index_d):
            cs = self.index_d[abs(parent)]
        elif 100 < abs(parent) < 300 and abs(parent) != 130:
            cs = _family_cs(211, "pi+-")
        elif 300 < abs(parent) < 1000 or abs(parent) in [130, 10313, 10323]:
            cs = _family_cs(321, "K+-")
        elif abs(parent) > 1000 and abs(parent) < 5000:
            cs = _family_cs(2212, "nucleon")
        elif 5 < abs(parent) < 23:
            info(15, "returning 0 cross-section for lepton", parent)
            return np.zeros(self.energy_grid.d)
        else:
            info(1, "Strange case for parent, using zero cross section.")
            cs = 0.0

        if not mbarn:
            return self.mbarn2cm2 * cs
        return cs


class ContinuousLosses:
    """Class for managing the dictionary of hadron-air cross-sections.

    Args:
      mceq_hdf_db (object): instance of :class:`MCEq.data.HDF5Backend`
      material (str): name of the material (not fully implemented)
    """

    def __init__(self, mceq_hdf_db, physics=None):
        from MCEq import config

        self._physics = config.physics if physics is None else physics
        #: MCEq HDF5Backend reference
        self.mceq_db = mceq_hdf_db
        #: reference to energy grid
        self.energy_grid = mceq_hdf_db.energy_grid
        #: List of active parents
        self.parents = None
        #: Dictionary containing the distribuiton matrices
        self.index_d = None
        # Load defaults
        self.load_db()

    def __getitem__(self, parent):
        """Return the cross section in :math:`\\text{cm}^2` as
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
        if "generic" not in index and self._physics.generic_losses_all_charged:
            raise Exception("New data file needed to support generic losses.")
        self.generic_spl = InterpolatedUnivariateSpline(
            np.log(index["generic"][0]), np.log(index["generic"][1]), k=1
        )
