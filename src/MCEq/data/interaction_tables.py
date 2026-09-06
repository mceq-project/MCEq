"""Interaction yield matrices: the ``Interactions`` consumer.

Plan section 1's ``data/interaction_tables.py``, §13.2 item 4. The class moved
here verbatim from ``data/__init__.py`` (which becomes a pure re-export
surface). ``normalize_hadronic_model_name`` is imported by name from
``model_names`` rather than read off the package namespace, so this module has
no import-time edge back to ``data/__init__``.
"""

import numpy as np

from MCEq.data.model_names import normalize_hadronic_model_name
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
