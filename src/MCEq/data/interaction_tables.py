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

    def __init__(self, mceq_hdf_db, *, physics):
        from collections import defaultdict

        self._physics = physics
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
        #         #: (tuple) selection of a band of coeffictients (in xf)
        # self.band = None
        #: (tuple) modified particle combination for error prop.
        self.mod_pprod = defaultdict(dict)
        #: (dict) absolute channel overrides, keyed by the table's own
        #: ``(parent, child)`` key shape, applied in ``get_matrix`` (R2-B).
        #: Survives ``load`` (the channel is a policy on the table, not a
        #: slice of the file) and is the only write path for external
        #: matrices (``MCEqRun.inject_ddm``); see ``set_channel_override``.
        self.channel_overrides = {}

    def set_channel_override(self, parent, child, matrix):
        """Override the yield matrix of one channel (R2-B).

        ``get_matrix`` returns *matrix* for this channel from now on,
        across ``load``/``set_interaction_model`` reloads, so
        ``get_matrix``, the particle wiring built from it
        (``set_hadronic_channels``), ``dN_dxlab``, and the assembled
        ``int_m`` all agree with the injected matrix, and the injection
        is no longer a ``hadr_yields`` dict write that the next reload
        silently wipes (data §5, risk 18). Bare PDG ints are accepted
        and normalised to helicity-0 keys.

        The override is returned as stored, like the raw ``index_d``
        matrices are; treat it as read-only. ``mod_pprod`` factors
        still multiply on top afterwards, so a Barr-style modification
        composes with an injected matrix instead of replacing it.

        No membership validation: a channel absent from the loaded
        model simply never gets wired (``set_hadronic_channels`` only
        iterates the relations), exactly as for a DB channel the
        particle filters dropped.
        """
        import numpy as np

        parent = (parent, 0) if isinstance(parent, int) else tuple(parent)
        child = (child, 0) if isinstance(child, int) else tuple(child)
        self.channel_overrides[(parent, child)] = np.asarray(matrix)
        info(5, f"channel override set for {parent} -> {child}")
        return True

    def clear_channel_overrides(self):
        """Drop every ``set_channel_override`` injection.

        The table reverts to the file until the caller re-wires (driver:
        ``MCEqRun.regenerate_matrices``).
        """
        n = len(self.channel_overrides)
        self.channel_overrides = {}
        info(1, f"{n} channel overrides cleared.")
        return n

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

        # Dispatch on the caller's primary BEFORE binding the locals (B1 fix,
        # R3): the pre-fix code bound ``prim_pdg, symm_pdg = 2212, 2112`` first
        # and then tested the overwritten name, so the swap could never fire
        # and a 2112 primary was treated as a proton -- its isospin copies
        # landed on the proton's partner keys (or, in the kaon leg, on the
        # primary's own entry). A 2112 primary now gets symm 2212.
        if prim_pdg == 2112:
            prim_pdg, symm_pdg = 2112, 2212
        else:
            prim_pdg, symm_pdg = 2212, 2112

        # p->pi+ = n-> pi-, p->pi- = n-> pi+
        if abs(sec_pdg) == 211:
            # Add the same mod to the isospin symmetric particle combination
            mpli[(symm_pdg, -sec_pdg)][("isospin", args)] = kmat

            # R3 deletes the unflavoured (221/223/333) coupling that used to
            # sit here: its guard tested bare ints against the
            # (pdg, helicity) keys of ``self.parents``, so it never fired --
            # and no shipped database carries those ids anyway, so "fixing"
            # the comparison would have written six dead keys for no physics.
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

        # R2-B: absolute overrides replace the file matrix; mod_pprod
        # factors still multiply on top below, so the two mechanisms
        # compose rather than fight (data §5, risk 18).
        if (parent, child) in self.channel_overrides:
            m = self.channel_overrides[(parent, child)]
        else:
            m = self.index_d[(parent, child)]

        # R3 deletes the `disable_leading_mesons` veto with its
        # `ie = 50` window here: since the index gained helicity the guard
        # `abs(child) < 2000` raises TypeError on the (pdg, hel) tuple before
        # the window is ever read, so the feature has been dead for as long as
        # this layout has existed.
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
