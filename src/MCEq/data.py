import six
import numpy as np
import h5py
from os.path import join, isfile
from collections import defaultdict
from MCEq import config, asarray
from .misc import normalize_hadronic_model_name, info, _eval_energy_cuts

# TODO: Convert this to some functional generic class. Very erro prone to
# enter stuff by hand
equivalences = {
    "SIBYLL23": {
        -4132: 4122,
        -4122: 4122,
        -3334: -3312,
        -3322: -2112,
        -3212: -3122,
        -413: -411,
        113: 211,
        221: 211,
        111: 211,
        310: 130,
        413: 411,
        3212: 3122,
        3334: 3312,
    },
    "SIBYLL21": {
        -3322: 2112,
        -3312: 2212,
        -3222: 2212,
        -3212: 2112,
        -3122: 2112,
        -3112: 2212,
        -2212: 2212,
        -2112: 2112,
        310: 130,
        111: 211,
        3112: 2212,
        3122: 2112,
        3212: 2112,
        3222: 2212,
        3312: 2212,
        3322: 2112,
    },
    "QGSJET01": {
        -4122: 2212,
        -3322: 2212,
        -3312: 2212,
        -3222: 2212,
        -3122: 2212,
        -3112: 2212,
        -2212: 2212,
        -2112: 2212,
        -421: 321,
        -411: 321,
        -211: 211,
        -321: 321,
        111: 211,
        221: 211,
        130: 321,
        310: 321,
        411: 321,
        421: 321,
        2112: 2212,
        3112: 2212,
        3122: 2212,
        3222: 2212,
        3312: 2212,
        3322: 2212,
        4122: 2212,
    },
    "QGSJETII": {
        -3122: -2112,
        111: 211,
        113: 211,
        221: 211,
        310: 130,
        3122: 2112,
    },
    "DPMJET": {
        -4122: -3222,
        -3334: -3312,
        -3212: -3122,
        -431: -321,
        -421: -321,
        -413: -321,
        -411: -321,
        310: 130,
        113: 211,
        221: 211,
        111: 211,
        411: 321,
        413: 321,
        421: 321,
        431: 321,
        3212: 3122,
        3334: 3312,
        4122: 3222,
    },
    "EPOSLHC": {
        -3334: 2212,
        -3322: -3122,
        -3312: 2212,
        -3222: -2212,
        -3212: -3122,
        -3112: 2212,
        111: 211,
        113: 211,
        221: 211,
        310: 130,
        3112: -2212,
        3212: 3122,
        3222: 2212,
        3312: -2212,
        3322: 3122,
        3334: -2212,
    },
    "PYTHIA8": {
        -3122: -2112,
        -431: -321,
        -421: -321,
        -413: -321,
        -411: -321,
        111: 211,
        113: 211,
        221: 211,
        310: 321,
        130: 321,
        411: 321,
        413: 321,
        421: 321,
        431: 321,
        3122: 2112,
    },
}

equivalences["FLUKA"] = equivalences["DPMJET"]
class HDF5Backend(object):
    """Provides access to tabulated data stored in an HDF5 file.

    The file contains all necessary ingredients to run MCEq, i.e. no
    other files are required. This database is not maintained in git
    and it will change infrequently.
    """

    def __init__(self, medium=config.interaction_medium):
        info(2, "Opening HDF5 file", config.mceq_db_fname)
        self.had_fname = join(config.data_dir, config.mceq_db_fname)
        if not isfile(self.had_fname):
            raise Exception(
                'MCEq DB file {0} not found in "data" directory.'.format(
                    config.mceq_db_fname
                )
            )

        self.em_fname = join(config.data_dir, config.em_db_fname)
        if config.enable_em and not isfile(self.had_fname):
            raise Exception(
                'Electromagnetic DB file {0} not found in "data" directory.'.format(
                    config.em_db_fname
                )
            )

        with h5py.File(self.had_fname, "r") as mceq_db:
            from MCEq.misc import energy_grid

            ca = mceq_db["common"].attrs
            self.version = (
                mceq_db.attrs["version"] if "version" in mceq_db.attrs else "1.0.0"
            )
            self.min_idx, self.max_idx, self._cuts = _eval_energy_cuts(
                ca["e_grid"], config.e_min, config.e_max
            )
            self._energy_grid = energy_grid(
                ca["e_grid"][self._cuts],
                ca["e_bins"][self.min_idx : self.max_idx + 1],
                ca["widths"][self._cuts],
                int(self.max_idx - self.min_idx),
            )
            self.dim_full = int(ca["e_dim"])

        self.medium = medium

    @property
    def energy_grid(self):
        return self._energy_grid

    def _gen_db_dictionary(self, hdf_root, indptrs, equivalences={}):
        from scipy.sparse import csr_matrix

        index_d = {}
        relations = defaultdict(lambda: [])
        particle_list = []
        if "description" in hdf_root.attrs:
            description = hdf_root.attrs["description"]
        else:
            description = None
        mat_data = np.asarray(hdf_root[:, :], dtype=config.floatlen)
        indptr_data = np.asarray(indptrs[:])
        len_data = hdf_root.attrs["len_data"]
        if hdf_root.attrs["tuple_idcs"].shape[1] == 4:
            model_particles = sorted(
                list(set(hdf_root.attrs["tuple_idcs"][:, (0, 2)].flatten().tolist()))
            )
        else:
            model_particles = sorted(
                list(set(hdf_root.attrs["tuple_idcs"].flatten().tolist()))
            )

        exclude = config.adv_set["disabled_particles"]
        read_idx = 0
        available_parents = [
            (pdg, parity) for (pdg, parity) in (hdf_root.attrs["tuple_idcs"][:, :2])
        ]
        available_parents = sorted(list(set(available_parents)))

        # Reverse equivalences
        eqv_lookup = defaultdict(lambda: [])
        for k in equivalences:
            eqv_lookup[(equivalences[k], 0)].append((k, 0))

        for tupidx, tup in enumerate(hdf_root.attrs["tuple_idcs"]):
            if len(tup) == 4:
                parent_pdg, child_pdg = tuple(tup[:2]), tuple(tup[2:])
            elif len(tup) == 2:
                parent_pdg, child_pdg = (tup[0], 0), (tup[1], 0)
            else:
                raise Exception("Failed decoding parent-child relation.")

            if (abs(parent_pdg[0]) in exclude) or (abs(child_pdg[0]) in exclude):
                read_idx += len_data[tupidx]
                continue
            parent_pdg = int(parent_pdg[0]), (parent_pdg[1])
            child_pdg = int(child_pdg[0]), (child_pdg[1])

            particle_list.append(parent_pdg)
            particle_list.append(child_pdg)
            index_d[(parent_pdg, child_pdg)] = asarray(
                (
                    csr_matrix(
                        (
                            mat_data[0, read_idx : read_idx + len_data[tupidx]],
                            mat_data[1, read_idx : read_idx + len_data[tupidx]],
                            indptr_data[tupidx, :],
                        ),
                        shape=(self.dim_full, self.dim_full),
                    )[self._cuts, self.min_idx : self.max_idx]
                ).toarray()
            )
            relations[parent_pdg].append(child_pdg)

            info(
                20,
                "This parent {0} is used for interactions of".format(parent_pdg[0]),
                [p[0] for p in eqv_lookup[parent_pdg]],
                condition=len(equivalences) > 0,
            )
            if config.assume_nucleon_interactions_for_exotics:
                for eqv_parent in eqv_lookup[parent_pdg]:
                    if eqv_parent[0] not in model_particles:
                        info(
                            10,
                            "No equiv. replacement needed of",
                            eqv_parent,
                            "for",
                            parent_pdg,
                            "parent.",
                        )
                        continue
                    elif eqv_parent in available_parents:
                        info(
                            10,
                            "Parent {0} has dedicated simulation.".format(
                                eqv_parent[0]
                            ),
                        )
                        continue
                    particle_list.append(eqv_parent)
                    index_d[(eqv_parent, child_pdg)] = index_d[(parent_pdg, child_pdg)]
                    relations[eqv_parent] = relations[parent_pdg]
                    info(
                        15,
                        "equivalence of {0} and {1} interactions".format(
                            eqv_parent[0], parent_pdg[0]
                        ),
                    )

            read_idx += len_data[tupidx]

        return {
            "parents": sorted(list(relations)),
            "particles": sorted(list(set(particle_list))),
            "relations": dict(relations),
            "index_d": dict(index_d),
            "description": description,
        }

    def _check_subgroup_exists(self, subgroup, mname):
        available_models = [m for m in list(subgroup) if "indptrs" not in m]
        if mname not in available_models:
            info(0, "Invalid choice/model", mname)
            info(0, "Choose from:\n", "\n".join(available_models))
            raise Exception("Unknown selections.")

    def interaction_db(self, interaction_model_name):
        mname = normalize_hadronic_model_name(interaction_model_name)
        info(10, "Generating interaction db. mname={0}".format(mname))
        with h5py.File(self.had_fname, "r") as mceq_db:
            if (
                self.medium not in mceq_db["hadronic_interactions"]
                or mname not in mceq_db["hadronic_interactions"][self.medium]
            ) and config.fallback_to_air_cs:
                self._check_subgroup_exists(
                    mceq_db["hadronic_interactions"]["air"], mname
                )
                info(
                    1,
                    (
                        "Production matrices for {0} in {1} not found."
                        + "Fall-back to air."
                    ).format(mname, self.medium),
                )
                medium = "air"
            else:
                self._check_subgroup_exists(
                    mceq_db["hadronic_interactions"]["air"], mname
                )
                medium = self.medium

            if "SIBYLL21" in mname:
                eqv = equivalences["SIBYLL21"]
            elif "SIBYLL23" in mname:
                eqv = equivalences["SIBYLL23"]
            elif "QGSJET01" in mname:
                eqv = equivalences["QGSJET01"]
            elif "QGSJETII" in mname:
                eqv = equivalences["QGSJETII"]
            elif "DPMJET" in mname:
                eqv = equivalences["DPMJET"]
            elif "EPOSLHC" in mname:
                eqv = equivalences["EPOSLHC"]
            elif "PYTHIA8" in mname:
                eqv = equivalences["PYTHIA8"]
            elif "FLUKA" in mname:
                eqv = equivalences["FLUKA"]
            else:
                raise ValueError("Unknown equivalence table for", mname)
            int_index = self._gen_db_dictionary(
                mceq_db["hadronic_interactions"][medium][mname],
                mceq_db["hadronic_interactions"][medium][mname + "_indptrs"],
                equivalences=eqv,
            )

        # Append electromagnetic interaction matrices from EmCA
        if config.enable_em:
            if medium == "ice":
                info(5, "Electromagnetic cross sections for ice replaced by water.")
                medium = "water"

            with h5py.File(self.em_fname, "r") as em_db:
                info(2, "Injecting EmCA matrices into interaction_db.")
                self._check_subgroup_exists(em_db, "electromagnetic")
                self._check_subgroup_exists(em_db["electromagnetic"], self.medium)
                em_index = self._gen_db_dictionary(
                    em_db["electromagnetic"][self.medium]["emca_mats"],
                    em_db["electromagnetic"][self.medium]["emca_mats" + "_indptrs"],
                )
            if config.muon_helicity_dependence:
                # This is only approximately valid and is done for consistency.
                # Typically electrons would quickly depolarize due to multiple
                # scattering but this requires additional matrices for
                # (-11,1) -> (-11,0) etc. that are not available now.
                from itertools import product

                info(
                    5,
                    "Copy bremsstrahlung and photon emission to "
                    + "polarised electrons and muons.",
                )
                for pid, h in product([11, -11, 13, -13], [-1, 1]):
                    em_index["index_d"][((pid, h), (pid, h))] = em_index["index_d"][
                        ((pid, 0), (pid, 0))
                    ]
                    em_index["index_d"][((pid, h), (22, 0))] = em_index["index_d"][
                        ((pid, 0), (22, 0))
                    ]
                    em_index["parents"].append((pid, h))

                em_index["relations"] = defaultdict(lambda: [])
                em_index["particles"] = []

                for idx_tup in em_index["index_d"]:
                    parent, child = idx_tup
                    em_index["relations"][parent].append(child)
                    em_index["particles"].append(parent)
                    em_index["particles"].append(child)

            int_index["parents"] = sorted(int_index["parents"] + em_index["parents"])
            int_index["particles"] = sorted(
                list(set(int_index["particles"] + em_index["particles"]))
            )
            int_index["relations"].update(em_index["relations"])
            int_index["index_d"].update(em_index["index_d"])

        if int_index["description"] is not None:
            int_index["description"] += "\nInteraction model name: " + mname
        else:
            int_index["description"] = "Interaction model name: " + mname

        return int_index

    def decay_db(self, decay_dset_name):
        info(10, "Generating decay db. dset_name={0}".format(decay_dset_name))

        with h5py.File(self.had_fname, "r") as mceq_db:
            if config.muon_helicity_dependence:
                if decay_dset_name != "polarized":
                    info(
                        0,
                        "Warning: "
                        + f"Does this decay dataset '{decay_dset_name}'"
                        + " include polarization?",
                    )
                else:
                    decay_dset_name = "polarized"
                info(2, "Using helicity dependent decays.")

            self._check_subgroup_exists(mceq_db["decays"], decay_dset_name)
            dec_index = self._gen_db_dictionary(
                mceq_db["decays"][decay_dset_name],
                mceq_db["decays"][decay_dset_name + "_indptrs"],
            )

            # Refresh the metadata after modifying the index
            dec_index["relations"] = defaultdict(lambda: [])
            dec_index["particles"] = []

            for idx_tup in dec_index["index_d"]:
                parent, child = idx_tup
                dec_index["relations"][parent].append(child)
                dec_index["particles"].append(parent)
                dec_index["particles"].append(child)

            dec_index["parents"] = sorted(list(dec_index["relations"]))
            dec_index["particles"] = sorted(list(set(dec_index["particles"])))
        return dec_index

    def cs_db(self, interaction_model_name):
        mname = normalize_hadronic_model_name(interaction_model_name)
        medium = self.medium
        if "SIBYLL23C" in mname or "SIBYLL23DSTAR" in mname:
            info(5, f"{mname} cross sections replaced by 23D.")
            mname = "SIBYLL23D"

        if "DPMJETIII19" in mname or "FLUKA" in mname:
            info(5, f"{mname} cross sections replaced by DPMIII191.")
            mname = "DPMJETIII191"

        if config.adv_set["forced_int_cs"] is not None:
            mname = config.adv_set["forced_int_cs"]
            info(1, "All interaction cross sections forced to", mname)

        if medium == "air-legacy" and "SIBYLL23" not in mname:
            info(5, "air-legacy target replaced by air for", mname)
            medium = "air"
        elif medium == "ice":
            info(5, "ice target replaced by water for", mname)
            medium = "water"

        with h5py.File(self.had_fname, "r") as mceq_db:
            self._check_subgroup_exists(mceq_db["cross_sections"], medium)
            self._check_subgroup_exists(mceq_db["cross_sections"][medium], mname)
            cs_db = mceq_db["cross_sections"][medium][mname]
            cs_data = cs_db[:]
            index_d = {}
            parents = list(cs_db.attrs["projectiles"])
            for ip, p in enumerate(parents):
                index_d[p] = cs_data[self._cuts, ip]

        if config.adv_set["replace_meson_cross_sections_with"] is not None:
            mname_mesons = config.adv_set["replace_meson_cross_sections_with"]
            info(1, "Meson cross sections forced to", mname_mesons)
            with h5py.File(self.had_fname, "r") as mceq_db:
                self._check_subgroup_exists(mceq_db["cross_sections"], medium)
                self._check_subgroup_exists(
                    mceq_db["cross_sections"][medium], mname_mesons
                )
                mes_cs_db = mceq_db["cross_sections"][medium][mname_mesons]
                mes_cs_data = mes_cs_db[:]
                mes_parents = list(mes_cs_db.attrs["projectiles"])
                for ip, p in enumerate(mes_parents):
                    if p in index_d and (100 < abs(p) < 2000):
                        info(1, "Meson cross sections for", p, "replaced.")
                        index_d[p] = mes_cs_data[self._cuts, ip]

        # Append electromagnetic interaction cross sections from EmCA
        if config.enable_em:
            with h5py.File(self.em_fname, "r") as em_db:
                info(2, "Injecting EmCA matrices into interaction_db.")
                self._check_subgroup_exists(em_db, "electromagnetic")
                self._check_subgroup_exists(em_db["electromagnetic"], medium)
                em_cs = em_db["electromagnetic"][medium]["cs"][:]
                em_parents = list(
                    em_db["electromagnetic"][medium]["cs"].attrs["projectiles"]
                )

                for ip, p in enumerate(em_parents):
                    if p in index_d:
                        raise Exception("EM cross sections already in database?")
                    index_d[p] = em_cs[ip, self._cuts]
                parents += em_parents

        return {"parents": parents, "index_d": index_d}

    def continuous_loss_db(self):
        with h5py.File(self.had_fname, "r") as mceq_db:
            self._check_subgroup_exists(mceq_db["continuous_losses"], self.medium)
            if config.enable_em or not config.enable_cont_rad_loss:
                loss_case = "ionization"
            else:
                loss_case = "total"
            cl_db = mceq_db["continuous_losses"][self.medium][loss_case]
            # No rad losses for hadrons implememented
            cl_db_hadrons = mceq_db["continuous_losses"][self.medium]["total"]
            index_d = {}
            generic_dedx = None

            for k in list(cl_db):
                if k != "hadron":
                    for hel in [0, 1, -1]:
                        index_d[(int(k), hel)] = cl_db[k][self._cuts]
                else:
                    # Tuple (boost, dEdx)
                    generic_dedx = (cl_db_hadrons[k][0], cl_db_hadrons[k][1])

            # if config.enable_em:
            #     with h5py.File(self.em_fname, "r") as em_db:
            #         info(2, "Injecting EmCA matrices into interaction_db.")
            #         self._check_subgroup_exists(em_db, "electromagnetic")
            #         for hel in [0, 1, -1]:
            #             index_d[(11, hel)] = em_db["electromagnetic"][self.medium][
            #                 "dEdX 11"
            #             ][self._cuts]
            #             index_d[(-11, hel)] = em_db["electromagnetic"][self.medium][
            #                 "dEdX -11"
            #             ][self._cuts]
        if generic_dedx is not None:
            return {
                "parents": sorted(list(index_d)),
                "index_d": index_d,
                "generic": generic_dedx,
            }
        else:
            return {"parents": sorted(list(index_d)), "index_d": index_d}


class Interactions(object):
    """Class for managing the dictionary of interaction yield matrices.

    Args:
    mceq_hdf_db (object): instance of :class:`MCEq.data.HDF5Backend`
    """

    def __init__(self, mceq_hdf_db):
        from collections import defaultdict

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
        self.mod_pprod = defaultdict(lambda: {})

    def load(self, interaction_model, parent_list=None):
        from MCEq.misc import is_charm_pdgid

        self.iam = normalize_hadronic_model_name(interaction_model)
        # Load tables and index from file
        index = self.mceq_db.interaction_db(self.iam)
        disabled_particles = config.adv_set["disabled_particles"]
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
        if config.adv_set["disable_charm_pprod"]:
            self.parents = [p for p in self.parents if not is_charm_pdgid(p[0])]
        if config.adv_set["disable_interactions_of_unstable"]:
            self.parents = [
                p for p in self.parents if p[0] not in [2212, 2112, -2212, -2112]
            ]
        if config.adv_set["allowed_projectiles"]:
            self.parents = [
                p for p in self.parents if p[0] in config.adv_set["allowed_projectiles"]
            ]

        if config.adv_set["fix_dpmjet_neutral_kaons"] and "DPMJET" in self.iam:
            # Fix bug in DPMJET-III K0 production matrices
            # The numbers for the mixture of K= and K- are obtained
            # from fitting the true zfactors for DPMJET-III with
            # a sum of K+ and K-. The different values are expected
            # from quark counting rules. This bug will be resolved
            # in future versions.
            info(3, "Applying fix for neutral kaons in DPMJET.")
            for p in self.parents:
                if abs(p[0]) < 100:
                    continue
                if p[0] in [2212, 2112]:
                    # From fit to fixed distributions
                    self.index_d[(p, (310, 0))] = 0.5 * (
                        0.84 * self.index_d[(p, (321, 0))]
                        + 1.09 * self.index_d[(p, (-321, 0))]
                    )
                else:
                    # Generic isospin for other primaries
                    self.index_d[(p, (310, 0))] = 0.5 * (
                        self.index_d[(p, (321, 0))] + self.index_d[(p, (-321, 0))]
                    )
                self.index_d[(p, (130, 0))] = np.copy(self.index_d[(p, (310, 0))])

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
        if config.adv_set["disable_direct_leptons"]:
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
        from MCEq.misc import gen_xmat

        info(2, "Generating modification matrix for", x_func.__name__, args)

        xmat = gen_xmat(self.energy_grid)

        # select the relevant slice of interaction matrix
        modmat = x_func(xmat, self.energy_grid.c, *args)
        # Set lower triangular indices to 0. (should be not necessary)
        modmat[np.tril_indices(self.energy_grid.d, -1)] = 0.0

        return asarray(modmat)

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

        if config.use_isospin_sym and prim_pdg not in [2212, 2112]:
            raise Exception("Unsupported primary for isospin symmetries.")

        if (x_func.__name__, args) in mpli[(pstup)]:
            info(
                5,
                " no changes to particle production"
                + " modification matrix of {0}/{1} for {2},{3}".format(
                    prim_pdg, sec_pdg, x_func.__name__, args
                ),
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
            + " matrix of {0}/{1} for {2},{3}".format(
                prim_pdg, sec_pdg, x_func.__name__, args
            ),
        )

        kmat = self._gen_mod_matrix(x_func, *args)
        mpli[pstup][(x_func.__name__, args)] = kmat

        info(5, 'modification "strength"', np.sum(kmat) / np.count_nonzero(kmat))

        if not config.use_isospin_sym:
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
                    "{0}: {1} -> {2}, func: {3}, arg: {4}".format(
                        i + j, prim_pdg, sec_pdg, argname, argv
                    ),
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
            raise Exception(
                ("trying to get empty matrix {0} -> {1}").format(parent, child)
            )

        m = self.index_d[(parent, child)]

        if (
            config.adv_set["disable_leading_mesons"]
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
                "using modified particle production for {0}/{1}".format(
                    parent[0], child[0]
                ),
            )
            i = 0
            m = np.copy(m)
            for args, mmat in six.iteritems(self.mod_pprod[(parent[0], child[0])]):
                info(10, i, (parent[0], child[0]), args, np.sum(mmat), np.sum(m))
                i += 1
                m *= mmat

        return m


class Decays(object):
    """Class for managing the dictionary of decay yield matrices.

    Args:
      mceq_hdf_db (object): instance of :class:`MCEq.data.HDF5Backend`
    """

    def __init__(self, mceq_hdf_db, override_decay_db_name=None):
        #: MCEq HDF5Backend reference
        self.mceq_db = mceq_hdf_db
        #: (list) List of particles in the decay matrices
        self.parent_list = []
        self._default_decay_dset = (
            override_decay_db_name if override_decay_db_name else config.decay_db_name
        )

        if self._default_decay_dset is None:
            if config.muon_helicity_dependence:
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
            raise Exception("Parent {0} not in decay database.".format(parent_pdg))

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
            raise Exception(
                ("trying to get empty matrix {0} -> {1}").format(parent, child)
            )

        return self.index_d[(parent, child)]


class InteractionCrossSections(object):
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

    def __init__(self, mceq_hdf_db, interaction_model="DPMJETIII191"):
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
        """Returns inelastic ``parent``-air cross-section
        :math:`\\sigma_{inel}^{proj-Air}(E)` as vector spanned over
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
        if parent in list(self.index_d):
            cs = self.index_d[parent]
        elif abs(parent) in list(self.index_d):
            cs = self.index_d[abs(parent)]
        elif 100 < abs(parent) < 300 and abs(parent) != 130:
            cs = self.index_d[211]
        elif 300 < abs(parent) < 1000 or abs(parent) in [130, 10313, 10323]:
            info(15, message_templ.format(parent, "K+-"))
            cs = self.index_d[321]
        elif abs(parent) > 1000 and abs(parent) < 5000:
            info(15, message_templ.format(parent, "nucleon"))
            cs = self.index_d[2212]
        elif 5 < abs(parent) < 23:
            info(15, "returning 0 cross-section for lepton", parent)
            return np.zeros(self.energy_grid.d)
        else:
            info(1, "Strange case for parent, using zero cross section.")
            cs = 0.0

        if not mbarn:
            return self.mbarn2cm2 * cs
        else:
            return cs


class ContinuousLosses(object):
    """Class for managing the dictionary of hadron-air cross-sections.

    Args:
      mceq_hdf_db (object): instance of :class:`MCEq.data.HDF5Backend`
      material (str): name of the material (not fully implemented)
    """

    def __init__(self, mceq_hdf_db):
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
        if "generic" not in index and config.generic_losses_all_charged:
            raise Exception("New data file needed to support generic losses.")
        self.generic_spl = InterpolatedUnivariateSpline(
            np.log(index["generic"][0]), np.log(index["generic"][1]), k=1
        )
