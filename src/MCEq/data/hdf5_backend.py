"""The HDF5 backend: the consumer classes' view of the cross-section tables.

The materialized numpy reads are :mod:`MCEq.data.hdf5_store`; the EM
selection policy is :mod:`MCEq.data.em_tables`. This module owns the class
that opens the databases, decodes channel packs into interaction
dictionaries, and applies the hadronic-model / medium / low-energy
blending policy on top. Re-exported as ``MCEq.data.HDF5Backend``.
"""

from collections import defaultdict
from os.path import isfile, join

import numpy as np

from MCEq.data import em_tables, hdf5_store
from MCEq.data.blending import blend_cross_sections, blend_yields, he_le_weight
from MCEq.data.energy_grid import EnergyGrid, _eval_energy_cuts
from MCEq.data.equivalences import (
    apply_equivalences,
    equivalences,
    reverse_equivalences,
)
from MCEq.data.model_names import family_of, normalize_hadronic_model_name
from MCEq.misc import info


class HDF5Backend:
    """Provides access to tabulated data stored in an HDF5 file.

    The file contains all necessary ingredients to run MCEq, i.e. no
    other files are required. This database is not maintained in git
    and it will change infrequently.

    ``paths``, ``grid``, ``physics`` and ``em`` are the settings groups this
    backend reads from; a group left as ``None`` falls back to the live view
    MCEq.config publishes, which reads the same flat names as before.
    """

    def __init__(
        self,
        medium=None,
        low_energy_model=None,
        he_le_transition=80.0,
        he_le_trwidth=0.3,
        paths=None,
        grid=None,
        physics=None,
        em=None,
    ):
        from MCEq import config

        # A group left as None resolves to the live view on MCEq.config, so an
        # un-injected backend reads exactly what it read before, per read.
        paths = config.paths if paths is None else paths
        grid = self._grid = config.grid if grid is None else grid
        physics = self._physics = config.physics if physics is None else physics
        self._em = config.em if em is None else em
        # Resolved here rather than as a signature default: a default would bind
        # when this module is first imported, which happens lazily from
        # MCEqRun.__init__, so whether a caller's `config.interaction_medium`
        # was seen depended on whether it was written before that import.
        medium = physics.interaction_medium if medium is None else medium

        info(2, "Opening HDF5 file", paths.mceq_db_fname)
        self.had_fname = join(paths.data_dir, paths.mceq_db_fname)
        if not isfile(self.had_fname):
            raise Exception(
                f'MCEq DB file {paths.mceq_db_fname} not found in "data" directory.'
            )
        self._had = hdf5_store.HDF5Store(self.had_fname)

        self.em_fname = join(paths.data_dir, paths.em_db_fname)
        if physics.enable_em and not isfile(self.em_fname):
            _n = paths.em_db_fname
            raise Exception(
                f'Electromagnetic DB file {_n} not found in "data" directory.'
            )
        # Constructed unconditionally even though the EM file may be absent
        # (and must stay absent-constructible): HDF5Store never opens.
        self._em_store = hdf5_store.HDF5Store(self.em_fname)

        # DIAGNOSTIC (em-20dec-binning-test): when grid.em_standalone_grid is
        # set, the energy grid is taken from the EM DB instead of the hadronic
        # DB, so the EM cascade can run on a finer bins/decade grid than the
        # (10/dec) hadronic DB. The inert hadronic interaction/decay matrices
        # are then skipped and the e± ionization continuous-loss curve is
        # interpolated onto the EM grid (see interaction_db / decay_db / cs_db /
        # continuous_loss_db below). Default off → standard behaviour unchanged.
        self._em_standalone = bool(grid.em_standalone_grid)
        grid_fname = self.em_fname if self._em_standalone else self.had_fname

        self.version = self._had.attrs().get("version", "1.0.0")

        ca = hdf5_store.HDF5Store(grid_fname).attrs("common")
        self._e_grid_full = np.asarray(ca["e_grid"])
        self.min_idx, self.max_idx, self._cuts = _eval_energy_cuts(
            ca["e_grid"], grid.e_min, grid.e_max
        )

        self._energy_grid = EnergyGrid(
            ca["e_grid"][self._cuts],
            ca["e_bins"][self.min_idx : self.max_idx + 1],
            ca["widths"][self._cuts],
            int(self.max_idx - self.min_idx),
        )

        # 2D databases are detected by the ``k_dim`` attribute on the
        # ``common`` group; there is no config flag.
        self.is_2d = "k_dim" in ca
        if self.is_2d:
            self.n_k = int(ca["k_dim"])
            self.k_grid = np.asarray(ca["k_grid"])
        else:
            self.n_k = 1
            self.k_grid = np.asarray([0])

        # Full CSR dimension: in 2D it spans n_k Hankel modes * energy grid.
        if self.is_2d:
            self.dim_full = int(ca["e_dim"]) * self.n_k
        else:
            self.dim_full = int(ca["e_dim"])

        self.medium = medium
        self.low_energy_model = (
            normalize_hadronic_model_name(low_energy_model)
            if low_energy_model is not None
            else None
        )
        self.he_le_transition = float(he_le_transition)
        self.he_le_trwidth = float(he_le_trwidth)
        if self.he_le_transition <= 0.0:
            raise ValueError("he_le_transition must be positive")
        if self.he_le_trwidth < 0.0:
            raise ValueError("he_le_trwidth must be non-negative")

    @property
    def energy_grid(self):
        return self._energy_grid

    def _gen_db_dictionary(self, pack, equivalences={}):
        index_d = {}
        relations = defaultdict(list)
        particle_list = []
        description = pack.description
        mat_data = np.asarray(pack.data, dtype=self._grid.dtype)
        indptr_data = np.asarray(pack.indptrs)
        len_data = pack.len_data
        tuple_idcs = pack.tuple_idcs
        if tuple_idcs.shape[1] == 4:
            model_particles = sorted(
                list(set(tuple_idcs[:, (0, 2)].flatten().tolist()))
            )
        else:
            model_particles = sorted(list(set(tuple_idcs.flatten().tolist())))

        exclude = self._physics.filters["disabled_particles"]
        read_idx = 0
        available_parents = [(pdg, parity) for (pdg, parity) in (tuple_idcs[:, :2])]
        available_parents = sorted(list(set(available_parents)))

        # Reverse equivalences
        eqv_lookup = reverse_equivalences(equivalences)

        for tupidx, tup in enumerate(tuple_idcs):
            # In 2D each channel stores n_k Hankel-mode blocks back-to-back,
            # so the flat read length per channel scales with n_k.
            if self.is_2d:
                expand_len = self.n_k
            else:
                expand_len = 1
            # Helicity information handling
            if len(tup) == 4:
                parent_pdg, child_pdg = tuple(tup[:2]), tuple(tup[2:])
            elif len(tup) == 2:
                parent_pdg, child_pdg = (tup[0], 0), (tup[1], 0)
            else:
                raise Exception("Failed decoding parent-child relation.")

            if (abs(parent_pdg[0]) in exclude) or (abs(child_pdg[0]) in exclude):
                read_idx += expand_len * len_data[tupidx]
                continue
            parent_pdg = int(parent_pdg[0]), (parent_pdg[1])
            child_pdg = int(child_pdg[0]), (child_pdg[1])

            particle_list.append(parent_pdg)
            particle_list.append(child_pdg)
            index_d[(parent_pdg, child_pdg)] = hdf5_store.unpack_channel(
                mat_data,
                indptr_data[tupidx, :],
                read_idx,
                expand_len * len_data[tupidx],
                dim_full=self.dim_full,
                is_2d=self.is_2d,
                n_k=self.n_k,
                min_idx=self.min_idx,
                max_idx=self.max_idx,
                cuts=self._cuts,
            )

            relations[parent_pdg].append(child_pdg)

            info(
                20,
                f"This parent {parent_pdg[0]} is used for interactions of",
                [p[0] for p in eqv_lookup[parent_pdg]],
                condition=len(equivalences) > 0,
            )
            if self._physics.assume_nucleon_interactions_for_exotics:
                # Aliases, deliberately: the stand-in shares this channel's
                # matrix object and this parent's `relations` list, which later
                # iterations of this loop keep appending to.
                apply_equivalences(
                    eqv_lookup,
                    parent_pdg,
                    child_pdg,
                    model_particles,
                    available_parents,
                    particle_list,
                    index_d,
                    relations,
                )

            read_idx += expand_len * len_data[tupidx]

        return {
            "parents": sorted(list(relations)),
            "particles": sorted(list(set(particle_list))),
            "relations": dict(relations),
            "index_d": dict(index_d),
            "description": description,
        }

    def _check_subgroup_exists(self, store, path, mname):
        available_models = [m for m in store.members(path) if "indptrs" not in m]
        if mname not in available_models:
            info(0, "Invalid choice/model", mname)
            info(0, "Choose from:\n", "\n".join(available_models))
            raise Exception("Unknown selections.")

    def _he_le_weight(self):
        """:func:`MCEq.data.blending.he_le_weight` on this backend's settings.

        Reads the dtype off the injected grid group; ``__init__`` always
        binds it, so there is no live-config fallback here (the
        ``object.__new__`` shells that needed one are gone).
        """
        return he_le_weight(
            self._energy_grid.c,
            self.he_le_transition,
            self.he_le_trwidth,
            self._grid.dtype,
        )

    def _blend_interaction_dbs(self, he_index, le_index, he_name, le_name):
        """:func:`MCEq.data.blending.blend_yields` on this backend's settings."""
        return blend_yields(
            he_index,
            le_index,
            he_name,
            le_name,
            self._he_le_weight(),
            self.he_le_transition,
            self.he_le_trwidth,
        )

    def interaction_db(self, interaction_model_name):
        mname = normalize_hadronic_model_name(interaction_model_name)
        if self.low_energy_model is None or mname == self.low_energy_model:
            return self._interaction_db_single(mname)
        he_index = self._interaction_db_single(mname)
        le_index = self._interaction_db_single(self.low_energy_model)
        return self._blend_interaction_dbs(
            he_index, le_index, mname, self.low_energy_model
        )

    def _interaction_db_single(self, interaction_model_name):
        mname = normalize_hadronic_model_name(interaction_model_name)
        info(10, f"Generating interaction db. mname={mname}")
        had = "hadronic_interactions"
        if (
            not self._had.has(f"{had}/{self.medium}")
            or not self._had.has(f"{had}/{self.medium}/{mname}")
        ) and self._physics.fallback_to_air_cs:
            self._check_subgroup_exists(self._had, f"{had}/air", mname)
            info(
                1,
                (
                    f"Production matrices for {mname} in {self.medium} not found."
                    + "Fall-back to air."
                ),
            )
            medium = "air"
        else:
            self._check_subgroup_exists(self._had, f"{had}/{self.medium}", mname)
            medium = self.medium

        family = family_of(mname)
        if family is None:
            raise ValueError("Unknown equivalence table for", mname)
        eqv = equivalences[family]
        if self._em_standalone:
            # Hadronic matrices live on the (coarser) hadronic grid; they
            # are inert for a γ/e± cascade. Skip them so they never hit the
            # dim_full reshape against the EM grid.
            int_index = {
                "parents": [],
                "particles": [],
                "relations": {},
                "index_d": {},
                "description": None,
            }
        else:
            int_index = self._gen_db_dictionary(
                self._had.read_channel_pack(f"{had}/{medium}", mname),
                equivalences=eqv,
            )

        # Append electromagnetic interaction matrices from the EM database
        if self._physics.enable_em:
            medium = em_tables.medium_for_em(medium)

            info(2, "Injecting EM matrices into interaction_db.")
            self._check_subgroup_exists(self._em_store, "/", "electromagnetic")
            self._check_subgroup_exists(self._em_store, "electromagnetic", self.medium)
            em_group = em_tables.emca_group_path(
                self._em_store, self.medium, "interaction_db", self._em.air_density
            )
            em_index = self._gen_db_dictionary(
                self._em_store.read_channel_pack(em_group, "emca_mats"),
            )
            if self._physics.muon_helicity_dependence:
                em_tables.helicity_duplicate(em_index)

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
        info(10, f"Generating decay db. dset_name={decay_dset_name}")

        if self._em_standalone:
            # Decay matrices live on the hadronic grid and are inert for a
            # γ/e± cascade (the only decaying secondaries are sub-permille
            # muons from γ→μ⁺μ⁻). Skip them.
            return {
                "parents": [],
                "particles": [],
                "relations": defaultdict(list),
                "index_d": {},
                "description": None,
            }

        if self._physics.muon_helicity_dependence:
            if decay_dset_name != "polarized":
                info(
                    0,
                    "Warning: "
                    + f"Does this decay dataset '{decay_dset_name}'"
                    + " include polarization?",
                )
            info(2, "Using helicity dependent decays.")

        self._check_subgroup_exists(self._had, "decays", decay_dset_name)

        if self.is_2d and decay_dset_name == "polarized":
            # 2D databases declare ``layout='superset'`` on the
            # ``polarized`` dataset: the full channel set including the
            # helicity-resolved muon entries, loadable directly. A 2D
            # database without this attribute stores only a helicity
            # overlay (an unsupported legacy layout) and must be rebuilt
            # with current mceq-maintenance-tools.
            layout = self._had.attrs("decays/polarized").get("layout", None)
            if isinstance(layout, bytes):
                layout = layout.decode()
            if layout != "superset":
                raise RuntimeError(
                    f"2D database '{self.had_fname}': the 'polarized' "
                    "decay dataset does not declare layout='superset'. "
                    "Legacy delta-layout 2D databases are not supported; "
                    "rebuild the database with current "
                    "mceq-maintenance-tools."
                )
        dec_index = self._gen_db_dictionary(
            self._had.read_channel_pack("decays", decay_dset_name),
        )
        return dec_index

    def cs_db(self, interaction_model_name):
        mname = normalize_hadronic_model_name(interaction_model_name)
        if self.low_energy_model is None or mname == self.low_energy_model:
            return self._cs_db_single(mname)

        he_index = self._cs_db_single(mname)
        le_index = self._cs_db_single(self.low_energy_model)
        return blend_cross_sections(
            he_index,
            le_index,
            mname,
            self.low_energy_model,
            self._he_le_weight(),
        )

    def _cs_db_single(self, interaction_model_name):
        mname = normalize_hadronic_model_name(interaction_model_name)
        medium = self.medium
        if "SIBYLL23C" in mname or "SIBYLL23DSTAR" in mname:
            info(5, f"{mname} cross sections replaced by 23D.")
            mname = "SIBYLL23D"

        # Modern databases carry native FLUKA cross sections. Preserve the
        # historical DPMJET fallback only for older files that do not.
        if "FLUKA" in mname:
            cs_root = "cross_sections"
            direct_medium = medium if self._had.has(f"{cs_root}/{medium}") else None
            direct = direct_medium is not None and self._had.has(
                f"{cs_root}/{direct_medium}/{mname}"
            )
            if (
                not direct
                and self._physics.fallback_to_air_cs
                and self._had.has(f"{cs_root}/air")
            ):
                if self._had.has(f"{cs_root}/air/{mname}"):
                    medium = "air"
                    direct = True
            if not direct:
                for fallback in ("DPMJETIII191", "DPMJETIII193"):
                    if direct_medium is not None and self._had.has(
                        f"{cs_root}/{direct_medium}/{fallback}"
                    ):
                        info(5, f"{mname} cross sections replaced by {fallback}.")
                        mname = fallback
                        break

        filters = self._physics.filters
        if filters["forced_int_cs"] is not None:
            mname = filters["forced_int_cs"]
            info(1, "All interaction cross sections forced to", mname)

        if medium == "air-legacy" and "SIBYLL23" not in mname:
            info(5, "air-legacy target replaced by air for", mname)
            medium = "air"

        index_d = {}
        parents = []
        if not self._em_standalone:
            self._check_subgroup_exists(self._had, "cross_sections", medium)
            self._check_subgroup_exists(self._had, f"cross_sections/{medium}", mname)
            cs_data, cs_attrs = self._had.read_table(f"cross_sections/{medium}/{mname}")
            if "parents" not in cs_attrs:
                raise RuntimeError(
                    f"Cross-section table '{medium}/{mname}' in "
                    f"'{self.had_fname}' has no 'parents' attribute. "
                    "Legacy databases using 'projectiles' are not "
                    "supported; rebuild the database with current "
                    "mceq-maintenance-tools."
                )
            parents = list(cs_attrs["parents"])
            for ip, p in enumerate(parents):
                index_d[p] = cs_data[self._cuts, ip]

        if filters["replace_meson_cross_sections_with"] is not None:
            mname_mesons = filters["replace_meson_cross_sections_with"]
            info(1, "Meson cross sections forced to", mname_mesons)
            self._check_subgroup_exists(self._had, "cross_sections", medium)
            self._check_subgroup_exists(
                self._had, f"cross_sections/{medium}", mname_mesons
            )
            mes_cs_data, mes_attrs = self._had.read_table(
                f"cross_sections/{medium}/{mname_mesons}"
            )
            mes_parents = list(mes_attrs["parents"])
            for ip, p in enumerate(mes_parents):
                if p in index_d and (100 < abs(p) < 2000):
                    info(1, "Meson cross sections for", p, "replaced.")
                    index_d[p] = mes_cs_data[self._cuts, ip]

        # Append electromagnetic interaction cross sections from the EM database
        if self._physics.enable_em:
            info(2, "Injecting EM matrices into interaction_db.")
            self._check_subgroup_exists(self._em_store, "/", "electromagnetic")
            self._check_subgroup_exists(self._em_store, "electromagnetic", medium)
            em_cs_group = em_tables.em_cs_group_path(
                self._em_store, medium, "cs_db", self._em.air_density
            )
            em_cs, em_cs_attrs = self._em_store.read_table(f"{em_cs_group}/cs")
            em_parents = list(em_cs_attrs["projectiles"])

            for ip, p in enumerate(em_parents):
                if p in index_d:
                    raise Exception("EM cross sections already in database?")
                index_d[p] = em_cs[ip, self._cuts]
            parents += em_parents

        return {"parents": parents, "index_d": index_d}

    def continuous_loss_db(self):
        self._check_subgroup_exists(self._had, "continuous_losses", self.medium)
        if self._physics.enable_em or not self._physics.enable_cont_rad_loss:
            loss_case = "ionization"
        else:
            loss_case = "total"
        self._check_subgroup_exists(
            self._had, f"continuous_losses/{self.medium}", loss_case
        )
        cl_db = self._had.read_group_datasets(
            f"continuous_losses/{self.medium}/{loss_case}"
        )
        # No radiative losses for hadrons implemented
        cl_db_hadrons = self._had.read_group_datasets(
            f"continuous_losses/{self.medium}/total"
        )
        index_d = {}
        generic_dedx = None

        # In em_standalone mode the loss curves are stored on the hadronic
        # grid; interpolate them onto the (finer) EM grid before applying
        # the energy cuts. dE/dX is smooth (Bethe-Bloch) and stored with a
        # negative sign, so interpolate the value linearly in log(E) — a
        # log-log interpolation would take log of a negative number.
        had_eg = (
            np.asarray(self._had.attrs("common")["e_grid"])
            if self._em_standalone
            else None
        )

        for k in cl_db:
            if k != "hadron":
                if self._em_standalone:
                    dedx = np.interp(
                        np.log(self._e_grid_full),
                        np.log(had_eg),
                        np.asarray(cl_db[k]),
                    )[self._cuts]
                else:
                    dedx = cl_db[k][self._cuts]
                for hel in [0, 1, -1]:
                    index_d[(int(k), hel)] = dedx
            else:
                # Tuple (boost, dEdx)
                generic_dedx = (cl_db_hadrons[k][0], cl_db_hadrons[k][1])

        if generic_dedx is not None:
            return {
                "parents": sorted(list(index_d)),
                "index_d": index_d,
                "generic": generic_dedx,
            }
        return {"parents": sorted(list(index_d)), "index_d": index_d}
