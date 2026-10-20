"""HDF5 database backend: reads, decodes and blends interaction, decay, cross-section and loss tables."""

import json
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from os.path import isfile, join
from pathlib import Path

import numpy as np

from MCEq import config
from MCEq.data import download, hdf5_store
from MCEq.data.hadronic_models import (
    equivalences_of,
    mapped_cross_section,
    reverse_equivalences,
)
from MCEq.misc import (
    _eval_energy_cuts,
    energy_grid,
    info,
    normalize_hadronic_model_name,
)


class HDF5Backend:
    """Provides access to tabulated data stored in an HDF5 file.

    Reads the hadronic database and, when EM interactions are enabled, a
    separate EM database. ``medium=None`` means ``config.interaction_medium``.
    """

    @contextmanager
    def _read_session(self):
        """Scope hadronic and enabled EM reads to one handle per file."""
        with ExitStack() as stack:
            stack.enter_context(self._had.session())
            if config.enable_em:
                stack.enter_context(self._em_store.session())
            yield

    def scattering_composition(self):
        """Elemental composition of the medium, rows ``(Z, A [g/mol], atom fraction)``.

        Read from ``material_scattering/<medium>/composition``; the screened
        Coulomb scattering model needs it.
        """
        group = f"material_scattering/{self.medium}"
        if not self._had.has(f"{group}/composition"):
            raise ValueError(
                f"{group}/composition is absent from {self.had_fname}. Use a "
                "database with this medium or set "
                "config.muon_scattering_model = 'gaussian'."
            )
        comp = np.asarray(self._had.read_group_datasets(group)["composition"], float)
        if (
            comp.ndim != 2
            or comp.shape[1] != 3
            or not np.all(np.isfinite(comp))
            or np.any(comp <= 0)
            or np.any(comp[:, 0] != np.round(comp[:, 0]))
            or not np.isclose(comp[:, 2].sum(), 1.0, rtol=0, atol=1e-6)
        ):
            raise ValueError(f"Invalid composition at {group}/composition")
        return comp

    def scattering_rate_table(self):
        """``(rates, l_grid, e_grid, attrs)`` of the precomputed screened
        Coulomb rates of the medium cut to the energy window, or None.

        ``material_scattering/<medium>/screened_coulomb_rates`` (``(n_k, n_e)``
        on the file's grid, its inputs as attributes for ``scattering.mode_rates``
        to check), read once per medium: rebuilds then reopen nothing.
        """
        cache = self.__dict__.setdefault("_scattering_rate_tables", {})
        if self.medium not in cache:
            path = f"material_scattering/{self.medium}/screened_coulomb_rates"
            got, cache[self.medium] = self._had.read_optional_table(path), None
            if got is not None:
                rates, attrs, common = got
                l_grid = np.asarray(common.get("k_grid", []), dtype=float)
                if np.shape(rates) == (l_grid.size, np.size(common["e_grid"])):
                    cut = rates[:, self._cuts]
                    cache[self.medium] = cut, l_grid, self._energy_grid.c, attrs
        return cache[self.medium]

    def __init__(
        self,
        medium=None,
        low_energy_model=None,
        he_le_transition=80.0,
        he_le_trwidth=0.3,
    ):
        medium = config.interaction_medium if medium is None else medium

        info(2, "Opening HDF5 file", config.mceq_db_fname)
        self.had_fname = join(config.data_dir, config.mceq_db_fname)
        if not isfile(self.had_fname):
            raise Exception(
                f'MCEq DB file {config.mceq_db_fname} not found in "data" directory.'
            )
        self._had = hdf5_store.HDF5Store(self.had_fname)
        # A release package (root attr ``mceq_db_manifest``) resolves models
        # it lacks through the release manifest; a monolithic file does not.
        self._packages = (
            ModelStoreCache(self._had)
            if "mceq_db_manifest" in self._had.attrs()
            else None
        )

        self.em_fname = join(config.data_dir, config.em_db_fname)
        if config.enable_em and not isfile(self.em_fname):
            _n = config.em_db_fname
            raise Exception(
                f'Electromagnetic DB file {_n} not found in "data" directory.'
            )
        self._em_store = hdf5_store.HDF5Store(self.em_fname)

        with self._read_session():
            # A pure electromagnetic database may use its own energy grid.
            self._em_standalone = bool(config.em_standalone_grid)
            grid_store = self._em_store if self._em_standalone else self._had

            self.version = self._had.attrs().get("version", "1.0.0")

            ca = grid_store.attrs("common")
            self._e_grid_full = np.asarray(ca["e_grid"])
            self.min_idx, self.max_idx, self._cuts = _eval_energy_cuts(
                ca["e_grid"], config.e_min, config.e_max
            )

            self._energy_grid = energy_grid(
                ca["e_grid"][self._cuts],
                ca["e_bins"][self.min_idx : self.max_idx + 1],
                ca["widths"][self._cuts],
                int(self.max_idx - self.min_idx),
            )

            # 2D databases carry ``k_dim`` on the ``common`` group. Their
            # modes are the Legendre moments <P_l(cos theta)> with l = k_grid.
            self.is_2d = "k_dim" in ca
            self.angular_basis = hdf5_store.angular_basis(ca, grid_store.fname)
            if self.is_2d:
                self.n_k = int(ca["k_dim"])
                self.k_grid = np.asarray(ca["k_grid"])
            else:
                self.n_k = 1
                self.k_grid = np.asarray([0])

            # Full CSR dimension: in 2D it spans n_k modes * energy grid.
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

    def _gen_db_dictionary(self, pack, eqv=None):
        index_d = {}
        relations = defaultdict(list)
        particle_list = []
        description = pack.description
        mat_data = np.asarray(pack.data, dtype=config.floatlen)
        indptr_data = np.asarray(pack.indptrs)
        len_data = pack.len_data
        tuple_idcs = pack.tuple_idcs
        if tuple_idcs.shape[1] == 4:
            model_particles = sorted(
                list(set(tuple_idcs[:, (0, 2)].flatten().tolist()))
            )
        else:
            model_particles = sorted(list(set(tuple_idcs.flatten().tolist())))

        exclude = config.adv_set["disabled_particles"]
        read_idx = 0
        available_parents = [(pdg, parity) for (pdg, parity) in (tuple_idcs[:, :2])]
        available_parents = sorted(list(set(available_parents)))

        # Reverse equivalences
        eqv_lookup = reverse_equivalences(eqv or {})

        for tupidx, tup in enumerate(tuple_idcs):
            # In 2D each channel stores n_k mode blocks back-to-back,
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
                condition=bool(eqv),
            )
            if config.assume_nucleon_interactions_for_exotics:
                # A stand-in shares this parent's matrix and `relations` list.
                for eqv_parent in eqv_lookup[parent_pdg]:
                    if eqv_parent[0] not in model_particles:
                        continue
                    if eqv_parent in available_parents:
                        info(10, f"Parent {eqv_parent[0]} has dedicated simulation.")
                        continue
                    particle_list.append(eqv_parent)
                    index_d[(eqv_parent, child_pdg)] = index_d[(parent_pdg, child_pdg)]
                    relations[eqv_parent] = relations[parent_pdg]
                    info(
                        15,
                        f"equivalence of {eqv_parent[0]} and {parent_pdg[0]} interactions",
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

    def _model_store(self, mname, medium, tree):
        """Store holding ``tree/<medium>/<mname>``: the primary file, then the
        manifest-listed release packages, else None."""
        if self._had.has(f"{tree}/{medium}/{mname}"):
            return self._had
        if self._packages is not None:
            return self._packages.get(mname, medium, tree)
        return None

    def has_interaction_model(self, mname):
        """True if yields of ``mname`` exist for this medium, in the primary
        file or a release package; the air fallback does not count."""
        mname = normalize_hadronic_model_name(mname)
        return self._model_store(mname, self.medium, "hadronic_interactions") is not None

    def _resolve_interaction_store(self, mname):
        """``(store, medium)`` of the yield pack of ``mname`` in package mode,
        with the air fallback."""
        had = "hadronic_interactions"
        store = self._model_store(mname, self.medium, had)
        if store is not None:
            return store, self.medium
        if config.fallback_to_air_cs and self.medium != "air":
            store = self._model_store(mname, "air", had)
            if store is not None:
                info(
                    1,
                    (
                        f"Production matrices for {mname} in {self.medium} not found."
                        + "Fall-back to air."
                    ),
                )
                return store, "air"
        raise missing_model_error(self._packages, mname, self.medium)

    def _he_le_weight(self):
        """:func:`he_le_weight` on this backend's settings."""
        return he_le_weight(
            self._energy_grid.c,
            self.he_le_transition,
            self.he_le_trwidth,
            config.floatlen,
        )

    def _blend_interaction_dbs(self, he_index, le_index, he_name, le_name):
        """:func:`blend_yields` on this backend's settings."""
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
        blended = self._blend_interaction_dbs(
            he_index, le_index, mname, self.low_energy_model
        )
        # A blended column is bin-averaged only where both models are.
        he_ba, le_ba = he_index.get("bin_average"), le_index.get("bin_average")
        blended["bin_average"] = (
            (he_ba[0], he_ba[1] & le_ba[1])
            if he_ba and le_ba and he_ba[0] == le_ba[0]
            else None
        )
        return blended

    def _interaction_db_single(self, interaction_model_name):
        mname = normalize_hadronic_model_name(interaction_model_name)
        info(10, f"Generating interaction db. mname={mname}")
        had = "hadronic_interactions"
        store = self._had
        if self._packages is not None:
            if self._em_standalone:
                # A photon/electron cascade uses no hadronic interactions,
                # so a database without the model must still load.
                medium = self.medium
            else:
                store, medium = self._resolve_interaction_store(mname)
        elif (
            not self._had.has(f"{had}/{self.medium}")
            or not self._had.has(f"{had}/{self.medium}/{mname}")
        ) and config.fallback_to_air_cs:
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

        eqv = equivalences_of(mname)
        if eqv is None:
            raise ValueError("Unknown equivalence table for", mname)
        if self._em_standalone:
            # Hadronic matrices are on the hadronic grid and unused in a
            # γ/e± cascade on the EM grid.
            int_index = {
                "parents": [],
                "particles": [],
                "relations": {},
                "index_d": {},
                "description": None,
            }
        else:
            int_index = self._gen_db_dictionary(
                store.read_channel_pack(f"{had}/{medium}", mname),
                eqv=eqv,
            )
            int_index["bin_average"] = self._bin_average_marks(
                store.attrs(f"{had}/{medium}/{mname}"),
                "bin_average_projectiles",
                eqv if config.assume_nucleon_interactions_for_exotics else None,
            )

        # Append electromagnetic interaction matrices from the EM database
        if config.enable_em:
            info(2, "Injecting EM matrices into interaction_db.")
            self._check_subgroup_exists(self._em_store, "/", "electromagnetic")
            self._check_subgroup_exists(self._em_store, "electromagnetic", self.medium)
            em_index = self._gen_db_dictionary(
                self._em_store.read_channel_pack(
                    f"electromagnetic/{self.medium}", "emca_mats"
                ),
            )

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
            # Decay matrices are on the hadronic grid and negligible for a
            # γ/e± cascade (only muons from γ→μ⁺μ⁻ decay).
            return {
                "parents": [],
                "particles": [],
                "relations": defaultdict(list),
                "index_d": {},
                "description": None,
            }

        if config.muon_helicity():
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
            # The 2D ``polarized`` dataset must hold the full channel set
            # including helicity-resolved muons (``layout='superset'``).
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
        dec_index["bin_average"] = self._bin_average_marks(
            self._had.attrs("decays"), "bin_average_parents"
        )
        return dec_index

    def _bin_average_marks(self, attrs, key, equivalences=None):
        """``(radius, {pdg})`` of a stencil-averaged table, ``None`` without marks.

        Equivalence stand-ins borrow the marks of the parent whose tables they
        use.
        """
        if not config.bin_average_rates or ("bin_average_radius" not in attrs):
            return None
        pdgs = {int(p) for p in attrs.get(key, ())}
        for stand_in, aliases in reverse_equivalences(equivalences or {}).items():
            if stand_in[0] in pdgs:
                pdgs |= {a[0] for a in aliases}
        return int(attrs["bin_average_radius"]), pdgs

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

        # FLUKA cross sections fall back to DPMJET for files without them.
        if "FLUKA" in mname:
            cs_root = "cross_sections"
            direct = self._model_store(mname, medium, cs_root) is not None
            if (
                not direct
                and config.fallback_to_air_cs
                and self._model_store(mname, "air", cs_root) is not None
            ):
                medium = "air"
                direct = True
            if not direct:
                for fallback in ("DPMJETIII191", "DPMJETIII193"):
                    if self._model_store(fallback, medium, cs_root) is not None:
                        info(5, f"{mname} cross sections replaced by {fallback}.")
                        mname = fallback
                        break

        filters = config.adv_set
        if filters["forced_int_cs"] is not None:
            mname = filters["forced_int_cs"]
            info(1, "All interaction cross sections forced to", mname)

        if medium == "air-legacy" and "SIBYLL23" not in mname:
            info(5, "air-legacy target replaced by air for", mname)
            medium = "air"

        index_d = {}
        parents = []
        if not self._em_standalone:
            cs_store = self._model_store(mname, medium, "cross_sections")
            if cs_store is None:
                if self._packages is not None:
                    raise missing_model_error(self._packages, mname, medium)
                self._check_subgroup_exists(self._had, "cross_sections", medium)
                self._check_subgroup_exists(
                    self._had, f"cross_sections/{medium}", mname
                )
                raise Exception("Unknown selections.")
            cs_data, cs_attrs = cs_store.read_table(f"cross_sections/{medium}/{mname}")
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
            mes_store = self._model_store(mname_mesons, medium, "cross_sections")
            if mes_store is None:
                if self._packages is not None:
                    raise missing_model_error(self._packages, mname_mesons, medium)
                self._check_subgroup_exists(self._had, "cross_sections", medium)
                self._check_subgroup_exists(
                    self._had, f"cross_sections/{medium}", mname_mesons
                )
                raise Exception("Unknown selections.")
            mes_cs_data, mes_attrs = mes_store.read_table(
                f"cross_sections/{medium}/{mname_mesons}"
            )
            mes_parents = list(mes_attrs["parents"])
            for ip, p in enumerate(mes_parents):
                if p in index_d and (100 < abs(p) < 2000):
                    info(1, "Meson cross sections for", p, "replaced.")
                    index_d[p] = mes_cs_data[self._cuts, ip]

        # Append electromagnetic interaction cross sections from the EM database
        if config.enable_em:
            info(2, "Injecting EM cross sections into cs_db.")
            self._check_subgroup_exists(self._em_store, "/", "electromagnetic")
            self._check_subgroup_exists(self._em_store, "electromagnetic", medium)
            em_cs, em_cs_attrs = self._em_store.read_table(
                f"electromagnetic/{medium}/cs"
            )
            em_parents = list(em_cs_attrs["projectiles"])

            for ip, p in enumerate(em_parents):
                if p in index_d:
                    raise Exception("EM cross sections already in database?")
                index_d[p] = em_cs[ip, self._cuts]
            parents += em_parents

        return {"parents": parents, "index_d": index_d}

    def continuous_loss_db(self):
        self._check_subgroup_exists(self._had, "continuous_losses", self.medium)
        if config.enable_em or not config.enable_cont_rad_loss:
            loss_case = "ionization"
        else:
            loss_case = "total"
        self._check_subgroup_exists(
            self._had, f"continuous_losses/{self.medium}", loss_case
        )
        cl_db = self._had.read_group_datasets(
            f"continuous_losses/{self.medium}/{loss_case}"
        )
        # The generic hadron curve: the total loss (hadrons have no radiative
        # part), or the ionization loss when the EM cascade is on.
        cl_db_hadrons = (
            cl_db
            if config.enable_em
            else self._had.read_group_datasets(f"continuous_losses/{self.medium}/total")
        )
        index_d = {}
        generic_dedx = None

        # In em_standalone mode the loss curves are on the hadronic grid;
        # interpolate onto the EM grid before the energy cuts. dE/dX is
        # stored negative, so interpolate linearly in log(E).
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


def he_le_weight(energy, transition, trwidth, dtype):
    """High-energy model weight on the active kinetic-energy grid.

    Args:
        energy: energy grid centers in GeV (``energy_grid.c``).
        transition: HE/LE transition energy in GeV.
        trwidth: full 10--90 % width of the transition in decades; ``0.0``
            makes the weight a hard switch at ``transition``.
        dtype: weight dtype; ``None`` gives float64.
    """
    energy = np.asarray(energy, dtype=float)
    if trwidth == 0.0:
        return (energy >= transition).astype(dtype)
    # Define trwidth as the full 10--90% width in log10 energy.
    arg = 2.0 * np.log(9.0) * np.log10(energy / transition) / trwidth
    arg = np.clip(arg, -700.0, 700.0)
    return np.asarray(1.0 / (1.0 + np.exp(-arg)), dtype=dtype)


def blend_yields(he_index, le_index, he_name, le_name, w_he, transition, trwidth):
    """Blend yield matrices column-wise over the HE model's channel set.

    Channels absent from the LE model keep the HE matrix. The weight
    ``w_he`` of :func:`he_le_weight` depends on the projectile energy (last
    axis) only, so it broadcasts over the Legendre modes of 2D
    ``(n_k, n_e, n_e)`` tensors. Both models must be on the same energy grid,
    with the transition inside both models' coverage. ``transition`` and
    ``trwidth`` only enter the description string.
    """
    w_he = w_he[np.newaxis, :]
    w_le = 1.0 - w_he
    he_yields = he_index["index_d"]
    le_yields = le_index["index_d"]
    blended = {}
    for channel, he_matrix in he_yields.items():
        if channel in le_yields:
            blended[channel] = he_matrix * w_he + le_yields[channel] * w_le
        else:
            blended[channel] = he_matrix

    relations = defaultdict(list)
    particles = set()
    for parent, child in blended:
        relations[parent].append(child)
        particles.add(parent)
        particles.add(child)
    return {
        "parents": sorted(relations),
        "particles": sorted(particles),
        "relations": dict(relations),
        "index_d": blended,
        "description": (
            f"Runtime HE/LE blend: {he_name} + {le_name}; "
            f"transition={transition:g} GeV, "
            f"10-90 width={trwidth:g} decades"
        ),
    }


def blend_cross_sections(he_index, le_index, he_name, le_name, w_he):
    """Blend inelastic cross sections per projectile, mapping across models.

    A projectile carried by only one model is mapped onto the other model's
    equivalent column (:func:`~MCEq.data.hadronic_models.mapped_cross_section`);
    without an equivalent it stays unblended. ``he_name``/``le_name`` are
    normalised model names and select the equivalence tables.
    """
    w_le = 1.0 - w_he
    blended = {}
    for projectile, he_cs in he_index["index_d"].items():
        le_cs = mapped_cross_section(le_index["index_d"], projectile, le_name)
        # Leave HE-only projectiles unchanged.
        blended[projectile] = he_cs if le_cs is None else he_cs * w_he + le_cs * w_le
    # LE-only projectiles (e.g. FLUKA's n, nbar, K0, hyperons) are kept: own
    # sigma below the transition, the mapped HE equivalent above it. Otherwise
    # get_cs would fall back to the proton column and, for antibaryons, miss
    # the annihilation part.
    for projectile, le_cs in le_index["index_d"].items():
        if projectile in blended:
            continue
        he_cs = mapped_cross_section(he_index["index_d"], projectile, he_name)
        blended[projectile] = le_cs if he_cs is None else he_cs * w_he + le_cs * w_le
    return {"parents": sorted(blended), "index_d": blended}


class ModelStoreCache:
    """Models the primary file lacks, served from release packages on its grid.

    The manifest is read on the first miss, each package is opened once, and
    models no package provides are remembered.
    """

    def __init__(self, primary):
        # A manifest named in the package's root attributes overrides the
        # config default.
        declared = primary.attrs().get("mceq_db_manifest")
        self.manifest_name = str(declared) if declared else config.mceq_db_manifest
        common = primary.attrs("common")
        self._grid = download.grid_signature(common)
        self._e_grid = np.ascontiguousarray(np.asarray(common["e_grid"]))
        self._manifest = None
        self._stores = {}
        self._unavailable = set()

    @property
    def manifest(self):
        if self._manifest is None:
            self._manifest = download.ensure_manifest(config, self.manifest_name)
        return self._manifest

    def get(self, model, medium, tree):
        """Store holding ``tree/<medium>/<model>``, or None if no package on
        this grid lists the model for the medium."""
        key = (model, medium)
        if key in self._unavailable:
            return None
        for name in download.files_for(self.manifest, self._grid, model, medium):
            store = self._store(name)
            if store.has(f"{tree}/{medium}/{model}"):
                return store
        self._unavailable.add(key)
        return None

    def _store(self, name):
        store = self._stores.get(name)
        if store is None:
            path = download.ensure_package(config, name, self.manifest)
            store = hdf5_store.HDF5Store(str(path))
            self._check_grid(store, name)
            self._stores[name] = store
        return store

    def _check_grid(self, store, name):
        """Refuse a package whose ``common/e_grid`` is not byte-identical to
        the primary's: the two would index each other's bins silently. A 2D
        package must be booked in Legendre moments, like the primary."""
        common = store.attrs("common")
        hdf5_store.angular_basis(common, store.fname)
        other = np.ascontiguousarray(np.asarray(common["e_grid"]))
        if (
            other.shape != self._e_grid.shape
            or other.tobytes() != self._e_grid.tobytes()
        ):
            raise Exception(
                f"Data package {name} ({store.fname}) has a different energy grid "
                f"than the primary database {config.mceq_db_fname}. The "
                "manifest lists it on the same grid, so one of the two files is "
                "stale: delete it and fetch again with "
                f"python -m MCEq.data.download --file {name}"
            )


def missing_model_error(cache, model, medium):
    """The error raised when neither the primary file nor any package on its
    grid can serve ``model`` in ``medium``."""
    try:
        offered = download.models_on_grid(cache.manifest, cache._grid)
        manifest_note = (
            f"Models on this grid per {cache.manifest_name}: {json.dumps(offered)}."
        )
    except Exception as ex:  # manifest unavailable: say so, still actionable
        manifest_note = f"(release manifest not readable: {ex})"
    return Exception(
        f"Interaction model {model} for medium {medium!r} is not in the "
        f"primary database {config.mceq_db_fname} and no package of the data "
        f"release provides it on the same grid. {manifest_note} To register a "
        f"single-model file, add an entry to {Path(config.data_dir) / cache.manifest_name} "
        "(see the data installation docs)."
    )
