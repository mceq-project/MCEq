"""Initial condition (primary flux or user spectra) of the cascade."""

import numpy as np

from MCEq.misc import info


class InitialState:
    """Primary model, initial-condition vector ``phi0`` and its restore list.

    Args:
      system: the :class:`MCEq.driver.system.CascadeSystem` providing the
        particle manager and energy grid.
    """

    def __init__(self, system):
        self._system = system
        self.phi0 = np.zeros(1)
        self._restore_initial_condition = []
        self.pmodel = None
        self.get_nucleon_spectrum = None

    def set_primary_model(self, model_class_or_object, tag=None):
        """Sets primary flux model.

        This functions is quick and does not require re-generation of
        matrices.

        Args:
          model_class_or_object: a :class:`crflux.models.PrimaryFlux`
            subclass, or an instance of one
          tag: passed as a single positional argument to the class when
            constructing the model object
        """

        assert not isinstance(model_class_or_object, tuple), (
            "Primary model can not be supplied as tuples"
        )

        # Check if classs or object supplied
        if not isinstance(model_class_or_object, type):
            assert any(
                [
                    "PrimaryFlux" in b.__name__
                    for b in model_class_or_object.__class__.__bases__
                ]
            ), "model_class_or_object is not derived from crflux.models.PrimaryFlux"
            info(5, "Primary model supplied as object")
            self.pmodel = model_class_or_object
        else:
            # Initialize primary model object
            info(5, "Primary model supplied as class")
            self.pmodel = model_class_or_object(tag)

        info(1, f"Primary model set to {self.pmodel.name}")

        # Replayed by name after interaction-model changes.
        self._restore_initial_condition = [("set_primary_model", self.pmodel)]
        self.get_nucleon_spectrum = self.pmodel.p_and_n_flux

        pman = self._system.pman
        minimal_energy = self._system._cfg.physics.minimal_primary_energy
        if (2212, 0) not in pman or (2112, 0) not in pman:
            raise Exception(
                "No nucleons in eqn system, primary flux model can not be used."
            )
        proton, neutron = pman[(2212, 0)], pman[(2112, 0)]
        e_tot = self._system._energy_grid.c + 0.5 * (proton.mass + neutron.mass)

        min_idx = np.argmin(np.abs(e_tot - minimal_energy))
        self.phi0 *= 0
        p_top, n_top = self.get_nucleon_spectrum(e_tot[min_idx:])[1:]
        self.phi0[min_idx + proton.lidx : proton.uidx] = 1e-4 * p_top

        if not neutron.is_resonance:
            self.phi0[min_idx + neutron.lidx : neutron.uidx] = 1e-4 * n_top
        else:
            info(
                2,
                "Neutrons not part of equation system,",
                "substituting initial flux with protons.",
            )
            self.phi0[min_idx + proton.lidx : proton.uidx] += 1e-4 * n_top

    def set_single_primary_particle(
        self, E, corsika_id=None, pdg_id=None, append=False
    ):
        """Set type and kinetic energy of a single primary nucleus to
        calculation of particle yields.

        The functions uses the superposition theorem, where the flux of
        a nucleus with mass A and charge Z is modeled by using Z protons
        and A-Z neutrons at energy :math:`E_{nucleon}= E_{nucleus} / A`
        The nucleus type is defined via :math:`\\text{CORSIKA ID} = A*100 + Z`. For
        example iron has the CORSIKA ID 5226.

        Single leptons or hadrons can be defined by specifiying `pdg_id` instead of
        `corsika_id`.

        The `append` argument can be used to compose an initial state with
        multiple particles. If it is `False` the initial condition is reset to zero
        before adding the particle.

        The kinetic energy per nucleon must fit the active energy grid and the
        deposition: three bins matching E^0..E^2 at the bin centres, or, for
        species whose tables the database marks as stencil-averaged, 2r + 1
        bins whose reconstructed density has the moments of a delta
        (:func:`MCEq.species.bin_average.delta_deposition`).

        Args:
          E (float): kinetic energy of a nucleus in GeV
          corsika_id (int): ID of a nucleus (see text)
          pdg_id (int): PDG ID of a particle
          append (bool): If True, keep previous state and append a new particle.
        """
        import warnings

        from scipy.linalg import solve

        from MCEq.misc import getAZN

        if corsika_id and pdg_id:
            raise Exception("Provide either corsika or PDG ID")

        info(2, f"CORSIKA ID {corsika_id}, PDG ID {pdg_id}, energy {E:5.3g} GeV")

        if corsika_id:
            # CORSIKA codes nuclei as 100 * A + Z; 14 is the proton
            n_nucleons, n_protons = (
                (1, 1) if corsika_id == 14 else divmod(corsika_id, 100)
            )
            n_neutrons = n_nucleons - n_protons
        elif pdg_id:
            n_nucleons, n_protons, n_neutrons = getAZN(pdg_id)

        En = E / float(n_nucleons) if n_nucleons > 0 else E

        if En < np.min(self._system._energy_grid.c):
            raise Exception("energy per nucleon too low for primary " + str(corsika_id))

        if append is False:
            self._restore_initial_condition = [
                ("set_single_primary_particle", E, corsika_id, pdg_id, False)
            ]
            self.phi0 *= 0.0
        else:
            self._restore_initial_condition.append(
                ("set_single_primary_particle", E, corsika_id, pdg_id, True)
            )
        egrid = self._system._energy_grid.c
        ebins = self._system._energy_grid.b
        ewidths = self._system._energy_grid.w

        info(
            3,
            (
                f"superposition: n_protons={n_protons}, n_neutrons={n_neutrons}, "
                + f"energy per nucleon={En:5.3g} GeV"
            ),
        )

        cenbin = np.argwhere(En < ebins)[0][0] - 1

        if self._deposit_for_stencil(En, n_nucleons, n_protons, n_neutrons, pdg_id):
            return

        if cenbin == 0 or cenbin == len(egrid) - 1:
            # The outermost bin has no neighbour on one side for the
            # three-bin moment match; deposit the particle there whole.
            if n_nucleons == 0:
                parts = [(pdg_id, 1.0)]
            else:
                parts = [(2212, n_protons), (2112, n_neutrons)]
            for pid, k in parts:
                if k > 0:
                    lidx = self._system.pman[pid].lidx
                    self.phi0[lidx + cenbin] += k / ewidths[cenbin]
            return

        # Equalize the first three moments for 3 normalizations around the central
        # bin
        emat = np.vstack(
            (
                ewidths[cenbin - 1 : cenbin + 2],
                ewidths[cenbin - 1 : cenbin + 2] * egrid[cenbin - 1 : cenbin + 2],
                ewidths[cenbin - 1 : cenbin + 2] * egrid[cenbin - 1 : cenbin + 2] ** 2,
            )
        )

        if n_nucleons == 0:
            # This case handles other exotic projectiles
            b_particle = np.array([1.0, En, En**2])
            lidx = self._system.pman[pdg_id].lidx
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.phi0[lidx + cenbin - 1 : lidx + cenbin + 2] += solve(
                    emat, b_particle
                )
            return

        if n_protons > 0:
            b_protons = np.array([n_protons, En * n_protons, En**2 * n_protons])
            p_lidx = self._system.pman[2212].lidx
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.phi0[p_lidx + cenbin - 1 : p_lidx + cenbin + 2] += solve(
                    emat, b_protons
                )
        if n_neutrons > 0:
            b_neutrons = np.array([n_neutrons, En * n_neutrons, En**2 * n_neutrons])
            n_lidx = self._system.pman[2112].lidx
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.phi0[n_lidx + cenbin - 1 : n_lidx + cenbin + 2] += solve(
                    emat, b_neutrons
                )

    def _deposit_for_stencil(self, En, n_nucleons, n_protons, n_neutrons, pdg_id):
        """Deposit a single particle for stencil-averaged tables, if they apply.

        Stencil-averaged tables (:mod:`MCEq.species.bin_average`) read the
        cells around a bin as a smooth density, so the particle is deposited
        on 2r + 1 bins whose reconstructed density has the log-energy moments
        of a delta. Returns False (nothing deposited) when no deposited
        species carries a stencil mark, or within r bins of the grid edge,
        where the three-bin deposition is used.
        """
        from MCEq.species.bin_average import delta_deposition

        pman = self._system.pman
        if n_nucleons == 0:
            parts = [(pman[pdg_id], 1.0)]
        else:
            parts = [(pman[2212], n_protons), (pman[2112], n_neutrons)]
            parts = [(p, k) for p, k in parts if k > 0]
        radius = {
            max(getattr(p, "_int_stencil", 0), getattr(p, "_dec_stencil", 0))
            for p, _ in parts
        }
        if radius == {0}:
            return False
        if len(radius) != 1 or 0 in radius:
            raise ValueError(
                "single primary: deposited species disagree on the bin-average "
                f"stencil ({sorted(radius)}); the database marks are inconsistent"
            )
        r = radius.pop()
        grid = self._system._energy_grid
        c = int(np.searchsorted(grid.b, En, side="right")) - 1
        if c < r or c > len(grid.c) - 1 - r:
            # The 2r + 1 bins would leave the grid; near the edge the tables
            # read a constant density anyway, and a one-sided moment match
            # oscillates. Use the three-bin deposition there.
            info(
                2,
                f"single primary at {En:5.3g} GeV is within {r} bins of the grid "
                "edge; using the three-bin deposition",
            )
            return False
        first, contents = delta_deposition(grid.b, En, r)
        sl = slice(first, first + len(contents))
        for p, k in parts:
            self.phi0[p.lidx + sl.start : p.lidx + sl.stop] += k * contents / grid.w[sl]
        return True

    def set_initial_spectrum(self, spectrum, pdg_id, append=False):
        """Set a user-defined spectrum for an arbitrary species as initial condition.

        This function is an equivalent to :func:`set_single_primary_particle`. It
        allows to define an arbitrary spectrum for each available particle species
        as initial condition for the integration. Set the `append`
        argument to `True` for subsequent species to define initial
        spectra combined from different particles.

        The spectrum is supplied as one differential-spectrum value per
        kinetic-energy bin and copied into the selected particle's
        state-vector slice without converting between energy and momentum
        conventions.

        Args:
          spectrum (np.array): differential spectrum on the kinetic-energy grid
          pdg_id (int): PDG ID in case of a particle
        """

        info(2, f"PDG ID {pdg_id}")

        if not append:
            self._restore_initial_condition = [
                ("set_initial_spectrum", spectrum, pdg_id, append)
            ]
            self.phi0 *= 0
        else:
            self._restore_initial_condition.append(
                ("set_initial_spectrum", spectrum, pdg_id, append)
            )
        if len(spectrum) != self._system._energy_grid.d:
            raise Exception("Lengths of spectrum and energy grid do not match.")

        self.phi0[self._system.pman[pdg_id].lidx : self._system.pman[pdg_id].uidx] += (
            spectrum
        )

    def get_initial_state(self):
        """Return a copy of the current initial-condition vector ``phi0``.

        This is the state vector composed by the most recent
        :meth:`set_primary_model` / :meth:`set_single_primary_particle` /
        :meth:`set_initial_spectrum` calls — the same vector
        :meth:`solve` propagates. Use it to assemble columns for
        :meth:`solve_batch` without touching private attributes.

        Returns:
          (np.ndarray[dim_states]): copy of the initial state vector.
        """
        return self.phi0.copy()

    def initial_state(self, components):
        """Compose and return an initial-state column without mutating
        the instance.

        Builds a ``(dim_states,)`` vector from one or more components
        using the same machinery as :meth:`set_single_primary_particle`
        and :meth:`set_initial_spectrum`, then restores the previous
        initial condition. The intended use is assembling the columns of
        a :meth:`solve_batch` initial-state matrix::

            # response matrix: one column per primary energy
            phi0 = np.stack(
                [mceq.initial_state({"E": E, "pdg_id": 2212}) for E in E_primaries],
                axis=1,
            )
            res = mceq.solve_batch(phi0)

        Args:
          components (dict | list[dict]): one component dict or a list of
            component dicts (summed). Each component is either

            - a single primary: ``{"E": <GeV>, "corsika_id": <A*100+Z>}``
              or ``{"E": <GeV>, "pdg_id": <PDG>}`` (forwarded to
              :meth:`set_single_primary_particle`), or
            - a user spectrum: ``{"spectrum": <array on the
              kinetic-energy grid>, "pdg_id": <PDG>}`` (forwarded to
              :meth:`set_initial_spectrum`).

        Returns:
          (np.ndarray[dim_states]): the composed initial-state column.
        """
        if isinstance(components, dict):
            components = [components]
        if not components:
            raise ValueError("initial_state: components must not be empty")

        saved_phi0 = self.phi0.copy()
        saved_restore = list(self._restore_initial_condition)
        try:
            for i, comp in enumerate(components):
                comp = dict(comp)
                append = i > 0
                if "spectrum" in comp:
                    spectrum = comp.pop("spectrum")
                    pdg_id = comp.pop("pdg_id")
                    if comp:
                        raise ValueError(
                            f"initial_state: unknown keys {sorted(comp)} in "
                            f"spectrum component"
                        )
                    self.set_initial_spectrum(spectrum, pdg_id, append=append)
                elif "E" in comp:
                    E = comp.pop("E")
                    unknown = set(comp) - {"corsika_id", "pdg_id"}
                    if unknown:
                        raise ValueError(
                            f"initial_state: unknown keys {sorted(unknown)} in "
                            f"single-primary component"
                        )
                    self.set_single_primary_particle(E, append=append, **comp)
                else:
                    raise ValueError(
                        "initial_state: each component needs either 'E' "
                        "(single primary) or 'spectrum' + 'pdg_id' "
                        f"(user spectrum); got keys {sorted(comp)}"
                    )
            return self.phi0.copy()
        finally:
            self.phi0 = saved_phi0
            self._restore_initial_condition = saved_restore

    def resize(self, restore_list, replay):
        """Reallocate ``phi0`` and call ``replay(name, args)`` per recorded entry."""
        self.phi0 = np.zeros(self._system.pman.dim_states)
        if len(restore_list) > 0:
            for con in restore_list:
                replay(con[0], con[1:])
