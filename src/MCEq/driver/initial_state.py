"""Initial state: primary models to the ``phi0`` vector.

``InitialState`` composes the initial-condition vector the solver starts
from — the machinery behind ``MCEqRun.set_primary_model`` /
``set_single_primary_particle`` / ``set_initial_spectrum`` /
``initial_state()`` — and owns the restore list whose *method names* the
facade replays after every species-dimension change (names, never bound
methods). The pman/energy-grid reads route through the system, ``_phi0``
is ``phi0`` here, and the resize of the composition vector is
:meth:`resize` (driven by the facade, which zeroes its own ``_solution``
and replays the list).

Reads the physics system (pman, energy grid) but never the facade.
"""

import numpy as np

from MCEq.misc import info


class InitialState:
    """The primary-model configuration and the composed ``phi0``.

    Args:
      system: the :class:`MCEq.driver.system.CascadeSystem` the particle
        manager and energy grid are read from.
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
          interaction_model (:class:`CRFluxModel.PrimaryFlux`): reference
          to primary model **class**
          tag (tuple): positional argument list for model class
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

        # Save primary flux model for restoration after interaction model
        # changes. Store the method *name*, not a bound method — see the
        # comment in ``_resize_vectors_and_restore`` (PR #163).
        self._restore_initial_condition = [("set_primary_model", self.pmodel)]
        # TODO: Maybe needs to catch the removal of the np.vectorize
        # self.get_nucleon_spectrum = np.vectorize(self.pmodel.p_and_n_flux)
        self.get_nucleon_spectrum = self.pmodel.p_and_n_flux

        try:
            self._system.pman.dim_states
        except AttributeError:
            self.finalize_pmodel = True

        # Set initial condition
        minimal_energy = self._system._cfg.physics.minimal_primary_energy
        if (2212, 0) in self._system.pman and (2112, 0) in self._system.pman:
            e_tot = self._system._energy_grid.c + 0.5 * (
                self._system.pman[(2212, 0)].mass + self._system.pman[(2112, 0)].mass
            )
        else:
            raise Exception(
                "No nucleons in eqn system, primary flux model can not be used."
            )

        min_idx = np.argmin(np.abs(e_tot - minimal_energy))
        self.phi0 *= 0
        p_top, n_top = self.get_nucleon_spectrum(e_tot[min_idx:])[1:]
        if (2212, 0) in self._system.pman:
            self.phi0[
                min_idx + self._system.pman[(2212, 0)].lidx : self._system.pman[
                    (2212, 0)
                ].uidx
            ] = 1e-4 * p_top
        else:
            info(
                1,
                "Protons not in equation system, can not set primary flux.",
            )

        if (2112, 0) in self._system.pman and not self._system.pman[
            (2112, 0)
        ].is_resonance:
            self.phi0[
                min_idx + self._system.pman[(2112, 0)].lidx : self._system.pman[
                    (2112, 0)
                ].uidx
            ] = 1e-4 * n_top
        elif (2212, 0) in self._system.pman:
            info(
                2,
                "Neutrons not part of equation system,",
                "substituting initial flux with protons.",
            )
            self.phi0[
                min_idx + self._system.pman[(2212, 0)].lidx : self._system.pman[
                    (2212, 0)
                ].uidx
            ] += 1e-4 * n_top

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

        A continuous input energy range is allowed between
        :math:`50*A~ \\text{GeV} < E_\\text{nucleus} < 10^{10}*A \\text{GeV}`.

        Args:
          E (float): kinetic energy of a nucleus in GeV
          corsika_id (int): ID of a nucleus (see text)
          pdg_id (int): PDG ID of a particle
          append (bool): If True, keep previous state and append a new particle.
        """
        import warnings

        from scipy.linalg import solve

        from MCEq.misc import getAZN, getAZN_corsika

        if corsika_id and pdg_id:
            raise Exception("Provide either corsika or PDG ID")

        info(2, f"CORSIKA ID {corsika_id}, PDG ID {pdg_id}, energy {E:5.3g} GeV")

        if corsika_id:
            n_nucleons, n_protons, n_neutrons = getAZN_corsika(corsika_id)
        elif pdg_id:
            n_nucleons, n_protons, n_neutrons = getAZN(pdg_id)

        En = E / float(n_nucleons) if n_nucleons > 0 else E

        if En < np.min(self._system._energy_grid.c):
            raise Exception("energy per nucleon too low for primary " + str(corsika_id))

        if append is False:
            # Store ``False`` explicitly so the replay does not silently
            # default to overwriting on the first call of an append chain.
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

    def set_initial_spectrum(self, spectrum, pdg_id, append=False):
        """Set a user-defined spectrum for an arbitrary species as initial condition.

        This function is an equivalent to :func:`set_single_primary_particle`. It
        allows to define an arbitrary spectrum for each available particle species
        as initial condition for the integration. Set the `append`
        argument to `True` for subsequent species to define initial
        spectra combined from different particles.

        The (differential) spectrum has to be distributed on the energy
        grid as dN/dptot, i.e. divided by the bin widths and with the
        total momentum units in GeV(/c).

        Args:
          spectrum (np.array): spectrum dN/dptot
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
            - a user spectrum: ``{"spectrum": <array dN/dptot>,
              "pdg_id": <PDG>}`` (forwarded to
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
        """Reallocate ``phi0`` to the current system size and replay.

        The original ``_resize_vectors_and_restore`` first line, verbatim,
        plus the restore loop; the caller (facade) zeroes its own solution
        vector and passes the replay of the stored method names, since the
        restore entries name facade methods (module docstring).
        """
        self.phi0 = np.zeros(self._system.pman.dim_states)
        if len(restore_list) > 0:
            for con in restore_list:
                replay(con[0], con[1:])
