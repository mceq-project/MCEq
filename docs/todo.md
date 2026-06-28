# TODO

Pending follow-ups that don't belong to a specific PR yet.

## 3D atmosphere: auto-pick `interaction_medium` from `density_model.env_name`

A WIP draft existed on the `feature/etd1-solver` branch that, in
`MCEqRun.__init__` and `MCEqRun.set_density_model`, did:

```python
if hasattr(self.density_model, 'env_name') and self.density_model.env_name:
    self.medium = self.density_model.env_name
    config.interaction_medium = self.density_model.env_name
```

**Intent.** `GeneralizedTarget` exposes an `env_name` attribute naming the
material (e.g. `"air"`, `"rock"`, `"water"`). `self.medium` selects which
HDF5 subgroup is loaded by `data.HDF5Backend` and which target nucleus
mass is used by `misc._target_masses[config.interaction_medium.lower()]`.
The hook auto-aligns those settings with the density model so users don't
have to set `config.interaction_medium = "rock"` separately before
constructing `MCEqRun`.

The `__init__` placement (before `HDF5Backend(medium=self.medium)` is
constructed) is correct.

**Why it was reverted.** Two issues to resolve before merging:

1. **`set_density_model` hunk is a partial no-op.** It updates
   `self.medium` and the config global but does **not** rebuild
   `self._mceq_db = HDF5Backend(medium=...)`, re-call
   `Interactions.load(...)`, or refresh `_int_cs`. So if a user swaps to
   a target with a different `env_name` after init, the interaction
   matrices are still keyed off the original medium. The rebuild needs
   to happen here, or the second hunk should be dropped.

2. **Side-effect on a module-level global.** Setting
   `config.interaction_medium = ...` mutates module state that survives
   across `MCEqRun` instances. Two coexisting `MCEqRun`s with different
   `env_name`s trample each other's global, and the global is never
   restored when the model is replaced with one that lacks `env_name`.
   Either keep the change purely on the instance (`self.medium`) and
   thread it through every consumer, or save/restore around the mutation.

Both issues are tractable. Land alongside the rest of the 3D-atm work on
the `claude/km3net-density-model-rD59O` branch.
