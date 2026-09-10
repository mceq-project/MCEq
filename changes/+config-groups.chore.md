`MCEq.config` exposes grouped views of its settings — `config.grid`,
`config.physics`, `config.paths` and so on, one per layer of the target
architecture. They read and write through to the flat names, so a component
handed `config.grid` still sees a later `config.e_min = ...`, and dict settings
stay shared by identity. Components take these instead of the module as the
refactor proceeds.

Removed along the way: `MCEqParticle.dN_dxf` (needs a `sibyll23c_aux.ppd` that
ships with no release) and `_interaction_threshold` (its only call site was
commented out), the `config.kernel_config == "CUDA"` branches in `dN_dxlab`,
`dNdec_dxlab` and `dN_dEkin` (config spells it `cuda_etd2`, so they never ran),
and the `mkl_spmm_tile` / `accelerate_spmm_tile` lookups, which read names
`config` never declared.
