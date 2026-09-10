**Removed** `MCEq.charm_models` (decision D6): the abstract `CharmModel`
base and `MRS_charm` hook for replacing a yield table with a custom charm
model. Unreachable from the shipped API since the WHR/MRS charm tables
left the cross-section database — no `Yields.set_custom_charm_model`
entry point ever existed in v2, and no caller outside the module and its
own test referenced the names.
