Every example notebook in `docs/examples/` and `examples/` was refreshed for
the v2 layout (M15): kernelspec and language_info are now `python3` everywhere
(nine declared `python2`, one `mceq_dev_396`), `set_theta_deg` is gone from
all executed code, the CI database pin is present in every notebook that
solves, and `Muon spectra.ipynb` lost its six Python-2 print statements.
`DDM_example.ipynb` joined the gallery, ported from the (to-be-archived)
`mceq-examples` repository: `MCEq.ddm` becomes `MCEq.models.ddm.ddm`, the
baseline model is `SIBYLL21` instead of the non-redistributable
`SIBYLL2.3d`, and the injection now targets the DDM instance (the upstream
cell injected into the comparison run, an upstream typo). The executed
notebooks were re-run at the v2 tree and their stored outputs refreshed.
The four CI-ignored notebooks stay ignored — M15 re-measured every reason
against both shipped databases; the workflow comments carry the details.
