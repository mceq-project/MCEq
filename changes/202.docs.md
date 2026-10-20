The examples moved to a top-level `examples/` folder, grouped by topic
(`getting_started`, `atmosphere`, `fluxes`, `hadronic`, `showers_2d`), and
are now marimo notebooks: each runs as a script, opens interactively behind a
**Build & solve** button, and exports to a clean Jupyter notebook
(`scripts/export_marimo.py`) and to a static HTML page with its figures and links to the marimo and Jupyter
files (`scripts/build_examples.py`). All examples run on the v2 release databases.
New: `hadronic/modify_particle_yields` (from the former `mceq-examples`
repository) and two validations against CORSIKA 7.8050,
`showers_2d/corsika_p1tev_vertical` (angular distributions, 1 TeV proton) and
`showers_2d/corsika_p1pev_60deg` (spectra at seven depths, 1 PeV at 60°),
with steering cards and reduced reference histograms in
`examples/data/corsika/`. `MCEqRun.solve`, `solve_batch` and `solve_fullsky`
take a `progress(done, total)` callback.
