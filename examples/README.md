# MCEq examples

Each example is a [marimo](https://marimo.io) notebook: a plain Python file
that runs as a script, opens as an interactive notebook, and exports to a
clean Jupyter notebook or a static HTML page.

| Folder | Examples |
|---|---|
| `getting_started/` | `basic_flux` (start here): muon and neutrino fluxes at ground; `compare_primary_fluxes`: effect of the cosmic-ray model |
| `atmosphere/` | `density_profiles`: every density model as density vs depth; `spectrum_vs_atmosphere`: seasonal and site dependence of lepton fluxes |
| `fluxes/` | `zenith_distribution_icecube`, `zenith_averaged_neutrinos`, `muon_spectra` (Auger, IceCube), `full_sky_carousel` (`solve_fullsky`, geomagnetic cutoff) |
| `hadronic/` | `kpi_demonstration` and `modify_particle_yields` (`set_mod_pprod`), `ddm` (Data-Driven Model), `partial_hadron_contribution` and `partial_hadron_fractions` (lepton parents) |
| `showers_2d/` | `angular_shower_development` (Legendre 2D cascade); `corsika_p1tev_vertical` and `corsika_p1pev_60deg`: the validation against CORSIKA 7 |
| `data/corsika/` | CORSIKA steering cards, reduced reference histograms (`c7_reference.npz`) and their reduction scripts for the two comparisons |

## Run an example

```sh
pip install -e . --group notebooks
marimo edit examples/getting_started/basic_flux.py   # interactive
python examples/getting_started/basic_flux.py        # as a script
```

Opening a notebook only renders the configuration form. Nothing is built or
solved until you click **Build & solve**; after that, applying a new
configuration recomputes only the cells that depend on it.

The examples download the release databases they need on first use: the 1D
package is 76 MB; the 2D examples need the 456 MB Legendre package, and the
1 PeV CORSIKA comparison also the 968 MB SIBYLL 2.3e 2D package.

## Jupyter

In a marimo session, **Download Jupyter notebook** exports the notebook with
the applied configuration; the example pages link the same file. Locally,

```sh
python scripts/export_marimo.py examples/getting_started/basic_flux.py
```

writes the same file next to the source. The export replaces the
configuration form with plain assignments of the applied values and removes
every marimo construct, so it runs with MCEq, NumPy, Matplotlib and crflux
only. Exported `.ipynb` files are build products and are not committed.

## The example pages

```sh
python scripts/build_examples.py -j 4          # all examples
python scripts/build_examples.py -k corsika    # a subset
```

runs every notebook with all gates open (`MCEQ_EXAMPLES_AUTORUN=1`) and writes,
under `docs/_build/site/examples/`, one HTML page per example with its solved
figures, the Jupyter export, the marimo source, an `index.html` and a
`manifest.json` (source commit, per-notebook status and run time). The Sphinx
build links the gallery when that directory exists. The full build needs the
release databases and takes about half an hour on a workstation; it is not
part of the Read the Docs build or of pull-request CI, which runs
`marimo check --strict` and the Jupyter-export tests only.

## Writing an example

Use `getting_started/basic_flux.py` as the template:

- A title and one sentence naming the observable.
- `## 01 · Configuration`: one compact form (cells `configuration_defaults`,
  `configuration_controls`, `selected_configuration`). Explain each option:
  what it does, its default, when to change it. Submit changes together.
- The cells `run_gate` and `run_stop` (via `MCEq.notebook.gate_closed`); the
  first compute cell consumes the `proceed` token.
- Build the `MCEqRun` in one cell and solve in the next, inside
  `solve_progress` (or `loop_progress` for repeated solves); `solve`,
  `solve_batch` and `solve_fullsky` take a `progress` callback.
- Readout, then a focused figure with a local `plt.rc_context`.
- A final `export_notebook` cell with `download_jupyter("<stem>.ipynb", ...)`.
- Only static Markdown in `mo.md`; cells other than the reserved ones must
  not use `mo`, or the Jupyter export fails.

Keep the cascade grid wider than the plotted interval, and keep the run time
of a default configuration to a few minutes. Shared helpers are in
`src/MCEq/notebook.py`. The Jupyter conversion uses marimo's private
converter, so marimo is pinned (0.24.0) and `tests/test_notebook_export.py`
exports every example.

## Comparisons with CORSIKA

`data/corsika/<case>/` holds what is needed to reproduce a comparison
without the raw air-shower output: the steering card (`steering.inp`), a
README with the CORSIKA build, statistics, histogram definitions and the table
of MCEq settings that match the CORSIKA run, the reduced histograms
(`c7_reference.npz`, a few tens of kB) and the script that produced them from
the aggregated CORSIKA output.
