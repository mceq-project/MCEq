# Basic-flux notebook template

`docs/examples/Basic_flux.py` is the source; `Basic_flux.ipynb` is its clean Jupyter
export. Both show the same configuration and three-panel flux figure.

## Run and download

Open **Atmospheric lepton fluxes** in the browser gallery. The first launch
installs Python packages and loads the model tables; the calculation runs on
your device. Use the dropdowns and sliders in section 01, then press **Apply configuration**
to recalculate. Changes are grouped so dragging a slider does not repeatedly solve
the cascade. Energy sliders use log₁₀(E/GeV).
The browser bundle includes **SIBYLL2.3e** and **QGSJET-II-04**, CORSIKA atmospheres,
and the NumPy solver (`auto` or `numpy_etd2`). Low-energy controls are disabled in this two-model browser bundle. Local Python
also offers DPMJET-III-19.3 for the low-energy model and installed native backends.

**Download Jupyter notebook**, below the figure, converts the current saved
source in memory. It writes the last **applied** control values as a plain Python
configuration cell and removes the widgets, download cell, marimo imports, and
marimo metadata. Pending form edits are not exported until applied. It requires no subprocess or server.
Marimo's standard Export menu remains available, but does not apply this cleanup.
The downloaded notebook runs locally with this MCEq version, NumPy, Matplotlib,
and crflux; marimo is not required. It uses the full public database, downloaded
on first calculation. Browser packages and data require internet on first load.

For local editing and the same cleaned export:

```sh
pip install -e . --group notebooks
marimo edit docs/examples/Basic_flux.py
marimo check --strict docs/examples/Basic_flux.py
python scripts/export_marimo.py docs/examples/Basic_flux.py
```

## Build the browser gallery

```sh
python scripts/build_browser_gallery.py
python -m http.server --directory docs/_build/site 8000
```

Open `http://localhost:8000/browser/`. To use an existing public database or
choose another output directory:

```sh
python scripts/build_browser_gallery.py \
  --database /path/to/mceq_db_lext_dpm193_v142.h5 \
  --output /tmp/mceq-gallery
```

The build copies the selected air-interaction tables losslessly, retaining the
full energy grid and decay tables. The default two-model database is about
41 MB, versus 289 MB for the full source. `--models` accepts database group names
and must include the default `SIBYLL23E`. No reduced-precision or truncated-grid
approximation is introduced.

A gallery-only Python wheel supplies MCEq to Pyodide. CORSIKA uses the same scalar
formulas as the compiled atmosphere; the solver uses the existing NumPy fallback.
Native MKL/CUDA backends and MSIS atmospheres require local Python. The generated
wheel is a deployment artifact, not a replacement for the normal MCEq package.

`manifest.json` records source commit, dirty status, models, sizes and SHA-256
hashes. Generated wheels, databases, and HTML stay outside version control.
Read the Docs builds this gallery before Sphinx; Sphinx copies it under
`browser/`. Notebook CI also saves a downloadable static-gallery artifact.
This first port sits alongside the existing examples.

## Style for subsequent examples

- A title and one sentence describing the observable.
- One compact configuration form, grouped by interactions, primary/solver,
  energy grid, and geometry/transition; units beside each control. Submit changes
  together. The Jupyter export materializes the applied values as assignments.
- Construction and solving together so a configuration edit rebuilds the run.
- Readout followed by a focused figure, with local `plt.rc_context` styling.
- Static Markdown and ordinary Python for the calculation. Reserve
  `configuration_defaults`, `configuration_controls`, `selected_configuration`,
  and `export_notebook` for the shared form/export pattern. The converter reads
  literal defaults for CLI export and applied settings for browser downloads.

Keep the cascade grid wider than the plotted interval. Low-energy transition
settings require valid overlap in the chosen model tables. The conversion uses
marimo's private converter, so the gallery pins marimo 0.24.0 and tests the
cleanup before upgrading it.
