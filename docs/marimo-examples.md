# Editing the basic-flux example

`docs/examples/Basic_flux.py` is the source; `Basic_flux.ipynb` is its Jupyter export.
With MCEq and the `notebooks` dependency group installed, run from the repository root:

```sh
marimo edit docs/examples/Basic_flux.py
marimo check docs/examples/Basic_flux.py
python scripts/export_marimo.py docs/examples/Basic_flux.py
```

The Jupyter export uses ordinary Python, Markdown, and Matplotlib. It does not
require marimo. The helper uses `marimo export ipynb --sort top-down`, removes
the unused Markdown-helper import, and saves without outputs for a small source file.
The current public MCEq database downloads on the first calculation.

Use this structure for subsequent examples:

- A title and one sentence describing the observable.
- One visible configuration cell, grouped by physics, energy grid, and backend;
  units and alternative values belong beside the assignments.
- Construction and solving in one cell so a configuration edit rebuilds the run.
- Readout followed by a focused figure, using local `plt.rc_context` styling.
- Static Markdown and plain Python; no widgets or notebook-specific output helpers.

Keep the cascade grid wider than the displayed energy interval. Low-energy
blending is off until a model is selected; its transition and width require
valid overlap in the chosen model tables.
