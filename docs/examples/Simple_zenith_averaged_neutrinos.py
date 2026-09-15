# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo==0.24.0", "nbformat", "MCEq>=2.0.0", "matplotlib>=3.10", "numpy", "crflux>1.0.5"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Zenith-averaged neutrinos")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Zenith-averaged neutrino flux

    Down-going muon and electron neutrinos averaged over 11 bins in
    $\cos\theta$, split into conventional and prompt components. The plainest
    full example: one solve loop, two plots, one ASCII file.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from MCEq import config
    from MCEq.core import MCEqRun
    from MCEq.notebook import (
        apply_kernel_and_grid,
        download_jupyter,
        loop_progress,
    )

    return (
        MCEqRun,
        apply_kernel_and_grid,
        config,
        download_jupyter,
        loop_progress,
        mo,
        np,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 01 · Configuration
    """)
    return


@app.cell(hide_code=True)
def configuration_defaults():
    defaults = {
        "database": "mceq_db_lext_dpm193_v142.h5",
        "interaction_model": "SIBYLL21",
        "primary": "HillasGaisser2012:H3a",
        "theta_deg": 0.0,
        "e_min": 1.0,
        "e_max": 1.0e9,
        "backend": "auto",
    }
    return (defaults,)


@app.cell(hide_code=True)
def configuration_controls(defaults, mo):
    from MCEq.notebook import configuration_controls as build_controls

    controls, _options = build_controls(
        defaults,
        database=defaults["database"],
        show=("interaction_model", "primary", "energy", "backend"),
    )
    controls
    return (controls,)


@app.cell(hide_code=True)
def selected_configuration(controls, defaults, mo):
    from MCEq.notebook import apply_configuration, primary_model_from

    settings = apply_configuration(controls, defaults)
    database = settings["database"]
    interaction_model = settings["interaction_model"]
    primary_model = primary_model_from(settings["primary"])
    e_min, e_max = settings["e_min"], settings["e_max"]
    backend = settings["backend"]
    mo.md(
        f"**Applied:** {interaction_model} · {settings['primary']} · "
        f"{e_min:g}–{e_max:g} GeV"
    )
    return (
        backend, database, e_max, e_min, interaction_model, primary_model,
        settings,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 02 · Average over $\cos\theta$

    Equidistant in $\cos\theta$ from vertical to horizontal, 11 points.
    """)
    return


@app.cell
def _(MCEqRun, apply_kernel_and_grid, backend, config, database, e_max, e_min,
      interaction_model, np, primary_model):
    apply_kernel_and_grid(backend, e_min, e_max, 0)
    config.mceq_db_fname = database

    angles = np.arccos(np.linspace(1, 0, 11)) * 180.0 / np.pi
    mag = 3

    mceq_run = MCEqRun(
        interaction_model=interaction_model,
        primary_model=primary_model,
        theta_deg=0.0,
    )
    e_grid = mceq_run.e_grid

    pairs = {
        "mu": ("mu+", "mu-"), "numu": ("numu", "antinumu"),
        "nue": ("nue", "antinue"), "nutau": ("nutau", "antinutau"),
    }
    prefix = {
        "mu_conv": ("conv", "mu"), "mu_pr": ("pr", "mu"), "mu_total": ("total", "mu"),
        "numu_conv": ("conv", "numu"), "numu_pr": ("pr", "numu"),
        "numu_total": ("total", "numu"),
        "nue_conv": ("conv", "nue"), "nue_pr": ("pr", "nue"),
        "nue_total": ("total", "nue"), "nutau_pr": ("total", "nutau"),
    }
    flux = {frac: np.zeros_like(e_grid) for frac in prefix}

    def scan(advance):
        for theta in angles:
            mceq_run.set_zenith_azimuth(theta)
            mceq_run.solve()
            advance(title=f"{theta:.0f} deg")
            for frac, (track, group) in prefix.items():
                flux[frac] += sum(
                    mceq_run.get_solution(f"{track}_{p}", mag)
                    for p in pairs[group]
                )

    with loop_progress(len(angles), "zenith average") as advance:
        scan(advance)
    flux = {frac: total / len(angles) for frac, total in flux.items()}
    return (e_grid, flux, mag)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 03 · Spectra

    Solid black: total; dash-dot blue: conventional; dashed red: prompt.
    """)
    return


@app.cell
def _(e_grid, flux, mag, np, plt):
    figures = []
    for pref, lab in [("numu_", r"\nu_\mu"), ("nue_", r"\nu_e")]:
        fig, ax = plt.subplots(figsize=(4.2, 3))
        ax.loglog(e_grid, flux[pref + "total"], color="k", ls="-", lw=1.5)
        ax.loglog(
            e_grid, flux[pref + "conv"], color="b", ls="-.", lw=1.5,
            label=rf"conventional ${lab}$",
        )
        ax.loglog(
            e_grid, flux[pref + "pr"], color="r", ls="--", lw=1.5,
            label=f"prompt ${lab}$",
        )
        ax.set_xlim(10, 1e7)
        ax.set_ylim(1e-5, 10)
        ax.set_xlabel(rf"$E_{{{lab}}}$ [GeV]")
        ax.set_ylabel(
            r"$\Phi_{" + lab + "}$ (E/GeV)$^{" + str(mag) + " }$"
            + "(cm$^{2}$ s sr GeV)$^{-1}$"
        )
        ax.legend(loc="upper right", frameon=False, numpoints=1)
        fig.tight_layout()
        figures.append(fig)
    figures
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 04 · ASCII export
    """)
    return


@app.cell
def _(e_grid, flux, np):
    import os
    import tempfile

    components = [
        "mu_conv", "mu_pr", "mu_total",
        "numu_conv", "numu_pr", "numu_total",
        "nue_conv", "nue_pr", "nue_total", "nutau_pr",
    ]
    desktop = tempfile.mkdtemp(prefix="mceq_example_")
    destination = os.path.join(desktop, "H3a_zenith_av.txt")
    np.savetxt(
        destination,
        np.column_stack([e_grid, *[flux[frac] for frac in components]]),
        fmt="%6.5E",
        header=(
            "lepton flux scaled with E**3. Order (E, mu_conv, mu_pr, mu_total, "
            "numu_conv, numu_pr, numu_total, nue_conv, nue_pr, nue_total, "
            "nutau_pr)"
        ),
    )
    f"Zenith-averaged spectra saved to `{destination}`."
    return (destination,)


@app.cell(hide_code=True)
def export_notebook(download_jupyter, settings):
    download_jupyter(
        filename="Simple_zenith_averaged_neutrinos.ipynb", configuration=settings
    )
    return


if __name__ == "__main__":
    app.run()
