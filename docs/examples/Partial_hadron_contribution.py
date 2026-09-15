# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo==0.24.0", "nbformat", "MCEq>=2.0.0", "matplotlib>=3.10", "numpy", "crflux>1.0.5"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Partial hadron contributions (vintage)")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Partial hadron contributions — vintage

    Per-mother decomposition of the ground-level lepton fluxes, reproducing
    Figure 9 of [Fedynitch et al., arXiv:1503.00544](https://arxiv.org/abs/1503.00544).
    The original used `set_obs_particles`; the v2 replacement is
    `ParticleManager.track_leptons_from`, so all mother groups coexist in one
    matrix and the whole decomposition is a **single solve**. Thunman primary,
    60° zenith, absolute $E^3$ fluxes. Needs the full public database (charm).
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
        solve_progress,
    )

    return (
        MCEqRun,
        apply_kernel_and_grid,
        config,
        download_jupyter,
        mo,
        np,
        plt,
        solve_progress,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 01 · Configuration

    Primary (Thunman) and zenith (60°) are fixed by the reproduction; controls
    set the hadronic model, cascade grid, and solver.
    """)
    return


@app.cell(hide_code=True)
def configuration_defaults():
    defaults = {
        "database": "mceq_db_lext_dpm193_v142.h5",
        "interaction_model": "SIBYLL23E",
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
        show=("interaction_model", "energy", "backend"),
    )
    controls
    return (controls,)


@app.cell(hide_code=True)
def selected_configuration(controls, defaults, mo):
    from MCEq.notebook import apply_configuration

    settings = apply_configuration(controls, defaults)
    database = settings["database"]
    interaction_model = settings["interaction_model"]
    e_min, e_max = settings["e_min"], settings["e_max"]
    backend = settings["backend"]
    mo.md(
        f"**Applied:** {interaction_model} · Thunman · 60° · "
        f"{e_min:g}–{e_max:g} GeV"
    )
    return (
        backend, database, e_max, e_min, interaction_model, settings,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 02 · Register mother groups and solve once

    Groups are matched against whatever the interaction model actually tracks;
    unlisted members are skipped, so the figure degrades gracefully to the
    species present.
    """)
    return


@app.cell
def _(MCEqRun, apply_kernel_and_grid, backend, config, database, e_max, e_min,
      interaction_model, np):
    import crflux.models as pm

    apply_kernel_and_grid(backend, e_min, e_max, 0)
    config.mceq_db_fname = database

    mceq_run = MCEqRun(
        interaction_model=interaction_model,
        primary_model=(pm.Thunman, None),
        theta_deg=60.0,
    )

    # positive-PDG mothers; antiparticles are pulled in automatically
    mother_groups = [
        ("$D^\\pm$", ["D+", "D-"], "grpD", "orange"),
        ("$D^0$", ["D0", "Dbar0"], "grpD0", "mediumblue"),
        ("$D_s$", ["D_s+", "D_s-"], "grpDs", "cyan"),
    ]

    requested = []
    for _title, members, prefix, _color in mother_groups:
        present = []
        for name in members:
            try:
                present.append(mceq_run.pman[name].pdg_id)
            except KeyError:
                pass
        if present:
            mceq_run.pman.track_leptons_from(present, prefix + "_")
            requested.append(prefix)

    mceq_run.regenerate_matrices()
    e_grid = mceq_run.e_grid
    return (e_grid, mceq_run, mother_groups, requested)


@app.cell
def _(interaction_model, mceq_run, solve_progress):
    with solve_progress(f"{interaction_model} · hadron decomposition") as progress:
        mceq_run.solve(progress=progress)
    "solved"
    return


@app.cell
def _(mceq_run, np):
    mag = 3
    leptons = {
        "mu": ("mu+", "mu-"),
        "numu": ("numu", "antinumu"),
        "nue": ("nue", "antinue"),
        "nutau": ("nutau", "antinutau"),
    }

    def flux(name):
        try:
            return mceq_run.get_solution(name, mag)
        except Exception:
            return np.zeros_like(mceq_run.e_grid)

    def combined(prefix, base):
        return sum(flux(f"{prefix}_{p}") for p in leptons[base])

    def total(base):
        return combined("total", base)

    def conv(base):
        return combined("conv", base)

    def prompt(base):
        return combined("pr", base)

    return combined, conv, flux, mag, prompt, total


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 03 · Four panels, mothers over total

    Black: total (solid), conventional (dotted), prompt (dashed); colours per
    mother species from the tracking tables (D, Ds) and the built-in pion /
    kaon / muon-decay categories.
    """)
    return


@app.cell
def _(combined, conv, e_grid, mother_groups, plt, prompt, requested, total):
    def panel(ax, base, ylim, xlab, ylab, sources):
        ax.loglog()
        ax.plot(e_grid, total(base), ls="-", lw=2, color="black", label="total")
        ax.plot(e_grid, conv(base), ls=":", lw=2, color="black", label="conv.")
        ax.plot(e_grid, prompt(base), ls="--", lw=2, color="black",
                label="prompt")
        for color, name, prefix in sources:
            values = combined(prefix, base)
            if np.all(values <= 0):
                continue
            ax.plot(e_grid, values, ls="-", lw=1.5, color=color, label=name)
        ax.set_xlim(10, 1e9)
        ax.set_ylim(*ylim)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.legend(loc="upper right", frameon=False, numpoints=1, ncol=2,
                  fontsize="small")

    charm_sources = [
        (color, title, prefix)
        for title, _members, prefix, color in mother_groups
        if prefix in requested
    ]
    pion_kaon = [("red", r"$\pi$", "pi"), ("green", "$K$", "k")]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    panel(axes[0, 0], "mu", (1e-6, 1e3), r"$E_{\mu}$ [GeV]",
          r"$\Phi_{\mu}$ (E/GeV)$^{3}$ (cm$^{2}$ s sr GeV)$^{-1}$",
          pion_kaon + charm_sources)
    panel(axes[0, 1], "numu", (1e-7, 1e1), r"$E_{\nu_\mu}$ [GeV]",
          r"$\Phi_{\nu_\mu}$ (E/GeV)$^{3}$ (cm$^{2}$ s sr GeV)$^{-1}$",
          pion_kaon + charm_sources)
    panel(axes[1, 0], "nue", (1e-7, 1e1), r"$E_{\nu_e}$ [GeV]",
          r"$\Phi_{\nu_e}$ (E/GeV)$^{3}$ (cm$^{2}$ s sr GeV)$^{-1}$",
          pion_kaon + charm_sources + [("pink", r"$\mu$ decay", "mu")])
    panel(axes[1, 1], "nutau", (1e-9, 1e-2), r"$E_{\nu_\tau}$ [GeV]",
          r"$\Phi_{\nu_\tau}$ (E/GeV)$^{3}$ (cm$^{2}$ s sr GeV)$^{-1}$",
          charm_sources)
    fig.tight_layout(pad=0.4)
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 03 · Four panels, mothers over total

    Black: total (solid), conventional (dotted), prompt (dashed); colour per
    mother group. The tau panel keeps $\tau$ and the open charm.
    """)
    return


@app.cell(hide_code=True)
def export_notebook(download_jupyter, settings):
    download_jupyter(
        filename="Partial_hadron_contribution.ipynb", configuration=settings
    )
    return


if __name__ == "__main__":
    app.run()
