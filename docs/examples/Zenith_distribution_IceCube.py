# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo==0.24.0", "nbformat", "MCEq>=2.0.0", "matplotlib>=3.10", "numpy", "crflux>1.0.5"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Zenith distribution at IceCube")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Zenith distribution of neutrino fluxes

    Neutrino and muon fluxes at fixed energy versus $\cos\theta$ for the two
    MSIS00 seasons at the South Pole. Zenith runs 0° (downward) to 180°
    (upward through Earth); the atmosphere is centered on IceCube, and here
    geomagnetic cutoffs are off, so azimuth averaging is exact.
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

    The season pair and the 37-bin $\cos\theta$ grid are the fixed subject of
    the comparison.
    """)
    return


@app.cell(hide_code=True)
def configuration_defaults():
    defaults = {
        "database": "mceq_db_lext_dpm193_v142.h5",
        "interaction_model": "SIBYLL21",
        "primary": "HillasGaisser2012:H3a",
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
def run_gate(mo):
    gate = mo.ui.run_button(label="Build & solve", kind="success")
    gate
    return (gate,)


@app.cell(hide_code=True)
def run_stop(gate, mo):
    interactive = mo.app_meta().mode in ("edit", "run")
    mo.stop(
        interactive and not gate.value,
        mo.md("Adjust the settings above, then click **Build & solve**."),
    )
    proceed = True
    return (proceed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 02 · Spectra per zenith and season

    One solve per (zenith, season); the full spectrum is stored and single
    energy bins are extracted at plot time.
    """)
    return


@app.cell
def _(MCEqRun, apply_kernel_and_grid, backend, config, database, e_max, e_min,
      interaction_model, np, primary_model, proceed):
    _ = proceed
    apply_kernel_and_grid(backend, e_min, e_max, 0)
    config.mceq_db_fname = database

    cos_theta = np.linspace(-1, 1, 37)
    angles = np.arccos(cos_theta) / np.pi * 180.0
    mag = 3

    tracks = {
        "mu_conv": ("conv", ("mu+", "mu-")),
        "mu_pr": ("pr", ("mu+", "mu-")),
        "mu_total": ("total", ("mu+", "mu-")),
        "mu_pi": ("pi", ("mu+", "mu-")),
        "mu_k": ("k", ("mu+", "mu-")),
        "numu_conv": ("conv", ("numu", "antinumu")),
        "numu_pr": ("pr", ("numu", "antinumu")),
        "numu_total": ("total", ("numu", "antinumu")),
        "numu_pi": ("pi", ("numu", "antinumu")),
        "numu_k": ("k", ("numu", "antinumu")),
        "nue_conv": ("conv", ("nue", "antinue")),
        "nue_pr": ("pr", ("nue", "antinue")),
        "nue_total": ("total", ("nue", "antinue")),
        "nue_pi": ("pi", ("nue", "antinue")),
        "nue_k": ("k", ("nue", "antinue")),
        "nutau_pr": ("total", ("nutau", "antinutau")),
    }

    mceq_run = MCEqRun(
        interaction_model=interaction_model,
        primary_model=primary_model,
        theta_deg=0.0,
        density_model=("MSIS00_IC", ("SouthPole", "January")),
    )
    e_grid = mceq_run.e_grid

    all_flux = {}

    def scan(advance):
        for season, atmosphere in [
            ("summer", ("MSIS00_IC", ("SouthPole", "January"))),
            ("winter", ("MSIS00_IC", ("SouthPole", "July"))),
        ]:
            mceq_run.set_density_model(atmosphere)
            season_flux = {frac: [] for frac in tracks}
            for theta in angles:
                mceq_run.set_zenith_azimuth(theta)
                mceq_run.solve()
                advance(title=f"{season} · {theta:.0f} deg")
                for frac, (track, particles) in tracks.items():
                    season_flux[frac].append(
                        sum(
                            mceq_run.get_solution(f"{track}_{p}", mag)
                            for p in particles
                        )
                    )
            all_flux[season] = season_flux

    with loop_progress(
        2 * len(angles), "zenith distribution, both seasons"
    ) as advance:
        scan(advance)
    return (all_flux, cos_theta, e_grid, mag)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 03 · Muon neutrinos, three energies

    Solid: summer (January); dashed: winter (July). Blue/green/red: pion, kaon,
    prompt contributions. The energy index is clamped to the grid, which is
    shallower in the reduced database.
    """)
    return


@app.cell
def _(all_flux, cos_theta, e_grid, mag, np, plt):
    def draw_numu():
        fig_numu, axes = plt.subplots(1, 3, sharex=True, figsize=(9.8, 3.0))
        for i, (e_target, ylim) in enumerate(
            zip([10.0, 10e3, 1e6], [2e-5, 4e-14, 6e-22])
        ):
            ax = axes[i]
            idx = min(
                int(np.argmin(np.abs(e_target - e_grid))) + 1, e_grid.size - 1
            )
            ax.text(0.01, 0.9, rf" $E(\nu_\mu)$={e_grid[idx]:2.1e} GeV",
                    transform=ax.transAxes)
            for sid, season in enumerate(["summer", "winter"]):
                for frac, color, label in [
                    ("pi", "b", r"from $\pi$ decay"),
                    ("k", "g", "from $K$ decay"),
                    ("pr", "r", "from prompt decays"),
                ]:
                    ax.plot(
                        cos_theta,
                        [f[idx] / e_grid[idx] ** mag
                         for f in all_flux[season][f"numu_{frac}"]],
                        label=(label if (i == 0 and sid == 0) else "_nolabel_"),
                        ls="-" if sid == 0 else "--", lw=1.5, color=color,
                    )
            ax.set_ylim(0, ylim)
            ax.set_xlabel(r"$\cos(\theta)$ (IceCube)")
        axes[0].set_ylabel(r"$\Phi_{\nu_\mu}$ [cm$^{2}$ s sr GeV]$^{-1}$")
        fig_numu.legend(*axes[0].get_legend_handles_labels(), loc=(0.45, 0.9),
                        ncol=3, frameon=False)
        fig_numu.tight_layout(rect=[0.0, 0.0, 1.0, 0.92], pad=0.05)
        return fig_numu

    draw_numu()
    return


@app.cell
def _(all_flux, cos_theta, e_grid, mag, np, plt):
    def draw_muons():
        fig_mu, axes = plt.subplots(1, 3, sharex=True, figsize=(9.8, 3.0))
        for i, (e_target, ylim) in enumerate(
            zip([10.0, 10e3, 1e6], [7e-5, 12e-14, 15e-22])
        ):
            ax = axes[i]
            idx = min(
                int(np.argmin(np.abs(e_target - e_grid))) + 1, e_grid.size - 1
            )
            ax.text(0.01, 0.9, rf" $E(\mu)$={e_grid[idx]:2.1e} GeV",
                    transform=ax.transAxes)
            for sid, season in enumerate(["summer", "winter"]):
                for frac, color, label in [
                    ("pi", "b", r"from $\pi$ decay"),
                    ("k", "g", "from $K$ decay"),
                    ("pr", "r", "from prompt decays"),
                ]:
                    ax.plot(
                        cos_theta,
                        [f[idx] / e_grid[idx] ** mag
                         for f in all_flux[season][f"mu_{frac}"]],
                        label=(label if (i == 0 and sid == 0) else "_nolabel_"),
                        ls="-" if sid == 0 else "--", lw=1.5, color=color,
                    )
            ax.set_ylim(0, ylim)
            ax.set_xlabel(r"$\cos(\theta)$ (IceCube)")
        axes[0].set_ylabel(r"$\Phi_{\mu}$ [cm$^{2}$ s sr GeV]$^{-1}$")
        axes[0].set_xlim(0, 1)
        fig_mu.legend(*axes[0].get_legend_handles_labels(), loc=(0.45, 0.9),
                      ncol=3, frameon=False)
        fig_mu.tight_layout(rect=[0.0, 0.0, 1.0, 0.92], pad=0.05)
        return fig_mu

    draw_muons()
    return


@app.cell(hide_code=True)
def export_notebook(download_jupyter, settings):
    download_jupyter(
        filename="Zenith_distribution_IceCube.ipynb", configuration=settings
    )
    return


if __name__ == "__main__":
    app.run()
