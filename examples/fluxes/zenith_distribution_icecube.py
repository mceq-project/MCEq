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
    # Zenith distribution of neutrino fluxes at IceCube

    Muon-neutrino and muon fluxes at fixed energy versus $\cos\theta$, split
    by parent (pion, kaon, prompt), for South-Pole summer and winter.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from MCEq import config
    from MCEq.core import MCEqRun
    from MCEq.notebook import apply_kernel_and_grid, solve_progress

    return MCEqRun, apply_kernel_and_grid, config, mo, np, plt, solve_progress


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 01 · Configuration

    The atmosphere is fixed to `MSIS00_IC`: NRLMSISE-00 profiles centred on
    IceCube, so a zenith angle covers the full sky from 0° (down-going) to
    180° (up-going, the cascade develops above the far side of the Earth).
    January and July are the austral summer and winter. The controls set
    the hadronic model, the primary spectrum, the cascade grid and the
    backend; keep `e_max` at least one decade above the highest plotted
    energy ($10^6$ GeV).
    """)
    return


@app.cell(hide_code=True)
def configuration_defaults():
    defaults = {
        "database": "mceq_base_air_1d_v2.h5",
        "interaction_model": "SIBYLL23E",
        "primary": "HillasGaisser2012:H3a",
        "e_min": 1.0,
        "e_max": 1.0e8,
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
    from MCEq.notebook import gate_closed

    mo.stop(
        gate_closed(gate),
        mo.md("Adjust the settings above, then click **Build & solve**."),
    )
    proceed = True
    return (proceed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 02 · Batch over zenith and season

    Twenty $\cos\theta$ bin centres times two seasons are 40 independent
    cascades sharing one operator. `solve_batch(conditions=...)` takes one
    dict per member (`zenith_deg`, `density_model`), builds one integration
    path each, and packs them onto a carousel of `carousel_K` columns that
    advance together. A narrow carousel keeps the columns busy, because the
    near-horizontal paths need several times more steps than the vertical
    ones.
    """)
    return


@app.cell
def _(MCEqRun, apply_kernel_and_grid, backend, config, database, e_max, e_min,
      interaction_model, primary_model, proceed):
    _ = proceed
    apply_kernel_and_grid(backend, e_min, e_max, 0)
    config.mceq_db_fname = database

    seasons = {
        "summer": ("MSIS00_IC", ("SouthPole", "January")),
        "winter": ("MSIS00_IC", ("SouthPole", "July")),
    }
    mceq_run = MCEqRun(
        interaction_model=interaction_model,
        low_energy_model="FLUKA20251",  # below 80 GeV
        primary_model=primary_model,
        theta_deg=0.0,
        density_model=seasons["summer"],
    )
    return mceq_run, seasons


@app.cell
def _(interaction_model, mceq_run, np, seasons, solve_progress):
    cos_theta = np.linspace(-0.95, 0.95, 20)
    angles = np.degrees(np.arccos(cos_theta))
    conditions = [
        {"zenith_deg": float(zenith), "density_model": atmosphere}
        for atmosphere in seasons.values()
        for zenith in angles
    ]
    with solve_progress(
        f"{interaction_model} · {len(conditions)} cascades"
    ) as progress:
        result = mceq_run.solve_batch(
            conditions=conditions, carousel_K=8, progress=progress
        )
    result
    return cos_theta, result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 03 · Fluxes by parent

    The lepton prefixes `pi_`, `k_` and `pr_` of `get_solution` select the
    decays of pions, kaons and charmed hadrons. Column `k` of the batch is
    season-major: `k = i_season * n_zenith + i_zenith`.
    """)
    return


@app.cell
def _(cos_theta, mceq_run, np, result, seasons):
    e_grid = mceq_run.e_grid
    targets = [10.0, 1.0e4, 1.0e6]  # GeV
    e_idx = [int(np.argmin(np.abs(e_grid - energy))) for energy in targets]
    lepton_pairs = {"numu": ("numu", "antinumu"), "mu": ("mu+", "mu-")}
    # zenith_flux[lepton][parent][season] has shape (n_energies, n_zenith)
    zenith_flux = {
        lepton: {
            parent: {
                season: np.array([
                    [
                        sum(
                            result.get_solution(
                                f"{parent}_{particle}",
                                k=i_season * cos_theta.size + i_zen,
                            )[idx]
                            for particle in particles
                        )
                        for i_zen in range(cos_theta.size)
                    ]
                    for idx in e_idx
                ])
                for i_season, season in enumerate(seasons)
            }
            for parent in ("pi", "k", "pr")
        }
        for lepton, particles in lepton_pairs.items()
    }
    e_plot = e_grid[e_idx]
    return e_plot, zenith_flux


@app.cell
def _(cos_theta, e_plot, interaction_model, plt, zenith_flux):
    def zenith_figure(lepton, symbol, cos_range):
        parent_styles = {
            "pi": (r"$\pi$ decays", "#17857E"),
            "k": (r"$K$ decays", "#263342"),
            "pr": ("prompt", "#C46B30"),
        }
        season_styles = {"summer": "-", "winter": "--"}
        with plt.rc_context({
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#263342",
            "axes.edgecolor": "#B3BAC2",
            "xtick.color": "#52606D",
            "ytick.color": "#52606D",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }):
            figure, axes = plt.subplots(
                1, 3, figsize=(11.5, 3.9), sharex=True, layout="constrained"
            )
            for i_e, ax in enumerate(axes):
                for parent, (parent_label, color) in parent_styles.items():
                    for season, linestyle in season_styles.items():
                        ax.plot(
                            cos_theta,
                            zenith_flux[lepton][parent][season][i_e],
                            color=color, linestyle=linestyle, linewidth=1.8,
                            label=f"{parent_label}, {season}",
                        )
                ax.set_title(
                    rf"$E_{{{symbol}}}$ = {e_plot[i_e]:.2g} GeV",
                    loc="left", fontsize=12, pad=10,
                )
                ax.set_xlim(cos_range)
                ax.set_ylim(bottom=0)
                ax.set_xlabel(r"$\cos\theta$ (IceCube)")
                ax.grid(axis="y", color="#E6E9ED", linewidth=0.7)
                ax.set_axisbelow(True)
            axes[0].set_ylabel(
                rf"$\Phi_{{{symbol}}}$ [GeV$^{{-1}}$ cm$^{{-2}}$ s$^{{-1}}$ sr$^{{-1}}$]"
            )
            figure.legend(
                *axes[0].get_legend_handles_labels(), frameon=False,
                loc="outside lower center", ncol=3, fontsize=10,
            )
            figure.suptitle(
                f"{interaction_model}  /  solid: summer (January), "
                "dashed: winter (July)",
                fontsize=11, color="#52606D",
            )
        return figure

    fig_numu = zenith_figure("numu", r"\nu_\mu + \bar\nu_\mu", (-1.0, 1.0))
    fig_numu
    return (zenith_figure,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Conventional neutrinos peak at the horizon, where pions and kaons decay
    before they interact; the prompt flux is nearly isotropic. The seasonal
    difference comes from the temperature of the production layer and is
    largest for the pion component at high energy. Muons are shown for the
    down-going hemisphere only (the Earth absorbs the up-going ones).
    """)
    return


@app.cell
def _(zenith_figure):
    fig = zenith_figure("mu", r"\mu^+ + \mu^-", (0.0, 1.0))
    fig
    return (fig,)


@app.cell(hide_code=True)
def export_notebook(settings):
    from MCEq.notebook import download_jupyter

    download_jupyter(filename="zenith_distribution_icecube.ipynb", configuration=settings)
    return


if __name__ == "__main__":
    app.run()
