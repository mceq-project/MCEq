# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo==0.24.0", "nbformat", "MCEq>=2.0.0", "matplotlib>=3.10", "numpy", "crflux>1.0.5"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Atmospheric lepton fluxes")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Atmospheric lepton fluxes

    Muons and neutrinos at ground level, for one cosmic-ray model and zenith angle.
    """)
    return


@app.cell
def _():
    import crflux.models as pm
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from MCEq import config
    from MCEq.config.detect import resolve_kernel
    from MCEq.config.run import RunConfig
    from MCEq.core import MCEqRun
    from MCEq.notebook import prepare_database

    return MCEqRun, RunConfig, config, mo, np, plt, pm, prepare_database, resolve_kernel


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 01 · Configuration

    Edit this cell and run. Model choices require the corresponding database tables.
    """)
    return


@app.cell
def _(pm):
    # Hadronic interactions
    interaction_model = "SIBYLL2.3e"  # browser: also "QGSJET-II-04"
    database = "mceq_db_lext_dpm193_v142.h5"
    low_energy_model = None        # None: no additional low-energy model
    he_le_transition = 80.0        # GeV; used when low_energy_model is set
    he_le_trwidth = 0.3            # log10(E/GeV) decades, 10–90% width; 0: hard switch

    # Primary cosmic rays and atmosphere
    cr_model = (pm.HillasGaisser2012, "H3a")  # also "H4a", or (pm.GaisserHonda, None)
    theta_deg = 60.0               # zenith angle from vertical [deg]
    atmosphere = ("CORSIKA", ("BK_USStd", None))

    # Cascade energy grid [GeV]; keep a buffer around the plotted range
    e_min = 1.0
    e_max = 1.0e11

    # Solver: auto chooses a CPU backend; CUDA is opt-in
    backend = "auto"              # browser: auto / numpy_etd2; native: also mkl_etd2, accelerate_etd2, cuda_etd2

    return (
        atmosphere,
        backend,
        cr_model,
        database,
        e_max,
        e_min,
        he_le_transition,
        he_le_trwidth,
        interaction_model,
        low_energy_model,
        theta_deg,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 02 · Solve the cascade
    """)
    return


@app.cell
async def _(
    MCEqRun,
    RunConfig,
    atmosphere,
    backend,
    config,
    cr_model,
    database,
    e_max,
    e_min,
    he_le_transition,
    he_le_trwidth,
    interaction_model,
    low_energy_model,
    prepare_database,
    resolve_kernel,
    theta_deg,
):
    config.mceq_db_fname = await prepare_database(database)
    config.e_min, config.e_max = e_min, e_max
    config.kernel_config = resolve_kernel(backend)
    config.debug_level = 0

    # Construct and solve together so configuration edits rebuild the cascade.
    mceq_run = MCEqRun(
        interaction_model=interaction_model,
        primary_model=cr_model,
        theta_deg=theta_deg,
        density_model=atmosphere,
        low_energy_model=low_energy_model,
        he_le_transition=he_le_transition,
        he_le_trwidth=he_le_trwidth,
        config=RunConfig.snapshot(),
    )
    mceq_run.solve()
    return (mceq_run,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 03 · Fluxes at ground

    Particle + antiparticle sums, weighted by $E^3$. Conventional and prompt
    components are shown separately.
    """)
    return


@app.cell
def _(mceq_run):
    energy = mceq_run.e_grid
    species = {
        r"$\mu^+ + \mu^-$": ("mu+", "mu-"),
        r"$\nu_\mu + \bar\nu_\mu$": ("numu", "antinumu"),
        r"$\nu_e + \bar\nu_e$": ("nue", "antinue"),
    }
    fluxes = {
        label: {
            component: sum(
                mceq_run.get_solution(f"{component}_{particle}", mag=3)
                for particle in particles
            )
            for component in ("total", "conv", "pr")
        }
        for label, particles in species.items()
    }
    return energy, fluxes


@app.cell
def _(cr_model, energy, fluxes, interaction_model, np, plt, theta_deg):
    plot_range = (10.0, 1.0e7)  # GeV; inside the cascade grid
    plot_mask = (energy >= plot_range[0]) & (energy <= plot_range[1])
    curve_styles = {
        "total": ("Total", "#263342", "-"),
        "conv": ("Conventional", "#17857E", "--"),
        "pr": ("Prompt", "#C46B30", ":"),
    }

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
        fig, axes = plt.subplots(
            1, 3, figsize=(11.5, 4.4), sharex=True, sharey=True, layout="constrained"
        )
        for ax, (species_label, components) in zip(axes, fluxes.items()):
            for component, (curve_label, color, linestyle) in curve_styles.items():
                values = components[component]
                visible = plot_mask & (values > 0) & np.isfinite(values)
                ax.loglog(
                    energy[visible], values[visible], label=curve_label,
                    color=color, linestyle=linestyle, linewidth=2,
                )
            ax.set_title(species_label, loc="left", fontsize=14, pad=12)
            ax.set_xlim(plot_range)
            ax.set_ylim(1.0e-6, 1.0)
            ax.set_xlabel(r"Kinetic energy $E$ [GeV]")
            ax.grid(axis="y", which="major", color="#E6E9ED", linewidth=0.7)
            ax.set_axisbelow(True)

        axes[0].set_ylabel(r"$E^3\,\mathrm{d}\Phi/\mathrm{d}E$ [GeV$^2$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
        fig.legend(
            *axes[0].get_legend_handles_labels(), frameon=False,
            loc="outside lower center", ncol=3, fontsize=10,
        )
        primary_label = cr_model[1] or cr_model[0].__name__
        fig.suptitle(
            f"{interaction_model}  /  {primary_label}  /  {theta_deg:g}° zenith",
            fontsize=11, color="#52606D",
        )

    fig
    return (fig,)


@app.cell(hide_code=True)
def export_notebook():
    from MCEq.notebook import download_jupyter

    download_jupyter(filename="Basic_flux.ipynb")
    return


if __name__ == "__main__":
    app.run()
