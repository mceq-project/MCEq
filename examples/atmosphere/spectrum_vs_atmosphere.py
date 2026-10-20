# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo==0.24.0", "nbformat", "MCEq>=2.0.0", "matplotlib>=3.10", "numpy", "crflux>1.0.5"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="Atmosphere dependence of lepton fluxes",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Atmosphere dependence of lepton fluxes

    Muon and $\nu_\mu$ fluxes at the South Pole in January and August, relative
    to the US Standard atmosphere: a warmer, less dense stratosphere lets more
    pions and kaons decay before they interact. Reproduces Figure 5 of
    [Fedynitch et al. 2015](http://inspirehep.net/record/1346929);
    values shifted slightly as MCEq evolved.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    from MCEq import config
    from MCEq.core import MCEqRun
    from MCEq.notebook import apply_kernel_and_grid, solve_progress

    return MCEqRun, apply_kernel_and_grid, config, mo, plt, solve_progress


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 01 · Configuration

    The atmospheres and the two zenith angles are the fixed subject of the
    comparison: the South Pole in January (austral summer) and August
    (winter), each as a CORSIKA five-layer parametrization
    (`("CORSIKA", ("PL_SouthPole", month))`) and as NRLMSISE-00
    (`("MSIS00", ("SouthPole", month))`), against the US Standard reference
    `("CORSIKA", ("BK_USStd", None))`. The form tunes the baseline run:
    hadronic interaction model, primary spectrum, cascade energy grid, and
    solver backend.
    """)
    return


@app.cell(hide_code=True)
def configuration_defaults():
    defaults = {
        "database": "mceq_base_air_1d_v2.h5",
        "interaction_model": "SIBYLL23E",
        "primary": "HillasGaisser2012:H3a",
        "e_min": 1.0,
        "e_max": 1.0e9,
        "backend": "auto",
    }
    return (defaults,)


@app.cell(hide_code=True)
def configuration_controls(defaults):
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
        backend,
        database,
        e_max,
        e_min,
        interaction_model,
        primary_model,
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
    ## 02 · Five atmospheres × two zenith angles in one batch

    `solve_batch(conditions=...)` takes one dict per member with
    `density_model` and `zenith_deg`; it builds one integration path per
    member and runs all ten on a single operator. The inclined (85°) paths
    are the long ones; the original figure used 90°, which costs about three
    times more steps for nearly the same ratios.
    """)
    return


@app.cell
def _(
    MCEqRun,
    apply_kernel_and_grid,
    backend,
    config,
    database,
    e_max,
    e_min,
    interaction_model,
    primary_model,
    proceed):
    _ = proceed
    apply_kernel_and_grid(backend, e_min, e_max, 0)
    config.mceq_db_fname = database

    baseline = ("CORSIKA", ("BK_USStd", None))
    mceq_run = MCEqRun(
        interaction_model=interaction_model,
        low_energy_model="FLUKA20251",  # below 80 GeV
        primary_model=primary_model,
        theta_deg=0.0,
        density_model=baseline,
    )
    return baseline, mceq_run


@app.cell
def _(baseline, mceq_run, solve_progress):
    atmospheres = {
        "CKA SP/Jan": ("CORSIKA", ("PL_SouthPole", "January")),
        "CKA SP/Aug": ("CORSIKA", ("PL_SouthPole", "August")),
        "MSIS00 SP/Jan": ("MSIS00", ("SouthPole", "January")),
        "MSIS00 SP/Aug": ("MSIS00", ("SouthPole", "August")),
        "US Std": baseline,
    }
    zeniths = (0.0, 85.0)
    members = [(label, theta) for label in atmospheres for theta in zeniths]
    conditions = [
        {"density_model": atmospheres[label], "zenith_deg": theta}
        for label, theta in members
    ]
    with solve_progress(f"{len(members)} atmosphere × zenith members") as progress:
        batch = mceq_run.solve_batch(conditions=conditions, progress=progress)
    return atmospheres, batch, members, zeniths


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 03 · Ratios to US Standard

    Total (conventional + prompt) particle + antiparticle fluxes, divided by
    the US Standard flux at the same zenith angle. Solid: vertical; dashed:
    85°.
    """)
    return


@app.cell
def _(batch, members, mceq_run):
    energy = mceq_run.e_grid
    species = {"mu": ("mu+", "mu-"), "numu": ("numu", "antinumu")}
    fluxes = {
        (lepton, label, theta): sum(
            batch.get_solution(f"total_{particle}", k=k, mag=3)
            for particle in particles
        )
        for lepton, particles in species.items()
        for k, (label, theta) in enumerate(members)
    }
    return energy, fluxes


@app.cell
def _(atmospheres, energy, fluxes, interaction_model, plt, zeniths):
    plot_range = (1.0, 1.0e8)  # GeV; inside the cascade grid
    plot_mask = (energy >= plot_range[0]) & (energy <= plot_range[1])
    colors = {
        "CKA SP/Jan": "#C46B30",
        "CKA SP/Aug": "#17857E",
        "MSIS00 SP/Jan": "#8C3B1C",
        "MSIS00 SP/Aug": "#263342",
    }
    panels = {"mu": r"Muons $\mu^+ + \mu^-$", "numu": r"Muon neutrinos $\nu_\mu + \bar\nu_\mu$"}

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
            1, 2, figsize=(10.5, 4.2), sharey=True, layout="constrained"
        )
        for ax, (lepton, panel_label) in zip(axes, panels.items()):
            ax.axhline(1.0, color="#B3BAC2", linewidth=0.8)
            for label in atmospheres:
                if label not in colors:
                    continue
                for theta, linestyle in zip(zeniths, ("-", "--")):
                    reference = fluxes[(lepton, "US Std", theta)]
                    valid = plot_mask & (reference > 0)
                    ratio = fluxes[(lepton, label, theta)][valid] / reference[valid]
                    ax.semilogx(
                        energy[valid], ratio, color=colors[label],
                        linestyle=linestyle, linewidth=1.8,
                        label=label if theta == zeniths[0] else "_nolegend_",
                    )
            ax.set_title(panel_label, loc="left", fontsize=13, pad=10)
            ax.set_xlim(plot_range)
            ax.set_ylim(0.75, 1.15)
            ax.set_xlabel(r"Kinetic energy $E$ [GeV]")
            ax.grid(axis="y", which="major", color="#E6E9ED", linewidth=0.7)
            ax.set_axisbelow(True)
        axes[0].set_ylabel(r"$\Phi(\mathrm{atm})\,/\,\Phi(\mathrm{US\ Std})$")
        fig.legend(
            *axes[0].get_legend_handles_labels(), frameon=False,
            loc="outside lower center", ncol=4, fontsize=10,
        )
        fig.suptitle(
            f"{interaction_model}  /  solid {zeniths[0]:g}° · dashed {zeniths[1]:g}°",
            fontsize=11, color="#52606D",
        )
    fig
    return (fig,)


@app.cell(hide_code=True)
def export_notebook(settings):
    from MCEq.notebook import download_jupyter

    download_jupyter(filename="spectrum_vs_atmosphere.ipynb", configuration=settings)
    return


if __name__ == "__main__":
    app.run()
