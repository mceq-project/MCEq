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
    from MCEq.core import MCEqRun
    from MCEq.notebook import solve_progress

    return (
        MCEqRun,
        config,
        mo,
        np,
        plt,
        pm,
        resolve_kernel,
        solve_progress,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 01 · Configuration

    Interaction model, cosmic-ray flux, and cascade energy range.
    """)
    return


@app.cell(hide_code=True)
def configuration_defaults():
    defaults = {
        "interaction_model": "SIBYLL2.3e",
        "database": "mceq_base_air_1d_v2.h5",
        "low_energy_model": "FLUKA20251",
        "he_le_transition": 80.0,
        "he_le_trwidth": 0.3,
        "cr_model": "H3a",
        "theta_deg": 60.0,
        "e_min": 1.0,
        "e_max": 1.0e11,
        "backend": "auto",
    }
    return (defaults,)


@app.cell(hide_code=True)
def configuration_controls(defaults, mo):
    import math

    from MCEq.notebook import backend_options as _backend_options

    backend_options = _backend_options()

    controls = mo.md("""
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,350px),1fr));gap:1.5rem 2rem">
    <div>

    <strong>Interactions</strong>

    {interaction_model}

    {low_energy_model}

    </div><div>

    <strong>Primary & solver</strong>

    {cr_model}

    {backend}

    </div><div>

    <strong>Energy grid</strong>

    {log_e_min}

    {log_e_max}

    </div><div>

    <strong>Geometry & transition</strong>

    {theta_deg}

    {he_le_transition}

    {he_le_trwidth}

    </div></div>
    """).batch(
        interaction_model=mo.ui.dropdown(
            ["SIBYLL2.3e", "QGSJET-II-04", "EPOS-LHC", "DPMJET-III-19.3"],
            value=defaults["interaction_model"],
            label="Interaction model", full_width=True,
        ),
        low_energy_model=mo.ui.dropdown(
            {"FLUKA 2025.1": "FLUKA20251", "None": "none"},
            value="FLUKA 2025.1", full_width=True,
            label="Low-energy model (below the transition)",
        ),
        cr_model=mo.ui.dropdown(
            ["H3a", "H4a", "Gaisser–Honda"], value=defaults["cr_model"],
            label="Cosmic-ray model", full_width=True,
        ),
        backend=mo.ui.dropdown(
            backend_options, value="Automatic", label="Backend", full_width=True,
        ),
        log_e_min=mo.ui.slider(
            -1, 2, step=0.5, value=math.log10(defaults["e_min"]),
            label="e_min · log₁₀(E/GeV)", show_value=True, full_width=True,
        ),
        log_e_max=mo.ui.slider(
            7, 11, step=0.5, value=math.log10(defaults["e_max"]),
            label="e_max · log₁₀(E/GeV)", show_value=True, full_width=True,
        ),
        theta_deg=mo.ui.slider(
            0, 90, step=5, value=defaults["theta_deg"],
            label="Zenith angle [°]", show_value=True, full_width=True,
        ),
        he_le_transition=mo.ui.slider(
            10, 200, step=10, value=defaults["he_le_transition"],
            label="Low-energy transition [GeV]", show_value=True, full_width=True,
        ),
        he_le_trwidth=mo.ui.slider(
            0, 1, step=0.1, value=defaults["he_le_trwidth"],
            label="Transition width [decades]", show_value=True, full_width=True,
        ),
    ).form(submit_button_label="Apply configuration", bordered=False)
    controls
    return (controls,)


@app.cell(hide_code=True)
def selected_configuration(controls, defaults, mo, pm):
    settings = dict(defaults)
    if controls.value is not None:
        settings.update(controls.value)
        settings["e_min"] = 10.0 ** settings.pop("log_e_min")
        settings["e_max"] = 10.0 ** settings.pop("log_e_max")
        settings["low_energy_model"] = (
            None if settings["low_energy_model"] == "none" else settings["low_energy_model"]
        )

    interaction_model = settings["interaction_model"]
    database = settings["database"]
    low_energy_model = settings["low_energy_model"]
    he_le_transition = settings["he_le_transition"]
    he_le_trwidth = settings["he_le_trwidth"]
    cr_model = (
        (pm.GaisserHonda, None) if settings["cr_model"] == "Gaisser–Honda"
        else (pm.HillasGaisser2012, settings["cr_model"])
    )
    theta_deg = settings["theta_deg"]
    atmosphere = ("CORSIKA", ("BK_USStd", None))
    e_min, e_max = settings["e_min"], settings["e_max"]
    backend = settings["backend"]
    mo.md(
        f"**Applied:** {interaction_model} · {settings['cr_model']} · "
        f"{theta_deg:g}° · {e_min:g}–{e_max:g} GeV"
    )
    return (
        atmosphere, backend, cr_model, database, e_max, e_min,
        he_le_transition, he_le_trwidth, interaction_model, low_energy_model,
        settings, theta_deg,
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
    ## 02 · Solve the cascade
    """)
    return


@app.cell
def _(
    MCEqRun,
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
    resolve_kernel,
    theta_deg,
    proceed):
    _ = proceed
    config.mceq_db_fname = database
    config.e_min, config.e_max = e_min, e_max
    config.kernel_config = resolve_kernel(backend)
    config.debug_level = 0

    # Configuration edits rebuild the run; the next cell solves it.
    mceq_run = MCEqRun(
        interaction_model=interaction_model,
        primary_model=cr_model,
        theta_deg=theta_deg,
        density_model=atmosphere,
        low_energy_model=low_energy_model,
        he_le_transition=he_le_transition,
        he_le_trwidth=he_le_trwidth,
    )
    return (mceq_run,)


@app.cell
def _(interaction_model, mceq_run, solve_progress, theta_deg):
    with solve_progress(
        f"{interaction_model} · {theta_deg:g}° zenith"
    ) as progress:
        mceq_run.solve(progress=progress)
    "solved"
    return


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
def export_notebook(settings):
    from MCEq.notebook import download_jupyter

    download_jupyter(filename="basic_flux.ipynb", configuration=settings)
    return


if __name__ == "__main__":
    app.run()
