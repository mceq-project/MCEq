# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo==0.24.0", "nbformat", "MCEq>=2.0.0", "matplotlib>=3.10", "numpy", "crflux>1.0.5"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Dependence on primary CR flux")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dependence on primary cosmic-ray flux

    Zenith-averaged sea-level muon and neutrino fluxes for four primary
    spectra: the knee-region composition models move the prompt component
    far more than the conventional one.
    """)
    return


@app.cell
def _():
    import crflux.models as pm
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from MCEq import config
    from MCEq.core import MCEqRun
    from MCEq.notebook import apply_kernel_and_grid, loop_progress

    return MCEqRun, apply_kernel_and_grid, config, loop_progress, mo, np, plt, pm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 01 · Configuration

    The four primary spectra are the fixed subject: Hillas–Gaisser 2012 H3a
    and H4a (Gaisser 2012) and Gaisser–Stanev–Tilav 3-gen and 4-gen (2013).
    They agree below the knee and differ in how the flux and its nuclear
    composition continue to $10^{10}$ GeV. The form sets the hadronic
    interaction model (`interaction_model` of `MCEqRun`; the list holds every
    model of the v2 release on this grid, extra models are downloaded on first
    use), the cascade energy grid (`config.e_min`, `config.e_max`; the prompt
    flux needs `e_max` well above the plotted range), and the solver backend
    (`config.kernel_config`, `"auto"` picks the fastest available).
    """)
    return


@app.cell(hide_code=True)
def configuration_defaults():
    defaults = {
        "database": "mceq_base_air_1d_v2.h5",
        "interaction_model": "SIBYLL23E",
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
    mo.md(f"**Applied:** {interaction_model} · {e_min:g}–{e_max:g} GeV")
    return backend, database, e_max, e_min, interaction_model, settings


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
    ## 02 · Four primaries per solve, five zenith angles

    Each primary spectrum gives one initial state (`set_primary_model`, then
    `get_initial_state`). Stacked as the columns of `phi0`, the four are
    propagated together by `solve_batch(phi0=...)`: one operator, one
    integration path, four spectra. The zenith average uses the midpoint rule
    on five $\cos\theta$ bins of $[0, 1]$ ($\cos\theta = 0.9, 0.7, \dots,
    0.1$); inclined paths are longer, so the 84° solve dominates the run time.
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
    pm,
    proceed):
    _ = proceed
    apply_kernel_and_grid(backend, e_min, e_max, 0)
    config.mceq_db_fname = database

    primary_models = [
        (pm.HillasGaisser2012, "H3a"),
        (pm.HillasGaisser2012, "H4a"),
        (pm.GaisserStanevTilav, "3-gen"),
        (pm.GaisserStanevTilav, "4-gen"),
    ]
    mceq_run = MCEqRun(
        interaction_model=interaction_model,
        low_energy_model="FLUKA20251",  # below 80 GeV
        primary_model=primary_models[0],
        theta_deg=0.0,
        density_model=("CORSIKA", ("BK_USStd", None)),
    )
    return mceq_run, primary_models


@app.cell
def _(loop_progress, mceq_run, np, primary_models):
    cos_theta = np.linspace(0.9, 0.1, 5)  # bin midpoints of [0, 1]
    angles = np.degrees(np.arccos(cos_theta))

    initial_states, primary_names = [], []
    for primary_model in primary_models:
        mceq_run.set_primary_model(*primary_model)
        initial_states.append(mceq_run.get_initial_state())
        primary_names.append(mceq_run.pmodel.name)
    phi0 = np.stack(initial_states, axis=1)  # column k = primary k

    batches = []
    with loop_progress(len(angles), "zenith scan, 4 primaries per solve") as advance:
        for theta in angles:
            mceq_run.set_zenith_azimuth(float(theta))
            batches.append(mceq_run.solve_batch(phi0=phi0))
            advance(title=f"{theta:.0f}°")
    return angles, batches, primary_names


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 03 · Zenith-averaged fluxes

    Particle + antiparticle sums, weighted by $E^3$ and averaged over the
    five $\cos\theta$ bins. Solid: total; dashed: conventional; dotted:
    prompt.
    """)
    return


@app.cell
def _(batches, mceq_run, np, primary_names):
    energy = mceq_run.e_grid
    mag = 3
    species = {
        "mu": ("mu+", "mu-"),
        "numu": ("numu", "antinumu"),
        "nue": ("nue", "antinue"),
    }
    fluxes = {
        name: {
            lepton: {
                component: np.mean(
                    [
                        sum(
                            batch.get_solution(f"{component}_{particle}", k=k, mag=mag)
                            for particle in particles
                        )
                        for batch in batches
                    ],
                    axis=0,
                )
                for component in ("total", "conv", "pr")
            }
            for lepton, particles in species.items()
        }
        for k, name in enumerate(primary_names)
    }
    return energy, fluxes, mag


@app.cell
def _(energy, fluxes, interaction_model, np, plt):
    plot_range = (10.0, 1.0e8)  # GeV; inside the default cascade grid
    plot_mask = (energy >= plot_range[0]) & (energy <= plot_range[1])
    primary_colors = ["#263342", "#17857E", "#C46B30", "#7B5EA7"]
    component_styles = {"total": ("-", 2.0), "conv": ("--", 1.2), "pr": (":", 1.4)}
    panels = {
        "mu": r"$\mu^+ + \mu^-$",
        "numu": r"$\nu_\mu + \bar\nu_\mu$",
        "nue": r"$\nu_e + \bar\nu_e$",
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
        for ax, (lepton, panel_label) in zip(axes, panels.items()):
            for (name, by_lepton), color in zip(fluxes.items(), primary_colors):
                for component, (linestyle, linewidth) in component_styles.items():
                    values = by_lepton[lepton][component]
                    visible = plot_mask & (values > 0) & np.isfinite(values)
                    ax.loglog(
                        energy[visible], values[visible], color=color,
                        linestyle=linestyle, linewidth=linewidth,
                        label=name if component == "total" else "_nolegend_",
                    )
            ax.set_title(panel_label, loc="left", fontsize=14, pad=12)
            ax.set_xlim(plot_range)
            ax.set_ylim(1.0e-6, 1.0)
            ax.set_xlabel(r"Kinetic energy $E$ [GeV]")
            ax.grid(axis="y", which="major", color="#E6E9ED", linewidth=0.7)
            ax.set_axisbelow(True)

        axes[0].set_ylabel(
            r"$E^3\,\mathrm{d}\Phi/\mathrm{d}E$ [GeV$^2$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]"
        )
        fig.legend(
            *axes[0].get_legend_handles_labels(), frameon=False,
            loc="outside lower center", ncol=4, fontsize=10,
        )
        fig.suptitle(
            f"{interaction_model}  /  zenith-averaged, cos θ ∈ [0, 1]  /  "
            "solid total · dashed conv · dotted prompt",
            fontsize=11, color="#52606D",
        )

    fig
    return (fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 04 · ASCII export

    One text file per primary model with the zenith-averaged $E^3$-weighted
    spectra, for external processing. The files go to a temporary directory;
    change `output_dir` to keep them.
    """)
    return


@app.cell
def _(energy, fluxes, mag, np):
    import re
    import tempfile
    from pathlib import Path

    output_dir = Path(tempfile.mkdtemp(prefix="mceq_primary_fluxes_"))
    columns = [
        (lepton, component)
        for lepton in ("mu", "numu", "nue")
        for component in ("conv", "pr", "total")
    ]
    header = f"lepton flux scaled with E**{mag}. Columns: E [GeV], " + ", ".join(
        f"{lepton}_{component}" for lepton, component in columns
    )
    written = []
    for primary_name, primary_fluxes in fluxes.items():
        stem = re.sub(r"[^A-Za-z0-9.-]+", "_", primary_name).strip("_")
        path = output_dir / f"lepton_flux_{stem}.txt"
        np.savetxt(
            path,
            np.column_stack(
                [energy]
                + [primary_fluxes[lepton][component] for lepton, component in columns]
            ),
            fmt="%6.5E",
            header=header,
        )
        written.append(path.name)
    print(output_dir, written)
    return


@app.cell(hide_code=True)
def export_notebook(settings):
    from MCEq.notebook import download_jupyter

    download_jupyter(filename="compare_primary_fluxes.ipynb", configuration=settings)
    return


if __name__ == "__main__":
    app.run()
