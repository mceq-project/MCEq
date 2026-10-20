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

    Down-going muon and electron neutrino fluxes averaged over the upper
    hemisphere ($\cos\theta \in [0, 1]$), split into conventional and prompt
    components, and written to an ASCII table.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from MCEq import config
    from MCEq.core import MCEqRun
    from MCEq.notebook import apply_kernel_and_grid, loop_progress

    return MCEqRun, apply_kernel_and_grid, config, loop_progress, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 01 · Configuration

    - **Interaction model**: hadronic yields from the release database.
      `SIBYLL23E` (default) and `FLUKA20251` ship in the base package; the
      other models are fetched on first use.
    - **Primary cosmic rays**: crflux spectrum injected at the top of the
      atmosphere; H3a is the common reference.
    - **Energy grid**: `config.e_min`/`config.e_max` cut the cascade grid.
      The upper edge must sit well above the highest energy you plot,
      because the parent flux above it is missing.
    - **Backend**: `config.kernel_config`; `auto` picks MKL or Accelerate when
      available, otherwise NumPy.
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
    ## 02 · Solve each zenith angle

    Eleven zenith angles equidistant in $\cos\theta$, from vertical to
    horizontal. `set_zenith_azimuth` changes the slant path without
    rebuilding the operators, so each `solve()` reuses the same run. The
    near-horizontal paths are the longest (most integration steps).
    """)
    return


@app.cell
def _(MCEqRun, apply_kernel_and_grid, backend, config, database, e_max, e_min,
      interaction_model, primary_model, proceed):
    _ = proceed
    apply_kernel_and_grid(backend, e_min, e_max, 0)
    config.mceq_db_fname = database

    mceq_run = MCEqRun(
        interaction_model=interaction_model,
        primary_model=primary_model,
        theta_deg=0.0,
        density_model=("CORSIKA", ("BK_USStd", None)),
    )
    return (mceq_run,)


@app.cell
def _(loop_progress, mceq_run, np):
    cos_theta = np.linspace(1.0, 0.0, 11)
    angles = np.degrees(np.arccos(cos_theta))
    mag = 3
    pairs = {
        "mu": ("mu+", "mu-"),
        "numu": ("numu", "antinumu"),
        "nue": ("nue", "antinue"),
        "nutau": ("nutau", "antinutau"),
    }
    components = {
        "mu_conv": ("conv", "mu"), "mu_pr": ("pr", "mu"), "mu_total": ("total", "mu"),
        "numu_conv": ("conv", "numu"), "numu_pr": ("pr", "numu"),
        "numu_total": ("total", "numu"),
        "nue_conv": ("conv", "nue"), "nue_pr": ("pr", "nue"),
        "nue_total": ("total", "nue"), "nutau_pr": ("total", "nutau"),
    }
    per_angle = {name: [] for name in components}
    with loop_progress(angles.size, "zenith scan") as advance:
        for theta in angles:
            mceq_run.set_zenith_azimuth(theta)
            mceq_run.solve()
            advance(title=f"{theta:.0f}° zenith")
            for name, (track, group) in components.items():
                per_angle[name].append(
                    sum(
                        mceq_run.get_solution(f"{track}_{particle}", mag=mag)
                        for particle in pairs[group]
                    )
                )
    return mag, per_angle


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 03 · Average over $\cos\theta$

    Particle and antiparticle are summed per angle; the plain mean over the
    equidistant $\cos\theta$ points approximates the hemisphere average.
    """)
    return


@app.cell
def _(mceq_run, np, per_angle):
    e_grid = mceq_run.e_grid
    flux = {name: np.mean(values, axis=0) for name, values in per_angle.items()}
    return e_grid, flux


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 04 · Spectra
    """)
    return


@app.cell
def _(e_grid, flux, interaction_model, mag, np, plt):
    plot_range = (10.0, 1.0e7)  # GeV; inside the cascade grid
    plot_mask = (e_grid >= plot_range[0]) & (e_grid <= plot_range[1])
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
            1, 2, figsize=(9.5, 4.2), sharex=True, sharey=True, layout="constrained"
        )
        for ax, (prefix, label) in zip(
            axes, [("numu", r"\nu_\mu + \bar\nu_\mu"), ("nue", r"\nu_e + \bar\nu_e")]
        ):
            for component, (curve_label, color, linestyle) in curve_styles.items():
                values = flux[f"{prefix}_{component}"]
                visible = plot_mask & (values > 0) & np.isfinite(values)
                ax.loglog(
                    e_grid[visible], values[visible], label=curve_label,
                    color=color, linestyle=linestyle, linewidth=2,
                )
            ax.set_title(f"${label}$", loc="left", fontsize=14, pad=12)
            ax.set_xlim(plot_range)
            ax.set_ylim(1.0e-6, 1.0)
            ax.set_xlabel(r"Kinetic energy $E$ [GeV]")
            ax.grid(axis="y", which="major", color="#E6E9ED", linewidth=0.7)
            ax.set_axisbelow(True)
        axes[0].set_ylabel(
            rf"$E^{mag}\,\langle\Phi\rangle_{{\cos\theta}}$ "
            r"[GeV$^2$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]"
        )
        fig.legend(
            *axes[0].get_legend_handles_labels(), frameon=False,
            loc="outside lower center", ncol=3, fontsize=10,
        )
        fig.suptitle(
            f"{interaction_model}  /  zenith average 0°–90°",
            fontsize=11, color="#52606D",
        )

    fig
    return (fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 05 · ASCII export

    Columns: energy, then the averaged $E^3$-weighted fluxes. The file goes
    to a temporary directory; change `destination` to keep it.
    """)
    return


@app.cell
def _(e_grid, flux, np):
    import os
    import tempfile

    columns = [
        "mu_conv", "mu_pr", "mu_total",
        "numu_conv", "numu_pr", "numu_total",
        "nue_conv", "nue_pr", "nue_total", "nutau_pr",
    ]
    destination = os.path.join(
        tempfile.mkdtemp(prefix="mceq_example_"), "zenith_averaged_fluxes.txt"
    )
    np.savetxt(
        destination,
        np.column_stack([e_grid, *[flux[name] for name in columns]]),
        fmt="%6.5E",
        header="lepton flux scaled with E**3. Columns: E [GeV], " + ", ".join(columns),
    )
    f"Zenith-averaged spectra saved to {destination}"
    return


@app.cell(hide_code=True)
def export_notebook(settings):
    from MCEq.notebook import download_jupyter

    download_jupyter(filename="zenith_averaged_neutrinos.ipynb", configuration=settings)
    return


if __name__ == "__main__":
    app.run()
