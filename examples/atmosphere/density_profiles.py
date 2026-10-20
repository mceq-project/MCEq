# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo==0.24.0", "nbformat", "MCEq>=2.0.0", "matplotlib>=3.10", "numpy"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Atmosphere density-depth relations")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Atmosphere density–depth relations

    Every atmosphere model MCEq ships, as mass density against slant depth,
    without solving a cascade: the profiles differ most in the upper
    atmosphere, where the mesons of the cascade decide between decay and
    interaction.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from MCEq.core import DENSITY_MODELS
    from MCEq.geometry import atmosphere_parameters

    return DENSITY_MODELS, atmosphere_parameters, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 01 · Configuration

    An atmosphere is selected in `MCEqRun(density_model=(model, config))` or
    `set_density_model`, with `model` a key of `MCEq.core.DENSITY_MODELS`
    (the classes live in `MCEq.geometry.density_profiles` and
    `MCEq.geometry.msis21`); the default is
    `("CORSIKA", ("BK_USStd", None))`. This notebook builds the classes
    directly:

    - **CORSIKA**: Linsley's five-layer parametrizations, `config = (location, season)`
      from `atmosphere_parameters.list_available_corsika_atmospheres()`.
    - **MSIS00 / MSIS21**: NRLMSISE-00 and NRLMSIS 2.1 (the latter needs the
      `nrlmsis` package) at a site of `atmosphere_parameters.LOCATIONS` and a
      month, `config = (site, month)`.
    - **Isothermal**: one scale height (6.3 km) and 1300 g/cm² at the ground; a
      textbook reference, not a realistic atmosphere.
    - AIRS satellite profiles need external tables and are not shown.

    The form picks the site and month of the MSIS profiles and the inclined
    zenith angle shown next to the vertical one. The observation level is sea
    level (`config.h_obs`) for all models, so site-specific fits are
    extrapolated below their ground: the monthly South Pole (`SDR_SouthPole`)
    fits are constrained only above the ice surface at 2.8 km
    (≈ 690 g/cm² vertical, dotted line), and some months, October most of
    all, diverge below it. Set `config.h_obs` to the site altitude before
    building the model to stop the column there.
    """)
    return


@app.cell(hide_code=True)
def configuration_defaults():
    defaults = {
        "site": "SouthPole",
        "month": "January",
        "theta_deg": 70.0,
    }
    return (defaults,)


@app.cell(hide_code=True)
def configuration_controls(atmosphere_parameters, defaults, mo):
    controls = mo.md("""
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,350px),1fr));gap:1rem 2rem">

    {site}

    {month}

    {theta_deg}

    </div>
    """).batch(
        site=mo.ui.dropdown(
            list(atmosphere_parameters.LOCATIONS), value=defaults["site"],
            label="MSIS site", full_width=True,
        ),
        month=mo.ui.dropdown(
            list(atmosphere_parameters.MONTH_TO_DAY_OF_YEAR), value=defaults["month"],
            label="MSIS month", full_width=True,
        ),
        theta_deg=mo.ui.slider(
            0, 85, step=5, value=defaults["theta_deg"],
            label="Inclined zenith angle [°]", show_value=True, full_width=True,
        ),
    ).form(submit_button_label="Apply configuration", bordered=False)
    controls
    return (controls,)


@app.cell(hide_code=True)
def selected_configuration(controls, defaults, mo):
    settings = dict(defaults)
    if controls.value is not None:
        settings.update(controls.value)

    site = settings["site"]
    month = settings["month"]
    theta_deg = settings["theta_deg"]
    mo.md(f"**Applied:** MSIS at {site} in {month} · inclined {theta_deg:g}°")
    return month, settings, site, theta_deg


@app.cell(hide_code=True)
def run_gate(mo):
    gate = mo.ui.run_button(label="Build profiles", kind="success")
    gate
    return (gate,)


@app.cell(hide_code=True)
def run_stop(gate, mo):
    from MCEq.notebook import gate_closed

    mo.stop(
        gate_closed(gate),
        mo.md("Adjust the settings above, then click **Build profiles**."),
    )
    proceed = True
    return (proceed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 02 · Profiles

    `set_theta` integrates the density along the trajectory and fits the
    splines; `max_X` is the slant depth at the observation level and
    `X2rho` the density at slant depth $X$. Logarithmic axes resolve the
    upper atmosphere ($X \lesssim 200$ g/cm²), where the profiles differ
    most and where an isothermal atmosphere is a straight line of slope 1.
    """)
    return


@app.cell
def _(DENSITY_MODELS, atmosphere_parameters, month, np, site, theta_deg, proceed):
    _ = proceed

    def density_curve(atmosphere, theta):
        atmosphere.set_theta(theta)
        depth = np.geomspace(1.0, atmosphere.max_X, 2000)
        return depth, atmosphere.X2rho(depth)

    def build(model, model_config):
        return DENSITY_MODELS[model](*model_config)

    corsika_sets = atmosphere_parameters.list_available_corsika_atmospheres()
    # CORSIKA parametrizations except the monthly South Pole series (SDR_*).
    corsika_curves = {
        f"{location}" + (f" / {season}" if season else ""): density_curve(
            build("CORSIKA", (location, season)), 0.0
        )
        for location, season in corsika_sets
        if not location.startswith("SDR_")
    }
    # Monthly South Pole parametrizations: the seasonal cycle.
    monthly_curves = {
        season: density_curve(build("CORSIKA", (location, season)), 0.0)
        for location, season in corsika_sets
        if location == "SDR_SouthPole"
    }

    family_models = [
        ("CORSIKA US Std", "CORSIKA", ("BK_USStd", None)),
        (f"MSIS00 {site}/{month}", "MSIS00", (site, month)),
        (f"MSIS21 {site}/{month}", "MSIS21", (site, month)),
        ("Isothermal", "Isothermal", (None, None)),
    ]
    family_curves = {}
    for family_label, family_model, family_config in family_models:
        try:
            atmosphere = build(family_model, family_config)
            family_curves[family_label] = [
                density_curve(atmosphere, theta) for theta in (0.0, theta_deg)
            ]
        except ImportError as error:  # MSIS21 without the nrlmsis package
            print(f"Skipping {family_label}: {error}")
    return corsika_curves, family_curves, monthly_curves


@app.cell
def _(corsika_curves, family_curves, monthly_curves, np, plt, theta_deg):
    density_scale = 1.0e3  # g/cm³ -> 10⁻³ g/cm³
    south_pole_surface = 690.0  # g/cm², vertical depth of the ice surface
    family_colors = ["#263342", "#17857E", "#C46B30", "#8A94A0"]

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
            1, 3, figsize=(13.5, 4.8), sharey=True, layout="constrained"
        )

        corsika_colors = plt.cm.tab20(np.linspace(0.0, 1.0, len(corsika_curves)))
        for (label, (depth, rho)), color in zip(corsika_curves.items(), corsika_colors):
            axes[0].plot(depth, density_scale * rho, color=color, linewidth=1.3, label=label)
        axes[0].set_title("CORSIKA parametrizations · vertical", loc="left", fontsize=12)
        axes[0].legend(frameon=False, fontsize=7.5, loc="lower right")

        month_colors = plt.cm.twilight(np.linspace(0.0, 1.0, len(monthly_curves), endpoint=False))
        for (label, (depth, rho)), color in zip(monthly_curves.items(), month_colors):
            axes[1].plot(depth, density_scale * rho, color=color, linewidth=1.3, label=label[:3])
        axes[1].axvline(south_pole_surface, color="#8A94A0", linestyle=":", linewidth=1.0)
        axes[1].set_title("CORSIKA SDR South Pole, monthly · vertical", loc="left", fontsize=12)
        axes[1].legend(frameon=False, fontsize=8, ncol=2, loc="lower right")

        for (label, curves), color in zip(family_curves.items(), family_colors):
            for (depth, rho), linestyle in zip(curves, ("-", "--")):
                axes[2].plot(
                    depth, density_scale * rho, color=color, linestyle=linestyle,
                    linewidth=1.6, label=label if linestyle == "-" else "_nolegend_",
                )
        axes[2].set_title(
            f"Model families · solid 0°, dashed {theta_deg:g}°", loc="left", fontsize=12
        )
        axes[2].legend(frameon=False, fontsize=9, loc="lower right")

        for ax in axes:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(1.0, 4000.0)
            ax.set_xlabel(r"Slant depth $X$ [g/cm$^2$]")
            ax.grid(which="major", color="#E6E9ED", linewidth=0.7)
            ax.set_axisbelow(True)
        axes[0].set_ylim(1.0e-3, 5.0)
        axes[0].set_ylabel(r"Mass density $\rho$ [$10^{-3}$ g/cm$^3$]")

    fig
    return (fig,)


@app.cell(hide_code=True)
def export_notebook(settings):
    from MCEq.notebook import download_jupyter

    download_jupyter(filename="density_profiles.ipynb", configuration=settings)
    return


if __name__ == "__main__":
    app.run()
