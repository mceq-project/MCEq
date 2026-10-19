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

    The density classes directly, without solving a cascade: mass density
    against slant depth for five profiles at vertical and 70°.
    Reproduces Figure 4 of
    [Fedynitch et al., IPRD 2015](http://inspirehep.net/record/1346929).
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from MCEq.environment import registry
    from MCEq.notebook import download_jupyter

    return download_jupyter, mo, np, plt, registry


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 01 · Configuration

    This notebook touches only the atmosphere layer, so no interaction model,
    primary spectrum, or solver is configured; the profile set below is the
    subject.
    """)
    return


@app.cell(hide_code=True)
def selected_configuration():
    settings = {}
    return (settings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 02 · Profiles

    `set_theta` triggers the path integration and cache look-up; `max_X` is
    the slant depth at the observation level for the current angle.
    """)
    return


@app.cell
def _(np, registry):
    profiles = [
        ("US Standard", "black", ("CORSIKA", ("BK_USStd", None))),
        ("CKA, SP/Aug", "red", ("CORSIKA", ("PL_SouthPole", "August"))),
        ("CKA, SP/Jan", "green", ("CORSIKA", ("PL_SouthPole", "January"))),
        ("MSIS00, SP/Aug", "cyan", ("MSIS00", ("SouthPole", "August"))),
        ("MSIS00, SP/Jan", "blue", ("MSIS00", ("SouthPole", "January"))),
    ]
    built = []
    for profile_name, profile_color, model in profiles:
        built.append(
            (profile_name, profile_color, registry.build(model[0], model[1]))
        )
    return (built,)


@app.cell
def _(built, np, plt):
    fig, ax = plt.subplots(figsize=(4, 3))
    for theta, linestyle in [(0.0, "-"), (70.0, "--")]:
        for name, color, profile_obj in built:
            profile_obj.set_theta(theta)
            x_vec = np.linspace(0, profile_obj.max_X, 10000)
            ax.plot(
                x_vec, 1e3 / profile_obj.r_X2rho(x_vec),
                label=name if linestyle == "-" else "_nolabel_",
                lw=1.5, ls=linestyle, color=color,
            )
    ax.set_xlim(0, 4000)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax.set_xlabel("slant depth $X$ in [g/cm$^2$]")
    ax.set_ylabel(r"mass density $\rho$ in [10$^{-3}$g/cm$^3$]")
    ax.set_title("solid: vertical · dashed: 70°", fontsize=9)
    fig.tight_layout()
    fig
    return (fig,)


@app.cell(hide_code=True)
def export_notebook(download_jupyter, settings):
    download_jupyter(
        filename="Plot_density_depth_relations.ipynb", configuration=settings
    )
    return


if __name__ == "__main__":
    app.run()
