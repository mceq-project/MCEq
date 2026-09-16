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

    Flux ratios relative to the US Standard atmosphere for four alternative
    profiles (CORSIKA South Pole Jan/Aug, MSIS00 South Pole Jan/Aug) at two
    zenith angles. Reproduces Figure 5 of
    [Fedynitch et al., IPRD 2015](http://inspirehep.net/record/1346929);
    values shifted slightly as MCEq evolved.
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
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 01 · Configuration

    The atmosphere set and the two zenith angles are the fixed subject of the
    comparison; the controls tune the baseline run.
    """)
    return


@app.cell(hide_code=True)
def configuration_defaults():
    defaults = {
        "database": "mceq_db_lext_dpm193_v142.h5",
        "interaction_model": "SIBYLL21",
        "primary": "HillasGaisser2012:H3a",
        "theta_deg": 0.0,
        "e_min": 1.0,
        "e_max": 1.0e7,
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
    ## 02 · Compute fluxes for five atmospheres × two zeniths

    Inclined trajectories need more integration steps: the 90° rows are the
    slow ones. Total (conventional + prompt) fluxes are stored per case.
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

    mceq_run = MCEqRun(
        interaction_model=interaction_model,
        primary_model=primary_model,
        theta_deg=0.0,
    )

    atmospheres = [
        (("CORSIKA", ("PL_SouthPole", "January")), "red"),
        (("CORSIKA", ("PL_SouthPole", "August")), "lightblue"),
        (("MSIS00", ("SouthPole", "January")), "darkred"),
        (("MSIS00", ("SouthPole", "August")), "blue"),
        (("CORSIKA", ("BK_USStd", None)), "black"),
    ]

    mag = 3
    mu_flux, numu_flux = {}, {}

    def scan(advance):
        for atmosphere, color in atmospheres:
            mceq_run.set_density_model(atmosphere)
            for theta in [0.0, 90.0]:
                mceq_run.set_zenith_azimuth(theta)
                mceq_run.solve()
                advance(title=atmosphere[0])
                mu_flux[(theta, atmosphere, color)] = (
                    mceq_run.get_solution("total_mu+", mag)
                    + mceq_run.get_solution("total_mu-", mag)
                )
                numu_flux[(theta, atmosphere, color)] = (
                    mceq_run.get_solution("total_numu", mag)
                    + mceq_run.get_solution("total_antinumu", mag)
                )

    with loop_progress(2 * len(atmospheres), "atmosphere x zenith scan") as advance:
        scan(advance)
    e_grid = mceq_run.e_grid
    return e_grid, mu_flux, numu_flux


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 03 · Ratios to US Standard

    Solid: vertical; dashed: 90°. The US Std curves are unity by construction.
    """)
    return


@app.cell
def _(e_grid, mu_flux, numu_flux, plt):
    titles = {
        ("CORSIKA", ("PL_SouthPole", "January")): "CKA SP/Jan",
        ("CORSIKA", ("PL_SouthPole", "August")): "CKA SP/Aug",
        ("MSIS00", ("SouthPole", "January")): "MSIS00 SP/Jan",
        ("MSIS00", ("SouthPole", "August")): "MSIS00 SP/Aug",
        ("CORSIKA", ("BK_USStd", None)): "USStd",
    }
    baseline = ("CORSIKA", ("BK_USStd", None))
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5))
    for axis, flux, name in zip(
        axes, (mu_flux, numu_flux), ("Muons", "Muon neutrinos")
    ):
        for (p_theta, p_profile, p_color) in flux:
            if p_profile == baseline:
                continue
            ls = "-" if p_theta < 90.0 else "--"
            label = titles[p_profile] if p_theta < 90.0 else "_nolabel_"
            axis.semilogx(
                e_grid,
                flux[(p_theta, p_profile, p_color)]
                / flux[(p_theta, baseline, "black")],
                ls=ls, lw=1.5, color=p_color, label=label,
            )
        axis.set_xlim(1, 1e9)
        axis.set_ylim(0.75, 1.13)
        axis.set_title(name, fontsize=10)
        axis.legend(loc="lower left", ncol=2, frameon=False, fontsize=10)
    axes[0].set_xlabel(r"$E_{\mu}$ [GeV]")
    axes[0].set_ylabel(r"$\Phi_{\mu}$(atm)$/\Phi_{\mu}$(USStd)")
    axes[1].set_xlabel(r"$E_{\nu_\mu}$ [GeV]")
    axes[1].set_ylabel(r"$\Phi_{\nu_\mu}$(atm)$/\Phi_{\nu_\mu}$(USStd)")
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def export_notebook(download_jupyter, settings):
    download_jupyter(
        filename="Dependence_of_spectrum_on_atmosphere.ipynb",
        configuration=settings,
    )
    return


if __name__ == "__main__":
    app.run()
