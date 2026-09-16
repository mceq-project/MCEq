# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo==0.24.0", "nbformat", "MCEq>=2.0.0", "matplotlib>=3.10", "numpy", "crflux>1.0.5"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Data-Driven Model (DDM)")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data-Driven Model (DDM)

    The DDM replaces selected charm-production channels of a baseline hadronic
    model while keeping the rest
    ([paper](https://inspirehep.net/literature/2088973)). Compare baseline
    vs DDM-injected fluxes and ratios.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    from MCEq import config
    from MCEq.core import MCEqRun
    from MCEq.notebook import (
        apply_kernel_and_grid,
        download_jupyter,
        solve_progress,
    )

    mpl.rcParams["figure.dpi"] = 150
    return MCEqRun, apply_kernel_and_grid, config, download_jupyter, mo, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 01 · Configuration
    """)
    return


@app.cell(hide_code=True)
def configuration_defaults():
    defaults = {
        "database": "mceq_db_lext_dpm193_v142.h5",
        "interaction_model": "SIBYLL21",
        "primary": "HillasGaisser2012:H3a",
        "theta_deg": 60.0,
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
        show=("interaction_model", "primary", "zenith", "energy", "backend"),
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
    theta_deg = settings["theta_deg"]
    e_min, e_max = settings["e_min"], settings["e_max"]
    backend = settings["backend"]
    mo.md(
        f"**Applied:** {interaction_model} · {settings['primary']} · "
        f"{theta_deg:g}° · {e_min:g}–{e_max:g} GeV"
    )
    return (
        backend,
        database,
        e_max,
        e_min,
        interaction_model,
        primary_model,
        settings,
        theta_deg,
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
    ## 02 · Baseline vs DDM

    Two runs share the configuration; `inject_ddm` swaps the charm channels of
    the second. The DDM needs a baseline because it only overrides selected
    production channels.
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
    theta_deg,
    proceed):
    _ = proceed
    from MCEq.models.ddm.ddm import DataDrivenModel

    apply_kernel_and_grid(backend, e_min, e_max, 0)
    config.mceq_db_fname = database

    ddm = DataDrivenModel(str(config.data_dir / "DDM_1.0.npy"))
    ddm_summary = str(ddm).strip().replace("\n", " · ")

    mceq_default = MCEqRun(
        interaction_model=interaction_model,
        primary_model=primary_model,
        theta_deg=theta_deg,
    )
    mceq_ddm = MCEqRun(
        interaction_model=interaction_model,
        primary_model=primary_model,
        theta_deg=theta_deg,
    )
    mceq_ddm.inject_ddm(ddm)

    mceq_default.set_zenith_azimuth(theta_deg)
    mceq_ddm.set_zenith_azimuth(theta_deg)
    return ddm_summary, mceq_ddm, mceq_default


@app.cell
def _(interaction_model, mceq_ddm, mceq_default, solve_progress, theta_deg):
    with solve_progress(
        f"{interaction_model} · {theta_deg:g}° · default"
    ) as progress:
        mceq_default.solve(progress=progress)
    with solve_progress(
        f"{interaction_model} · {theta_deg:g}° · DDM 1.0"
    ) as progress:
        mceq_ddm.solve(progress=progress)
    "both solved"
    return


@app.cell(hide_code=True)
def configuration_summary(ddm_summary, mo):
    mo.md(f"""
    **Injected DDM table:** {ddm_summary}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 03 · Fluxes and ratios

    Solid: total; dashed: conventional. Lower panels show baseline/DDM.
    """)
    return


@app.cell
def _(mceq_ddm, mceq_default, plt):
    mag = 3
    e_grid = mceq_default.e_grid

    def spectra(run):
        sums = {
            key: sum(
                run.get_solution(f"{component}_{p}", mag) for p in particles
            )
            for key, component, particles in [
                ("mu_conv", "conv", ("mu+", "mu-")),
                ("mu_total", "total", ("mu+", "mu-")),
                ("numu_conv", "conv", ("numu", "antinumu")),
                ("numu_total", "total", ("numu", "antinumu")),
                ("nue_conv", "conv", ("nue", "antinue")),
                ("nue_total", "total", ("nue", "antinue")),
            ]
        }
        sums.update({
            "mu_charge": run.get_solution("total_mu+", mag)
            / run.get_solution("total_mu-", mag),
            "numu_ratio": run.get_solution("total_numu", mag)
            / run.get_solution("total_antinumu", mag),
            "nue_ratio": run.get_solution("total_nue", mag)
            / run.get_solution("total_antinue", mag),
        })
        return sums

    flux_ddm, flux_base = spectra(mceq_ddm), spectra(mceq_default)

    def compare(total_key, conv_key, xlabel, ylabel, ylim):
        fig, axes = plt.subplots(
            2, 1, figsize=(4.5, 4.5), sharex=True,
            gridspec_kw={"height_ratios": (3, 1)},
        )
        color = axes[0].loglog(e_grid, flux_ddm[total_key], ls="-", lw=1.5,
                               label="DDM")[0].get_color()
        axes[0].loglog(e_grid, flux_ddm[conv_key], ls="--", lw=1.5, color=color)
        color2 = axes[0].loglog(e_grid, flux_base[total_key], ls="-", lw=1.5,
                                label="baseline", alpha=0.55)[0].get_color()
        axes[0].loglog(e_grid, flux_base[conv_key], ls="--", lw=1.5,
                       color=color2, alpha=0.55)
        axes[1].semilogx(e_grid, flux_base[total_key] / flux_ddm[total_key],
                         ls="-", lw=1.5, c="k")
        axes[1].semilogx(e_grid, flux_base[conv_key] / flux_ddm[conv_key],
                         ls="--", lw=1.5, c="k")
        axes[0].set_xlim(10, 1e7)
        if ylim:
            axes[0].set_ylim(*ylim)
        axes[0].set_ylabel(ylabel)
        axes[0].legend(loc="upper right", frameon=False, numpoints=1)
        axes[1].set_ylim(0.82, 1.38)
        axes[1].set_ylabel("baseline/DDM")
        axes[1].set_xlabel(xlabel)
        fig.tight_layout(h_pad=-0.5)
        return fig

    fig_mu = compare(
        "mu_total", "mu_conv", r"$E_{\mu}$ [GeV]",
        r"$\Phi_{\mu}$ (E/GeV)$^{3}$ (cm$^{2}$ s sr GeV)$^{-1}$", (2e-5, 1),
    )
    fig_numu = compare(
        "numu_total", "numu_conv", r"$E_{\nu_\mu}$ [GeV]",
        r"$\Phi_{\nu_\mu}$ (E/GeV)$^{3}$ (cm$^{2}$ s sr GeV)$^{-1}$",
        (2e-5, 1e-1),
    )
    fig_nue = compare(
        "nue_total", "nue_conv", r"$E_{\nu_e}$ [GeV]",
        r"$\Phi_{\nu_e}$ (E/GeV)$^{3}$ (cm$^{2}$ s sr GeV)$^{-1}$",
        (2e-5, 1e-1),
    )

    ratio_figs = []
    for key, title, ylabel, ylim in [
        ("mu_charge", r"$E_{\mu}$ [GeV]", "Muon charge ratio", (1.15, 1.6)),
        ("numu_ratio", r"$E_{\nu_\mu}$ [GeV]",
         r"${\nu_\mu}/\bar{\nu}_\mu$ ratio", (1.05, 1.95)),
        ("nue_ratio", r"$E_{\nu_e}$ [GeV]", r"${\nu_e}/\bar{\nu}_e$ ratio",
         (1.0, 1.45)),
    ]:
        fig, axes = plt.subplots(
            2, 1, figsize=(4.5, 4.5), sharex=True,
            gridspec_kw={"height_ratios": (3, 1)},
        )
        color = axes[0].semilogx(e_grid, flux_ddm[key], ls="-", lw=1.5,
                                 label="DDM")[0].get_color()
        axes[0].semilogx(e_grid, flux_base[key], ls="-", lw=1.5, color="k",
                         alpha=0.55, label="baseline")
        axes[1].semilogx(e_grid, flux_base[key] / flux_ddm[key], ls="-", lw=1.5,
                         c="k")
        axes[0].set_xlim(10, 1e7)
        axes[0].set_ylim(*ylim)
        axes[0].set_ylabel(ylabel)
        axes[0].legend(loc="upper right", frameon=False, numpoints=1)
        axes[1].set_ylim(0.82, 1.38)
        axes[1].set_ylabel("baseline/DDM")
        axes[1].set_xlabel(title)
        fig.tight_layout(h_pad=-0.5)
        ratio_figs.append(fig)
    return fig_mu, fig_nue, fig_numu, ratio_figs


@app.cell
def _(fig_mu, fig_nue, fig_numu):
    [fig_mu, fig_numu, fig_nue]
    return


@app.cell
def _(ratio_figs):
    ratio_figs
    return


@app.cell(hide_code=True)
def export_notebook(download_jupyter, settings):
    download_jupyter(filename="DDM_example.ipynb", configuration=settings)
    return


if __name__ == "__main__":
    app.run()
