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

    Replacing a baseline model's pion, kaon and nucleon yields by fits to
    fixed-target data changes the muon and neutrino fluxes by 10–20 % and
    raises their charge ratios.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import marimo as mo
    import numpy as np

    from MCEq.core import MCEqRun
    from MCEq.notebook import (
        apply_kernel_and_grid,
        loop_progress,
    )

    return (
        MCEqRun,
        apply_kernel_and_grid,
        loop_progress,
        mo,
        np,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 01 · Configuration

    The DDM ([Fedynitch & Huber, PRD 106, 083018](https://inspirehep.net/literature/2088973))
    describes $p$-C and $\pi^-$-C production of $\pi^\pm$, $K^\pm$, $p$,
    $\bar p$ and $n$ by splines fitted to NA49/NA61 data at a few beam
    energies; $n$ and $\pi^+$ projectiles follow by isospin symmetry and
    $K^0_{S,L}$ from the $K^\pm$ average. Everything it does not cover — other projectiles, other
    secondaries, charm — comes from the **baseline** interaction model chosen
    here. The DDM paper used SIBYLL-2.3d (`SIBYLL23D`, fetched with the
    extra-models package on first use).
    """)
    return


@app.cell(hide_code=True)
def configuration_defaults():
    defaults = {
        "database": "mceq_base_air_1d_v2.h5",
        "interaction_model": "SIBYLL23D",
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
        f"**Applied:** baseline {interaction_model} · {settings['primary']} · "
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
    ## 02 · Baseline and DDM runs

    `MCEq.ddm.DataDrivenModel()` loads the spline table `DDM_1.0.npy` shipped
    in `config.data_dir`; its arguments restrict the energy range
    (`e_min`/`e_max`), the channels (`enable_channels`) or projectiles
    (`exclude_projectiles`). `MCEqRun.inject_ddm(ddm)` turns the splines into
    yield matrices on the run's energy grid, installs them as channel
    overrides of the baseline tables, and rebuilds the matrices. Two runs
    with identical settings keep the comparison clean.
    """)
    return


@app.cell
def _(
    MCEqRun,
    apply_kernel_and_grid,
    backend,
    database,
    e_max,
    e_min,
    interaction_model,
    primary_model,
    theta_deg,
    proceed):
    _ = proceed
    from MCEq import config
    from MCEq.ddm import DataDrivenModel

    apply_kernel_and_grid(backend, e_min, e_max, 0)
    config.mceq_db_fname = database

    ddm = DataDrivenModel()
    ddm_channels = sorted(ddm.channels)

    runs = {}
    for _label in ("baseline", "DDM"):
        runs[_label] = MCEqRun(
            interaction_model=interaction_model,
            low_energy_model="FLUKA20251",  # below 80 GeV
            primary_model=primary_model,
            theta_deg=theta_deg,
        )
    runs["DDM"].inject_ddm(ddm)
    return ddm_channels, runs


@app.cell
def _(loop_progress, runs):
    with loop_progress(len(runs), "Baseline and DDM") as advance:
        for _run in runs.values():
            _run.solve()
            advance()
    "both solved"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 03 · Fluxes, ratios and charge ratios

    Top: $E^3$-weighted total fluxes (solid) and conventional component
    (dashed). Middle: DDM / baseline. Bottom: particle / antiparticle ratio,
    the quantity most directly constrained by the charge-resolved
    fixed-target data.
    """)
    return


@app.cell
def _(ddm_channels, runs):
    leptons = {
        "mu": ("mu+", "mu-"),
        "numu": ("numu", "antinumu"),
        "nue": ("nue", "antinue"),
    }
    spectra = {
        label: {
            lepton: {
                "total": sum(run.get_solution(f"total_{p}", mag=3) for p in pair),
                "conv": sum(run.get_solution(f"conv_{p}", mag=3) for p in pair),
                "charge": run.get_solution(f"total_{pair[0]}")
                / run.get_solution(f"total_{pair[1]}"),
            }
            for lepton, pair in leptons.items()
        }
        for label, run in runs.items()
    }
    e_grid = runs["baseline"].e_grid
    channel_summary = ", ".join(f"{p}→{s}" for p, s in ddm_channels)
    print(f"DDM channels ({len(ddm_channels)}): {channel_summary}")
    return e_grid, spectra


@app.cell
def _(e_grid, interaction_model, np, plt, spectra, theta_deg):
    plot_range = (10.0, 1.0e6)  # GeV
    plot_mask = (e_grid >= plot_range[0]) & (e_grid <= plot_range[1])
    run_colors = {"baseline": "#263342", "DDM": "#C46B30"}
    titles = {
        "mu": (r"$\mu^+ + \mu^-$", r"$\mu^+/\mu^-$"),
        "numu": (r"$\nu_\mu + \bar\nu_\mu$", r"$\nu_\mu/\bar\nu_\mu$"),
        "nue": (r"$\nu_e + \bar\nu_e$", r"$\nu_e/\bar\nu_e$"),
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
            3, 3, figsize=(11.5, 8.0), sharex=True, layout="constrained",
            gridspec_kw={"height_ratios": (3, 1.2, 1.6)},
        )
        for column, (lepton, (flux_title, charge_label)) in enumerate(titles.items()):
            flux_ax, ratio_ax, charge_ax = axes[:, column]
            for label, color in run_colors.items():
                curves = spectra[label][lepton]
                flux_ax.loglog(
                    e_grid[plot_mask], curves["total"][plot_mask], color=color,
                    linewidth=2, label=f"{label} total",
                )
                flux_ax.loglog(
                    e_grid[plot_mask], curves["conv"][plot_mask], color=color,
                    linewidth=1.5, linestyle="--", label=f"{label} conv.",
                )
                charge_ax.semilogx(
                    e_grid[plot_mask], curves["charge"][plot_mask], color=color,
                    linewidth=2,
                )
            for component, linestyle in (("total", "-"), ("conv", "--")):
                baseline = spectra["baseline"][lepton][component]
                ratio = np.divide(
                    spectra["DDM"][lepton][component], baseline,
                    out=np.ones_like(baseline), where=baseline > 0,
                )
                ratio_ax.semilogx(
                    e_grid[plot_mask], ratio[plot_mask], color="#C46B30",
                    linestyle=linestyle, linewidth=1.8,
                )
            ratio_ax.axhline(1.0, color="#B3BAC2", linewidth=0.8)
            flux_ax.set_title(flux_title, loc="left", fontsize=14, pad=10)
            flux_ax.set_xlim(plot_range)
            flux_ax.set_ylim(1.0e-5, 1.0)
            ratio_ax.set_ylim(0.7, 1.3)
            charge_ax.set_ylabel(charge_label)
            charge_ax.set_xlabel(r"Kinetic energy $E$ [GeV]")
            for ax in (flux_ax, ratio_ax, charge_ax):
                ax.grid(axis="y", which="major", color="#E6E9ED", linewidth=0.7)
                ax.set_axisbelow(True)
        axes[0, 0].set_ylabel(
            r"$E^3\,\mathrm{d}\Phi/\mathrm{d}E$ [GeV$^2$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]"
        )
        axes[1, 0].set_ylabel("DDM / baseline")
        fig.legend(
            *axes[0, 0].get_legend_handles_labels(), frameon=False,
            loc="outside lower center", ncol=4, fontsize=10,
        )
        fig.suptitle(
            f"baseline {interaction_model}  /  {theta_deg:g}° zenith",
            fontsize=11, color="#52606D",
        )

    fig
    return (fig,)


@app.cell(hide_code=True)
def export_notebook(settings):
    from MCEq.notebook import download_jupyter

    download_jupyter(filename="ddm.ipynb", configuration=settings)
    return


if __name__ == "__main__":
    app.run()
