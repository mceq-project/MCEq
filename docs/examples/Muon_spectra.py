# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo==0.24.0", "nbformat", "MCEq>=2.0.0", "matplotlib>=3.10", "numpy", "scipy", "crflux>1.0.5"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Muon spectra at Auger and IceCube")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Muon spectra at Auger and IceCube

    Prompt and conventional muon (and IceCube neutrino) spectra for single
    proton and iron primaries, solved on the atmospheric depth grid
    (original by H. Dembinski). Interaction models offered are the ones in the
    selected database; this demo needs the full public database because single
    primaries at $10^{8}$–$10^{10}$ GeV need the full energy grid.
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
        np,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 01 · Configuration

    Observatories and compositions are fixed (Auger: 1425 m, 67°, $10^{10}$
    GeV; IceCube: 2835 m, 13°, $10^{8}$ GeV). Pick the interaction models to
    compare — each adds four solves.
    """)
    return


@app.cell(hide_code=True)
def configuration_defaults():
    defaults = {
        "database": "mceq_db_lext_dpm193_v142.h5",
        "models": ["EPOSLHC", "SIBYLL23E"],
        "e_min": 1.0,
        "e_max": 1.0e11,
        "backend": "auto",
    }
    return (defaults,)


@app.cell(hide_code=True)
def configuration_controls(defaults, mo):
    from MCEq.notebook import configuration_controls as build_controls

    controls, options = build_controls(
        defaults,
        database=defaults["database"],
        show=("models_multi", "energy", "backend"),
    )
    controls
    return (controls, options)


@app.cell(hide_code=True)
def selected_configuration(controls, defaults, mo):
    from MCEq.notebook import apply_configuration

    settings = apply_configuration(controls, defaults)
    database = settings["database"]
    models = settings["models"]
    e_min, e_max = settings["e_min"], settings["e_max"]
    backend = settings["backend"]
    mo.md(
        f"**Applied:** {', '.join(models)} · {e_min:g}–{e_max:g} GeV · "
        f"{len(models) * 4} solves expected"
    )
    return (backend, database, e_max, e_min, models, settings)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 02 · Solve per experiment, model, and composition
    """)
    return


@app.cell
def _(MCEqRun, apply_kernel_and_grid, backend, config, database, e_max, e_min,
      models, np):
    from scipy.integrate import trapezoid

    import crflux.models as pm

    apply_kernel_and_grid(backend, e_min, e_max, 0)
    config.mceq_db_fname = database

    experiments = {
        "auger": {"h_obs": 1425, "theta_deg": 67, "E0": 1e10,
                  "atmosphere": ("CORSIKA", ("BK_USStd", None))},
        "icecube": {"h_obs": 2835, "theta_deg": 13, "E0": 1e8,
                    "atmosphere": ("MSIS00_IC", ("SouthPole", "January"))},
    }
    primaries = {"proton": {"pdg_id": 2212}, "iron": {"corsika_id": 5626}}

    spectra, height_grid = {}, {}

    def scan(advance):
        for experiment, site in experiments.items():
            config.h_obs = site["h_obs"]
            spectra[experiment] = {}
            for interaction_model in models:
                per_primary = spectra[experiment][interaction_model] = {}
                for prim, ids in primaries.items():
                    advance(title=f"{experiment} · {interaction_model} · {prim}")
                    run = MCEqRun(
                        interaction_model,
                        primary_model=(pm.HillasGaisser2012, "H3a"),
                        theta_deg=site["theta_deg"],
                    )
                    run.set_density_model(site["atmosphere"])
                    run.set_single_primary_particle(site["E0"], **ids)
                    Xvec = np.arange(1, run.density_model.max_X, 20)
                    run.solve(int_grid=Xvec, grid_var="X")
                    if experiment not in height_grid:
                        height_grid[experiment] = (
                            Xvec,
                            run.density_model.s_lX2h(np.log(Xvec)) / 1e2,
                        )

                    store = per_primary[prim] = {"egrid": run.e_grid}
                    for component in ("total", "conv", "pr"):
                        store["mu_" + component] = (
                            run.get_solution(f"{component}_mu+", 0)
                            + run.get_solution(f"{component}_mu-", 0)
                        )
                    store["mu_long"] = np.array([
                        trapezoid(
                            run.get_solution("total_mu+", 0, grid_idx=snap)
                            + run.get_solution("total_mu-", 0, grid_idx=snap),
                            run.e_grid,
                        )
                        for snap in range(len(run.grid_sol))
                    ])
                    if experiment == "icecube":
                        for neutrino in ("numu", "nue", "nutau"):
                            for component in ("total", "conv", "pr"):
                                if neutrino == "nutau" and component != "total":
                                    continue
                                store[f"{neutrino}_{component}"] = (
                                    run.get_solution(f"{component}_{neutrino}", 0)
                                    + run.get_solution(
                                        f"{component}_anti{neutrino}", 0
                                    )
                                )

    with loop_progress(
        len(experiments) * len(models) * len(primaries),
        "detector x model x primary scans",
    ) as advance:
        scan(advance)
    return (height_grid, spectra)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 03 · Atmosphere columns and muon yield with depth

    The shaded band marks the observatory altitude.
    """)
    return


@app.cell
def _(height_grid, plt, spectra):
    def plot_columns():
        titles = {
            "auger": r"Auger: 1425 m, $\theta = 67^\circ$",
            "icecube": r"IceCube: 2835 m, $\theta = 13^\circ$",
        }
        h_obs = {"auger": 1425, "icecube": 2835}
        figures = []
        for experiment, (depth, height) in height_grid.items():
            fig, ax = plt.subplots()
            ax.set_title(titles[experiment])
            ax.plot(height / 1e3, depth, "k-")
            ax.axvspan(0, h_obs[experiment] / 1e3, color="k", alpha=0.2)
            ax.set_xlim(0, 50)
            ax.set_xlabel("$h$/km")
            ax.set_ylabel("$X_\\mathrm{atm}/\\mathrm{g\\,cm^{-2}}$")
            figures.append(fig)
        return figures

    plot_columns()
    return


@app.cell
def _(height_grid, plt, spectra):
    def plot_longitudinal():
        titles = {
            "auger": r"Auger: 1425 m, $E_0 = 10^{10}\,$GeV $\theta = 67^\circ$",
            "icecube": r"IceCube: 2835 m, $E_0 = 10^{8}\,$GeV $\theta = 13^\circ$",
        }
        h_obs = {"auger": 1425, "icecube": 2835}
        figures = []
        for experiment, (depth, height) in height_grid.items():
            for interaction_model, per_primary in spectra[experiment].items():
                fig, ax = plt.subplots()
                ax.set_title(titles[experiment])
                for prim, store in per_primary.items():
                    ax.plot(
                        height / 1e3, store["mu_long"], "-",
                        color={"proton": "r", "iron": "b"}[prim], label=prim,
                    )
                ax.axvspan(0, h_obs[experiment] / 1e3, color="k", alpha=0.2,
                           zorder=1)
                ax.set_xlim(0, 20)
                ax.legend(title=interaction_model)
                ax.set_xlabel("$h$/km")
                ax.set_ylabel("$N_\\mu(E_\\mu > 1\\,\\mathrm{GeV})$")
                figures.append(fig)
        return figures

    plot_longitudinal()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 04 · Spectra, composition ratio, and neutrinos

    Total and prompt muon spectra per primary, then the iron/proton ratio and
    its model spread, then the IceCube muon-neutrino spectra.
    """)
    return


@app.cell
def _(plt, spectra):
    def plot_muon_spectra():
        titles = {
            "auger": r"Auger: 1425 m, $E_0 = 10^{10}\,$GeV $\theta = 67^\circ$",
            "icecube": r"IceCube: 2835 m, $E_0 = 10^{8}\,$GeV $\theta = 13^\circ$",
        }
        figures = []
        for experiment in spectra:
            for interaction_model, per_primary in spectra[experiment].items():
                fig, ax = plt.subplots()
                ax.set_title(titles[experiment])
                for prim, store in per_primary.items():
                    color = {"proton": "r", "iron": "b"}[prim]
                    label = {"proton": "p", "iron": "Fe"}[prim]
                    ax.loglog(store["egrid"], store["mu_total"], "-",
                              color=color, lw=2, label=label + " total")
                    ax.loglog(store["egrid"], store["mu_pr"], ":",
                              color=color, label=label + " prompt")
                ax.set_xlim(0.1, 1e8)
                ax.set_ylim(1e-14, 1e7)
                ax.set_xlabel(r"$E_\mu$/GeV")
                ax.legend(title=interaction_model)
                figures.append(fig)
        return figures

    plot_muon_spectra()
    return


@app.cell
def _(plt, spectra):
    def plot_composition_ratio():
        titles = {
            "auger": r"Auger: 1425 m, $E_0 = 10^{10}\,$GeV $\theta = 67^\circ$",
            "icecube": r"IceCube: 2835 m, $E_0 = 10^{8}\,$GeV $\theta = 13^\circ$",
        }
        families = {"EPOSLHC": "k", "SIBYLL": "b", "DPMJET": "g", "QGSJET": "r"}
        figures = []
        for experiment in spectra:
            fig, ax = plt.subplots()
            ax.set_title(titles[experiment], x=0.55)
            for interaction_model, per_primary in spectra[experiment].items():
                ratio = (
                    per_primary["iron"]["mu_total"]
                    / per_primary["proton"]["mu_total"]
                )
                family = next(
                    (name for name in families if name in interaction_model),
                    None,
                )
                ax.semilogx(
                    per_primary["proton"]["egrid"], ratio, "-",
                    color=families.get(family, "0.5"), label=interaction_model,
                )
            ax.set_xlim(1, {"auger": 1e8, "icecube": 1e6}[experiment])
            ax.set_ylim(1, 3.2)
            ax.set_xlabel(r"$E_\mu$/GeV")
            ax.set_ylabel(r"iron / proton")
            ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
            figures.append(fig)
        return figures

    plot_composition_ratio()
    return


@app.cell
def _(plt, spectra):
    def plot_icnu():
        figures = []
        for interaction_model, per_primary in spectra["icecube"].items():
            fig, ax = plt.subplots()
            ax.set_title(
                r"IceCube: 2835 m, $E_0 = 10^{8}\,$GeV $\theta = 13^\circ$"
            )
            for prim, store in per_primary.items():
                color = {"proton": "r", "iron": "b"}[prim]
                label = {"proton": "p", "iron": "Fe"}[prim]
                ax.loglog(store["egrid"], store["numu_total"], "-",
                          color=color, lw=2, label=f"{label} total")
                ax.loglog(store["egrid"], store["numu_pr"], ":",
                          color=color, label=f"{label} prompt")
            ax.set_xlim(1, 1e8)
            ax.set_ylim(1e-14, 1e7)
            ax.set_xlabel(r"$E_{\nu_\mu}$/GeV")
            ax.legend(title=interaction_model)
            figures.append(fig)
        return figures

    plot_icnu()
    return


@app.cell(hide_code=True)
def export_notebook(download_jupyter, settings):
    download_jupyter(filename="Muon_spectra.ipynb", configuration=settings)
    return


if __name__ == "__main__":
    app.run()
