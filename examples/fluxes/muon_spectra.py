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
    # Muons from single proton and iron primaries at Auger and IceCube

    Muon number along the shower axis, muon spectra at the observation
    level, and the iron/proton ratio for single primaries, per interaction
    model (original by H. Dembinski).
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

    The two sites are fixed: Auger at 1425 m, 67° zenith, $E_0 = 10^{10}$
    GeV in the CORSIKA US-standard atmosphere; IceCube at 2835 m, 13°,
    $E_0 = 10^{8}$ GeV in the South-Pole January MSIS00 atmosphere.

    - **Interaction models**: each selected model is one `MCEqRun` (one
      matrix build) and four solves. Models outside the base package are
      fetched on first use.
    - **Energy grid**: the primary must sit on the grid, so `e_max` has to
      exceed $10^{10}$ GeV for Auger. A proton at $E_0$ and an iron nucleus
      at $E_0/56$ per nucleon are both deposited on it with
      `set_single_primary_particle`.
    """)
    return


@app.cell(hide_code=True)
def configuration_defaults():
    defaults = {
        "database": "mceq_base_air_1d_v2.h5",
        "models": ["EPOSLHC", "SIBYLL23E"],
        "e_min": 1.0,
        "e_max": 1.0e11,
        "backend": "auto",
    }
    return (defaults,)


@app.cell(hide_code=True)
def configuration_controls(defaults, mo):
    from MCEq.notebook import configuration_controls as build_controls

    controls, _options = build_controls(
        defaults,
        database=defaults["database"],
        show=("models_multi", "energy", "backend"),
    )
    controls
    return (controls,)


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
        f"{len(models) * 4} solves"
    )
    return backend, database, e_max, e_min, models, settings


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
    ## 02 · Solve per model, site and primary

    `solve(int_grid=X)` stores the state at each slant depth in `X`
    (every 20 g/cm² down to the ground); `get_solution(..., grid_idx=i)`
    reads snapshot `i`. `config.h_obs` sets the observation level of the
    CORSIKA atmosphere when it is created; the IceCube-centred MSIS00
    atmosphere carries its own surface elevation.
    """)
    return


@app.cell
def _(MCEqRun, apply_kernel_and_grid, config, backend, database, e_max, e_min,
      loop_progress, models, np, proceed):
    _ = proceed
    import crflux.models as pm
    from scipy.integrate import trapezoid

    apply_kernel_and_grid(backend, e_min, e_max, 0)
    config.mceq_db_fname = database

    sites = {
        "Auger": {"h_obs": 1425.0, "theta_deg": 67.0, "E0": 1e10,
                  "atmosphere": ("CORSIKA", ("BK_USStd", None))},
        "IceCube": {"h_obs": 2835.0, "theta_deg": 13.0, "E0": 1e8,
                    "atmosphere": ("MSIS00_IC", ("SouthPole", "January"))},
    }
    primaries = {"p": {"pdg_id": 2212}, "Fe": {"corsika_id": 5626}}

    # results[site][model][primary] -> dict of arrays; columns[site] -> (X, h)
    results = {_site: {_model: {} for _model in models} for _site in sites}
    columns = {}
    with loop_progress(
        len(models) * len(sites) * len(primaries), "model × site × primary"
    ) as advance:
        for _model in models:
            run = MCEqRun(
                interaction_model=_model,
                low_energy_model="FLUKA20251",  # below 80 GeV
                primary_model=(pm.HillasGaisser2012, "H3a"),
                theta_deg=0.0,
            )
            for _site, _spec in sites.items():
                config.h_obs = _spec["h_obs"]
                run.set_density_model(_spec["atmosphere"])
                run.set_zenith_azimuth(_spec["theta_deg"])
                _depths = np.arange(1.0, run.density_model.max_X, 20.0)
                columns[_site] = (
                    _depths, run.density_model.s_lX2h(np.log(_depths)) / 1e5
                )
                for _primary, _ids in primaries.items():
                    run.set_single_primary_particle(_spec["E0"], **_ids)
                    run.solve(int_grid=_depths)
                    advance(title=f"{_model} · {_site} · {_primary}")
                    _store = {"e_grid": run.e_grid}
                    for _component in ("total", "pr"):
                        _store[f"mu_{_component}"] = (
                            run.get_solution(f"{_component}_mu+")
                            + run.get_solution(f"{_component}_mu-")
                        )
                        _store[f"numu_{_component}"] = (
                            run.get_solution(f"{_component}_numu")
                            + run.get_solution(f"{_component}_antinumu")
                        )
                    _store["n_mu"] = np.array([
                        trapezoid(
                            run.get_solution("total_mu+", grid_idx=i)
                            + run.get_solution("total_mu-", grid_idx=i),
                            run.e_grid,
                        )
                        for i in range(len(_depths))
                    ])
                    results[_site][_model][_primary] = _store
            run.close()
    config.h_obs = 0.0
    return columns, primaries, results, sites


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 03 · Muon number along the shower axis

    $N_\mu(E_\mu > 1\,\mathrm{GeV})$, the integral of the muon spectrum at
    each snapshot, versus the altitude of that depth on the inclined axis.
    The grey band is below the observation level.
    """)
    return


@app.cell
def _(columns, models, np, plt, primaries, results, sites):
    model_colors = dict(zip(
        models, ["#17857E", "#C46B30", "#263342", "#7A5BA6", "#B8A030", "#52606D"]
    ))
    primary_styles = {"p": "-", "Fe": "--"}
    plot_style = {
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
    }

    with plt.rc_context(plot_style):
        fig_long, axes_long = plt.subplots(
            1, 2, figsize=(11.5, 4.2), layout="constrained"
        )
        for _ax, (_site, _spec) in zip(axes_long, sites.items()):
            _depths, height_km = columns[_site]
            for _model in models:
                for _primary in primaries:
                    _ax.plot(
                        height_km, results[_site][_model][_primary]["n_mu"],
                        color=model_colors[_model],
                        linestyle=primary_styles[_primary], linewidth=1.8,
                        label=f"{_model}, {_primary}",
                    )
            _ax.axvspan(0, _spec["h_obs"] / 1e3, color="#B3BAC2", alpha=0.3)
            _ax.set_xlim(0, 20)
            _ax.set_xlabel("Altitude on the shower axis [km]")
            _ax.set_title(
                f"{_site}: {_spec['h_obs']:g} m, {_spec['theta_deg']:g}°, "
                f"$E_0 = 10^{{{np.log10(_spec['E0']):.0f}}}$ GeV",
                loc="left", fontsize=12, pad=10,
            )
            _ax.grid(axis="y", color="#E6E9ED", linewidth=0.7)
            _ax.set_axisbelow(True)
        axes_long[0].set_ylabel(r"$N_\mu(E_\mu > 1\,\mathrm{GeV})$")
        fig_long.legend(
            *axes_long[0].get_legend_handles_labels(), frameon=False,
            loc="outside lower center", ncol=4, fontsize=10,
        )
    fig_long
    return model_colors, plot_style, primary_styles


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 04 · Ground spectra and composition ratio

    Top: muon spectra at the observation level (solid: total, dotted:
    prompt from charm decays). Bottom: iron over proton for the total
    spectrum; iron yields more low-energy muons because its energy per
    nucleon is 56 times lower, and fewer at the highest energies.
    """)
    return


@app.cell
def _(model_colors, models, plot_style, plt, results, sites):
    with plt.rc_context(plot_style):
        fig_spec, axes_spec = plt.subplots(
            2, 2, figsize=(11.5, 7.2), sharex="col", layout="constrained",
            gridspec_kw={"height_ratios": [2, 1]},
        )
        for _col, _site in enumerate(sites):
            _ax, _ax_ratio = axes_spec[0, _col], axes_spec[1, _col]
            for _model in models:
                _color = model_colors[_model]
                for _primary, _linewidth in (("p", 1.2), ("Fe", 2.2)):
                    _store = results[_site][_model][_primary]
                    _ax.loglog(
                        _store["e_grid"], _store["mu_total"], color=_color,
                        linewidth=_linewidth, label=f"{_model}, {_primary}",
                    )
                    _ax.loglog(
                        _store["e_grid"], _store["mu_pr"], color=_color,
                        linewidth=_linewidth, linestyle=":",
                    )
                _proton = results[_site][_model]["p"]
                _ratio = results[_site][_model]["Fe"]["mu_total"] / _proton["mu_total"]
                _ax_ratio.semilogx(_proton["e_grid"], _ratio, color=_color, linewidth=1.8)
            _ax.set_xlim(1.0, 1.0e8 if _site == "Auger" else 1.0e6)
            _ax.set_ylim(1e-14, 1e7)
            _ax.set_title(_site, loc="left", fontsize=12, pad=10)
            _ax_ratio.set_ylim(0.5, 3.2)
            _ax_ratio.axhline(1.0, color="#B3BAC2", linewidth=0.8)
            _ax_ratio.set_xlabel(r"Muon energy $E_\mu$ [GeV]")
            for _axis in (_ax, _ax_ratio):
                _axis.grid(axis="y", which="major", color="#E6E9ED", linewidth=0.7)
                _axis.set_axisbelow(True)
        axes_spec[0, 0].set_ylabel(r"$\mathrm{d}N_\mu/\mathrm{d}E$ [GeV$^{-1}$]")
        axes_spec[1, 0].set_ylabel("Fe / p")
        fig_spec.legend(
            *axes_spec[0, 0].get_legend_handles_labels(), frameon=False,
            loc="outside lower center", ncol=4, fontsize=10,
        )
    fig_spec
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 05 · Muon neutrinos at IceCube

    $\nu_\mu + \bar\nu_\mu$ from the same IceCube showers, total and prompt.
    """)
    return


@app.cell
def _(model_colors, models, plot_style, plt, results):
    with plt.rc_context(plot_style):
        fig, ax_nu = plt.subplots(figsize=(6.5, 4.4), layout="constrained")
        for _model in models:
            for _primary, _linewidth in (("p", 1.2), ("Fe", 2.2)):
                _store = results["IceCube"][_model][_primary]
                ax_nu.loglog(
                    _store["e_grid"], _store["numu_total"],
                    color=model_colors[_model], linewidth=_linewidth,
                    label=f"{_model}, {_primary}",
                )
                ax_nu.loglog(
                    _store["e_grid"], _store["numu_pr"],
                    color=model_colors[_model], linewidth=_linewidth, linestyle=":",
                )
        ax_nu.set_xlim(1.0, 1.0e8)
        ax_nu.set_ylim(1e-14, 1e7)
        ax_nu.set_xlabel(r"Neutrino energy $E_{\nu_\mu}$ [GeV]")
        ax_nu.set_ylabel(r"$\mathrm{d}N_{\nu_\mu}/\mathrm{d}E$ [GeV$^{-1}$]")
        ax_nu.set_title(
            r"IceCube, $E_0 = 10^8$ GeV, 13° (dotted: prompt)",
            loc="left", fontsize=12, pad=10,
        )
        ax_nu.grid(axis="y", which="major", color="#E6E9ED", linewidth=0.7)
        ax_nu.legend(frameon=False, fontsize=9)
    fig
    return (fig,)


@app.cell(hide_code=True)
def export_notebook(settings):
    from MCEq.notebook import download_jupyter

    download_jupyter(filename="muon_spectra.ipynb", configuration=settings)
    return


if __name__ == "__main__":
    app.run()
