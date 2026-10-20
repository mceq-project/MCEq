# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo==0.24.0", "nbformat", "MCEq>=2.0.0", "matplotlib>=3.10", "numpy", "crflux>1.0.5"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Modifying the K/π ratio")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Modifying the K/π ratio

    Raising the kaon yield in proton–air collisions hardens the neutrino flux
    and shifts the $\nu_\mu$ budget from pion to kaon decays; raising the pion
    yield matters most at low energies, where pions dominate.
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

    The baseline hadronic model, primary flux, zenith angle and cascade energy
    range. The modifications below are applied on top of the chosen model; the
    atmosphere is the IceCube-centred MSIS-00 profile (South Pole, January).
    """)
    return


@app.cell(hide_code=True)
def configuration_defaults():
    defaults = {
        "database": "mceq_base_air_1d_v2.h5",
        "interaction_model": "SIBYLL23E",
        "primary": "HillasGaisser2012:H3a",
        "theta_deg": 60.0,
        "e_min": 1.0,
        "e_max": 1.0e5,
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
        backend, database, e_max, e_min, interaction_model, primary_model,
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
    ## 02 · Reweight pion and kaon production

    `MCEqRun.set_mod_pprod(projectile, secondary, x_func, args)` multiplies the
    yield matrix of one channel by `x_func(x, E, *args)`, where $x = E_{\rm sec}/E_{\rm proj}$.
    Here `x_func` is a flat factor $1+\delta$ in $p\to\pi^\pm$ or $p\to K^\pm$:

    | scenario | $\pi^+$ | $\pi^-$ | $K^+$ | $K^-$ |
    |---|---|---|---|---|
    | nominal | – | – | – | – |
    | π raised | +15 % | +12 % | – | – |
    | K raised | – | – | +45 % | +55 % |

    After the modifications `regenerate_matrices(skip_decay_matrix=True)`
    rebuilds only the interaction matrix (decays are unchanged), and
    `unset_mod_pprod()` returns to the nominal model. Neutrinos are split by
    mother with MCEq's default tracking categories `pi_`, `k_` + `K0_`, and
    `mu_`.
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

    apply_kernel_and_grid(backend, e_min, e_max, 0)
    config.mceq_db_fname = database

    mceq_run = MCEqRun(
        interaction_model=interaction_model,
        primary_model=primary_model,
        theta_deg=theta_deg,
        density_model=("MSIS00_IC", ("SouthPole", "January")),
    )
    return (mceq_run,)


@app.cell
def _(loop_progress, mceq_run, np):
    def flat_factor(xmat, egrid, label, delta):
        """x_func for set_mod_pprod: a constant factor (1 + delta)."""
        return (1.0 + delta) * np.ones_like(xmat)

    scenarios = {
        "nominal": [],
        "π raised": [(2212, 211, 0.15), (2212, -211, 0.12)],
        "K raised": [(2212, 321, 0.45), (2212, -321, 0.55)],
    }
    mothers = {
        "total": ("total_",),
        "pi": ("pi_",),
        "k": ("k_", "K0_"),
        "mu": ("mu_",),
    }
    neutrinos = {"numu": ("numu", "antinumu"), "nue": ("nue", "antinue")}

    fluxes = {}
    with loop_progress(len(scenarios), "K/π reweighting") as advance:
        for _scenario, _modifications in scenarios.items():
            mceq_run.unset_mod_pprod(dont_fill=True)
            for _projectile, _secondary, _delta in _modifications:
                mceq_run.set_mod_pprod(
                    _projectile, _secondary, flat_factor, ("delta", _delta)
                )
            mceq_run.regenerate_matrices(skip_decay_matrix=True)
            mceq_run.solve()
            fluxes[_scenario] = {
                (flavor, mother): sum(
                    mceq_run.get_solution(prefix + particle, mag=3)
                    for prefix in prefixes
                    for particle in particles
                )
                for flavor, particles in neutrinos.items()
                for mother, prefixes in mothers.items()
            }
            advance()
    mceq_run.unset_mod_pprod()
    e_grid = mceq_run.e_grid
    return e_grid, fluxes


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 03 · Decomposition by mother

    Colour marks the scenario, line style the mother: solid total, dashed
    from $\pi^\pm$, dash-dotted from $K^{\pm,0}$, dotted from muon decay.
    The lower panels show each modified total relative to the nominal one.
    """)
    return


@app.cell
def _(e_grid, fluxes, interaction_model, np, plt, theta_deg):
    plot_range = (1.0, 2.0e3)  # GeV
    plot_mask = (e_grid >= plot_range[0]) & (e_grid <= plot_range[1])
    scenario_colors = {
        "nominal": "#263342",
        "π raised": "#17857E",
        "K raised": "#C46B30",
    }
    mother_styles = {
        "total": ("total", "-"),
        "pi": (r"from $\pi^\pm$", "--"),
        "k": (r"from $K^{\pm,0}$", "-."),
        "mu": (r"from $\mu^\pm$", ":"),
    }
    flavors = {
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
            2, 2, figsize=(11, 6.4), sharex=True, layout="constrained",
            gridspec_kw={"height_ratios": (3, 1)},
        )
        for column, (flavor, flavor_label) in enumerate(flavors.items()):
            ax, ratio_ax = axes[0, column], axes[1, column]
            nominal_total = fluxes["nominal"][(flavor, "total")]
            for scenario, color in scenario_colors.items():
                for mother, (_label, linestyle) in mother_styles.items():
                    values = fluxes[scenario][(flavor, mother)]
                    visible = plot_mask & (values > 0)
                    ax.loglog(
                        e_grid[visible], values[visible], color=color,
                        linestyle=linestyle, linewidth=1.8,
                    )
                ratio = np.divide(
                    fluxes[scenario][(flavor, "total")], nominal_total,
                    out=np.ones_like(nominal_total), where=nominal_total > 0,
                )
                ratio_ax.semilogx(
                    e_grid[plot_mask], ratio[plot_mask], color=color, linewidth=1.8,
                )
            ax.set_title(flavor_label, loc="left", fontsize=14, pad=10)
            ax.set_xlim(plot_range)
            ax.grid(axis="y", which="major", color="#E6E9ED", linewidth=0.7)
            ratio_ax.set_ylim(0.95, 1.6)
            ratio_ax.set_xlabel(r"Neutrino energy $E$ [GeV]")
            ratio_ax.grid(axis="y", color="#E6E9ED", linewidth=0.7)
        axes[0, 0].set_ylabel(
            r"$E^3\,\mathrm{d}\Phi/\mathrm{d}E$ [GeV$^2$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]"
        )
        axes[1, 0].set_ylabel("total / nominal")

        handles = [
            plt.Line2D([], [], color=color, linewidth=2.5, label=scenario)
            for scenario, color in scenario_colors.items()
        ] + [
            plt.Line2D([], [], color="#52606D", linestyle=linestyle, label=label)
            for label, linestyle in mother_styles.values()
        ]
        fig.legend(
            handles=handles, frameon=False, loc="outside lower center",
            ncol=7, fontsize=10,
        )
        fig.suptitle(
            f"{interaction_model}  /  {theta_deg:g}° zenith  /  MSIS-00 South Pole, January",
            fontsize=11, color="#52606D",
        )

    fig
    return (fig,)


@app.cell(hide_code=True)
def export_notebook(settings):
    from MCEq.notebook import download_jupyter

    download_jupyter(filename="kpi_demonstration.ipynb", configuration=settings)
    return


if __name__ == "__main__":
    app.run()
