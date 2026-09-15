# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo==0.24.0", "nbformat", "MCEq>=2.0.0", "matplotlib>=3.10", "numpy", "crflux>1.0.5"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Modifying the K/pi ratio")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Modifying the K/pi ratio

    `set_mod_pprod` overrides hadron production with a custom weight function.
    Here the pion yield is raised and the kaon yield lowered, and the pion- vs
    kaon-derived neutrino components are compared against the nominal flux.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from MCEq import config
    from MCEq.core import MCEqRun
    from MCEq.notebook import apply_kernel_and_grid, download_jupyter

    return MCEqRun, apply_kernel_and_grid, config, download_jupyter, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 01 · Configuration
    """)
    return


@app.cell(hide_code=True)
def configuration_defaults():
    defaults = {
        "database": "mceq_db_v140reduced_compact.h5",
        "interaction_model": "SIBYLL21",
        "primary": "HillasGaisser2012:H3a",
        "theta_deg": 60.0,
        "e_min": 1.0,
        "e_max": 1.0e5,
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
def _(mo):
    mo.md(r"""
    ## 02 · Reweight pions and kaons

    Three configurations, each at zeniths 0°, 60°, 90° over the IceCube
    (MSIS00 South Pole / Jan) atmosphere: nominal, pions boosted by +15%, and
    kaons raised. Only the production matrices change, so `regenerate_matrices`
    keeps the decay tables.
    """)
    return


@app.cell
def _(MCEqRun, apply_kernel_and_grid, backend, config, database, e_max, e_min,
      interaction_model, np, primary_model):
    apply_kernel_and_grid(backend, e_min, e_max, 0)
    config.mceq_db_fname = database

    angles = np.rad2deg(np.arccos(np.array([1.0, 0.5, 0.0])))
    mag = 3

    def weight_fun(xmat, egrid, pname, value):
        return (1 + value) * np.ones_like(xmat)

    mceq = MCEqRun(
        interaction_model=interaction_model,
        primary_model=primary_model,
        theta_deg=0.0,
    )
    mceq.set_density_model(("MSIS00_IC", ("SouthPole", "January")))
    e_grid = mceq.e_grid

    def compute_fluxes():
        from collections import defaultdict

        flux = defaultdict(list)
        for theta in angles:
            mceq.set_zenith_azimuth(theta)
            mceq.solve()
            for key, components in {
                "numu_pi": ["pi_numu", "pi_antinumu"],
                "numu_k": ["k_numu", "k_antinumu", "K0_numu", "K0_antinumu"],
                "numu_mu": ["mu_numu", "mu_antinumu"],
                "numu_total": ["total_numu", "total_antinumu"],
                "nue_pi": ["pi_nue", "pi_antinue"],
                "nue_k": ["k_nue", "k_antinue", "K0_nue", "K0_antinue"],
                "nue_mu": ["mu_nue", "mu_antinue"],
                "nue_total": ["total_nue", "total_antinue"],
            }.items():
                flux[key].append(
                    sum(mceq.get_solution(c, mag) for c in components)
                )
        return dict(flux)

    mceq.unset_mod_pprod()
    mceq.matrix_builder.construct_matrices(skip_decay_matrix=True)
    nominal = compute_fluxes()

    mceq.set_mod_pprod(2212, 211, weight_fun, ("a", 0.15))
    mceq.set_mod_pprod(2212, -211, weight_fun, ("a", 0.12))
    mceq.regenerate_matrices(skip_decay_matrix=True)
    piup = compute_fluxes()
    mceq.unset_mod_pprod()

    mceq.set_mod_pprod(2212, 321, weight_fun, ("a", 0.45))
    mceq.set_mod_pprod(2212, -321, weight_fun, ("a", 0.55))
    mceq.regenerate_matrices(skip_decay_matrix=True)
    kdown = compute_fluxes()
    mceq.unset_mod_pprod()
    return e_grid, kdown, nominal, piup

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 03 · Decomposition

    Black: nominal, red: kaon-raised. Line style marks the mother: solid total,
    dashed from pions, dot-dash from kaons, dotted from muon decay.
    """)
    return


@app.cell
def _(e_grid, kdown, nominal, plt):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.3), sharex=True, sharey=True)
    for axis, group, ylim in [
        (axes[0], "numu", (5e-4, 2e-1)),
        (axes[1], "nue", (5e-4, 9e-2)),
    ]:
        for style, data in [("k", nominal), ("r", kdown)]:
            color = axis.loglog(e_grid, data[f"{group}_total"][1], c=style)[0].get_color()
            axis.loglog(e_grid, data[f"{group}_pi"][1], c=color, ls="--")
            axis.loglog(e_grid, data[f"{group}_k"][1], c=color, ls="-.")
            axis.loglog(e_grid, data[f"{group}_mu"][1], c=color, ls=":")
        axis.set_xlim(1.0, 1500)
        axis.set_ylim(*ylim)
        axis.tick_params(labelsize=11)
    axes[0].set_xlabel("Muon neutrino energy $E$ (GeV)", fontsize="large")
    axes[1].set_xlabel("Electron neutrino energy $E$ (GeV)", fontsize="large")
    axes[0].set_ylabel(
        r"$E^3\,\Phi$ (GeV$^{-2}\,$s$\,$cm$^{2}\,$sr$)^{-1}$", fontsize="large"
    )
    axes[0].text(0.75, 0.9, r"$\nu_\mu$ + $\bar{\nu}_\mu$",
                 transform=axes[0].transAxes, fontsize="large")
    axes[1].text(0.75, 0.9, r"$\nu_{\rm e}$ + $\bar{\nu}_{\rm e}$",
                 transform=axes[1].transAxes, fontsize="large")
    legend_lines, legend_labels = [], []
    for name, ls in zip(
        ["total", r"from $\pi^\pm$", r"from $K^{\pm,0}$", r"from $\mu^\pm$"],
        ["-", "--", "-.", ":"],
    ):
        legend_lines.append(plt.plot([0, 0], [0, 0], ls=ls, color="k")[0])
        legend_labels.append(name)
    fig.legend(
        legend_lines, legend_labels, loc="upper center", mode="expand", ncol=4,
        frameon=True, fontsize="large",
    )
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.9], w_pad=0.0, h_pad=0.5)
    fig
    return (fig,)


@app.cell(hide_code=True)
def export_notebook(download_jupyter, settings):
    download_jupyter(filename="KPi_demonstration.ipynb", configuration=settings)
    return


if __name__ == "__main__":
    app.run()
