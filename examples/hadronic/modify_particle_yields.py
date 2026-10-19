# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo==0.24.0", "nbformat", "MCEq>=2.0.0", "matplotlib>=3.10", "numpy", "crflux>1.0.5"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Modifying pion and kaon yields")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Modifying pion and kaon yields

    Constant factors on the $\pi^\pm$ and $K^\pm$ yields of proton–air
    collisions shift each $\nu_\mu$ component by roughly the charge-weighted
    mean of its factors; the total follows the kaon change where kaons
    dominate, at high energy and near the vertical.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from MCEq.core import MCEqRun
    from MCEq.notebook import (
        apply_kernel_and_grid,
        loop_progress,
        prepare_database,
        primary_model_from,
    )

    return (
        MCEqRun,
        apply_kernel_and_grid,
        loop_progress,
        mo,
        np,
        plt,
        prepare_database,
        primary_model_from,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 01 · Configuration

    The hadronic model whose yields are modified, the four energy-independent
    modifications of the proton–air yields (in percent; $0$ leaves a channel
    untouched), and the zenith angles. The atmosphere is the IceCube-centred
    NRLMSISE-00 profile (`MSIS00_IC`, South Pole, January), the primary flux
    H3a. Every zenith angle costs two solves (nominal and modified);
    near-horizontal ones are the slowest.
    """)
    return


@app.cell(hide_code=True)
def configuration_defaults():
    defaults = {
        "database": "mceq_base_air_1d_v2.h5",
        "interaction_model": "DPMJETIII193",
        "primary": "HillasGaisser2012:H3a",
        "pi_plus": 15,
        "pi_minus": -12,
        "k_plus": 45,
        "k_minus": -55,
        "zeniths": [0.0, 60.0, 90.0],
        "e_min": 1.0,
        "e_max": 1.0e6,
        "backend": "auto",
    }
    return (defaults,)


@app.cell(hide_code=True)
def configuration_controls(defaults, mo):
    from MCEq.notebook import backend_options, database_models

    models = database_models(defaults["database"])
    backends = backend_options()
    zenith_choices = ["0", "30", "45", "60", "75", "85", "90"]

    def percent_slider(key, label):
        return mo.ui.slider(
            -80, 100, step=5, value=defaults[key], label=label + " [%]",
            show_value=True, full_width=True,
        )

    controls = mo.md("""
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,350px),1fr));gap:1.5rem 2rem">
    <div>

    <strong>Model & solver</strong>

    {interaction_model}

    {backend}

    {zeniths}

    </div><div>

    <strong>Pion yields in p–air</strong>

    {pi_plus}

    {pi_minus}

    </div><div>

    <strong>Kaon yields in p–air</strong>

    {k_plus}

    {k_minus}

    </div></div>
    """).batch(
        interaction_model=mo.ui.dropdown(
            models, value=defaults["interaction_model"],
            label="Interaction model", full_width=True,
        ),
        backend=mo.ui.dropdown(
            backends,
            value=next(
                (k for k, v in backends.items() if v == defaults["backend"]),
                "Automatic",
            ),
            label="Backend", full_width=True,
        ),
        zeniths=mo.ui.multiselect(
            zenith_choices, value=[f"{z:g}" for z in defaults["zeniths"]],
            label="Zenith angles [°]", full_width=True,
        ),
        pi_plus=percent_slider("pi_plus", "π⁺"),
        pi_minus=percent_slider("pi_minus", "π⁻"),
        k_plus=percent_slider("k_plus", "K⁺"),
        k_minus=percent_slider("k_minus", "K⁻"),
    ).form(submit_button_label="Apply configuration", bordered=False)
    controls
    return (controls,)


@app.cell(hide_code=True)
def selected_configuration(controls, defaults, mo, primary_model_from):
    settings = dict(defaults)
    if controls.value is not None:
        settings.update(controls.value)
        settings["zeniths"] = sorted(float(z) for z in settings["zeniths"]) or [0.0]

    database = settings["database"]
    interaction_model = settings["interaction_model"]
    primary_model = primary_model_from(settings["primary"])
    pi_plus, pi_minus = settings["pi_plus"], settings["pi_minus"]  # percent
    k_plus, k_minus = settings["k_plus"], settings["k_minus"]  # percent
    zeniths = settings["zeniths"]
    e_min, e_max = settings["e_min"], settings["e_max"]
    backend = settings["backend"]
    mo.md(
        f"**Applied:** {interaction_model} · "
        f"π⁺ {pi_plus:+.0f}% · π⁻ {pi_minus:+.0f}% · "
        f"K⁺ {k_plus:+.0f}% · K⁻ {k_minus:+.0f}% · "
        f"θ = {', '.join(f'{z:g}' for z in zeniths)}°"
    )
    return (
        backend, database, e_max, e_min, interaction_model, k_minus, k_plus,
        pi_minus, pi_plus, primary_model, settings, zeniths,
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
    ## 02 · Nominal fluxes

    The run is built once per model. The nominal solve does not depend on
    the modification percentages, so changing only those reuses it and
    re-solves just the modified case.
    """)
    return


@app.cell
async def _(
    MCEqRun,
    apply_kernel_and_grid,
    backend,
    database,
    e_max,
    e_min,
    interaction_model,
    prepare_database,
    primary_model,
    proceed):
    _ = proceed
    from MCEq import config

    apply_kernel_and_grid(backend, e_min, e_max, 0)
    config.mceq_db_fname = await prepare_database(database)

    mceq_run = MCEqRun(
        interaction_model=interaction_model,
        primary_model=primary_model,
        theta_deg=0.0,
        density_model=("MSIS00_IC", ("SouthPole", "January")),
    )
    return (mceq_run,)


@app.cell
def _():
    def numu_components(mceq_run, zeniths, advance):
        """Solve each zenith angle; return numu + antinumu by mother, E^3-weighted.

        The mother split uses MCEq's default tracking categories: ``pi_``,
        ``k_`` (charged kaons) + ``K0_`` (K0_L, K0_S) and ``mu_``.
        """
        mothers = {
            "total": ("total_",),
            "pi": ("pi_",),
            "k": ("k_", "K0_"),
            "mu": ("mu_",),
        }
        fluxes = {}
        for theta in zeniths:
            mceq_run.set_zenith_azimuth(theta)
            mceq_run.solve()
            fluxes[theta] = {
                mother: sum(
                    mceq_run.get_solution(prefix + particle, mag=3)
                    for prefix in prefixes
                    for particle in ("numu", "antinumu")
                )
                for mother, prefixes in mothers.items()
            }
            advance()
        return fluxes

    return (numu_components,)


@app.cell
def _(loop_progress, mceq_run, numu_components, zeniths):
    with loop_progress(len(zeniths), "Nominal · zenith scan") as _advance:
        nominal = numu_components(mceq_run, zeniths, _advance)
    e_grid = mceq_run.e_grid
    "nominal solved"
    return e_grid, nominal


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 03 · Modify the yields and re-solve

    `MCEqRun.set_mod_pprod(projectile, secondary, x_func, args)` multiplies the
    yield matrix of one channel, element by element, by a weight
    `x_func(x, E, *args)` evaluated on the matrix grid, with
    $x = E_{\rm sec}/E_{\rm proj}$ and $E$ the projectile energy. The weight
    can be any function of $x$ and $E$ (e.g. an uncertainty band that grows
    towards large $x$); here it is the constant $1+\delta$. With the default
    `config.use_isospin_sym = True` MCEq also applies the isospin partners:
    $p\to\pi^\pm$ modifies $n\to\pi^\mp$, $p\to K^\pm$ modifies $n\to K^\pm$,
    and $K^0_{S,L}$ production receives half of each charged-kaon change.

    The modification is only stored by `set_mod_pprod`; the transport matrix
    is not touched until `regenerate_matrices()` rebuilds it
    (`skip_decay_matrix=True` keeps the unchanged decay matrix).
    `unset_mod_pprod()` clears all modifications and rebuilds the nominal
    matrices; always clear before setting a new value for the same channel,
    since a second call with the same function and label is ignored.
    """)
    return


@app.cell
def _(k_minus, k_plus, loop_progress, mceq_run, nominal, np, numu_components,
      pi_minus, pi_plus, zeniths):
    _ = nominal  # the nominal scan runs first, on unmodified matrices

    # (projectile, secondary) -> fractional change of the yield
    modifications = {
        (2212, 211): pi_plus / 100.0,
        (2212, -211): pi_minus / 100.0,
        (2212, 321): k_plus / 100.0,
        (2212, -321): k_minus / 100.0,
    }

    def flat_weight(xmat, egrid, label, delta):
        """x_func for set_mod_pprod: the constant factor (1 + delta)."""
        return (1.0 + delta) * np.ones_like(xmat)

    mceq_run.unset_mod_pprod(dont_fill=True)
    for (projectile, secondary), delta in modifications.items():
        if delta != 0.0:
            mceq_run.set_mod_pprod(
                projectile, secondary, flat_weight, ("delta", delta)
            )
    mceq_run.regenerate_matrices(skip_decay_matrix=True)
    try:
        with loop_progress(len(zeniths), "Modified · zenith scan") as _advance:
            modified = numu_components(mceq_run, zeniths, _advance)
    finally:
        # Leave the run nominal for any later cell or re-execution.
        mceq_run.unset_mod_pprod()
    "modified solved"
    return (modified,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 04 · Effect on the $\nu_\mu$ flux

    One column per zenith angle. Top: $\nu_\mu+\bar\nu_\mu$ from all
    sources (solid), pion (dashed), kaon (dash-dotted) and muon (dotted)
    decays; nominal in dark, modified in orange. Bottom: modified / nominal
    per component. The $\nu_\mu+\bar\nu_\mu$ sum responds to a mean of the
    $+$ and $-$ factors weighted by the charge ratio: proton–air produces
    mostly $\pi^+$ and $K^+$, so the default $+45\,\%/-55\,\%$ kaon change
    leaves only about $+5\,\%$ in the kaon component. Muon-decay neutrinos
    inherit the pion change. The kaon share of $\nu_\mu$ grows with energy
    and is largest near the vertical, so there the total follows the kaon
    ratio; near the horizon pions decay before interacting up to higher
    energies and the pion ratio sets the total over a wider range.
    """)
    return


@app.cell
def _(e_grid, interaction_model, k_minus, k_plus, modified, nominal, np,
      pi_minus, pi_plus, plt, zeniths):
    plot_range = (2.0, 1.0e5)  # GeV
    plot_mask = (e_grid >= plot_range[0]) & (e_grid <= plot_range[1])
    case_colors = {"nominal": "#263342", "modified": "#C46B30"}
    mother_styles = {
        "total": ("total", "-"),
        "pi": (r"from $\pi^\pm$", "--"),
        "k": (r"from $K^{\pm,0}$", "-."),
        "mu": (r"from $\mu^\pm$", ":"),
    }
    ratio_colors = {"total": "#263342", "pi": "#17857E", "k": "#C46B30", "mu": "#8A6FB0"}

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
            2, len(zeniths), figsize=(4.0 * len(zeniths) + 1.5, 6.4),
            sharex=True, sharey="row", layout="constrained", squeeze=False,
            gridspec_kw={"height_ratios": (3, 1.4)},
        )
        for column, theta in enumerate(zeniths):
            ax, ratio_ax = axes[0, column], axes[1, column]
            for case, fluxes in (("nominal", nominal), ("modified", modified)):
                for mother, (_label, linestyle) in mother_styles.items():
                    values = fluxes[theta][mother]
                    visible = plot_mask & (values > 0)
                    ax.loglog(
                        e_grid[visible], values[visible],
                        color=case_colors[case], linestyle=linestyle,
                        linewidth=1.8,
                    )
            for mother, (label, linestyle) in mother_styles.items():
                reference = nominal[theta][mother]
                ratio = np.divide(
                    modified[theta][mother], reference,
                    out=np.full_like(reference, np.nan), where=reference > 0,
                )
                ratio_ax.semilogx(
                    e_grid[plot_mask], ratio[plot_mask], label=label,
                    color=ratio_colors[mother], linestyle=linestyle,
                    linewidth=1.8,
                )
            ax.set_title(f"θ = {theta:g}°", loc="left", fontsize=13, pad=10)
            ax.set_xlim(plot_range)
            ax.set_ylim(1.0e-4, 1.0)
            ax.grid(axis="y", which="major", color="#E6E9ED", linewidth=0.7)
            ratio_ax.axhline(1.0, color="#B3BAC2", linewidth=0.8)
            ratio_ax.set_xlabel(r"Neutrino energy $E$ [GeV]")
            ratio_ax.grid(axis="y", color="#E6E9ED", linewidth=0.7)
        axes[0, 0].set_ylabel(
            r"$E^3\,\mathrm{d}\Phi/\mathrm{d}E$ [GeV$^2$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]"
        )
        axes[1, 0].set_ylabel("modified / nominal")

        handles = [
            plt.Line2D([], [], color=color, linewidth=2.5, label=case)
            for case, color in case_colors.items()
        ] + [
            plt.Line2D([], [], color="#52606D", linestyle=linestyle, label=label)
            for label, linestyle in mother_styles.values()
        ]
        fig.legend(
            handles=handles, frameon=False, loc="outside lower center",
            ncol=6, fontsize=10,
        )
        fig.suptitle(
            f"{interaction_model}  /  π⁺ {pi_plus:+.0f}%  π⁻ {pi_minus:+.0f}%  "
            f"K⁺ {k_plus:+.0f}%  K⁻ {k_minus:+.0f}%  /  MSIS-00 South Pole, January",
            fontsize=11, color="#52606D",
        )

    fig
    return (fig,)


@app.cell(hide_code=True)
def export_notebook(settings):
    from MCEq.notebook import download_jupyter

    download_jupyter(filename="modify_particle_yields.ipynb", configuration=settings)
    return


if __name__ == "__main__":
    app.run()
