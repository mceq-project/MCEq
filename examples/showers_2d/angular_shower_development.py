# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo==0.24.0", "nbformat", "MCEq>=2.0.0", "matplotlib>=3.10", "numpy", "crflux>1.0.5"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Angular shower development (2D cascade)")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Angular shower development (2D cascade)

    In a single proton shower, GeV muon neutrinos spread to several degrees
    around the shower axis, while those of a few tens of GeV stay within
    about a degree.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from MCEq import config
    from MCEq.core import MCEqRun
    from MCEq.notebook import apply_kernel_and_grid, solve_progress

    return MCEqRun, apply_kernel_and_grid, config, mo, np, plt, solve_progress


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 01 · Configuration

    A 2D run is selected by the database: `config.mceq_db_fname =
    "mceq_base_air_2d_legendre_v2.h5"`, the default 2D package of the data
    release (FLUKA20251; SIBYLL23E comes from a companion model package,
    fetched on first use). It stores every yield as Legendre moments
    $\langle P_l(\cos\theta)\rangle$ on 90 degrees $l$ up to 2000, and MCEq
    transports all modes at once, so a 2D solve is far more expensive than a
    1D one. The cascade grid is therefore kept just wide enough for the
    chosen primary energy.
    """)
    return


@app.cell(hide_code=True)
def configuration_defaults():
    defaults = {
        "database": "mceq_base_air_2d_legendre_v2.h5",
        "interaction_model": "FLUKA20251",
        "primary_energy": 100.0,
        "theta_deg": 30.0,
        "e_min": 0.5,
        "e_max": 300.0,
        "backend": "auto",
    }
    return (defaults,)


@app.cell(hide_code=True)
def configuration_controls(defaults, mo):
    from MCEq.notebook import backend_options, database_models

    models = database_models(defaults["database"])
    backends = backend_options()
    controls = mo.md("""
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,350px),1fr));gap:1.5rem 2rem">
    <div>

    <strong>Shower</strong>

    {interaction_model}

    {primary_energy}

    {theta_deg}

    </div><div>

    <strong>Solver</strong>

    {backend}

    </div></div>
    """).batch(
        interaction_model=mo.ui.dropdown(
            models, value=defaults["interaction_model"],
            label="Interaction model", full_width=True,
        ),
        primary_energy=mo.ui.dropdown(
            {"20 GeV": 20.0, "50 GeV": 50.0, "100 GeV": 100.0, "200 GeV": 200.0},
            value=f"{defaults['primary_energy']:g} GeV",
            label="Primary proton energy", full_width=True,
        ),
        theta_deg=mo.ui.slider(
            0, 80, step=5, value=defaults["theta_deg"],
            label="Shower inclination [°]", show_value=True, full_width=True,
        ),
        backend=mo.ui.dropdown(
            backends,
            value=next(
                (k for k, v in backends.items() if v == defaults["backend"]),
                "Automatic",
            ),
            label="Backend", full_width=True,
        ),
    ).form(submit_button_label="Apply configuration", bordered=False)
    controls
    return (controls,)


@app.cell(hide_code=True)
def selected_configuration(controls, defaults, mo):
    settings = dict(defaults)
    if controls.value is not None:
        settings.update(controls.value)

    database = settings["database"]
    interaction_model = settings["interaction_model"]
    primary_energy = settings["primary_energy"]
    theta_deg = settings["theta_deg"]
    e_min, e_max = settings["e_min"], settings["e_max"]
    backend = settings["backend"]
    mo.md(
        f"**Applied:** {interaction_model} · {primary_energy:g} GeV proton · "
        f"{theta_deg:g}° · {database}"
    )
    return (
        backend, database, e_max, e_min, interaction_model, primary_energy,
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
    ## 02 · Build and solve the 2D run

    The physics switches keep their defaults: muon multiple scattering
    (`config.muon_multiple_scattering`), helicity-dependent muon decays and
    continuous energy losses on, the electromagnetic cascade off. To shrink
    the state vector, which every one of the 90 modes carries, the default
    lepton-origin tracking is switched off, D mesons and $K^0_S$ are forced
    to decay at production (`adv_set["force_resonance"]`), and photons,
    $\pi^0$, electrons and $\nu_\tau$ are dropped
    (`adv_set["disabled_particles"]`). The path controller uses the 1D step
    settings (`eps=0.03`, `dX_max=5` g/cm²) instead of the tighter 2D default;
    on this grid that changes the angular densities by less than $10^{-3}$
    and halves the runtime. The state is recorded at 15 km, 5 km and sea
    level.
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
    np,
    primary_energy,
    theta_deg,
    proceed):
    _ = proceed
    apply_kernel_and_grid(backend, e_min, e_max, 0)
    config.mceq_db_fname = database
    config.enable_default_tracking = False
    config.adv_set["force_resonance"] = [421, 431, 411, 310]
    config.adv_set["disabled_particles"] = [22, 111, 16, 11]

    mceq_2d = MCEqRun(
        interaction_model=interaction_model,
        low_energy_model="FLUKA20251",  # below 80 GeV
        primary_model=None,
        theta_deg=theta_deg,
        density_model=("CORSIKA", ("USStd", None)),
    )
    mceq_2d.set_single_primary_particle(primary_energy, pdg_id=2212)
    density = mceq_2d.density_model
    altitudes_km = np.array([15.0, 5.0, 0.0])
    # h2X(0) can round a hair above max_X, which would skip the
    # sea-level snapshot; stay just below it.
    save_depths = np.minimum(
        density.h2X(altitudes_km * 1e5), density.max_X * (1.0 - 1e-9)
    )
    return altitudes_km, mceq_2d, save_depths


@app.cell
def _(interaction_model, mceq_2d, save_depths, solve_progress, theta_deg):
    with solve_progress(
        f"2D · {interaction_model} · {theta_deg:g}°"
    ) as progress:
        mceq_2d.solve(int_grid=save_depths, eps=0.03, dX_max=5.0, progress=progress)
    (
        f"{mceq_2d.integration_path[0]} steps · {mceq_2d._mceq_db.n_k} "
        f"Legendre modes · {mceq_2d.dim_states} states per mode"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 03 · Read out the modes

    `grid_sol` holds one flat vector per recorded depth; reshaped to
    `(n_k, dim_states)`, row $l$ is the moment $\langle P_l\rangle$ of
    $\mathrm{d}N/\mathrm{d}E$ for every species and energy. Row $l=0$ is the
    angle-integrated spectrum, the 1D result.
    `convert_to_theta_space(modes, pdg_id, hel, log_theta=True)` sums the
    Legendre series (interpolated in $l$, Jackson-damped) and returns
    `(l_grid, moments, theta, f)`, with `f[depth][energy]` the density in the
    planar measure $\theta\,\mathrm{d}\theta$, i.e.
    $\mathrm{d}N/\mathrm{d}E\,\mathrm{d}\theta = f\,\theta$.
    """)
    return


@app.cell
def _(mceq_2d, np):
    n_k = mceq_2d._mceq_db.n_k
    mode_history = [
        snapshot.reshape(n_k, mceq_2d.dim_states) for snapshot in mceq_2d.grid_sol
    ]
    energy = mceq_2d.e_grid
    energy_bins = mceq_2d.e_bins
    numu_slice = slice(
        mceq_2d.pman.pdg2mceqidx[(14, 0)] * len(energy),
        (mceq_2d.pman.pdg2mceqidx[(14, 0)] + 1) * len(energy),
    )
    # Angle-integrated numu spectra dN/dE, one per depth
    numu_1d = np.array([modes[0, numu_slice] for modes in mode_history])
    _, _, theta, numu_f = mceq_2d.convert_to_theta_space(
        mode_history, pdg_id=14, hel=0, theta_res=400, log_theta=True
    )
    numu_f = np.asarray(numu_f)  # (depth, energy, theta)
    return energy, energy_bins, numu_1d, numu_f, theta


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 04 · Angular profiles of $\nu_\mu$

    Top: the angle-integrated $\nu_\mu$ spectrum, $E\,\mathrm{d}N/\mathrm{d}E$,
    at the three depths. Bottom: $f\,\theta^2/a_0$, the fraction of the
    neutrinos in an energy bin per logarithmic interval of the angle to the
    shower axis ($\mathrm{d}N/\mathrm{d}\ln\theta$ normalised to unit area),
    for three neutrino energies.
    The peak position is the typical angle, set by the transverse momenta in
    production and decay over the energy; it shrinks with energy, more
    slowly than $1/E$. The shapes barely change below 15 km: most neutrinos
    are produced high up. The small ripples at large angles are the
    truncation of the Legendre series at $l = 2000$.
    """)
    return


@app.cell
def _(
    altitudes_km,
    energy,
    energy_bins,
    interaction_model,
    np,
    numu_1d,
    numu_f,
    plt,
    primary_energy,
    theta,
    theta_deg):
    depth_colors = ["#8FA3B8", "#17857E", "#263342"]
    energy_colors = ["#C46B30", "#17857E", "#263342"]
    probe_energies = [1.0, 5.0, 25.0]  # GeV
    probe_bins = np.digitize(probe_energies, energy_bins) - 1
    theta_degrees = np.degrees(theta)

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
        fig = plt.figure(figsize=(11.5, 7.6), layout="constrained")
        top, bottom = fig.subfigures(2, 1, height_ratios=(1, 1.15))
        spectrum_ax = top.subplots()
        for depth_index, altitude in enumerate(altitudes_km):
            values = energy * numu_1d[depth_index]
            visible = (values > 0) & (energy <= primary_energy)
            spectrum_ax.loglog(
                energy[visible], values[visible], linewidth=2,
                color=depth_colors[depth_index], label=f"{altitude:g} km",
            )
        spectrum_ax.set_xlim(energy_bins[0], primary_energy)
        spectrum_ax.set_xlabel(r"Neutrino energy $E$ [GeV]")
        spectrum_ax.set_ylabel(r"$E\,\mathrm{d}N/\mathrm{d}E$")
        spectrum_ax.set_title(
            r"$\nu_\mu$, angle-integrated ($l=0$ mode)", loc="left", fontsize=13
        )
        spectrum_ax.grid(axis="y", which="major", color="#E6E9ED", linewidth=0.7)
        spectrum_ax.legend(frameon=False, title="altitude", fontsize=10)

        angle_axes = bottom.subplots(1, len(altitudes_km), sharey=True)
        for depth_index, (ax, altitude) in enumerate(zip(angle_axes, altitudes_km)):
            for color, bin_index in zip(energy_colors, probe_bins):
                # Unit area in ln(theta): int f theta^2 dln(theta) = a_0
                values = (
                    numu_f[depth_index, bin_index] * theta**2
                    / numu_1d[depth_index, bin_index]
                )
                visible = theta_degrees > 0.05
                ax.semilogx(
                    theta_degrees[visible], values[visible], linewidth=2,
                    color=color,
                    label=(
                        f"{energy_bins[bin_index]:.2g}–"
                        f"{energy_bins[bin_index + 1]:.2g} GeV"
                    ),
                )
            ax.set_title(f"{altitude:g} km", loc="left", fontsize=13)
            ax.set_xlim(0.05, 90)
            ax.set_xlabel(r"Angle to shower axis $\theta$ [°]")
            ax.grid(axis="y", which="major", color="#E6E9ED", linewidth=0.7)
        angle_axes[0].set_ylabel(r"$(1/N)\,\mathrm{d}N/\mathrm{d}\ln\theta$")
        bottom.legend(
            *angle_axes[0].get_legend_handles_labels(), frameon=False,
            loc="outside lower center", ncol=3, fontsize=10,
        )
        fig.suptitle(
            f"{interaction_model}  /  {primary_energy:g} GeV proton  /  "
            f"{theta_deg:g}° inclination  /  US Standard",
            fontsize=11, color="#52606D",
        )

    fig
    return (fig,)


@app.cell(hide_code=True)
def export_notebook(settings):
    from MCEq.notebook import download_jupyter

    download_jupyter(
        filename="angular_shower_development.ipynb", configuration=settings
    )
    return


if __name__ == "__main__":
    app.run()
