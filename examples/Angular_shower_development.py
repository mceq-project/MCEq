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

    The 2D angular cascade on the v2 ETD2RK integrator: interaction and decay
    matrices are one block-diagonal stitched CSR of shape
    $(n_k \cdot N, n_k \cdot N)$; muon multiple Coulomb scattering enters the
    diagonal as $-\kappa^2\theta_s^2(E)/4$; the angular reconstruction uses the
    legacy cubic-interp + trapezoidal Hankel inversion. 2D mode is detected
    automatically from a database carrying a `k_dim` attribute — this notebook
    needs a **2D interaction database** (not redistributable with MCEq; see
    `docs/examples` README pointer); a 1D database will not run it.

    This notebook is intentionally left out of the CI notebook gate: the only
    2D asset compatible with the v2 backend is the FLUKA 2D database, which
    cannot be shipped or downloaded in CI.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import ticker

    from MCEq import config
    from MCEq.core import MCEqRun
    from MCEq.notebook import apply_kernel_and_grid, download_jupyter

    return (
        MCEqRun, apply_kernel_and_grid, config, download_jupyter, mo, np, plt,
        ticker,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 01 · Configuration

    The 2D database is discovered from whatever carries `k_dim` under the MCEq
    data directory. Physics switches (EM off, muon losses and helicity
    dependence on) are those of the original demo.
    """)
    return


@app.cell(hide_code=True)
def configuration_defaults():
    defaults = {
        "database_2d": "mceq_db_v2_fluka2d_rc7.h5",
        "interaction_model": "FLUKA20251",
        "primary": "GlobalSplineFitBeta",
        "theta_deg": 30.0,
        "e_min": 0.1,
        "e_max": 1.0e4,
        "backend": "auto",
    }
    return (defaults,)


@app.cell(hide_code=True)
def configuration_controls(defaults, mo):
    import math

    from MCEq.notebook import backend_options

    def available_2d_databases():
        import h5py

        found = []
        for candidate in sorted(config.data_dir.glob("*.h5")):
            try:
                with h5py.File(candidate, "r") as source:
                    if "k_dim" in dict(source["common"].attrs):
                        found.append(candidate.name)
            except Exception:
                continue
        return found

    options = available_2d_databases()
    present = defaults["database_2d"] in options
    database = mo.ui.dropdown(
        options or [defaults["database_2d"]],
        value=defaults["database_2d"],
        label="2D database", full_width=True, disabled=not options,
    )
    backends = backend_options()
    status = mo.md(
        "⚠️ no database with `k_dim` found — copy a 2D asset into the MCEq "
        "data directory" if not options else "✅ 2D database found"
    )
    controls = mo.md(
        "<div style=\"display:grid;grid-template-columns:repeat(auto-fit,"
        "minmax(min(100%,350px),1fr));gap:1rem 2rem\">\n"
        "<div>\n\n**2D database**\n\n{database}</div><div>\n\n"
        "**Geometry**\n\n{theta_deg}\n\n</div><div>\n\n**Cascade grid**"
        "\n\n{log_e_min}\n\n{log_e_max}\n\n</div><div>\n\n**Solver**\n\n"
        "{backend}\n\n</div></div>"
    ).batch(
        database=database,
        theta_deg=mo.ui.slider(
            0, 80, step=5, value=defaults["theta_deg"],
            label="Shower inclination [°]", show_value=True, full_width=True,
        ),
        log_e_min=mo.ui.slider(
            -1, 1, step=0.5, value=math.log10(defaults["e_min"]),
            label="e_min · log₁₀(E/GeV)", show_value=True, full_width=True,
        ),
        log_e_max=mo.ui.slider(
            3, 6, step=0.5, value=math.log10(defaults["e_max"]),
            label="e_max · log₁₀(E/GeV)", show_value=True, full_width=True,
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
    [status, controls]
    return (controls, present)


@app.cell(hide_code=True)
def selected_configuration(controls, defaults, mo, present):
    from MCEq.notebook import apply_configuration

    settings = apply_configuration(controls, defaults)
    database_2d = settings["database_2d"]
    interaction_model = settings["interaction_model"]
    theta_deg = settings["theta_deg"]
    e_min, e_max = settings["e_min"], settings["e_max"]
    backend = settings["backend"]
    mo.md(
        "**Applied:** "
        + (
            f"{interaction_model} · {database_2d} · {theta_deg:g}° · "
            f"{e_min:g}–{e_max:g} GeV"
            if present
            else "no 2D database available — sections below are skipped"
        )
    )
    return (
        backend, database_2d, e_max, e_min, interaction_model, present,
        settings, theta_deg,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 02 · Build the 2D run

    `GlobalSplineFitBeta` primary, a single 100 GeV proton injected at 15 km /
    5 km / 200 m / sea level snapshots.
    """)
    return


@app.cell
def _(MCEqRun, apply_kernel_and_grid, backend, config, database_2d, e_max,
      e_min, interaction_model, np, present, theta_deg):
    if present:
        import crflux.models as pm

        apply_kernel_and_grid(backend, e_min, e_max, 0)
        config.mceq_db_fname = database_2d
        config.enable_default_tracking = False
        config.enable_em = False
        config.enable_em_ion = False
        config.muon_energy_loss = True
        config.generic_losses_all_charged = True
        config.enable_cont_rad_loss = True
        config.enable_energy_loss = True
        config.muon_helicity_dependence = True
        config.muon_multiple_scattering = True
        config.density_model = ("CORSIKA", ("USStd", None))
        config.adv_set["force_resonance"] = [421, 431, 411, 310]
        config.adv_set["disabled_particles"] = [22, 111, 16, 11]

        mceq_2d = MCEqRun(
            theta_deg=theta_deg,
            interaction_model=interaction_model,
            primary_model=(pm.GlobalSplineFitBeta, None),
        )
        density = mceq_2d.density_model
        mceq_egrid = mceq_2d.e_grid
        mceq_ebins = mceq_2d.e_bins
        # ``h2X(0)`` round-trips a hair above max_X, which stops the path
        # from snapshotting at the surface; nudge below max_X to keep all four.
        save_depths = np.minimum(
            density.h2X([1500000, 500000, 20000, 0]),
            density.max_X * (1.0 - 1e-9),
        )
        mceq_2d.set_single_primary_particle(100, pdg_id=2212)
        ediffs = np.diff(mceq_ebins)
    else:
        mceq_2d = density = mceq_egrid = mceq_ebins = save_depths = ediffs = None
    return (
        density, ediffs, mceq_2d, mceq_ebins, mceq_egrid, save_depths,
    )


@app.cell
def _(mceq_2d, mo, np, present, save_depths):
    def solve_2d():
        import time

        t0 = time.perf_counter()
        mceq_2d.solve(int_grid=save_depths)
        elapsed = time.perf_counter() - t0
        return (
            f"v2-2D ETD2RK solve: {elapsed:.1f} s, "
            f"{mceq_2d.integration_path[0]} steps, "
            f"{mceq_2d._mceq_db.n_k} Hankel modes, "
            f"dim_states = {mceq_2d.dim_states}"
        )

    status_2d = solve_2d() if present else "skipped — no 2D database"
    status_2d
    return (status_2d,)


@app.cell
def _(mceq_2d, np, present):
    if present:
        n_k = mceq_2d._mceq_db.n_k
        hankel_history = [
            snapshot.reshape(n_k, mceq_2d.dim_states)
            for snapshot in mceq_2d.grid_sol
        ]
    else:
        hankel_history = None
    hankel_history is not None and f"{len(hankel_history)} depth snapshots"
    return (hankel_history,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 03 · Reconstruction

    `convert_to_theta_space` returns $(\theta\text{ grid}, E, \theta,
    \text{depth})$ angular densities by inverse Hankel transform; the $k=0$
    row is the 1D limit for overlay.
    """)
    return


@app.cell
def _(hankel_history, mceq_2d, np, present):
    if present:
        theta_by_particle = {}
        solution_1d_by_particle = {}
        for pdg in (12, 13, 14):
            helicities = [-1, 0, 1] if pdg == 13 else [0]
            for hel in helicities:
                start = mceq_2d.pman.pdg2mceqidx[(pdg, hel)] * len(
                    mceq_2d.e_grid
                )
                end = (mceq_2d.pman.pdg2mceqidx[(pdg, hel)] + 1) * len(
                    mceq_2d.e_grid
                )
                _, _, theta_range, ang_vs_depth = (
                    mceq_2d.convert_to_theta_space(hankel_history, pdg, hel)
                )
                theta_by_particle[(pdg, hel)] = (
                    np.array(theta_range),
                    np.array(ang_vs_depth),
                )
                solution_1d_by_particle[(pdg, hel)] = np.array([
                    hankel[0, start:end] for hankel in hankel_history
                ])
            theta_by_particle[pdg] = theta_by_particle[(pdg, 0)]
            solution_1d_by_particle[pdg] = solution_1d_by_particle[(pdg, 0)]
    else:
        theta_by_particle = solution_1d_by_particle = None
    (
        "reconstructed: " + ", ".join(str(p) for p in (12, 13, 14))
        if present else "skipped — no 2D database"
    )
    return (solution_1d_by_particle, theta_by_particle)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 04 · 1D spectrum from the $k=0$ Hankel mode
    """)
    return


@app.cell
def _(density, ediffs, mceq_egrid, plt, present, save_depths,
      solution_1d_by_particle):
    def plot_1d():
        pdg = 14
        depth_ind = 3
        latex = {12: r"$\nu_\mu$", 14: r"$\nu_\mu$", 13: r"$\mu^-$"}[pdg]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(
            mceq_egrid,
            solution_1d_by_particle[pdg][depth_ind] * ediffs,
            lw=2.5, label="1D limit (k=0)", color="C0",
        )
        ax.grid(True)
        ax.set_xscale("log")
        ax.set_xlim(0.8, 1000)
        ax.set_xlabel("$E$ [GeV]")
        ax.set_ylabel(r"$(\mathrm{d}N/\mathrm{d}E) \times \Delta E$")
        ax.legend(loc="upper right")
        ax.set_title(
            latex + f" at {density.X2h(save_depths[depth_ind]) / 1e5:.0f} km"
            f" ($X = {save_depths[depth_ind]:.0f}\\,\\mathrm{{g\\,cm^{{-2}}}}$)"
        )
        return fig

    figure_1d = plot_1d() if present else "skipped — no 2D database"
    figure_1d
    return (figure_1d,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 05 · Angular densities at fixed energies

    $E\,\mathrm{d}N/\mathrm{d}E\,\mathrm{d}\theta$ scaled by $\theta$ for
    direct comparison with e.g. CORSIKA output, per depth and energy bin.
    """)
    return


@app.cell
def _(density, ediffs, mceq_ebins, plt, present, save_depths, theta_by_particle):
    def plot_2d():
        pdg = 14
        energies = [1, 2, 5]
        bins = np.digitize(energies, mceq_ebins) - 1
        colors = ["#24abd4", "#1c4ea3", "#4e1675"]
        fig, axes = plt.subplots(
            1, len(save_depths) - 1, figsize=(4 * (len(save_depths) - 1), 5)
        )
        for ix, ax in enumerate(axes[: len(save_depths) - 1]):
            theta_range, ang = theta_by_particle[pdg]
            for i, this_ebin in enumerate(bins):
                curve = ang[ix][this_ebin] * theta_range * (
                    np.diff(theta_range)[0]
                )
                ax.plot(
                    np.degrees(theta_range), curve * ediffs[this_ebin],
                    lw=2.5, color=colors[i],
                    label=(
                        rf"$E$ = {mceq_ebins[this_ebin]:.1f}.."
                        f"{mceq_ebins[this_ebin + 1]:.1f} GeV"
                    ),
                )
            ax.set_title(
                rf"$\nu_\mu$ at {density.X2h(save_depths[ix]) / 1e5:.0f} km"
                f" ($X = {save_depths[ix]:.0f}$)"
            )
            ax.set_xlim(0, 30)
            ax.set_xlabel(r"$\theta\,[^\circ]$")
            ax.set_xscale("symlog", linthresh=1)
            ax.legend(fontsize="small")
        fig.tight_layout()
        return fig

    figure_2d = plot_2d() if present else "skipped — no 2D database"
    figure_2d
    return (figure_2d,)


@app.cell(hide_code=True)
def export_notebook(download_jupyter, settings):
    download_jupyter(
        filename="Angular_shower_development.ipynb", configuration=settings
    )
    return


if __name__ == "__main__":
    app.run()
