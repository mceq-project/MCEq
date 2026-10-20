# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo==0.24.0", "nbformat", "MCEq>=2.0.0", "matplotlib>=3.10", "numpy", "scipy", "tqdm", "crflux>1.0.5"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Full-sky fluxes with solve_fullsky")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Full-sky neutrino fluxes and the geomagnetic cutoff

    Muon-neutrino flux on a (zenith, azimuth) sky grid above Tokyo, solved in
    one batched call, with and without the geomagnetic rigidity cutoff.
    """)
    return


@app.cell
def _():
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from MCEq import config
    from MCEq.core import MCEqRun
    from MCEq.notebook import apply_kernel_and_grid, loop_progress, solve_progress

    return (
        MCEqRun,
        apply_kernel_and_grid,
        config,
        loop_progress,
        mo,
        np,
        plt,
        solve_progress,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 01 · Configuration

    The physics is pinned so that the batch machinery is the subject:
    SIBYLL-2.3e above 80 GeV blended with FLUKA below
    (`low_energy_model="FLUKA20251"`), the H3a primary flux, and the
    NRLMSISE-00 January atmosphere at Tokyo, a site with cutoffs of 7–30 GV.
    The FLUKA blend matters here: the cutoff removes primaries below a few
    tens of GeV, and their secondaries come from the low-energy model.

    - **Sky-grid step**: spacing of the zenith (0° up to below the horizon)
      and azimuth grids; the number of pixels grows with the inverse
      square.
    - **Energy grid**: the cutoff acts on GeV-scale leptons; `e_min` must
      stay below the plotted 1 GeV and `e_max` need not be large.
    - **Backend**: `config.kernel_config` for the solver.
    """)
    return


@app.cell(hide_code=True)
def configuration_defaults():
    defaults = {
        "database": "mceq_base_air_1d_v2.h5",
        "step_deg": 20.0,
        "e_min": 0.1,
        "e_max": 1.0e6,
        "backend": "auto",
    }
    return (defaults,)


@app.cell(hide_code=True)
def configuration_controls(defaults, mo):
    import math

    from MCEq.notebook import backend_options

    backends = backend_options()
    backend_label = next(
        (label for label, value in backends.items() if value == defaults["backend"]),
        "Automatic",
    )
    controls = mo.md(
        "<div style=\"display:grid;grid-template-columns:repeat(auto-fit,"
        "minmax(min(100%,350px),1fr));gap:1rem 2rem\">\n"
        "<div>\n\n**Sky grid**\n\n{step_deg}\n\n</div><div>\n\n"
        "**Cascade grid**\n\n{log_e_min}\n\n{log_e_max}\n\n</div><div>\n\n"
        "**Solver**\n\n{backend}\n\n</div></div>"
    ).batch(
        step_deg=mo.ui.slider(
            10, 30, step=5, value=defaults["step_deg"],
            label="Sky grid step [°]", show_value=True, full_width=True,
        ),
        log_e_min=mo.ui.slider(
            -1, 0, step=0.5, value=math.log10(defaults["e_min"]),
            label="e_min · log₁₀(E/GeV)", show_value=True, full_width=True,
        ),
        log_e_max=mo.ui.slider(
            4, 9, step=0.5, value=math.log10(defaults["e_max"]),
            label="e_max · log₁₀(E/GeV)", show_value=True, full_width=True,
        ),
        backend=mo.ui.dropdown(
            backends, value=backend_label, label="Backend", full_width=True,
        ),
    ).form(submit_button_label="Apply configuration", bordered=False)
    controls
    return (controls,)


@app.cell(hide_code=True)
def selected_configuration(controls, defaults, mo):
    from MCEq.notebook import apply_configuration

    settings = apply_configuration(controls, defaults)
    database = settings["database"]
    step_deg = settings["step_deg"]
    e_min, e_max = settings["e_min"], settings["e_max"]
    backend = settings["backend"]
    mo.md(
        f"**Applied:** {step_deg:g}° sky grid · {e_min:g}–{e_max:g} GeV"
    )
    return backend, database, e_max, e_min, settings, step_deg


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
    ## 02 · Run and sky grid

    The geomagnetic cutoff is a property of the run
    (`MCEqRun(..., geomagnetic_cutoff=True)`, default off) or of a single
    `solve_fullsky(..., geomagnetic_cutoff=...)` call. It needs an
    atmosphere with a geographic location and a primary model with mass
    groups.
    """)
    return


@app.cell
def _(MCEqRun, apply_kernel_and_grid, backend, config, database, e_max, e_min,
      np, step_deg, proceed):
    _ = proceed
    import crflux.models as pm

    apply_kernel_and_grid(backend, e_min, e_max, 0)
    config.mceq_db_fname = database

    mceq = MCEqRun(
        interaction_model="SIBYLL23E",
        low_energy_model="FLUKA20251",
        primary_model=(pm.HillasGaisser2012, "H3a"),
        theta_deg=0.0,
        density_model=("MSIS00", ("Tokyo", "January")),
    )
    zenith_grid = np.arange(0.0, 90.0, step_deg)
    azimuth_grid = np.arange(0.0, 360.0, step_deg)
    (
        f"dim_states = {mceq.dim_states}, {mceq.e_grid.size} energy bins · "
        f"sky grid {zenith_grid.size} × {azimuth_grid.size} = "
        f"{zenith_grid.size * azimuth_grid.size} pixels"
    )
    return azimuth_grid, mceq, zenith_grid


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 03 · Serial scan versus one batched call

    Without a cutoff the MSIS00 atmosphere does not depend on azimuth, so the
    zenith grid alone is the full answer. The serial loop solves one zenith
    after the other; `solve_fullsky(zenith_grid)` builds one integration
    path per zenith and advances all columns together on the carousel
    (`carousel_K`, default `min(K, 128)` columns). Both must agree.
    """)
    return


@app.cell
def _(loop_progress, mceq, np, time, zenith_grid):
    # A grid energy near 1 GeV, so that no interpolation enters the comparison
    e_index = int(np.argmin(np.abs(mceq.e_grid - 1.0)))
    energy = float(mceq.e_grid[e_index])
    serial_numu = np.empty(zenith_grid.size)
    _start = time.perf_counter()
    with loop_progress(zenith_grid.size, "serial zenith scan") as advance:
        for _i, _zenith in enumerate(zenith_grid):
            mceq.set_zenith_azimuth(float(_zenith))
            mceq.solve()
            advance()
            _flux = mceq.get_solution("total_numu") + mceq.get_solution(
                "total_antinumu"
            )
            serial_numu[_i] = _flux[e_index]
    serial_seconds = time.perf_counter() - _start
    f"serial: {serial_seconds:.1f} s for {zenith_grid.size} zenith angles"
    return energy, serial_numu, serial_seconds


@app.cell
def _(mceq, serial_seconds, solve_progress, time, zenith_grid):
    _start = time.perf_counter()
    with solve_progress(f"solve_fullsky, {zenith_grid.size} zeniths") as progress:
        no_cutoff = mceq.solve_fullsky(
            zenith_grid, geomagnetic_cutoff=False, progress=progress
        )
    batch_seconds = time.perf_counter() - _start
    (
        f"solve_fullsky: {batch_seconds:.1f} s (serial {serial_seconds:.1f} s) · "
        f"steps per column {no_cutoff.nsteps_per_col.tolist()}"
    )
    return (no_cutoff,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 04 · Full sky with the rigidity cutoff

    With `geomagnetic_cutoff=True`, `solve_fullsky` looks up the cutoff
    $R_c$ of every pixel with `MCEq.geometry.cutoff.get_cutoff_map`
    (tabulated for named sites such as Tokyo; traced with `gtracr` and
    cached for other locations) and removes, per pixel, the primary nuclei
    with rigidity below $R_c$ from the initial state
    (`MCEq.batch.build_phi0_with_cutoff`). Every pixel is now a separate
    column.
    """)
    return


@app.cell
def _(azimuth_grid, mceq, solve_progress, zenith_grid):
    from MCEq.geometry.cutoff import get_cutoff_map

    cutoff_GV = get_cutoff_map(mceq.density_model, zenith_grid, azimuth_grid)
    with solve_progress(
        f"solve_fullsky with cutoff, {zenith_grid.size * azimuth_grid.size} pixels"
    ) as progress_cut:
        with_cutoff = mceq.solve_fullsky(
            zenith_grid, azimuth_grid, geomagnetic_cutoff=True,
            progress=progress_cut,
        )
    with_cutoff
    return cutoff_GV, with_cutoff


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 05 · Skymaps

    `result.skymap(name, E)` returns the `(n_zenith, n_azimuth)` flux at
    energy `E`, interpolated between grid points (here `E` is a grid
    point). Left: the cutoff map (azimuth
    measured as in the atmosphere model). Middle: the fraction of the ~1 GeV
    $\nu_\mu + \bar\nu_\mu$ flux that survives the cutoff, which follows the
    cutoff map. Right: the azimuth-averaged zenith dependence; the serial
    scan (crosses) and the batched solve (line) coincide.
    """)
    return


@app.cell
def _(azimuth_grid, cutoff_GV, energy, no_cutoff, np, plt, serial_numu,
      with_cutoff, zenith_grid):
    def numu_map(result):
        return result.skymap("total_numu", energy) + result.skymap(
            "total_antinumu", energy
        )

    reference = numu_map(no_cutoff)  # (n_zenith, 1)
    suppression = numu_map(with_cutoff) / reference

    def edges(centres, step):
        return np.append(centres - step / 2, centres[-1] + step / 2)

    step = azimuth_grid[1] - azimuth_grid[0]
    az_edges, zen_edges = edges(azimuth_grid, step), edges(zenith_grid, step)

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
        fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), layout="constrained")
        for ax, values, label, cmap in (
            (axes[0], cutoff_GV, "Rigidity cutoff $R_c$ [GV]", "cividis"),
            (axes[1], suppression, "Flux with / without cutoff", "viridis"),
        ):
            mesh = ax.pcolormesh(az_edges, zen_edges, values, cmap=cmap)
            fig.colorbar(mesh, ax=ax, label=label, pad=0.02)
            ax.set_xlabel("Azimuth [°]")
            ax.set_ylabel("Zenith [°]")
            ax.set_ylim(zen_edges[-1], 0.0)
        axes[0].set_title("Cutoff map, Tokyo", loc="left", fontsize=12, pad=10)
        axes[1].set_title(
            rf"$\nu_\mu + \bar\nu_\mu$ at {energy:.2g} GeV", loc="left",
            fontsize=12, pad=10,
        )

        cos_theta = np.cos(np.radians(zenith_grid))
        ax = axes[2]
        ax.plot(
            cos_theta, reference[:, 0], color="#263342", linewidth=2,
            label="solve_fullsky, no cutoff",
        )
        ax.plot(
            cos_theta, serial_numu, linestyle="none", marker="+", markersize=10,
            color="#C46B30", label="serial scan, no cutoff",
        )
        ax.plot(
            cos_theta, numu_map(with_cutoff).mean(axis=1), color="#17857E",
            linewidth=2, linestyle="--", label="with cutoff, azimuth average",
        )
        ax.set_xlabel(r"$\cos\theta$")
        ax.set_ylabel(r"$\Phi$ [GeV$^{-1}$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
        ax.set_title("Zenith dependence", loc="left", fontsize=12, pad=10)
        ax.grid(axis="y", color="#E6E9ED", linewidth=0.7)
        ax.legend(frameon=False, fontsize=9)
    fig
    return (fig,)


@app.cell(hide_code=True)
def export_notebook(settings):
    from MCEq.notebook import download_jupyter

    download_jupyter(filename="full_sky_carousel.ipynb", configuration=settings)
    return


if __name__ == "__main__":
    app.run()
