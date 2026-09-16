# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo==0.24.0", "nbformat", "MCEq>=2.0.0", "matplotlib>=3.10", "numpy", "tqdm", "crflux>1.0.5"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Full-sky fluxes with solve_fullsky")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Full-sky neutrino fluxes with `solve_fullsky`

    `MCEqRun.solve_fullsky` propagates one primary spectrum through every
    (zenith, azimuth) pixel of a sky grid in a **single multi-RHS solve**:
    the stage-5 LPT carousel packs pixels into a `K_pipe`-column pipeline, so
    total kernel work is $\sum_{\text{pixels}} \text{nsteps} \times
    \text{ms/RHS}$, not $\max(\text{nsteps}) \times K_{\text{total}}$. Works
    on all four backends.

    Sections below: serial baseline vs one-shot carousel, per-pixel flux
    extraction, a 2-D `phi0` hook, and the geomagnetic rigidity cutoff
    (requires `pymsis` for the MSIS21 atmospheres and `gtracr` for the cutoff
    map; sections skip themselves cleanly when a package is missing).
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
    from MCEq.notebook import (
        apply_kernel_and_grid,
        download_jupyter,
        loop_progress,
        solve_progress,
    )

    return (
        MCEqRun, apply_kernel_and_grid, config, download_jupyter, loop_progress,
        mo, np, plt, solve_progress, time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 01 · Configuration

    The full-sky machinery is the subject, so the demo pins SIBYLL2.3d / H3a
    and a South-Pole atmosphere; the controls set the sky-grid step and the
    cascade grid.
    """)
    return


@app.cell(hide_code=True)
def configuration_defaults():
    defaults = {
        "database": "mceq_db_lext_dpm193_v142.h5",
        "step_deg": 30.0,
        "e_min": 1.0,
        "e_max": 1.0e7,
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
            -1, 3, step=0.5, value=math.log10(defaults["e_min"]),
            label="e_min · log₁₀(E/GeV)", show_value=True, full_width=True,
        ),
        log_e_max=mo.ui.slider(
            4, 11, step=0.5, value=math.log10(defaults["e_max"]),
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
    e_min, e_max = settings["e_min"], settings["e_max"]
    backend = settings["backend"]
    step_deg = settings["step_deg"]
    n_zen = int(round(180.0 / step_deg)) + 1
    n_az = int(round(360.0 / step_deg))
    mo.md(
        f"**Applied:** {e_min:g}–{e_max:g} GeV · sky grid {n_zen}×{n_az} = "
        f"{n_zen * n_az} pixels"
    )
    return (
        backend, database, e_max, e_min, n_az, n_zen, settings, step_deg,
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
    ## 02 · Run and environment

    At the South Pole the column is azimuth-symmetric but the full-sky
    machinery still exercises $K = n_{zen} \times n_{az}$ distinct paths.
    Geometry defaults to NRLMSIS 2.1 when `pymsis` is installed, otherwise to
    CORSIKA US Standard with cutoffs off.
    """)
    return


@app.cell
def _(MCEqRun, apply_kernel_and_grid, backend, config, database, e_max, e_min, proceed):
    _ = proceed
    import importlib.util

    has_msis21 = importlib.util.find_spec("pymsis") is not None
    has_gtracr = importlib.util.find_spec("gtracr") is not None
    use_geomagnetic = has_msis21 and has_gtracr

    apply_kernel_and_grid(backend, e_min, e_max, 0)
    config.mceq_db_fname = database

    atmosphere = (
        ("MSIS21_IC", ("SouthPole", "January"))
        if has_msis21
        else ("CORSIKA", ("BK_USStd", None))
    )
    import crflux.models as pm

    mceq = MCEqRun(
        interaction_model="SIBYLL23D",
        primary_model=(pm.HillasGaisser2012, "H3a"),
        theta_deg=0.0,
        density_model=atmosphere,
    )
    f"MSIS21: {has_msis21} · gtracr: {has_gtracr} · atmosphere: "
    f"{atmosphere[0]} · dim_states={mceq.dim_states}, "
    f"n_E={mceq.e_grid.size}"
    return atmosphere, has_gtracr, has_msis21, mceq, use_geomagnetic


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 03 · Sky grid
    """)
    return


@app.cell
def _(mceq, n_az, n_zen, np, step_deg):
    zen_centres = np.linspace(0.0, 180.0, n_zen)
    az_centres = np.arange(n_az) * step_deg
    K = n_zen * n_az
    E_target_GeV = np.array([1.0, 10.0, 100.0])
    f"grid: {n_zen} zen × {n_az} az = K = {K} pixels"
    return E_target_GeV, K, az_centres, zen_centres


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 04 · Baseline: serial pixel scan

    A sub-sampled grid keeps the baseline quick; the same call is what
    `solve_fullsky` replaces.
    """)
    return


@app.cell
def _(E_target_GeV, K, az_centres, loop_progress, mceq, np, zen_centres):
    zen_sub = zen_centres[::3]
    az_sub = az_centres[::9] if az_centres.size >= 9 else az_centres
    K_sub = zen_sub.size * az_sub.size

    log_grid = np.log(mceq.e_grid)
    log_targets = np.log(E_target_GeV)
    numu_sub = np.zeros((E_target_GeV.size, zen_sub.size, az_sub.size))

    t0s = time.time()
    with loop_progress(K_sub, "serial pixel scan") as advance:
        for i_z, zen in enumerate(zen_sub):
            for i_a, az in enumerate(az_sub):
                mceq.set_zenith_azimuth(float(zen), float(az))
                mceq.solve()
                advance()
                flux = mceq.get_solution("total_numu", 0) + mceq.get_solution(
                    "total_antinumu", 0
                )
                numu_sub[:, i_z, i_a] = np.exp(
                    np.interp(log_targets, log_grid, np.log(np.maximum(flux, 1e-300)))
                )
    t_serial = time.time() - t0s
    f"serial wall: {t_serial:.1f}s ({t_serial / K_sub * 1e3:.1f} ms / pixel)"
    return (K_sub, numu_sub, t_serial)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 05 · One-shot carousel

    `carousel_K=None` resolves to `min(K, 128)`. The cutoff is disabled here
    when active so the serial baseline and the carousel solve the same
    physics; toggle it in the cell to see the skymap effect.
    """)
    return


@app.cell
def _(K, az_centres, mceq, solve_progress, t_serial, zen_centres):
    t0c = time.time()
    with solve_progress(f"carousel full sky (K = {K})") as progress:
        res = mceq.solve_fullsky(
            zen_centres, az_centres, geomagnetic_cutoff=False,
            progress=progress,
        )
    t_carousel = time.time() - t0c
    (
        f"solve_fullsky (carousel): {t_carousel:.2f}s for K = {K} · "
        f"nsteps/pixel min {res.nsteps_per_col.min()} mean "
        f"{res.nsteps_per_col.mean():.0f} max {res.nsteps_per_col.max()} · "
        f"sol shape {res.sol.shape}"
    )
    return res, t_carousel


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 06 · Zenith dependence, carousel vs serial

    Azimuth-averaged; red crosses are the serial baseline (agreement verifies
    the multi-RHS path). `res.skymap('total_numu', E)` is the one-line
    shortcut; the manual slicing is kept to show the state-vector layout.
    """)
    return


@app.cell
def _(E_target_GeV, K, az_centres, numu_sub, n_zen, np, plt, res, zen_centres):
    pname2pref = res._mceq.pman.pname2pref
    e_grid = np.asarray(res._mceq.e_grid)
    numu_pixel_grid = np.zeros((e_grid.size, K), dtype=np.float64)
    for base in ("numu", "antinumu"):
        for variant in (base, base + "_l", base + "_r"):
            if variant in pname2pref:
                p = pname2pref[variant]
                numu_pixel_grid += res.sol[p.lidx : p.uidx, :]

    log_E = np.log(e_grid)
    log_flux = np.log(np.maximum(numu_pixel_grid, 1e-300))
    numu_at_E = np.empty((E_target_GeV.size, K), dtype=np.float64)
    for i, lt in enumerate(np.log(E_target_GeV)):
        ji = int(np.clip(np.searchsorted(log_E, lt, side="right"), 1,
                          log_E.size - 1))
        w = (lt - log_E[ji - 1]) / (log_E[ji] - log_E[ji - 1])
        numu_at_E[i] = np.exp((1.0 - w) * log_flux[ji - 1] + w * log_flux[ji])
    numu_at_E = numu_at_E.reshape(E_target_GeV.size, n_zen, az_centres.size)

    cos_theta = np.cos(np.deg2rad(zen_centres))
    cos_theta_sub = np.cos(np.deg2rad(np.asarray(zen_centres)[::3]))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for i, energy in enumerate(E_target_GeV):
        ax = axes[i]
        ax.semilogy(cos_theta, numu_at_E[i].mean(axis=1), "k-", lw=1.5,
                    label="solve_fullsky (az avg)")
        ax.plot(cos_theta_sub, numu_sub[i].mean(axis=1), "r+", ms=8,
                label="serial scan" if i == 0 else None)
        ax.set_xlabel(r"$\cos\theta$")
        ax.set_title(rf"$E_{{\nu_\mu}} = {energy:g}$ GeV")
    axes[0].set_ylabel(r"$\Phi_{\nu_\mu + \bar\nu_\mu}$ [(cm$^2$ s sr GeV)$^{-1}$]")
    axes[0].legend(loc="best", fontsize="small", frameon=False)
    fig.tight_layout()
    fig
    return (numu_at_E,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 07 · Geomagnetic cutoff and per-pixel primaries

    With `pymsis` + `gtracr`, `geomagnetic_cutoff=None` (default for
    MSIS-based atmospheres) builds and caches the rigidity map on first call.
    A 2-D `phi0` of shape `(dim_states, K)` lets the caller modify primaries
    per pixel — the demo below zeroes the up-going hemisphere purely to
    exercise the hook.
    """)
    return


@app.cell
def _(az_centres, mceq, np, solve_progress, use_geomagnetic, zen_centres):
    if use_geomagnetic:
        phi0_1d = mceq._phi0.copy()
        phi0_2d = np.broadcast_to(phi0_1d[:, None], (mceq.dim_states, K)).copy()
        upgoing = np.repeat(zen_centres[:, None] > 90.0, az_centres.size, axis=1)
        phi0_2d[:, upgoing.ravel()] = 0.0
        with solve_progress("masked carousel (up-going vetoed)") as _progress:
            sol_masked = mceq.solve_fullsky(
                zen_centres, az_centres, phi0=phi0_2d, progress=_progress
            )
        masked_note = (
            f"masked solve sol shape = {sol_masked.sol.shape} (up-going "
            "hemisphere zeroed)"
        )
    else:
        masked_note = (
            "skipped: this section needs `pymsis` and `gtracr` "
            "(`pip install pymsis gtracr`)"
        )
    masked_note
    return (masked_note,)


@app.cell(hide_code=True)
def export_notebook(download_jupyter, settings):
    download_jupyter(filename="Full_sky_carousel.ipynb", configuration=settings)
    return


if __name__ == "__main__":
    app.run()
