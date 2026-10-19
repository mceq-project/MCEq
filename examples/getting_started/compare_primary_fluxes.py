# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo==0.24.0", "nbformat", "MCEq>=2.0.0", "matplotlib>=3.10", "numpy", "crflux>1.0.5"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Dependence on primary CR flux")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dependence on primary cosmic-ray flux

    Sea-level muon and neutrino fluxes for four primary spectra
    (HG2012 H3a/H4a, GST 3-/4-gen), zenith-averaged over
    $\cos\theta\in[0,1]$ in 11 bins. The comparison set is the point of this
    notebook; the controls change the hadronic model, geometry, grid, and
    solver.
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
    """)
    return


@app.cell(hide_code=True)
def configuration_defaults():
    defaults = {
        "database": "mceq_db_lext_dpm193_v142.h5",
        "interaction_model": "SIBYLL21",
        "theta_deg": 0.0,
        "e_min": 1.0,
        "e_max": 1.0e9,
        "backend": "auto",
    }
    return (defaults,)


@app.cell(hide_code=True)
def configuration_controls(defaults, mo):
    from MCEq.notebook import configuration_controls as build_controls

    controls, _options = build_controls(
        defaults,
        database=defaults["database"],
        show=("interaction_model", "zenith", "energy", "backend"),
    )
    controls
    return (controls,)


@app.cell(hide_code=True)
def selected_configuration(controls, defaults, mo):
    from MCEq.notebook import apply_configuration

    settings = apply_configuration(controls, defaults)

    database = settings["database"]
    interaction_model = settings["interaction_model"]
    theta_deg = settings["theta_deg"]
    e_min, e_max = settings["e_min"], settings["e_max"]
    backend = settings["backend"]
    mo.md(
        f"**Applied:** {interaction_model} · {theta_deg:g}° reference zenith · "
        f"{e_min:g}–{e_max:g} GeV"
    )
    return (
        backend, database, e_max, e_min, interaction_model, settings, theta_deg,
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
    ## 02 · Zenith-averaged fluxes per primary model

    Fluxes are summed as particle + antiparticle, scaled by $E^3$, and
    averaged over the $\cos\theta$ grid for each primary spectrum.
    """)
    return


@app.cell
def _(MCEqRun, apply_kernel_and_grid, backend, config, database, e_max, e_min,
      interaction_model, np, theta_deg, proceed):
    _ = proceed
    import crflux.models as pm

    apply_kernel_and_grid(backend, e_min, e_max, 0)
    config.mceq_db_fname = database

    primary_models = [
        (pm.HillasGaisser2012, "H3a"),
        (pm.HillasGaisser2012, "H4a"),
        (pm.GaisserStanevTilav, "3-gen"),
        (pm.GaisserStanevTilav, "4-gen"),
    ]
    angles = np.arccos(np.linspace(1, 0, 11)) * 180.0 / np.pi
    mag = 3
    prefix = {
        "mu_conv": ("conv", ("mu+", "mu-")),
        "mu_pr": ("pr", ("mu+", "mu-")),
        "mu_total": ("total", ("mu+", "mu-")),
        "numu_conv": ("conv", ("numu", "antinumu")),
        "numu_pr": ("pr", ("numu", "antinumu")),
        "numu_total": ("total", ("numu", "antinumu")),
        "nue_conv": ("conv", ("nue", "antinue")),
        "nue_pr": ("pr", ("nue", "antinue")),
        "nue_total": ("total", ("nue", "antinue")),
        "nutau_pr": ("total", ("nutau", "antinutau")),
    }

    mceq_run = MCEqRun(
        interaction_model=interaction_model,
        primary_model=primary_models[0],
        theta_deg=theta_deg,
    )
    e_grid = mceq_run.e_grid

    p_spectrum_flux = []

    def scan(advance):
        for pmodel in primary_models:
            mceq_run.set_primary_model(*pmodel)
            flux = {frac: np.zeros_like(e_grid) for frac in prefix}
            for theta in angles:
                mceq_run.set_zenith_azimuth(theta)
                mceq_run.solve()
                advance(title=f"{pmodel[0].__name__} {(pmodel[1] or '')}")
                for frac, (track, particles) in prefix.items():
                    flux[frac] += sum(
                        mceq_run.get_solution(f"{track}_{particle}", mag)
                        for particle in particles
                    )
            p_spectrum_flux.append((
                {frac: total / len(angles) for frac, total in flux.items()},
                mceq_run.pmodel.sname,
                mceq_run.pmodel.name,
            ))

    with loop_progress(
        len(primary_models) * len(angles), "zenith x primary scan"
    ) as advance:
        scan(advance)
    return (e_grid, mag, p_spectrum_flux)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 03 · Comparison

    Solid: total; dashed: conventional; dotted: prompt.
    """)
    return


@app.cell
def _(e_grid, mag, p_spectrum_flux, plt):
    figures = []
    with plt.rc_context({"figure.figsize": (4.5, 3.5)}):
        for pref, lab in [("numu_", r"\nu_\mu"), ("mu_", r"\mu"), ("nue_", r"\nu_e")]:
            fig, ax = plt.subplots()
            for entry, col in zip(p_spectrum_flux, ["k", "r", "g", "b"]):
                saved, label, _ = entry
                ax.loglog(
                    e_grid, saved[pref + "total"], color=col, ls="-", lw=2.5,
                    label=label, alpha=0.4,
                )
                ax.loglog(
                    e_grid, saved[pref + "conv"], color=col, ls="--", lw=1,
                    label="_nolabel_",
                )
                ax.loglog(
                    e_grid, saved[pref + "pr"], color=col, ls=":", lw=1,
                    label="_nolabel_",
                )
            ax.set_xlim(50, 1e9)
            ax.set_ylim(1e-5, 1)
            ax.set_xlabel(rf"$E_{{{lab}}}$ [GeV]")
            ax.set_ylabel(
                r"$\Phi_{" + lab + "}$ (E/GeV)$^{" + str(mag) + " }$"
                + "(cm$^{2}$ s sr GeV)$^{-1}$"
            )
            ax.legend(loc="upper right")
            fig.tight_layout()
            figures.append(fig)
    figures
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 04 · ASCII export

    Zenith-averaged spectra per primary model, for external processing.
    """)
    return


@app.cell
def _(e_grid, np, p_spectrum_flux):
    import os
    import tempfile

    components = [
        "mu_conv", "mu_pr", "mu_total",
        "numu_conv", "numu_pr", "numu_total",
        "nue_conv", "nue_pr", "nue_total", "nutau_pr",
    ]
    desktop = tempfile.mkdtemp(prefix="mceq_example_")
    for saved_flux, p_sname, _ in p_spectrum_flux:
        np.savetxt(
            os.path.join(desktop, "numu_flux_" + p_sname + ".txt"),
            np.column_stack([e_grid, *[saved_flux[frac] for frac in components]]),
            fmt="%6.5E",
            header=(
                "lepton flux scaled with E**3. Order (E, mu_conv, mu_pr, "
                "mu_total, numu_conv, numu_pr, numu_total, nue_conv, nue_pr, "
                "nue_total, nutau_pr)"
            ),
        )
    return (desktop,)


@app.cell(hide_code=True)
def export_notebook(download_jupyter, settings):
    download_jupyter(
        filename="Compare_primary_fluxes.ipynb", configuration=settings
    )
    return


if __name__ == "__main__":
    app.run()
