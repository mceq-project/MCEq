# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo==0.24.0", "nbformat", "MCEq>=2.0.0", "matplotlib>=3.10", "numpy", "crflux>1.0.5"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Partial hadron contributions (modern)")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Partial hadron contributions — modern

    The v2 way to decompose lepton fluxes by mother hadron:
    `ParticleManager.track_leptons_from` registers the mother groups *before*
    construction finishes, so every group lives in one matrix and the whole
    decomposition costs **one solve** — unlike the vintage
    `set_obs_particles` loop over separate runs. Pick the charm groups to
    register; pion, kaon, and muon-decay sources use MCEq's built-in tracking
    prefixes. Fractions are shown relative to the total flux.
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
        solve_progress,
    )

    return (
        MCEqRun,
        apply_kernel_and_grid,
        config,
        download_jupyter,
        mo,
        np,
        plt,
        solve_progress,
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
        "interaction_model": "SIBYLL23E",
        "primary": "HillasGaisser2012:H3a",
        "theta_deg": 60.0,
        "e_min": 1.0,
        "e_max": 1.0e9,
        "backend": "auto",
        "charm_groups": ["D±", "D0", "Ds"],
    }
    return (defaults,)


@app.cell(hide_code=True)
def configuration_controls(defaults, mo):
    from MCEq.notebook import configuration_controls as build_controls

    charm = mo.ui.multiselect(
        ["D±", "D0", "Ds"], value=defaults["charm_groups"],
        label="Charm groups to track", full_width=True,
    )
    controls, _options = build_controls(
        defaults,
        database=defaults["database"],
        show=("interaction_model", "primary", "zenith", "energy", "backend"),
    )
    charm_and_form = [charm, controls]
    charm_and_form
    return (charm_and_form,)


@app.cell(hide_code=True)
def selected_configuration(charm_and_form, defaults, mo):
    from MCEq.notebook import apply_configuration, primary_model_from

    charm_widget = charm_and_form[0]
    charm_groups = (
        charm_widget.value
        if charm_widget.value is not None
        else defaults["charm_groups"]
    )
    settings = apply_configuration(charm_and_form[1], defaults)
    database = settings["database"]
    interaction_model = settings["interaction_model"]
    primary_model = primary_model_from(settings["primary"])
    theta_deg = settings["theta_deg"]
    e_min, e_max = settings["e_min"], settings["e_max"]
    backend = settings["backend"]
    mo.md(
        f"**Applied:** {interaction_model} · {settings['primary']} · "
        f"{theta_deg:g}° · charm: {', '.join(charm_groups) or 'none'}"
    )
    return (
        backend, charm_groups, database, e_max, e_min, interaction_model,
        primary_model, settings, theta_deg,
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
    ## 02 · Register groups, solve once
    """)
    return


@app.cell
def _(MCEqRun, apply_kernel_and_grid, backend, charm_groups, config, database,
      e_max, e_min, interaction_model, primary_model, theta_deg, proceed):
    _ = proceed
    apply_kernel_and_grid(backend, e_min, e_max, 0)
    config.mceq_db_fname = database

    charm_options = {
        "D±": (["D+", "D-"], "grpDpm_"),
        "D0": (["D0", "Dbar0"], "grpD0_"),
        "Ds": (["D_s+", "D_s-"], "grpDs_"),
    }
    registered = {
        "pi": r"$\pi$", "k": "$K$", "mu": r"$\mu$ decay",
    }
    # values are display labels; keys are tracking prefixes

    mceq_run = MCEqRun(
        interaction_model=interaction_model,
        primary_model=primary_model,
        theta_deg=theta_deg,
    )
    for group, (members, track_prefix) in charm_options.items():
        if group not in charm_groups:
            continue
        present = []
        for member in members:
            try:
                present.append(mceq_run.pman[member].pdg_id)
            except KeyError:
                pass
        if present:
            mceq_run.pman.track_leptons_from(present, track_prefix)
            registered[track_prefix.rstrip("_")] = group

    mceq_run.regenerate_matrices()
    return (mceq_run, registered)


@app.cell
def _(interaction_model, mceq_run, solve_progress, theta_deg):
    with solve_progress(
        f"{interaction_model} · {theta_deg:g}° · charm decomposition"
    ) as progress:
        mceq_run.solve(progress=progress)
    "solved"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 03 · Fractions of the total flux

    Solid: mother contribution / total; dashed: conventional share; dotted:
    prompt share. Groups the model does not track (charm in older models) are
    skipped automatically.
    """)
    return


@app.cell
def _(mceq_run, np, plt, registered):
    mag = 3
    colors = {
        "pi": "red", "k": "green", "mu": "pink",
        "grpDpm": "orangered", "grpD0": "mediumblue", "grpDs": "cyan",
    }
    leptons = {
        "mu": ("mu+", "mu-"),
        "numu": ("numu", "antinumu"),
        "nue": ("nue", "antinue"),
        "nutau": ("nutau", "antinutau"),
    }

    def flux(species):
        try:
            curve = mceq_run.get_solution(species, mag)
            return np.where(np.isfinite(curve), curve, 0.0)
        except Exception:
            return None

    def combined(prefix, base):
        parts = [flux(f"{prefix}_{p}") for p in leptons[base]]
        if any(part is None for part in parts):
            return None
        return sum(parts)

    e_grid = mceq_run.e_grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    titles = {
        "mu": r"$\mu^+ + \mu^-$", "numu": r"$\nu_\mu + \bar\nu_\mu$",
        "nue": r"$\nu_e + \bar\nu_e$", "nutau": r"$\nu_\tau + \bar\nu_\tau$",
    }
    for ax, (base, _particles) in zip(axes.ravel(), list(leptons.items())):
        total = combined("total", base)
        if total is None or not np.any(total > 0):
            ax.set_visible(False)
            continue
        ax.loglog()
        ax.axhline(1.0, color="k", lw=1.0, ls="-")
        conv = combined("conv", base)
        prompt = combined("pr", base)
        if conv is not None and prompt is not None:
            ax.loglog(e_grid, np.maximum(conv, 1e-300) / total, ls="--",
                      lw=1.2, color="grey", label="conv / total")
            ax.loglog(e_grid, np.maximum(prompt, 1e-300) / total, ls=":",
                      lw=1.2, color="grey", label="prompt / total")
        for prefix, source_label in registered.items():
            curve = combined(prefix + "_", base)
            if curve is None or not np.any(curve > 0):
                continue
            ax.loglog(e_grid, curve / total, ls="-", lw=1.5,
                      color=colors.get(prefix, "magenta"), label=source_label)
        ax.set_xlim(10, 1e8)
        ax.set_ylim(1e-6, 3)
        ax.set_title(titles[base])
        ax.set_xlabel("E [GeV]")
        ax.legend(loc="lower left", frameon=False, fontsize="small")
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def export_notebook(download_jupyter, settings):
    download_jupyter(
        filename="Partial_hadron_fractions.ipynb", configuration=settings
    )
    return


if __name__ == "__main__":
    app.run()
