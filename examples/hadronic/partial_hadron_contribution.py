# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo==0.24.0", "nbformat", "MCEq>=2.0.0", "matplotlib>=3.10", "numpy", "crflux>1.0.5"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Partial hadron contributions (classic)")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Partial hadron contributions — classic figure

    Pions and kaons supply the lepton fluxes up to ~100 TeV; above that the
    prompt decays of charmed mesons take over, first for $\nu_e$, then
    $\nu_\mu$, and last for muons.

    This is the classic per-mother decomposition, Figure 9 of
    [Fedynitch et al., arXiv:1503.00544](https://arxiv.org/abs/1503.00544):
    a Thunman primary at 60° zenith, absolute $E^3$-weighted fluxes for
    $\mu$, $\nu_\mu$, $\nu_e$ and $\nu_\tau$, one curve per mother group.
    MCEq v1 computed it with `set_obs_particles`, one solve per group; that
    method is gone in v2, and the same curves come from tracking particles
    in a single solve. For the tracking API itself, with selectable groups
    and fractions of the total, see the companion notebook
    `partial_hadron_fractions.py`.
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
        solve_progress,
    )

    return (
        MCEqRun,
        apply_kernel_and_grid,
        mo,
        np,
        plt,
        solve_progress,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 01 · Configuration

    Primary (Thunman et al., the reference flux of the original figure) and
    zenith (60°) are fixed by the reproduction. The controls set the hadronic
    model — it must produce charm, as all SIBYLL-2.3 variants and DPMJET do —
    the cascade grid, and the solver backend. The figure extends to $10^9$ GeV,
    so keep `e_max` at least that high.
    """)
    return


@app.cell(hide_code=True)
def configuration_defaults():
    defaults = {
        "database": "mceq_base_air_1d_v2.h5",
        "interaction_model": "SIBYLL23E",
        "e_min": 1.0,
        "e_max": 1.0e9,
        "backend": "auto",
    }
    return (defaults,)


@app.cell(hide_code=True)
def configuration_controls(defaults):
    from MCEq.notebook import configuration_controls as build_controls

    controls, _options = build_controls(
        defaults,
        database=defaults["database"],
        show=("interaction_model", "energy", "backend"),
    )
    controls
    return (controls,)


@app.cell(hide_code=True)
def selected_configuration(controls, defaults, mo):
    from MCEq.notebook import apply_configuration

    settings = apply_configuration(controls, defaults)
    database = settings["database"]
    interaction_model = settings["interaction_model"]
    e_min, e_max = settings["e_min"], settings["e_max"]
    backend = settings["backend"]
    mo.md(
        f"**Applied:** {interaction_model} · Thunman · 60° · "
        f"{e_min:g}–{e_max:g} GeV"
    )
    return backend, database, e_max, e_min, interaction_model, settings


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
    ## 02 · Mother groups

    The conventional mothers need no setup: with `config.enable_default_tracking`
    (on by default) MCEq already tracks leptons from $\pi^\pm$ (`pi_`),
    $K^\pm$ (`k_`), $K^0_{S,L}$ (`K0_`) and muon decay (`mu_`), next to the
    `conv_`/`pr_` split. The charm groups are added with
    `pman.track_leptons_from(parents, prefix)` and the matrices rebuilt once.
    Whatever no group covers is shown as a remainder: *other conv.*
    (e.g. $\Lambda^0$ decays) and *other prompt* (leptons from the
    interaction vertex — unflavoured resonances such as $\eta$, $\omega$,
    $\phi$ — and charmed baryons, which the hadronic model decays itself).
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
    proceed):
    _ = proceed
    import crflux.models as pm

    from MCEq import config

    apply_kernel_and_grid(backend, e_min, e_max, 0)
    config.mceq_db_fname = database

    mceq_run = MCEqRun(
        interaction_model=interaction_model,
        primary_model=(pm.Thunman, None),
        theta_deg=60.0,
    )

    charm_groups = {
        "Dpm_": ["D+", "D-", "D0", "Dbar0"],
        "Ds_": ["D_s+", "D_s-"],
    }
    for _prefix, _members in charm_groups.items():
        _parents = [
            mceq_run.pman[name].pdg_id for name in _members if name in mceq_run.pman
        ]
        if _parents:
            mceq_run.pman.track_leptons_from(_parents, _prefix)
    mceq_run.regenerate_matrices()
    return (mceq_run,)


@app.cell
def _(interaction_model, mceq_run, solve_progress):
    with solve_progress(f"{interaction_model} · Thunman · 60°") as progress:
        mceq_run.solve(progress=progress)
    "solved"
    return


@app.cell
def _(mceq_run):
    leptons = {
        "mu": ("mu+", "mu-"),
        "numu": ("numu", "antinumu"),
        "nue": ("nue", "antinue"),
        "nutau": ("nutau", "antinutau"),
    }

    def summed(prefix, lepton):
        """E^3-weighted flux of one category, particle + antiparticle.

        Categories the model does not have (e.g. no charm) come back as zeros.
        """
        return sum(
            mceq_run.get_solution(prefix + particle, mag=3)
            for particle in leptons[lepton]
        )

    conventional = ("pi_", "k_", "K0_", "mu_")
    prompt = ("Dpm_", "Ds_")
    components = {}
    for _lepton in leptons:
        _parts = {
            prefix: summed(prefix, _lepton)
            for prefix in ("total_", "conv_", "pr_") + conventional + prompt
        }
        _parts["other_conv"] = _parts["conv_"] - sum(_parts[p] for p in conventional)
        _parts["other_pr"] = _parts["pr_"] - sum(_parts[p] for p in prompt)
        components[_lepton] = _parts
    e_grid = mceq_run.e_grid
    return components, e_grid


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 03 · Fluxes by mother

    Black: total (solid), conventional (dotted), prompt (dashed). Coloured:
    mother groups; grey and light-blue dashed: the conventional and prompt
    remainders.
    """)
    return


@app.cell
def _(components, e_grid, interaction_model, np, plt):
    sources = {
        "pi_": (r"$\pi^\pm$", "#C0392B", "-"),
        "k_": (r"$K^\pm$", "#17857E", "-"),
        "K0_": (r"$K^0_{S,L}$", "#7FB77E", "-"),
        "mu_": (r"$\mu$ decay", "#D98CB3", "-"),
        "Dpm_": (r"$D^\pm, D^0$", "#E08E2B", "-"),
        "Ds_": (r"$D_s$", "#3FA7D6", "-"),
        "other_conv": ("other conv.", "#8C959F", "--"),
        "other_pr": ("other prompt", "#9DC3E6", "--"),
    }
    totals = {
        "total_": ("total", "-"),
        "conv_": ("conv.", ":"),
        "pr_": ("prompt", "--"),
    }
    panels = {
        "mu": (r"$\mu^+ + \mu^-$", (1e-6, 1e0)),
        "numu": (r"$\nu_\mu + \bar\nu_\mu$", (1e-7, 1e-1)),
        "nue": (r"$\nu_e + \bar\nu_e$", (1e-7, 1e-1)),
        "nutau": (r"$\nu_\tau + \bar\nu_\tau$", (1e-9, 1e-3)),
    }
    plot_range = (10.0, 1.0e9)
    plot_mask = (e_grid >= plot_range[0]) & (e_grid <= plot_range[1])

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
            2, 2, figsize=(11.5, 8.0), sharex=True, layout="constrained"
        )
        for ax, (lepton, (title, ylim)) in zip(axes.ravel(), panels.items()):
            parts = components[lepton]
            for key, (label, linestyle) in totals.items():
                ax.loglog(
                    e_grid[plot_mask], parts[key][plot_mask], color="#263342",
                    linestyle=linestyle, linewidth=2.2, label=label,
                )
            for key, (label, color, linestyle) in sources.items():
                values = np.where(parts[key] > 0, parts[key], np.nan)
                ax.loglog(
                    e_grid[plot_mask], values[plot_mask], color=color,
                    linestyle=linestyle, linewidth=1.5, label=label,
                )
            ax.set_title(title, loc="left", fontsize=14, pad=10)
            ax.set_xlim(plot_range)
            ax.set_ylim(*ylim)
            ax.grid(axis="y", which="major", color="#E6E9ED", linewidth=0.7)
            ax.set_axisbelow(True)
        for ax in axes[1]:
            ax.set_xlabel(r"Kinetic energy $E$ [GeV]")
        for ax in axes[:, 0]:
            ax.set_ylabel(
                r"$E^3\,\mathrm{d}\Phi/\mathrm{d}E$ [GeV$^2$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]"
            )
        fig.legend(
            *axes[0, 0].get_legend_handles_labels(), frameon=False,
            loc="outside lower center", ncol=6, fontsize=10,
        )
        fig.suptitle(
            f"{interaction_model}  /  Thunman  /  60° zenith",
            fontsize=11, color="#52606D",
        )

    fig
    return (fig,)


@app.cell(hide_code=True)
def export_notebook(settings):
    from MCEq.notebook import download_jupyter

    download_jupyter(
        filename="partial_hadron_contribution.ipynb", configuration=settings
    )
    return


if __name__ == "__main__":
    app.run()
