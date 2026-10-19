# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo==0.24.0", "nbformat", "MCEq>=2.0.0", "matplotlib>=3.10", "numpy", "crflux>1.0.5"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Tracking lepton mothers")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tracking lepton mothers

    Which hadron a lepton came from, as a fraction of the total flux: charm
    decays overtake kaons between ~100 TeV ($\nu_e$) and a few PeV
    ($\nu_\mu$), and leptons produced directly at nucleon interaction
    vertices are the largest source of prompt muons.

    This notebook is a walkthrough of MCEq v2's particle tracking
    (`ParticleManager.track_leptons_from`): choose the mother groups, see
    which tracking species MCEq creates, and read every group from one
    solve. Tracking covers both decays and, with `from_interactions=True`,
    leptons emitted at the interaction vertex. For the classic absolute-flux
    figure with fixed groups (arXiv:1503.00544, Fig. 9), see
    `partial_hadron_contribution.py`.
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
        prepare_database,
        solve_progress,
    )

    return (
        MCEqRun,
        apply_kernel_and_grid,
        mo,
        np,
        plt,
        prepare_database,
        solve_progress,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 01 · Configuration

    Pick the mother groups to track, then the usual run settings. Pion,
    kaon and muon-decay sources are MCEq's default tracking categories
    (`pi_`, `k_` + `K0_`, `mu_`) and are always shown. The two *vertex*
    groups track leptons produced in the collisions of nucleons, or of
    charged pions and kaons, rather than in their decays. Use a model with
    charm (SIBYLL-2.3, DPMJET) for the $D$ groups; missing species are
    skipped.
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
        "e_max": 1.0e9,
        "backend": "auto",
        "mother_groups": ["D±", "D0", "Ds", "nucleon vertex", "π/K vertex"],
    }
    return (defaults,)


@app.cell(hide_code=True)
def configuration_controls(defaults, mo):
    from MCEq.notebook import configuration_controls as build_controls

    group_picker = mo.ui.multiselect(
        ["D±", "D0", "Ds", "nucleon vertex", "π/K vertex"],
        value=defaults["mother_groups"],
        label="Mother groups to track", full_width=True,
    )
    controls, _options = build_controls(
        defaults,
        database=defaults["database"],
        show=("interaction_model", "primary", "zenith", "energy", "backend"),
    )
    mo.vstack([group_picker, controls])
    return controls, group_picker


@app.cell(hide_code=True)
def selected_configuration(controls, defaults, group_picker, mo):
    from MCEq.notebook import apply_configuration, primary_model_from

    settings = apply_configuration(controls, defaults)
    settings["mother_groups"] = list(group_picker.value)
    database = settings["database"]
    interaction_model = settings["interaction_model"]
    primary_model = primary_model_from(settings["primary"])
    theta_deg = settings["theta_deg"]
    e_min, e_max = settings["e_min"], settings["e_max"]
    backend = settings["backend"]
    mother_groups = settings["mother_groups"]
    mo.md(
        f"**Applied:** {interaction_model} · {settings['primary']} · "
        f"{theta_deg:g}° · groups: {', '.join(mother_groups) or 'none'}"
    )
    return (
        backend, database, e_max, e_min, interaction_model, mother_groups,
        primary_model, settings, theta_deg,
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
    ## 02 · Register the groups

    `pman.track_leptons_from(parents, prefix, from_interactions=False)`
    creates one tracking species `prefix + lepton` for every lepton
    ($\mu^\pm$, $\nu$, $\bar\nu$ of all flavours) that the parents can
    produce. A tracking species is transported like its lepton but receives
    only the chosen channels, so it adds a row to the state vector and
    nothing else. `parents` are `(PDG ID, helicity)` keys, taken here from
    `pman[name].pdg_id`; antiparticles are listed explicitly (or use
    `include_antiparticle=True`). New species need new matrices, hence the
    single `regenerate_matrices()` afterwards; `get_solution(prefix + lepton)`
    then reads each group.
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
    mother_groups,
    prepare_database,
    primary_model,
    theta_deg,
    proceed):
    _ = proceed
    from MCEq import config

    apply_kernel_and_grid(backend, e_min, e_max, 0)
    config.mceq_db_fname = await prepare_database(database)

    # group label -> (parents, tracking prefix, from_interactions)
    group_definitions = {
        "D±": (["D+", "D-"], "Dpm_", False),
        "D0": (["D0", "Dbar0"], "D0_", False),
        "Ds": (["D_s+", "D_s-"], "Ds_", False),
        "nucleon vertex": (["p+", "pbar-", "n0", "nbar0"], "Nvtx_", True),
        "π/K vertex": (["pi+", "pi-", "K+", "K-"], "piKvtx_", True),
    }

    mceq_run = MCEqRun(
        interaction_model=interaction_model,
        primary_model=primary_model,
        theta_deg=theta_deg,
    )
    tracked = {}
    for _label in mother_groups:
        _members, _prefix, _from_interactions = group_definitions[_label]
        _parents = [
            mceq_run.pman[name].pdg_id for name in _members if name in mceq_run.pman
        ]
        if not _parents:
            continue
        mceq_run.pman.track_leptons_from(
            _parents, _prefix, from_interactions=_from_interactions
        )
        tracked[_label] = _prefix
    mceq_run.regenerate_matrices()

    for _label, _prefix in tracked.items():
        _species = [
            p.name for p in mceq_run.pman.cascade_particles
            if p.is_tracking and p.name.startswith(_prefix)
        ]
        print(f"{_label:>15}: {len(_species)} species, e.g. {', '.join(_species[:4])}")
    return mceq_run, tracked


@app.cell
def _(interaction_model, mceq_run, solve_progress, theta_deg):
    with solve_progress(
        f"{interaction_model} · {theta_deg:g}° · tracked mothers"
    ) as progress:
        mceq_run.solve(progress=progress)
    "solved"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 03 · Fractions of the total flux

    Each curve is a group's flux divided by the total (particle +
    antiparticle). Grey: conventional (dashed) and prompt (dotted) shares
    from the `conv_`/`pr_` split. Groups add up to one only where every
    source is tracked.
    """)
    return


@app.cell
def _(mceq_run, np, tracked):
    leptons = {
        "mu": ("mu+", "mu-"),
        "numu": ("numu", "antinumu"),
        "nue": ("nue", "antinue"),
        "nutau": ("nutau", "antinutau"),
    }
    sources = {
        r"$\pi^\pm$": "pi_",
        r"$K^\pm + K^0$": ("k_", "K0_"),
        r"$\mu$ decay": "mu_",
        **tracked,
    }

    def summed(prefixes, lepton):
        if isinstance(prefixes, str):
            prefixes = (prefixes,)
        return sum(
            mceq_run.get_solution(prefix + particle)
            for prefix in prefixes
            for particle in leptons[lepton]
        )

    fractions = {}
    for _lepton in leptons:
        _total = summed("total_", _lepton)
        _safe = np.where(_total > 0, _total, np.inf)
        fractions[_lepton] = {
            label: summed(prefixes, _lepton) / _safe
            for label, prefixes in {
                "conv.": "conv_", "prompt": "pr_", **sources
            }.items()
        }
    e_grid = mceq_run.e_grid
    return e_grid, fractions


@app.cell
def _(e_grid, fractions, interaction_model, np, plt, theta_deg):
    palette = [
        "#C0392B", "#17857E", "#D98CB3", "#E08E2B", "#3FA7D6",
        "#7C5CBF", "#263342", "#7FB77E",
    ]
    titles = {
        "mu": r"$\mu^+ + \mu^-$",
        "numu": r"$\nu_\mu + \bar\nu_\mu$",
        "nue": r"$\nu_e + \bar\nu_e$",
        "nutau": r"$\nu_\tau + \bar\nu_\tau$",
    }
    plot_range = (10.0, 1.0e8)
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
            2, 2, figsize=(11.5, 7.6), sharex=True, sharey=True,
            layout="constrained",
        )
        for ax, (lepton, title) in zip(axes.ravel(), titles.items()):
            curves = fractions[lepton]
            for label, linestyle in (("conv.", "--"), ("prompt", ":")):
                ax.loglog(
                    e_grid[plot_mask], curves[label][plot_mask], color="#8C959F",
                    linestyle=linestyle, linewidth=1.6, label=label,
                )
            source_labels = [k for k in curves if k not in ("conv.", "prompt")]
            for color, label in zip(palette, source_labels):
                values = np.where(curves[label] > 0, curves[label], np.nan)
                ax.loglog(
                    e_grid[plot_mask], values[plot_mask], color=color,
                    linewidth=1.8, label=label,
                )
            ax.set_title(title, loc="left", fontsize=14, pad=10)
            ax.set_xlim(plot_range)
            ax.set_ylim(1.0e-4, 1.5)
            ax.grid(axis="y", which="major", color="#E6E9ED", linewidth=0.7)
            ax.set_axisbelow(True)
        for ax in axes[1]:
            ax.set_xlabel(r"Kinetic energy $E$ [GeV]")
        for ax in axes[:, 0]:
            ax.set_ylabel("fraction of total flux")
        fig.legend(
            *axes[0, 0].get_legend_handles_labels(), frameon=False,
            loc="outside lower center", ncol=5, fontsize=10,
        )
        fig.suptitle(
            f"{interaction_model}  /  {theta_deg:g}° zenith",
            fontsize=11, color="#52606D",
        )

    fig
    return (fig,)


@app.cell(hide_code=True)
def export_notebook(settings):
    from MCEq.notebook import download_jupyter

    download_jupyter(
        filename="partial_hadron_fractions.ipynb", configuration=settings
    )
    return


if __name__ == "__main__":
    app.run()
