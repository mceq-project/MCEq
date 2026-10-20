# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo==0.24.0", "nbformat", "MCEq>=2.0.0", "matplotlib>=3.10", "numpy"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="MCEq 2D vs CORSIKA 7: 1 TeV proton")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MCEq 2D vs CORSIKA 7: a 1 TeV proton shower

    Angular and energy distributions of hadrons, muons and neutrinos from a
    vertical 1 TeV proton, computed by MCEq's 2D cascade equations and compared
    with 5 million CORSIKA 7 showers.

    This reproduces the MCEq v2 paper comparison (Figs. `fig:corsika_2d_spectra`
    and `fig:corsika_2d_angular`). Below a few GeV the transverse momentum
    from interactions and decays, together with muon multiple scattering, spreads
    secondaries over tens of degrees. A one-dimensional (straight-ahead) cascade
    has no angle, so only the 2D solver (energy × angle, expanded in Legendre
    modes) can be compared particle by particle.
    """)
    return


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    from MCEq import config
    from MCEq.config.detect import resolve_kernel
    from MCEq.core import MCEqRun
    from MCEq.notebook import solve_progress

    # The last solved run, keyed by its physics options: changing only the
    # plotting choices in the form below reuses it instead of solving again.
    solved_runs = {}
    return (
        Line2D, MCEqRun, Path, config, json, mo, np, plt, resolve_kernel,
        solve_progress, solved_runs,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 01 · CORSIKA setup

    CORSIKA 7.8050, patched with a β-exact `PRANGE` fix and isospin-corrected
    K*± decays, built with FLUKA 2025.1 as the low-energy model
    (`CORLEMODEL=FLUKA`), `THIN` and `NEUTRINO` options. 2500 runs × 2000
    showers; the runs differ only in `RUNNR` and the random seeds. The steering
    card of the first run, verbatim:
    """)
    return


@app.cell
def _(Path, json, np):
    def find_case_dir(name="p1tev_vertical"):
        """examples/data/corsika/<name>, from the notebook file or the working directory."""
        try:
            here = Path(__file__).resolve().parent
        except NameError:  # Jupyter: no __file__, the kernel runs in the notebook folder
            here = Path.cwd()
        for base in (here, here.parent, Path.cwd(), Path.cwd().parent):
            for candidate in (base / "data" / "corsika" / name,
                              base / "examples" / "data" / "corsika" / name):
                if (candidate / "c7_reference.npz").is_file():
                    return candidate
        raise FileNotFoundError(f"examples/data/corsika/{name} not found")

    case_dir = find_case_dir()
    steering_card = (case_dir / "steering.inp").read_text()
    reference = dict(np.load(case_dir / "c7_reference.npz"))
    reference_meta = json.loads(str(reference.pop("meta")))
    print(steering_card)
    print(
        f"Reference: {reference_meta['n_showers']:,} showers ({reference_meta['shards']}), "
        f"{', '.join(lv['name'] for lv in reference_meta['levels'])}"
    )
    return (reference,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    How each card line is matched on the MCEq side:

    | CORSIKA card | Meaning | MCEq counterpart |
    |---|---|---|
    | `PRMPAR 14`, `ERANGE 1000 1000` | proton, 1 TeV total energy | `set_single_primary_particle(1e3, pdg_id=2212)` (1 TeV kinetic; the 0.9 GeV difference is irrelevant) |
    | `THETAP 0 0` | vertical | `theta_deg=0` |
    | `FIXCHI 0` | primary starts at the top of the atmosphere | injection at X = 0 (default) |
    | `OBSLEV 15 / 5 / 0 km` | three observation levels | `solve(int_grid=…)` at 124.14 / 552.95 / 1036.10 g/cm² |
    | `ATMOD 1` | US Standard atmosphere (Linsley) | `density_model=("CORSIKA", ("USStd", None))` |
    | `HILOW 1500`, FLUKA build | FLUKA for every hadronic interaction | `interaction_model="FLUKA20251"`, `low_energy_model=None` |
    | `ELMFLG F F` | no EM cascade (no EGS4, no NKG) | `config.enable_em = False` (default) |
    | `ECUTS 0.1 0.1 0.3 0.3` | hadron and muon cut 0.1 GeV | `config.e_min = 0.1` (default) |
    | `MUMULT T` | Molière muon scattering, point-like nuclei | `muon_multiple_scattering` on, `muon_scattering_finite_nuclei = False` |
    | `MAGNET 1e-5 1e-5` | geomagnetic field off | MCEq has no field |
    | `THIN 1e-20 1e30 0`, `THINH 1 1` | no thinning (all weights 1) | MCEq is not a Monte Carlo |
    | `PHIP`, `ESLOPE`, `LONGI`, `RADNKG`, `ARRANG`, `STEPFC`, `HADFLG` | azimuth, slope, longitudinal and NKG output, defaults | not relevant here |

    From the particle files, each particle's kinetic energy and its angle θ to
    the primary axis were histogrammed at the three levels: 2° bins in four narrow
    energy bins for the angular distributions, and the MCEq energy grid
    (10 bins per decade) for the spectra. All values are particles per bin per
    primary shower with Poisson errors. `examples/data/corsika/p1tev_vertical/`
    holds the reduced histograms, their `README.md` and the reduction script.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 02 · MCEq setup

    The matched settings are fixed by the card above. Two knobs are left open:

    - **Muon scattering, nuclear size** (`config.muon_scattering_finite_nuclei`).
      MCEq's default screened-Coulomb kernel includes the nuclear form factor
      (finite nuclei), which suppresses large-angle single scatters. CORSIKA's
      Molière treatment uses point nuclei, so the matched setting is *point
      nuclei*. Switching to *finite nuclei* shows the size of that effect: at
      ground, 0.3–2 GeV muons become ≈ 2 % more forward-peaked.
    - **Backend**: which solver kernel integrates the cascade. Results agree to
      machine precision; MKL or CUDA are faster where installed.

    The species and levels to plot only change the figures; applying them reuses
    the last solution.
    """)
    return


@app.cell(hide_code=True)
def configuration_defaults():
    defaults = {
        "database": "mceq_base_air_2d_legendre_v2.h5",
        "interaction_model": "FLUKA20251",
        "low_energy_model": None,
        "density_model": ("CORSIKA", ("USStd", None)),
        "theta_deg": 0.0,
        "muon_scattering_finite_nuclei": False,
        "backend": "auto",
        "plot_species": ["p", "n", "pi", "mu", "numu", "nue"],
        "plot_levels": ["15 km", "5 km", "ground"],
    }
    return (defaults,)


@app.cell(hide_code=True)
def configuration_controls(defaults, mo):
    from MCEq.notebook import backend_options

    backends = backend_options()
    species_options = {
        "p": "p", "n": "n", "π±": "pi", "μ±": "mu", "νμ + ν̄μ": "numu", "νe + ν̄e": "nue",
    }
    controls = mo.md("""
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,350px),1fr));gap:1.5rem 2rem">
    <div>

    <strong>Physics & solver</strong>

    {nuclei}

    {backend}

    </div><div>

    <strong>Figures</strong>

    {plot_species}

    {plot_levels}

    </div></div>
    """).batch(
        nuclei=mo.ui.dropdown(
            {"Point nuclei (matches CORSIKA)": "point", "Finite nuclei (MCEq default)": "finite"},
            value=(
                "Finite nuclei (MCEq default)"
                if defaults["muon_scattering_finite_nuclei"]
                else "Point nuclei (matches CORSIKA)"
            ),
            label="Muon scattering", full_width=True,
        ),
        backend=mo.ui.dropdown(
            backends,
            value=next((k for k, v in backends.items() if v == defaults["backend"]), "Automatic"),
            label="Backend", full_width=True,
        ),
        plot_species=mo.ui.multiselect(
            species_options,
            value=[k for k, v in species_options.items() if v in defaults["plot_species"]],
            label="Species", full_width=True,
        ),
        plot_levels=mo.ui.multiselect(
            ["15 km", "5 km", "ground"], value=defaults["plot_levels"],
            label="Observation levels", full_width=True,
        ),
    ).form(submit_button_label="Apply configuration", bordered=False)
    controls
    return (controls,)


@app.cell(hide_code=True)
def selected_configuration(controls, defaults, mo):
    settings = dict(defaults)
    if controls.value is not None:
        _form = dict(controls.value)
        settings["muon_scattering_finite_nuclei"] = _form.pop("nuclei") == "finite"
        settings.update(_form)
        # keep the canonical order and at least one entry
        settings["plot_species"] = [
            s for s in defaults["plot_species"] if s in settings["plot_species"]
        ] or defaults["plot_species"]
        settings["plot_levels"] = [
            lv for lv in defaults["plot_levels"] if lv in settings["plot_levels"]
        ] or defaults["plot_levels"]

    database = settings["database"]
    interaction_model = settings["interaction_model"]
    low_energy_model = settings["low_energy_model"]
    density_model = settings["density_model"]
    theta_deg = settings["theta_deg"]
    muon_scattering_finite_nuclei = settings["muon_scattering_finite_nuclei"]
    backend = settings["backend"]
    plot_species = settings["plot_species"]
    plot_levels = settings["plot_levels"]
    mo.md(
        f"**Applied:** {interaction_model} · "
        f"{'finite' if muon_scattering_finite_nuclei else 'point'} nuclei · "
        f"backend {backend} · plotting {', '.join(plot_species)} at {', '.join(plot_levels)}"
    )
    return (
        backend, database, density_model, interaction_model, low_energy_model,
        muon_scattering_finite_nuclei, plot_levels, plot_species, settings,
        theta_deg,
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
    ## 03 · Solve the 2D cascade

    One 1 TeV proton is injected at the top of the atmosphere and the cascade is
    saved at the three observation depths. Building the 2D matrices and solving
    takes a few minutes on a CPU.
    """)
    return


@app.cell
def _(
    MCEqRun,
    backend,
    config,
    database,
    density_model,
    interaction_model,
    low_energy_model,
    muon_scattering_finite_nuclei,
    np,
    resolve_kernel,
    solved_runs,
    theta_deg,
    proceed):
    _ = proceed
    physics_key = (
        database, interaction_model, low_energy_model, repr(density_model),
        theta_deg, muon_scattering_finite_nuclei, backend,
    )
    if physics_key in solved_runs:
        mceq_run, save_depths = solved_runs[physics_key]
    else:
        solved_runs.clear()  # one 2D run at a time: each holds several GB
        config.mceq_db_fname = database
        config.kernel_config = resolve_kernel(backend)
        config.debug_level = 0
        config.enable_em = False  # ELMFLG F F
        config.muon_scattering_finite_nuclei = muon_scattering_finite_nuclei

        mceq_run = MCEqRun(
            theta_deg=theta_deg,
            interaction_model=interaction_model,
            primary_model=None,
            density_model=density_model,
            low_energy_model=low_energy_model,
        )
        _atmosphere = mceq_run.density_model
        # 15 km, 5 km, ground; h2X(0) lands a hair beyond max_X, so nudge it inside
        save_depths = np.minimum(
            _atmosphere.h2X(np.array([15.0e5, 5.0e5, 0.0])),
            _atmosphere.max_X * (1.0 - 1e-9),
        )
        mceq_run.set_single_primary_particle(1.0e3, pdg_id=2212)
    print("depths [g/cm²]:", np.round(save_depths, 2))
    return mceq_run, physics_key, save_depths


@app.cell
def _(
    interaction_model,
    mceq_run,
    muon_scattering_finite_nuclei,
    physics_key,
    save_depths,
    solve_progress,
    solved_runs):
    if physics_key not in solved_runs:
        with solve_progress(
            f"{interaction_model} 2D · "
            f"{'finite' if muon_scattering_finite_nuclei else 'point'} nuclei"
        ) as progress:
            mceq_run.solve(int_grid=save_depths, progress=progress)
        solved_runs[physics_key] = (mceq_run, save_depths)
    n_steps = mceq_run.integration_path[0]
    f"solved: {n_steps} steps, {mceq_run._mceq_db.n_k} Legendre modes"
    return (n_steps,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 04 · Read out like CORSIKA

    The solution at each depth is a set of Legendre-mode amplitudes per energy
    bin. `convert_to_theta_space` turns them into an angular density
    $\mathrm{d}N/(\mathrm{d}E\,\theta\,\mathrm{d}\theta)$ on a fine θ grid. Integrating
    $\theta$ times that density over each 2° bin and multiplying by the energy-bin
    width gives particles per bin per primary, the CORSIKA normalization. Muons
    are summed over the three helicity states MCEq tracks, particles over
    antiparticles.

    The angle-integrated spectrum is the $l=0$ mode. The solid "reconstructed"
    curve integrates the angular density instead; the two differ by the
    reconstruction closure (≤ 1 % here).
    """)
    return


@app.cell
def _(mceq_run, n_steps, np, reference):
    _ = n_steps  # read out after the solve
    species_states = {
        "p": [(2212, 0)],
        "n": [(2112, 0)],
        "pi": [(211, 0), (-211, 0)],
        "mu": [(pdg, hel) for pdg in (13, -13) for hel in (-1, 0, 1)],
        "numu": [(14, 0), (-14, 0)],
        "nue": [(12, 0), (-12, 0)],
    }
    theta_edges = reference["theta_edges_deg"]  # 2° bins 0–90, plus a ≥ 90° column
    angle_e_bins = reference["angle_e_bin_edges"]
    e_bins = mceq_run.e_bins
    n_e = len(e_bins) - 1
    n_theta = len(theta_edges) - 1
    # MCEq energy bins holding the four CORSIKA angular bins (same edges)
    angle_e_index = [int(np.argmin(np.abs(e_bins[:-1] - lo))) for lo, _ in angle_e_bins]
    assert np.allclose(e_bins[angle_e_index], angle_e_bins[:, 0], rtol=1e-3)

    n_k = mceq_run._mceq_db.n_k
    modes = [snapshot.reshape(n_k, mceq_run.dim_states) for snapshot in mceq_run.grid_sol]
    n_depths = len(modes)

    def bin_integral(theta, integrand, lo, hi):
        """∫ integrand dθ over [lo, hi], trapezoid with interpolated end points."""
        inside = (theta > lo) & (theta < hi)
        x = np.concatenate(([lo], theta[inside], [hi]))
        y = np.concatenate(
            ([np.interp(lo, theta, integrand)], integrand[inside], [np.interp(hi, theta, integrand)])
        )
        return np.trapezoid(y, x)

    mceq_angular, mceq_spec_k0, mceq_closure = {}, {}, {}
    for _name, _states in species_states.items():
        density, spec_k0 = None, np.zeros((n_depths, n_e))
        for _pdg, _hel in _states:
            _start = mceq_run.pman.pdg2mceqidx[(_pdg, _hel)] * n_e
            for _d in range(n_depths):
                spec_k0[_d] += modes[_d][0, _start:_start + n_e]
            _, _, theta, _rho = mceq_run.convert_to_theta_space(
                modes, _pdg, _hel, theta_res=1200
            )
            density = np.array(_rho) if density is None else density + np.array(_rho)
        # angle-integrated reconstruction / l=0 mode
        _closure = np.trapezoid(density * theta, theta, axis=2)
        mceq_closure[_name] = _closure / np.where(spec_k0 != 0, spec_k0, np.inf)
        mceq_spec_k0[_name] = spec_k0  # dN/dE per primary [1/GeV]

        hist = np.zeros((n_depths, len(angle_e_index), n_theta + 1))
        for _d in range(n_depths):
            for _i, _ie in enumerate(angle_e_index):
                _integrand = density[_d, _ie] * theta
                for _it in range(n_theta):
                    _lo, _hi = np.radians(theta_edges[_it:_it + 2])
                    hist[_d, _i, _it] = bin_integral(theta, _integrand, _lo, _hi)
                _lo90 = np.radians(theta_edges[-1])
                if theta[-1] > _lo90:
                    hist[_d, _i, n_theta] = bin_integral(theta, _integrand, _lo90, theta[-1])
                hist[_d, _i] *= e_bins[_ie + 1] - e_bins[_ie]
        mceq_angular[_name] = hist
    print("species read out:", ", ".join(mceq_angular))
    return e_bins, mceq_angular, mceq_closure, mceq_spec_k0, theta_edges


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 05 · Angular distributions

    Particles per 2° bin per primary versus the angle to the shower axis, in four
    narrow energy bins, up to 60°. Lines: MCEq; points: CORSIKA with Poisson
    errors. The lower panels show CORSIKA / MCEq (±15 %).
    """)
    return


@app.cell
def _(Line2D, mceq_angular, np, plot_levels, plot_species, plt, reference, theta_edges):
    species_labels = {
        "p": r"$p$", "n": r"$n$", "pi": r"$\pi^\pm$", "mu": r"$\mu^\pm$",
        "numu": r"$\nu_\mu+\bar\nu_\mu$", "nue": r"$\nu_e+\bar\nu_e$",
    }
    level_names = ["15 km", "5 km", "ground"]
    level_depths = [124.14, 552.95, 1036.10]
    style = {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": "#263342",
        "axes.edgecolor": "#B3BAC2",
        "xtick.color": "#52606D",
        "ytick.color": "#52606D",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
    ratio_limits = (0.85, 1.15)

    def plot_angular():
        energy_colors = ["#263342", "#17857E", "#C46B30", "#7A5BA6"]
        theta_max = 60.0
        centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
        n_keep = int(np.searchsorted(theta_edges, theta_max))
        e_edges = reference["angle_e_bin_edges"]
        levels = [level_names.index(lv) for lv in plot_levels]
        n_rows, n_cols = len(levels), len(plot_species)
        with plt.rc_context(style):
            fig = plt.figure(figsize=(2.15 * n_cols + 1.4, 3.3 * n_rows + 0.8), layout="constrained")
            grid = fig.add_gridspec(2 * n_rows, n_cols, height_ratios=[2.2, 1.0] * n_rows)
            for row, li in enumerate(levels):
                top = None
                for col, sp in enumerate(plot_species):
                    ax = fig.add_subplot(grid[2 * row, col], sharey=top)
                    top = top or ax
                    axr = fig.add_subplot(grid[2 * row + 1, col], sharex=ax)
                    model = mceq_angular[sp][li]
                    c7 = reference[f"ang_{sp}"][li]
                    c7_err = reference[f"ang_{sp}_err"][li]
                    for ie, color in enumerate(energy_colors):
                        ax.stairs(model[ie, :n_keep], theta_edges[: n_keep + 1], baseline=None, color=color, lw=1.4)
                        filled = c7_err[ie, :n_keep] > 0
                        ax.errorbar(
                            centers[:n_keep][filled], c7[ie, :n_keep][filled],
                            yerr=c7_err[ie, :n_keep][filled], fmt="o", ms=2.6,
                            elinewidth=0.8, color=color,
                        )
                        with np.errstate(invalid="ignore", divide="ignore"):
                            positive = np.where(model[ie, :n_keep] > 0, model[ie, :n_keep], np.nan)
                            ratio = c7[ie, :n_keep] / positive
                            ratio_err = c7_err[ie, :n_keep] / positive
                        axr.errorbar(
                            centers[:n_keep][filled], ratio[filled], yerr=ratio_err[filled],
                            fmt="o", ms=2.2, elinewidth=0.7, color=color,
                        )
                    ax.set_yscale("log")
                    ax.set_xlim(0, theta_max)
                    ax.tick_params(labelbottom=False)
                    ax.grid(axis="y", which="major", color="#E6E9ED", lw=0.6)
                    axr.axhline(1.0, color="#8A949E", lw=0.8)
                    axr.set_ylim(*ratio_limits)
                    axr.set_yticks([0.9, 1.0, 1.1])
                    if row == n_rows - 1:
                        axr.set_xlabel(r"$\theta$ [deg]")
                    if col == 0:
                        ax.set_ylabel(f"{level_names[li]} · {level_depths[li]:.0f} g/cm²\nN / (bin · primary)")
                        axr.set_ylabel("C7 / MCEq", fontsize=9)
                    else:
                        ax.tick_params(labelleft=False)
                        axr.tick_params(labelleft=False)
                    if row == 0:
                        ax.set_title(species_labels[sp], fontsize=12)
                ymax = top.get_ylim()[1]
                top.set_ylim(ymax * 10 ** -4.5, ymax)
            handles = [
                Line2D([], [], color=c, lw=1.4, marker="o", ms=3)
                for c in energy_colors
            ] + [Line2D([], [], color="k", lw=1.4), Line2D([], [], color="k", lw=0, marker="o", ms=3)]
            labels = [f"{lo:g}–{hi:g} GeV" for lo, hi in e_edges] + ["MCEq 2D", "CORSIKA 7"]
            fig.legend(handles, labels, loc="outside upper center", ncol=6, frameon=False, fontsize=9)
        return fig

    fig_angular = plot_angular()
    fig_angular
    return level_names, ratio_limits, species_labels, style


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 06 · Angle-integrated energy spectra

    $E\,\mathrm{d}N/\mathrm{d}E$ per primary at the three levels. Solid: MCEq
    reconstructed (angular density integrated over all angles); dashed: the
    $l=0$ mode; points: CORSIKA. CORSIKA hadrons include all angles, leptons
    θ < 90°. Shown up to 150 GeV (see caveats).
    """)
    return


@app.cell
def _(
    Line2D,
    e_bins,
    level_names,
    mceq_closure,
    mceq_spec_k0,
    np,
    plot_levels,
    plot_species,
    plt,
    ratio_limits,
    reference,
    species_labels,
    style):
    species_colors = {
        "p": "#4477AA", "n": "#EE6677", "pi": "#228833",
        "mu": "#CCBB44", "numu": "#AA3377", "nue": "#66CCEE",
    }
    e_edges = reference["e_edges"]
    # CORSIKA spectra use the MCEq grid edges; align by edge value, not index
    spectrum_index = np.searchsorted(e_bins, e_edges[:-1] * (1 - 1e-9))
    assert np.allclose(e_bins[spectrum_index], e_edges[:-1])
    assert np.allclose(e_bins[spectrum_index + 1], e_edges[1:])

    def plot_spectra():
        x_range = (0.1, 150.0)
        widths = np.diff(e_edges)
        centers = np.sqrt(e_edges[:-1] * e_edges[1:])
        shown = (centers > x_range[0]) & (centers < x_range[1])
        levels = [level_names.index(lv) for lv in plot_levels]
        with plt.rc_context(style):
            fig, axes = plt.subplots(
                2, len(levels), figsize=(4.2 * len(levels) + 0.6, 5.6), sharex=True,
                sharey="row", height_ratios=[2.3, 1.0], layout="constrained", squeeze=False,
            )
            for col, li in enumerate(levels):
                ax, axr = axes[0, col], axes[1, col]
                for sp in plot_species:
                    color = species_colors[sp]
                    k0 = centers * mceq_spec_k0[sp][li, spectrum_index]
                    recon = k0 * mceq_closure[sp][li, spectrum_index]
                    c7 = centers * reference[f"spec_{sp}"][li] / widths
                    c7_err = centers * reference[f"spec_{sp}_err"][li] / widths
                    filled = shown & (c7_err > 0)
                    ax.plot(centers[shown], k0[shown], color=color, lw=0.9, ls="--", alpha=0.8)
                    ax.plot(centers[shown], recon[shown], color=color, lw=1.5)
                    ax.errorbar(
                        centers[filled], c7[filled], yerr=c7_err[filled],
                        fmt="o", ms=2.6, elinewidth=0.8, color=color,
                    )
                    with np.errstate(invalid="ignore", divide="ignore"):
                        positive = np.where(recon > 0, recon, np.nan)
                    axr.errorbar(
                        centers[filled], (c7 / positive)[filled], yerr=(c7_err / positive)[filled],
                        fmt="o", ms=2.2, elinewidth=0.7, color=color,
                    )
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(*x_range)
                ax.set_title(level_names[li], fontsize=12)
                ax.grid(axis="y", which="major", color="#E6E9ED", lw=0.6)
                axr.axhline(1.0, color="#8A949E", lw=0.8)
                axr.set_ylim(*ratio_limits)
                axr.set_yticks([0.9, 1.0, 1.1])
                axr.set_xlabel(r"Kinetic energy $E$ [GeV]")
            axes[0, 0].set_ylabel(r"$E\,\mathrm{d}N/\mathrm{d}E$ per primary")
            axes[1, 0].set_ylabel("CORSIKA / MCEq")
            ymax = axes[0, 0].get_ylim()[1]
            axes[0, 0].set_ylim(ymax * 1e-5, ymax)
            handles = [Line2D([], [], color=species_colors[sp], lw=1.5) for sp in plot_species] + [
                Line2D([], [], color="k", lw=1.5),
                Line2D([], [], color="k", lw=0.9, ls="--"),
                Line2D([], [], color="k", lw=0, marker="o", ms=3),
            ]
            labels = [species_labels[sp] for sp in plot_species] + [
                "MCEq 2D (reconstructed)", r"MCEq $l=0$ mode", "CORSIKA 7",
            ]
            fig.legend(handles, labels, loc="outside upper center", ncol=5, frameon=False, fontsize=9)
        return fig

    fig_spectra = plot_spectra()
    fig_spectra
    return (spectrum_index,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 07 · Agreement in numbers

    CORSIKA / MCEq for the particle numbers summed over energy bands (MCEq:
    reconstructed spectrum), at the three levels. The ± is the CORSIKA Poisson
    error.
    """)
    return


@app.cell
def _(e_bins, level_names, mceq_closure, mceq_spec_k0, np, reference, spectrum_index):
    def agreement_table():
        # band edges are grid edges: 10^-1, 10^-0.9, 10^-0.7, 10^-0.3, 1, 10, 100 GeV
        bands = [(-1.0, -0.9), (-0.9, -0.7), (-0.7, -0.3), (-0.3, 0.0), (0.0, 1.0), (1.0, 2.0)]
        e_edges = reference["e_edges"]
        log_lo, log_hi = np.log10(e_edges[:-1]), np.log10(e_edges[1:])
        widths = np.diff(e_bins)[spectrum_index]
        n_showers = float(reference["n_showers"])
        lines = [
            f"{'species':7s} {'level':7s}"
            + "".join(f"{f'{10**lo:.3g}–{10**hi:.3g} GeV':>16s}" for lo, hi in bands)
        ]
        for sp in ("p", "n", "pi", "mu", "numu", "nue"):
            model = mceq_spec_k0[sp][:, spectrum_index] * mceq_closure[sp][:, spectrum_index] * widths
            for li, level in enumerate(level_names):
                cells = []
                for lo, hi in bands:
                    sel = (log_lo > lo - 1e-6) & (log_hi < hi + 1e-6)
                    c7 = reference[f"spec_{sp}"][li][sel].sum()
                    counts = ((reference[f"spec_{sp}_err"][li][sel] * n_showers) ** 2).sum()
                    m = model[li][sel].sum()
                    ratio = c7 / m
                    cells.append(f"{ratio:.3f} ± {ratio / np.sqrt(counts):.3f}" if counts > 0 else "–")
                lines.append(f"{sp:7s} {level:7s}" + "".join(f"{c:>16s}" for c in cells))
        return "\n".join(lines)

    print(agreement_table())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 08 · What to expect

    With point nuclei, MCEq and CORSIKA agree to 1 % from 0.2 GeV to 100 GeV for
    p, n, π±, μ± and ν_μ at every level (ν_μ +0.5 % above 10 GeV), and the
    angular shapes agree within the CORSIKA errors up to 60°. Known differences:

    - **Lowest bin, 0.1–0.126 GeV**: μ and ν_μ are 1–3 % low in CORSIKA, π± 3–5 %
      (with 2–4 % statistical errors), p +1–3 %. This bin sits at the
      cutoff (`ECUTS` 0.1 GeV) and the bottom of the MCEq grid.
    - **ν_e** (not in the paper figures): CORSIKA is ≈ 1 % low above 0.5 GeV and
      4–6 % low in the lowest bin.
    - **2 GeV at large angles**: beyond ≈ 25° (ground) to 40° (15 km) the 2 GeV
      MCEq distribution is 10⁻⁴–10⁻⁵ of its forward value and oscillates
      around zero. That is the floor of the Legendre reconstruction (90 modes),
      not physics; CORSIKA stays smooth there.
    - **Muon scattering**: with finite nuclei (MCEq's default), ground muons of
      0.3–2 GeV are tilted, CORSIKA / MCEq ≈ 0.98 near the axis to ≈ 1.03 at
      20–40°. Point nuclei remove most of it.
    - **The primary's own bin**: CORSIKA's `ERANGE` is total energy, so the
      surviving primary has 999.06 GeV kinetic energy and sits in the
      794–1000 GeV bin; MCEq's 1 TeV kinetic injection straddles the 1000 GeV
      edge. The 100–1000 GeV band is therefore not comparable bin by bin and
      is cut from the spectra.
    """)
    return


@app.cell(hide_code=True)
def export_notebook(settings):
    from MCEq.notebook import download_jupyter

    download_jupyter(filename="corsika_p1tev_vertical.ipynb", configuration=settings)
    return


if __name__ == "__main__":
    app.run()
