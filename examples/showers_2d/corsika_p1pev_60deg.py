# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo==0.24.0", "nbformat", "MCEq>=2.0.0", "matplotlib>=3.10", "numpy"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="MCEq vs CORSIKA 7: proton 1 PeV at 60°")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MCEq vs CORSIKA 7: proton 1 PeV at 60°

    Hadron, muon and neutrino spectra of a 1 PeV proton shower at seven slant
    depths (100 – 2000 g/cm²), MCEq 2D against 5000 CORSIKA 7 showers; above
    1 GeV the two agree to about 2 % down to 1000 g/cm².

    This is the comparison of the MCEq paper figures
    `fig:corsika_hadron_spectra` and `fig:corsika_integral_ratio`. The spectra
    are angle-integrated, yet MCEq runs in **2D** (Legendre angular modes,
    read out at $l = 0$): the 1D straight-ahead approximation moves every
    particle along the shower axis, which is fine above a few GeV but
    overestimates sub-GeV hadrons at depth, by up to ~75 % for 0.3 GeV neutrons at
    2000 g/cm² (section 05).

    **Download size.** The 2D run needs `mceq_base_air_2d_legendre_v2.h5`
    (456 MB) and the SIBYLL 2.3e 2D tables
    `mceq_model_SIBYLL23E_air_2d_legendre_v2.h5` (968 MB); MCEq fetches both
    on first use. The optional 1D comparison uses the default
    `mceq_base_air_1d_v2.h5` (76 MB).
    """)
    return


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from MCEq import config
    from MCEq.core import MCEqRun
    from MCEq.notebook import apply_kernel_and_grid, solve_progress

    return (
        MCEqRun,
        Path,
        apply_kernel_and_grid,
        config,
        json,
        mo,
        np,
        plt,
        solve_progress,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 01 · The CORSIKA run

    One shard of the production (2 showers); the other 2499 shards differ only
    in `RUNNR` (100000 + *I*) and `SEED` ((*I* + 1)·1000 + 1, 2, 3). Binary:
    CORSIKA 7.8050 built with SIBYLL 2.3e, FLUKA 2025.1, `THIN`, `NEUTRINO`,
    `CHARM` and the flat `HORIZONTAL` detector, patched with a β-exact PRANGE
    fix and isospin-corrected K*± decays. Details and the scoring procedure:
    `examples/data/corsika/p1pev_60deg/README.md`.
    """)
    return


@app.cell
def _(Path, json, np):
    def find_case_dir():
        """The reference data next to the examples, from a script, marimo or Jupyter."""
        try:
            here = Path(__file__).resolve().parent
        except NameError:  # Jupyter: the notebook directory is the working directory
            here = Path.cwd()
        for base in (here, Path.cwd()):
            candidate = base.parent / "data" / "corsika" / "p1pev_60deg"
            if candidate.is_dir():
                return candidate
        raise FileNotFoundError("examples/data/corsika/p1pev_60deg not found")

    case_dir = find_case_dir()
    steering_card = (case_dir / "steering.inp").read_text()
    reference = dict(np.load(case_dir / "c7_reference.npz"))
    reference_meta = json.loads(str(reference.pop("meta")))
    print(steering_card)
    return reference, reference_meta


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    | Card | Meaning | MCEq counterpart |
    |---|---|---|
    | `PRMPAR 14`, `ERANGE 1e6 1e6` | proton, 1 PeV | `set_single_primary_particle(1e6, pdg_id=2212)`, `primary_model=None` |
    | `THETAP 60 60` | zenith 60° (`PHIP` irrelevant without B field) | `theta_deg=60` |
    | `OBSLEV` × 7 | horizontal planes at slant 100 … 2000 g/cm² | `solve(int_grid=X, grid_var="X")` at `h2X(altitude)` |
    | no `ATMOD` | model 1: US standard (Linsley) | `CorsikaAtmosphere("USStd", None)` |
    | no `CURVED` (build option) | flat atmosphere | `config.r_E = 1e15` m |
    | no `HILOW` | SIBYLL above 80 GeV, FLUKA below | `low_energy_model="FLUKA20251"`, `he_le_transition=80` |
    | `ELMFLG F F` | no EM cascade (no photo-production) | `config.enable_em = False` (default) |
    | `MAGNET 0.00001 0.00001` | B ≈ 0 (azimuthal symmetry for the re-scoring) | MCEq has no field |
    | `MUMULT T` | Molière muon scattering, point nuclei | `config.muon_scattering_finite_nuclei = False` |
    | `ECUTS 0.3 0.3 …` | hadrons and muons dropped below 0.3 GeV | compare from 0.3 GeV; MCEq grid goes on to 0.1 GeV |
    | `THIN 1.E-6 1.0 0.` | thinning 10⁻⁶, **no radial thinning** | (MCEq is a mean, no sampling) |
    | `HADFLG 0 0 0 0 0 2` | CORSIKA defaults for nuclear fragmentation | irrelevant for a proton |

    CORSIKA scores on horizontal planes, MCEq on planes perpendicular to the
    shower axis. The reference therefore re-scores the CORSIKA particles onto the
    perpendicular plane through the core (a 30 g/cm² strip weighted by its
    azimuthal acceptance), so both sides count the same quantity: all particles
    crossing the plane at the given slant depth, any direction.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 02 · MCEq setup

    The defaults reproduce the paper figures. **Muon scattering** chooses the
    nuclear form factor of the muon multiple-scattering kernel: MCEq's default
    uses finite nuclei; CORSIKA's Molière treatment uses point nuclei, so the
    comparison does too (the difference is ≤ 0.2 % on these spectra, largest for
    sub-GeV muons at 2000 g/cm²). **Also solve 1D** adds a 1D run on the
    default 1D database (a few seconds) for section 05. The backend only
    changes speed.
    """)
    return


@app.cell(hide_code=True)
def configuration_defaults():
    defaults = {
        "database": "mceq_base_air_2d_legendre_v2.h5",
        "database_1d": "mceq_base_air_1d_v2.h5",
        "interaction_model": "SIBYLL23E",
        "low_energy_model": "FLUKA20251",
        "he_le_transition": 80.0,
        "he_le_trwidth": 0.3,
        "theta_deg": 60.0,
        "point_nuclei": True,
        "compare_1d": True,
        "backend": "auto",
    }
    return (defaults,)


@app.cell(hide_code=True)
def configuration_controls(defaults, mo):
    from MCEq.notebook import backend_options

    backends = backend_options()
    controls = mo.md("""
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:1rem 2rem">
    <div>

    <strong>Muon scattering</strong>

    {point_nuclei}

    </div><div>

    <strong>1D comparison</strong>

    {compare_1d}

    </div><div>

    <strong>Solver</strong>

    {backend}

    </div></div>
    """).batch(
        point_nuclei=mo.ui.dropdown(
            {"Point nuclei (CORSIKA Molière)": True, "Finite nuclei (MCEq default)": False},
            value="Point nuclei (CORSIKA Molière)" if defaults["point_nuclei"]
            else "Finite nuclei (MCEq default)",
            label="Nuclear form factor", full_width=True,
        ),
        compare_1d=mo.ui.checkbox(value=defaults["compare_1d"], label="Also solve 1D"),
        backend=mo.ui.dropdown(
            backends,
            value=next((k for k, v in backends.items() if v == defaults["backend"]), "Automatic"),
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
    database_1d = settings["database_1d"]
    interaction_model = settings["interaction_model"]
    low_energy_model = settings["low_energy_model"]
    he_le_transition = settings["he_le_transition"]
    he_le_trwidth = settings["he_le_trwidth"]
    theta_deg = settings["theta_deg"]
    point_nuclei = settings["point_nuclei"]
    compare_1d = settings["compare_1d"]
    backend = settings["backend"]
    mo.md(
        f"**Applied:** {interaction_model} + {low_energy_model} below "
        f"{he_le_transition:g} GeV · {theta_deg:g}° · "
        f"{'point' if point_nuclei else 'finite'} nuclei · "
        f"1D comparison {'on' if compare_1d else 'off'}"
    )
    return (
        backend,
        compare_1d,
        database,
        database_1d,
        he_le_transition,
        he_le_trwidth,
        interaction_model,
        low_energy_model,
        point_nuclei,
        settings,
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

    Flat Earth and CORSIKA's atmosphere 1 are set before the density model is
    built (the geometry reads `config.r_E` at construction). The seven depths
    are the slant depths of the `OBSLEV` altitudes along the 60° axis.
    (The paper's script raised the atmosphere by 12 km only to also read the
    solution on CORSIKA's horizontal planes, which needs axial depths beyond
    the ground; the on-axis spectra here are identical without it.)

    Expect a few minutes: building the 2D operators (90 Legendre modes) and
    about 1600 integration steps take 2–10 min on a multi-core CPU with MKL.
    """)
    return


@app.cell
def _(
    MCEqRun,
    apply_kernel_and_grid,
    backend,
    config,
    database,
    he_le_transition,
    he_le_trwidth,
    interaction_model,
    low_energy_model,
    np,
    point_nuclei,
    reference,
    theta_deg,
    proceed):
    _ = proceed
    from MCEq.geometry.density_profiles import CorsikaAtmosphere

    apply_kernel_and_grid(backend, 0.1, 1.0e11)  # MCEq default grid
    config.r_E = 1.0e15  # m: flat atmosphere, like CORSIKA without CURVED
    config.muon_scattering_finite_nuclei = not point_nuclei
    config.mceq_db_fname = database

    atmosphere = CorsikaAtmosphere("USStd", None)  # CORSIKA ATMOD 1
    atmosphere.set_theta(theta_deg)
    slant_depths = np.array([float(atmosphere.h2X(h)) for h in reference["alt_cm"]])

    mceq_2d = MCEqRun(
        interaction_model=interaction_model,
        primary_model=None,
        theta_deg=theta_deg,
        density_model=atmosphere,
        low_energy_model=low_energy_model,
        he_le_transition=he_le_transition,
        he_le_trwidth=he_le_trwidth,
    )
    mceq_2d.set_single_primary_particle(1.0e6, pdg_id=2212)
    print("slant depths of the OBSLEV planes [g/cm²]:", np.round(slant_depths, 2))
    return mceq_2d, slant_depths


@app.cell
def _(mceq_2d, slant_depths, solve_progress):
    with solve_progress("2D cascade · p 1 PeV · 60°") as progress:
        mceq_2d.solve(int_grid=slant_depths, grid_var="X", progress=progress)
    f"{mceq_2d.integration_path[0]} steps, {mceq_2d._mceq_db.n_k} Legendre modes"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 04 · MCEq vs CORSIKA

    `get_solution(..., grid_idx=i, integrate=True)` returns particles per
    energy bin at depth *i*; on a 2D run it reads the $l = 0$ Legendre row, the
    angle-integrated count, which is what the reference holds (per shower,
    same bins). Spectra are plotted as $E\,\mathrm{d}N/\mathrm{d}E$ = count / Δln E.
    """)
    return


@app.cell
def _(mceq_2d, np, reference):
    species = {
        "neutron": (r"$n + \bar n$", ("n0", "nbar0")),
        "proton": (r"$p + \bar p$", ("p+", "pbar-")),
        "pi_charged": (r"$\pi^+ + \pi^-$", ("pi+", "pi-")),
        "K_charged": (r"$K^+ + K^-$", ("K+", "K-")),
        "muon": (r"$\mu^+ + \mu^-$", ("mu+", "mu-")),
        "numu": (r"$\nu_\mu + \bar\nu_\mu$", ("numu", "antinumu")),
        "nue": (r"$\nu_e + \bar\nu_e$", ("nue", "antinue")),
    }

    def level_counts(run, names, n_levels):
        """Particles per kinetic-energy bin and primary, (level, bin)."""
        return np.array([
            sum(run.get_solution(name, grid_idx=i, integrate=True) for name in names)
            for i in range(n_levels)
        ])

    n_levels = len(reference["slant_gcm2"])
    mceq_counts = {key: level_counts(mceq_2d, names, n_levels) for key, (_, names) in species.items()}
    assert np.allclose(mceq_2d.e_bins, reference["e_edges"])
    return level_counts, mceq_counts, n_levels, species


@app.cell
def _(mceq_2d, mceq_counts, np, plt, reference, species):
    energy = mceq_2d.e_grid
    dlne = np.log(reference["e_edges"][1:] / reference["e_edges"][:-1])
    shown = (energy >= 0.3) & (energy <= 2.0e5)  # from ECUTS to well below the primary bin
    depth_colors = plt.get_cmap("viridis")(np.linspace(0.0, 0.88, len(reference["slant_gcm2"])))

    with plt.rc_context({
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
    }):
        fig_spectra = plt.figure(figsize=(11.5, 15.0), layout="constrained")
        outer = fig_spectra.add_gridspec(4, 2)
        for k, (key, (label, _)) in enumerate(species.items()):
            inner = outer[divmod(k, 2)].subgridspec(2, 1, height_ratios=[3, 1.2], hspace=0.05)
            ax = fig_spectra.add_subplot(inner[0])
            axr = fig_spectra.add_subplot(inner[1], sharex=ax)
            ymax = 0.0
            for i, (depth, color) in enumerate(zip(reference["slant_gcm2"], depth_colors)):
                c7 = reference[f"spec_{key}"][i]
                c7_err = reference[f"spec_{key}_err"][i]
                model = mceq_counts[key][i]
                ax.plot(energy[shown], (model / dlne)[shown], color=color, lw=1.6)
                points = shown & (c7 > 0)
                ax.errorbar(
                    energy[points], (c7 / dlne)[points], yerr=(c7_err / dlne)[points],
                    color=color, ls="none", marker="o", ms=2.8, elinewidth=0.8,
                )
                if points.any():
                    ymax = max(ymax, float(np.max((c7 / dlne)[points])))
                ok = points & (model > 0)
                axr.errorbar(
                    energy[ok], (c7 / model)[ok], yerr=(c7_err / model)[ok],
                    color=color, ls="none", marker="o", ms=2.2, elinewidth=0.7,
                )
            ax.set(xscale="log", yscale="log", ylim=(ymax / 1e4, ymax * 3))
            ax.tick_params(labelbottom=False)
            ax.set_title(label, loc="left", fontsize=13)
            ax.grid(axis="y", which="major", color="#E6E9ED", linewidth=0.7)
            axr.axhline(1.0, color="#52606D", lw=0.8)
            axr.set(xscale="log", xlim=(0.3, 2.0e5), ylim=(0.7, 1.3))
            axr.set_xlabel(r"Kinetic energy $E$ [GeV]")
            axr.set_ylabel("C7 / MCEq", fontsize=9)
            if k % 2 == 0:
                ax.set_ylabel(r"$E\,\mathrm{d}N/\mathrm{d}E$ per shower")

        legend_ax = fig_spectra.add_subplot(outer[3, 1])
        legend_ax.axis("off")
        handles = [
            plt.Line2D([], [], color="#263342", lw=1.6, label="MCEq 2D ($l = 0$)"),
            plt.Line2D([], [], color="#263342", ls="none", marker="o", ms=4,
                       label=f"CORSIKA 7 ({int(reference['n_showers'])} showers)"),
        ] + [
            plt.Line2D([], [], color=color, lw=4, label=f"X = {depth:g} g/cm²")
            for depth, color in zip(reference["slant_gcm2"], depth_colors)
        ]
        legend_ax.legend(handles=handles, loc="center", frameon=False, fontsize=11)
        fig_spectra.suptitle(
            "p, 1 PeV, 60°: SIBYLL 2.3e + FLUKA, EM off; slant depth along the axis",
            fontsize=11, color="#52606D",
        )

    fig_spectra
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Integrated above 1 GeV, as in the paper's Fig. 19. Error bars are the
    shower-to-shower standard error of CORSIKA's mean; points beyond ±15 % are
    drawn as arrows with their value.
    """)
    return


@app.cell
def _(mceq_2d, mceq_counts, np, plt, reference, species):
    above_1gev = mceq_2d.e_grid >= 1.0
    integral_ratio = {}
    for _key in species:
        _c7 = reference[f"spec_{_key}"][:, above_1gev].sum(axis=1)
        _model = mceq_counts[_key][:, above_1gev].sum(axis=1)
        integral_ratio[_key] = (_c7 / _model, reference[f"int_gt1_{_key}_err"] / _model)

    groups = (
        [("neutron", "#0077BB", "o"), ("proton", "#CC3311", "s"),
         ("pi_charged", "#009988", "^"), ("K_charged", "#EE7733", "D")],
        [("muon", "#33BBEE", "v"), ("numu", "#EE3377", "P"), ("nue", "#263342", "X")],
    )
    limits = (0.85, 1.15)
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
        fig_integral, axes_integral = plt.subplots(
            1, 2, figsize=(11.5, 4.4), sharex=True, sharey=True, layout="constrained"
        )
        for _ax, _group in zip(axes_integral, groups):
            for _j, (_key, _color, _marker) in enumerate(_group):
                _xs = reference["slant_gcm2"] * 10 ** (0.018 * (_j - (len(_group) - 1) / 2))
                _values, _errors = integral_ratio[_key]
                _ax.errorbar(_xs, _values, yerr=_errors, color=_color, marker=_marker, ms=5,
                            lw=1.2, elinewidth=0.9, label=species[_key][0])
                for _x, _v in zip(_xs, _values):
                    if not limits[0] <= _v <= limits[1]:
                        _low = _v < limits[0]
                        _y = limits[0] + 0.012 if _low else limits[1] - 0.012
                        _ax.plot(_x, _y, marker="v" if _low else "^", ms=6, color=_color)
                        _ax.annotate(f"{_v:.2f}", (_x, _y), xytext=(5, 0), textcoords="offset points",
                                    fontsize=8, color=_color, va="center")
            _ax.axhline(1.0, color="#52606D", lw=0.8)
            _ax.set(xscale="log", xlim=(85, 2400), ylim=limits, yticks=[0.9, 1.0, 1.1])
            _ax.set_xlabel(r"Slant depth $X$ [g cm$^{-2}$]")
            _ax.grid(axis="y", color="#E6E9ED", linewidth=0.7)
            _ax.legend(loc="upper center", ncol=len(_group), frameon=False, fontsize=10)
        axes_integral[0].set_ylabel(r"CORSIKA 7 / MCEq,  $E_\mathrm{kin} > 1$ GeV")

    fig_integral
    return (integral_ratio,)


@app.cell
def _(integral_ratio, reference, species):
    print("CORSIKA 7 / MCEq 2D, E_kin > 1 GeV  (± shower-to-shower SEM)")
    print(f"{'X [g/cm²]':>12s}" + "".join(f"{x:>14g}" for x in reference["slant_gcm2"]))
    for _key in species:
        _values, _errors = integral_ratio[_key]
        print(f"{_key:>12s}" + "".join(f"{v:>8.3f}±{e:.3f}" for v, e in zip(_values, _errors)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reading the comparison.** Above 1 GeV every species agrees within
    1–3 % from 100 to 1000 g/cm², with a depth-independent offset of about
    −2 % (CORSIKA below MCEq) common to all species. At 2000 g/cm² (300 m
    altitude, a few hadrons per shower above 1 GeV) the hadrons are statistics-limited:
    n 0.89 ± 0.04, p 0.95 ± 0.04, π± 1.07 ± 0.06, K± 0.84 ± 0.17; the
    leptons stay at 0.98. Sub-GeV (0.3–1 GeV) hadrons
    agree within about 5 % down to 1000 g/cm². The last bins below the primary
    energy are not shown: MCEq spreads the injected 1 PeV proton over three
    energy bins (moment-matched deposit), so bin-by-bin ratios there test the
    deposit, not the cascade.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 05 · Why 2D: the straight-ahead approximation

    The 1D cascade equation assumes all secondaries move along the shower
    axis, so a particle at slant depth $X$ has traversed exactly $X$ g/cm².
    Low-energy secondaries are produced at angles of order
    $p_\perp / p \sim 0.3\,\text{GeV} / E$ and are scattered further, so
    a sub-GeV hadron crossing the plane at depth $X$ has typically
    travelled a path $X/\cos\theta$ that is markedly longer: more interaction and
    decay lengths, more ionization loss. 1D ignores this path-length excess and keeps too
    many sub-GeV hadrons (and the muons and neutrinos they feed) at depth;
    above a few GeV the angles are small and 1D and 2D converge. The 2D run
    carries the angular distribution in Legendre modes and its $l = 0$ row is
    the properly angle-integrated count. Below, a 1D solve with the same
    models, atmosphere and depths (`mceq_base_air_1d_v2.h5`) divided by the 2D
    result.
    """)
    return


@app.cell
def _(
    MCEqRun,
    compare_1d,
    config,
    database_1d,
    he_le_transition,
    he_le_trwidth,
    interaction_model,
    level_counts,
    low_energy_model,
    mceq_2d,
    mceq_counts,
    n_levels,
    slant_depths,
    solve_progress,
    species,
    theta_deg,
):
    if compare_1d:
        config.mceq_db_fname = database_1d
        mceq_1d = MCEqRun(
            interaction_model=interaction_model,
            primary_model=None,
            theta_deg=theta_deg,
            density_model=mceq_2d.density_model,  # the same flat USStd atmosphere
            low_energy_model=low_energy_model,
            he_le_transition=he_le_transition,
            he_le_trwidth=he_le_trwidth,
        )
        mceq_1d.set_single_primary_particle(1.0e6, pdg_id=2212)
        with solve_progress("1D cascade · p 1 PeV · 60°") as progress_1d:
            mceq_1d.solve(int_grid=slant_depths, grid_var="X", progress=progress_1d)
        counts_1d = {
            key: level_counts(mceq_1d, names, n_levels) for key, (_, names) in species.items()
        }
        _sub_gev = (mceq_2d.e_grid >= 0.3) & (mceq_2d.e_grid < 1.0)
        for _key in ("neutron", "proton", "pi_charged", "muon"):
            _ratio = counts_1d[_key][:, _sub_gev].sum(axis=1) / mceq_counts[_key][:, _sub_gev].sum(axis=1)
            print(f"1D / 2D, 0.3-1 GeV, {_key:>10s}: " + " ".join(f"{r:.2f}" for r in _ratio))
    else:
        counts_1d = None
    "1D solved" if compare_1d else "1D comparison off"
    return (counts_1d,)


@app.cell
def _(counts_1d, mceq_2d, mceq_counts, np, plt, reference, species):
    if counts_1d is None:
        fig_1d = None
    else:
        _energy = mceq_2d.e_grid
        _shown = (_energy >= 0.3) & (_energy <= 1.0e4)
        _colors = plt.get_cmap("viridis")(np.linspace(0.0, 0.88, len(reference["slant_gcm2"])))
        with plt.rc_context({
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
        }):
            fig_1d, _axes = plt.subplots(
                2, 4, figsize=(11.5, 5.6), sharex=True, sharey=True, layout="constrained"
            )
            for _ax, (_key, (_label, _)) in zip(_axes.flat, species.items()):
                for _i, (_depth, _color) in enumerate(zip(reference["slant_gcm2"], _colors)):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        _ratio = counts_1d[_key][_i] / mceq_counts[_key][_i]
                    _ax.plot(_energy[_shown], _ratio[_shown], color=_color, lw=1.5,
                             label=f"{_depth:g} g/cm²")
                _ax.axhline(1.0, color="#52606D", lw=0.8)
                _ax.set_title(_label, loc="left", fontsize=12)
                _ax.set(xscale="log", ylim=(0.8, 2.0))
                _ax.grid(axis="y", color="#E6E9ED", linewidth=0.7)
            for _ax in _axes[1]:
                _ax.set_xlabel(r"Kinetic energy $E$ [GeV]")
            for _ax in _axes[:, 0]:
                _ax.set_ylabel("MCEq 1D / 2D")
            _axes.flat[-1].axis("off")
            _axes.flat[-1].legend(*_axes.flat[0].get_legend_handles_labels(),
                                  loc="center", frameon=False, title="slant depth")
    fig_1d
    return


@app.cell(hide_code=True)
def export_notebook(settings):
    from MCEq.notebook import download_jupyter

    download_jupyter(filename="corsika_p1pev_60deg.ipynb", configuration=settings)
    return


if __name__ == "__main__":
    app.run()
