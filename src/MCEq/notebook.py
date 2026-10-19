"""In-memory Jupyter downloads and notebook conveniences for the examples.

Marimo's private conversion API is confined to this module and checked against
the gallery's version. Importing MCEq does not require marimo.
"""

import ast
import re
from contextlib import ExitStack, contextmanager


def jupyter_bytes(app, configuration=None):
    """Export current cell source without running the cascade or spawning a process."""
    import nbformat
    from marimo._convert.ipynb import convert_from_ir_to_ipynb

    notebook = nbformat.reads(
        convert_from_ir_to_ipynb(app, sort_mode="top-down"), as_version=4
    )
    for cell in notebook.cells:
        if (
            configuration is None
            and cell.metadata.get("marimo", {}).get("name") == "configuration_defaults"
        ):
            assignment = ast.parse(cell.source).body[0]
            configuration = ast.literal_eval(assignment.value)
    cells = []
    for cell in notebook.cells:
        metadata = cell.metadata.pop("marimo", {})
        if metadata.get("name") in {
            "export_notebook",
            "configuration_defaults",
            "configuration_controls",
            "configuration_summary",
            "run_gate",
            "run_stop",
        }:
            continue
        if metadata.get("name") == "selected_configuration":
            settings = configuration
            if settings is None:
                tree = ast.parse(cell.source)
                assignment = next(
                    node
                    for node in tree.body
                    if isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Name)
                        and target.id == "settings"
                        for target in node.targets
                    )
                )
                settings = ast.literal_eval(assignment.value)
            cell.source = configuration_source(settings)
            cell.metadata = {}  # The plain configuration must be visible in Jupyter.
        if cell.cell_type == "code":
            source = "\n".join(
                line
                for line in cell.source.splitlines()
                if line != "import marimo as mo"
                and line.strip() != "_ = proceed"
            )
            # The gate token is a marimo-only dependency; drop it from
            # signatures so the exported notebook stands alone.
            source = re.sub(r",\s*proceed(?=\s*\))", "", source)
            source = re.sub(r"\(\s*proceed(?=\s*\))", "(", source)
            source = re.sub(r"[ \t]*proceed(?=[ \t]*\))", "", source)
            cell.source = source
            if any(
                isinstance(node, ast.Name) and node.id in {"mo", "marimo"}
                for node in ast.walk(ast.parse(cell.source))
            ):
                raise ValueError(
                    "Jupyter export still uses marimo; use static Markdown "
                    "and keep download controls in the export_notebook cell."
                )
            if not cell.source.strip():
                continue
        cells.append(cell)
    notebook.cells = cells
    notebook.metadata.pop("marimo", None)
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nbformat.validate(notebook)
    return nbformat.writes(notebook).encode("utf-8")


def configuration_source(settings):
    """Materialize the basic-flux controls as editable, ordinary Python assignments."""
    if "cr_model" not in settings:
        return generic_configuration_source(settings)
    primary = settings["cr_model"]
    if primary not in {"H3a", "H4a", "Gaisser–Honda"}:
        raise ValueError(f"Unsupported cosmic-ray model: {primary}")
    primary_code = (
        "(pm.GaisserHonda, None)"
        if primary == "Gaisser–Honda"
        else f"(pm.HillasGaisser2012, {primary!r})"
    )
    return f"""# Hadronic interactions
interaction_model = {settings["interaction_model"]!r}
database = {settings["database"]!r}
low_energy_model = {settings["low_energy_model"]!r}
he_le_transition = {settings["he_le_transition"]!r}  # GeV
he_le_trwidth = {settings["he_le_trwidth"]!r}  # log10(E/GeV) decades; 10–90% width

# Primary cosmic rays and atmosphere
cr_model = {primary_code}
theta_deg = {settings["theta_deg"]!r}  # degrees from vertical
atmosphere = ("CORSIKA", ("BK_USStd", None))

# Cascade energy grid [GeV]
e_min = {settings["e_min"]!r}
e_max = {settings["e_max"]!r}

# Solver
backend = {settings["backend"]!r}"""


#: Order used when materializing a generic configuration cell in Jupyter.
#: Keys missing from *settings* are skipped; unknown keys follow in sorted order.
_CONFIG_ORDER = (
    "database",
    "interaction_model",
    "low_energy_model",
    "he_le_transition",
    "he_le_trwidth",
    "primary",
    "density_model",
    "medium",
    "theta_deg",
    "h_obs",
    "e_min",
    "e_max",
    "backend",
)


def generic_configuration_source(settings):
    """Materialize a generic marimo configuration as plain Python assignments.

    Every setting becomes one module-level variable of the same name, so the
    exported notebook keeps running without marimo. The ``primary`` entry
    (``"ClassName"`` or ``"ClassName:tag"``) becomes a ``primary_model``
    ``(class, tag)`` tuple against ``import crflux.models as pm``.
    """
    lines = ["# MCEq example configuration (materialized from the form)"]
    rendered = set()
    for key in _CONFIG_ORDER:
        if key not in settings:
            continue
        rendered.add(key)
        if key == "primary":
            lines.append("import crflux.models as pm")
            lines.append("primary_model = " + primary_expression(settings[key]))
        elif key == "theta_deg":
            lines.append(f"theta_deg = {settings[key]!r}  # degrees from vertical")
        elif key == "e_min":
            lines.append(f"e_min = {settings[key]!r}  # GeV")
        elif key == "e_max":
            lines.append(f"e_max = {settings[key]!r}  # GeV")
        else:
            lines.append(f"{key} = {settings[key]!r}")
    for key in sorted(set(settings) - rendered):
        lines.append(f"{key} = {settings[key]!r}")
    return "\n".join(lines)


def primary_expression(primary):
    """Render ``"ClassName:tag"`` as a ``(pm.ClassName, tag)`` source string."""
    class_name, _, tag = str(primary).partition(":")
    tag = repr(tag) if tag else "None"
    return f"(pm.{class_name}, {tag})"


def database_models(database=None, medium="air"):
    """Hadronic models available for *database*, sorted.

    *database* is a package name of the data release (``None``: the
    configured ``config.mceq_db_fname``). The models are read from the
    release manifest: every package on the same grid as *database*, so the
    list includes models that MCEq fetches on first use.
    """
    from MCEq import config as _config
    from MCEq.data import download

    name = database or _config.mceq_db_fname
    manifest = download.local_manifest(_config) or download.ensure_manifest(_config)
    grid = manifest["files"][name]["grid"]
    return sorted(
        {
            model
            for entry in manifest["files"].values()
            if entry.get("grid") == grid
            for model in entry.get("models", {}).get(medium, [])
        }
    )


def backend_options():
    """Dropdown mapping of backend labels to ``kernel_config`` values."""
    import sys

    from MCEq.config import detect

    options = {"Automatic": "auto", "NumPy": "numpy_etd2"}
    if sys.platform == "emscripten":
        return options
    for label, capability in (
        ("MKL", "mkl"),
        ("Accelerate", "accelerate"),
        ("CUDA", "cuda"),
    ):
        if getattr(detect, "has_" + capability)():
            options[label] = capability + "_etd2"
    return options


#: Dropdown labels for crflux primary spectra, mapped to ``"ClassName"`` or
#: ``"ClassName:tag"``. Values must stay plain strings so form state and the
#: Jupyter export serialize them.
PRIMARY_OPTIONS = {
    "H3a": "HillasGaisser2012:H3a",
    "H4a": "HillasGaisser2012:H4a",
    "Gaisser–Honda": "GaisserHonda",
    "GST 3-gen": "GaisserStanevTilav:3-gen",
    "GST 4-gen": "GaisserStanevTilav:4-gen",
    "Combined G + HG H3a": "CombinedGHandHG:H3a",
    "Thunman": "Thunman",
    "Zatsepin–Sokolskaya (pamela)": "ZatsepinSokolskaya:pamela",
    "Power law E^-2.7": "SimplePowerlaw27",
    "Poly gonato": "PolyGonato",
}


def configuration_controls(defaults, database=None, medium="air", show=()):
    """Build the shared marimo configuration form used by the examples.

    Returns ``(controls, options)``: a batched form with an Apply button, and
    the enumerated options it was built from (interaction models, primaries,
    backends) so callers can label plots or gate optional sections.

    *defaults* is a dict read for initial values (``interaction_model``,
    ``primary``, ``theta_deg``, ``e_min``, ``e_max``, ``backend``); *show*
    selects the widget groups to render, in order. The interaction-model
    dropdown is populated from the database the notebook ships with, not
    from a live database switch, so the options stay honest for the demo.
    """
    import marimo as mo

    models = database_models(database, medium)
    backends = backend_options()

    widgets, slots = {}, {}
    if "interaction_model" in show:
        widgets["interaction_model"] = mo.ui.dropdown(
            {model: model for model in models},
            value=defaults.get("interaction_model", models[0]),
            label="Interaction model", full_width=True,
        )
        slots["interaction_model"] = ["interaction_model"]
    if "models_multi" in show:
        default_models = defaults.get(
            "models", [model for model in models if model in ("SIBYLL21", "QGSJETII04")][:2] or models[:1]
        )
        widgets["models"] = mo.ui.multiselect(
            models, value=default_models, label="Interaction models",
            full_width=True,
        )
        slots["models_multi"] = ["models"]
    if "primary" in show:
        widgets["primary"] = mo.ui.dropdown(
            PRIMARY_OPTIONS, value=_primary_label(defaults.get("primary", "H3a")),
            label="Primary cosmic rays", full_width=True,
        )
        slots["primary"] = ["primary"]
    if "zenith" in show:
        widgets["theta_deg"] = mo.ui.slider(
            0, 90, step=5, value=defaults.get("theta_deg", 0.0),
            label="Zenith angle [°]", show_value=True, full_width=True,
        )
        slots["zenith"] = ["theta_deg"]
    if "energy" in show:
        widgets["log_e_min"] = mo.ui.slider(
            -1, 3, step=0.5, value=_log10(defaults["e_min"]),
            label="e_min · log₁₀(E/GeV)", show_value=True, full_width=True,
        )
        widgets["log_e_max"] = mo.ui.slider(
            4, 11, step=0.5, value=_log10(defaults["e_max"]),
            label="e_max · log₁₀(E/GeV)", show_value=True, full_width=True,
        )
        slots["energy"] = ["log_e_min", "log_e_max"]
    if "backend" in show:
        widgets["backend"] = mo.ui.dropdown(
            backends, value=_backend_label(backends, defaults.get("backend", "auto")),
            label="Backend", full_width=True,
        )
        slots["backend"] = ["backend"]

    body = "".join(
        "".join(f"{{{key}}}\n\n" for key in slots[name]) for name in show
    )
    rendered = mo.md(
        "<div style=\"display:grid;"
        "grid-template-columns:repeat(auto-fit,minmax(min(100%,350px),1fr));"
        "gap:1rem 2rem\">\n" + body + "</div>"
    ).batch(**widgets).form(submit_button_label="Apply configuration", bordered=False)
    return rendered, {
        "interaction_models": models,
        "primaries": PRIMARY_OPTIONS,
        "backends": backends,
    }


def _log10(value):
    import math

    return math.log10(value)


def _primary_label(primary):
    """Dropdown label for a ``"ClassName:tag"`` code (labels are the keys)."""
    for label, value in PRIMARY_OPTIONS.items():
        if value == primary:
            return label
    return primary


def _backend_label(backends, kernel):
    for label, value in backends.items():
        if value == kernel:
            return label
    return "Automatic"


def apply_configuration(controls, defaults):
    """Merge applied form values into a settings dict, expanding shorthands.

    ``log_e_min``/``log_e_max`` become ``e_min``/``e_max``; keys not applied
    yet fall back to *defaults*.
    """
    settings = dict(defaults)
    if controls.value is not None:
        settings.update(controls.value)
    if "log_e_min" in settings:
        settings["e_min"] = 10.0 ** settings.pop("log_e_min")
    if "log_e_max" in settings:
        settings["e_max"] = 10.0 ** settings.pop("log_e_max")
    return settings


def apply_kernel_and_grid(backend, e_min, e_max, debug_level=0):
    """Push backend, energy grid, and debug level onto ``MCEq.config``."""
    from MCEq import config
    from MCEq.config.detect import resolve_kernel

    config.kernel_config = resolve_kernel(backend)
    config.e_min, config.e_max = e_min, e_max
    config.debug_level = debug_level


#: Environment variable that opens every Build & solve gate, for the static
#: HTML build (``marimo export html`` runs in "run" mode).
AUTORUN_ENV = "MCEQ_EXAMPLES_AUTORUN"


def gate_closed(gate):
    """True while an interactive session waits for the Build & solve button.

    Scripts, tests and exports run straight through; so does any session
    with :data:`AUTORUN_ENV` set to a non-empty value.
    """
    import os

    import marimo as mo

    if os.environ.get(AUTORUN_ENV):
        return False
    return mo.app_meta().mode in ("edit", "run") and not gate.value


@contextmanager
def solve_progress(title="Solving cascade", completion_title="Done"):
    """Yield a ``progress`` callback for one ``solve(progress=...)`` call.

    Inside a marimo kernel the callback drives the native progress widget,
    created on the driver's first call (where the step total becomes known)
    and appended to the enclosing cell. Anywhere else — Jupyter exports,
    plain Python, CI — it drives a tqdm bar instead, so a clean export
    needs no marimo. The bar is finalized when the block exits.

    Example::

        with solve_progress("SIBYLL23E · 60°") as progress:
            mceq.solve(progress=progress)
    """
    try:
        from marimo._runtime.context.utils import running_in_notebook

        in_marimo = running_in_notebook()
    except Exception:
        in_marimo = False

    stack = ExitStack()
    state = {}

    if in_marimo:
        import marimo as mo

        def progress(done, total):
            if "bar" not in state:
                state["bar"] = stack.enter_context(
                    mo.status.progress_bar(
                        total=total, title=title,
                        completion_title=completion_title,
                    )
                )
            bar = state["bar"]
            if done != bar.current:
                bar.update(increment=done - bar.current)
    else:
        from tqdm.auto import tqdm

        def progress(done, total):
            if "bar" not in state:
                state["bar"] = stack.enter_context(tqdm(total=total, desc=title))
            bar = state["bar"]
            if done != bar.n:
                bar.update(done - bar.n)

    try:
        yield progress
    finally:
        stack.close()


@contextmanager
def loop_progress(total, title="Solving cascade", completion_title="Done"):
    """Yield an ``advance()`` callable for a loop that solves repeatedly.

    One bar spans the whole loop (``advance()`` once per solved member),
    instead of a bar per solve. Same marimo-widget/tqdm split as
    :func:`solve_progress`.

    Example::

        with loop_progress(len(angles), "Zenith scan") as advance:
            for theta in angles:
                mceq.set_zenith_azimuth(theta)
                mceq.solve()
                advance()
    """
    try:
        from marimo._runtime.context.utils import running_in_notebook

        in_marimo = running_in_notebook()
    except Exception:
        in_marimo = False

    stack = ExitStack()

    if in_marimo:
        import marimo as mo

        bar = stack.enter_context(
            mo.status.progress_bar(
                total=total, title=title, completion_title=completion_title
            )
        )

        def advance(n=1, title=None):
            bar.update(increment=n, title=title)
    else:
        from tqdm.auto import tqdm

        bar = stack.enter_context(tqdm(total=total, desc=title))

        def advance(n=1, title=None):
            bar.update(n)
            if title is not None:
                bar.set_description(title)

    try:
        yield advance
    finally:
        stack.close()


def primary_model_from(primary):
    """Resolve ``"ClassName:tag"`` from the dropdown to a ``(class, tag)`` tuple."""
    import crflux.models as pm

    class_name, _, tag = str(primary).partition(":")
    return (getattr(pm, class_name), tag or None)


def download_jupyter(filename="notebook.ipynb", configuration=None):
    """Download the current saved source, read at click time (also in Pyodide's VFS)."""
    import marimo as mo
    from marimo._ast.app import InternalApp
    from marimo._ast.load import load_app
    from marimo._runtime.context import get_context

    context = get_context()
    source = context.filename
    if source is None:
        raise ValueError("Save the notebook before downloading Jupyter source")

    def export():
        app = load_app(source)
        if app is None:
            raise ValueError("The source contains no notebook")
        return jupyter_bytes(InternalApp(app), configuration=configuration)

    return mo.download(
        data=export,
        filename=filename,
        mimetype="application/x-ipynb+json",
        label="Download Jupyter notebook",
    )


async def prepare_database(database):
    """Select the native database or fetch the gallery subset in Pyodide."""
    import sys

    from MCEq import config

    if sys.platform != "emscripten":
        return database

    from pyodide.http import pyfetch

    name = "mceq_db_browser.h5"
    destination = config.data_dir / name
    if not destination.exists():
        response = await pyfetch("../public/mceq_db_browser.h5")
        if not response.ok:
            raise RuntimeError(
                f"Could not load gallery database: HTTP {response.status}"
            )
        destination.write_bytes(await response.bytes())
    return name
