"""In-memory Jupyter downloads for marimo examples, including Pyodide.

Marimo's private conversion API is confined to this module and checked against
the gallery's version. Importing MCEq does not require marimo.
"""

import ast


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
        }:
            continue
        if metadata.get("name") == "selected_configuration":
            settings = configuration
            if settings is None:
                assignment = ast.parse(cell.source).body[0]
                settings = ast.literal_eval(assignment.value)
            cell.source = configuration_source(settings)
            cell.metadata = {}  # The plain configuration must be visible in Jupyter.
        if cell.cell_type == "code":
            cell.source = "\n".join(
                line
                for line in cell.source.splitlines()
                if line != "import marimo as mo"
            )
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
    """Hadronic model group names available in *database*, sorted.

    *database* is a filename resolved against ``config.data_dir`` or an
    absolute path; ``None`` uses the configured default. ``*_indptrs``
    bookkeeping groups are filtered out.
    """
    import h5py

    from MCEq import config as _config

    path = _config.data_dir / (database or _config.mceq_db_fname)
    with h5py.File(path, "r") as source:
        return sorted(
            name
            for name in source[f"hadronic_interactions/{medium}"]
            if not name.endswith("_indptrs")
        )


def backend_options():
    """Dropdown mapping of backend labels to ``kernel_config`` values."""
    from MCEq.config.detect import available

    options = {"Automatic": "auto", "NumPy": "numpy_etd2"}
    for label, capability in (
        ("MKL", "mkl"),
        ("Accelerate", "accelerate"),
        ("CUDA", "cuda"),
    ):
        if available("has_" + capability):
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
