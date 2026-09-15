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
        }:
            continue
        if metadata.get("name") == "selected_configuration":
            cell.source = configuration_source(configuration)
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
