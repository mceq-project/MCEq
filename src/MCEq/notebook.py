"""In-memory Jupyter downloads for marimo examples, including Pyodide.

Marimo's private conversion API is confined to this module and checked against
the gallery's version. Importing MCEq does not require marimo.
"""

import ast


def jupyter_bytes(app):
    """Export current cell source without running the cascade or spawning a process."""
    import nbformat
    from marimo._convert.ipynb import convert_from_ir_to_ipynb

    notebook = nbformat.reads(
        convert_from_ir_to_ipynb(app, sort_mode="top-down"), as_version=4
    )
    cells = []
    for cell in notebook.cells:
        metadata = cell.metadata.pop("marimo", {})
        if metadata.get("name") == "export_notebook":
            continue
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


def download_jupyter(filename="notebook.ipynb"):
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
        return jupyter_bytes(InternalApp(app))

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
