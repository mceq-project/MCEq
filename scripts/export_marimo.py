"""Export a widget-free marimo example to Jupyter without a marimo dependency."""

import argparse
import ast
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    destination = args.source.with_suffix(".ipynb")

    with tempfile.TemporaryDirectory() as directory:
        exported = Path(directory) / "notebook.ipynb"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "marimo",
                "export",
                "ipynb",
                str(args.source),
                "--sort",
                "top-down",
                "-o",
                str(exported),
            ],
            check=True,
        )
        notebook = nbformat.read(exported, as_version=4)

    for cell in notebook.cells:
        cell.metadata.pop("marimo", None)
        if cell.cell_type != "code":
            continue
        # Static mo.md cells have become Markdown; their import is now unused.
        cell.source = "\n".join(
            line for line in cell.source.splitlines() if line != "import marimo as mo"
        )
        if any(
            isinstance(node, ast.Name) and node.id in {"mo", "marimo"}
            for node in ast.walk(ast.parse(cell.source))
        ):
            raise ValueError(
                "This example still uses marimo at runtime; "
                "use static Markdown and plain Python."
            )

    notebook.cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type != "code" or cell.source.strip()
    ]
    notebook.metadata.pop("marimo", None)
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nbformat.validate(notebook)
    nbformat.write(notebook, destination)
    print(destination)


if __name__ == "__main__":
    main()
