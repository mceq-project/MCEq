"""Export with the same in-memory path as the notebook's download button."""

import argparse
from pathlib import Path

from marimo._ast.app import InternalApp
from marimo._ast.load import load_app

from MCEq.notebook import jupyter_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    app = load_app(args.source)
    if app is None:
        parser.error("The source contains no notebook")
    destination = args.source.with_suffix(".ipynb")
    destination.write_bytes(jupyter_bytes(InternalApp(app)))
    print(destination)


if __name__ == "__main__":
    main()
