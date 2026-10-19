"""Build a static Pyodide basic-flux gallery from this checkout and a public database.

Generated wheels and HDF5 files are deployment artifacts, never source files.
Model tables are copied losslessly; the energy grid and decay tables are retained.
"""

import argparse
import base64
import csv
import hashlib
import html
import io
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import h5py
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def subset_database(source, target, models):
    with h5py.File(source, "r") as original, h5py.File(target, "w") as subset:
        for key, value in original.attrs.items():
            subset.attrs[key] = value
        # A self-contained file: no manifest lookups for missing models.
        subset.attrs.pop("mceq_db_manifest", None)
        subset.attrs["mceq_db_models"] = json.dumps({"air": list(models)})
        for name in ("common", "decays"):
            original.copy(name, subset)
        original.copy(
            "continuous_losses/air", subset.require_group("continuous_losses")
        )
        for group in ("cross_sections", "hadronic_interactions"):
            destination = subset.require_group(f"{group}/air")
            for model in models:
                original.copy(f"{group}/air/{model}", destination)
                pointers = f"{group}/air/{model}_indptrs"
                if pointers in original:
                    original.copy(pointers, destination)


def browser_wheel(directory, dependencies):
    # This artifact supplies Python sources to Pyodide. The browser uses the
    # scalar CORSIKA formulas and the existing NumPy ETD2 stage fallback.
    version = "2.0.0+browser"
    filename = directory / f"mceq-{version}-py3-none-any.whl"
    info = f"mceq-{version}.dist-info"
    files = {
        str(path.relative_to(ROOT / "src")): path.read_bytes()
        for path in sorted((ROOT / "src/MCEq").rglob("*.py"))
    }
    metadata = [
        "Metadata-Version: 2.1",
        "Name: MCEq",
        f"Version: {version}",
        "Summary: MCEq Python sources for the CORSIKA/NumPy browser gallery",
        *[f"Requires-Dist: {dep}" for dep in dependencies],
        "",
    ]
    files[f"{info}/METADATA"] = "\n".join(metadata).encode()
    files[f"{info}/WHEEL"] = (
        b"Wheel-Version: 1.0\nGenerator: MCEq browser gallery\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    )
    files[f"{info}/licenses/LICENSE"] = (ROOT / "LICENSE").read_bytes()
    records = io.StringIO()
    writer = csv.writer(records, lineterminator="\n")
    for name, content in files.items():
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(content).digest())
            .rstrip(b"=")
            .decode()
        )
        writer.writerow((name, f"sha256={digest}", len(content)))
    writer.writerow((f"{info}/RECORD", "", ""))
    files[f"{info}/RECORD"] = records.getvalue().encode()
    with zipfile.ZipFile(filename, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        for name, content in files.items():
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            wheel.writestr(entry, content)
    return filename


def configure_browser(index):
    """Set launch/save behavior in marimo 0.24's static bootstrap config."""
    document = index.read_text()
    pattern = re.compile(r'^(\s*"config": )(.+)(,)$', re.MULTILINE)
    match = pattern.search(document)
    if match is None:
        raise ValueError("marimo HTML bootstrap changed; review the gallery exporter")
    config = json.loads(match[2])
    config["runtime"]["auto_instantiate"] = True
    config["save"]["autosave"] = "after_delay"
    config["ai"]["enabled"] = False
    replacement = match[1] + json.dumps(config) + match[3]
    index.write_text(document[: match.start()] + replacement + document[match.end() :])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", type=Path,
        help="v2 release package; the default 1D package if omitted",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs/_build/site/browser"
    )
    parser.add_argument("--models", nargs="+", default=["SIBYLL23E"])
    args = parser.parse_args()
    if "SIBYLL23E" not in args.models:
        parser.error("--models must include SIBYLL23E, the notebook default")
    if args.database is None:
        from MCEq import config
        from MCEq.data.download import ensure_package

        args.database = Path(ensure_package(config, config.mceq_db_fname))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    app_dir = output / "basic-flux"
    public = app_dir / "public"
    public.mkdir(parents=True, exist_ok=True)
    database = public / "mceq_db_browser.h5"
    subset_database(args.database, database, args.models)

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    with tempfile.TemporaryDirectory() as directory:
        staging = Path(directory)
        wheel = browser_wheel(staging, project["dependencies"])
        source = (ROOT / "examples/getting_started/basic_flux.py").read_text()
        end = source.index("# ///", len("# /// script")) + len("# ///")
        dependencies = [
            "marimo==0.24.0",
            f"MCEq @ {wheel.as_uri()}",
            "matplotlib",
            "numpy",
            "nbformat",
            *project["dependencies"],
        ]
        header = '# /// script\n# requires-python = ">=3.12"\n# dependencies = [\n'
        header += "".join(
            f"#     {json.dumps(dep)},\n" for dep in dict.fromkeys(dependencies)
        )
        header += "# ]\n# ///"
        notebook = staging / "basic_flux.py"
        notebook.write_text(header + source[end:])
        subprocess.run(
            [
                sys.executable,
                "-m",
                "marimo",
                "export",
                "html-wasm",
                str(notebook),
                "--mode",
                "edit",
                "-o",
                str(app_dir),
                "-f",
            ],
            check=True,
        )
        configure_browser(app_dir / "index.html")
        wheel_hash = sha256(wheel)

    manifest = {
        "source_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        ),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "wheel_sha256": wheel_hash,
        "source_database": args.database.name,
        "source_database_sha256": sha256(args.database),
        "models": args.models,
        "medium": "air",
        "database_bytes": database.stat().st_size,
        "database_sha256": sha256(database),
        "table_policy": "lossless model subset; full energy grid and decay tables",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    model_labels = html.escape(" · ".join(args.models))
    (output / "index.html").write_text(
        """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MCEq · Interactive examples</title>
<style>body{font:17px/1.6 system-ui,sans-serif;color:#263342;background:#f7f9fb;margin:0}
main{max-width:850px;margin:70px auto;padding:0 24px}h1{font-size:40px;letter-spacing:-1px}
small{color:#52606d}a{color:#087e77}article{background:white;border:1px solid #dce4e9;border-radius:12px;padding:28px;margin-top:32px}
h2{margin-top:0}nav{margin-top:24px}nav a{margin-right:24px}</style>
<main><small>MCEq / EXAMPLES</small><h1>Atmospheric cascades, in your browser.</h1>
<p>Run and edit a notebook without installing Python. Download a Jupyter copy when you are ready to work locally.</p>
<article><h2>Atmospheric lepton fluxes</h2><p>Compare conventional, prompt and total muon and neutrino fluxes at ground level.</p>
<p><small>MODEL_LABELS · US standard atmosphere · NumPy solver</small></p>
<nav><a href="basic-flux/">Open notebook →</a></nav></article>
<p><small>The first launch downloads Python packages and the selected model tables. Calculations run on your device.</small></p></main></html>""".replace(
            "MODEL_LABELS", model_labels
        )
    )
    print(json.dumps(manifest, indent=2))
    print(f"Preview: python -m http.server --directory {output} 8000")


if __name__ == "__main__":
    main()
