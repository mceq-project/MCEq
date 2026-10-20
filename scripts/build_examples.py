"""Build the static example pages: run every marimo notebook under examples/.

For each ``examples/<topic>/<name>.py`` this writes, below ``--output``:

* ``<topic>/<name>.html`` -- ``marimo export html`` with every Build & solve
  gate open (:data:`MCEq.notebook.AUTORUN_ENV`), so the page shows the solved
  figures;
* ``<topic>/<name>.ipynb`` -- the clean Jupyter export of
  :func:`MCEq.notebook.jupyter_bytes` (no outputs, no marimo);
* ``<topic>/<name>.py`` -- the marimo source;
* ``index.html`` and ``manifest.json`` -- the gallery and its provenance.

The notebooks need the full release databases (2D examples about 1.5 GB) and
run for minutes each, so this runs on a workstation or a scheduled job, not in
the Read the Docs build. The default output is picked up by Sphinx through
``html_extra_path`` (docs/conf.py).

    python scripts/build_examples.py                    # all examples
    python scripts/build_examples.py -k corsika -j 2    # a subset, 2 at a time
"""

import argparse
import ast
import html
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

#: Gallery sections, in display order: directory -> heading.
TOPICS = {
    "getting_started": "Getting started",
    "atmosphere": "Atmosphere",
    "fluxes": "Lepton fluxes",
    "hadronic": "Hadronic interactions",
    "showers_2d": "Angular (2D) cascades and CORSIKA comparisons",
}


def notebooks(patterns=()):
    paths = sorted(
        path
        for topic in TOPICS
        for path in (EXAMPLES / topic).glob("*.py")
        if not path.name.startswith("_")
    )
    if patterns:
        paths = [p for p in paths if any(k in str(p) for k in patterns)]
    return paths


def describe(path):
    """``(title, first sentence)`` from the ``app_title`` and the first mo.md."""
    title, summary = path.stem.replace("_", " ").capitalize(), ""
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", getattr(node.func, "id", ""))
        if name == "App":
            for keyword in node.keywords:
                if keyword.arg == "app_title":
                    title = ast.literal_eval(keyword.value)
        elif name == "md" and node.args and not summary:
            try:
                text = ast.literal_eval(node.args[0])
            except ValueError:
                continue
            lines = [ln.strip() for ln in text.strip().splitlines()]
            body = " ".join(ln for ln in lines if ln and not ln.startswith("#"))
            summary = body.split(". ")[0].rstrip(".") + "." if body else ""
    return title, summary


def build_one(path, output):
    from marimo._ast.app import InternalApp
    from marimo._ast.load import load_app

    from MCEq.notebook import AUTORUN_ENV, jupyter_bytes

    target = output / path.parent.name
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target / path.name)
    (target / f"{path.stem}.ipynb").write_bytes(
        jupyter_bytes(InternalApp(load_app(path)))
    )
    env = dict(os.environ, **{AUTORUN_ENV: "1", "MPLBACKEND": "Agg"})
    start = time.time()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "html",
            str(path),
            "-o",
            str(target / f"{path.stem}.html"),
            "-f",
        ],
        cwd=path.parent,
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start
    ok = result.returncode == 0
    status = "ok" if ok else "FAILED"
    print(f"{status:6} {elapsed:6.1f}s  {path.relative_to(ROOT)}", flush=True)
    if not ok:
        print(result.stdout[-4000:], result.stderr[-4000:], sep="\n", flush=True)
    return {
        "source": str(path.relative_to(ROOT)),
        "ok": ok,
        "seconds": round(elapsed, 1),
    }


def write_index(output, paths, results):
    by_source = {r["source"]: r for r in results}
    sections = []
    for topic, heading in TOPICS.items():
        cards = []
        for path in (p for p in paths if p.parent.name == topic):
            title, summary = describe(path)
            stem = f"{topic}/{path.stem}"
            failed = not by_source.get(str(path.relative_to(ROOT)), {}).get("ok")
            cards.append(
                f'<article><h3><a href="{stem}.html">{html.escape(title)}</a>'
                + (" <small>(not built)</small>" if failed else "")
                + f"</h3><p>{html.escape(summary)}</p><nav>"
                f'<a href="{stem}.html">View</a>'
                f'<a href="{stem}.ipynb" download>Jupyter</a>'
                f'<a href="{stem}.py" download>marimo</a></nav></article>'
            )
        if cards:
            sections.append(f"<h2>{html.escape(heading)}</h2>" + "".join(cards))
    (output / "index.html").write_text(
        """<!doctype html>
<html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MCEq · Examples</title>
<style>body{font:16px/1.55 system-ui,sans-serif;color:#263342;background:#f7f9fb;margin:0}
main{max-width:880px;margin:56px auto;padding:0 24px}h1{font-size:36px;margin-bottom:4px}
h2{margin-top:40px;border-bottom:1px solid #dce4e9;padding-bottom:4px}
article{background:#fff;border:1px solid #dce4e9;border-radius:10px;padding:16px 22px;margin:14px 0}
h3{margin:0 0 4px}a{color:#087e77}nav a{margin-right:18px;font-size:14px}small{color:#8a5a2b}
p{margin:4px 0 8px}</style>
<main><h1>MCEq examples</h1>
<p>Each page is a marimo notebook run end to end. Download the Jupyter or marimo
source to change the configuration and run it locally
(<code>marimo edit &lt;file&gt;.py</code>).</p>
"""
        + "".join(sections)
        + "</main></html>\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs/_build/site/examples"
    )
    parser.add_argument("-k", nargs="*", default=(), help="path substrings")
    parser.add_argument("-j", type=int, default=1, help="parallel notebooks")
    args = parser.parse_args()

    paths = notebooks(args.k)
    if not paths:
        parser.error("no example matched")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.j) as pool:
        results = list(pool.map(lambda p: build_one(p, output), paths))

    # A partial build (-k) keeps the earlier entries of the manifest.
    manifest_path = output / "manifest.json"
    previous = (
        json.loads(manifest_path.read_text())["notebooks"]
        if manifest_path.exists() and args.k
        else []
    )
    merged = {r["source"]: r for r in previous + results}
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    )
    manifest_path.write_text(
        json.dumps(
            {
                "source_commit": commit,
                "source_dirty": dirty,
                "notebooks": sorted(merged.values(), key=lambda r: r["source"]),
            },
            indent=2,
        )
        + "\n"
    )
    write_index(output, notebooks(), list(merged.values()))
    failed = [r["source"] for r in results if not r["ok"]]
    print(f"{len(results) - len(failed)}/{len(results)} built -> {output}")
    sys.exit(bool(failed))


if __name__ == "__main__":
    main()
