"""The gallery's browser download and CLI export share one conversion path."""

import json
from pathlib import Path

import pytest

pytest.importorskip("marimo")
pytest.importorskip("nbformat")

from marimo._ast.app import InternalApp
from marimo._ast.load import load_app

from MCEq.notebook import jupyter_bytes

SOURCE = """import marimo
app = marimo.App()
@app.cell
def _():
    import marimo as mo
    value = 42
    return mo, value
@app.cell
def _(mo):
    mo.md("# Example")
    return
@app.cell
def _(value):
    result = value * 2
    return (result,)
@app.cell
def export_notebook():
    from MCEq.notebook import download_jupyter
    download_jupyter()
    return
"""


def test_export_keeps_calculation_and_removes_controls(tmp_path):
    path = tmp_path / "example.py"
    path.write_text(SOURCE)
    notebook = json.loads(jupyter_bytes(InternalApp(load_app(path))))
    code = "\n".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
    )
    assert "marimo" not in code
    assert "download_jupyter" not in code
    assert notebook["cells"][1]["cell_type"] == "markdown"
    namespace = {}
    exec(code, namespace)  # noqa: S102 — execute our fixed fixture
    assert namespace["result"] == 84
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    assert not any(c.get("outputs") for c in notebook["cells"])


def test_export_rejects_unconverted_widget(tmp_path):
    path = tmp_path / "example.py"
    path.write_text(SOURCE.replace("value = 42", "value = mo.ui.slider(1, 10)"))
    with pytest.raises(ValueError, match="still uses marimo"):
        jupyter_bytes(InternalApp(load_app(path)))


def test_basic_flux_export_is_current():
    source = Path(__file__).parents[1] / "docs/examples/Basic_flux.py"
    exported = json.loads(jupyter_bytes(InternalApp(load_app(source))))
    assert exported == json.loads(source.with_suffix(".ipynb").read_text())


def test_selected_controls_export_as_plain_configuration():
    source = Path(__file__).parents[1] / "docs/examples/Basic_flux.py"
    selected = {
        "interaction_model": "QGSJET-II-04",
        "database": "mceq_db_lext_dpm193_v142.h5",
        "low_energy_model": None,
        "he_le_transition": 80.0,
        "he_le_trwidth": 0.3,
        "cr_model": "H4a",
        "theta_deg": 30.0,
        "e_min": 10.0,
        "e_max": 1.0e9,
        "backend": "numpy_etd2",
    }
    exported = json.loads(jupyter_bytes(InternalApp(load_app(source)), selected))
    code = "\n".join(
        "".join(c["source"]) for c in exported["cells"] if c["cell_type"] == "code"
    )
    assert "controls" not in code
    assert "import marimo" not in code
    assert "settings" not in code
    cell = next(
        c
        for c in exported["cells"]
        if "# Hadronic interactions" in "".join(c["source"])
    )
    assert not cell["metadata"].get("jupyter", {}).get("source_hidden")
    assert "remove-input" not in cell["metadata"].get("tags", [])
    import crflux.models as pm

    namespace = {"pm": pm}
    exec("".join(cell["source"]), namespace)  # noqa: S102 — execute the exported configuration
    assert namespace["interaction_model"] == selected["interaction_model"]
    assert namespace["cr_model"] == (pm.HillasGaisser2012, "H4a")
    for key in ("e_min", "e_max", "theta_deg", "backend", "low_energy_model"):
        assert namespace[key] == selected[key]
