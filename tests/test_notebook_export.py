"""The gallery's browser download and CLI export share one conversion path."""

import json
from pathlib import Path

import pytest

pytest.importorskip("marimo")
nbformat = pytest.importorskip("nbformat")

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


GATE_SOURCE = """import marimo
app = marimo.App()
@app.cell
def _():
    import marimo as mo
    return (mo,)
@app.cell
def run_gate(mo):
    gate = mo.ui.run_button(label="Build & solve")
    gate
    return (gate,)
@app.cell
def run_stop(gate, mo):
    interactive = mo.app_meta().mode in ("edit", "run")
    mo.stop(interactive and not gate.value)
    proceed = True
    return (proceed,)
@app.cell
def _(mo, proceed):
    _ = proceed
    result = 42
    return (result,)
"""


def test_export_strips_run_gate(tmp_path):
    path = path = tmp_path / "gated.py"
    path.write_text(GATE_SOURCE)
    notebook = json.loads(jupyter_bytes(InternalApp(load_app(path))))
    code = "\n".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
    )
    for token in ("run_gate", "run_stop", "run_button", "mo.stop", "proceed"):
        assert token not in code
    namespace = {}
    exec(code, namespace)  # noqa: S102 — execute our fixed fixture
    assert namespace["result"] == 42


def test_export_rejects_unconverted_widget(tmp_path):
    path = tmp_path / "example.py"
    path.write_text(SOURCE.replace("value = 42", "value = mo.ui.slider(1, 10)"))
    with pytest.raises(ValueError, match="still uses marimo"):
        jupyter_bytes(InternalApp(load_app(path)))


EXAMPLES = sorted((Path(__file__).parents[1] / "examples").glob("*/*.py"))


@pytest.mark.parametrize("source", EXAMPLES, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_every_example_exports_to_plain_jupyter(source):
    """Each example converts without marimo left in the code cells."""
    exported = json.loads(jupyter_bytes(InternalApp(load_app(source))))
    nbformat.validate(nbformat.reads(json.dumps(exported), as_version=4))
    code = "\n".join(
        "".join(c["source"]) for c in exported["cells"] if c["cell_type"] == "code"
    )
    assert "import marimo" not in code


def test_selected_controls_export_as_plain_configuration():
    source = Path(__file__).parents[1] / "examples/getting_started/basic_flux.py"
    selected = {
        "interaction_model": "QGSJET-II-04",
        "database": "mceq_base_air_1d_v2.h5",
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
