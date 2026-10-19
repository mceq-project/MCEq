"""The static gallery keeps physical tables intact and exports a portable wheel."""

import importlib.util
import json
import zipfile
from pathlib import Path

import h5py
import numpy as np

_spec = importlib.util.spec_from_file_location(
    "build_browser_gallery",
    Path(__file__).parents[1] / "scripts/build_browser_gallery.py",
)
gallery = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gallery)


def test_subset_preserves_selected_tables(tmp_path):
    source, target = tmp_path / "source.h5", tmp_path / "subset.h5"
    with h5py.File(source, "w") as original:
        original.attrs["version"] = 142
        original.attrs["mceq_db_manifest"] = "mceq_db_manifest_v2.yaml"
        for name in ("common/grid", "decays/air/values", "continuous_losses/air/mu"):
            original.create_dataset(name, data=np.arange(6, dtype=np.float64))
        for group in ("cross_sections", "hadronic_interactions"):
            for model in ("KEEP", "OMIT"):
                data = original.create_dataset(f"{group}/air/{model}", data=np.eye(3))
                data.attrs["description"] = model
                original.create_dataset(f"{group}/air/{model}_indptrs", data=[0, 3])
    gallery.subset_database(source, target, ["KEEP"])
    with h5py.File(source) as original, h5py.File(target) as subset:
        # Self-contained: the manifest pointer is dropped, the model list
        # names only the kept models; every other attribute is copied.
        expected = dict(original.attrs)
        del expected["mceq_db_manifest"]
        expected["mceq_db_models"] = json.dumps({"air": ["KEEP"]})
        assert dict(subset.attrs) == expected
        names = []
        subset.visit(names.append)
        for name in names:
            if isinstance(subset[name], h5py.Dataset):
                np.testing.assert_array_equal(subset[name][...], original[name][...])
                assert dict(subset[name].attrs) == dict(original[name].attrs)
        assert "hadronic_interactions/air/OMIT" not in subset
        assert "cross_sections/air/OMIT_indptrs" not in subset


def test_browser_wheel_contains_python_and_no_database(tmp_path):
    wheel = gallery.browser_wheel(tmp_path, ["numpy", "h5py"])
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert "MCEq/notebook.py" in names
        assert "MCEq/geometry/corsika_numpy.py" in names
        assert not any(n.endswith((".so", ".h5", ".dll")) for n in names)
        metadata = archive.read("mceq-2.0.0+browser.dist-info/METADATA").decode()
        assert "Requires-Dist: h5py" in metadata


def test_bootstrap_runs_and_saves_browser_edits(tmp_path):
    path = tmp_path / "index.html"
    config = {"runtime": {}, "save": {}, "ai": {}}
    path.write_text('    "config": ' + json.dumps(config) + ",\n")
    gallery.configure_browser(path)
    config = json.loads(path.read_text().split(": ", 1)[1].rstrip(",\n"))
    assert config["runtime"]["auto_instantiate"] is True
    assert config["save"]["autosave"] == "after_delay"
