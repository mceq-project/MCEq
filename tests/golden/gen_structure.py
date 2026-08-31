"""Golden section `structure`: the static shape of the package.

Pins, from the source text alone (`ast`, no MCEq import, no build):

* the module inventory -- repo-relative path and `wc -l` line count for every
  `.py` under `src/MCEq` and `mceq_config`;
* the intra-package import graph as a sorted adjacency list, one `a -> b` entry
  per ordered pair, with a flag saying whether the import runs at import time or
  is deferred inside a function;
* the modules that import `MCEq.config` -- the ledger Phase 1 empties, except
  for `MCEq.core` (the driver, which is allowed to) and `mceq_config` (the shim
  whose entire job is to alias it);
* the import cycles.

THIS SECTION IS A RATCHET, NOT A FREEZE. Every phase of the refactor legitimately
changes it: modules move, split and shrink, and edges appear and disappear by
design. It fails so that the change is read and regenerated deliberately, the
way the `.importlinter` ledgers are shrunk deliberately. It earns its place in
the harness because an unreviewed change to the import graph is exactly what the
layer contracts exist to catch, and because the contracts have blind spots the
graph does not: a stale `exhaustive_ignores` entry for a module that is gone is
silent to import-linter, and visible here.

`diff_report` renders a mismatch as added/removed/resized lines; the golden test
calls it instead of the harness's generic per-key message, which for a string
array can only say that the shapes differ.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from ._harness import make_provenance

SECTION = "structure"

ROOT = Path(__file__).resolve().parents[2]
ROOTS = ("src/MCEq", "mceq_config")
PACKAGES = ("MCEq", "mceq_config")

#: Imports inside these run when they are called, not when the module loads.
#: A class body executes at import time and counts as an import-time import.
DEFERRED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def module_name(path: Path) -> str:
    """Dotted name of the module at `path`, e.g. `src/MCEq/data.py` -> MCEq.data."""
    parts = list(path.relative_to(ROOT).parts)
    if parts[0] == "src":
        parts = parts[1:]
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].removesuffix(".py")
    return ".".join(parts)


def source_files() -> list[Path]:
    return sorted(path for root in ROOTS for path in (ROOT / root).rglob("*.py"))


def count_lines(path: Path) -> int:
    """Number of newlines in `path`, which is what ``wc -l`` reports."""
    return path.read_bytes().count(b"\n")


def _imports(node, at_import_time: bool, found: list) -> None:
    """Collect `(node, at_import_time)` for every import statement below `node`."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.Import, ast.ImportFrom)):
            found.append((child, at_import_time))
        inner = at_import_time and not isinstance(child, DEFERRED_SCOPES)
        _imports(child, inner, found)


def _resolve(name: str, modules: set[str]) -> str | None:
    """Longest prefix of `name` that is a module in the package, else None.

    `from MCEq.misc import info` names an attribute, and the C extension in
    `from MCEq.geometry.nrlmsise00 import nrlmsise00` is not a `.py` file; both
    resolve to the module that holds them, as import-linter's graph does.
    """
    if name.split(".")[0] not in PACKAGES:
        return None
    while name:
        if name in modules:
            return name
        name, _, _ = name.rpartition(".")
    return None


def _targets(node, package: str, modules: set[str]):
    """Modules that one import statement depends on."""
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    else:
        base = node.module or ""
        if node.level:
            anchor = package
            for _ in range(node.level - 1):
                anchor = anchor.rpartition(".")[0]
            base = f"{anchor}.{base}" if base else anchor
        names = [f"{base}.{alias.name}" for alias in node.names]
    return [t for t in (_resolve(name, modules) for name in names) if t]


def import_graph(files: list[Path]) -> dict[tuple[str, str], bool]:
    """Ordered module pairs -> True when at least one of the imports is at import time.

    Self-imports are kept: `particlemanager` imports itself inside a method, and
    import-linter counts that dependency too.
    """
    modules = {module_name(path): path for path in files}
    known = set(modules)
    edges: dict[tuple[str, str], bool] = {}
    for name, path in sorted(modules.items()):
        tree = ast.parse(path.read_bytes(), filename=str(path))
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        statements: list = []
        _imports(tree, True, statements)
        for node, at_import_time in statements:
            for target in _targets(node, package, known):
                edges[name, target] = edges.get((name, target), False) or at_import_time
    return edges


def cycles(edges) -> list[str]:
    """Elementary cycles as `a -> b -> a`.

    Each cycle is enumerated once, from its lexicographically smallest member.
    """
    successors: dict[str, list[str]] = {}
    for source, target in sorted(edges):
        successors.setdefault(source, []).append(target)

    found: set[str] = set()

    def walk(start: str, node: str, path: list[str]) -> None:
        for nxt in successors.get(node, ()):
            if nxt == start:
                found.add(" -> ".join([*path, start]))
            elif nxt > start and nxt not in path:
                walk(start, nxt, [*path, nxt])

    for start in sorted(successors):
        walk(start, start, [start])
    return sorted(found)


def _text(values) -> np.ndarray:
    """String array whose dtype stays textual when there is nothing in it."""
    items = list(values)
    return np.array(items, dtype=np.str_ if items else "<U1")


def build() -> tuple[dict, dict]:
    """Produce (arrays, provenance) for this section. Mutates no process state."""
    files = source_files()
    paths = [path.relative_to(ROOT).as_posix() for path in files]
    lines = [count_lines(path) for path in files]
    edges = import_graph(files)
    pairs = sorted(edges)
    importers = [source for source, target in pairs if target == "MCEq.config"]

    arrays = {
        "modules/path": _text(paths),
        "modules/lines": np.array(lines, dtype=np.int64),
        "imports/edge": _text(f"{source} -> {target}" for source, target in pairs),
        "imports/at_import_time": np.array([edges[pair] for pair in pairs], dtype=bool),
        "config_importers": _text(importers),
        "cycles": _text(cycles(edges)),
    }

    provenance = make_provenance(
        SECTION,
        note=(
            "Static module inventory, import graph, MCEq.config importers and import "
            "cycles, parsed from the source with ast. A ratchet: every phase of the "
            "layered-architecture refactor changes it, and regenerating is the "
            "reviewed act of accepting the new shape."
        ),
        tolerances={},
        extra={
            "roots": list(ROOTS),
            "module_count": len(files),
            "total_lines": sum(lines),
            "edge_count": len(edges),
        },
    )
    return arrays, provenance


def _lookup(arrays: dict, keys: str, values: str) -> dict:
    """Two parallel arrays of a section, zipped into a mapping."""
    return dict(zip(arrays[keys].tolist(), arrays[values].tolist()))


def diff_report(golden: dict, produced: dict) -> list[str]:
    """Readable rendering of a `structure` mismatch, `+` added and `-` removed."""
    out: list[str] = []

    old = _lookup(golden, "modules/path", "modules/lines")
    new = _lookup(produced, "modules/path", "modules/lines")
    for path in sorted(set(new) - set(old)):
        out.append(f"  + module {path} ({new[path]} lines)")
    for path in sorted(set(old) - set(new)):
        out.append(f"  - module {path} ({old[path]} lines)")
    for path in sorted(set(old) & set(new)):
        if old[path] != new[path]:
            out.append(f"    module {path}: {old[path]} -> {new[path]} lines")
    if set(old) != set(new) or sum(old.values()) != sum(new.values()):
        out.append(
            f"    total: {len(old)} modules / {sum(old.values())} lines"
            f" -> {len(new)} / {sum(new.values())}"
        )

    for key, label in (
        ("imports/edge", "import"),
        ("config_importers", "MCEq.config importer"),
        ("cycles", "cycle"),
    ):
        was, now = set(golden[key].tolist()), set(produced[key].tolist())
        out += [f"  + {label} {item}" for item in sorted(now - was)]
        out += [f"  - {label} {item}" for item in sorted(was - now)]

    was_flags = _lookup(golden, "imports/edge", "imports/at_import_time")
    now_flags = _lookup(produced, "imports/edge", "imports/at_import_time")
    for edge in sorted(set(was_flags) & set(now_flags)):
        if was_flags[edge] != now_flags[edge]:
            when = "at import time" if now_flags[edge] else "deferred"
            out.append(f"    import {edge} is now {when}")

    return out
