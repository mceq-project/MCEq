"""`MCEq.operators` and the flat modules it replaced expose one surface.

`MCEq.operators.compiled` and `MCEq.operators.secant` are `MCEq.operator_assembly`
and `MCEq.secant` moved into the operators layer; the flat modules are silent
re-export shims for this release, warn in the next and are removed after
(refactoring plan, D14). This is the gate that keeps the move a relocation:
every name a flat module defined resolves from both paths and is the same
object, and the shim's `__all__` is exactly that set -- nothing added, nothing
lost.

`MOVED` lists the surface as of d3f06c2, the commit before the move. Not part
of it, and deliberately not re-exported: the incidental module aliases the
implementations import (`np`, `sp`, `SimpleNamespace`, `info`) and the private
helpers (`_permute_csr`, `_readout_matrix`, `_T_CACHE`). Nothing in the tree or
in the documented API reaches those through the flat path.
"""

import ast
import importlib
import subprocess
import sys

import pytest

#: Flat module -> (module that owns it now, the names it defined).
MOVED = {
    "MCEq.operator_assembly": (
        "MCEq.operators.compiled",
        (
            "CompiledOperator",
            "compile_operator",
            "identity_layout",
            "secant_coupling",
            "secant_layout",
            "split_diagonal",
        ),
    ),
    "MCEq.secant": (
        "MCEq.operators.secant",
        (
            "build_secant_kernel_ops",
            "secant_coupling_matrix",
        ),
    ),
}


@pytest.mark.parametrize("flat", sorted(MOVED))
def test_moved_names_are_one_object_on_both_paths(flat):
    """The shim forwards; it does not carry a second copy of anything."""
    new, names = MOVED[flat]
    old_module = importlib.import_module(flat)
    new_module = importlib.import_module(new)
    for name in names:
        assert hasattr(new_module, name), f"{new} does not define {name}"
        assert hasattr(old_module, name), f"{flat}.{name} no longer resolves"
        assert getattr(old_module, name) is getattr(new_module, name), (
            f"{flat}.{name} is not the same object as {new}.{name}"
        )


@pytest.mark.parametrize("flat", sorted(MOVED))
def test_shim_all_is_the_pinned_surface(flat):
    new, names = MOVED[flat]
    assert sorted(importlib.import_module(flat).__all__) == sorted(names)


@pytest.mark.parametrize("flat", sorted(MOVED))
def test_the_new_module_owns_the_definitions(flat):
    """Each name is defined in the operators module, not re-exported into it."""
    new, names = MOVED[flat]
    new_module = importlib.import_module(new)
    for name in names:
        owner = getattr(getattr(new_module, name), "__module__", None)
        assert owner == new, f"{new}.{name} is defined in {owner}"


def test_solvers_still_re_exports_the_operator_names():
    """`MCEq.solvers` is the path callers outside the package use."""
    import MCEq.operators.compiled as compiled
    import MCEq.solvers as solvers

    for name in (
        "CompiledOperator",
        "compile_operator",
        "secant_layout",
        "split_diagonal",
    ):
        assert name in solvers.__all__
        assert getattr(solvers, name) is getattr(compiled, name)


def test_importing_the_operators_package_is_free():
    """`MCEq.operators` re-exports nothing, so it pulls in no numerics.

    In a fresh interpreter: by the time a pytest session reaches this file it
    has imported the whole stack, so its own ``sys.modules`` says nothing about
    the cost of the import.
    """
    report = (
        "import sys\n"
        "import MCEq.operators\n"
        "print(sorted(m for m in ('numpy', 'scipy') if m in sys.modules))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", report], capture_output=True, text=True, check=True
    )
    loaded = ast.literal_eval(proc.stdout.strip().splitlines()[-1])
    assert loaded == [], (
        f"importing MCEq.operators imported {loaded}; the package __init__ "
        "re-exports nothing, so a caller pays only for the module it names"
    )
