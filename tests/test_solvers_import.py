"""`import MCEq.solvers` stays cheap on a machine with no accelerator.

The package pulls in all four backend modules through
``MCEq.solvers.etd2._BACKENDS``, so a dlopen, a cupy import or a hardware probe
at the top of any one of them would be paid by every user -- including one who
only ever runs the scipy backend, and one on a machine where the library is not
installed at all. Each backend defers that work into the constructor or the
function that needs it. This is the gate that keeps it there.

The check runs in a fresh interpreter: by the time a pytest session reaches
this file it has already imported MCEq, and quite possibly cupy, so nothing
about the parent process's ``sys.modules`` says anything about the cost of the
import.
"""

import ast
import subprocess
import sys
import textwrap

#: Lazily-resolved ``MCEq.config`` attributes that probe the platform. The
#: module's ``__getattr__`` caches each answer as a real module attribute, so a
#: name present in ``config.__dict__`` is one whose probe has run.
PROBES = ("has_mkl", "has_cuda", "has_accelerate", "mkl_path", "kernel_config")

#: Shared objects no backend may pull into the address space at import time.
ACCELERATOR_LIBS = ("libmkl", "libcudart", "libcublas", "libnvrtc", "libcusparse")

REPORT = textwrap.dedent(
    """
    import sys

    import MCEq.solvers  # noqa: F401

    from MCEq import config

    try:
        with open("/proc/self/maps") as fh:
            maps = fh.read()
    except OSError:  # not Linux; the sys.modules checks still apply
        maps = ""

    print(
        repr(
            {{
                "modules": sorted(
                    m for m in ("cupy", "MCEq.spacc", "MCEq.etd2_kernels")
                    if m in sys.modules
                ),
                "mkl_handle": config.mkl is not None,
                "probes": sorted(n for n in {probes!r} if n in config.__dict__),
                "libs": sorted(
                    {{lib for lib in {libs!r} if lib in maps}}
                ),
            }}
        )
    )
    """
).format(probes=PROBES, libs=ACCELERATOR_LIBS)


def test_importing_solvers_touches_no_accelerator():
    proc = subprocess.run(
        [sys.executable, "-c", REPORT],
        capture_output=True,
        text=True,
        check=True,
    )
    report = ast.literal_eval(proc.stdout.strip().splitlines()[-1])

    assert report["modules"] == [], (
        f"importing MCEq.solvers imported {report['modules']}; a backend module "
        "must not import its library at module level"
    )
    assert not report["mkl_handle"], (
        "importing MCEq.solvers loaded libmkl_rt; the handle belongs to the "
        "first MklSparseMatrix, not to the import"
    )
    assert report["probes"] == [], (
        f"importing MCEq.solvers resolved {report['probes']}; a backend module "
        "must not probe the platform at module level"
    )
    assert report["libs"] == [], (
        f"importing MCEq.solvers mapped {report['libs']} into the process"
    )
