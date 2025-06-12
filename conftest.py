import crflux.models as pm
from functools import lru_cache
import pytest
import pathlib

from MCEq.core import MCEqRun
from MCEq import config


@pytest.fixture(scope="session")
def mceq():
    config.debug_level = 1
    config.kernel_config = "numpy"
    config.cuda_gpu_id = 0
    if config.has_mkl:
        config.set_mkl_threads(2)

    return MCEqRun(
        interaction_model="SIBYLL23C",
        theta_deg=0.0,
        primary_model=(pm.HillasGaisser2012, "H3a"),
    )


@pytest.fixture
def msis_expected_file(request):
    test_dir = pathlib.Path(request.fspath).parent
    path = test_dir / "msis_expected.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Expected output file {path} not found. "
            "Please run the test with the expected output file."
        )
    return path
