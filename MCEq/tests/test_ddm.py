import pytest
from MCEq import ddm


@pytest.fixture(scope="module")
def ddm_obs():
    return ddm.DataDrivenModel()
