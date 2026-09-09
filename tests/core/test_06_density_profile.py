"""Density-model switching across every supported atmosphere (the xfail set included).

Split out of ``tests/test_core.py`` at M14 (TC-release); provenance in the
commit body. Fixture ``mceq_sib21`` comes from ``tests/conftest.py``.
"""

import sys

import pytest

if sys.platform.startswith("win") and sys.maxsize <= 2**32:
    pytest.skip("Skip model test on 32-bit Windows.", allow_module_level=True)

import MCEq.geometry.density_profiles as dprof
from MCEq.environment.parameters import list_available_corsika_atmospheres

msis_atmospheres = [
    "SouthPole",
    "Karlsruhe",
    "Geneva",
    "Tokyo",
    "GranSasso",
    "TelAviv",
    "KSC",
    "SoudanMine",
    "Tsukuba",
    "LynnLake",
    "PeaceRiver",
    "FtSumner",
]

season = ["January", "July"]

corsika_atmospheres = list_available_corsika_atmospheres()


test_densities_cases = []
# MSIS00: all atmospheres
for atmo in msis_atmospheres:
    for s in ["January", "July"]:
        test_densities_cases.append(
            pytest.param("MSIS00", (atmo, s), id=f"MSIS00-{atmo}-{s}")
        )

# MSIS00_IC and AIRS: only SouthPole
for model in ["MSIS00_IC", "AIRS"]:
    for s in ["January", "July"]:
        if model == "AIRS":
            test_densities_cases.append(
                pytest.param(
                    model,
                    ("SouthPole", s),
                    id=f"{model}-SouthPole-{s}",
                    marks=pytest.mark.xfail(reason="Fix issure #71"),
                )
            )
        else:
            test_densities_cases.append(
                pytest.param(model, ("SouthPole", s), id=f"{model}-SouthPole-{s}")
            )

for density_config in corsika_atmospheres:
    test_densities_cases.append(
        pytest.param(
            "CORSIKA",
            density_config,
            id=f"CORSIKA-{density_config}",
        )
    )

test_densities_cases.append(pytest.param("Isothermal", (None, None), id="Isothermal"))
test_densities_cases.append(
    pytest.param("GeneralizedTarget", (), id="GeneralizedTarget")
)
test_densities_cases.append(
    pytest.param(
        "Unknow",
        (),
        id="Unknown",
        marks=pytest.mark.xfail(reason="Fails for uknown density model"),
    )
)

profiles = {
    "MSIS00": dprof.MSIS00Atmosphere,
    "MSIS00_IC": dprof.MSIS00IceCubeCentered,
    "CORSIKA": dprof.CorsikaAtmosphere,
    "AIRS": dprof.AIRSAtmosphere,
    "Isothermal": dprof.IsothermalAtmosphere,
    "GeneralizedTarget": dprof.GeneralizedTarget,
}


@pytest.mark.parametrize("model, density_config", test_densities_cases)
def test_set_density_profile(mceq_sib21, model, density_config):
    try:
        mceq_sib21.set_density_model((model, density_config))
        mceq_sib21.solve()

        # test instances instead of str
        profile = profiles[model](*density_config)
        mceq_sib21.set_density_model(profile)
        mceq_sib21.solve()
    finally:
        # Restore the session-fixture default so later tests can still call
        # set_theta_deg / use the atmospheric path. GeneralizedTarget in
        # particular rejects angle changes and would poison the shared fixture.
        mceq_sib21.set_density_model(("CORSIKA", ("BK_USStd", None)))
