import numpy as np
import pytest

corsika_expected = [
    ("USStd", None, (1036.099233683902, 0.00015623258808300557)),
    ("BK_USStd", None, (1033.8094962133184, 0.00015782685585891685)),
    ("Karlsruhe", None, (1055.861981113731, 0.00016209949387937668)),
    ("ANTARES/KM3NeT-ORCA", "Summer", (986.9593811082788, 0.00015529574727367941)),
    ("ANTARES/KM3NeT-ORCA", "Winter", (988.4293864278521, 0.0001589317236294479)),
    ("KM3NeT-ARCA", "Summer", (1032.7184058861765, 0.00016954131888323744)),
    ("KM3NeT-ARCA", "Winter", (1039.3697214845179, 0.00016202068935405075)),
    ("KM3NeT", None, (1018.1547240905948, 0.0001609490344992944)),
    ("SouthPole", "December", (1011.4568036341923, 0.00014626903051217024)),
    ("SouthPole", "June", (1020.3505579524912, 0.00018246219074986874)),
    ("PL_SouthPole", "January", (1019.974568696789, 0.0001464549375212421)),
    ("PL_SouthPole", "August", (1019.9764946890782, 0.0001685608228906579)),
    ("SDR_SouthPole", "January", (1034.0143423913353, 0.00014632473385006882)),
    ("SDR_SouthPole", "February", (1035.9936617195242, 0.00014918253835319762)),
    ("SDR_SouthPole", "March", (1038.9308255875999, 0.00015298022770783904)),
    ("SDR_SouthPole", "April", (1041.9777521676683, 0.0001601901839449753)),
    ("SDR_SouthPole", "May", (1039.907580402069, 0.00016807635328451188)),
    ("SDR_SouthPole", "June", (1037.2666634880595, 0.00017542702214644906)),
    ("SDR_SouthPole", "July", (1035.7659887947702, 0.00018145735342421507)),
    ("SDR_SouthPole", "August", (1034.5825095745786, 0.0001865567510599077)),
    ("SDR_SouthPole", "September", (1033.8370290916944, 0.00018184168516617518)),
    ("SDR_SouthPole", "October", (1300.9045332765688, 0.0001729949911605945)),
    ("SDR_SouthPole", "November", (1021.2313399252452, 0.00016452181236376558)),
    ("SDR_SouthPole", "December", (1028.3233680860317, 0.00015434578972428948)),
]


# Test that all corsika atmospheres are tested
def test_cka_atm_completeness():
    from MCEq.geometry.atmosphere_parameters import list_available_corsika_atmospheres

    missing = []
    expected_entries = {(loc, season) for loc, season, _ in corsika_expected}
    for loc, season in list_available_corsika_atmospheres():
        if (loc, season) not in expected_entries:
            missing.append((loc, season))
    if missing:
        for i, (loc, season) in enumerate(missing):
            # Create reference data
            from MCEq.geometry.density_profiles import CorsikaAtmosphere

            cka_obj = CorsikaAtmosphere(loc, season)
            ref = (float(cka_obj.max_X), float(1.0 / cka_obj.r_X2rho(100.0)))
            missing[i] = (loc, season, ref)

    assert len(missing) == 0, f"Missing tests for {missing}."


ids = [f"{loc}-{season or 'None'}" for loc, season, _ in corsika_expected]


@pytest.mark.parametrize(("loc", "season", "expected"), corsika_expected, ids=ids)
def test_corsika_atm(loc, season, expected):
    from MCEq.geometry.density_profiles import CorsikaAtmosphere

    cka_obj = CorsikaAtmosphere(loc, season)
    assert np.allclose([cka_obj.max_X, 1.0 / cka_obj.r_X2rho(100.0)], expected)


msis00_expected = [
    ("SouthPole", "January", (1022.6914983678925, 0.00014380042112573175)),
    ("Karlsruhe", "January", (1041.2180457811605, 0.00016046129606232836)),
    ("Geneva", "January", (1044.6608866969684, 0.00016063221634835724)),
    ("Tokyo", "January", (1046.427667371285, 0.00016041531186210874)),
    ("SanGrasso", "January", (1048.6505423154006, 0.00016107650347480857)),
    ("TelAviv", "January", (1050.6431802896034, 0.00016342084740033518)),
    ("KSC", "January", (1050.2145039327452, 0.00016375664772178006)),
    ("SoudanMine", "January", (1033.3640270683418, 0.00015614485659072835)),
    ("Tsukuba", "January", (1045.785578319159, 0.00015970449150213374)),
    ("LynnLake", "January", (1019.9475650272982, 0.000153212909250962)),
    ("PeaceRiver", "January", (1020.3640351872195, 0.00015221038616604717)),
    ("FtSumner", "January", (1047.964376368261, 0.00016218804771381842)),
    ("SouthPole", "July", (1022.1737895082897, 0.00017812023753792838)),
]

ids = [f"{loc}-{season or 'None'}" for loc, season, _ in msis00_expected]


@pytest.mark.parametrize(("loc", "season", "expected"), msis00_expected, ids=ids)
def test_msis_atm(loc, season, expected):
    from MCEq.geometry.density_profiles import MSIS00Atmosphere

    msis_obj = MSIS00Atmosphere(loc, season)
    if expected is None:
        ref = (float(msis_obj.max_X), float(1.0 / msis_obj.r_X2rho(100.0)))
        msg = f"MSIS-00 reference data for {loc} in {season} not available. Creating a new one. {ref}"
        pytest.fail(msg)
    assert np.allclose([msis_obj.max_X, 1.0 / msis_obj.r_X2rho(100.0)], expected)
