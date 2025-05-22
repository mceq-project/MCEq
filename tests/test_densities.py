import numpy as np

def test_corsika_atm():

    from MCEq.geometry.density_profiles import CorsikaAtmosphere

    # Depth at surface and density at X=100 g/cm2
    cka_surf_100 = [
        (1036.099233683902, 0.00015623258808300557),
        (1033.8094962133184, 0.00015782685585891685),
        (1055.861981113731, 0.00016209949387937668),
        (986.9593811082788, 0.00015529574727367941),
        (988.4293864278521, 0.0001589317236294479),
        (1032.7184058861765, 0.00016954131888323744),
        (1039.3697214845179, 0.00016202068935405075),
        (1018.1547240905948, 0.0001609490344992944),
        (1011.4568036341923, 0.00014626903051217024),
        (1019.974568696789, 0.0001464549375212421),
        (1019.9764946890782, 0.0001685608228906579)
    ]
    for iatm, (loc, season) in enumerate([
        ("USStd", None),
        ("BK_USStd", None),
        ("Karlsruhe", None),
        ("ANTARES/KM3NeT-ORCA", 'Summer'),
        ("ANTARES/KM3NeT-ORCA", 'Winter'),
        ("KM3NeT-ARCA", 'Summer'),
        ("KM3NeT-ARCA", 'Winter'),
        ("KM3NeT", None),
        ('SouthPole', 'December'),
        ('PL_SouthPole', 'January'),
        ('PL_SouthPole', 'August'),
    ]):

        cka_obj = CorsikaAtmosphere(loc, season)
        assert np.allclose([cka_obj.max_X, 1. /
                            cka_obj.r_X2rho(100.)], cka_surf_100[iatm])


def test_msis_atm():

    from MCEq.geometry.density_profiles import MSIS00Atmosphere
    msis_surf_100 = [
        (1022.6914983678925, 0.00014380042112573175), (1041.2180457811605, 0.00016046129606232836),
        (1044.6608866969684, 0.00016063221634835724), (1046.427667371285, 0.00016041531186210874),
        (1048.6505423154006, 0.00016107650347480857), (1050.6431802896034, 0.00016342084740033518),
        (1050.2145039327452, 0.00016375664772178006), (1033.3640270683418, 0.00015614485659072835),
        (1045.785578319159, 0.00015970449150213374), (1019.9475650272982, 0.000153212909250962),
        (1020.3640351872195, 0.00015221038616604717), (1047.964376368261, 0.00016218804771381842)
    ]
    for iatm, (loc, season) in enumerate([
        ('SouthPole', "January"),
        ('Karlsruhe', "January"),
        ('Geneva', "January"),
        ('Tokyo', "January"),
        ('SanGrasso', "January"),
        ('TelAviv', "January"),
        ('KSC', "January"),
        ('SoudanMine', "January"),
        ('Tsukuba', "January"),
        ('LynnLake', "January"),
        ('PeaceRiver', "January"),
        ('FtSumner', "January")
    ]):

        msis_obj = MSIS00Atmosphere(loc, season)
        assert np.allclose([msis_obj.max_X, 1. /
                            msis_obj.r_X2rho(100.)], msis_surf_100[iatm])
