import numpy as np

# CORSIKA-style atmosphere parameters
# Data storage for atmosphere parameters
# Each key is a tuple (location, season), and value is a dict of parameters.
_cosika_atmosphere_params = {
    ("USStd", None): {
        "_aatm": np.array([-186.5562, -94.919, 0.61289, 0.0, 0.01128292]),
        "_batm": np.array([1222.6562, 1144.9069, 1305.5948, 540.1778, 1.0]),
        "_catm": np.array([994186.38, 878153.55, 636143.04, 772170.0, 1.0e9]),
        "_thickl": np.array([1036.102549, 631.100309, 271.700230, 3.039494, 0.001280]),
        "_hlay": np.array([0.0, 4.0e5, 1.0e6, 4.0e6, 1.0e7]),
    },
    ("BK_USStd", None): {
        "_aatm": np.array(
            [-149.801663, -57.932486, 0.63631894, 4.3545369e-4, 0.01128292]
        ),
        "_batm": np.array([1183.6071, 1143.0425, 1322.9748, 655.69307, 1.0]),
        "_catm": np.array([954248.34, 800005.34, 629568.93, 737521.77, 1.0e9]),
        "_thickl": np.array([1033.804941, 418.557770, 216.981635, 4.344861, 0.001280]),
        "_hlay": np.array([0.0, 7.0e5, 1.14e6, 3.7e6, 1.0e7]),
    },
    ("Karlsruhe", None): {
        "_aatm": np.array([-118.1277, -154.258, 0.4191499, 5.4094056e-4, 0.01128292]),
        "_batm": np.array([1173.9861, 1205.7625, 1386.7807, 555.8935, 1.0]),
        "_catm": np.array([919546.0, 963267.92, 614315.0, 739059.6, 1.0e9]),
        "_thickl": np.array([1055.858707, 641.755364, 272.720974, 2.480633, 0.001280]),
        "_hlay": np.array([0.0, 4.0e5, 1.0e6, 4.0e6, 1.0e7]),
    },
    ("KM3NeT", None): {  # Averaged over detector and season
        "_aatm": np.array(
            [
                -141.31449999999998,
                -8.256029999999999,
                0.6132505,
                -0.025998975,
                0.4024275,
            ]
        ),
        "_batm": np.array(
            [1153.0349999999999, 1263.3325, 1257.0724999999998, 404.85974999999996, 1.0]
        ),
        "_catm": np.array([967990.75, 668591.75, 636790.0, 814070.75, 21426175.0]),
        "_thickl": np.array(
            [
                1011.8521512499999,
                275.84507575000003,
                51.0230705,
                2.983134,
                0.21927724999999998,
            ]
        ),
        "_hlay": np.array([0.0, 993750.0, 2081250.0, 4150000.0, 6877500.0]),
    },
    ("ANTARES/KM3NeT-ORCA", "Summer"): {
        "_aatm": np.array([-158.85, -5.38682, 0.889893, -0.0286665, 0.50035]),
        "_batm": np.array([1145.62, 1176.79, 1248.92, 415.543, 1.0]),
        "_catm": np.array([998469.0, 677398.0, 636790.0, 823489.0, 16090500.0]),
        "_thickl": np.array([986.951713, 306.4668, 40.546793, 4.288721, 0.277182]),
        "_hlay": np.array([0, 9.0e5, 22.0e5, 38.0e5, 68.2e5]),
    },
    ("ANTARES/KM3NeT-ORCA", "Winter"): {
        "_aatm": np.array([-132.16, -2.4787, 0.298031, -0.0220264, 0.348021]),
        "_batm": np.array([1120.45, 1203.97, 1163.28, 360.027, 1.0]),
        "_catm": np.array([933697.0, 643957.0, 636790.0, 804486.0, 23109000.0]),
        "_thickl": np.array([988.431172, 273.033464, 37.185105, 1.162987, 0.192998]),
        "_hlay": np.array([0, 9.5e5, 22.0e5, 47.0e5, 68.2e5]),
    },
    ("KM3NeT-ARCA", "Summer"): {
        "_aatm": np.array([-157.857, -28.7524, 0.790275, -0.0286999, 0.481114]),
        "_batm": np.array([1190.44, 1171.0, 1344.78, 445.357, 1.0]),
        "_catm": np.array([1006100.0, 758614.0, 636790.0, 817384.0, 16886800.0]),
        "_thickl": np.array([1032.679434, 328.978681, 80.601135, 4.420745, 0.264112]),
        "_hlay": np.array([0, 9.0e5, 18.0e5, 38.0e5, 68.2e5]),
    },
    ("KM3NeT-ARCA", "Winter"): {
        "_aatm": np.array([-116.391, 3.5938, 0.474803, -0.0246031, 0.280225]),
        "_batm": np.array([1155.63, 1501.57, 1271.31, 398.512, 1.0]),
        "_catm": np.array([933697.0, 594398.0, 636790.0, 810924.0, 29618400.0]),
        "_thickl": np.array([1039.346286, 194.901358, 45.759249, 2.060083, 0.142817]),
        "_hlay": np.array([0, 12.25e5, 21.25e5, 43.0e5, 70.5e5]),
    },
    ("SouthPole", "December"): {  # MSIS-90-E for Dec
        "_aatm": np.array([-128.601, -39.5548, 1.13088, -0.00264960, 0.00192534]),
        "_batm": np.array([1139.99, 1073.82, 1052.96, 492.503, 1.0]),
        "_catm": np.array([861913.0, 744955.0, 675928.0, 829627.0, 5.8587010e9]),
        "_thickl": np.array([1011.398804, 588.128367, 240.955360, 3.964546, 0.000218]),
        "_hlay": np.array([0.0, 4.0e5, 1.0e6, 4.0e6, 1.0e7]),
    },
    ("SouthPole", "June"): {  # MSIS-90-E for June
        "_aatm": np.array([-163.331, -65.3713, 0.402903, -0.000479198, 0.00188667]),
        "_batm": np.array([1183.70, 1108.06, 1424.02, 207.595, 1.0]),
        "_catm": np.array([875221.0, 753213.0, 545846.0, 793043.0, 5.9787908e9]),
        "_thickl": np.array([1020.370363, 586.143464, 228.374393, 1.338258, 0.000214]),
        "_hlay": np.array([0.0, 4.0e5, 1.0e6, 4.0e6, 1.0e7]),
    },
    ("PL_SouthPole", "January"): {  # P. Lipari\'s Jan
        "_aatm": np.array([-113.139, -7.930635e06, -54.3888, -0.0, 0.00421033]),
        "_batm": np.array([1133.10, 1101.20, 1085.00, 1098.00, 1.0]),
        "_catm": np.array([861730.0, 826340.0, 790950.0, 682800.0, 2.6798156e9]),
        "_thickl": np.array(
            [1019.966898, 718.071682, 498.659703, 340.222344, 0.000478]
        ),
        "_hlay": np.array([0.0, 2.67e5, 5.33e5, 8.0e5, 1.0e7]),
    },
    ("PL_SouthPole", "August"): {  # P. Lipari\'s Aug
        "_aatm": np.array([-59.0293, -21.5794, -7.14839, 0.0, 0.000190175]),
        "_batm": np.array([1079.0, 1071.90, 1182.0, 1647.1, 1.0]),
        "_catm": np.array([764170.0, 699910.0, 635650.0, 551010.0, 59.329575e9]),
        "_thickl": np.array([1019.946057, 391.739652, 138.023515, 43.687992, 0.000022]),
        "_hlay": np.array([0.0, 6.67e5, 13.33e5, 2.0e6, 1.0e7]),
    },
    ("SDR_SouthPole", "January"): {
        "_aatm": np.array([-91.6956, 7.01491, 0.505452, -0.00181302, 0.00207722]),
        "_batm": np.array([1125.71, 1149.81, 1032.68, 490.789, 1.0]),
        "_catm": np.array([821621.0, 635444.0, 682968.0, 807327.0, 5.4303203e9]),
        "_thickl": np.array([1034.012527, 343.944792, 94.067894, 3.291329, 0.000236]),
        "_hlay": np.array([0.0, 7.8e5, 1.64e6, 4.04e6, 1.0e7]),
    },
    ("SDR_SouthPole", "February"): {
        "_aatm": np.array([-72.1988, 22.7002, 0.430171, -0.0012030, 0.00207722]),
        "_batm": np.array([1108.19, 1159.77, 1079.25, 523.956, 1.0]),
        "_catm": np.array([786271.0, 599986.0, 667432.0, 780919.0, 5.4303203e9]),
        "_thickl": np.array([1035.987489, 328.422968, 220.918411, 2.967188, 0.000236]),
        "_hlay": np.array([0.0, 8.0e5, 1.06e6, 4.04e6, 1.0e7]),
    },
    ("SDR_SouthPole", "March"): {
        "_aatm": np.array([-63.7290, -1.02799, 0.324414, -0.000490772, 0.00207722]),
        "_batm": np.array([1102.66, 1093.56, 1198.93, 589.827, 1.0]),
        "_catm": np.array([764831.0, 660389.0, 636118.0, 734909.0, 5.4303203e9]),
        "_thickl": np.array([1038.92306, 395.458197, 35.764007, 2.416571, 0.000236]),
        "_hlay": np.array([0.0, 6.7e5, 2.24e6, 4.04e6, 1.0e7]),
    },
    ("SDR_SouthPole", "April"): {
        "_aatm": np.array([-69.7259, -2.79781, 0.262692, -0.0000841695, 0.00207722]),
        "_batm": np.array([1111.70, 1128.64, 1413.98, 587.688, 1.0]),
        "_catm": np.array([766099.0, 641716.0, 588082.0, 693300.0, 5.4303203e9]),
        "_thickl": np.array([1041.972041, 342.512757, 33.817874, 1.731426, 0.000236]),
        "_hlay": np.array([0.0, 7.6e5, 2.2e6, 4.04e6, 1.0e7]),
    },
    ("SDR_SouthPole", "May"): {
        "_aatm": np.array([-78.5551, -5.33239, 0.312889, -0.0000920472, 0.00152236]),
        "_batm": np.array([1118.46, 1169.09, 1577.71, 452.177, 1.0]),
        "_catm": np.array([776648.0, 626683.0, 553087.0, 696835.0, 7.4095699e9]),
        "_thickl": np.array([1039.912752, 300.670386, 42.734726, 1.517136, 0.000173]),
        "_hlay": np.array([0.0, 8.4e5, 2.0e6, 3.97e6, 1.0e7]),
    },
    ("SDR_SouthPole", "June"): {
        "_aatm": np.array([-92.6125, -8.56450, 0.363986, 0.00207722, 0.00152236]),
        "_batm": np.array([1129.88, 1191.98, 1619.82, 411.586, 1.0]),
        "_catm": np.array([791177.0, 618840.0, 535235.0, 692253.0, 5.4303203e9]),
        "_thickl": np.array([1037.258545, 293.258798, 57.517838, 1.604677, 0.000236]),
        "_hlay": np.array([0.0, 8.5e5, 1.79e6, 3.84e6, 1.0e7]),
    },
    ("SDR_SouthPole", "July"): {
        "_aatm": np.array([-89.9639, -13.9697, 0.441631, -0.0000146525, 0.00207722]),
        "_batm": np.array([1125.73, 1180.47, 1581.43, 373.796, 1.0]),
        "_catm": np.array([784553.0, 628042.0, 531652.0, 703417.0, 5.4303203e9]),
        "_thickl": np.array([1035.760962, 291.018963, 79.913857, 1.808649, 0.000236]),
        "_hlay": np.array([0.0, 8.5e5, 1.59e6, 3.75e6, 1.0e7]),
    },
    ("SDR_SouthPole", "August"): {
        "_aatm": np.array([-90.4253, -18.7154, 0.513930, -0.00021565, 0.00152336]),
        "_batm": np.array([1125.01, 1175.60, 1518.03, 299.006, 1.0]),
        "_catm": np.array([781628.0, 633793.0, 533269.0, 737794.0, 7.4095699e9]),
        "_thickl": np.array([1034.576882, 288.770552, 102.500845, 1.854518, 0.000173]),
        "_hlay": np.array([0.0, 8.5e5, 1.44e6, 3.75e6, 1.0e7]),
    },
    ("SDR_SouthPole", "September"): {
        "_aatm": np.array([-91.6860, -23.3519, 0.891302, -0.000765666, 0.00207722]),
        "_batm": np.array([1125.53, 1169.77, 1431.26, 247.030, 1.0]),
        "_catm": np.array([786017.0, 645241.0, 545022.0, 805419.0, 5.4303203e9]),
        "_thickl": np.array([1033.830437, 289.992106, 132.664890, 2.758468, 0.000236]),
        "_hlay": np.array([0.0, 8.5e5, 1.3e6, 3.62e6, 1.0e7]),
    },
    ("SDR_SouthPole", "October"): {
        "_aatm": np.array([451.616, -85.5456, 2.06082, -0.0010760, 0.00207722]),
        "_batm": np.array([849.239, 1113.16, 1322.28, 372.242, 1.0]),
        "_catm": np.array([225286.0, 789340.0, 566132.0, 796434.0, 5.4303203e9]),
        "_thickl": np.array([1300.861796, 666.125666, 224.149961, 7.129645, 0.000236]),
        "_hlay": np.array([0.0, 3.1e5, 1.01e6, 3.15e6, 1.0e7]),
    },
    ("SDR_SouthPole", "November"): {
        "_aatm": np.array([-152.853, 4.22741, 1.38352, -0.00115014, 0.00207722]),
        "_batm": np.array([1174.09, 1272.49, 975.906, 481.615, 1.0]),
        "_catm": np.array([891602.0, 582119.0, 643130.0, 783786.0, 5.4303203e9]),
        "_thickl": np.array([1021.228204, 299.692876, 31.360220, 7.715379, 0.000236]),
        "_hlay": np.array([0.0, 8.5e5, 2.24e6, 3.24e6, 1.0e7]),
    },
    ("SDR_SouthPole", "December"): {
        "_aatm": np.array([-100.386, 5.43849, 0.399465, -0.00175472, 0.00207722]),
        "_batm": np.array([1128.71, 1198.10, 858.522, 480.142, 1.0]),
        "_catm": np.array([829352.0, 612649.0, 706104.0, 806875.0, 5.4303203e9]),
        "_thickl": np.array([1028.319337, 304.628519, 38.473287, 3.210929, 0.000236]),
        "_hlay": np.array([0.0, 8.5e5, 2.2e6, 4.04e6, 1.0e7]),
    },
}


_corsika_reference_table = """\
+---------------------+-------------------+------------------------------+
| location            |   CORSIKA MODATM  | Description/season           |
+=====================+===================+==============================+
| "USStd"             |          1        |  US Standard atmosphere      |
+---------------------+-------------------+------------------------------+
| "BK_USStd"          |         17        |  Bianca Keilhauer's USStd    |
+---------------------+-------------------+------------------------------+
| "Karlsruhe"         |          2        |  AT115 / Karlsruhe           |
+---------------------+-------------------+------------------------------+
| "SouthPole"         |     14 and 12     |  MSIS-90-E for Dec and June  |
+---------------------+-------------------+------------------------------+
|"PL_SouthPole"       |     15 and 16     |  P. Lipari's  Jan and Aug    |
+---------------------+-------------------+------------------------------+
|"SDR_SouthPole"      |      30 to 41     | S. De Ridder, every month    |
+---------------------+-------------------+------------------------------+
|"ANTARES/KM3NeT-ORCA"|         NA        |  PhD T. Heid                 |
+---------------------+-------------------+------------------------------+
| "KM3NeT-ARCA"       |         NA        |  PhD T. Heid                 |
+---------------------+-------------------+------------------------------+
"""

# NRLMSISE-00 related parameters
MONTH_TO_DAY_OF_YEAR = {
    "January": 1,
    "February": 32,
    "March": 60,
    "April": 91,
    "May": 121,
    "June": 152,
    "July": 182,
    "August": 213,
    "September": 244,
    "October": 274,
    "November": 305,
    "December": 335,
}

# Longitude, latitude, height (in cm)
LOCATIONS = {
    "SouthPole": (0.0, -90.0, 2834.0 * 100.0),
    "Karlsruhe": (8.4, 49.0, 110.0 * 100.0),
    "Geneva": (6.1, 46.2, 370.0 * 100.0),
    "Tokyo": (139.0, 35.0, 5.0 * 100.0),
    "SanGrasso": (13.5, 42.4, 5.0 * 100.0),
    "TelAviv": (34.8, 32.1, 5.0 * 100.0),
    "KSC": (-80.7, 32.1, 5.0 * 100.0),  # Kennedy Space Center
    "SoudanMine": (-92.2, 47.8, 5.0 * 100.0),
    "Tsukuba": (140.1, 36.2, 30.0 * 100.0),
    "LynnLake": (-101.1, 56.9, 360.0 * 100.0),
    "PeaceRiver": (-117.2, 56.15, 36000.0 * 100.0),
    "FtSumner": (-104.2, 34.5, 31000.0 * 100.0),
}

DAY_TIMES_SEC = {
    "day": 43200.0,  # 12:00 PM in seconds from midnight
    "night": 0.0,  # Midnight in seconds
}

# Default geophysical parameters for NRLMSISE-00
DEFAULT_F107A = 150.0
DEFAULT_F107 = 150.0
DEFAULT_AP = 4.0


def list_available_corsika_atmospheres(
    print_reference_table=False, format_output=False
):  # Renamed from list_available_corsika_atmospheres
    """Returns a list of available (location, season) tuples for CORSIKA-style models.

    Args:
      print_reference_table (bool): If True, prints the reference table
                                      for CORSIKA-style atmospheres.
    """
    if print_reference_table:
        print("Available CORSIKA-style atmosphere models and their references:")
        print(_corsika_reference_table)
    if format_output:
        print("Formatted output:")
        # Collect which seasons are available for each location and print
        # formatted output with seasons list listed with one tab in each line
        locations = {}
        for location, season in _cosika_atmosphere_params:
            if location not in locations:
                locations[location] = []
            if season is not None and season not in locations[location]:
                locations[location].append(season)
        for location, seasons in locations.items():
            print(f"{location}:")
            for season in seasons:
                print(f"\t{season}")

    return list(_cosika_atmosphere_params.keys())


def get_atmosphere_parameters(
    location, season
):  # Renamed from get_corsika_atmosphere_parameters
    (
        """
    Returns the atmospheric parameters for a given location and season for CORSIKA-style models.
    Parameters are based on CORSIKA parameterizations.

    Reference Table:
    """
        + _corsika_reference_table
        + """
    Args:
      location (str): The location identifier.
      season (str or None): The season identifier.

    Returns:
      tuple: (_aatm, _batm, _catm, _thickl, _hlay) numpy arrays.

    Raises:
      KeyError: if the (location, season) combination is not found.
    """
    )
    params = _cosika_atmosphere_params.get((location, season))
    if params is None:
        raise KeyError(
            f"CORSIKA atmosphere parameters not found for location='{location}', season='{season}'"
        )
    return (
        params["_aatm"],
        params["_batm"],
        params["_catm"],
        params["_thickl"],
        params["_hlay"],
    )


def get_nrlmsise00_defaults():
    """Returns the default geophysical parameters for NRLMSISE-00."""
    return DEFAULT_F107A, DEFAULT_F107, DEFAULT_AP


def get_location_data(location_name):
    """Returns longitude, latitude, and height for a given location name."""
    return LOCATIONS.get(location_name)


def get_month_day_of_year(month_name):
    """Returns the day of the year for a given month name."""
    return MONTH_TO_DAY_OF_YEAR.get(month_name)


def get_day_time_seconds(day_time_name):
    """Returns the time in seconds from midnight for a given day time name (e.g., 'day', 'night')."""
    return DAY_TIMES_SEC.get(day_time_name)
