import pytest
from MCEq import ddm
from MCEq import ddm_utils
from MCEq.particlemanager import _pdata
import numpy as np
import numpy.testing as npt
import unittest


@pytest.fixture(scope="module")
def ddm_fix():
    return ddm.DataDrivenModel(
        e_min=5.0,
        e_max=500.0,
        enable_channels=[(2212, 211)],
        exclude_projectiles=[111, 2112],
        enable_K0_from_isospin=True,
    )


@pytest.fixture(scope="module")
def mceq_qgs():
    import MCEq.core
    from crflux.models import HillasGaisser2012

    return MCEq.core.MCEqRun(
        interaction_model="QGSJETII04",
        theta_deg=0.0,
        primary_model=(HillasGaisser2012, "H3a"),
    )


class TestDDMEntry(unittest.TestCase):
    def setUp(self):
        self.entry = ddm._DDMEntry(
            ebeam=ddm_utils.fmteb(2.0),
            projectile=2212,
            secondary=211,
            x17=False,
            tck=(np.array([1, 2, 3]), np.array([4, 5, 6]), 3),
            cov=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            tv=1.0,
            te=0.1,
            spl_idx=1,
        )

    def test_knot_sigma(self):
        expected_result = np.array([0.1, 0.1, 0.1])
        np.testing.assert_array_equal(self.entry.knot_sigma, expected_result)

    def test_fl_ebeam(self):
        self.assertAlmostEqual(self.entry.fl_ebeam, 2.0)

    def test_n_knots(self):
        self.assertEqual(self.entry.n_knots, 3)

    def test_x_min(self):
        self.assertAlmostEqual(self.entry.x_min, _pdata.mass(211) / 2.0)


class TestDDMChannel(unittest.TestCase):
    def setUp(self):
        self.channel = ddm._DDMChannel(projectile=2212, secondary=211)
        self.channel.add_entry(
            ebeam=ddm_utils.fmteb(2.0),
            projectile=2212,
            secondary=211,
            x17=False,
            tck=(np.array([1, 2, 3]), np.array([4, 5, 6]), 3),
            cov=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            tv=1.0,
            te=1.0,
        )

    def test_entries(self):
        self.assertEqual(len(self.channel.entries), 1)

    def test_total_n_knots(self):
        self.assertEqual(self.channel.total_n_knots, 3)

    def test_n_splines(self):
        self.assertEqual(self.channel.n_splines, 1)

    def test_spline_indices(self):
        self.assertEqual(self.channel.spline_indices, [0])

    def test_get_entry_by_ebeam(self):
        entry = self.channel.get_entry(ebeam=2.0)
        self.assertEqual(entry.projectile, 2212)
        self.assertEqual(entry.secondary, 211)

    def test_get_entry_by_idx(self):
        entry = self.channel.get_entry(idx=0)
        self.assertEqual(entry.projectile, 2212)
        self.assertEqual(entry.secondary, 211)

    def test_get_entry_raises_error(self):
        with self.assertRaises(ValueError):
            self.channel.get_entry(ebeam=3.0)

    def test_str_representation(self):
        expected_output = (
            "\t2212 -> 211:\n\t\t0: ebeam = 2.0 GeV, x17=False, tune v|e=1.000|1.000\n"
        )
        self.assertEqual(str(self.channel), expected_output)


class TestDDMSplineDB(unittest.TestCase):
    def setUp(self):
        self.db = ddm.DDMSplineDB(
            enable_channels=[(2212, 211)],
            exclude_projectiles=[111, 2112],
        )

    def test_clone_entry(self):
        self.db.clone_entry(2212, 211, 158.0, 1000)
        entry = self.db.get_entry(2212, 211, ebeam=1000)
        self.assertEqual(entry.projectile, 2212)
        self.assertEqual(entry.secondary, 211)

    def test_get_spline_indices(self):
        indices = self.db.get_spline_indices(2212, 211)
        self.assertEqual(indices, [0, 1])

    def test_channels(self):
        channels = list(self.db.channels)
        self.assertEqual(len(channels), 1)
        channel = channels[0]
        self.assertEqual(channel.projectile, 2212)
        self.assertEqual(channel.secondary, 211)

    def test_get_entry_by_ebeam(self):
        entry = self.db.get_entry(2212, 211, ebeam=158)
        self.assertEqual(entry.projectile, 2212)
        self.assertEqual(entry.secondary, 211)
        self.assertEqual(entry.ebeam, ddm_utils.fmteb(158.0))

    def test_get_entry_by_idx(self):
        entry = self.db.get_entry(2212, 211, idx=0)
        self.assertEqual(entry.projectile, 2212)
        self.assertEqual(entry.secondary, 211)
        self.assertEqual(entry.ebeam, ddm_utils.fmteb(31.0))

    def test_get_entry_raises_error(self):
        with self.assertRaises(ValueError):
            self.db.get_entry(2212, 211, ebeam=123.0)

    def test_mk_channel(self):
        channel = self.db._mk_channel(2212, 211)
        self.assertEqual(channel, "2212-211")


class TestDataDrivenModel(unittest.TestCase):
    def setUp(self):
        self.model = ddm.DataDrivenModel(
            e_min=5.0,
            e_max=500.0,
            enable_channels=[(2212, 211)],
            exclude_projectiles=[111, 2112],
            enable_K0_from_isospin=True,
        )

    # def test_ddm_matrices(self):
    #     mceq = create_mceq_object()  # Create an MCEq object for testing
    #     matrices = self.model.ddm_matrices(mceq)
    #     self.assertEqual(len(matrices), 1)
    #     self.assertIn((2212, 211), matrices.keys())
    #     self.assertIsInstance(matrices[(2212, 211)], np.ndarray)

    def test_apply_tuning(self):
        self.model.apply_tuning(2212, 211, ebeam=158.0, tv=0.5, te=0.8)
        entry = self.model.spline_db.get_entry(2212, 211, ebeam=158.0)
        self.assertEqual(entry.tv, 0.5)
        self.assertEqual(entry.te, 0.8)

    def test_dn_dxl(self):
        x = np.linspace(0, 1, 100)
        dn_dxl, error = self.model.dn_dxl(x, 2212, 211, 158.0, return_error=True)
        self.assertEqual(len(dn_dxl), len(x))
        self.assertEqual(len(error), len(x))
        self.assertEqual(dn_dxl[0], 0)
        self.assertEqual(error[0], 0)

    def test_repr(self):
        expected_repr = (
            "DDM channels:\n\t2212 -> 211:\n\t\t0: ebeam = 31.0 GeV, x17=False, "
            + "tune v|e=1.000|1.000\n\t\t1: ebeam = 158.0 GeV, x17=False,"
            + " tune v|e=1.000|1.000\n\n"
        )
        self.assertEqual(repr(self.model), expected_repr)


class TestDDMUtils(unittest.TestCase):
    def setUp(self):
        self.model = ddm.DataDrivenModel(
            e_min=5.0,
            e_max=500.0,
            enable_channels=[(2212, 211)],
            exclude_projectiles=[111, 2112],
            enable_K0_from_isospin=True,
        )

    def test_fmteb(self):
        ebeam_float = 2.5
        ebeam_str = "2.5"
        ebeam_int = 3
        formatted_float = ddm_utils.fmteb(ebeam_float)
        formatted_str = ddm_utils.fmteb(ebeam_str)
        formatted_int = ddm_utils.fmteb(ebeam_int)
        self.assertEqual(formatted_float, "2.5")
        self.assertEqual(formatted_str, "2.5")
        self.assertEqual(formatted_int, "3.0")

    def test__spline_min_max_at_knot(self):
        tck = (
            np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            3,
        )
        sigma = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        iknot = 2
        tck_min, tck_max, h = ddm_utils._spline_min_max_at_knot(tck, iknot, sigma)
        expected_tck_min = (np.array([0, 1, 2, 3, 4]), np.array([1, 2, 2.7, 4, 5]), 3)
        expected_tck_max = (np.array([0, 1, 2, 3, 4]), np.array([1, 2, 3.3, 4, 5]), 3)
        expected_h = 0.3
        self.assertTrue(np.array_equal(tck_min[0], expected_tck_min[0]))
        self.assertTrue(np.array_equal(tck_min[1], expected_tck_min[1]))
        self.assertEqual(tck_min[2], expected_tck_min[2])
        self.assertTrue(np.array_equal(tck_max[0], expected_tck_max[0]))
        self.assertTrue(np.array_equal(tck_max[1], expected_tck_max[1]))
        self.assertEqual(tck_max[2], expected_tck_max[2])
        self.assertEqual(h, expected_h)

    def test_generate_DDM_matrix(self):
        import MCEq.core
        from crflux.models import HillasGaisser2012

        mceq_qgs = MCEq.core.MCEqRun(
            interaction_model="QGSJETII04",
            theta_deg=0.0,
            primary_model=(HillasGaisser2012, "H3a"),
        )

        channel = self.model.find_channel(2212, 211)

        generated_matrix = ddm_utils._generate_DDM_matrix(
            channel, mceq_qgs, e_min=20, e_max=50, average=True
        )
        expected_matrix = np.array(
            [
                [
                    0.0,
                    0.000983217365563208,
                    0.006211704918804481,
                    0.026537833076977973,
                    0.0656015917857339,
                    0.10123265537019738,
                ],
                [
                    0.0,
                    7.638437045704317e-05,
                    0.000991419546926868,
                    0.006507972292888666,
                    0.02629565697073082,
                    0.04999655619749746,
                ],
                [
                    0.0,
                    0.0,
                    7.702158302242417e-05,
                    0.0011836492400988143,
                    0.006900314243808511,
                    0.0217230936979197,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.00012213941307792056,
                    0.0014504806061470543,
                    0.007357451148747356,
                ],
                [0.0, 0.0, 0.0, 0.0, 0.00018539511358273218, 0.0010888189683184015],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.4570881508361719e-06],
            ]
        )

        npt.assert_allclose(generated_matrix, expected_matrix)


def test_eval_spline():
    x = np.linspace(0.1, 0.99, 10)
    tck = (
        np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.0]),
        3,
    )
    x17 = True
    cov = np.eye(6)

    # Test without error
    res = ddm_utils._eval_spline(x, tck, x17, cov)
    assert res.shape == (10,)
    assert np.all(res >= 0)

    # Test with error
    res, err = ddm_utils._eval_spline(x, tck, x17, cov, return_error=True)
    assert res.shape == (10,)
    assert err.shape == (10,)
    assert np.all(res >= 0)
    assert np.all(err >= 0)

    # Test with gamma_zfac
    gamma_zfac = 0.5
    res = ddm_utils._eval_spline(x, tck, x17, cov, gamma_zfac=gamma_zfac)
    assert res.shape == (10,)
    assert np.all(res >= 0)


def test_gen_dndx(ddm_fix):
    xbins = np.linspace(0.1, 1, 11)
    entry = ddm_fix.spline_db.get_entry(2212, 211, 31)
    dndx = ddm_utils._gen_dndx(xbins, entry)

    # Check the shape of the output
    assert len(dndx) == len(xbins) - 1

    # Check that dndx values are non-negative
    assert np.all(dndx >= 0)

    # Check that dndx values are zero where x < entry.x_min
    assert np.all(dndx[xbins[:-1] < entry.x_min] == 0)


def test_gen_averaged_dndx(ddm_fix):
    xbins = np.linspace(0.1, 1, 11)
    entry = ddm_fix.spline_db.get_entry(2212, 211, 31)
    averaged_dndx = ddm_utils._gen_averaged_dndx(xbins, entry)

    # Check the shape of the output
    assert averaged_dndx.shape == xbins[:-1].shape

    # Check that averaged dndx values are non-negative
    assert np.all(averaged_dndx >= 0)

    # Check that averaged dndx values are zero where x < entry.x_min
    assert np.all(averaged_dndx[xbins[:-1] < entry.x_min] == 0)


def test_calc_zfactor_and_error(ddm_fix):
    projectile = 2212
    secondary = 211
    ebeam = 158.0

    z_factor, z_error = ddm_utils.calc_zfactor_and_error(
        ddm_fix, projectile, secondary, ebeam
    )

    # Check that Z-factor values are non-negative
    assert np.all(z_factor >= 0)

    # Check zfactor result
    assert np.allclose(0.0485328868, z_factor)

    # Check zerror result
    assert np.allclose(0.0081361078, z_error)


def test_calc_zfactor_and_error2(ddm_fix):
    projectile = 2212
    secondary = 211
    ebeam = 158.0

    z_factor, z_error = ddm_utils.calc_zfactor_and_error2(
        ddm_fix, projectile, secondary, ebeam
    )

    # Check that Z-factor values are non-negative
    assert np.all(z_factor >= 0)

    # Check zfactor result
    assert np.allclose(0.0590098529, z_factor)

    # Check zerror result
    assert np.allclose(0.010476966, z_error)


# def test_gen_matrix_variations(ddm_fix, mceq_qgs):
#     from unittest.mock import MagicMock

#     # Mock the necessary objects and methods
#     model = MagicMock()
#     # model.e_min = ddm_fix.e_min
#     # model.e_max = ddm_fix.e_max
#     model.ddm_matrices = MagicMock(
#         return_value={
#             (2212, 2112): np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
#             (2212, 321): np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]]),
#         }
#     )
#     channel1 = MagicMock()
#     channel1.projectile = 2212
#     channel1.secondary = 2112
#     channel1.entries = [
#         MagicMock(
#             tck=([1, 2, 3], [4, 5, 6], 3),
#             knot_sigma=[0.1, 0.2, 0.3],
#             n_knots=3,
#             fl_ebeam=50.0,
#         ),
#         MagicMock(
#             tck=([7, 8, 9], [10, 11, 12], 3),
#             knot_sigma=[0.4, 0.5, 0.6],
#             n_knots=3,
#             fl_ebeam=100.0,
#         ),
#     ]
#     channel1.n_splines = 2
#     channel2 = MagicMock()
#     channel2.projectile = 2212
#     channel2.secondary = 321
#     channel2.entries = [
#         MagicMock(
#             tck=([13, 14, 15], [16, 17, 18], 3),
#             knot_sigma=[0.7, 0.8, 0.9],
#             n_knots=3,
#             fl_ebeam=50.0,
#         ),
#         MagicMock(
#             tck=([19, 20, 21], [22, 23, 24], 3),
#             knot_sigma=[1.0, 1.1, 1.2],
#             n_knots=3,
#             fl_ebeam=100.0,
#         ),
#     ]
#     channel2.n_splines = 2
#     model.spline_db.channels = [channel1, channel2]
    
#     class MyMockClass:
#    ...:     def __init__(self):
#    ...:         self.my_list = [1, 2, 3, 4, 5]
#    ...: 
#    ...:     def get_list_entry(self, index):
#    ...:         return self.my_list[index]


#     def get_entry(self, idx):
#         return self.entries[idx]

#     for ch in model.spline_db.channels:
#         ch.get_entry = MagicMock(side_effect=get_entry)

#     # Call the function
#     matrix_variations, isospin_partners = ddm_utils.gen_matrix_variations(
#         model, mceq_qgs
#     )

#     # Assert the results
#     assert len(matrix_variations) == 2
#     assert len(isospin_partners) == 1

#     assert matrix_variations[(2212, 2112)][0][0] == [
#         np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
#         np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
#         0.1,
#     ]
#     assert matrix_variations[(2212, 2112)][0][1] == [
#         np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
#         np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
#         0.1,
#     ]
#     assert matrix_variations[(2212, 2112)][1][0] == [
#         np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]]),
#         np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]]),
#         0.4,
#     ]
#     assert matrix_variations[(2212, 2112)][1][1] == [
#         np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]]),
#         np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]]),
#         0.4,
#     ]
#     assert matrix_variations[(2212, 321)][0][0] == [
#         np.array([[13, 14, 15], [16, 17, 18], [19, 20, 21]]),
#         np.array([[13, 14, 15], [16, 17, 18], [19, 20, 21]]),
#         0.7,
#     ]
#     assert matrix_variations[(2212, 321)][0][1] == [
#         np.array([[13, 14, 15], [16, 17, 18], [19, 20, 21]]),
#         np.array([[13, 14, 15], [16, 17, 18], [19, 20, 21]]),
#         0.7,
#     ]
#     assert matrix_variations[(2212, 321)][1][0] == [
#         np.array([[22, 23, 24], [25, 26, 27], [28, 29, 30]]),
#         np.array([[22, 23, 24], [25, 26, 27], [28, 29, 30]]),
#         1.0,
#     ]
#     assert matrix_variations[(2212, 321)][1][1] == [
#         np.array([[22, 23, 24], [25, 26, 27], [28, 29, 30]]),
#         np.array([[22, 23, 24], [25, 26, 27], [28, 29, 30]]),
#         1.0,
#     ]

#     assert isospin_partners[(2212, 321)][0] == (310, 130)
#     assert isospin_partners[(2212, 321)][1][0][0] == [
#         np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]]),
#         np.array([[221, 222, 223], [224, 225, 226], [227, 228, 229]]),
#         0.7,
#     ]
#     assert isospin_partners[(2212, 321)][1][0][1] == [
#         np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]]),
#         np.array([[221, 222, 223], [224, 225, 226], [227, 228, 229]]),
#         0.7,
#     ]
#     assert isospin_partners[(2212, 321)][1][1][0] == [
#         np.array([[22, 23, 24], [25, 26, 27], [28, 29, 30]]),
#         np.array([[222, 223, 224], [225, 226, 227], [228, 229, 230]]),
#         1.0,
#     ]
#     assert isospin_partners[(2212, 321)][1][1][1] == [
#         np.array([[22, 23, 24], [25, 26, 27], [28, 29, 30]]),
#         np.array([[222, 223, 224], [225, 226, 227], [228, 229, 230]]),
#         1.0,
#     ]
