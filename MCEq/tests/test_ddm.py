import pytest
from MCEq import ddm
from MCEq import ddm_utils
from MCEq.particlemanager import _pdata
import numpy as np
import unittest


@pytest.fixture(scope="module")
def ddm_obs():
    return ddm.DataDrivenModel()


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
