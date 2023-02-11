import unittest
import numpy as np
from MCEq.geometry.geometry import EarthGeometry


class TestEarthGeometry(unittest.TestCase):
    def setUp(self):
        self.eg = EarthGeometry()

    def test_check_angles(self):
        # Test exception when zenith angle is above max
        with self.assertRaises(Exception):
            self.eg._check_angles(np.pi / 2 + 0.1)

        # Test that no exception is raised when zenith angle is below max
        self.eg._check_angles(np.pi / 2 - 0.1)

    def test_A_1(self):
        theta = np.pi / 4
        expected_result = self.eg.r_obs * np.cos(theta)
        self.assertAlmostEqual(expected_result, self.eg._A_1(theta))

    def test_A_2(self):
        theta = np.pi / 4
        expected_result = self.eg.r_obs * np.sin(theta)
        self.assertAlmostEqual(expected_result, self.eg._A_2(theta))

    def test_pl(self):
        theta = np.pi / 4
        expected_result = np.sqrt(
            self.eg.r_top**2 - self.eg._A_2(theta) ** 2
        ) - self.eg._A_1(theta)
        self.assertAlmostEqual(expected_result, self.eg.pl(theta))

    def test_cos_th_star(self):
        theta = np.pi / 4
        expected_result = (self.eg._A_1(theta) + self.eg.pl(theta)) / self.eg.r_top
        self.assertAlmostEqual(expected_result, self.eg.cos_th_star(theta))

    def test_h(self):
        theta = np.pi / 4
        dl = 100
        expected_result = (
            np.sqrt(
                self.eg._A_2(theta) ** 2
                + (self.eg._A_1(theta) + self.eg.pl(theta) - dl) ** 2
            )
            - self.eg.r_E
        )
        self.assertAlmostEqual(expected_result, self.eg.h(dl, theta))

    def test_delta_l(self):
        theta = np.pi / 4
        h = 100
        expected_result = (
            self.eg._A_1(theta)
            + self.eg.pl(theta)
            - np.sqrt((h + self.eg.r_E) ** 2 - self.eg._A_2(theta) ** 2)
        )
        self.assertAlmostEqual(expected_result, self.eg.delta_l(h, theta))


if __name__ == "__main__":
    unittest.main()
