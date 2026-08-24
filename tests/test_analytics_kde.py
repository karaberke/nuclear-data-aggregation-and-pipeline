"""Gaussian KDE, including the log-space Jacobian correction."""

import unittest

import numpy as np

from backend.services import analytics

from tests.fixtures import make_series, simple_series


def trapezoid_area(x: np.ndarray, density: np.ndarray) -> float:
    return float(np.sum(np.diff(x) * (density[1:] + density[:-1]) / 2))


class KdeTests(unittest.TestCase):
    def test_kde_returns_an_independently_normalized_curve(self):
        result = analytics.kde_series(simple_series(), "linear", 1, 300)
        self.assertEqual(len(result.curves), 1)
        curve = result.curves[0]
        self.assertEqual(curve.x.size, 300)
        self.assertLess(abs(trapezoid_area(curve.x, curve.density) - 1), 1e-10)

    def test_log_kde_excludes_nonpositive_cross_sections(self):
        series = [
            make_series("A|I|D", [(1, 2), (2, 4), (3, 8), (4, 16), (5, 0)])
        ]
        result = analytics.kde_series(series, "log", 1, 50)
        self.assertEqual(result.excluded_nonpositive, 1)
        for curve in result.curves:
            self.assertTrue(np.all(curve.x > 0))

    def test_log_kde_is_also_unit_area_after_the_jacobian(self):
        result = analytics.kde_series(simple_series(), "log", 1, 300)
        curve = result.curves[0]
        self.assertLess(abs(trapezoid_area(curve.x, curve.density) - 1), 1e-10)

    def test_bandwidth_multiplier_is_clamped(self):
        wide = analytics.kde_series(simple_series(), "linear", 99, 200)
        at_cap = analytics.kde_series(simple_series(), "linear", 4, 200)
        np.testing.assert_allclose(
            wide.curves[0].density, at_cap.curves[0].density
        )

        narrow = analytics.kde_series(simple_series(), "linear", 0.001, 200)
        at_floor = analytics.kde_series(simple_series(), "linear", 0.25, 200)
        np.testing.assert_allclose(
            narrow.curves[0].density, at_floor.curves[0].density
        )

    def test_series_with_no_usable_values_emits_no_curve(self):
        series = [make_series("A|I|D", [(1, -1), (2, -2)])]
        result = analytics.kde_series(series, "log", 1, 50)
        self.assertEqual(result.curves, [])
        self.assertEqual(result.excluded_nonpositive, 2)


if __name__ == "__main__":
    unittest.main()
