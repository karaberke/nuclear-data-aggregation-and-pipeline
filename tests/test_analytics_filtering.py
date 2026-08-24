"""Range filtering and deterministic sampling."""

import unittest

import numpy as np

from backend.services import analytics

from tests.fixtures import simple_series


class FilterTests(unittest.TestCase):
    def test_range_filters_are_inclusive(self):
        result = analytics.filter_series(
            simple_series(),
            analytics.RangeFilters(energy_min=2, energy_max=3),
        )
        np.testing.assert_array_equal(result[0].energy, [2, 3])
        np.testing.assert_array_equal(result[0].sigma, [4, 8])

    def test_value_bounds_are_inclusive(self):
        result = analytics.filter_series(
            simple_series(),
            analytics.RangeFilters(value_min=4, value_max=8),
        )
        np.testing.assert_array_equal(result[0].sigma, [4, 8])

    def test_series_are_emptied_never_dropped(self):
        """The caller must be able to tell "no points in range" apart from
        "series not requested"."""
        result = analytics.filter_series(
            simple_series(), analytics.RangeFilters(energy_min=1e9)
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].size, 0)
        self.assertEqual(result[0].key, "A|I|D")

    def test_absent_bounds_impose_no_constraint(self):
        result = analytics.filter_series(
            simple_series(), analytics.RangeFilters()
        )
        self.assertEqual(result[0].size, 4)

    def test_stddev_is_filtered_alongside_the_values(self):
        series = analytics.series_from_points(
            "A|I|D",
            "A",
            "I",
            "D",
            [
                {"energy_MeV": 1, "cross_section_barns": 2,
                 "cross_section_stddev_barns": 0.1},
                {"energy_MeV": 2, "cross_section_barns": 4,
                 "cross_section_stddev_barns": None},
                {"energy_MeV": 3, "cross_section_barns": 8,
                 "cross_section_stddev_barns": 0.3},
            ],
        )
        result = analytics.filter_series(
            [series], analytics.RangeFilters(energy_min=2)
        )
        self.assertEqual(result[0].size, 2)
        # The missing uncertainty stays missing, never coerced to zero.
        self.assertTrue(np.isnan(result[0].stddev[0]))
        self.assertAlmostEqual(result[0].stddev[1], 0.3)


class SamplingTests(unittest.TestCase):
    def test_deterministic_sampling_retains_endpoints(self):
        result = analytics.deterministic_sample(
            np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), 3
        )
        np.testing.assert_array_equal(result, [0, 2, 5])

    def test_sampling_is_a_noop_below_the_cap(self):
        values = np.array([0.0, 1.0, 2.0])
        np.testing.assert_array_equal(
            analytics.deterministic_sample(values, 7), values
        )

    def test_single_sample_takes_the_first_value(self):
        result = analytics.deterministic_sample(np.array([4.0, 9.0, 16.0]), 1)
        np.testing.assert_array_equal(result, [4])


if __name__ == "__main__":
    unittest.main()
