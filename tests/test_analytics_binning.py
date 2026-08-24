"""Binning, integration, and coverage.

Ported from the TypeScript SPA's `analytics.test.mjs`. Test names and magic
numbers are preserved from it deliberately - they are the reference the
parity fixtures were captured against.
"""

import unittest

import numpy as np

from backend.services import analytics
from backend.services.multigroup import structure_edges

from tests.fixtures import (
    boundary_series,
    line_series,
    make_series,
    simple_series,
    sparse_gap_series,
)


class MeanBinningTests(unittest.TestCase):
    def test_binning_calculates_means_and_omits_empty_mean_bins(self):
        result = analytics.bin_series(simple_series(), 10, "linear")
        self.assertEqual(result.x_scale, "linear")

        mean_records = [
            record
            for record in result.records
            if record.cross_section_mean is not None
        ]
        self.assertGreater(len(mean_records), 0)
        self.assertLessEqual(len(mean_records), 4)

        weighted_sum = sum(
            record.cross_section_mean * record.point_count
            for record in mean_records
        )
        # 2 + 4 + 8 + 16; every sample lands in exactly one group.
        self.assertEqual(weighted_sum, 30)

    def test_bin_series_produces_group_count_plus_one_edges(self):
        automatic = analytics.bin_series(simple_series(), 10, "linear")
        self.assertEqual(automatic.edges.size, 11)

        custom = analytics.bin_series(
            simple_series(), 10, "linear", np.array([1.0, 2.0, 3.0, 4.0])
        )
        self.assertEqual(custom.edges.size, 4)

    def test_log_binning_reports_nonpositive_energy_values(self):
        series = [make_series("A|I|D", [(0, 1), (1, 2), (2, 4), (3, 8)])]
        result = analytics.bin_series(series, 10, "log")
        self.assertEqual(result.excluded_nonpositive, 1)

    def test_bin_centers_use_the_right_mean_for_the_scale(self):
        """Not covered directly by the TypeScript suite."""
        linear = analytics.bin_series(
            line_series(), 10, "linear", np.array([2.0, 8.0])
        )
        self.assertAlmostEqual(linear.records[0].bin_center, 5.0)

        log = analytics.bin_series(
            simple_series(), 10, "log", np.array([1.0, 4.0])
        )
        # Geometric, not arithmetic: sqrt(1 * 4) == 2, not 2.5.
        self.assertAlmostEqual(log.records[0].bin_center, 2.0)


class IntegralPreservationTests(unittest.TestCase):
    def test_density_recovers_the_exact_integral_under_uniform_edges(self):
        result = analytics.bin_series(line_series(), 10, "linear")
        total_area = sum(
            record.cross_section_group_average * record.bin_width
            for record in result.records
            if record.cross_section_group_average is not None
        )
        # integral of x dx from 0 to 10 == 50
        self.assertLess(abs(total_area - 50), 1e-9)

    def test_custom_edges_preserve_the_integral_over_the_full_domain(self):
        result = analytics.bin_series(
            line_series(), 10, "linear", np.array([0.0, 2.0, 10.0])
        )
        np.testing.assert_array_equal(result.edges, [0, 2, 10])
        self.assertEqual(len(result.records), 2)

        total_area = sum(
            record.cross_section_group_average * record.bin_width
            for record in result.records
            if record.cross_section_group_average is not None
        )
        self.assertLess(abs(total_area - 50), 1e-9)

    def test_partial_edges_preserve_only_that_portion_of_the_integral(self):
        result = analytics.bin_series(
            line_series(), 10, "linear", np.array([2.0, 8.0])
        )
        self.assertEqual(len(result.records), 1)

        total_area = sum(
            record.cross_section_group_average * record.bin_width
            for record in result.records
            if record.cross_section_group_average is not None
        )
        expected = (8**2 - 2**2) / 2
        self.assertLess(abs(total_area - expected), 1e-9)
        self.assertGreater(
            abs(total_area - 50), 1, "must not equal the full-domain integral"
        )

    def test_a_group_edge_on_a_data_point_is_shared_by_both_neighbors(self):
        result = analytics.bin_series(
            boundary_series(), 10, "linear", np.array([0.0, 5.0, 10.0])
        )
        first, second = result.records[0], result.records[1]
        # Trapezoids: (0+50)/2 * 5 == 125 and (50+10)/2 * 5 == 150.
        self.assertLess(abs(first.cross_section_integral - 125), 1e-9)
        self.assertLess(abs(second.cross_section_integral - 150), 1e-9)

        total = sum(
            record.cross_section_group_average * record.bin_width
            for record in result.records
            if record.cross_section_group_average is not None
        )
        self.assertLess(abs(total - 275), 1e-9)

    def test_consecutive_gap_groups_are_interpolation_covered(self):
        edges = np.arange(12, dtype=np.float64)  # [0, 1, ..., 11]
        result = analytics.bin_series(
            sparse_gap_series(), 10, "linear", edges
        )
        # Groups 2..7 lie strictly inside the gap between energy 1 and 10.
        mid_groups = [
            record
            for record in result.records
            if 2 <= record.group_index <= 7
        ]
        self.assertGreater(len(mid_groups), 0)
        self.assertTrue(
            all(
                record.point_count == 0
                and record.cross_section_group_average is not None
                for record in mid_groups
            ),
            "gap groups must be interpolation-covered with no landed sample",
        )

        total = sum(
            record.cross_section_integral
            for record in result.records
            if record.cross_section_integral is not None
        )
        self.assertLess(abs(total - 55), 1e-9)


class CoverageTests(unittest.TestCase):
    def test_coverage_is_fractional_for_groups_straddling_the_domain(self):
        # Each 10-wide group half-overlaps the [0, 10] data domain.
        result = analytics.bin_series(
            line_series(), 10, "linear", np.array([-5.0, 5.0, 15.0])
        )
        first, second = result.records[0], result.records[1]
        self.assertLess(abs(first.coverage_fraction - 0.5), 1e-9)
        self.assertLess(abs(second.coverage_fraction - 0.5), 1e-9)

    def test_coverage_is_exactly_one_when_a_group_matches_the_domain(self):
        result = analytics.bin_series(
            line_series(), 10, "linear", np.array([0.0, 10.0])
        )
        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.records[0].coverage_fraction, 1)

    def test_uncomputable_values_stay_none_never_zero(self):
        """A null becoming 0.0 would be a physics error, not a rounding one."""
        result = analytics.bin_series(
            sparse_gap_series(),
            10,
            "linear",
            np.array([0.0, 1.0, 3.0, 5.0, 11.0]),
        )
        gap = [record for record in result.records if record.point_count == 0]
        self.assertTrue(gap)
        for record in gap:
            self.assertIsNone(record.cross_section_mean)
            self.assertIsNotNone(record.cross_section_group_average)


class EdgeValidationTests(unittest.TestCase):
    def test_bin_series_rejects_malformed_custom_edges(self):
        cases = {
            "too few": [1.0],
            "descending": [3.0, 1.0, 2.0],
            "non-finite": [1.0, float("nan"), 3.0],
            "duplicate": [1.0, 1.0, 2.0],
        }
        for label, edges in cases.items():
            with self.subTest(label=label):
                with self.assertRaises(analytics.BinEdgeError):
                    analytics.bin_series(
                        simple_series(), 10, "linear", np.array(edges)
                    )

        with self.subTest(label="nonpositive on a log scale"):
            with self.assertRaises(analytics.BinEdgeError):
                analytics.bin_series(
                    simple_series(), 10, "log", np.array([-1.0, 1.0, 10.0])
                )

    def test_edge_errors_carry_displayable_messages(self):
        with self.assertRaises(analytics.BinEdgeError) as caught:
            analytics.bin_series(
                simple_series(), 10, "linear", np.array([3.0, 1.0])
            )
        self.assertEqual(
            str(caught.exception), "Bin edges must be strictly ascending."
        )

    def test_automatic_bounds_reject_a_nonpositive_minimum_on_a_log_scale(self):
        """The same rule custom edges get, applied to the automatic extent.

        Left unchecked this reached `math.log10` and raised a bare ValueError,
        which the callers that handle the user-correctable family do not catch.
        """
        for minimum in (0.0, -1.0):
            with self.subTest(minimum=minimum):
                with self.assertRaises(analytics.BinEdgeError) as caught:
                    analytics.bin_series(
                        line_series(),
                        10,
                        "log",
                        None,
                        analytics.EnergyBounds(minimum=minimum, maximum=10.0),
                    )
                self.assertEqual(
                    str(caught.exception),
                    "Minimum energy must be positive for a logarithmic scale.",
                )

    def test_a_nonpositive_minimum_is_fine_on_a_linear_scale(self):
        result = analytics.bin_series(
            line_series(),
            10,
            "linear",
            None,
            analytics.EnergyBounds(minimum=0.0, maximum=10.0),
        )
        self.assertTrue(result.records)

    def test_automatic_bounds_reject_an_inverted_range(self):
        """Inverted bounds used to build a descending edge array silently.

        `np.searchsorted` in `_bin_indices` is undefined on one, so the result
        was an empty grouping blamed on the filters rather than the bounds.
        """
        for scale in ("linear", "log"):
            with self.subTest(scale=scale):
                with self.assertRaises(analytics.BinEdgeError) as caught:
                    analytics.bin_series(
                        line_series(),
                        10,
                        scale,
                        None,
                        analytics.EnergyBounds(minimum=8.0, maximum=2.0),
                    )
                self.assertEqual(
                    str(caught.exception),
                    "Minimum energy must be less than maximum energy.",
                )

    def test_equal_bounds_are_still_widened_not_rejected(self):
        """`_expanded_extent` handles a degenerate extent deliberately."""
        result = analytics.bin_series(
            line_series(),
            10,
            "log",
            None,
            analytics.EnergyBounds(minimum=5.0, maximum=5.0),
        )
        self.assertGreater(result.edges.size, 1)
        self.assertTrue(bool(np.all(np.diff(result.edges) > 0)))


class StandardStructureTests(unittest.TestCase):
    def test_binning_against_a_bundled_structure(self):
        edges = structure_edges("VITAMIN-J175")
        result = analytics.bin_series(line_series(), 10, "log", edges)
        self.assertEqual(result.edges.size, 176)
        self.assertTrue(result.records)


if __name__ == "__main__":
    unittest.main()
