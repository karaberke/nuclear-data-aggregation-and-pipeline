"""The Ratio / Difference comparison.

Covers the orientation (comparison / reference), grid merging across
different energy grids, exact baseline crossings, and the undefined-value
rules. The invariant under test throughout is that a point above the baseline
means the comparison is larger and a point below means the reference is.
"""

import unittest

import numpy as np

from backend.services import analytics

from tests.fixtures import make_series


def compare(reference, comparison, **kwargs):
    """One pair, unwrapped - the shape most tests here want."""
    return analytics.compare_cross_sections(reference, [comparison], **kwargs)[0]


class IdenticalCurveTests(unittest.TestCase):
    def test_identical_curves_are_ratio_one_and_zero_percent(self):
        reference = make_series("A|I|D", [(1, 2), (2, 4), (3, 8)])
        comparison = make_series("B|I|D", [(1, 2), (2, 4), (3, 8)])

        result = compare(reference, comparison)

        self.assertTrue(result.valid.all())
        np.testing.assert_array_equal(result.ratio, np.ones(3))
        np.testing.assert_array_equal(result.percent_difference, np.zeros(3))

    def test_comparison_twice_the_reference_is_ratio_two_and_100_percent(self):
        reference = make_series("A|I|D", [(1, 2), (2, 4), (3, 8)])
        comparison = make_series("B|I|D", [(1, 4), (2, 8), (3, 16)])

        result = compare(reference, comparison)

        np.testing.assert_array_equal(result.ratio, np.full(3, 2.0))
        np.testing.assert_array_equal(
            result.percent_difference, np.full(3, 100.0)
        )

    def test_comparison_half_the_reference_is_ratio_half_and_minus_50_percent(self):
        reference = make_series("A|I|D", [(1, 2), (2, 4), (3, 8)])
        comparison = make_series("B|I|D", [(1, 1), (2, 2), (3, 4)])

        result = compare(reference, comparison)

        np.testing.assert_array_equal(result.ratio, np.full(3, 0.5))
        np.testing.assert_array_equal(
            result.percent_difference, np.full(3, -50.0)
        )


class GridMergeTests(unittest.TestCase):
    def test_different_energy_grids_are_interpolated_not_index_matched(self):
        """Index matching would pair (1,10) with (1,20) and call it 2x."""
        reference = make_series("A|I|D", [(0, 0), (10, 10)])
        comparison = make_series("B|I|D", [(0, 0), (5, 10), (10, 20)])

        result = compare(reference, comparison)

        # Union of both grids plus the overlap boundaries.
        np.testing.assert_array_equal(result.energy, [0.0, 5.0, 10.0])
        # Reference is the line y = x, so it is 5 at the comparison's own
        # midpoint - a value present in neither source array.
        np.testing.assert_array_equal(result.reference_sigma, [0.0, 5.0, 10.0])
        np.testing.assert_array_equal(result.comparison_sigma, [0.0, 10.0, 20.0])
        np.testing.assert_array_equal(result.ratio[1:], [2.0, 2.0])

    def test_the_grid_is_the_sorted_union_of_both_series(self):
        reference = make_series("A|I|D", [(1, 1), (3, 3), (5, 5)])
        comparison = make_series("B|I|D", [(1, 1), (2, 2), (4, 4), (5, 5)])

        result = compare(reference, comparison)

        np.testing.assert_array_equal(result.energy, [1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertTrue(np.all(np.diff(result.energy) > 0))


class CrossingTests(unittest.TestCase):
    def test_a_single_crossing_lands_exactly_on_the_baseline(self):
        reference = make_series("A|I|D", [(0, 0), (10, 10)])
        comparison = make_series("B|I|D", [(0, 10), (10, 0)])

        result = compare(reference, comparison)

        np.testing.assert_array_equal(result.crossing_energies, [5.0])
        index = int(np.searchsorted(result.energy, 5.0))
        self.assertEqual(result.energy[index], 5.0)
        self.assertEqual(result.ratio[index], 1.0)
        self.assertEqual(result.percent_difference[index], 0.0)

    def test_every_crossing_is_retained_and_ordered(self):
        """Three changes of dominance must produce three baseline crossings."""
        reference = make_series(
            "A|I|D", [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]
        )
        comparison = make_series(
            "B|I|D", [(0, 0), (1, 2), (2, 0), (3, 2), (4, 0)]
        )

        result = compare(reference, comparison)

        self.assertEqual(result.crossing_energies.size, 4)
        np.testing.assert_allclose(
            result.crossing_energies, [0.5, 1.5, 2.5, 3.5]
        )
        self.assertTrue(np.all(np.diff(result.crossing_energies) > 0))

        for energy in result.crossing_energies:
            index = int(np.searchsorted(result.energy, energy))
            self.assertAlmostEqual(float(result.ratio[index]), 1.0, places=12)
            self.assertAlmostEqual(
                float(result.percent_difference[index]), 0.0, places=10
            )

    def test_dominance_changes_sign_across_each_crossing(self):
        reference = make_series("A|I|D", [(0, 1), (1, 1), (2, 1)])
        comparison = make_series("B|I|D", [(0, 0), (1, 2), (2, 0)])

        result = compare(reference, comparison)
        above = result.percent_difference > 0

        # Above the baseline exactly where the comparison is the larger curve.
        larger = result.comparison_sigma > result.reference_sigma
        np.testing.assert_array_equal(above[result.valid], larger[result.valid])


class SwapTests(unittest.TestCase):
    def test_swapping_reference_and_comparison_gives_the_reciprocal(self):
        reference = make_series("A|I|D", [(1, 2), (2, 4), (3, 1)])
        comparison = make_series("B|I|D", [(1, 8), (2, 2), (3, 5)])

        forward = compare(reference, comparison)
        backward = compare(comparison, reference)

        # A point undefined in either direction has no reciprocal to compare.
        both = forward.valid & backward.valid
        self.assertTrue(both.any())
        np.testing.assert_allclose(
            forward.ratio[both], 1.0 / backward.ratio[both], rtol=1e-12
        )

    def test_percent_difference_stays_consistent_when_the_reference_swaps(self):
        """(1 + D/100)(1 + D_swapped/100) == 1, since each is R and 1/R."""
        reference = make_series("A|I|D", [(1, 2), (2, 4), (3, 1)])
        comparison = make_series("B|I|D", [(1, 8), (2, 2), (3, 5)])

        forward = compare(reference, comparison)
        backward = compare(comparison, reference)
        both = forward.valid & backward.valid

        product = (1 + forward.percent_difference[both] / 100) * (
            1 + backward.percent_difference[both] / 100
        )
        np.testing.assert_allclose(product, np.ones(int(both.sum())), rtol=1e-12)


class UndefinedValueTests(unittest.TestCase):
    def test_zero_reference_produces_a_gap_not_an_infinity(self):
        reference = make_series("A|I|D", [(1, 0.0), (2, 1.0)])
        comparison = make_series("B|I|D", [(1, 5.0), (2, 1.0)])

        result = compare(reference, comparison)

        self.assertFalse(bool(result.valid[0]))
        self.assertTrue(np.isnan(result.ratio[0]))
        self.assertTrue(np.isnan(result.percent_difference[0]))
        self.assertFalse(np.isinf(result.ratio).any())
        self.assertEqual(
            result.invalid_reason[0], analytics.REASON_REFERENCE_ZERO
        )

    def test_the_source_values_survive_at_an_undefined_point(self):
        """The export still has to show what was there."""
        reference = make_series("A|I|D", [(1, 0.0), (2, 1.0)])
        comparison = make_series("B|I|D", [(1, 5.0), (2, 1.0)])

        result = compare(reference, comparison)

        self.assertEqual(result.reference_sigma[0], 0.0)
        self.assertEqual(result.comparison_sigma[0], 5.0)

    def test_both_zero_is_still_undefined(self):
        reference = make_series("A|I|D", [(1, 0.0), (2, 1.0)])
        comparison = make_series("B|I|D", [(1, 0.0), (2, 1.0)])

        result = compare(reference, comparison)

        self.assertFalse(bool(result.valid[0]))
        self.assertTrue(np.isnan(result.ratio[0]))
        self.assertEqual(result.invalid_reason[0], analytics.REASON_BOTH_ZERO)

    def test_a_zero_reference_never_becomes_zero_ratio(self):
        """Substituting 0 would read as 'the comparison is 100% smaller'."""
        reference = make_series("A|I|D", [(1, 0.0), (2, 1.0)])
        comparison = make_series("B|I|D", [(1, 5.0), (2, 1.0)])

        result = compare(reference, comparison)

        self.assertNotEqual(result.ratio[0], 0.0)

    def test_excluded_points_are_counted(self):
        reference = make_series("A|I|D", [(1, 0.0), (2, 1.0), (3, 0.0)])
        comparison = make_series("B|I|D", [(1, 5.0), (2, 1.0), (3, 2.0)])

        result = compare(reference, comparison)

        self.assertEqual(result.excluded_count, 2)
        self.assertEqual(result.valid_count, 1)


class OverlapTests(unittest.TestCase):
    def test_non_overlapping_domains_give_a_descriptive_empty_result(self):
        reference = make_series("A|I|D", [(1, 1), (2, 2)])
        comparison = make_series("B|I|D", [(5, 1), (6, 2)])

        result = compare(reference, comparison)

        self.assertEqual(result.size, 0)
        self.assertIsNotNone(result.unavailable_reason)
        self.assertIn("do not overlap", result.unavailable_reason)

    def test_no_extrapolation_happens_outside_the_common_domain(self):
        reference = make_series("A|I|D", [(0, 1), (10, 11)])
        comparison = make_series("B|I|D", [(4, 1), (20, 17)])

        result = compare(reference, comparison)

        self.assertEqual(result.overlap_min, 4.0)
        self.assertEqual(result.overlap_max, 10.0)
        self.assertEqual(float(result.energy.min()), 4.0)
        self.assertEqual(float(result.energy.max()), 10.0)

    def test_the_energy_filter_narrows_the_overlap(self):
        reference = make_series("A|I|D", [(0, 1), (10, 11)])
        comparison = make_series("B|I|D", [(0, 1), (10, 11)])

        result = compare(reference, comparison, energy_min=2.0, energy_max=6.0)

        self.assertEqual(result.overlap_min, 2.0)
        self.assertEqual(result.overlap_max, 6.0)
        self.assertGreaterEqual(float(result.energy.min()), 2.0)
        self.assertLessEqual(float(result.energy.max()), 6.0)


class DiscontinuityTests(unittest.TestCase):
    """A repeated energy carries two cross sections, so it is a jump."""

    def build(self):
        # Comparison is flat at 2, jumps to 5 at E=2, stays at 5. The
        # reference sits at 3 throughout, so the curves change dominance
        # *through* the jump and never actually meet.
        reference = make_series("A|I|D", [(1, 3), (2, 3), (3, 3)])
        comparison = make_series("B|I|D", [(1, 2), (2, 2), (2, 5), (3, 5)])
        return reference, comparison

    def test_a_jump_does_not_create_a_false_crossing(self):
        reference, comparison = self.build()

        result = compare(reference, comparison)

        self.assertEqual(result.crossing_energies.size, 0)

    def test_a_jump_breaks_the_line_instead_of_averaging(self):
        reference, comparison = self.build()

        result = compare(reference, comparison)
        index = int(np.searchsorted(result.energy, 2.0))

        self.assertEqual(result.energy[index], 2.0)
        self.assertFalse(bool(result.valid[index]))
        self.assertTrue(np.isnan(result.ratio[index]))
        self.assertEqual(
            result.invalid_reason[index], analytics.REASON_DISCONTINUITY
        )

    def test_the_averaged_value_is_never_produced(self):
        """(2 + 5) / 2 == 3.5 is in neither evaluation."""
        reference, comparison = self.build()

        result = compare(reference, comparison)

        self.assertNotIn(3.5, set(result.comparison_sigma.tolist()))


class CompatibilityTests(unittest.TestCase):
    def test_a_different_nuclide_is_rejected(self):
        reference = make_series("A|U235|MT1", [(1, 1), (2, 2)])
        comparison = make_series("B|Pu239|MT1", [(1, 1), (2, 2)])

        with self.assertRaises(analytics.IncompatibleSeriesError) as caught:
            compare(reference, comparison)

        self.assertIn("U235", str(caught.exception))
        self.assertIn("Pu239", str(caught.exception))

    def test_a_different_reaction_channel_is_rejected(self):
        reference = make_series("A|U235|MT18", [(1, 1), (2, 2)])
        comparison = make_series("A|U235|MT102", [(1, 1), (2, 2)])

        with self.assertRaises(analytics.IncompatibleSeriesError) as caught:
            compare(reference, comparison)

        self.assertIn("MT18", str(caught.exception))
        self.assertIn("MT102", str(caught.exception))

    def test_a_series_cannot_be_compared_against_itself(self):
        series = make_series("A|U235|MT1", [(1, 1), (2, 2)])

        with self.assertRaises(analytics.IncompatibleSeriesError):
            compare(series, series)

    def test_no_pair_is_computed_when_one_is_incompatible(self):
        """A rejected selection must not render a partial chart."""
        reference = make_series("A|U235|MT1", [(1, 1), (2, 2)])
        good = make_series("B|U235|MT1", [(1, 1), (2, 2)])
        bad = make_series("C|Pu239|MT1", [(1, 1), (2, 2)])

        with self.assertRaises(analytics.IncompatibleSeriesError):
            analytics.compare_cross_sections(reference, [good, bad])


class DominantLabelTests(unittest.TestCase):
    def test_the_larger_curve_is_named(self):
        reference = make_series("A|I|D", [(1, 1), (2, 4)])
        comparison = make_series("B|I|D", [(1, 2), (2, 2)])

        result = compare(reference, comparison)
        labels = analytics.dominant_labels(result, "REF", "CMP")

        self.assertEqual(labels[0], "CMP")
        self.assertEqual(labels[-1], "REF")

    def test_equality_reads_as_equal_rather_than_naming_a_winner(self):
        reference = make_series("A|I|D", [(1, 2), (2, 4)])
        comparison = make_series("B|I|D", [(1, 2), (2, 4)])

        result = compare(reference, comparison)
        labels = analytics.dominant_labels(result, "REF", "CMP")

        self.assertEqual(list(labels), ["Equal within tolerance"] * 2)


if __name__ == "__main__":
    unittest.main()
