"""The binning performance budget.

The same 500 ms bar the TypeScript suite set, deliberately unchanged so it
cannot be quietly lowered during the migration: 30,000 points against the
725-group SAND-II725 structure.
"""

import os
import time
import unittest

from backend.services import analytics
from backend.services.multigroup import structure_edges

from tests.fixtures import performance_series

BUDGET_SECONDS = 0.5


@unittest.skipIf(
    os.getenv("SKIP_PERF_TESTS"), "SKIP_PERF_TESTS set (loaded CI machine)"
)
class BinningPerformanceTests(unittest.TestCase):
    def test_thirty_thousand_points_into_725_groups_is_fast(self):
        series = performance_series()
        edges = structure_edges("SAND-II725")

        start = time.perf_counter()
        result = analytics.bin_series(series, 10, "log", edges)
        elapsed = time.perf_counter() - start

        self.assertEqual(result.edges.size, 726)
        self.assertLess(
            elapsed,
            BUDGET_SECONDS,
            f"expected under {BUDGET_SECONDS}s, took {elapsed:.3f}s",
        )

    def test_five_series_stay_within_budget(self):
        """The real worst case: a full 5-series comparison."""
        import dataclasses

        base = performance_series()[0]
        series = [
            dataclasses.replace(base, key=f"P{index}|I|D")
            for index in range(5)
        ]
        edges = structure_edges("SAND-II725")

        start = time.perf_counter()
        analytics.bin_series(series, 10, "log", edges)
        elapsed = time.perf_counter() - start

        self.assertLess(
            elapsed,
            BUDGET_SECONDS * 5,
            f"5-series binning took {elapsed:.3f}s",
        )

    def test_kde_stays_interactive(self):
        """The bandwidth slider re-runs this on every release."""
        series = performance_series()
        start = time.perf_counter()
        analytics.kde_series(series, "log", 1.0)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, BUDGET_SECONDS, f"KDE took {elapsed:.3f}s")


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(
    os.getenv("SKIP_PERF_TESTS"), "SKIP_PERF_TESTS set (loaded CI machine)"
)
class ComparisonPerformanceTests(unittest.TestCase):
    """Two 30,000-point curves on different grids, merged and compared.

    The merge must not degrade to a pairwise scan: an O(N*M) comparison of
    two 30,000-point series is 9e8 operations and would miss this budget by
    orders of magnitude rather than by a little.
    """

    def two_series(self):
        import numpy as np

        reference = performance_series()[0]
        # A different grid, so the merge and interpolation both do real work
        # rather than lining up index for index.
        shifted = analytics.SeriesArrays(
            key="Q|I|D",
            database="Q",
            isotope="I",
            dataset="D",
            energy=reference.energy * 1.0001,
            sigma=np.roll(reference.sigma, 11),
        )
        return reference, shifted

    def test_two_thirty_thousand_point_series_compare_quickly(self):
        reference, comparison = self.two_series()

        start = time.perf_counter()
        results = analytics.compare_cross_sections(reference, [comparison])
        elapsed = time.perf_counter() - start

        result = results[0]
        # The merged grid really is the union of both, not one of them.
        self.assertGreater(result.size, 30_000)
        self.assertLess(
            elapsed,
            BUDGET_SECONDS,
            f"expected under {BUDGET_SECONDS}s, took {elapsed:.3f}s",
        )

    def test_crossings_are_found_without_downsampling(self):
        """Every crossing, at full resolution - the budget must not buy that."""
        reference, comparison = self.two_series()

        start = time.perf_counter()
        result = analytics.compare_cross_sections(reference, [comparison])[0]
        elapsed = time.perf_counter() - start

        self.assertGreater(result.crossing_energies.size, 0)
        self.assertLess(elapsed, BUDGET_SECONDS)
