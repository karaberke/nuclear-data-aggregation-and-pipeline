"""Parity: the Python port reproduces the TypeScript implementation.

The fixtures below were generated from the compiled TypeScript and
committed, which is what lets this suite keep protecting the numerics now
that the SPA is gone: parity is enforced in CI forever rather than checked
once by hand. The generator itself was removed with the rest of the
TypeScript — recover it from git history if the goldens ever need rebuilding,
and treat any need to do so as a red flag, since these values are supposed to
be frozen.

Tolerances are per-field and deliberately tight. The one place where exact
bit-equality is NOT guaranteed is `cross_section_mean`: `np.bincount`
accumulates in array order while the TypeScript accumulated sequentially with
`sums[i] += value`, which can differ in the last 1-2 ulp on real data.
"""

import json
import unittest
from pathlib import Path

import numpy as np

from backend.services import analytics

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "parity"

# Exact: integers and identifiers.
EXACT_FIELDS = ("series_key", "group_index", "point_count")
# Derived purely from the edges - should agree to the last ulp.
EDGE_FIELDS = ("bin_start", "bin_end", "bin_width", "bin_center", "coverage_fraction")
# Accumulated sums; see the module docstring.
VALUE_FIELDS = (
    "cross_section_mean",
    "cross_section_group_average",
    "cross_section_integral",
)

EDGE_TOLERANCE = 1e-12
VALUE_TOLERANCE = 1e-9


def load(name: str) -> dict:
    with (FIXTURE_DIR / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def series_from_fixture(key: str, points: list[dict]) -> analytics.SeriesArrays:
    database, isotope, dataset = key.split("|")
    return analytics.series_from_points(key, database, isotope, dataset, points)


class ParityTestCase(unittest.TestCase):
    """Shared comparison helpers."""

    @classmethod
    def setUpClass(cls):
        cls.source = load("source_series.json")

    def simple(self):
        return series_from_fixture("DB|Co59|MT102", self.source["simple"])

    def line(self):
        return series_from_fixture("DB|H1|MT1", self.source["line"])

    def boundary(self):
        return series_from_fixture("DB|U235|MT18", self.source["boundary"])

    def sparse_gap(self):
        return series_from_fixture("DB|Fe56|MT2", self.source["sparseGap"])

    def big(self):
        return series_from_fixture(
            "TENDL-2019|Co59|MT102", self.source["big"]
        )

    def comparison(self):
        return [
            series_from_fixture(item["key"], item["points"])
            for item in self.source["comparison"]
        ]

    def assert_close(self, actual, expected, tolerance, label):
        """Compare allowing None <-> null, which must match exactly."""
        if expected is None or actual is None:
            # A null becoming 0.0 is a physics bug, not a rounding difference.
            self.assertIs(
                actual if actual is None else None,
                expected if expected is None else None,
                f"{label}: null mismatch (python={actual!r}, ts={expected!r})",
            )
            return
        self.assertAlmostEqual(
            actual,
            expected,
            delta=max(abs(expected) * tolerance, tolerance),
            msg=f"{label}: {actual!r} != {expected!r}",
        )

    def assert_grouping_matches(self, fixture_name: str, result):
        expected = load(fixture_name)

        np.testing.assert_allclose(
            result.edges,
            np.asarray(expected["edges"], dtype=np.float64),
            rtol=EDGE_TOLERANCE,
            atol=0,
            err_msg=f"{fixture_name}: edges differ",
        )
        self.assertEqual(result.x_scale, expected["xScale"], fixture_name)
        self.assertEqual(
            result.excluded_nonpositive,
            expected["excludedNonpositive"],
            fixture_name,
        )
        self.assertEqual(
            len(result.records),
            len(expected["records"]),
            f"{fixture_name}: record count differs",
        )

        for index, (actual, want) in enumerate(
            zip(result.records, expected["records"])
        ):
            where = f"{fixture_name}[{index}]"
            for field in EXACT_FIELDS:
                self.assertEqual(
                    getattr(actual, field), want[field], f"{where}.{field}"
                )
            for field in EDGE_FIELDS:
                self.assert_close(
                    getattr(actual, field),
                    want[field],
                    EDGE_TOLERANCE,
                    f"{where}.{field}",
                )
            for field in VALUE_FIELDS:
                self.assert_close(
                    getattr(actual, field),
                    want[field],
                    VALUE_TOLERANCE,
                    f"{where}.{field}",
                )


class BinningParityTests(ParityTestCase):
    def test_automatic_linear_edges(self):
        self.assert_grouping_matches(
            "bin_automatic_linear_10.json",
            analytics.bin_series([self.simple()], 10, "linear"),
        )

    def test_automatic_log_edges_over_two_thousand_points(self):
        self.assert_grouping_matches(
            "bin_automatic_log_40.json",
            analytics.bin_series([self.big()], 40, "log"),
        )

    def test_automatic_edges_with_explicit_bounds(self):
        self.assert_grouping_matches(
            "bin_automatic_linear_bounded.json",
            analytics.bin_series(
                [self.line()],
                10,
                "linear",
                None,
                analytics.EnergyBounds(minimum=2, maximum=8),
            ),
        )

    def test_custom_edges(self):
        self.assert_grouping_matches(
            "bin_custom_0_2_10.json",
            analytics.bin_series(
                [self.line()], 10, "linear", np.array([0.0, 2.0, 10.0])
            ),
        )

    def test_custom_edges_covering_part_of_the_domain(self):
        self.assert_grouping_matches(
            "bin_custom_partial_2_8.json",
            analytics.bin_series(
                [self.line()], 10, "linear", np.array([2.0, 8.0])
            ),
        )

    def test_custom_edges_extending_past_the_data(self):
        self.assert_grouping_matches(
            "bin_custom_negative_span.json",
            analytics.bin_series(
                [self.boundary()], 10, "linear", np.array([-5.0, 5.0, 15.0])
            ),
        )

    def test_groups_spanning_a_gap_are_interpolation_covered(self):
        self.assert_grouping_matches(
            "bin_sparse_gap.json",
            analytics.bin_series(
                [self.sparse_gap()],
                10,
                "linear",
                np.array([0.0, 2.0, 4.0, 6.0, 8.0, 12.0]),
            ),
        )

    def test_vitamin_j175_structure(self):
        edges = np.asarray(self.source["vitaminJEdges"], dtype=np.float64)
        self.assert_grouping_matches(
            "bin_vitamin_j175.json",
            analytics.bin_series([self.big()], 40, "log", edges),
        )

    def test_multiple_series_share_one_edge_set(self):
        self.assert_grouping_matches(
            "bin_comparison_log.json",
            analytics.bin_series(self.comparison(), 40, "log"),
        )


class KdeParityTests(ParityTestCase):
    def assert_kde_matches(self, fixture_name: str, result):
        expected = load(fixture_name)
        self.assertEqual(result.x_scale, expected["xScale"])
        self.assertEqual(
            result.excluded_nonpositive, expected["excludedNonpositive"]
        )

        flattened = [
            {
                "series_key": curve.series_key,
                "cross_section_barns": float(x),
                "density": float(density),
            }
            for curve in result.curves
            for x, density in zip(curve.x, curve.density)
        ]
        self.assertEqual(len(flattened), len(expected["records"]), fixture_name)

        for index, (actual, want) in enumerate(
            zip(flattened, expected["records"])
        ):
            where = f"{fixture_name}[{index}]"
            self.assertEqual(actual["series_key"], want["series_key"], where)
            self.assert_close(
                actual["cross_section_barns"],
                want["cross_section_barns"],
                EDGE_TOLERANCE,
                f"{where}.x",
            )
            self.assert_close(
                actual["density"],
                want["density"],
                VALUE_TOLERANCE,
                f"{where}.density",
            )

    def test_linear_kde(self):
        self.assert_kde_matches(
            "kde_linear_bw1.json",
            analytics.kde_series([self.big()], "linear", 1, 300),
        )

    def test_log_kde_applies_the_jacobian(self):
        self.assert_kde_matches(
            "kde_log_bw05.json",
            analytics.kde_series([self.big()], "log", 0.5, 50),
        )

    def test_multi_series_kde_shares_one_domain(self):
        self.assert_kde_matches(
            "kde_comparison_log.json",
            analytics.kde_series(self.comparison(), "log", 1.5, 120),
        )


class FilteringParityTests(ParityTestCase):
    def test_inclusive_range_filtering(self):
        expected = load("filter_series.json")
        filters = analytics.RangeFilters(
            energy_min=expected["filters"]["energyMin"],
            energy_max=expected["filters"]["energyMax"],
            value_min=expected["filters"]["valueMin"],
            value_max=expected["filters"]["valueMax"],
        )
        result = analytics.filter_series([self.big()], filters)
        self.assertEqual(len(result), len(expected["series"]))

        for actual, want in zip(result, expected["series"]):
            self.assertEqual(actual.key, want["key"])
            self.assertEqual(actual.size, len(want["points"]))
            np.testing.assert_allclose(
                actual.energy,
                [point["energy_MeV"] for point in want["points"]],
                rtol=EDGE_TOLERANCE,
            )
            np.testing.assert_allclose(
                actual.sigma,
                [point["cross_section_barns"] for point in want["points"]],
                rtol=EDGE_TOLERANCE,
            )


class SamplingParityTests(ParityTestCase):
    def test_deterministic_sampling_picks_identical_indices(self):
        expected = load("deterministic_sample.json")
        for case in expected["cases"]:
            with self.subTest(size=case["size"], n=len(case["values"])):
                result = analytics.deterministic_sample(
                    np.asarray(case["values"], dtype=np.float64), case["size"]
                )
                np.testing.assert_array_equal(
                    result, np.asarray(case["result"], dtype=np.float64)
                )


if __name__ == "__main__":
    unittest.main()
