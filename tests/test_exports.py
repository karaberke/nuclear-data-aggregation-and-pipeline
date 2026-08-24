"""CSV rendering, quoting, and JavaScript-compatible number formatting."""

import json
import unittest
from pathlib import Path

from backend.services import exports

V8_FIXTURE = Path(__file__).parent / "fixtures" / "v8_number_strings.json"


class NumberFormattingTests(unittest.TestCase):
    """`repr(float)` and V8's `String(number)` disagree on presentation.

    Energies are in MeV and routinely reach 1e-11, so this is most of the
    energy column rather than a corner case.
    """

    def test_matches_v8_across_adversarial_doubles(self):
        with V8_FIXTURE.open(encoding="utf-8") as handle:
            cases = json.load(handle)
        self.assertGreater(len(cases), 30)
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(exports.format_number(value), expected)

    def test_integral_doubles_lose_the_trailing_zero(self):
        self.assertEqual(exports.format_number(2.0), "2")
        self.assertEqual(exports.format_number(-3.0), "-3")

    def test_small_magnitudes_switch_to_exponential_at_1e_minus_7(self):
        self.assertEqual(exports.format_number(1e-6), "0.000001")
        self.assertEqual(exports.format_number(1e-7), "1e-7")

    def test_large_magnitudes_switch_to_exponential_at_1e21(self):
        self.assertEqual(exports.format_number(1e20), "100000000000000000000")
        self.assertEqual(exports.format_number(1e21), "1e+21")

    def test_none_renders_blank_never_zero(self):
        self.assertEqual(exports.format_number(None), "")

    def test_negative_zero_renders_as_zero(self):
        self.assertEqual(exports.format_number(-0.0), "0")


class CsvCellTests(unittest.TestCase):
    def test_plain_values_are_unquoted(self):
        self.assertEqual(exports.csv_cell("ENDF/B-VIII.0"), "ENDF/B-VIII.0")
        self.assertEqual(exports.csv_cell("Co59"), "Co59")

    def test_commas_quotes_and_newlines_are_quoted(self):
        self.assertEqual(exports.csv_cell("a,b"), '"a,b"')
        self.assertEqual(exports.csv_cell('say "hi"'), '"say ""hi"""')
        self.assertEqual(exports.csv_cell("line\nbreak"), '"line\nbreak"')

    def test_mt_labels_containing_a_comma_are_quoted(self):
        # "MT 16 — (n,2n)" contains a comma inside the parentheses.
        from backend.services.mt_labels import format_mt_label

        self.assertEqual(
            exports.csv_cell(format_mt_label(16)), '"MT 16 — (n,2n)"'
        )

    def test_none_becomes_an_empty_field(self):
        self.assertEqual(exports.csv_cell(None), "")


class WriteCsvTests(unittest.TestCase):
    def test_framing_uses_lf_and_no_trailing_newline(self):
        csv = exports.write_csv(["a", "b"], [[1, 2], [3, 4]])
        self.assertEqual(csv, "a,b\n1,2\n3,4")
        self.assertNotIn("\r", csv)

    def test_header_only_when_there_are_no_rows(self):
        self.assertEqual(exports.write_csv(["a", "b"], []), "a,b")

    def test_missing_uncertainties_stay_blank(self):
        csv = exports.write_csv(
            exports.FILTERED_HEADER,
            [["DB", "Co59", "MT102", 1.5, 2.5, None]],
        )
        self.assertTrue(csv.endswith(",1.5,2.5,"))

    def test_bar_header_matches_the_legacy_column_order(self):
        self.assertEqual(
            exports.BAR_HEADER,
            [
                "database", "isotope", "dataset", "series",
                "group_index", "mt", "mt_label",
                "bin_start", "bin_end", "bin_center", "bin_width",
                "point_count", "coverage_percent",
                "cross_section_mean", "cross_section_group_average",
                "cross_section_integral",
                "group_structure_name", "rebinning_method",
            ],
        )


if __name__ == "__main__":
    unittest.main()


class ComparisonCsvTests(unittest.TestCase):
    """The export must be readable as the numbers actually plotted."""

    def build(self):
        from backend.dash_ui import presenters
        from backend.services import analytics
        from tests.fixtures import make_series

        reference = make_series("A|I|D", [(1, 1.0), (2, 2.0), (3, 4.0)])
        comparison = make_series("B|I|D", [(1, 2.0), (2, 2.0), (3, 2.0)])
        results = analytics.compare_cross_sections(reference, [comparison])
        labels = {"A|I|D": "ENDF/B-VIII.0", "B|I|D": "JEFF-3.3"}
        return results, labels, presenters

    def rows(self, content):
        lines = content.split("\n")
        header = lines[0].split(",")
        return [dict(zip(header, line.split(","))) for line in lines[1:]]

    def test_the_header_is_the_documented_one(self):
        self.assertEqual(
            exports.COMPARISON_HEADER[:5],
            [
                "energy_MeV",
                "reference_series",
                "comparison_series",
                "reference_cross_section_b",
                "comparison_cross_section_b",
            ],
        )
        self.assertEqual(
            exports.COMPARISON_HEADER[5:],
            [
                "ratio",
                "percent_difference",
                "dominant_series",
                "valid",
                "invalid_reason",
                "is_crossing",
            ],
        )

    def test_exported_values_match_the_plotted_values(self):
        results, labels, presenters = self.build()
        plotted = presenters.comparison_rows(results, labels, "ratio")[0]

        rows = self.rows(presenters.comparison_csv(results, labels))

        self.assertEqual(len(rows), plotted["values"].size)
        for row, value in zip(rows, plotted["values"]):
            self.assertEqual(row["ratio"], exports.format_number(float(value)))

    def test_the_orientation_matches_the_chart(self):
        results, labels, presenters = self.build()

        rows = self.rows(presenters.comparison_csv(results, labels))

        # Comparison 2.0 over reference 1.0 is 2, not 0.5.
        self.assertEqual(rows[0]["reference_series"], "ENDF/B-VIII.0")
        self.assertEqual(rows[0]["comparison_series"], "JEFF-3.3")
        self.assertEqual(rows[0]["ratio"], "2")

    def test_undefined_rows_are_kept_with_a_reason(self):
        from backend.dash_ui import presenters
        from backend.services import analytics
        from tests.fixtures import make_series

        reference = make_series("A|I|D", [(1, 0.0), (2, 1.0)])
        comparison = make_series("B|I|D", [(1, 5.0), (2, 1.0)])
        results = analytics.compare_cross_sections(reference, [comparison])
        labels = {"A|I|D": "REF", "B|I|D": "CMP"}

        rows = self.rows(presenters.comparison_csv(results, labels))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["ratio"], "")
        self.assertEqual(rows[0]["percent_difference"], "")
        self.assertEqual(rows[0]["invalid_reason"], "reference_zero")
        self.assertEqual(rows[0]["valid"], "false")
        # The source values survive so the reader can see what was there.
        self.assertEqual(rows[0]["reference_cross_section_b"], "0")
        self.assertEqual(rows[0]["comparison_cross_section_b"], "5")

    def test_crossings_are_flagged(self):
        from backend.dash_ui import presenters
        from backend.services import analytics
        from tests.fixtures import make_series

        reference = make_series("A|I|D", [(0, 0.0), (10, 10.0)])
        comparison = make_series("B|I|D", [(0, 10.0), (10, 0.0)])
        results = analytics.compare_cross_sections(reference, [comparison])
        labels = {"A|I|D": "REF", "B|I|D": "CMP"}

        rows = self.rows(presenters.comparison_csv(results, labels))
        flagged = [row for row in rows if row["is_crossing"] == "true"]

        self.assertEqual(len(flagged), 1)
        self.assertEqual(flagged[0]["energy_MeV"], "5")
        self.assertEqual(flagged[0]["ratio"], "1")
