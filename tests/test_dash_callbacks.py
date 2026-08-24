"""Tier 2: callback helpers and the input-validation layer.

The callback bodies themselves are exercised over the real wire protocol in
tests/test_dash_integration.py, which additionally catches serialization
failures (numpy scalars, None in customdata) that direct calls would miss.
"""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend.dash_ui import callbacks, components, figures, inputs, presenters
from backend.services import analytics

from tests.fixtures import line_series, make_series, simple_series


class LayeringTests(unittest.TestCase):
    def test_callbacks_module_does_no_arithmetic(self):
        """The rule that keeps calculation out of the callback layer.

        Grep-checkable, and cheap enough to assert on every run.
        """
        source = Path(callbacks.__file__).read_text(encoding="utf-8")
        for banned in ("import numpy", "import math", "from numpy", "from math"):
            self.assertNotIn(banned, source, f"callbacks.py must not {banned}")

    def test_the_chart_key_set_has_exactly_one_definition(self):
        """Every enumeration of the four charts must agree.

        The tab strip is the source; `CHART_KEYS`, the axis defaults, and the
        renderer/CSV dispatch in C10/C13 all key off it. These used to be
        written out independently - and the two dispatch ladders even ran in
        different orders - so a fifth chart could be added to one and missed
        by another with nothing failing.
        """
        tabs = {tab["value"] for tab in components.CHART_TABS}
        self.assertEqual(callbacks.CHART_KEYS, tabs)
        self.assertEqual(set(callbacks.AXIS_DEFAULTS), tabs)
        self.assertIn(callbacks.DEFAULT_CHART, tabs)

    def test_both_chart_dispatches_cover_every_chart_key(self):
        """C10 renders and C13 exports the same set of charts.

        Read off the source rather than executed, because driving all four
        needs a loaded query - `tests/test_dash_integration.py` does that. This
        catches the cheap mistake: adding a branch to one ladder only.
        """
        source = Path(callbacks.__file__).read_text(encoding="utf-8")
        render = source.split("def render_chart")[1].split("def ")[0]
        export = source.split("def export_processed_csv")[1].split("def ")[0]
        for key in callbacks.CHART_KEYS:
            # The last chart in each ladder is the unconditional `else`, so it
            # is named by its helper rather than by a literal comparison.
            in_render = f'"{key}"' in render or f"_render_{key}" in render
            in_export = f'"{key}"' in export or f"{key}_csv" in export
            self.assertTrue(in_render, f"C10 does not handle {key}")
            self.assertTrue(in_export, f"C13 does not handle {key}")


class SelectionRuleTests(unittest.TestCase):
    def test_multi_select_requires_two_to_five(self):
        self.assertFalse(callbacks.selection_is_valid("databases", "databases", ["a"]))
        self.assertTrue(callbacks.selection_is_valid("databases", "databases", ["a", "b"]))
        self.assertTrue(
            callbacks.selection_is_valid("databases", "databases", list("abcde"))
        )
        self.assertFalse(
            callbacks.selection_is_valid("databases", "databases", list("abcdef"))
        )

    def test_pinned_dimensions_require_exactly_one(self):
        self.assertFalse(callbacks.selection_is_valid("databases", "isotopes", []))
        self.assertTrue(callbacks.selection_is_valid("databases", "isotopes", ["I"]))
        self.assertFalse(
            callbacks.selection_is_valid("databases", "isotopes", ["I", "J"])
        )

    def test_hints_match_the_mode(self):
        self.assertEqual(callbacks.hint_for("databases", "databases"), "Choose 2–5")
        self.assertEqual(callbacks.hint_for("databases", "isotopes"), "Choose one")

    def test_natural_sort_keeps_mt_codes_in_numeric_order(self):
        self.assertEqual(
            callbacks.natural_sort(["MT102", "MT1", "MT16", "MT2", "MT107"]),
            ["MT1", "MT2", "MT16", "MT102", "MT107"],
        )

    def test_natural_sort_handles_isotope_names(self):
        self.assertEqual(
            callbacks.natural_sort(["U235", "H1", "Co59", "H2"]),
            ["Co59", "H1", "H2", "U235"],
        )

    def test_notice_reports_exclusions_only_when_there_are_some(self):
        self.assertEqual(callbacks.notice(1200, 0), "Showing 1,200 filtered points.")
        self.assertIn("3 nonpositive", callbacks.notice(10, 3))


class FilterValidationTests(unittest.TestCase):
    def test_blank_inputs_mean_no_constraint(self):
        filters = inputs.read_filters(None, "", None, "")
        self.assertIsNone(filters.energy_min)
        self.assertIsNone(filters.value_max)

    def test_inverted_energy_range_is_rejected(self):
        with self.assertRaises(inputs.InputError) as caught:
            inputs.read_filters(10, 1, None, None)
        self.assertEqual(
            str(caught.exception),
            "Minimum energy must be less than maximum energy.",
        )

    def test_inverted_value_range_is_rejected(self):
        with self.assertRaises(inputs.InputError) as caught:
            inputs.read_filters(None, None, 10, 1)
        self.assertEqual(
            str(caught.exception),
            "Minimum cross section must be less than maximum.",
        )

    def test_non_numeric_input_is_rejected_by_label(self):
        with self.assertRaises(inputs.InputError) as caught:
            inputs.read_filters("abc", None, None, None)
        self.assertEqual(str(caught.exception), "Minimum energy must be finite.")


class BinEdgeInputTests(unittest.TestCase):
    def test_blank_falls_back_to_automatic(self):
        self.assertIsNone(inputs.parse_custom_bin_edges(""))
        self.assertIsNone(inputs.parse_custom_bin_edges("   "))
        self.assertIsNone(inputs.parse_custom_bin_edges(None))

    def test_commas_and_whitespace_both_separate(self):
        np.testing.assert_allclose(
            inputs.parse_custom_bin_edges("1, 2  3,4"), [1, 2, 3, 4]
        )

    def test_non_numeric_edges_are_rejected(self):
        with self.assertRaises(inputs.InputError) as caught:
            inputs.parse_custom_bin_edges("1, two, 3")
        self.assertEqual(
            str(caught.exception), "Bin edges must be a list of numbers."
        )

    def test_standard_mode_requires_a_preset(self):
        with self.assertRaises(inputs.InputError):
            inputs.read_bar_controls("standard", 40, None, "", None, None, "mean")

    def test_automatic_mode_rejects_an_inverted_energy_range(self):
        """Mirrors the rule `read_filters` applies to the display filters."""
        with self.assertRaises(inputs.InputError) as caught:
            inputs.read_bar_controls(
                "automatic", 40, None, "", 100, 1, "mean"
            )
        self.assertEqual(
            str(caught.exception), "Min energy must be less than Max energy."
        )

    def test_automatic_mode_rejects_an_empty_energy_range(self):
        with self.assertRaises(inputs.InputError):
            inputs.read_bar_controls("automatic", 40, None, "", 5, 5, "mean")

    def test_automatic_mode_accepts_a_one_sided_bound(self):
        config = inputs.read_bar_controls(
            "automatic", 40, None, "", 1, None, "mean"
        )
        self.assertEqual(config.energy_bounds.minimum, 1)
        self.assertIsNone(config.energy_bounds.maximum)

    def test_standard_mode_loads_the_preset_edges(self):
        config = inputs.read_bar_controls(
            "standard", 40, "VITAMIN-J175", "", None, None, "mean"
        )
        self.assertEqual(config.custom_edges.size, 176)
        self.assertEqual(config.structure_name, "VITAMIN-J175")

    def test_automatic_mode_names_itself_with_the_bin_count(self):
        config = inputs.read_bar_controls(
            "automatic", 40, None, "", None, None, "mean"
        )
        self.assertEqual(config.structure_name, "Automatic (40 bins)")
        self.assertIsNone(config.custom_edges)

    def test_bar_mode_titles_avoid_probability_language(self):
        self.assertEqual(
            inputs.bar_mode_title("density"),
            "Energy-averaged cross section (barns)",
        )
        self.assertEqual(
            inputs.bar_mode_title("mean"), "Mean Cross Section (barns)"
        )
        self.assertNotIn("probability", inputs.bar_mode_title("density").lower())
        self.assertNotIn("barns/MeV", inputs.bar_mode_title("density"))


class SeriesLabelTests(unittest.TestCase):
    def test_label_follows_the_comparison_dimension(self):
        item = simple_series()[0]
        self.assertEqual(inputs.series_label(item, "databases"), "A")
        self.assertEqual(inputs.series_label(item, "isotopes"), "I")
        self.assertEqual(inputs.series_label(item, "datasets"), "D")
        self.assertEqual(inputs.series_label(item, "single"), "A · I · D")

    def test_colors_are_assigned_by_loaded_position(self):
        meta = presenters.series_metadata(simple_series() * 3, "single")
        self.assertEqual(
            [item["color"] for item in meta],
            figures.PALETTE[:3],
        )


class BarGeometryTests(unittest.TestCase):
    """go.Bar positions bars in DATA units, then applies the axis transform."""

    def test_bar_geometry_spans_the_bin_exactly(self):
        """center +/- width/2 must reproduce the original edges.

        This is the invariant that matters; it holds for both axis types
        because Plotly applies the log transform after positioning. An
        earlier revision used the geometric centre with a log10 width, which
        rendered bars ~7x too narrow and truncated the x-range - confirmed
        against rendered pixel positions in a browser.
        """
        starts = np.array([1.0, 10.0, 2.0])
        ends = np.array([10.0, 100.0, 8.0])
        for scale in ("linear", "log"):
            with self.subTest(scale=scale):
                centers, widths = figures._bar_geometry(starts, ends, scale)
                np.testing.assert_allclose(centers - widths / 2, starts)
                np.testing.assert_allclose(centers + widths / 2, ends)

    def test_geometry_is_the_arithmetic_center_on_both_scales(self):
        starts = np.array([1.0])
        ends = np.array([10.0])
        for scale in ("linear", "log"):
            centers, widths = figures._bar_geometry(starts, ends, scale)
            self.assertAlmostEqual(float(centers[0]), 5.5)
            self.assertAlmostEqual(float(widths[0]), 9.0)

    def test_display_bin_center_stays_geometric_on_a_log_scale(self):
        """The tooltip/CSV centre is a separate, physics-facing quantity."""
        result = analytics.bin_series(
            simple_series(), 10, "log", np.array([1.0, 100.0])
        )
        self.assertAlmostEqual(result.records[0].bin_center, 10.0)


class SectionRevealTests(unittest.TestCase):
    """A `<section>` must be revealed as a block, never as a flex row."""

    def test_section_reveal_is_block_not_flex(self):
        """The regression guard for the overlapping-controls bug.

        `SHOWN` is flex, which is right for the inline `.bin-controls-group`
        divs it was written for. Applied to a `<section>` it laid the entire
        control stack out beside the chart and overflowed the viewport.
        """
        self.assertEqual(components.SECTION_SHOWN, {"display": "block"})
        self.assertNotEqual(components.SECTION_SHOWN, components.SHOWN)


class LegendTests(unittest.TestCase):
    def test_base_layout_legend_is_display_only(self):
        """Shown so it reaches the exported PNG; inert so it cannot lie.

        Clicking a Plotly legend entry hides a trace client-side without
        telling the server, which would leave the chart disagreeing with the
        pill checklist that actually drives binning and KDE normalization.
        """
        layout = figures.base_layout("x", "y", "log", "log")
        self.assertTrue(layout.showlegend)
        self.assertFalse(layout.legend.itemclick)
        self.assertFalse(layout.legend.itemdoubleclick)

    def test_bar_figure_emits_one_legend_entry_per_series(self):
        """Coverage splits a series across two traces, not two legend rows.

        The third series below has only partial bins, so keying the legend
        entry off the full-coverage trace would leave it unlabelled.
        """
        series = [
            # Spans both bins fully.
            make_series("W|I|D", [(0.0, 1.0), (10.0, 1.0)]),
            # Spans [0, 7]: bin one full, bin two partial.
            make_series("M|I|D", [(0.0, 1.0), (7.0, 1.0)]),
            # Spans [4, 6]: both bins partial, neither full.
            make_series("N|I|D", [(4.0, 1.0), (6.0, 1.0)]),
        ]
        result = analytics.bin_series(
            series, 2, "linear", np.array([0.0, 5.0, 10.0])
        )
        rows, _ = presenters.bar_rows(
            result,
            {item.key: item for item in series},
            "cross_section_group_average",
            "linear",
        )
        labels = {item.key: item.key for item in series}
        colors = {
            item.key: figures.color_for(index)
            for index, item in enumerate(series)
        }
        figure = figures.bar_figure(
            rows, labels, colors, "linear", "linear", "Value"
        )

        legend_names = [
            trace.name for trace in figure.data if trace.showlegend
        ]
        self.assertEqual(sorted(legend_names), ["M|I|D", "N|I|D", "W|I|D"])
        # The split is real - more traces than legend entries.
        self.assertGreater(len(figure.data), len(legend_names))
        for trace in figure.data:
            self.assertEqual(trace.legendgroup, trace.name)


class MessageFigureTests(unittest.TestCase):
    def test_message_figure_is_never_a_blank_grid(self):
        figure = figures.message_figure("No points remain.")
        self.assertEqual(len(figure.data), 0)
        self.assertEqual(len(figure.layout.annotations), 1)
        self.assertEqual(figure.layout.annotations[0].text, "No points remain.")
        self.assertFalse(figure.layout.xaxis.visible)


class PresenterTests(unittest.TestCase):
    def test_missing_values_render_blank_never_zero(self):
        self.assertEqual(presenters._cell(None), "")
        self.assertEqual(presenters._cell(0.0), "0")

    def test_partial_coverage_is_flagged_per_group(self):
        series = line_series()
        result = analytics.bin_series(
            series, 10, "linear", np.array([-5.0, 5.0, 15.0])
        )
        rows, _ = presenters.bar_rows(
            result,
            {item.key: item for item in series},
            "cross_section_group_average",
            "linear",
        )
        self.assertTrue(rows[0]["partial"].all())

    def test_log_y_excludes_nonpositive_bar_values(self):
        series = line_series()
        result = analytics.bin_series(series, 10, "linear")
        rows, excluded = presenters.bar_rows(
            result,
            {item.key: item for item in series},
            "cross_section_group_average",
            "log",
        )
        self.assertGreaterEqual(excluded, 0)
        for row in rows:
            self.assertTrue((row["value"] > 0).all())


if __name__ == "__main__":
    unittest.main()


class PromptMessageTests(unittest.TestCase):
    """Guards against string-interpolation slips in the status line.

    An earlier revision built these as "Select " + hint.lower() + " isotope.",
    which rendered as "Select choose one isotope."
    """

    def test_prompts_read_as_sentences(self):
        self.assertEqual(
            callbacks.prompt_for("single", "databases"), "Choose a database."
        )
        self.assertEqual(
            callbacks.prompt_for("single", "isotopes"), "Choose an isotope."
        )
        self.assertEqual(
            callbacks.prompt_for("single", "datasets"), "Choose a dataset."
        )

    def test_multi_select_prompts_are_plural(self):
        self.assertEqual(
            callbacks.prompt_for("databases", "databases"),
            "Choose 2–5 databases.",
        )
        self.assertEqual(
            callbacks.prompt_for("isotopes", "isotopes"),
            "Choose 2–5 isotopes.",
        )

    def test_no_prompt_stutters(self):
        for mode in ("single", "databases", "isotopes", "datasets"):
            for key in ("databases", "isotopes", "datasets"):
                message = callbacks.prompt_for(mode, key)
                with self.subTest(mode=mode, key=key):
                    self.assertNotIn("choose", message[1:].lower())
                    self.assertTrue(message.endswith("."))


class RenderingEnvironmentTests(unittest.TestCase):
    """WebGL is absent in many VDI / locked-down corporate browsers."""

    def test_webgl_is_used_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(figures.webgl_enabled())

    def test_force_svg_disables_scattergl(self):
        with patch.dict(os.environ, {"PLOTLY_FORCE_SVG": "1"}, clear=True):
            self.assertFalse(figures.webgl_enabled())

    def test_large_traces_fall_back_to_svg_when_forced(self):
        import numpy as np

        from backend.services import analytics

        big = analytics.SeriesArrays(
            key="A|I|D", database="A", isotope="I", dataset="D",
            energy=np.linspace(1, 1000, 5000),
            sigma=np.linspace(1, 2, 5000),
        )
        prepared, _ = analytics.line_values([big], "linear", "linear", False)
        with patch.dict(os.environ, {"PLOTLY_FORCE_SVG": "1"}, clear=True):
            figure = figures.line_figure(
                prepared, {"A|I|D": "A"}, {"A|I|D": "#0a84ff"},
                "linear", "linear", False,
            )
        self.assertEqual(figure.data[0].type, "scatter")

        with patch.dict(os.environ, {}, clear=True):
            figure = figures.line_figure(
                prepared, {"A|I|D": "A"}, {"A|I|D": "#0a84ff"},
                "linear", "linear", False,
            )
        self.assertEqual(figure.data[0].type, "scattergl")


class ComparisonControlTests(unittest.TestCase):
    def series(self, *keys):
        return [
            make_series(key, [(1, 1.0), (2, 2.0), (3, 4.0)]) for key in keys
        ]

    def test_two_series_default_to_first_reference_second_comparison(self):
        loaded = self.series("A|I|D", "B|I|D")

        config = inputs.read_comparison_controls(None, None, None, loaded)

        self.assertEqual(config.reference.key, "A|I|D")
        self.assertEqual([item.key for item in config.comparisons], ["B|I|D"])

    def test_a_single_series_cannot_be_compared(self):
        with self.assertRaises(inputs.InputError):
            inputs.read_comparison_controls(
                None, None, None, self.series("A|I|D")
            )

    def test_more_than_two_series_compare_every_other_visible_one(self):
        loaded = self.series("A|I|D", "B|I|D", "C|I|D")

        config = inputs.read_comparison_controls("ratio", "B|I|D", None, loaded)

        self.assertEqual(config.reference.key, "B|I|D")
        self.assertEqual(
            sorted(item.key for item in config.comparisons),
            ["A|I|D", "C|I|D"],
        )

    def test_the_reference_is_never_compared_against_itself(self):
        loaded = self.series("A|I|D", "B|I|D")

        config = inputs.read_comparison_controls(
            "ratio", "A|I|D", ["A|I|D", "B|I|D"], loaded
        )

        self.assertNotIn("A|I|D", [item.key for item in config.comparisons])

    def test_an_unknown_reference_falls_back_to_the_first_series(self):
        """The chart must draw before the selector-population callback lands."""
        loaded = self.series("A|I|D", "B|I|D")

        config = inputs.read_comparison_controls("ratio", "gone", None, loaded)

        self.assertEqual(config.reference.key, "A|I|D")

    def test_an_unknown_metric_falls_back_to_ratio(self):
        loaded = self.series("A|I|D", "B|I|D")

        config = inputs.read_comparison_controls("nonsense", None, None, loaded)

        self.assertEqual(config.metric, "ratio")

    def test_the_formula_states_the_orientation(self):
        self.assertEqual(
            inputs.comparison_formula("ENDF/B-VIII.0", "JEFF-3.3"),
            "JEFF-3.3 / ENDF/B-VIII.0",
        )

    def test_value_filters_are_dropped_but_energy_filters_survive(self):
        filters = analytics.RangeFilters(
            energy_min=1.0, energy_max=9.0, value_min=2.0, value_max=8.0
        )

        reduced = inputs.energy_only_filters(filters)

        self.assertEqual(reduced.energy_min, 1.0)
        self.assertEqual(reduced.energy_max, 9.0)
        self.assertIsNone(reduced.value_min)
        self.assertIsNone(reduced.value_max)


class ComparisonFigureTests(unittest.TestCase):
    """The baseline is the whole point: it is what 'larger' is measured from."""

    def rows(self, metric):
        reference = make_series("A|I|D", [(1, 1.0), (2, 2.0), (3, 4.0)])
        comparison = make_series("B|I|D", [(1, 2.0), (2, 2.0), (3, 2.0)])
        results = analytics.compare_cross_sections(reference, [comparison])
        labels = {"A|I|D": "REF", "B|I|D": "CMP"}
        return presenters.comparison_rows(results, labels, metric)

    def test_ratio_mode_baseline_is_one(self):
        figure = figures.comparison_figure(
            self.rows("ratio"), {"B|I|D": "#000"}, "log", "ratio"
        )

        self.assertEqual([shape.y0 for shape in figure.layout.shapes], [1.0])
        self.assertEqual(
            [note.text for note in figure.layout.annotations],
            ["Equal cross sections"],
        )

    def test_percent_mode_baseline_is_zero(self):
        figure = figures.comparison_figure(
            self.rows("percent"), {"B|I|D": "#000"}, "log", "percent"
        )

        self.assertEqual([shape.y0 for shape in figure.layout.shapes], [0.0])
        self.assertEqual(
            [note.text for note in figure.layout.annotations],
            ["No difference"],
        )

    def test_the_y_axis_is_linear_in_both_modes(self):
        for metric in ("ratio", "percent"):
            with self.subTest(metric=metric):
                figure = figures.comparison_figure(
                    self.rows(metric), {"B|I|D": "#000"}, "log", metric
                )
                self.assertEqual(figure.layout.yaxis.type, "linear")

    def test_the_axis_titles_are_the_documented_ones(self):
        self.assertEqual(
            inputs.comparison_axis_title("ratio"),
            "Cross-Section Ratio (comparison / reference)",
        )
        self.assertEqual(
            inputs.comparison_axis_title("percent"),
            "Difference Relative to Reference (%)",
        )

    def test_the_baseline_is_always_inside_the_visible_range(self):
        """A ratio that never approaches 1 must still show the equality line."""
        reference = make_series("A|I|D", [(1, 1.0), (2, 1.0)])
        comparison = make_series("B|I|D", [(1, 50.0), (2, 60.0)])
        results = analytics.compare_cross_sections(reference, [comparison])
        rows = presenters.comparison_rows(
            results, {"A|I|D": "REF", "B|I|D": "CMP"}, "ratio"
        )

        figure = figures.comparison_figure(rows, {"B|I|D": "#000"}, "log", "ratio")
        low, high = figure.layout.yaxis.range

        self.assertLess(low, 1.0)
        self.assertGreater(high, 1.0)

    def test_large_ratios_are_not_clipped(self):
        reference = make_series("A|I|D", [(1, 1.0), (2, 1.0)])
        comparison = make_series("B|I|D", [(1, 1.0), (2, 1000.0)])
        results = analytics.compare_cross_sections(reference, [comparison])
        rows = presenters.comparison_rows(
            results, {"A|I|D": "REF", "B|I|D": "CMP"}, "ratio"
        )

        figure = figures.comparison_figure(rows, {"B|I|D": "#000"}, "log", "ratio")

        self.assertGreaterEqual(figure.layout.yaxis.range[1], 1000.0)

    def test_numeric_customdata_stays_an_ndarray(self):
        """`.tolist()` here would drop the base64 typed-array encoding."""
        rows = self.rows("ratio")

        self.assertIsInstance(rows[0]["customdata"], np.ndarray)
        self.assertEqual(rows[0]["customdata"].dtype, np.float64)
        self.assertEqual(rows[0]["customdata"].shape[1], 4)

    def test_the_hover_names_every_required_field(self):
        figure = figures.comparison_figure(
            self.rows("ratio"), {"B|I|D": "#000"}, "log", "ratio"
        )
        template = figure.data[0].hovertemplate

        for field in ("Energy", "Comparison", "Reference", "Ratio",
                      "Difference", "Dominant"):
            with self.subTest(field=field):
                self.assertIn(field, template)

    def test_gaps_are_not_connected_across_undefined_points(self):
        figure = figures.comparison_figure(
            self.rows("ratio"), {"B|I|D": "#000"}, "log", "ratio"
        )

        self.assertFalse(figure.data[0].connectgaps)


class ComparisonNoticeTests(unittest.TestCase):
    def test_excluded_and_crossing_counts_are_reported(self):
        text = callbacks.comparison_notice(100, 3, 2, False)

        self.assertIn("100 comparison points", text)
        self.assertIn("3 undefined", text)
        self.assertIn("2 baseline crossing", text)

    def test_dropped_value_filters_are_explained(self):
        text = callbacks.comparison_notice(10, 0, 0, True)

        self.assertIn("Min/Max cross section do not apply", text)

    def test_nothing_extra_is_said_when_nothing_was_dropped(self):
        text = callbacks.comparison_notice(10, 0, 0, False)

        self.assertEqual(text, "Showing 10 comparison points.")
