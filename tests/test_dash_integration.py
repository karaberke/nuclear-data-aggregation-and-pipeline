"""Tier 3: the real Dash wire protocol over the FastAPI app.

These POST to `_dash-update-component` exactly as the browser does, which is
what catches serialization failures - numpy scalars, `None` inside
customdata, oversized figures - that calling the functions directly would
not. The output-id strings are derived from `_dash-dependencies` rather than
hardcoded, because `allow_duplicate` appends a content hash to them.
"""

import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app
from backend.services.cache import get_cache

SESSION = "0" * 32

# A small synthetic table, so no JANIS subprocess is involved.
FAKE_TABLE = tuple((10.0**exponent, 2.0 + exponent) for exponent in range(-4, 4))


def fake_parsed_table(database, isotope, dataset, value, field="SIG"):
    if value == "xs_stddev":
        return tuple((energy, sigma * 0.1) for energy, sigma in FAKE_TABLE)
    return FAKE_TABLE


class WireProtocolTestCase(unittest.TestCase):
    """Shared plumbing for driving callbacks over HTTP."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.dependencies = cls.client.get("/_dash-dependencies").json()

    def setUp(self):
        get_cache().clear()

    def stub_janis(
        self,
        *,
        table=fake_parsed_table,
        databases=("DB-A", "DB-B"),
        isotopes=("Co59",),
        datasets=("MT102",),
        quantities=("xs", "xs_stddev"),
    ):
        """Patch every JANIS entry point for the rest of the test.

        `enterContext` (TestCase, 3.11+) unwinds these at tearDown, so there
        is no `with` block and no chance of a patch leaking when an assertion
        fails mid-test - which the hand-written `try/finally` stacks this
        replaces could do. The same five-patch stanza was previously repeated
        seven times, twice with drifted return values.

        Pass `table=None` for the callbacks that must not reach the parsed
        table at all; every other argument overrides one return value.
        """
        if table is not None:
            self.enterContext(
                patch("backend.charts.get_parsed_table", side_effect=table)
            )
        for name, value in (
            ("list_databases", databases),
            ("list_isotopes", isotopes),
            ("list_all_datasets", datasets),
            ("list_quantities", quantities),
        ):
            self.enterContext(
                patch(f"backend.jar_runner.{name}", return_value=list(value))
            )

    def callback_for(self, trigger: str):
        """Find the callback whose Inputs include `id.property`."""
        for callback in self.dependencies:
            for item in callback["inputs"]:
                if f"{item['id']}.{item['property']}" == trigger:
                    return callback
        raise AssertionError(f"no callback triggered by {trigger}")

    @staticmethod
    def parse_outputs(output: str) -> list[dict] | dict:
        """Turn Dash's `output` string into the `outputs` spec it expects.

        Multi-output callbacks encode their targets as
        `..id.prop...id.prop..`, and `allow_duplicate` appends `@<hash>` to
        the property - which must be preserved.
        """
        if output.startswith("..") and output.endswith(".."):
            parts = output[2:-2].split("...")
            specs = []
            for part in parts:
                identifier, _, prop = part.rpartition(".")
                specs.append({"id": identifier, "property": prop})
            return specs
        identifier, _, prop = output.rpartition(".")
        return {"id": identifier, "property": prop}

    def callback_for_output(self, target: str):
        """Find the callback that writes `id.property`.

        Needed when a callback shares every one of its inputs with another -
        the ratio chart's formula caption reacts to the same three inputs the
        chart itself does, so triggers alone cannot tell them apart.
        """
        for callback in self.dependencies:
            if target in callback["output"]:
                return callback
        raise AssertionError(f"no callback writes {target}")

    def fire(
        self,
        trigger: str,
        values: dict,
        state: dict | None = None,
        output: str | None = None,
    ):
        """Invoke a callback the way the browser would."""
        callback = (
            self.callback_for_output(output)
            if output
            else self.callback_for(trigger)
        )
        state = state or {}

        def resolve(spec, source):
            key = f"{spec['id']}.{spec['property']}"
            return {
                "id": spec["id"],
                "property": spec["property"],
                "value": source.get(key),
            }

        payload = {
            "output": callback["output"],
            "outputs": self.parse_outputs(callback["output"]),
            "inputs": [resolve(spec, values) for spec in callback["inputs"]],
            "state": [resolve(spec, state) for spec in callback.get("state", [])],
            "changedPropIds": [trigger],
        }
        response = self.client.post(
            "/_dash-update-component",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        return response

    def run_a_query(self, reaction_type="xs"):
        """Drive the Run button and return the resulting query-store payload."""
        self.stub_janis()
        response = self.fire(
            "run-btn.n_clicks",
            {"run-btn.n_clicks": 1},
            {
                "session-store.data": {"sid": SESSION},
                "comparison-mode.value": "databases",
                "databases.value": ["DB-A", "DB-B"],
                "isotopes.value": "Co59",
                "datasets.value": "MT102",
                "reaction-type.value": reaction_type,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["response"]


class QueryWorkflowTests(WireProtocolTestCase):
    def test_run_populates_the_store_and_legend(self):
        result = self.run_a_query()
        store = result["query-store"]["data"]
        self.assertEqual(len(store["series"]), 2)
        self.assertEqual(store["sid"], SESSION)
        self.assertIn("Loaded 2 series", result["status-line"]["children"])
        self.assertEqual(len(result["series-legend"]["value"]), 2)
        # The panels themselves are not this callback's job - they follow the
        # store, in ResultPanelTests below.

    def test_changing_the_selection_clears_the_store(self):
        """The first half of the panel-staleness chain.

        C2 (comparison mode) and C3 (databases) must both drop the loaded
        result, because the panels key off exactly this.
        """
        self.run_a_query()
        switched = self.fire(
            "comparison-mode.value", {"comparison-mode.value": "isotopes"}
        ).json()["response"]
        narrowed = self.fire(
            "databases.value",
            {"databases.value": ["DB-A"]},
            {"comparison-mode.value": "databases"},
        ).json()["response"]

        self.assertIsNone(switched["query-store"]["data"])
        self.assertIsNone(narrowed["query-store"]["data"])


class ResultPanelTests(WireProtocolTestCase):
    """The panels must never outlive the result they describe.

    Regression: C6 was the only writer of these four properties, so they only
    changed when a query *ran*. Changing the databases or the comparison mode
    after a run left the Analysis panel on screen against a selection it no
    longer matched, with both export buttons still live.
    """

    def panels_for(self, store):
        # Addressed by output, not trigger: `query-store.data` is also an
        # Input to the chart-render callback, which the trigger lookup would
        # find first.
        response = self.fire(
            "query-store.data",
            {"query-store.data": store},
            output="analysis-section.style",
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["response"]

    def test_a_cleared_store_hides_the_panels(self):
        result = self.panels_for(None)
        self.assertEqual(result["analysis-section"]["style"], {"display": "none"})
        self.assertEqual(result["chart-section"]["style"], {"display": "none"})
        self.assertTrue(result["export-btn"]["disabled"])
        self.assertTrue(result["export-processed-btn"]["disabled"])

    def test_a_cleared_store_also_empties_the_legend(self):
        """A legend listing the previous comparison's series is stale UI.

        Hiding the section is enough today, but only because nothing else
        reveals it. Emptying the legend makes it true regardless.
        """
        result = self.panels_for(None)
        self.assertEqual(result["series-legend"]["options"], [])
        self.assertEqual(result["series-legend"]["value"], [])

    def test_a_loaded_store_leaves_the_legend_to_the_run_callback(self):
        """`no_update`, so the run callback's labels stand.

        Returning real values here would also re-fire the chart render, which
        reads series-legend.value as an Input.
        """
        store = self.run_a_query()["query-store"]["data"]
        result = self.panels_for(store)
        self.assertNotIn(
            "series-legend", result, "C16 must not overwrite a live legend"
        )

    def test_a_loaded_store_reveals_the_panels_as_blocks(self):
        """`display: block`, not flex.

        SHOWN is `display: flex`, which on a <section> lays the whole control
        stack out beside the chart instead of above it.
        """
        store = self.run_a_query()["query-store"]["data"]
        result = self.panels_for(store)
        self.assertEqual(result["analysis-section"]["style"], {"display": "block"})
        self.assertEqual(result["chart-section"]["style"], {"display": "block"})
        self.assertFalse(result["export-btn"]["disabled"])
        self.assertFalse(result["export-processed-btn"]["disabled"])

    def test_store_payload_carries_no_data_points(self):
        """The 30,000-point arrays must never reach the browser."""
        store = self.run_a_query()["query-store"]["data"]
        serialized = json.dumps(store)
        self.assertLess(len(serialized), 4096, "query store payload too large")
        for series in store["series"]:
            self.assertNotIn("points", series)
            self.assertNotIn("energy", series)

    def test_store_payload_is_json_serializable(self):
        """numpy scalars would raise here rather than in the browser."""
        store = self.run_a_query()["query-store"]["data"]
        json.dumps(store)
        for series in store["series"]:
            self.assertIsInstance(series["n_points"], int)

    def test_each_series_gets_a_distinct_palette_color(self):
        store = self.run_a_query()["query-store"]["data"]
        colors = [series["color"] for series in store["series"]]
        self.assertEqual(len(set(colors)), len(colors))


class RenderTests(WireProtocolTestCase):
    def base_inputs(self, store, chart="line", **overrides):
        values = {
            "query-store.data": store,
            "series-legend.value": [s["key"] for s in store["series"]],
            "chart-tabs.value": chart,
            "axis-store.data": {
                "line": {"x": "log", "y": "log"},
                "bar": {"x": "log", "y": "log"},
                "kde": {"x": "log", "y": "log"},
                "comparison": {"x": "log", "y": "linear"},
            },
            "energy-min.value": None,
            "energy-max.value": None,
            "value-min.value": None,
            "value-max.value": None,
            "group-structure-mode.value": "automatic",
            "bin-count.value": 40,
            "group-energy-min.value": None,
            "group-energy-max.value": None,
            "group-structure-preset.value": "VITAMIN-J175",
            "bin-edges.value": "",
            "bar-mode.value": "mean",
            "bandwidth.value": 1,
            "comparison-metric.value": "ratio",
            "comparison-reference.value": None,
            "comparison-targets.value": None,
        }
        values.update(overrides)
        return values

    def render(self, store, chart="line", **overrides):
        self.stub_janis()
        response = self.fire(
            "query-store.data", self.base_inputs(store, chart, **overrides)
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["response"]

    def test_line_chart_renders_one_trace_per_series(self):
        store = self.run_a_query()["query-store"]["data"]
        result = self.render(store, "line")
        figure = result["chart-graph"]["figure"]
        self.assertEqual(len(figure["data"]), 2)
        self.assertEqual(figure["layout"]["xaxis"]["type"], "log")

    def test_hiding_a_series_drops_its_trace(self):
        store = self.run_a_query()["query-store"]["data"]
        only_first = [store["series"][0]["key"]]
        result = self.render(store, "line", **{"series-legend.value": only_first})
        self.assertEqual(len(result["chart-graph"]["figure"]["data"]), 1)

    def test_bar_chart_renders_and_reports_a_notice(self):
        store = self.run_a_query()["query-store"]["data"]
        result = self.render(store, "bar")
        self.assertGreater(len(result["chart-graph"]["figure"]["data"]), 0)
        self.assertIn("Showing", result["chart-notice"]["children"])

    def test_kde_chart_renders(self):
        store = self.run_a_query()["query-store"]["data"]
        result = self.render(store, "kde")
        figure = result["chart-graph"]["figure"]
        self.assertEqual(len(figure["data"]), 2)
        self.assertEqual(
            figure["layout"]["yaxis"]["title"]["text"], "Probability Density"
        )

    def test_bar_mode_switch_changes_the_axis_title(self):
        store = self.run_a_query()["query-store"]["data"]
        mean = self.render(store, "bar", **{"bar-mode.value": "mean"})
        density = self.render(store, "bar", **{"bar-mode.value": "density"})
        self.assertEqual(
            mean["chart-graph"]["figure"]["layout"]["yaxis"]["title"]["text"],
            "Mean Cross Section (barns)",
        )
        self.assertEqual(
            density["chart-graph"]["figure"]["layout"]["yaxis"]["title"]["text"],
            "Energy-averaged cross section (barns)",
        )

    def test_standard_structure_is_applied(self):
        store = self.run_a_query()["query-store"]["data"]
        result = self.render(
            store,
            "bar",
            **{"group-structure-mode.value": "standard"},
        )
        self.assertGreater(len(result["chart-graph"]["figure"]["data"]), 0)

    def test_bin_edges_autofill_fires_once_then_stops(self):
        """The self-loop must terminate in exactly two passes."""
        store = self.run_a_query()["query-store"]["data"]
        first = self.render(store, "bar", **{"bin-edges.value": ""})
        filled = first["bin-edges"]["value"]
        self.assertTrue(filled, "edges should be auto-filled when blank")

        second = self.render(store, "bar", **{"bin-edges.value": filled})
        self.assertNotIn(
            "bin-edges", second, "second pass must not rewrite the edges"
        )

    def test_invalid_filter_range_shows_a_message_not_a_traceback(self):
        store = self.run_a_query()["query-store"]["data"]
        result = self.render(
            store, "line", **{"energy-min.value": 10, "energy-max.value": 1}
        )
        self.assertEqual(
            result["chart-error"]["children"],
            "Minimum energy must be less than maximum energy.",
        )
        figure = result["chart-graph"]["figure"]
        self.assertEqual(len(figure["data"]), 0)
        self.assertEqual(len(figure["layout"]["annotations"]), 1)

    def test_malformed_custom_edges_show_an_inline_error(self):
        store = self.run_a_query()["query-store"]["data"]
        result = self.render(
            store,
            "bar",
            **{
                "group-structure-mode.value": "custom",
                "bin-edges.value": "1, two, 3",
            },
        )
        self.assertEqual(
            result["chart-error"]["children"],
            "Bin edges must be a list of numbers.",
        )

    def test_descending_custom_edges_show_an_inline_error(self):
        store = self.run_a_query()["query-store"]["data"]
        result = self.render(
            store,
            "bar",
            **{
                "group-structure-mode.value": "custom",
                "bin-edges.value": "10, 5, 1",
            },
        )
        self.assertEqual(
            result["chart-error"]["children"],
            "Bin edges must be strictly ascending.",
        )

    def test_filters_that_remove_everything_report_it(self):
        store = self.run_a_query()["query-store"]["data"]
        result = self.render(store, "line", **{"energy-min.value": 1e30})
        self.assertIn(
            "No points remain", str(result["chart-graph"]["figure"])
        )

    def test_uncertainty_adds_a_band_trace(self):
        store = self.run_a_query(reaction_type="xs_stddev")["query-store"]["data"]
        result = self.render(store, "line")
        # One band + one line per series.
        self.assertEqual(len(result["chart-graph"]["figure"]["data"]), 4)


class AxisMemoryTests(WireProtocolTestCase):
    def test_each_chart_type_remembers_its_own_scales(self):
        stored = {
            "line": {"x": "log", "y": "log"},
            "bar": {"x": "linear", "y": "linear"},
            "kde": {"x": "log", "y": "linear"},
        }
        response = self.fire(
            "chart-tabs.value",
            {
                "chart-tabs.value": "bar",
                "x-scale.value": "log",
                "y-scale.value": "log",
            },
            {"axis-store.data": stored},
        )
        result = response.json()["response"]
        self.assertEqual(result["x-scale"]["value"], "linear")
        self.assertEqual(result["y-scale"]["value"], "linear")
        # Switching tabs must not rewrite the store.
        self.assertNotIn("axis-store", result)

    def test_changing_a_scale_writes_only_the_active_chart(self):
        stored = {
            "line": {"x": "log", "y": "log"},
            "bar": {"x": "log", "y": "log"},
            "kde": {"x": "log", "y": "log"},
        }
        response = self.fire(
            "x-scale.value",
            {
                "chart-tabs.value": "kde",
                "x-scale.value": "linear",
                "y-scale.value": "log",
            },
            {"axis-store.data": stored},
        )
        result = response.json()["response"]
        updated = result["axis-store"]["data"]
        self.assertEqual(updated["kde"], {"x": "linear", "y": "log"})
        self.assertEqual(updated["line"], {"x": "log", "y": "log"})
        # No write-back into the selects, which is what breaks the loop.
        self.assertNotIn("x-scale", result)

    def test_bar_controls_appear_only_on_the_bar_tab(self):
        stored = {"bar": {"x": "log", "y": "log"}}
        result = self.fire(
            "chart-tabs.value",
            {
                "chart-tabs.value": "bar",
                "x-scale.value": "log",
                "y-scale.value": "log",
            },
            {"axis-store.data": stored},
        ).json()["response"]
        self.assertEqual(result["bin-controls"]["style"], {"display": "flex"})
        self.assertEqual(result["kde-controls"]["style"], {"display": "none"})


class DownloadTests(WireProtocolTestCase):
    def test_filtered_export_returns_csv_content(self):
        store = self.run_a_query()["query-store"]["data"]
        self.stub_janis()
        response = self.fire(
            "export-btn.n_clicks",
            {"export-btn.n_clicks": 1},
            {
                "query-store.data": store,
                "series-legend.value": [
                    s["key"] for s in store["series"]
                ],
                "energy-min.value": None,
                "energy-max.value": None,
                "value-min.value": None,
                "value-max.value": None,
            },
        )
        payload = response.json()["response"]["download-filtered"]["data"]
        self.assertEqual(
            payload["filename"], "nuclear-cross-section-comparison.csv"
        )
        lines = payload["content"].splitlines()
        self.assertEqual(
            lines[0],
            "database,isotope,dataset,energy_MeV,cross_section_barns,"
            "cross_section_stddev_barns",
        )
        self.assertEqual(len(lines), 1 + 2 * len(FAKE_TABLE))

    def test_processed_export_matches_the_active_chart(self):
        store = self.run_a_query()["query-store"]["data"]
        self.stub_janis()
        response = self.fire(
            "export-processed-btn.n_clicks",
            {"export-processed-btn.n_clicks": 1},
            {
                "query-store.data": store,
                "series-legend.value": [
                    s["key"] for s in store["series"]
                ],
                "chart-tabs.value": "bar",
                "axis-store.data": {"bar": {"x": "log", "y": "log"}},
                "energy-min.value": None,
                "energy-max.value": None,
                "value-min.value": None,
                "value-max.value": None,
                "group-structure-mode.value": "automatic",
                "bin-count.value": 40,
                "group-energy-min.value": None,
                "group-energy-max.value": None,
                "group-structure-preset.value": "VITAMIN-J175",
                "bin-edges.value": "",
                "bar-mode.value": "mean",
                "bandwidth.value": 1,
            },
        )
        payload = response.json()["response"]["download-processed"]["data"]
        self.assertEqual(
            payload["filename"], "nuclear-cross-section-processed-bar.csv"
        )
        header = payload["content"].splitlines()[0]
        self.assertTrue(header.startswith("database,isotope,dataset,series,"))
        self.assertIn("cross_section_group_average", header)
        self.assertIn("rebinning_method", header)


class CacheRecoveryTests(WireProtocolTestCase):
    def test_an_evicted_result_re_materializes_rather_than_failing(self):
        """The handle embeds the query, so a miss costs latency, not an error."""
        store = self.run_a_query()["query-store"]["data"]
        get_cache().clear()

        self.stub_janis()
        response = self.fire(
            "query-store.data",
            RenderTests.base_inputs(self, store, "line"),
        )
        result = response.json()["response"]
        self.assertEqual(len(result["chart-graph"]["figure"]["data"]), 2)
        self.assertEqual(result["chart-error"]["children"], "")


if __name__ == "__main__":
    unittest.main()


# The two libraries must actually differ for a ratio to be interesting; the
# shared FAKE_TABLE returns the same points for every database.
def diverging_parsed_table(database, isotope, dataset, value, field="SIG"):
    if value == "xs_stddev":
        return tuple((energy, sigma * 0.1) for energy, sigma in FAKE_TABLE)
    if database == "DB-B":
        # Crosses the reference part-way along, so a baseline crossing exists.
        return tuple(
            (energy, sigma * (0.5 + index * 0.4))
            for index, (energy, sigma) in enumerate(FAKE_TABLE)
        )
    return FAKE_TABLE


class ComparisonChartTests(RenderTests):
    """Tier 3 for the Ratio / Difference tab.

    Worth driving over the wire rather than calling the builder directly: the
    merged grid carries NaN at every undefined point and a string `hovertext`
    column beside a float64 `customdata` block, and only a real serialization
    round trip proves Dash accepts that combination.
    """

    def render_comparison(self, store, **overrides):
        self.stub_janis(table=diverging_parsed_table)
        response = self.fire(
            "query-store.data",
            self.base_inputs(store, "comparison", **overrides),
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["response"]

    def test_the_comparison_chart_serializes_over_the_wire(self):
        store = self.run_a_query()["query-store"]["data"]

        result = self.render_comparison(store)
        figure = result["chart-graph"]["figure"]

        self.assertEqual(len(figure["data"]), 1)
        self.assertIn("Showing", result["chart-notice"]["children"])
        self.assertEqual(result["chart-error"]["children"], "")

    def test_ratio_mode_labels_and_baseline(self):
        store = self.run_a_query()["query-store"]["data"]

        figure = self.render_comparison(store)["chart-graph"]["figure"]

        self.assertEqual(
            figure["layout"]["yaxis"]["title"]["text"],
            "Cross-Section Ratio (comparison / reference)",
        )
        self.assertEqual(figure["layout"]["yaxis"]["type"], "linear")
        self.assertEqual(figure["layout"]["shapes"][0]["y0"], 1)

    def test_percent_mode_labels_and_baseline(self):
        store = self.run_a_query()["query-store"]["data"]

        figure = self.render_comparison(
            store, **{"comparison-metric.value": "percent"}
        )["chart-graph"]["figure"]

        self.assertEqual(
            figure["layout"]["yaxis"]["title"]["text"],
            "Difference Relative to Reference (%)",
        )
        self.assertEqual(figure["layout"]["yaxis"]["type"], "linear")
        self.assertEqual(figure["layout"]["shapes"][0]["y0"], 0)

    def test_swapping_the_reference_inverts_the_reading(self):
        store = self.run_a_query()["query-store"]["data"]
        first, second = (series["key"] for series in store["series"])

        forward = self.render_comparison(
            store,
            **{
                "comparison-reference.value": first,
                "comparison-targets.value": [second],
            },
        )["chart-graph"]["figure"]
        backward = self.render_comparison(
            store,
            **{
                "comparison-reference.value": second,
                "comparison-targets.value": [first],
            },
        )["chart-graph"]["figure"]

        self.assertNotEqual(forward["data"][0]["name"], backward["data"][0]["name"])
        self.assertIn(" / ", forward["data"][0]["name"])

    def test_value_filters_do_not_apply_to_this_chart(self):
        store = self.run_a_query()["query-store"]["data"]

        result = self.render_comparison(
            store, **{"value-min.value": 1e9, "value-max.value": 1e10}
        )

        # A value window that empties every other chart still draws here.
        self.assertEqual(len(result["chart-graph"]["figure"]["data"]), 1)
        self.assertIn(
            "Min/Max cross section do not apply",
            result["chart-notice"]["children"],
        )

    def test_an_incompatible_comparison_mode_explains_itself(self):
        """Comparing MT channels is rejected, never silently plotted."""
        self.stub_janis(databases=["DB-A"], datasets=["MT102", "MT16"])
        response = self.fire(
            "run-btn.n_clicks",
            {"run-btn.n_clicks": 1},
            {
                "session-store.data": {"sid": SESSION},
                "comparison-mode.value": "datasets",
                "databases.value": "DB-A",
                "isotopes.value": "Co59",
                "datasets.value": ["MT102", "MT16"],
                "reaction-type.value": "xs",
            },
        )
        store = response.json()["response"]["query-store"]["data"]

        result = self.render_comparison(store)

        message = result["chart-error"]["children"]
        self.assertIn("MT102", message)
        self.assertIn("MT16", message)
        self.assertEqual(len(result["chart-graph"]["figure"]["data"]), 0)

    def selectors(self, store, reference=None, targets=None, clicks=0):
        """Drive C14. Triggered via `comparison-swap`, which is its own input:
        `query-store.data` would match the chart-render callback first."""
        first, second = (series["key"] for series in store["series"])
        response = self.fire(
            "comparison-swap.n_clicks",
            {
                "query-store.data": store,
                "series-legend.value": [first, second],
                "comparison-swap.n_clicks": clicks,
            },
            {
                "comparison-reference.value": reference,
                "comparison-targets.value": targets,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["response"]

    def test_the_selectors_default_to_first_reference_second_comparison(self):
        store = self.run_a_query()["query-store"]["data"]
        first, second = (series["key"] for series in store["series"])

        result = self.selectors(store)

        self.assertEqual(result["comparison-reference"]["value"], first)
        self.assertEqual(result["comparison-targets"]["value"], [second])

    def test_swap_exchanges_the_two_roles(self):
        store = self.run_a_query()["query-store"]["data"]
        first, second = (series["key"] for series in store["series"])

        result = self.selectors(
            store, reference=first, targets=[second], clicks=1
        )

        self.assertEqual(result["comparison-reference"]["value"], second)
        self.assertEqual(result["comparison-targets"]["value"], [first])

    def test_the_reference_is_never_offered_as_its_own_comparison(self):
        store = self.run_a_query()["query-store"]["data"]
        first, _ = (series["key"] for series in store["series"])

        result = self.selectors(store, reference=first)
        offered = [
            option["value"] for option in result["comparison-targets"]["options"]
        ]

        self.assertNotIn(first, offered)

    def test_the_formula_states_the_orientation(self):
        store = self.run_a_query()["query-store"]["data"]
        first, second = (series["key"] for series in store["series"])
        labels = {s["key"]: s["label"] for s in store["series"]}

        response = self.fire(
            "comparison-reference.value",
            {
                "comparison-reference.value": first,
                "comparison-targets.value": [second],
                "query-store.data": store,
            },
            output="comparison-formula.children",
        )

        self.assertEqual(
            response.json()["response"]["comparison-formula"]["children"],
            f"{labels[second]} / {labels[first]}",
        )

    def test_the_processed_export_covers_this_chart(self):
        store = self.run_a_query()["query-store"]["data"]

        self.stub_janis(table=diverging_parsed_table)
        response = self.fire(
            "export-processed-btn.n_clicks",
            {"export-processed-btn.n_clicks": 1},
            {
                "query-store.data": store,
                "series-legend.value": [
                    series["key"] for series in store["series"]
                ],
                "chart-tabs.value": "comparison",
                "comparison-metric.value": "ratio",
            },
        )
        content = response.json()["response"]["download-processed"]["data"][
            "content"
        ]

        self.assertTrue(content.startswith("energy_MeV,reference_series,"))
        self.assertIn("ratio", content.split("\n")[0])
