"""The Dash callback graph.

Every callback body is: coerce inputs -> call a service -> build an output ->
handle `InputError`/`DomainError`. No scientific calculation happens here.

Enforced by convention and checkable by grep: **this module must not import
`numpy` or `math`.** `tests/test_dash_callbacks.py` asserts it.
"""

import functools
import logging
import re
from dataclasses import dataclass

from dash import Input, Output, State, ctx, dcc, no_update
from fastapi.concurrency import run_in_threadpool

from .. import charts, jar_runner
from ..services import analytics, query_store
from ..services.errors import DomainError
from ..services.mt_labels import dataset_option_label
from . import components, figures, presenters
from .inputs import (
    InputError,
    as_list,
    bar_mode_title,
    comparison_formula,
    energy_only_filters,
    read_bar_controls,
    read_comparison_controls,
    read_filters,
)

logger = logging.getLogger(__name__)

# One source of truth for the series limit. The UI check in
# `selection_is_valid` and the pydantic bound in `CrossSectionQuery` are
# deliberately both enforced - but off the same number, so they cannot drift
# into the UI offering a series the server will reject.
MAX_SERIES = charts.MAX_QUERY_SERIES
FIELD = "SIG"

# Single source of truth lives in components.py, which layout.py also reads
# for the `axis-store` initial value. Re-exported here so existing references
# (and tests) keep working.
AXIS_DEFAULTS = components.AXIS_DEFAULTS

# Every control that feeds `resolve_chart_data` and the four `compute_*`
# helpers, in the order both callback signatures take them.
#
# Declared once because C10 (the chart) and C13 (its CSV) must read exactly
# the same controls: they already share one compute path so the export cannot
# be computed differently from the chart, but the dependency lists were
# duplicated, so adding a control to one and forgetting the other would put
# the drift back. C10 takes these as Input (re-render on change), C13 as State
# (read on click) - only the id/property pairs are shared, never the type.
CHART_CONTROLS = (
    ("query-store", "data"),
    ("series-legend", "value"),
    ("chart-tabs", "value"),
    ("axis-store", "data"),
    ("energy-min", "value"),
    ("energy-max", "value"),
    ("value-min", "value"),
    ("value-max", "value"),
    ("group-structure-mode", "value"),
    ("bin-count", "value"),
    ("group-energy-min", "value"),
    ("group-energy-max", "value"),
    ("group-structure-preset", "value"),
    ("bin-edges", "value"),
    ("bar-mode", "value"),
    ("bandwidth", "value"),
    ("comparison-metric", "value"),
    ("comparison-reference", "value"),
    ("comparison-targets", "value"),
)
CHART_INPUTS = [Input(cid, prop) for cid, prop in CHART_CONTROLS]
CHART_STATES = [State(cid, prop) for cid, prop in CHART_CONTROLS]

# The one enumeration of the chart keys, taken from the tab strip the user
# actually sees so a new tab cannot appear without this set growing with it.
# C10 dispatches to a `_render_*` and C13 to a `presenters.*_csv` over exactly
# these; the two used to spell the set out independently, in different orders,
# which is how an export ends up handling a chart the renderer does not (or
# the reverse). They remain separate if/elif ladders on purpose - the
# per-chart arguments are heterogeneous (bar takes seven extras, kde one,
# comparison three, line none), so a uniform dispatch table would need every
# renderer to accept one bundled context, which is more indirection than the
# branches it would replace. Sharing the *set* removes the drift that matters.
CHART_KEYS = frozenset(tab["value"] for tab in components.CHART_TABS)
DEFAULT_CHART = "line"

# One name for the user-correctable family. Three different tuples used to be
# spelled out across C10 and C13; each was individually reachable-correct, but
# a reader could not tell "deliberately narrow" from "someone forgot one".
USER_CORRECTABLE = (
    InputError,
    DomainError,
    analytics.BinEdgeError,
    analytics.IncompatibleSeriesError,
)

CACHE_MISS_MESSAGE = (
    "This result is no longer cached. Press Run comparison to reload it."
)

# Shown from the instant Run is pressed until C6 returns. Until this existed
# the first Run gave no feedback at all - the chart panel that holds the only
# dcc.Loading is still display:none at that point - so a query that can
# legitimately take minutes looked like the click had done nothing, which is
# what had users pressing Run again and burning a second JANIS slot.
RUNNING_MESSAGE = "Querying JANIS… large selections can take a few minutes."

NO_COMPARISON_MESSAGE = (
    "No comparison points remain for the selected energy range."
)

ALL_UNDEFINED_MESSAGE = (
    "Every comparison point is undefined: the reference cross section is "
    "zero or missing across the whole overlapping energy range."
)


# --------------------------------------------------------------------------
# Helpers (no Dash types below this line except in the callbacks themselves)
# --------------------------------------------------------------------------


def is_multi(mode: str, key: str) -> bool:
    return mode == key


def hint_for(mode: str, key: str) -> str:
    """Short hint shown under a selector."""
    return "Choose 2–5" if is_multi(mode, key) else "Choose one"


# Singular/plural nouns for the prompt messages.
NOUNS = {
    "databases": ("a database", "2–5 databases"),
    "isotopes": ("an isotope", "2–5 isotopes"),
    "datasets": ("a dataset", "2–5 datasets"),
}


def prompt_for(mode: str, key: str) -> str:
    """Full sentence telling the user what to pick next."""
    singular, plural = NOUNS[key]
    return f"Choose {plural if is_multi(mode, key) else singular}."


def selection_is_valid(mode: str, key: str, values: list[str]) -> bool:
    """The comparison-shape invariant, checked on the UI side.

    Exactly one dimension may be a 2-5 value multi-select; the other two are
    pinned to exactly one value each. `CrossSectionQuery` enforces the same
    rule server-side, so this is defence in depth rather than the only gate.
    """
    if is_multi(mode, key):
        return 2 <= len(values) <= MAX_SERIES
    return len(values) == 1


def build_query(
    databases: list[str],
    isotopes: list[str],
    datasets: list[str],
    reaction_type: str,
) -> charts.CrossSectionQuery:
    return charts.CrossSectionQuery(
        databases=databases,
        isotopes=isotopes,
        datasets=datasets,
        field=FIELD,
        reaction_type=reaction_type,
    )


def restore_series(store: dict | None):
    """Re-materialize the series behind a query handle.

    The handle carries the full query, so a cache miss re-runs
    `query_store.load_query`, which normally hits the parsed-table LRU and
    only reaches JANIS if that expired too. TTL expiry, LRU eviction, and
    worker routing therefore cost latency rather than an error.
    """
    if not store or "query" not in store:
        return None, None
    session_id = query_store.ensure_session_id(store.get("sid"))
    query = build_query(
        store["query"]["databases"],
        store["query"]["isotopes"],
        store["query"]["datasets"],
        store["query"].get("reaction_type", "xs"),
    )
    handle = query_store.load_query(session_id, query)
    return handle, query


def visible_filtered(store, visible_keys, filters):
    """Series that are both legend-visible and inside the filter ranges."""
    handle, query = restore_series(store)
    if handle is None:
        return None, None
    allowed = set(visible_keys or [])
    subset = [item for item in handle.series if item.key in allowed]
    return analytics.filter_series(subset, filters), query


# The two reasons there is nothing to compute, distinguished because the chart
# and the export word them differently.
CACHE_MISS = "cache_miss"
NO_DATA = "no_data"


@dataclass(frozen=True, slots=True)
class ChartData:
    """The resolved inputs a chart and its CSV export are both computed from.

    C10 (`render_chart`) and C13 (`export_processed_csv`) used to derive all of
    this independently: the axis scales, the filters, the comparison-only
    energy-bounds rule, and the visible/filtered series. Two hand-maintained
    copies of one pipeline is how a CSV silently stops matching the chart it
    claims to export, so both now go through `resolve_chart_data`.

    `filters` is the *unnarrowed* set even on the comparison chart, because the
    notice line has to say whether a value bound was ignored - see
    `value_filter_dropped`. The narrowing itself already happened by then.
    """

    chart: str
    x_scale: str
    y_scale: str
    series: list | None
    filters: object
    include_stddev: bool
    unavailable: str | None = None

    @property
    def value_filter_dropped(self) -> bool:
        return (
            self.filters.value_min is not None
            or self.filters.value_max is not None
        )


def resolve_chart_data(
    store, visible, chart, axes,
    energy_min, energy_max, value_min, value_max,
) -> ChartData:
    """Resolve the control panel into the series and scales to compute from.

    Raises `InputError` for bad control values; the callers catch it.
    """
    axes = axes or {}
    scales = axes.get(chart, AXIS_DEFAULTS.get(chart, AXIS_DEFAULTS["line"]))
    filters = read_filters(energy_min, energy_max, value_min, value_max)

    # Energy bounds only on the comparison chart. A cross-section value bound
    # removes points by magnitude and can empty the middle of a curve, which
    # would make the merged-grid interpolation span energies with no source
    # data behind them.
    effective = energy_only_filters(filters) if chart == "comparison" else filters
    series, _ = visible_filtered(store, visible, effective)

    unavailable = None
    if series is None:
        unavailable = CACHE_MISS
    elif not series or all(item.size == 0 for item in series):
        unavailable = NO_DATA

    return ChartData(
        chart=chart,
        x_scale=scales["x"],
        y_scale=scales["y"],
        series=series,
        filters=filters,
        include_stddev=store.get("reaction_type") == "xs_stddev",
        unavailable=unavailable,
    )


# --- Per-chart analytics. Shared so the chart and its CSV can never be
# computed from different arguments. Presentation stays with each caller. ---


def compute_line(data: ChartData):
    return analytics.line_values(
        data.series, data.x_scale, data.y_scale, data.include_stddev
    )


def compute_bar(
    data: ChartData, structure_mode, bin_count, preset, edges_text,
    group_energy_min, group_energy_max, bar_mode,
):
    config = read_bar_controls(
        structure_mode, bin_count, preset, edges_text,
        group_energy_min, group_energy_max, bar_mode,
    )
    result = analytics.bin_series(
        data.series, config.bin_count, data.x_scale,
        config.custom_edges, config.energy_bounds,
    )
    return config, result


def compute_kde(data: ChartData, bandwidth):
    return analytics.kde_series(data.series, data.x_scale, float(bandwidth or 1))


def compute_comparison(data: ChartData, metric, reference_key, target_keys):
    config = read_comparison_controls(
        metric, reference_key, target_keys, data.series
    )
    results = analytics.compare_cross_sections(
        config.reference, config.comparisons
    )
    return config, results


def labels_and_colors(store) -> tuple[dict, dict]:
    meta = (store or {}).get("series", [])
    return (
        {item["key"]: item["label"] for item in meta},
        {item["key"]: item["color"] for item in meta},
    )


def offload(func):
    """Run a callback body in a worker thread so the event loop stays free.

    Dash's FastAPI backend executes a *sync* callback inline on the event
    loop - `response_data = ctx.run(partial_func)` in
    `dash/backends/_fastapi.py::serve_callback`, whose surrounding handler is
    an `async def` registered as a FastAPI route. One slow callback therefore
    blocks every other request in the process, including `/api/health`.
    Measured before this shim: three concurrent 3s queries took 9.0s rather
    than 3.0s, and a health check fired 0.5s in could not even begin until
    all three had finished.

    Returning a coroutine makes Dash `await` the result instead, so the body
    runs in anyio's worker threadpool and the loop stays free. This is also
    what finally makes the JANIS admission gate and semaphore do anything:
    without overlapping arrivals they were never contended.

    `contextvars` propagate through `run_in_threadpool`, so `dash.ctx` and
    `callback_context` still resolve inside the body.
    """

    @functools.wraps(func)
    async def shim(*args, **kwargs):
        return await run_in_threadpool(func, *args, **kwargs)

    return shim


# --- Per-chart renderers. Module level, not nested inside register():
# they close over nothing from it, and CHART_HANDLERS below needs to name
# them. Each returns the same 4-tuple C10 outputs. ---

def _render_line(data, labels, colors, uirevision):
    prepared, excluded = compute_line(data)
    shown = sum(entry["energy"].size for entry in prepared)
    if shown == 0:
        message = "No points remain for the selected filters and scales."
        return figures.message_figure(message), "", "", no_update
    figure = figures.line_figure(
        prepared, labels, colors, data.x_scale, data.y_scale,
        data.include_stddev, uirevision,
    )
    return figure, notice(shown, excluded), "", no_update

def _render_bar(
    data, labels, colors, uirevision,
    structure_mode, bin_count, preset, edges_text,
    group_energy_min, group_energy_max, bar_mode,
):
    config, result = compute_bar(
        data, structure_mode, bin_count, preset, edges_text,
        group_energy_min, group_energy_max, bar_mode,
    )
    series_by_key = {item.key: item for item in data.series}
    rows, excluded = presenters.bar_rows(
        result, series_by_key, config.field, data.y_scale
    )
    if not rows:
        message = "No bins remain for the selected filters and scales."
        return figures.message_figure(message), "", "", no_update

    figure = figures.bar_figure(
        rows, labels, colors, data.x_scale, data.y_scale,
        bar_mode_title(config.mode), uirevision,
    )
    shown = sum(int(row["value"].size) for row in rows)

    # Reproduce the legacy side effect: fill the edge box on the first
    # successful bar render only while it is blank. On the re-trigger the
    # box is populated, so this loop runs exactly twice and stops.
    edges_output = no_update
    if not (edges_text or "").strip() and result.edges.size:
        edges_output = ", ".join(
            f"{float(edge):.6g}" for edge in result.edges
        )

    return (
        figure,
        notice(shown, excluded + result.excluded_nonpositive),
        "",
        edges_output,
    )

def _render_kde(data, labels, colors, bandwidth, uirevision):
    result = compute_kde(data, bandwidth)
    if not result.curves:
        message = "No KDE values remain for the selected filters."
        return figures.message_figure(message), "", "", no_update
    figure = figures.kde_figure(
        result, labels, colors, data.x_scale, data.y_scale, uirevision
    )
    shown = sum(int(curve.x.size) for curve in result.curves)
    return figure, notice(shown, result.excluded_nonpositive), "", no_update

def _render_comparison(
    data, labels, colors, uirevision,
    metric, reference_key, target_keys,
):
    config, results = compute_comparison(
        data, metric, reference_key, target_keys
    )
    rows = presenters.comparison_rows(results, labels, config.metric)
    if not rows:
        # Every pair produced an empty grid; surface the first concrete
        # reason rather than an unexplained blank chart.
        reasons = [
            result.unavailable_reason
            for result in results
            if result.unavailable_reason
        ]
        message = reasons[0] if reasons else NO_COMPARISON_MESSAGE
        return figures.message_figure(message), "", message, no_update

    shown = sum(result.valid_count for result in results)
    if shown == 0:
        return (
            figures.message_figure(ALL_UNDEFINED_MESSAGE),
            "",
            ALL_UNDEFINED_MESSAGE,
            no_update,
        )

    figure = figures.comparison_figure(
        rows, colors, data.x_scale, config.metric, uirevision
    )
    excluded = sum(result.excluded_count for result in results)
    crossings = sum(
        int(result.crossing_energies.size) for result in results
    )
    return (
        figure,
        comparison_notice(
            shown, excluded, crossings, data.value_filter_dropped
        ),
        "",
        no_update,
    )


def register(dash_app) -> None:
    """Attach every callback. Functions stay module-level and importable so
    `tests/test_dash_callbacks.py` can call them directly."""

    # -- C1 -----------------------------------------------------------------
    @dash_app.callback(
        Output("session-store", "data"),
        Input("app-location", "pathname"),
        State("session-store", "data"),
    )
    @offload
    def init_session(_pathname, existing):
        current = (existing or {}).get("sid")
        if query_store.valid_session_id(current):
            return no_update
        return {"sid": query_store.new_session_id()}

    # -- C2 -----------------------------------------------------------------
    @dash_app.callback(
        Output("databases", "multi"),
        Output("databases", "value"),
        Output("databases", "options"),
        Output("databases", "disabled"),
        Output("databases-hint", "children"),
        Output("isotopes", "multi"),
        Output("isotopes", "value"),
        Output("isotopes", "options"),
        Output("isotopes", "disabled"),
        Output("isotopes-hint", "children"),
        Output("datasets", "multi"),
        Output("datasets", "value"),
        Output("datasets", "options"),
        Output("datasets", "disabled"),
        Output("datasets-hint", "children"),
        Output("query-store", "data"),
        Output("status-line", "children"),
        Input("comparison-mode", "value"),
    )
    @offload
    def configure_mode(mode):
        """Changing the comparison dimension clears every selection."""
        try:
            options = [
                {"label": name, "value": name}
                for name in jar_runner.list_databases()
            ]
            status = "Choose the required selections for this mode."
        except RuntimeError as error:
            logger.exception("listing databases failed")
            options = []
            status = f"Could not list databases: {error}"

        return (
            is_multi(mode, "databases"), None, options, False,
            hint_for(mode, "databases"),
            is_multi(mode, "isotopes"), None, [], True,
            hint_for(mode, "isotopes"),
            is_multi(mode, "datasets"), None, [], True,
            hint_for(mode, "datasets"),
            None,
            status,
        )

    # -- C3 -----------------------------------------------------------------
    @dash_app.callback(
        Output("isotopes", "options", allow_duplicate=True),
        Output("isotopes", "value", allow_duplicate=True),
        Output("isotopes", "disabled", allow_duplicate=True),
        Output("datasets", "options", allow_duplicate=True),
        Output("datasets", "value", allow_duplicate=True),
        Output("datasets", "disabled", allow_duplicate=True),
        Output("query-store", "data", allow_duplicate=True),
        Output("status-line", "children", allow_duplicate=True),
        Input("databases", "value"),
        State("comparison-mode", "value"),
        prevent_initial_call=True,
    )
    @offload
    def load_isotopes(databases, mode):
        selected = as_list(databases)
        if not selection_is_valid(mode, "databases", selected):
            return [], None, True, [], None, True, None, prompt_for(
                mode, "databases"
            )
        try:
            common = jar_runner.list_common_isotopes(selected, FIELD)
        except RuntimeError as error:
            logger.exception("listing isotopes failed")
            return [], None, True, [], None, True, None, str(error)

        if not common:
            return [], None, True, [], None, True, None, (
                "No common Isotope options."
            )
        options = [{"label": name, "value": name} for name in natural_sort(common)]
        return options, None, False, [], None, True, None, prompt_for(
            mode, "isotopes"
        )

    # -- C4 -----------------------------------------------------------------
    @dash_app.callback(
        Output("datasets", "options", allow_duplicate=True),
        Output("datasets", "value", allow_duplicate=True),
        Output("datasets", "disabled", allow_duplicate=True),
        Output("query-store", "data", allow_duplicate=True),
        Output("status-line", "children", allow_duplicate=True),
        Input("isotopes", "value"),
        State("databases", "value"),
        State("comparison-mode", "value"),
        prevent_initial_call=True,
    )
    @offload
    def load_datasets(isotopes, databases, mode):
        selected_isotopes = as_list(isotopes)
        selected_databases = as_list(databases)
        if not selection_is_valid(mode, "isotopes", selected_isotopes):
            return [], None, True, None, prompt_for(mode, "isotopes")
        try:
            common = jar_runner.list_common_datasets(
                selected_databases, selected_isotopes, FIELD
            )
        except RuntimeError as error:
            logger.exception("listing datasets failed")
            return [], None, True, None, str(error)

        if not common:
            return [], None, True, None, "No common Dataset options."

        options = [
            {"label": dataset_option_label(name), "value": name}
            for name in natural_sort(common)
        ]
        return options, None, False, None, prompt_for(mode, "datasets")

    # -- C5 -----------------------------------------------------------------
    @dash_app.callback(
        Output("selection-summary", "children"),
        Output("run-btn", "disabled"),
        Input("comparison-mode", "value"),
        Input("databases", "value"),
        Input("isotopes", "value"),
        Input("datasets", "value"),
        Input("reaction-type", "value"),
    )
    @offload
    def update_selection_ui(mode, databases, isotopes, datasets, reaction_type):
        selections = {
            "databases": as_list(databases),
            "isotopes": as_list(isotopes),
            "datasets": as_list(datasets),
        }
        mode_label = next(
            option["label"]
            for option in components.COMPARISON_MODES
            if option["value"] == mode
        )
        summary = "\n".join(
            [
                f"Mode: {mode_label}",
                f"Database: {', '.join(selections['databases']) or '—'}",
                f"Field: {FIELD}",
                f"Isotope: {', '.join(selections['isotopes']) or '—'}",
                f"Dataset: {', '.join(selections['datasets']) or '—'}",
                f"Reaction: {reaction_type}",
            ]
        )
        ready = all(
            selection_is_valid(mode, key, values)
            for key, values in selections.items()
        )
        return summary, not ready

    # -- C6 -----------------------------------------------------------------
    @dash_app.callback(
        Output("query-store", "data", allow_duplicate=True),
        Output("status-line", "children", allow_duplicate=True),
        Output("status-line", "className"),
        Output("series-legend", "options"),
        Output("series-legend", "value"),
        Input("run-btn", "n_clicks"),
        State("session-store", "data"),
        State("comparison-mode", "value"),
        State("databases", "value"),
        State("isotopes", "value"),
        State("datasets", "value"),
        State("reaction-type", "value"),
        # Locks the button for the whole query and says so, client-side.
        #
        # `running` is NOT background-callback-only in Dash 4 - unlike
        # `cancel`/`progress`/`progress_default`, whose docstrings say so
        # explicitly. This app has no callback manager and must not grow one.
        #
        # The renderer applies these "on" values BEFORE issuing the request,
        # so the button greys out on the click itself rather than a round trip
        # later. That is the whole point: one JANIS slot and one JVM per press.
        #
        # No `allow_duplicate=True` here even though C5 owns `run-btn.disabled`
        # and C6 itself owns both `status-line` properties. `running` entries
        # are client-side side-updates keyed by the plain "id.property" string,
        # not registered outputs, so they bypass duplicate validation entirely
        # - and `allow_duplicate` would hash the property name into one that
        # matches no component.
        #
        # The `status-line` off-values are near no-ops on purpose: the renderer
        # applies `runningOff` before the callback's real outputs, and all three
        # of this callback's return paths write both properties, so C6 always
        # wins. They only take effect if the request never returns a payload,
        # where clearing a stale progress message is the correct outcome.
        running=[
            (Output("run-btn", "disabled"), True, False),
            (Output("run-btn", "children"), "Running…", "Run comparison"),
            (Output("status-line", "children"), RUNNING_MESSAGE, ""),
            (Output("status-line", "className"), "", ""),
        ],
        prevent_initial_call=True,
    )
    @offload
    def run_query(
        _clicks, session, mode, databases, isotopes, datasets, reaction_type
    ):
        session_id = query_store.ensure_session_id((session or {}).get("sid"))
        try:
            query = build_query(
                as_list(databases),
                as_list(isotopes),
                as_list(datasets),
                reaction_type,
            )
            handle = query_store.load_query(session_id, query)
        except DomainError as error:
            return None, str(error), "error", [], []
        except Exception as error:  # noqa: BLE001
            logger.exception("query failed")
            return None, f"Query failed: {error}", "error", [], []

        meta = presenters.series_metadata(handle.series, mode)
        store = {
            "sid": session_id,
            "query_id": handle.query_id,
            "comparison_mode": mode,
            "query": query.model_dump(mode="json"),
            "series": meta,
            "reaction_type": reaction_type,
        }
        status = (
            f"Loaded {len(handle.series)} series with "
            f"{handle.total_points:,} points."
        )
        return (
            store,
            status,
            "",
            components.legend_options(meta),
            [item["key"] for item in meta],
        )

    # -- C8 -----------------------------------------------------------------
    @dash_app.callback(
        Output("axis-store", "data"),
        Output("x-scale", "value"),
        Output("y-scale", "value"),
        Output("bin-controls", "style"),
        Output("kde-controls", "style"),
        Output("comparison-controls", "style"),
        Output("y-scale", "disabled"),
        Input("chart-tabs", "value"),
        Input("x-scale", "value"),
        Input("y-scale", "value"),
        State("axis-store", "data"),
    )
    @offload
    def sync_axes(chart, x_scale, y_scale, stored):
        """Per-chart axis memory in one callback.

        Split across two callbacks this becomes a write-back loop: restoring
        the stored values fires the store-writer, which rewrites what it just
        restored. Returning `no_update` on the half that did not trigger
        breaks the cycle - Dash does not re-fire an Input whose Output was
        `no_update`.
        """
        stored = stored or dict(AXIS_DEFAULTS)
        bin_style = components.SHOWN if chart == "bar" else components.HIDDEN
        kde_style = components.SHOWN if chart == "kde" else components.HIDDEN
        comparison = chart == "comparison"
        comparison_style = components.SHOWN if comparison else components.HIDDEN

        if ctx.triggered_id == "chart-tabs" or ctx.triggered_id is None:
            current = stored.get(chart, AXIS_DEFAULTS["line"])
            return (
                no_update, current["x"], current["y"],
                bin_style, kde_style, comparison_style, comparison,
            )

        updated = dict(stored)
        # The comparison chart's y-axis is not user-configurable, so never let
        # a stale dropdown value write "log" back into its slot.
        updated[chart] = {
            "x": x_scale,
            "y": "linear" if comparison else y_scale,
        }
        return (
            updated, no_update, no_update,
            bin_style, kde_style, comparison_style, comparison,
        )

    # -- C9 -----------------------------------------------------------------
    @dash_app.callback(
        Output("automatic-group-controls", "style"),
        Output("standard-group-controls", "style"),
        Output("custom-group-controls", "style"),
        Input("group-structure-mode", "value"),
    )
    @offload
    def toggle_group_structure_controls(mode):
        return (
            components.SHOWN if mode == "automatic" else components.HIDDEN,
            components.SHOWN if mode == "standard" else components.HIDDEN,
            components.SHOWN if mode == "custom" else components.HIDDEN,
        )

    # -- C10 ----------------------------------------------------------------
    @dash_app.callback(
        Output("chart-graph", "figure"),
        Output("chart-notice", "children"),
        Output("chart-error", "children"),
        Output("bin-edges", "value", allow_duplicate=True),
        *CHART_INPUTS,
        prevent_initial_call=True,
    )
    @offload
    def render_chart(
        store, visible, chart, axes,
        energy_min, energy_max, value_min, value_max,
        structure_mode, bin_count, group_energy_min, group_energy_max,
        preset, edges_text, bar_mode, bandwidth,
        comparison_metric, comparison_reference, comparison_targets,
    ):
        if not store:
            return figures.message_figure(), "", "", no_update

        uirevision = f"{store.get('query_id')}:{chart}"
        labels, colors = labels_and_colors(store)

        try:
            data = resolve_chart_data(
                store, visible, chart, axes,
                energy_min, energy_max, value_min, value_max,
            )
            if data.unavailable == CACHE_MISS:
                return (
                    figures.message_figure(CACHE_MISS_MESSAGE),
                    "",
                    CACHE_MISS_MESSAGE,
                    no_update,
                )
            # The comparison chart deliberately does NOT stop on NO_DATA: with
            # too few series `read_comparison_controls` raises a message that
            # names the actual problem ("Select at least two series...") rather
            # than blaming the filters.
            if data.unavailable == NO_DATA and chart != "comparison":
                message = "No points remain for the selected filters."
                return figures.message_figure(message), "", "", no_update

            if chart == "comparison":
                return _render_comparison(
                    data, labels, colors, uirevision,
                    comparison_metric, comparison_reference,
                    comparison_targets,
                )
            if chart == "line":
                return _render_line(data, labels, colors, uirevision)
            if chart == "bar":
                return _render_bar(
                    data, labels, colors, uirevision,
                    structure_mode, bin_count, preset, edges_text,
                    group_energy_min, group_energy_max, bar_mode,
                )
            return _render_kde(data, labels, colors, bandwidth, uirevision)
        except USER_CORRECTABLE as error:
            return figures.message_figure(str(error)), "", str(error), no_update
        except Exception as error:  # noqa: BLE001
            logger.exception("chart render failed")
            message = f"Could not draw the chart: {error}"
            return figures.message_figure(message), "", message, no_update

    # -- C11 ----------------------------------------------------------------
    @dash_app.callback(
        Output("energy-min", "value"),
        Output("energy-max", "value"),
        Output("value-min", "value"),
        Output("value-max", "value"),
        Output("axis-store", "data", allow_duplicate=True),
        Output("x-scale", "value", allow_duplicate=True),
        Output("y-scale", "value", allow_duplicate=True),
        Output("group-structure-mode", "value"),
        Output("bin-count", "value"),
        Output("group-energy-min", "value"),
        Output("group-energy-max", "value"),
        Output("bin-edges", "value", allow_duplicate=True),
        Output("bar-mode", "value"),
        Output("bandwidth", "value"),
        Output("comparison-metric", "value"),
        Input("reset-controls", "n_clicks"),
        prevent_initial_call=True,
    )
    @offload
    def reset_controls(_clicks):
        return (
            None, None, None, None,
            dict(AXIS_DEFAULTS), "log", "log",
            "automatic", 40, None, None, "", "mean", 1, "ratio",
        )

    # -- C12 ----------------------------------------------------------------
    @dash_app.callback(
        Output("download-filtered", "data"),
        Output("chart-error", "children", allow_duplicate=True),
        Input("export-btn", "n_clicks"),
        State("query-store", "data"),
        State("series-legend", "value"),
        State("energy-min", "value"),
        State("energy-max", "value"),
        State("value-min", "value"),
        State("value-max", "value"),
        prevent_initial_call=True,
    )
    @offload
    def export_filtered_csv(
        _clicks, store, visible, energy_min, energy_max, value_min, value_max
    ):
        try:
            filters = read_filters(energy_min, energy_max, value_min, value_max)
            series, _ = visible_filtered(store, visible, filters)
            if series is None:
                return no_update, CACHE_MISS_MESSAGE
            if not series or all(item.size == 0 for item in series):
                return no_update, "No filtered data is available to export."
            content = presenters.filtered_csv(series)
        except (InputError, DomainError) as error:
            return no_update, str(error)

        return (
            dcc.send_string(
                content, "nuclear-cross-section-comparison.csv"
            ),
            "",
        )

    # -- C13 ----------------------------------------------------------------
    @dash_app.callback(
        Output("download-processed", "data"),
        Output("chart-error", "children", allow_duplicate=True),
        Input("export-processed-btn", "n_clicks"),
        *CHART_STATES,
        prevent_initial_call=True,
    )
    @offload
    def export_processed_csv(
        _clicks, store, visible, chart, axes,
        energy_min, energy_max, value_min, value_max,
        structure_mode, bin_count, group_energy_min, group_energy_max,
        preset, edges_text, bar_mode, bandwidth,
        comparison_metric, comparison_reference, comparison_targets,
    ):
        try:
            # Same resolution the chart uses, so the export is of exactly what
            # was plotted - including the comparison chart's energy-only rule.
            data = resolve_chart_data(
                store, visible, chart, axes,
                energy_min, energy_max, value_min, value_max,
            )
            if data.unavailable == CACHE_MISS:
                return no_update, CACHE_MISS_MESSAGE
            if data.unavailable == NO_DATA:
                return no_update, "No processed data is available to export."

            labels, _ = labels_and_colors(store)
            series_by_key = {item.key: item for item in data.series}

            if chart == "line":
                prepared, _ = compute_line(data)
                content = presenters.line_csv(prepared, labels)
            elif chart == "bar":
                config, result = compute_bar(
                    data, structure_mode, bin_count, preset, edges_text,
                    group_energy_min, group_energy_max, bar_mode,
                )
                content = presenters.bar_csv(
                    result, series_by_key, labels, config.structure_name
                )
            elif chart == "comparison":
                _, results = compute_comparison(
                    data, comparison_metric, comparison_reference,
                    comparison_targets,
                )
                content = presenters.comparison_csv(results, labels)
            else:
                content = presenters.kde_csv(
                    compute_kde(data, bandwidth), series_by_key, labels
                )
        except USER_CORRECTABLE as error:
            return no_update, str(error)

        return (
            dcc.send_string(
                content, f"nuclear-cross-section-processed-{chart}.csv"
            ),
            "",
        )


    # -- C14 ----------------------------------------------------------------
    @dash_app.callback(
        Output("comparison-reference", "options"),
        Output("comparison-reference", "value"),
        Output("comparison-targets", "options"),
        Output("comparison-targets", "value"),
        Input("query-store", "data"),
        Input("series-legend", "value"),
        Input("comparison-swap", "n_clicks"),
        State("comparison-reference", "value"),
        State("comparison-targets", "value"),
        prevent_initial_call=True,
    )
    @offload
    def sync_comparison_selectors(store, visible, _swap, reference, targets):
        """Keep the orientation selectors in step with what is loaded.

        `comparison-reference.value` is a State rather than an Input on
        purpose: it is also an Output here, and Dash rejects a callback that
        feeds its own output back in. The formula caption reacts to those
        values in C15 instead.
        """
        meta = (store or {}).get("series", [])
        allowed = set(visible or [])
        options = [
            {"label": item["label"], "value": item["key"]}
            for item in meta
            if item["key"] in allowed
        ]
        keys = [option["value"] for option in options]
        if len(keys) < 2:
            return options, (keys[0] if keys else None), options, []

        chosen = [key for key in as_list(targets) if key in keys]
        if reference not in keys:
            reference = None

        if ctx.triggered_id == "comparison-swap" and reference and chosen:
            # Exchange the two roles: the first comparison becomes the
            # reference, and the old reference is compared against it.
            promoted = chosen[0]
            chosen = [reference] + [
                key for key in chosen[1:] if key != promoted
            ]
            reference = promoted

        if reference is None:
            reference = keys[0]
        chosen = [key for key in chosen if key != reference]
        if not chosen:
            # Two series loaded: the second is compared against the first.
            # More than two: every remaining visible series gets a line.
            chosen = [key for key in keys if key != reference]

        return (
            options,
            reference,
            [option for option in options if option["value"] != reference],
            chosen,
        )

    # -- C15 ----------------------------------------------------------------
    @dash_app.callback(
        Output("comparison-formula", "children"),
        Input("comparison-reference", "value"),
        Input("comparison-targets", "value"),
        Input("query-store", "data"),
        prevent_initial_call=True,
    )
    @offload
    def sync_comparison_formula(reference, targets, store):
        """Spell out the orientation, because 'ratio' does not."""
        if not reference:
            return ""
        labels, _ = labels_and_colors(store)
        reference_label = labels.get(reference, reference)
        names = [
            labels.get(key, key)
            for key in as_list(targets)
            if key != reference
        ]
        if not names:
            return ""
        return " · ".join(
            comparison_formula(reference_label, name) for name in names
        )

    # -- C16 ----------------------------------------------------------------
    @dash_app.callback(
        Output("analysis-section", "style"),
        Output("chart-section", "style"),
        Output("export-btn", "disabled"),
        Output("export-processed-btn", "disabled"),
        Output("series-legend", "options", allow_duplicate=True),
        Output("series-legend", "value", allow_duplicate=True),
        Input("query-store", "data"),
        prevent_initial_call=True,
    )
    @offload
    def sync_result_panels(store):
        """Reveal the result panels only while a result is actually loaded.

        `query-store` is the single source of truth for "is there a result",
        and C2/C3/C4 all clear it the moment the selection changes. Deriving
        the panels from it here is what stops them outliving their data.

        C6 used to be the sole writer of these four properties, so they only
        ever changed when a query *ran*. Change the databases or switch
        comparison mode after a run and the Analysis panel stayed on screen -
        filters, toolbar, legend, and live export buttons - describing a
        result that no longer matched the selection. The chart itself did
        reset (C10 reads the same store), which is what made the stale panel
        look merely empty rather than obviously wrong. The enabled export
        buttons were the worst of it: pressing one reported "This result is
        no longer cached", when nothing had expired at all.

        `SECTION_SHOWN`, never `SHOWN`: `SHOWN` is `display: flex`, which on a
        <section> lays the whole control stack out beside the chart.

        `prevent_initial_call=True` is safe because `query-store` is
        `storage_type="memory"` - always empty on load - and the layout
        defaults are already this hidden/disabled state.
        """
        if not store:
            # The legend is emptied, not just hidden. It lives inside
            # #analysis-section so hiding the section is enough *today*, but a
            # legend still listing the previous comparison's series is exactly
            # the stale state this callback exists to prevent - leaving it
            # populated relies on the section never being revealed by any
            # other route.
            return components.HIDDEN, components.HIDDEN, True, True, [], []
        return (
            components.SECTION_SHOWN, components.SECTION_SHOWN, False, False,
            # C6 fills the legend from the series metadata in the same
            # response that sets the store; `no_update` leaves its values
            # alone. It also stops Dash re-firing the chart render, which
            # reads series-legend.value as an Input.
            no_update, no_update,
        )


def comparison_notice(
    shown: int, excluded: int, crossings: int, value_filter_dropped: bool
) -> str:
    text = f"Showing {shown:,} comparison points."
    if excluded:
        text += (
            f" {excluded:,} undefined point(s) excluded "
            "(zero, missing, or discontinuous reference)."
        )
    if crossings:
        text += f" {crossings:,} baseline crossing(s)."
    if value_filter_dropped:
        text += (
            " Min/Max cross section do not apply to this chart: filtering by "
            "value can empty the middle of a curve, which would make the "
            "ratio span energies with no data."
        )
    return text


def notice(shown: int, excluded: int) -> str:
    text = f"Showing {shown:,} filtered points."
    if excluded:
        text += f" {excluded:,} nonpositive value(s) excluded for log scale."
    return text


def natural_sort(values: list[str]) -> list[str]:
    """Order like the browser's `localeCompare(..., {numeric: true})`.

    Keeps MT1 < MT2 < MT16 < MT102 and H1 < Co59 < U235 rather than the
    lexicographic MT1 < MT102 < MT16.
    """

    def key(value: str):
        return [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", value)
        ]

    return sorted(values, key=key)
