"""Dash page layout.

Reproduces the structure of the TypeScript SPA's `index.html` (removed at
cutover; see git history). The component ids here are asserted by
`tests/test_dash_smoke.py` — keep that list in sync when renaming one.

One deliberate difference from the original markup: `#status-line` used to
multiplex selection messages, render counts, and validation errors. Here
those are split across three elements so that no Dash output needs five
producers competing through `allow_duplicate`:

    #status-line   selection and query state    (C2, C3, C4, C6)
    #chart-notice  render counts and exclusions (C10 only)
    #chart-error   validation errors            (C10, C12, C13)

The analysis panel is split in two: `#analysis-section` holds the filters and
display controls, `#chart-section` holds the graph beneath them. C6 reveals
both together.
"""

from dash import dcc, html

from ..services import multigroup
from . import components, figures


def page_header() -> html.Header:
    return html.Header(
        [
            html.H1("Nuclear Data Explorer"),
            html.P(
                [
                    "Evaluated neutron cross-section measurements sourced from ",
                    html.A(
                        "JANIS",
                        href="https://www.oecd-nea.org/jcms/pl_39933/what-is-janis",
                        target="_blank",
                        rel="noopener noreferrer",
                    ),
                    ".",
                ],
                className="subtitle",
            ),
        ]
    )


def stores() -> list:
    return [
        # Page-load trigger for the session-id callback.
        dcc.Location(id="app-location"),
        # Per-tab cache partition key. A partition key, not a credential.
        dcc.Store(id="session-store", storage_type="session"),
        # ~1.5 KB handle: cache key, the query, and per-series metadata.
        # The 30,000-point arrays never leave the server.
        dcc.Store(id="query-store", storage_type="memory"),
        # Independent axis memory per chart type, surviving a reload.
        dcc.Store(
            id="axis-store",
            storage_type="session",
            data=components.AXIS_DEFAULTS,
        ),
        dcc.Download(id="download-filtered"),
        dcc.Download(id="download-processed"),
    ]


def selection_panel() -> html.Section:
    return html.Section(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Data selection"),
                            html.P(
                                "Choose one comparison dimension and up to "
                                "five series."
                            ),
                        ]
                    )
                ],
                className="panel-heading",
            ),
            components.selection_panel_controls(),
            components.action_row(),
            html.Div(
                id="status-line", role="status", **{"aria-live": "polite"}
            ),
            html.Pre("No selections yet.", id="selection-summary"),
        ],
        className="panel selection-panel",
        id="selection-section",
    )


def analysis_panel() -> html.Section:
    preset_options = multigroup.structure_options()
    return html.Section(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Analysis"),
                            html.P(
                                "Filters and display controls reuse the "
                                "fetched data - changing them never re-runs "
                                "JANIS."
                            ),
                        ]
                    ),
                    html.Button(
                        "Reset controls",
                        id="reset-controls",
                        className="action reset-button",
                        n_clicks=0,
                    ),
                ],
                className="panel-heading",
            ),
            components.filter_grid(),
            dcc.Tabs(
                id="chart-tabs",
                value="line",
                className="chart-tabs",
                children=[
                    dcc.Tab(
                        label=tab["label"],
                        value=tab["value"],
                        className="tab",
                        selected_className="tab--selected",
                    )
                    for tab in components.CHART_TABS
                ],
            ),
            components.chart_toolbar(preset_options),
            dcc.Checklist(
                id="series-legend",
                options=[],
                value=[],
                inline=True,
                className="series-legend",
                inputClassName="legend-check",
                labelClassName="legend-item",
            ),
        ],
        className="panel analysis-panel",
        id="analysis-section",
        style=components.HIDDEN,
    )


def chart_panel() -> html.Section:
    """The graph, in its own full-width card below the controls."""
    return html.Section(
        [
            html.Div(id="chart-notice", className="chart-notice", role="status"),
            html.Div(id="chart-error", className="chart-error", role="alert"),
            dcc.Loading(
                dcc.Graph(
                    id="chart-graph",
                    figure=figures.message_figure(),
                    config=figures.GRAPH_CONFIG,
                ),
                type="default",
                className="chart-container",
            ),
        ],
        className="panel chart-panel",
        id="chart-section",
        style=components.HIDDEN,
    )


def build_layout() -> html.Div:
    children = [
        page_header(),
        html.Main(
            stores() + [selection_panel(), analysis_panel(), chart_panel()],
            className="wrapper",
        ),
    ]
    return html.Div(children, id="page-root")
