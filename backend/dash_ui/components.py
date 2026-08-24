"""Reusable layout fragments.

Presentation only - no calculation, no service calls.
"""

from dash import dcc, html

# Same five colours the Vega spec used, in the same order.
PALETTE = ["#0a84ff", "#ff375f", "#30b0c7", "#ff9f0a", "#bf5af2"]

COMPARISON_MODES = [
    {"label": "Single series", "value": "single"},
    {"label": "Compare databases", "value": "databases"},
    {"label": "Compare isotopes", "value": "isotopes"},
    {"label": "Compare datasets", "value": "datasets"},
]

REACTION_TYPES = [
    {"label": "Cross section", "value": "xs"},
    {"label": "Cross section ± std. deviation", "value": "xs_stddev"},
]

SCALES = [
    {"label": "Linear", "value": "linear"},
    {"label": "Log", "value": "log"},
]

GROUP_STRUCTURE_MODES = [
    {"label": "Automatic", "value": "automatic"},
    {"label": "Standard structure", "value": "standard"},
    {"label": "Custom edges", "value": "custom"},
]

BAR_MODES = [
    {"label": "Mean", "value": "mean"},
    {"label": "Group Average / Density", "value": "density"},
]

CHART_TABS = [
    {"label": "Raw line", "value": "line"},
    {"label": "Binned bars", "value": "bar"},
    {"label": "Kernel Density Estimation", "value": "kde"},
    {"label": "Ratio / Difference", "value": "comparison"},
]

# Default axis scale per chart. Lives here, not in `callbacks.py`, because
# both `layout.py` (as the `axis-store` initial value) and `callbacks.py` (as
# the fallback when the store is empty) need it, and this module is the one
# neither imports from the other to reach. It was previously written out in
# full in both places - the same four entries and the same explanatory
# comment - so a change to one silently left the other stale.
AXIS_DEFAULTS = {
    "line": {"x": "log", "y": "log"},
    "bar": {"x": "log", "y": "log"},
    "kde": {"x": "log", "y": "log"},
    # Linear and fixed - percent difference is signed, so a log y-axis would
    # drop every energy at which the reference is the larger evaluation.
    "comparison": {"x": "log", "y": "linear"},
}

# "Ratio / Difference" names the chart; the metric names the calculation.
# Both are spelled out because "ratio difference" would describe neither.
COMPARISON_METRICS = [
    {"label": "Ratio", "value": "ratio"},
    {"label": "Percent Difference vs Reference", "value": "percent"},
]

PLACEHOLDER_MESSAGE = "Run a query to compare cross-section data"

HIDDEN = {"display": "none"}
SHOWN = {"display": "flex"}
# `<section>` reveal. NOT `SHOWN`: flex on a section lays its children out in a
# row, which put the whole control stack beside the chart and overflowed the
# viewport. The legacy SPA revealed these panels with the `hidden` attribute,
# whose removal restores the element's default `display: block`; that is the
# value being stated here.
SECTION_SHOWN = {"display": "block"}


def control(label: str, component, hint: str | None = None, **kwargs):
    """A labelled form control matching the original `.control` block."""
    hint_id = kwargs.pop("hint_id", None)
    children = [html.Span(label), component]
    if hint is not None:
        # Dash rejects id=None, so only set it when one was requested.
        extra = {"id": hint_id} if hint_id else {}
        children.append(html.Small(hint, **extra))
    return html.Label(children, className="control", **kwargs)


def compact_control(label: str, component, **kwargs):
    """A toolbar-sized labelled control."""
    return html.Label(
        [html.Span(label), component], className="compact-control", **kwargs
    )


def cascade_control(label: str, control_id: str) -> html.Div:
    """One rung of the database -> isotope -> dataset cascade.

    All three start empty and disabled and are populated by C2/C3/C4; the
    only thing that differed between them was the label and the id, so they
    were three copies of the same eight lines. The `-hint` id is derived
    rather than passed, because `hint_for`/`prompt_for` in callbacks.py key
    off exactly this same name.
    """
    return control(
        label,
        dcc.Dropdown(
            id=control_id, options=[], disabled=True,
            className="dash-dropdown",
        ),
        "Choose one",
        hint_id=f"{control_id}-hint",
    )


def selection_panel_controls() -> html.Div:
    return html.Div(
        [
            control(
                "Comparison mode",
                dcc.Dropdown(
                    id="comparison-mode",
                    options=COMPARISON_MODES,
                    value="single",
                    clearable=False,
                    className="dash-dropdown",
                ),
                "Select what changes between series",
            ),
            cascade_control("Database", "databases"),
            control(
                "Field",
                dcc.Input(
                    id="field", value="SIG", disabled=True, type="text"
                ),
                "Fixed cross-section field",
            ),
            cascade_control("Isotope", "isotopes"),
            cascade_control("Dataset", "datasets"),
            control(
                "Reaction values",
                dcc.Dropdown(
                    id="reaction-type",
                    options=REACTION_TYPES,
                    value="xs",
                    clearable=False,
                    className="dash-dropdown",
                ),
                "Uncertainty appears on the raw chart",
            ),
        ],
        className="controls-grid selection-grid",
    )


def action_row() -> html.Div:
    return html.Div(
        [
            html.Button(
                "Run comparison",
                id="run-btn",
                className="action run-button",
                disabled=True,
                n_clicks=0,
            ),
            html.Button(
                "Export filtered CSV",
                id="export-btn",
                className="action export-button",
                disabled=True,
                n_clicks=0,
            ),
            html.Button(
                "Export graph CSV",
                id="export-processed-btn",
                className="action export-processed-button",
                disabled=True,
                n_clicks=0,
            ),
        ],
        className="action-row",
    )


def number_input(component_id: str) -> dcc.Input:
    """A numeric field that commits on blur or Enter.

    Module level rather than nested in `filter_grid`, because `bin_controls`
    needs the identical widget and had re-inlined it twice - so the debounce
    rationale below only governed four of the six numeric fields.

    debounce=True commits on blur or Enter. The TypeScript used a 150ms
    trailing debounce, which was free there because the work was in-process;
    here every render is a round trip carrying a large figure, so blur/Enter
    is both cheaper and a more natural commit point for a numeric field.
    """
    return dcc.Input(
        id=component_id,
        type="number",
        debounce=True,
        placeholder="Automatic",
    )


def filter_grid() -> html.Div:
    return html.Div(
        [
            control("Minimum energy (MeV)", number_input("energy-min")),
            control("Maximum energy (MeV)", number_input("energy-max")),
            control("Minimum cross section", number_input("value-min")),
            control("Maximum cross section", number_input("value-max")),
        ],
        className="controls-grid filter-grid",
    )


def bin_controls(preset_options: list[dict]) -> html.Div:
    return html.Div(
        [
            compact_control(
                "Group structure",
                dcc.Dropdown(
                    id="group-structure-mode",
                    options=GROUP_STRUCTURE_MODES,
                    value="automatic",
                    clearable=False,
                    className="dash-dropdown",
                ),
            ),
            html.Div(
                [
                    html.Label(
                        [
                            html.Span("Bins"),
                            dcc.Slider(
                                id="bin-count",
                                min=10,
                                max=100,
                                step=1,
                                value=40,
                                marks=None,
                                # mouseup is the default, stated explicitly so
                                # it is not "fixed" into drag-storm behaviour.
                                updatemode="mouseup",
                                tooltip={
                                    "placement": "bottom",
                                    "always_visible": True,
                                },
                            ),
                        ],
                        className="range-control",
                    ),
                    control(
                        "Min energy (MeV)",
                        number_input("group-energy-min"),
                    ),
                    control(
                        "Max energy (MeV)",
                        number_input("group-energy-max"),
                    ),
                ],
                id="automatic-group-controls",
                className="bin-controls-group",
            ),
            html.Div(
                [
                    control(
                        "Preset",
                        dcc.Dropdown(
                            id="group-structure-preset",
                            options=preset_options,
                            value=(
                                preset_options[0]["value"]
                                if preset_options
                                else None
                            ),
                            clearable=False,
                            className="dash-dropdown",
                        ),
                    )
                ],
                id="standard-group-controls",
                className="bin-controls-group",
                style=HIDDEN,
            ),
            html.Div(
                [
                    control(
                        "Bin edges",
                        dcc.Input(
                            id="bin-edges",
                            type="text",
                            debounce=True,
                            placeholder="Automatic",
                            value="",
                        ),
                    )
                ],
                id="custom-group-controls",
                className="bin-controls-group",
                style=HIDDEN,
            ),
            compact_control(
                "Bar value",
                dcc.Dropdown(
                    id="bar-mode",
                    options=BAR_MODES,
                    value="mean",
                    clearable=False,
                    className="dash-dropdown",
                ),
            ),
            html.Small(
                "Group Average / Density is the energy-averaged cross "
                "section ∫σ(E)dE / ΔE, in barns — not a "
                "probability density.",
                className="control-note",
                id="bar-mode-note",
            ),
        ],
        id="bin-controls",
        className="bin-controls-group",
        style=HIDDEN,
    )


def kde_controls() -> html.Div:
    return html.Div(
        [
            html.Label(
                [
                    html.Span("Bandwidth ×"),
                    dcc.Slider(
                        id="bandwidth",
                        min=0.25,
                        max=4,
                        step=0.05,
                        value=1,
                        marks=None,
                        updatemode="mouseup",
                        tooltip={
                            "placement": "bottom",
                            "always_visible": True,
                        },
                    ),
                ],
                className="range-control",
            )
        ],
        id="kde-controls",
        className="bin-controls-group",
        style=HIDDEN,
    )


def comparison_controls() -> html.Div:
    """Metric, orientation, and the formula caption for the ratio chart.

    The formula caption is not decoration: "ratio" alone does not say which
    series is divided by which, and getting it backwards inverts the reading
    of every point on the chart.
    """
    return html.Div(
        [
            compact_control(
                "Metric",
                dcc.Dropdown(
                    id="comparison-metric",
                    options=COMPARISON_METRICS,
                    value="ratio",
                    clearable=False,
                    className="dash-dropdown",
                ),
            ),
            compact_control(
                "Reference",
                dcc.Dropdown(
                    id="comparison-reference",
                    options=[],
                    clearable=False,
                    className="dash-dropdown",
                ),
            ),
            compact_control(
                "Comparison",
                dcc.Dropdown(
                    id="comparison-targets",
                    options=[],
                    multi=True,
                    className="dash-dropdown",
                ),
            ),
            html.Button(
                "Swap",
                id="comparison-swap",
                className="action reset-button",
                n_clicks=0,
                title="Exchange the reference and comparison series",
            ),
            html.Span(
                "", id="comparison-formula", className="comparison-formula"
            ),
        ],
        id="comparison-controls",
        className="bin-controls-group",
        style=HIDDEN,
    )


def chart_toolbar(preset_options: list[dict]) -> html.Div:
    return html.Div(
        [
            compact_control(
                "X scale",
                dcc.Dropdown(
                    id="x-scale",
                    options=SCALES,
                    value="log",
                    clearable=False,
                    className="dash-dropdown",
                ),
            ),
            compact_control(
                "Y scale",
                dcc.Dropdown(
                    id="y-scale",
                    options=SCALES,
                    value="log",
                    clearable=False,
                    className="dash-dropdown",
                ),
            ),
            bin_controls(preset_options),
            kde_controls(),
            comparison_controls(),
        ],
        className="chart-toolbar",
    )


def legend_options(series_meta: list[dict]) -> list[dict]:
    """Checklist options carrying a colour swatch, as the old pills did."""
    return [
        {
            "label": html.Span(
                [
                    html.Span(
                        className="legend-swatch",
                        style={"backgroundColor": item["color"]},
                    ),
                    html.Span(item["label"]),
                ]
            ),
            "value": item["key"],
        }
        for item in series_meta
    ]
