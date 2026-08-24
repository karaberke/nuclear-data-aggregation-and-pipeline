"""Plotly figure construction.

Reproduces the three Vega-Lite specs the TypeScript built. Presentation
only: every number arriving here has already been computed by
`backend.services.analytics`.

Two Plotly-specific traps are handled explicitly and each has a unit test:

* `go.Bar` positions a bar at `x` +/- `width / 2` in **data** units and only
  then applies the axis transform, so a log x-axis still wants the arithmetic
  centre and the raw width - passing log10-space values is the bug, not the
  fix, and renders bars ~7x too narrow (see `_bar_geometry`);
* `np.ndarray` passed straight into a trace is serialized as a base64 typed
  array, which is dramatically smaller than a JSON number list. Never call
  `.tolist()` on the way into a figure.
"""

import numpy as np
import plotly.graph_objects as go

from ..deployment import env_flag
from ..services import analytics
from .components import PALETTE, PLACEHOLDER_MESSAGE
from .inputs import comparison_axis_title

# Above this many visible points, switch the raw line to WebGL. Scattergl
# keeps 5 x 30,000 points interactive where SVG would not.
SCATTERGL_THRESHOLD = 4_000


def webgl_enabled() -> bool:
    """Whether to use Scattergl for large traces.

    Set `PLOTLY_FORCE_SVG=1` for environments without WebGL - remote
    desktops, VDI sessions, and locked-down corporate browsers all turn up
    without it, and Plotly renders a "WebGL is not supported" placeholder
    rather than falling back on its own. SVG is slower with tens of
    thousands of points but always renders.
    """
    return not env_flag("PLOTLY_FORCE_SVG")


FONT_FAMILY = '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
TEXT_COLOR = "#1d1d1f"
MUTED_COLOR = "#6e6e73"
GRID_COLOR = "rgba(0, 0, 0, 0.08)"
LINE_COLOR = "rgba(0, 0, 0, 0.12)"

GRAPH_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "nuclear-cross-section",
        "scale": 2,
    },
}


def color_for(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


def base_layout(
    x_title: str,
    y_title: str,
    x_scale: str,
    y_scale: str,
    title: str | None = None,
    uirevision: str | None = None,
) -> go.Layout:
    return go.Layout(
        title=dict(text=title, x=0.5, xanchor="center") if title else None,
        height=420,
        autosize=True,
        # Top margin leaves room for the horizontal legend above the plot.
        margin=dict(l=70, r=20, t=76 if title else 56, b=52),
        # Transparent so the frosted-glass panel shows through.
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, color=TEXT_COLOR, size=12),
        # Display only. The HTML pill checklist remains the *control* - it is
        # a server-side Input, so hiding a series genuinely re-bins and
        # re-normalizes the KDE. This legend exists so the PNG the modebar
        # camera produces is self-describing; `itemclick`/`itemdoubleclick`
        # are off precisely so it can never diverge from that server state.
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
            itemclick=False,
            itemdoubleclick=False,
        ),
        hovermode="closest",
        hoverlabel=dict(font=dict(family=FONT_FAMILY, size=12)),
        xaxis=dict(
            title=x_title,
            type=x_scale,
            gridcolor=GRID_COLOR,
            linecolor=LINE_COLOR,
            zeroline=False,
        ),
        yaxis=dict(
            title=y_title,
            type=y_scale,
            gridcolor=GRID_COLOR,
            linecolor=LINE_COLOR,
            zeroline=False,
        ),
        # Preserves pan/zoom across control changes, but resets on a new
        # query or a chart-type switch. Without it every filter tweak snaps
        # the viewport back and the app feels broken.
        uirevision=uirevision,
    )


def message_figure(text: str = PLACEHOLDER_MESSAGE) -> go.Figure:
    """An explicit message, never an empty grid.

    A Plotly figure with zero traces and no annotation renders as a bare
    axis pair, which is exactly the "blank chart" failure to avoid.
    """
    figure = go.Figure()
    figure.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        annotations=[
            dict(
                text=text,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(family=FONT_FAMILY, size=14, color=MUTED_COLOR),
            )
        ],
    )
    return figure


def _hover_prefix(item) -> str:
    return (
        f"<b>{item.database}</b><br>"
        f"{item.isotope} · {item.dataset}<br>"
    )


def line_figure(
    prepared: list[dict],
    labels: dict[str, str],
    colors: dict[str, str],
    x_scale: str,
    y_scale: str,
    include_stddev: bool,
    uirevision: str | None = None,
) -> go.Figure:
    """Raw pointwise curves, with optional per-point uncertainty rules."""
    figure = go.Figure()
    total_points = sum(entry["energy"].size for entry in prepared)
    use_gl = webgl_enabled() and total_points > SCATTERGL_THRESHOLD
    trace_type = go.Scattergl if use_gl else go.Scatter
    band_type = go.Scattergl if use_gl else go.Scatter

    for entry in prepared:
        item = entry["series"]
        if entry["energy"].size == 0:
            continue
        color = colors[item.key]

        # The uncertainty band, drawn BENEATH the line as the Vega `rule`
        # layer was. One trace of interleaved [lo, hi, nan] segments: nan is
        # a segment break, so 30k per-point rules cost one WebGL trace
        # instead of 30k SVG shapes. (error_y is not an option - Scattergl
        # does not support it.)
        if include_stddev and entry["y_low"] is not None:
            low = entry["y_low"]
            high = entry["y_high"]
            valid = ~(np.isnan(low) | np.isnan(high))
            if valid.any():
                energies = entry["energy"][valid]
                breaks = np.full(energies.size, np.nan)
                figure.add_trace(
                    band_type(
                        x=np.stack([energies, energies, breaks]).T.ravel(),
                        y=np.stack([low[valid], high[valid], breaks]).T.ravel(),
                        mode="lines",
                        line=dict(color=color, width=1),
                        opacity=0.4,
                        hoverinfo="skip",
                        showlegend=False,
                        legendgroup=item.key,
                        name=f"{labels[item.key]} uncertainty",
                    )
                )

        # Constant fields are baked into the template string; only the
        # varying stddev goes in customdata. A 30k x 5 string customdata
        # array would dominate the payload for no added information.
        hover = _hover_prefix(item) + (
            "Energy: %{x:.6g} MeV<br>Cross section: %{y:.6g} barns"
        )
        customdata = None
        if include_stddev and entry["stddev"] is not None:
            customdata = entry["stddev"]
            hover += "<br>Std. deviation: %{customdata:.6g} barns"

        figure.add_trace(
            trace_type(
                # numpy arrays, never .tolist() - Dash serializes ndarray as
                # base64 typed arrays.
                x=entry["energy"],
                y=entry["sigma"],
                mode="lines",
                line=dict(color=color, width=2),
                legendgroup=item.key,
                name=labels[item.key],
                customdata=customdata,
                hovertemplate=hover + "<extra></extra>",
            )
        )

    figure.update_layout(
        base_layout(
            "Energy (MeV)",
            "Cross Section (barns)",
            x_scale,
            y_scale,
            uirevision=uirevision,
        )
    )
    return figure


def _bar_geometry(
    starts: np.ndarray, ends: np.ndarray, x_scale: str
) -> tuple[np.ndarray, np.ndarray]:
    """Bar centres and widths that span exactly [start, end].

    `go.Bar` positions a bar at `x` +/- `width / 2` in **data** units and
    only then applies the axis transform, so the arithmetic centre and the
    raw width are correct on a log axis too. Using the geometric centre with
    a log10 width - the intuitive reading of "axis units" - renders bars
    roughly 7x too narrow and truncates the x-range; verified against
    rendered pixel positions, see test_bar_geometry_spans_the_bin_exactly.

    `x_scale` is accepted so the signature documents that the caller has
    considered the axis type, and so a future axis type can change this
    without touching call sites.
    """
    return (starts + ends) / 2, ends - starts


def bar_figure(
    rows: list[dict],
    labels: dict[str, str],
    colors: dict[str, str],
    x_scale: str,
    y_scale: str,
    value_title: str,
    uirevision: str | None = None,
) -> go.Figure:
    """Binned bars, split by coverage so partial groups are flagged."""
    figure = go.Figure()
    # A series emits up to two traces (full coverage, partial coverage), but
    # earns exactly one legend entry. Keyed off "first trace emitted for this
    # series" rather than "the full-coverage one", because a series whose bins
    # are *all* partial would otherwise go unlabelled.
    labelled: set[str] = set()

    for entry in rows:
        item = entry["series"]
        color = colors[item.key]
        for partial in (False, True):
            mask = entry["partial"] == partial
            if not mask.any():
                continue

            starts = entry["bin_start"][mask]
            ends = entry["bin_end"][mask]
            centers, widths = _bar_geometry(starts, ends, x_scale)

            marker = dict(
                color=color,
                opacity=0.28 if partial else 0.5,
                line=dict(color=color, width=1.5),
            )
            if partial:
                # go.Bar.marker.line has no `dash`, so the Vega dashed
                # stroke becomes hatching - which also reads better at 725
                # groups, where adjacent borders merge visually.
                marker["pattern"] = dict(
                    shape="/",
                    fgcolor=color,
                    bgcolor="rgba(0,0,0,0)",
                    size=4,
                    solidity=0.25,
                )

            suffix = " · partial coverage" if partial else ""
            # `name` never reaches the tooltip - every hovertemplate below
            # ends in `<extra></extra>` - so the suffix stays in the hover
            # text and the legend shows the plain series label.
            show_in_legend = item.key not in labelled
            labelled.add(item.key)
            figure.add_trace(
                go.Bar(
                    x=centers,
                    width=widths,
                    y=entry["value"][mask],
                    marker=marker,
                    legendgroup=item.key,
                    showlegend=show_in_legend,
                    name=labels[item.key],
                    customdata=entry["customdata"][mask],
                    hovertemplate=(
                        _hover_prefix(item)
                        + "%{customdata[0]}<br>"
                        + "Group %{customdata[1]}<br>"
                        + "Range: %{customdata[2]} – %{customdata[3]} MeV<br>"
                        + "Center: %{customdata[4]} · Width: %{customdata[5]}<br>"
                        + "Points: %{customdata[6]} · Coverage: %{customdata[7]}%<br>"
                        + "Mean: %{customdata[8]} barns<br>"
                        + "Group average: %{customdata[9]} barns<br>"
                        + "Integral: %{customdata[10]} barn·MeV"
                        + suffix
                        + "<extra></extra>"
                    ),
                )
            )

    figure.update_layout(
        base_layout(
            "Energy bin (MeV)",
            value_title,
            x_scale,
            y_scale,
            uirevision=uirevision,
        )
    )
    figure.update_layout(barmode="overlay", bargap=0)
    return figure


def kde_figure(
    result: analytics.KdeResult,
    labels: dict[str, str],
    colors: dict[str, str],
    x_scale: str,
    y_scale: str,
    uirevision: str | None = None,
) -> go.Figure:
    """Kernel density curves - 200 points each, so plain SVG is fine."""
    figure = go.Figure()
    for curve in result.curves:
        figure.add_trace(
            go.Scatter(
                x=curve.x,
                y=curve.density,
                mode="lines",
                line=dict(color=colors[curve.series_key], width=2.5),
                legendgroup=curve.series_key,
                name=labels[curve.series_key],
                hovertemplate=(
                    "Cross section: %{x:.6g} barns<br>"
                    "Density: %{y:.6g}<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        base_layout(
            "Cross Section (barns)",
            # This one genuinely is a probability density, unlike the bar
            # chart's "Density" mode.
            "Probability Density",
            x_scale,
            y_scale,
            uirevision=uirevision,
        )
    )
    return figure

BASELINES = {
    # (baseline value, annotation) per metric. The baseline is the whole point
    # of this chart: above it the comparison is larger, below it the reference.
    "ratio": (1.0, "Equal cross sections"),
    "percent": (0.0, "No difference"),
}


def _comparison_y_range(rows: list[dict], baseline: float) -> list[float]:
    """Data extent widened to always contain the baseline.

    Plotly shapes - which is what `add_hline` produces - take no part in
    autoranging, so a comparison that never approaches equality would push the
    baseline off-screen and leave the reader with no reference to judge
    against. Large values are still never clipped, and zoom is unaffected.
    """
    finite = [
        row["values"][np.isfinite(row["values"])]
        for row in rows
        if row["values"].size
    ]
    values = np.concatenate(finite) if finite else np.empty(0)
    if values.size == 0:
        return [baseline - 1.0, baseline + 1.0]

    low = min(float(values.min()), baseline)
    high = max(float(values.max()), baseline)
    pad = (high - low) * 0.05 if high > low else max(abs(baseline), 1.0) * 0.05
    return [low - pad, high + pad]


def comparison_figure(
    rows: list[dict],
    colors: dict[str, str],
    x_scale: str,
    metric: str,
    uirevision: str | None = None,
) -> go.Figure:
    """Ratio or percent-difference curves against a labelled equality line.

    The y-axis is linear in both metrics and is not configurable: percent
    difference is signed, so a log axis would drop every energy at which the
    reference is the larger evaluation - exactly half the information the
    chart exists to show.
    """
    baseline, baseline_label = BASELINES[metric]
    figure = go.Figure()

    total_points = sum(row["values"].size for row in rows)
    use_gl = webgl_enabled() and total_points > SCATTERGL_THRESHOLD
    trace_type = go.Scattergl if use_gl else go.Scatter

    for row in rows:
        result = row["result"]
        reference_label = row["reference_label"]
        comparison_label = row["comparison_label"]
        # Coloured by the comparison series, so a line keeps the colour its
        # series already has in the legend and every other chart.
        color = colors.get(result.comparison_key, color_for(0))

        figure.add_trace(
            trace_type(
                x=result.energy,
                y=row["values"],
                mode="lines",
                line=dict(color=color, width=2),
                # NaN at an undefined point must read as a break in the curve,
                # never as a straight line drawn across it.
                connectgaps=False,
                legendgroup=result.comparison_key,
                name=f"{comparison_label} / {reference_label}",
                customdata=row["customdata"],
                hovertext=row["hovertext"],
                hovertemplate=(
                    "Energy: %{x:.6g} MeV<br>"
                    f"Comparison: {comparison_label}<br>"
                    f"Reference: {reference_label}<br>"
                    "Comparison σ: %{customdata[1]:.6g} b<br>"
                    "Reference σ: %{customdata[0]:.6g} b<br>"
                    "Ratio: %{customdata[2]:.6g}<br>"
                    "Difference: %{customdata[3]:+.4g}%<br>"
                    "Dominant: %{hovertext}"
                    "<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        base_layout(
            "Energy (MeV)",
            comparison_axis_title(metric),
            x_scale,
            "linear",
            uirevision=uirevision,
        )
    )
    figure.update_yaxes(range=_comparison_y_range(rows, baseline))
    # After `update_layout`, which replaces the layout object wholesale and
    # would otherwise discard the shape this adds.
    figure.add_hline(
        y=baseline,
        line_dash="dash",
        line_color=MUTED_COLOR,
        line_width=1.5,
        annotation_text=baseline_label,
        annotation_position="top left",
        annotation_font=dict(family=FONT_FAMILY, size=11, color=MUTED_COLOR),
    )
    return figure
