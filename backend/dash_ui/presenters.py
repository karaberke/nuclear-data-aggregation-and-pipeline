"""Shape service results for figures, downloads, and the legend.

Formatting only. The numbers are already final when they arrive here; this
module decides how they are labelled, coloured, and rendered as text.
"""

import numpy as np

from ..services import analytics, exports, mt_labels
from .components import PALETTE
from .inputs import series_label


def series_metadata(
    series: list[analytics.SeriesArrays], comparison_mode: str
) -> list[dict]:
    """Stable per-series label and colour.

    Colour is assigned by position in the LOADED order and then persisted in
    the query store, so a series keeps its colour when others are hidden.
    """
    return [
        {
            "key": item.key,
            "database": item.database,
            "isotope": item.isotope,
            "dataset": item.dataset,
            "label": series_label(item, comparison_mode),
            "color": PALETTE[index % len(PALETTE)],
            "n_points": item.size,
        }
        for index, item in enumerate(series)
    ]


def _cell(value) -> str:
    """Hover text for one numeric cell.

    Missing values render blank, never `null` and never 0 - a group with no
    landed sample has no mean, which is different from a mean of zero.
    """
    if value is None:
        return ""
    return exports.format_number(round(float(value), 12))


def bar_rows(
    result: analytics.GroupingResult,
    series_by_key: dict[str, analytics.SeriesArrays],
    field: str,
    y_scale: str,
) -> tuple[list[dict], int]:
    """Group records to per-series arrays for `figures.bar_figure`."""
    grouped: dict[str, list[analytics.GroupPoint]] = {}
    excluded = 0

    for record in result.records:
        value = getattr(record, field)
        if value is None:
            continue
        if y_scale == "log" and value <= 0:
            excluded += 1
            continue
        grouped.setdefault(record.series_key, []).append(record)

    rows: list[dict] = []
    for key, records in grouped.items():
        item = series_by_key[key]
        mt = mt_labels.parse_mt_from_dataset(item.dataset)
        label = mt_labels.format_mt_label(mt)

        rows.append(
            {
                "series": item,
                "bin_start": np.array(
                    [record.bin_start for record in records], dtype=np.float64
                ),
                "bin_end": np.array(
                    [record.bin_end for record in records], dtype=np.float64
                ),
                "value": np.array(
                    [getattr(record, field) for record in records],
                    dtype=np.float64,
                ),
                "partial": np.array(
                    [record.coverage_fraction < 1 for record in records],
                    dtype=bool,
                ),
                "customdata": np.array(
                    [
                        [
                            label,
                            str(record.group_index),
                            _cell(record.bin_start),
                            _cell(record.bin_end),
                            _cell(record.bin_center),
                            _cell(record.bin_width),
                            str(record.point_count),
                            _cell(record.coverage_fraction * 100),
                            _cell(record.cross_section_mean),
                            _cell(record.cross_section_group_average),
                            _cell(record.cross_section_integral),
                        ]
                        for record in records
                    ],
                    dtype=object,
                ),
            }
        )

    return rows, excluded


def _uncertainty_at(stddev, index) -> float | None:
    """One point's standard deviation, or None where JANIS reported none.

    Missing must stay None all the way to `csv_cell`, which renders it as an
    empty field. Coercing it to 0.0 would claim zero uncertainty, which is a
    physics error, not a formatting one - `series_from_points` stores these
    as NaN for exactly that reason.
    """
    if stddev is None or np.isnan(stddev[index]):
        return None
    return float(stddev[index])


def filtered_csv(series: list[analytics.SeriesArrays]) -> str:
    """One row per visible, filtered raw point."""
    rows = []
    for item in series:
        stddev = item.stddev
        for index in range(item.size):
            uncertainty = _uncertainty_at(stddev, index)
            rows.append(
                [
                    item.database,
                    item.isotope,
                    item.dataset,
                    float(item.energy[index]),
                    float(item.sigma[index]),
                    uncertainty,
                ]
            )
    return exports.write_csv(exports.FILTERED_HEADER, rows)


def line_csv(prepared: list[dict], labels: dict[str, str]) -> str:
    rows = []
    for entry in prepared:
        item = entry["series"]
        stddev = entry["stddev"]
        for index in range(entry["energy"].size):
            uncertainty = _uncertainty_at(stddev, index)
            rows.append(
                [
                    item.database,
                    item.isotope,
                    item.dataset,
                    labels[item.key],
                    float(entry["energy"][index]),
                    float(entry["sigma"][index]),
                    uncertainty,
                ]
            )
    return exports.write_csv(exports.LINE_HEADER, rows)


def bar_csv(
    result: analytics.GroupingResult,
    series_by_key: dict[str, analytics.SeriesArrays],
    labels: dict[str, str],
    structure_name: str,
) -> str:
    """Every emitted group, with all three value columns.

    Deliberately not filtered to whichever field is currently plotted: the
    export shows mean, group average, and integral together regardless of
    the chart's Bar value setting, and includes partially covered and
    zero-landed-sample groups.
    """
    rows = []
    for record in result.records:
        item = series_by_key[record.series_key]
        mt = mt_labels.parse_mt_from_dataset(item.dataset)
        rows.append(
            [
                item.database,
                item.isotope,
                item.dataset,
                labels[item.key],
                record.group_index,
                mt,
                mt_labels.format_mt_label(mt),
                record.bin_start,
                record.bin_end,
                record.bin_center,
                record.bin_width,
                record.point_count,
                record.coverage_fraction * 100,
                record.cross_section_mean,
                record.cross_section_group_average,
                record.cross_section_integral,
                structure_name,
                exports.REBINNING_METHOD,
            ]
        )
    return exports.write_csv(exports.BAR_HEADER, rows)


def kde_csv(
    result: analytics.KdeResult,
    series_by_key: dict[str, analytics.SeriesArrays],
    labels: dict[str, str],
) -> str:
    rows = []
    for curve in result.curves:
        item = series_by_key[curve.series_key]
        for x, density in zip(curve.x, curve.density):
            rows.append(
                [
                    item.database,
                    item.isotope,
                    item.dataset,
                    labels[item.key],
                    float(x),
                    float(density),
                ]
            )
    return exports.write_csv(exports.KDE_HEADER, rows)


def comparison_rows(
    results: list[analytics.ComparisonSeries],
    labels: dict[str, str],
    metric: str,
) -> list[dict]:
    """Comparison pairs as figure-ready arrays.

    The hover carries eight fields, which is enough for a naive customdata
    array to dominate the payload at 60,000 merged grid points. It is split by
    cost instead: the two series names are constant per trace and get baked
    into the template by `figures.comparison_figure`, the four varying numbers
    stay a float64 array that Dash serializes base64, and only the dominance
    text - three distinct strings, which gzip collapses - travels as
    `hovertext`. Keeping the strings out of `customdata` is what stops the
    whole array from falling back to object dtype and losing base64.
    """
    rows: list[dict] = []
    for result in results:
        if result.size == 0:
            continue
        reference_label = labels.get(result.reference_key, result.reference_key)
        comparison_label = labels.get(
            result.comparison_key, result.comparison_key
        )
        values = (
            result.ratio if metric == "ratio" else result.percent_difference
        )
        rows.append(
            {
                "result": result,
                "reference_label": reference_label,
                "comparison_label": comparison_label,
                "values": values,
                "customdata": np.stack(
                    [
                        result.reference_sigma,
                        result.comparison_sigma,
                        result.ratio,
                        result.percent_difference,
                    ],
                    axis=-1,
                ),
                # `.tolist()` is correct here and only here: a string column is
                # never base64-encoded anyway, so the ndarray buys nothing.
                "hovertext": analytics.dominant_labels(
                    result, reference_label, comparison_label
                ).tolist(),
            }
        )
    return rows


def comparison_csv(
    results: list[analytics.ComparisonSeries],
    labels: dict[str, str],
) -> str:
    """Every merged-grid point, undefined ones included.

    A row whose ratio is undefined is kept with blank calculated fields and a
    populated `invalid_reason`, so the export shows where the comparison could
    not be made rather than silently omitting those energies. The source cross
    sections are always present. Orientation matches the chart exactly:
    comparison / reference.
    """
    rows = []
    for result in results:
        if result.size == 0:
            continue
        reference_label = labels.get(result.reference_key, result.reference_key)
        comparison_label = labels.get(
            result.comparison_key, result.comparison_key
        )
        dominant = analytics.dominant_labels(
            result, reference_label, comparison_label
        )
        crossing = (
            np.isin(result.energy, result.crossing_energies)
            if result.crossing_energies.size
            else np.zeros(result.energy.shape, dtype=bool)
        )
        reasons = result.invalid_reason

        for index in range(result.size):
            is_valid = bool(result.valid[index])
            rows.append(
                [
                    float(result.energy[index]),
                    reference_label,
                    comparison_label,
                    float(result.reference_sigma[index]),
                    float(result.comparison_sigma[index]),
                    float(result.ratio[index]) if is_valid else None,
                    (
                        float(result.percent_difference[index])
                        if is_valid
                        else None
                    ),
                    str(dominant[index]),
                    is_valid,
                    "" if reasons is None else str(reasons[index]),
                    bool(crossing[index]),
                ]
            )
    return exports.write_csv(exports.COMPARISON_HEADER, rows)
