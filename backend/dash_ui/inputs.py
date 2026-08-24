"""Coerce and validate raw callback inputs.

This module exists so that every callback body reduces to:
coerce -> call a service -> build a figure -> `except InputError`. Keeping
the coercion here is what makes "no scientific calculation inside callbacks"
mechanically checkable: `callbacks.py` imports neither `numpy` nor `math`.

Error messages are the ones the TypeScript produced, so the inline error
text is unchanged for users.
"""

import math
import re
from dataclasses import dataclass

import numpy as np

from ..services import analytics, multigroup

SPLIT_PATTERN = re.compile(r"[,\s]+")


class InputError(ValueError):
    """A user-correctable problem with the control values."""


@dataclass(frozen=True, slots=True)
class BarControls:
    bin_count: int
    custom_edges: np.ndarray | None
    energy_bounds: analytics.EnergyBounds | None
    mode: str
    field: str
    structure_name: str


def optional_number(value, label: str) -> float | None:
    """Blank stays blank; anything non-numeric is a user error."""
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise InputError(f"{label} must be finite.") from error
    if not math.isfinite(number):
        raise InputError(f"{label} must be finite.")
    return number


def read_filters(
    energy_min, energy_max, value_min, value_max
) -> analytics.RangeFilters:
    filters = analytics.RangeFilters(
        energy_min=optional_number(energy_min, "Minimum energy"),
        energy_max=optional_number(energy_max, "Maximum energy"),
        value_min=optional_number(value_min, "Minimum cross section"),
        value_max=optional_number(value_max, "Maximum cross section"),
    )
    if (
        filters.energy_min is not None
        and filters.energy_max is not None
        and filters.energy_min >= filters.energy_max
    ):
        raise InputError("Minimum energy must be less than maximum energy.")
    if (
        filters.value_min is not None
        and filters.value_max is not None
        and filters.value_min >= filters.value_max
    ):
        raise InputError("Minimum cross section must be less than maximum.")
    return filters


def parse_custom_bin_edges(text: str | None) -> np.ndarray | None:
    """Parse the free-text edge list. Blank falls back to automatic edges."""
    if not text or not text.strip():
        return None
    parts = [part for part in SPLIT_PATTERN.split(text.strip()) if part]
    try:
        values = [float(part) for part in parts]
    except ValueError as error:
        raise InputError("Bin edges must be a list of numbers.") from error
    if any(not math.isfinite(value) for value in values):
        raise InputError("Bin edges must be a list of numbers.")
    return np.array(values, dtype=np.float64)


def bar_mode_field(mode: str) -> str:
    return (
        "cross_section_group_average"
        if mode == "density"
        else "cross_section_mean"
    )


def bar_mode_title(mode: str) -> str:
    return (
        "Energy-averaged cross section (barns)"
        if mode == "density"
        else "Mean Cross Section (barns)"
    )


def read_bar_controls(
    structure_mode: str,
    bin_count,
    preset: str | None,
    edges_text: str | None,
    group_energy_min,
    group_energy_max,
    bar_mode: str,
) -> BarControls:
    """Resolve the active Group Structure mode into edges/bounds."""
    try:
        count = int(bin_count) if bin_count is not None else 40
    except (TypeError, ValueError) as error:
        raise InputError("Bin count must be a whole number.") from error

    custom_edges: np.ndarray | None = None
    energy_bounds: analytics.EnergyBounds | None = None

    if structure_mode == "standard":
        if not preset:
            raise InputError("Choose a group structure.")
        try:
            custom_edges = multigroup.structure_edges(preset)
        except multigroup.UnknownStructureError as error:
            raise InputError("Choose a group structure.") from error
        structure_name = preset
    elif structure_mode == "custom":
        custom_edges = parse_custom_bin_edges(edges_text)
        structure_name = "Custom edges"
    else:
        energy_bounds = analytics.EnergyBounds(
            minimum=optional_number(group_energy_min, "Min energy"),
            maximum=optional_number(group_energy_max, "Max energy"),
        )
        # Same rule `read_filters` applies to the display filters. Caught here
        # as well as in `analytics._validate_automatic_bounds` so the message
        # names the controls the user actually typed into; the service-side
        # check is what also covers the scale-dependent positivity rule, which
        # needs an x_scale this function is not given.
        if (
            energy_bounds.minimum is not None
            and energy_bounds.maximum is not None
            and energy_bounds.minimum >= energy_bounds.maximum
        ):
            raise InputError("Min energy must be less than Max energy.")
        structure_name = f"Automatic ({count} bins)"

    return BarControls(
        bin_count=count,
        custom_edges=custom_edges,
        energy_bounds=energy_bounds,
        mode=bar_mode,
        field=bar_mode_field(bar_mode),
        structure_name=structure_name,
    )


def series_label(item, comparison_mode: str) -> str:
    """Label a series by whichever dimension varies across the comparison."""
    if comparison_mode == "databases":
        return item.database
    if comparison_mode == "isotopes":
        return item.isotope
    if comparison_mode == "datasets":
        return item.dataset
    # U+00B7 MIDDLE DOT, matching the TypeScript.
    return f"{item.database} · {item.isotope} · {item.dataset}"


def as_list(value) -> list[str]:
    """Normalize a Dash dropdown value, which may be scalar or a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if item]
    return [value]


COMPARISON_METRICS = ("ratio", "percent")

COMPARISON_AXIS_TITLES = {
    "ratio": "Cross-Section Ratio (comparison / reference)",
    "percent": "Difference Relative to Reference (%)",
}


@dataclass(frozen=True, slots=True)
class ComparisonControls:
    metric: str
    reference: analytics.SeriesArrays
    comparisons: list[analytics.SeriesArrays]


def comparison_axis_title(metric: str) -> str:
    return COMPARISON_AXIS_TITLES.get(
        metric, COMPARISON_AXIS_TITLES["ratio"]
    )


def comparison_formula(reference_label: str, comparison_label: str) -> str:
    """The orientation, spelled out: 'JEFF-3.3 / ENDF/B-VIII.0'.

    Shown next to the controls because "ratio" alone does not say which way
    round the division goes, and the answer inverts the whole chart.
    """
    return f"{comparison_label} / {reference_label}"


def read_comparison_controls(
    metric: str | None,
    reference_key: str | None,
    target_keys,
    series: list[analytics.SeriesArrays],
) -> ComparisonControls:
    """Resolve the Ratio / Difference selection into concrete series.

    Falls back to "first series is the reference, everything else is compared
    against it" whenever a selection is missing, so the chart draws on the
    first render rather than waiting for the selector-population callback to
    land.
    """
    if len(series) < 2:
        raise InputError(
            "Select at least two series to compare. Use a comparison mode "
            "that loads more than one evaluation."
        )

    chosen = metric if metric in COMPARISON_METRICS else "ratio"
    by_key = {item.key: item for item in series}

    if reference_key not in by_key:
        reference_key = series[0].key
    reference = by_key[reference_key]

    requested = [key for key in as_list(target_keys) if key in by_key]
    # Comparing a series against itself is the one selection the UI can
    # produce that has no meaning; drop it rather than erroring, because the
    # reference dropdown can legitimately land on an already-selected target.
    requested = [key for key in requested if key != reference_key]
    if not requested:
        requested = [item.key for item in series if item.key != reference_key]
    if not requested:
        raise InputError(
            "Choose at least one series to compare against the reference."
        )

    return ComparisonControls(
        metric=chosen,
        reference=reference,
        comparisons=[by_key[key] for key in requested],
    )


def energy_only_filters(
    filters: analytics.RangeFilters,
) -> analytics.RangeFilters:
    """The energy bounds alone, dropping the cross-section value bounds.

    Used by the Ratio / Difference chart. A value bound removes points by
    cross-section magnitude, which can empty out the middle of a curve; the
    merged-grid interpolation would then span that hole and draw a ratio
    across energies where one of the two evaluations has no data behind it.
    """
    return analytics.RangeFilters(
        energy_min=filters.energy_min, energy_max=filters.energy_max
    )
