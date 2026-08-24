"""Cross-section transformations: filtering, binning, integration, KDE.

Ported from the TypeScript SPA's `analytics.ts` (removed at cutover; see git
history). Numerical behaviour is preserved deliberately -
`tests/test_parity_analytics.py` compares this module against golden fixtures
captured from that implementation, so changes here that alter results will
fail that suite rather than drift silently.

Two places where an obvious "improvement" would break parity, both marked
inline below:

* automatic edges are built with scalar `float.__pow__`, not `np.power`,
  because a 1-ulp difference can move a point that sits exactly on an edge
  into the neighbouring bin;
* `_sweep_group_integrals` stays a Python loop, because its cursor carry
  (a point on an edge is shared by both neighbouring groups) is stateful.
"""

import math
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

ScaleChoice = Literal["linear", "log"]
BarMode = Literal["mean", "density"]

MAX_KDE_SAMPLES = 5_000
KDE_EVALUATION_COUNT = 200
MIN_AUTOMATIC_BINS = 10
MAX_AUTOMATIC_BINS = 100


class BinEdgeError(ValueError):
    """Bin edges the binning/integration algorithms cannot use safely.

    The messages are the exact strings the TypeScript threw, so the UI can
    surface them verbatim.
    """


@dataclass(frozen=True, slots=True)
class SeriesArrays:
    """One series' points as parallel arrays, sorted by ascending energy.

    Sorting is done once, at load time, rather than inside `bin_series` on
    every render as the TypeScript did. Every function here assumes ascending
    `energy` as a precondition.
    """

    key: str
    database: str
    isotope: str
    dataset: str
    energy: np.ndarray
    sigma: np.ndarray
    stddev: np.ndarray | None = None

    @property
    def size(self) -> int:
        return int(self.energy.size)


@dataclass(frozen=True, slots=True)
class RangeFilters:
    energy_min: float | None = None
    energy_max: float | None = None
    value_min: float | None = None
    value_max: float | None = None


@dataclass(frozen=True, slots=True)
class EnergyBounds:
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True, slots=True)
class GroupPoint:
    series_key: str
    group_index: int
    bin_start: float
    bin_end: float
    bin_width: float
    bin_center: float
    point_count: int
    cross_section_mean: float | None
    cross_section_group_average: float | None
    cross_section_integral: float | None
    coverage_fraction: float


@dataclass(frozen=True, slots=True)
class GroupingResult:
    records: list[GroupPoint]
    edges: np.ndarray
    x_scale: ScaleChoice
    excluded_nonpositive: int


@dataclass(frozen=True, slots=True)
class KdeCurve:
    series_key: str
    x: np.ndarray
    density: np.ndarray


@dataclass(frozen=True, slots=True)
class KdeResult:
    curves: list[KdeCurve]
    x_scale: ScaleChoice
    excluded_nonpositive: int


def series_from_points(
    key: str,
    database: str,
    isotope: str,
    dataset: str,
    points: list[dict],
) -> SeriesArrays:
    """Build a sorted `SeriesArrays` from `charts.build_records` output.

    Missing standard deviations become `np.nan`, never 0 - JANIS does not
    report a stddev at every energy the central value has, and conflating
    "unknown" with "zero uncertainty" would be a physics error.
    """
    energy = np.fromiter(
        (point["energy_MeV"] for point in points), dtype=np.float64, count=len(points)
    )
    sigma = np.fromiter(
        (point["cross_section_barns"] for point in points),
        dtype=np.float64,
        count=len(points),
    )
    stddev = None
    if points and "cross_section_stddev_barns" in points[0]:
        stddev = np.array(
            [
                np.nan
                if point.get("cross_section_stddev_barns") is None
                else float(point["cross_section_stddev_barns"])
                for point in points
            ],
            dtype=np.float64,
        )

    # Sort once. JANIS tables normally arrive ascending, so this is usually a
    # single O(n) scan that finds nothing to do.
    if energy.size > 1 and not np.all(np.diff(energy) >= 0):
        order = np.argsort(energy, kind="stable")
        energy = energy[order]
        sigma = sigma[order]
        if stddev is not None:
            stddev = stddev[order]

    return SeriesArrays(
        key=key,
        database=database,
        isotope=isotope,
        dataset=dataset,
        energy=energy,
        sigma=sigma,
        stddev=stddev,
    )


def filter_series(
    series: list[SeriesArrays], filters: RangeFilters
) -> list[SeriesArrays]:
    """Keep points falling inside every configured inclusive range.

    Series are never dropped, only emptied - the caller distinguishes "this
    series has no points in range" from "this series was not requested".
    """
    filtered: list[SeriesArrays] = []
    for item in series:
        mask = np.ones(item.energy.shape, dtype=bool)
        if filters.energy_min is not None:
            mask &= item.energy >= filters.energy_min
        if filters.energy_max is not None:
            mask &= item.energy <= filters.energy_max
        if filters.value_min is not None:
            mask &= item.sigma >= filters.value_min
        if filters.value_max is not None:
            mask &= item.sigma <= filters.value_max

        if mask.all():
            filtered.append(item)
            continue

        filtered.append(
            replace(
                item,
                energy=item.energy[mask],
                sigma=item.sigma[mask],
                stddev=None if item.stddev is None else item.stddev[mask],
            )
        )
    return filtered


def deterministic_sample(values: np.ndarray, maximum_size: int) -> np.ndarray:
    """Evenly distributed source samples, without randomness.

    Always retains the first and last value. The index expression keeps the
    TypeScript's operand order (`i * (n - 1) / (m - 1)`) so the IEEE-754
    result - and therefore the chosen indices - are identical.
    """
    count = values.size
    if count <= maximum_size:
        return values
    if maximum_size <= 1:
        return values[:1]

    positions = (
        np.arange(maximum_size, dtype=np.float64) * (count - 1) / (maximum_size - 1)
    )
    return values[np.floor(positions).astype(np.int64)]


def _expanded_extent(
    minimum: float, maximum: float, scale: ScaleChoice
) -> tuple[float, float]:
    if minimum != maximum:
        return minimum, maximum
    if scale == "log" and minimum > 0:
        root_ten = math.sqrt(10)
        return minimum / root_ten, maximum * root_ten
    padding = abs(minimum) * 0.05 or 0.5
    return minimum - padding, maximum + padding


def _build_edges(
    minimum: float, maximum: float, count: int, scale: ScaleChoice
) -> np.ndarray:
    start, end = _expanded_extent(minimum, maximum, scale)
    if scale == "log":
        log_start = math.log10(start)
        step = (math.log10(end) - log_start) / count
        # Scalar pow, NOT np.power: CPython's float.__pow__ and V8's Math.pow
        # both call libm, so the edges match the TypeScript bit for bit. A
        # 1-ulp shift here can move a point sitting exactly on an edge into
        # the neighbouring group.
        return np.array(
            [10 ** (log_start + index * step) for index in range(count + 1)],
            dtype=np.float64,
        )

    step = (end - start) / count
    return np.array(
        [start + index * step for index in range(count + 1)], dtype=np.float64
    )


def _validate_custom_edges(edges: np.ndarray, scale: ScaleChoice) -> None:
    if edges.size < 2:
        raise BinEdgeError("Bin edges must contain at least two values.")
    if not np.all(np.isfinite(edges)):
        raise BinEdgeError("Bin edges must be finite numbers.")
    if scale == "log" and np.any(edges <= 0):
        raise BinEdgeError("Bin edges must be positive for a logarithmic scale.")
    if np.any(np.diff(edges) <= 0):
        raise BinEdgeError("Bin edges must be strictly ascending.")


def _validate_automatic_bounds(
    lower: float, upper: float, scale: ScaleChoice
) -> None:
    """Check the resolved automatic-mode extent before building edges.

    The custom-edge path is validated by `_validate_custom_edges`; this is its
    counterpart for the extent the automatic path derives from the data and the
    caller's `EnergyBounds`. Without it a user-supplied bound reaches
    `_build_edges` unchecked, where `minimum <= 0` on a log scale raises a bare
    `ValueError` from `math.log10` - not a `BinEdgeError`, so callers that
    handle the user-correctable family miss it - and an inverted range silently
    produces a *descending* edge array, which `np.searchsorted` in
    `_bin_indices` is not defined on.

    Equal bounds stay legal: `_expanded_extent` widens them deliberately.
    """
    if lower > upper:
        raise BinEdgeError("Minimum energy must be less than maximum energy.")
    if scale == "log" and lower <= 0:
        raise BinEdgeError(
            "Minimum energy must be positive for a logarithmic scale."
        )


def _bin_indices(energy: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of the TypeScript's per-point binary search.

    `side="right"` yields the largest index whose edge is <= the value, which
    is what the hand-written loop computed. The two fix-ups reproduce its
    boundary handling: a value equal to the final edge belongs to the last
    group, and anything outside the edge span is rejected.
    """
    count = edges.size - 1
    indices = np.searchsorted(edges, energy, side="right") - 1
    indices[energy == edges[-1]] = count - 1
    indices[(energy < edges[0]) | (energy > edges[-1])] = -1
    return indices


def _interpolate_between(
    prev_energy: float,
    prev_value: float,
    next_energy: float,
    next_value: float,
    energy: float,
) -> float:
    span = next_energy - prev_energy
    if span <= 0:
        return prev_value
    fraction = (energy - prev_energy) / span
    return prev_value + fraction * (next_value - prev_value)


def _sweep_group_integrals(
    energy: list[float], sigma: list[float], edges: np.ndarray
) -> tuple[list[float | None], list[float]]:
    """Trapezoidal integral of the piecewise-linear curve over each group.

    Single O(groups + points) sweep: `point_idx`/`cursor` only ever advance
    across the whole sweep, never resetting per group, which is what keeps
    this from being the O(groups * points) a per-group rescan would cost -
    required for a 725-group structure against a ~30,000-point series.

    Deliberately NOT vectorized. Three coupled properties make it stateful:
    the `point_idx = cursor` carry (a point landing exactly on an edge is
    re-examined by the next group, so both neighbours share it), the
    `point_idx == 0` clamp, and per-series domain clipping. The loop is over
    groups (<= 726), not points, so the Python overhead is negligible.

    Iterates over Python lists rather than numpy arrays: scalar indexing of
    an ndarray costs ~5x more per element because of scalar boxing, and this
    is the only hot Python loop in the codebase.
    """
    group_count = edges.size - 1
    integrals: list[float | None] = [None] * group_count
    coverages: list[float] = [0.0] * group_count
    n = len(energy)
    domain_start = energy[0]
    domain_end = energy[-1]
    point_idx = 0

    for group in range(group_count):
        lo = edges[group]
        hi = edges[group + 1]
        overlap_lo = max(lo, domain_start)
        overlap_hi = min(hi, domain_end)

        if overlap_lo >= overlap_hi:
            continue

        while point_idx < n and energy[point_idx] < overlap_lo:
            point_idx += 1

        # overlap_lo <= energy[0] whenever point_idx == 0 here, so there is
        # no "point before index 0" to bracket against - clamp instead.
        prev_energy = overlap_lo
        if point_idx == 0:
            prev_value = sigma[0]
        else:
            prev_value = _interpolate_between(
                energy[point_idx - 1],
                sigma[point_idx - 1],
                energy[point_idx],
                sigma[point_idx],
                overlap_lo,
            )

        integral = 0.0
        cursor = point_idx
        while cursor < n and energy[cursor] < overlap_hi:
            point_energy = energy[cursor]
            if point_energy > overlap_lo:
                integral += (
                    (point_energy - prev_energy)
                    * (prev_value + sigma[cursor])
                    / 2
                )
                prev_energy = point_energy
                prev_value = sigma[cursor]
            cursor += 1

        # JavaScript would throw on an out-of-range index here; Python would
        # silently wrap to the LAST element and return a plausible wrong
        # number. Both bounds provably hold - assert so a future change to
        # the clipping logic fails loudly instead.
        assert 1 <= cursor < n, f"sweep cursor {cursor} out of range for {n} points"

        last_value = _interpolate_between(
            energy[cursor - 1],
            sigma[cursor - 1],
            energy[cursor],
            sigma[cursor],
            overlap_hi,
        )
        integral += (overlap_hi - prev_energy) * (prev_value + last_value) / 2

        integrals[group] = integral
        coverages[group] = overlap_hi - overlap_lo
        # Not cursor + 1: the next group may legitimately re-examine this
        # point when it sits exactly on the shared edge.
        point_idx = cursor

    return integrals, coverages


def bin_series(
    series: list[SeriesArrays],
    bin_count: int,
    scale_choice: ScaleChoice,
    custom_edges: np.ndarray | None = None,
    energy_bounds: EnergyBounds | None = None,
) -> GroupingResult:
    """Aggregate every series against shared energy edges.

    Each group reports two independent quantities:

    * `cross_section_mean` / `point_count` - the arithmetic mean of the raw
      samples that land in the group. Depends on local sample density, not on
      the shape of the curve.
    * `cross_section_group_average` / `cross_section_integral` - the
      energy-averaged cross section from the piecewise-linear integral of the
      series' own points. Defined whenever a group overlaps that series'
      data, *even with no landed sample*, and satisfies
      `sum(average_i * width_i) == integral` exactly over the covered range,
      for uniform or arbitrary edges.

    `coverage_fraction` reports how much of the group's width that series'
    data actually covers.
    """
    x_scale = scale_choice
    excluded_nonpositive = 0

    if custom_edges is not None:
        edges = np.asarray(custom_edges, dtype=np.float64)
        _validate_custom_edges(edges, x_scale)
        if x_scale == "log":
            excluded_nonpositive = sum(
                int(np.count_nonzero(item.energy <= 0)) for item in series
            )
    else:
        usable = []
        for item in series:
            if x_scale == "log":
                positive = item.energy[item.energy > 0]
                excluded_nonpositive += item.size - positive.size
                usable.append(positive)
            else:
                usable.append(item.energy)

        combined = [part for part in usable if part.size]
        if not combined:
            return GroupingResult([], np.empty(0), x_scale, excluded_nonpositive)

        data_min = min(float(part.min()) for part in combined)
        data_max = max(float(part.max()) for part in combined)
        lower = (
            data_min
            if energy_bounds is None or energy_bounds.minimum is None
            else energy_bounds.minimum
        )
        upper = (
            data_max
            if energy_bounds is None or energy_bounds.maximum is None
            else energy_bounds.maximum
        )
        _validate_automatic_bounds(lower, upper, x_scale)
        count = max(
            MIN_AUTOMATIC_BINS,
            min(MAX_AUTOMATIC_BINS, int(round(bin_count))),
        )
        edges = _build_edges(lower, upper, count, x_scale)

    group_count = edges.size - 1
    starts = edges[:-1]
    ends = edges[1:]
    widths = ends - starts
    centers = (
        np.sqrt(starts * ends) if x_scale == "log" else (starts + ends) / 2
    )

    records: list[GroupPoint] = []

    for item in series:
        if x_scale == "log":
            usable_mask = item.energy > 0
            energy = item.energy[usable_mask]
            sigma = item.sigma[usable_mask]
        else:
            energy = item.energy
            sigma = item.sigma

        totals = np.zeros(group_count, dtype=np.int64)
        sums = np.zeros(group_count, dtype=np.float64)
        if energy.size:
            indices = _bin_indices(energy, edges)
            landed = indices >= 0
            if landed.any():
                landed_indices = indices[landed]
                totals = np.bincount(landed_indices, minlength=group_count)
                sums = np.bincount(
                    landed_indices, weights=sigma[landed], minlength=group_count
                )

        if energy.size >= 2:
            integrals, coverages = _sweep_group_integrals(
                energy.tolist(), sigma.tolist(), edges
            )
        else:
            integrals = [None] * group_count
            coverages = [0.0] * group_count

        for index in range(group_count):
            has_mean = totals[index] > 0
            integral = integrals[index]
            if not has_mean and integral is None:
                continue

            width = float(widths[index])
            records.append(
                GroupPoint(
                    series_key=item.key,
                    group_index=index,
                    bin_start=float(starts[index]),
                    bin_end=float(ends[index]),
                    bin_width=width,
                    bin_center=float(centers[index]),
                    point_count=int(totals[index]),
                    cross_section_mean=(
                        float(sums[index] / totals[index]) if has_mean else None
                    ),
                    cross_section_group_average=(
                        None if integral is None else integral / width
                    ),
                    cross_section_integral=integral,
                    coverage_fraction=min(
                        1.0, max(0.0, coverages[index] / width)
                    ),
                )
            )

    return GroupingResult(records, edges, x_scale, excluded_nonpositive)


def _silverman_bandwidth(values: np.ndarray) -> float:
    """Silverman's rule of thumb on the (possibly log-transformed) values."""
    if values.size < 2:
        deviation = 0.0
    else:
        # ddof=1: the TypeScript divides by (n - 1).
        deviation = float(np.std(values, ddof=1))
    if deviation > 0:
        return 1.06 * deviation * values.size ** -0.2
    magnitude = abs(float(values[0])) if values.size else 1.0
    return max(magnitude * 0.01, 0.01)


def kde_series(
    series: list[SeriesArrays],
    scale_choice: ScaleChoice,
    bandwidth_multiplier: float,
    evaluation_count: int = KDE_EVALUATION_COUNT,
    maximum_samples: int = MAX_KDE_SAMPLES,
) -> KdeResult:
    """Independently normalized Gaussian KDE curves on one shared domain."""
    x_scale = scale_choice
    excluded_nonpositive = (
        sum(int(np.count_nonzero(item.sigma <= 0)) for item in series)
        if x_scale == "log"
        else 0
    )

    transformed: list[tuple[str, np.ndarray]] = []
    for item in series:
        values = item.sigma[item.sigma > 0] if x_scale == "log" else item.sigma
        sampled = deterministic_sample(values, maximum_samples)
        transformed.append(
            (item.key, np.log10(sampled) if x_scale == "log" else sampled)
        )

    populated = [values for _, values in transformed if values.size]
    if not populated:
        return KdeResult([], x_scale, excluded_nonpositive)

    domain_min = min(float(values.min()) for values in populated)
    domain_max = max(float(values.max()) for values in populated)

    multiplier = max(0.25, min(4.0, bandwidth_multiplier))
    bandwidths = {
        key: (_silverman_bandwidth(values) * multiplier if values.size else 0.0)
        for key, values in transformed
    }
    maximum_bandwidth = max(bandwidths.values(), default=0.0)

    # 'linear' even in log mode: these values are already log10-transformed.
    base_start, base_end = _expanded_extent(domain_min, domain_max, "linear")
    domain_start = base_start - 3 * maximum_bandwidth
    domain_end = base_end + 3 * maximum_bandwidth
    point_count = max(2, evaluation_count)
    positions = (
        domain_start
        + np.arange(point_count, dtype=np.float64)
        * (domain_end - domain_start)
        / (point_count - 1)
    )

    normalizer = math.sqrt(2 * math.pi)
    curves: list[KdeCurve] = []

    for key, values in transformed:
        if not values.size:
            continue
        bandwidth = bandwidths[key]
        # 200 x 5000 kernel evaluations as one matrix op rather than a
        # 1,000,000-iteration double loop.
        standardized = (positions[:, None] - values[None, :]) / bandwidth
        density = np.exp(-0.5 * standardized * standardized).sum(axis=1) / (
            values.size * bandwidth * normalizer
        )

        if x_scale == "log":
            x = np.power(10.0, positions)
            # Jacobian for the log10 change of variables.
            density = density / (x * math.log(10))
        else:
            x = positions

        # `np.trapezoid` computes exactly this expression, so the two are
        # bit-identical; it is written out because dividing by the *same*
        # expression is what makes the unit-area assertion in
        # tests/test_analytics_kde.py exact by construction rather than
        # approximately true. (An older comment here claimed the reason was
        # the np.trapz -> np.trapezoid rename pinning the NumPy version. That
        # is stale: pyproject requires numpy>=2.0, where np.trapz no longer
        # exists at all.)
        area = float(np.sum(np.diff(x) * (density[1:] + density[:-1]) / 2))
        if area > 0:
            density = density / area

        curves.append(KdeCurve(series_key=key, x=x, density=density))

    return KdeResult(curves, x_scale, excluded_nonpositive)


def line_values(
    series: list[SeriesArrays],
    x_scale: ScaleChoice,
    y_scale: ScaleChoice,
    include_stddev: bool,
) -> tuple[list[dict], int]:
    """Prepare raw pointwise traces, dropping values a log axis cannot show.

    Plotly silently drops nonpositive values on a log axis; excluding them
    here instead keeps the reported count authoritative. Mirrors
    `computeLineValues` in the former app.ts, including nulling `y_low` when
    a log y-axis would reject the lower error bound.
    """
    prepared: list[dict] = []
    excluded = 0

    for item in series:
        mask = np.ones(item.energy.shape, dtype=bool)
        if x_scale == "log":
            mask &= item.energy > 0
        if y_scale == "log":
            mask &= item.sigma > 0
        excluded += int(np.count_nonzero(~mask))

        energy = item.energy[mask]
        sigma = item.sigma[mask]
        stddev = None
        if include_stddev and item.stddev is not None:
            stddev = item.stddev[mask]

        low = high = None
        if stddev is not None:
            low = sigma - stddev
            high = sigma + stddev
            if y_scale == "log":
                low = np.where(low > 0, low, np.nan)

        prepared.append(
            {
                "series": item,
                "energy": energy,
                "sigma": sigma,
                "stddev": stddev,
                "y_low": low,
                "y_high": high,
            }
        )

    return prepared, excluded


# ---------------------------------------------------------------------------
# Ratio / difference comparison
# ---------------------------------------------------------------------------
#
# Compares one evaluation against another as a function of incident energy.
# The orientation is always comparison / reference, in one of two metrics:
#
#     ratio    R(E) = sigma_c(E) / sigma_r(E)              baseline 1
#     percent  D(E) = 100 * (sigma_c(E) - sigma_r(E)) / sigma_r(E)   baseline 0
#
# Both are computed in a single pass, so switching the displayed metric never
# repeats the interpolation work.

ComparisonMetric = Literal["ratio", "percent"]

# Relative tolerance for the hover "Dominant" label ONLY. It never invalidates
# a point and never appears in a denominator - it exists so that a point which
# is equal to within rounding does not claim a winner. Deliberately distinct
# from a denominator tolerance, which would silently discard physically
# meaningful small nonzero cross sections.
EQUALITY_RELATIVE_TOLERANCE = 1e-9

REASON_VALID = ""
REASON_REFERENCE_ZERO = "reference_zero"
REASON_BOTH_ZERO = "both_zero"
REASON_NONFINITE_REFERENCE = "nonfinite_reference"
REASON_NONFINITE_COMPARISON = "nonfinite_comparison"
REASON_DISCONTINUITY = "discontinuity"

# Wide enough for the longest reason above; a fixed-width unicode array keeps
# this a compact numpy column rather than a list of Python strings.
_REASON_DTYPE = "<U21"


class IncompatibleSeriesError(ValueError):
    """Two series whose ratio would not be a meaningful dimensionless quantity.

    A `ValueError` rather than an `errors.DomainError`, following
    `BinEdgeError`: this module stays independent of the transport layer, and
    the Dash callbacks already catch that family for inline display.
    """


@dataclass(frozen=True, slots=True)
class ComparisonSeries:
    """One reference/comparison pair evaluated on their merged energy grid.

    Every array is sorted by ascending energy and has the same length.
    `ratio` and `percent_difference` are `np.nan` wherever `valid` is False -
    never +/-inf, and never 0, because "undefined" and "equal" are different
    statements. `reference_sigma`/`comparison_sigma` keep their source values
    at those energies so an export can still show what was there.

    `unavailable_reason` is set only when the pair produced no grid at all
    (no energy overlap, or an empty series); the arrays are then empty. It is
    how a `list[ComparisonSeries]` can carry a descriptive empty result.
    """

    reference_key: str
    comparison_key: str

    energy: np.ndarray
    reference_sigma: np.ndarray
    comparison_sigma: np.ndarray

    ratio: np.ndarray
    percent_difference: np.ndarray

    valid: np.ndarray
    invalid_reason: np.ndarray | None

    overlap_min: float
    overlap_max: float
    crossing_energies: np.ndarray

    unavailable_reason: str | None = None

    @property
    def size(self) -> int:
        return int(self.energy.size)

    @property
    def valid_count(self) -> int:
        return int(np.count_nonzero(self.valid))

    @property
    def excluded_count(self) -> int:
        return int(self.energy.size - np.count_nonzero(self.valid))


def check_comparable(
    reference: SeriesArrays, comparison: SeriesArrays
) -> None:
    """Reject pairs whose ratio would not be dimensionless in the intended sense.

    Cross sections are always barns against MeV here (`field="SIG"`), so there
    is never a unit to convert - the whole compatibility question reduces to
    whether the two series describe the same nuclide and the same reaction
    channel. The comparison-shape invariant means that in practice this passes
    for a `databases` comparison and fails for `isotopes`/`datasets`, which is
    correct: those vary precisely the quantities that must match.
    """
    if reference.key == comparison.key:
        raise IncompatibleSeriesError(
            "The reference and comparison are the same series. Choose a "
            "different series to compare against."
        )
    if reference.isotope != comparison.isotope:
        raise IncompatibleSeriesError(
            f"Cannot compare {comparison.isotope} against {reference.isotope}: "
            "a ratio between different nuclides is not a meaningful "
            "dimensionless quantity. Use Compare databases to hold the "
            "nuclide fixed."
        )
    if reference.dataset != comparison.dataset:
        raise IncompatibleSeriesError(
            f"Cannot compare {comparison.dataset} against {reference.dataset}: "
            "a ratio between different reaction channels is not a meaningful "
            "dimensionless quantity. Use Compare databases to hold the "
            "reaction fixed."
        )


def _empty_comparison(
    reference: SeriesArrays,
    comparison: SeriesArrays,
    reason: str,
    overlap_min: float = math.nan,
    overlap_max: float = math.nan,
) -> ComparisonSeries:
    def empty(dtype=np.float64) -> np.ndarray:
        return np.empty(0, dtype=dtype)

    return ComparisonSeries(
        reference_key=reference.key,
        comparison_key=comparison.key,
        energy=empty(),
        reference_sigma=empty(),
        comparison_sigma=empty(),
        ratio=empty(),
        percent_difference=empty(),
        valid=empty(bool),
        invalid_reason=empty(_REASON_DTYPE),
        overlap_min=overlap_min,
        overlap_max=overlap_max,
        crossing_energies=empty(),
        unavailable_reason=reason,
    )


def _duplicate_energies(energy: np.ndarray) -> np.ndarray:
    """Energies repeated inside one series - a discontinuity, not a duplicate.

    A repeated energy carries two different cross sections, so averaging them
    would invent a value that is in neither evaluation, and interpolating
    across the pair would draw a false diagonal through a jump. The energies
    are returned so the caller can break the line and suppress crossings
    there instead.
    """
    if energy.size < 2:
        return np.empty(0, dtype=np.float64)
    return np.unique(energy[:-1][np.diff(energy) == 0])


def _merge_grids(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Sorted, deduplicated union of two already-sorted arrays.

    `searchsorted` + `insert` runs in O(M log N + N + M) entirely inside
    numpy. A true O(N+M) two-way merge would need a Python loop, which at
    N = 30,000 costs orders of magnitude more than the log factor saves -
    numpy has no linear merge primitive to call instead. `np.union1d` is
    avoided because its full sort of the concatenation is strictly worse.
    """
    if left.size == 0:
        merged = right
    elif right.size == 0:
        merged = left
    else:
        merged = np.insert(left, np.searchsorted(left, right), right)
    if merged.size < 2:
        return np.asarray(merged, dtype=np.float64)
    # One grid point per distinct energy. Runs of equal values are collapsed
    # here and handled separately as discontinuities, which is what keeps
    # np.interp from being asked to resolve an ambiguous x.
    keep = np.concatenate(([True], np.diff(merged) > 0))
    return np.asarray(merged[keep], dtype=np.float64)


def _crossing_energies(
    grid: np.ndarray, difference: np.ndarray, discontinuities: np.ndarray
) -> np.ndarray:
    """Energies where the two curves cross, solved exactly on each segment.

    Under piecewise-linear interpolation `d(E) = sigma_c - sigma_r` is linear
    between adjacent grid points, so a sign change brackets exactly one root
    at `E0 - d0 * (E1 - E0) / (d1 - d0)`. Inserting it into the output grid is
    what makes every crossing of the source curves land exactly on the
    baseline rather than being stepped over.
    """
    if grid.size < 2:
        return np.empty(0, dtype=np.float64)

    d0, d1 = difference[:-1], difference[1:]
    e0, e1 = grid[:-1], grid[1:]
    span = e1 - e0
    delta = d1 - d0

    # Equivalent to `d0 * d1 < 0` but without the overflow-to-inf that a
    # product of two large barns values can hit, which would silently drop a
    # legitimate crossing.
    opposite_signs = (d0 < 0) != (d1 < 0)
    crosses = (
        opposite_signs
        & (d0 != 0)
        & (d1 != 0)
        & np.isfinite(d0)
        & np.isfinite(d1)
        & (span > 0)
        & (delta != 0)
    )
    if discontinuities.size:
        # An interval with a jump at either end has no single linear segment
        # to solve on, so any root found there would be fabricated.
        crosses &= ~np.isin(e0, discontinuities)
        crosses &= ~np.isin(e1, discontinuities)

    if not crosses.any():
        return np.empty(0, dtype=np.float64)

    lo, hi = e0[crosses], e1[crosses]
    roots = lo - d0[crosses] * (span[crosses] / delta[crosses])
    # Rounding can push the root a ulp outside its own bracket; clipping keeps
    # the inserted energy inside the interval that produced it, so the merged
    # grid stays sorted.
    return np.unique(np.clip(roots, lo, hi))


def _compare_pair(
    reference: SeriesArrays,
    comparison: SeriesArrays,
    energy_min: float | None,
    energy_max: float | None,
) -> ComparisonSeries:
    if reference.size == 0 or comparison.size == 0:
        return _empty_comparison(
            reference,
            comparison,
            "One of the selected series has no points in the current range.",
        )

    overlap_min = max(float(reference.energy[0]), float(comparison.energy[0]))
    overlap_max = min(float(reference.energy[-1]), float(comparison.energy[-1]))
    if energy_min is not None:
        overlap_min = max(overlap_min, energy_min)
    if energy_max is not None:
        overlap_max = min(overlap_max, energy_max)

    if overlap_min > overlap_max:
        return _empty_comparison(
            reference,
            comparison,
            "The selected series do not overlap in energy, so there is "
            "nothing to compare.",
            overlap_min,
            overlap_max,
        )

    # Only the grid is clipped. Interpolation still runs against the full
    # source arrays so a point sitting on the overlap boundary is bracketed by
    # its true neighbours rather than by a truncated endpoint.
    def clipped(energy: np.ndarray) -> np.ndarray:
        lo = np.searchsorted(energy, overlap_min, side="left")
        hi = np.searchsorted(energy, overlap_max, side="right")
        return energy[lo:hi]

    grid = _merge_grids(clipped(reference.energy), clipped(comparison.energy))
    # The two overlap boundaries are always grid points, so the comparison
    # spans the whole common domain even when neither series has a point there.
    grid = _merge_grids(grid, np.array([overlap_min, overlap_max], dtype=np.float64))

    discontinuities = np.union1d(
        _duplicate_energies(reference.energy),
        _duplicate_energies(comparison.energy),
    )

    # np.interp is piecewise-linear, matching `_interpolate_between` above,
    # and clamps rather than extrapolates - unreachable anyway, since the grid
    # is contained in the overlap by construction.
    ref_sigma = np.interp(grid, reference.energy, reference.sigma)
    cmp_sigma = np.interp(grid, comparison.energy, comparison.sigma)

    crossings = _crossing_energies(grid, cmp_sigma - ref_sigma, discontinuities)
    if crossings.size:
        grid = _merge_grids(grid, crossings)
        ref_sigma = np.interp(grid, reference.energy, reference.sigma)
        cmp_sigma = np.interp(grid, comparison.energy, comparison.sigma)

    finite_ref = np.isfinite(ref_sigma)
    finite_cmp = np.isfinite(cmp_sigma)
    on_discontinuity = (
        np.isin(grid, discontinuities)
        if discontinuities.size
        else np.zeros(grid.shape, dtype=bool)
    )
    zero_ref = finite_ref & (ref_sigma == 0)
    both_zero = zero_ref & finite_cmp & (cmp_sigma == 0)

    valid = finite_ref & finite_cmp & ~zero_ref & ~on_discontinuity

    # Assigned least-specific first so the more specific reason overwrites:
    # both_zero beats reference_zero, and a discontinuity - a statement about
    # the grid rather than the values - beats either.
    reason = np.full(grid.shape, REASON_VALID, dtype=_REASON_DTYPE)
    reason[~finite_cmp] = REASON_NONFINITE_COMPARISON
    reason[~finite_ref] = REASON_NONFINITE_REFERENCE
    reason[zero_ref] = REASON_REFERENCE_ZERO
    reason[both_zero] = REASON_BOTH_ZERO
    reason[on_discontinuity] = REASON_DISCONTINUITY

    ratio = np.full(grid.shape, np.nan, dtype=np.float64)
    percent = np.full(grid.shape, np.nan, dtype=np.float64)
    # `where=valid` means the division is never executed at an undefined
    # point, so no inf is produced to clean up afterwards. Percent difference
    # is taken directly from (c - r) / r rather than from 100 * (R - 1): the
    # two are algebraically equal and agree exactly on 2x and 0.5x, but the
    # direct form keeps its precision near equality, which is where this chart
    # is read most closely.
    with np.errstate(invalid="ignore", over="ignore"):
        np.divide(cmp_sigma, ref_sigma, out=ratio, where=valid)
        np.divide(cmp_sigma - ref_sigma, ref_sigma, out=percent, where=valid)
        percent *= 100.0

    return ComparisonSeries(
        reference_key=reference.key,
        comparison_key=comparison.key,
        energy=grid,
        reference_sigma=ref_sigma,
        comparison_sigma=cmp_sigma,
        ratio=ratio,
        percent_difference=percent,
        valid=valid,
        invalid_reason=reason,
        overlap_min=overlap_min,
        overlap_max=overlap_max,
        crossing_energies=crossings,
    )


def compare_cross_sections(
    reference: SeriesArrays,
    comparisons: list[SeriesArrays],
    *,
    energy_min: float | None = None,
    energy_max: float | None = None,
) -> list[ComparisonSeries]:
    """Compare each series in `comparisons` against `reference`.

    Raises `IncompatibleSeriesError` if any pair describes a different nuclide
    or reaction channel - checked for every pair before any work is done, so a
    rejected selection never renders a partial chart.
    """
    for comparison in comparisons:
        check_comparable(reference, comparison)
    return [
        _compare_pair(reference, comparison, energy_min, energy_max)
        for comparison in comparisons
    ]


def dominant_labels(
    result: ComparisonSeries,
    reference_label: str,
    comparison_label: str,
) -> np.ndarray:
    """Per-point winner text for the hover readout.

    Uses `EQUALITY_RELATIVE_TOLERANCE` so a ratio equal to 1 within rounding -
    every inserted crossing, for instance - reads as equal rather than
    claiming a winner on the last bit.
    """
    labels = np.full(result.energy.shape, "", dtype=object)
    if result.size == 0:
        return labels
    deviation = result.ratio - 1.0
    with np.errstate(invalid="ignore"):
        larger = result.valid & (deviation > EQUALITY_RELATIVE_TOLERANCE)
        smaller = result.valid & (deviation < -EQUALITY_RELATIVE_TOLERANCE)
    labels[result.valid] = "Equal within tolerance"
    labels[larger] = comparison_label
    labels[smaller] = reference_label
    return labels
