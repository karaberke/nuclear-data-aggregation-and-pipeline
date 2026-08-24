"""CSV generation for the two export buttons.

Byte-compatible with the former `app.ts` implementation, which matters
because `tests/test_exports.py` diffs against fixtures captured from it.

The subtle part is `format_number`. V8's `String(number)` and CPython's
`repr(float)` both produce shortest round-trip digits, so they agree on
*which* digits - but they disagree on presentation:

    value      JavaScript      Python repr
    2.0        "2"             "2.0"
    1e-5       "0.00001"       "1e-05"
    1e-7       "1e-7"          "1e-07"
    1e21       "1e+21"         "1e+21"

Energies here are in MeV and routinely reach 1e-11, so this is not a corner
case - it is most of the energy column. `format_number` reimplements
ECMA-262 Number::toString so the exported bytes match.
"""

import io
import math
import re

CSV_QUOTE_PATTERN = re.compile(r'[",\n]')

FILTERED_HEADER = [
    "database",
    "isotope",
    "dataset",
    "energy_MeV",
    "cross_section_barns",
    "cross_section_stddev_barns",
]

LINE_HEADER = [
    "database",
    "isotope",
    "dataset",
    "series",
    "energy_MeV",
    "cross_section_barns",
    "cross_section_stddev_barns",
]

BAR_HEADER = [
    "database",
    "isotope",
    "dataset",
    "series",
    "group_index",
    "mt",
    "mt_label",
    "bin_start",
    "bin_end",
    "bin_center",
    "bin_width",
    "point_count",
    "coverage_percent",
    "cross_section_mean",
    "cross_section_group_average",
    "cross_section_integral",
    "group_structure_name",
    "rebinning_method",
]

KDE_HEADER = [
    "database",
    "isotope",
    "dataset",
    "series",
    "cross_section_barns",
    "density",
]

COMPARISON_HEADER = [
    "energy_MeV",
    "reference_series",
    "comparison_series",
    "reference_cross_section_b",
    "comparison_cross_section_b",
    "ratio",
    "percent_difference",
    "dominant_series",
    "valid",
    "invalid_reason",
    "is_crossing",
]

REBINNING_METHOD = "pointwise integration"


def format_number(value) -> str:
    """Render a float the way JavaScript's `String(number)` would.

    Implements the ECMA-262 Number::toString algorithm: find the shortest
    digit string `s` (k digits) and exponent `n` with `s * 10**(n-k) == m`,
    then choose fixed or exponential notation from `n`.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)

    x = float(value)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    if x == 0:
        # JavaScript renders both 0 and -0 as "0".
        return "0"

    sign = "-" if x < 0 else ""
    text = repr(abs(x))

    if "e" in text or "E" in text:
        mantissa, _, exponent_text = text.lower().partition("e")
        exponent = int(exponent_text)
    else:
        mantissa, exponent = text, 0

    integer_part, _, fraction_part = mantissa.partition(".")
    digits = int(integer_part + fraction_part)
    power = exponent - len(fraction_part)

    # Minimize k by removing trailing zeros, as the spec requires.
    while digits % 10 == 0 and digits >= 10:
        digits //= 10
        power += 1

    s = str(digits)
    k = len(s)
    n = power + k

    if k <= n <= 21:
        return sign + s + "0" * (n - k)
    if 0 < n <= 21:
        return sign + s[:n] + "." + s[n:]
    if -6 < n <= 0:
        return sign + "0." + "0" * (-n) + s
    # Exponential notation.
    exponent_sign = "+" if n - 1 >= 0 else "-"
    suffix = f"e{exponent_sign}{abs(n - 1)}"
    if k == 1:
        return sign + s + suffix
    return sign + s[0] + "." + s[1:] + suffix


def csv_cell(value) -> str:
    """Quote a single field exactly as the TypeScript `csvCell` did."""
    if value is None:
        return ""
    text = value if isinstance(value, str) else format_number(value)
    if CSV_QUOTE_PATTERN.search(text):
        escaped = text.replace('"', '""')
        return f'"{escaped}"'
    return text


def write_csv(header: list[str], rows) -> str:
    """Join rows with the TypeScript's exact framing.

    Hand-rolled rather than using `csv.writer`, which also quotes '\\r' and
    emits CRLF line endings by default - both would break the byte diff.
    """
    buffer = io.StringIO()
    buffer.write(",".join(header))
    for row in rows:
        buffer.write("\n")
        buffer.write(",".join(csv_cell(cell) for cell in row))
    return buffer.getvalue()
