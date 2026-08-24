"""Shared fixtures, mirroring the four the TypeScript suite used."""

import numpy as np

from backend.services import analytics


def make_series(key: str, points: list[tuple[float, float]]):
    database, isotope, dataset = key.split("|")
    return analytics.series_from_points(
        key,
        database,
        isotope,
        dataset,
        [
            {"energy_MeV": energy, "cross_section_barns": value}
            for energy, value in points
        ],
    )


def simple_series():
    return [make_series("A|I|D", [(1, 2), (2, 4), (3, 8), (4, 16)])]


def line_series():
    """A straight line y = x over [0, 10].

    Its integral over any sub-interval has an easy closed form:
    integral from a to b of x dx == (b**2 - a**2) / 2.
    """
    return [make_series("L|I|D", [(0, 0), (10, 10)])]


def boundary_series():
    """A peak so the segments meeting at energy=5 have different slopes.

    Lets a boundary-exactly-on-a-point test tell a correct shared bracket
    apart from an accidental match.
    """
    return [make_series("B|I|D", [(0, 0), (5, 50), (10, 10)])]


def sparse_gap_series():
    """A wide data gap between energy 1 and 10.

    Exercises several consecutive groups that have interpolation coverage
    but no landed sample.
    """
    return [make_series("S|I|D", [(0, 2), (1, 4), (10, 6), (11, 8)])]


def performance_series(point_count: int = 30_000):
    index = np.arange(point_count, dtype=np.float64)
    energy = 1e-5 * (60 / 1e-5) ** (index / (point_count - 1))
    sigma = 1 + np.sin(index / 37) * 0.5
    return [
        analytics.SeriesArrays(
            key="P|I|D",
            database="P",
            isotope="I",
            dataset="D",
            energy=energy,
            sigma=sigma,
        )
    ]
