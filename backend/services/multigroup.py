"""Standard JANIS/NEA multigroup energy group structures.

Verified against the required edge counts and confirmed to already be in MeV
(see the canonical VITAMIN-J175 19.6403 MeV upper bound and WIMS69's ~1e-5 eV
to 10 MeV span, both of which match reading these values directly as MeV).

Ported from the TypeScript SPA (removed at cutover; see git history). The
edge values live in `data/multigroup_structures.json`, generated from that
source, so they are the identical IEEE-754 doubles - JSON round-trips a
double exactly in both languages.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA_PATH = Path(__file__).resolve().parent / "data" / "multigroup_structures.json"

UNIT_TO_MEV = {"MeV": 1.0, "keV": 1e-3, "eV": 1e-6}


class UnknownStructureError(KeyError):
    """Raised when a preset name is not in the bundled structure set."""


def normalize_structure(structure: dict) -> list[float]:
    """Convert a structure's edges to MeV, ascending.

    JANIS group-structure files may be listed in either direction.
    Downstream validation (strictly ascending, finite, positive-if-log) is
    `analytics.bin_series`'s job once these are passed in as custom edges.
    """
    factor = UNIT_TO_MEV[structure["unit"]]
    converted = [edge * factor for edge in structure["edges"]]
    if converted and converted[0] > converted[-1]:
        converted.reverse()
    return converted


@lru_cache(maxsize=1)
def _raw_structures() -> tuple[dict, ...]:
    with DATA_PATH.open(encoding="utf-8") as handle:
        return tuple(json.load(handle)["structures"])


@lru_cache(maxsize=1)
def _normalized() -> dict[str, np.ndarray]:
    """Normalize every bundled structure once.

    Cached because SAND-II725 is 726 edges and the bar chart re-reads it on
    every slider movement.
    """
    return {
        structure["name"]: np.asarray(
            normalize_structure(structure), dtype=np.float64
        )
        for structure in _raw_structures()
    }


def structure_names() -> list[str]:
    """Bundled preset names, in their declared order."""
    return [structure["name"] for structure in _raw_structures()]


def structure_options() -> list[dict[str, str]]:
    """Dropdown options labelled with each structure's group count."""
    return [
        {
            "label": f"{name} ({len(edges) - 1} groups)",
            "value": name,
        }
        for name, edges in _normalized().items()
    ]


def structure_edges(name: str) -> np.ndarray:
    """Return one preset's ascending MeV edges.

    The returned array is the cached instance - treat it as read-only.
    `bin_series` never mutates the edges it is given.
    """
    try:
        return _normalized()[name]
    except KeyError as error:
        raise UnknownStructureError(
            f"Unknown group structure: {name}"
        ) from error
