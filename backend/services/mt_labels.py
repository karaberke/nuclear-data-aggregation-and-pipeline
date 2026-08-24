"""ENDF-6 Appendix B reaction definitions for the MT codes this app queries.

Maintained by hand and stored locally - never scraped at runtime. Ported
from the TypeScript SPA's `mtReactions.ts` (removed at cutover); the emitted
strings are byte-identical to it, including the em dash (U+2014) separator
and the Greek letters in the MT 102 and MT 107 labels.

The keys mirror `jar_runner.DESIRED_DATASETS`.
"""

import re
from typing import NamedTuple


class MtReactionInfo(NamedTuple):
    endf_notation: str
    """Exact ENDF-6 Appendix B notation (z = generic incident particle)."""
    label: str
    """Exact ENDF-6 Appendix B reaction name."""
    display_label: str
    """Neutron-specific short form used in the visible "MT <n> - ..." label."""


MT_REACTIONS: dict[int, MtReactionInfo] = {
    1: MtReactionInfo(
        "(n,total)", "Neutron total cross sections", "Total"
    ),
    2: MtReactionInfo(
        "(z,z0)",
        "Elastic scattering cross section for incident particles",
        "Elastic",
    ),
    16: MtReactionInfo(
        "(z,2n)", "Production of two neutrons and a residual", "(n,2n)"
    ),
    18: MtReactionInfo(
        "(z,fission)", "Particle-induced fission", "(n,f) Fission"
    ),
    102: MtReactionInfo(
        "(z,γ)", "Radiative capture", "(n,γ) Radiative capture"
    ),
    103: MtReactionInfo(
        "(z,p)", "Production of a proton, plus a residual", "(n,p)"
    ),
    107: MtReactionInfo(
        "(z,α)",
        "Production of an alpha particle, plus a residual",
        "(n,α)",
    ),
}

_DATASET_PATTERN = re.compile(r"^MT(\d+)$", re.IGNORECASE)

UNKNOWN_REACTION = "Unknown reaction"


def parse_mt_from_dataset(dataset: str) -> int | None:
    """Extract the numeric MT code from a dataset string like "MT102"."""
    match = _DATASET_PATTERN.match(dataset.strip())
    return int(match.group(1)) if match else None


def format_mt_label(mt: int | None) -> str:
    """Format "MT <n> — <reaction>", falling back to a bare code."""
    if mt is None:
        return UNKNOWN_REACTION
    info = MT_REACTIONS.get(mt)
    # U+2014 EM DASH, matching the TypeScript exactly.
    return f"MT {mt} — {info.display_label}" if info else f"MT {mt}"


def dataset_option_label(dataset: str) -> str:
    """Human-readable dropdown text for a raw dataset value."""
    return format_mt_label(parse_mt_from_dataset(dataset))
