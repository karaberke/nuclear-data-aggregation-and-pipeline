import re
import subprocess
import threading
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from .deployment import max_concurrent_queries

JAR_DIR = Path(__file__).parent / "Janis_all_jars"
JAR_PATH = str(JAR_DIR / "Janis.jar")

# EXFOR is deliberately absent: its dataset level is experimental entry ids
# ("X4(,2N)47-AG-106"), not MT numbers, so it intersects to nothing against any
# evaluated library and defeats the MT labelling the UI is built around.
DESIRED_DATABASES = [
    "BROND-3.1", "CENDL-3.2", "ENDF/B-VII.1", "ENDF/B-VIII.0",
    "FENDL-3.1b", "IRDFF-II", "JEFF-4.0", "EAF-2010",
    "JENDL-4.0", "JENDL-4.0u", "TENDL-2019", "RUSFOND-2010",
]
DESIRED_DATASETS = ["MT1", "MT2", "MT16", "MT18", "MT102", "MT103", "MT107"]
REACTION_TYPES = ["xs", "xs_stddev"]

# Isotope listings carry the odd raw material id ("MAT9437") alongside real
# targets. Only those are dropped - `LiNat`, `CNat`, `SiNat` and `n` are
# genuine targets, so a stricter "looks like a nuclide" pattern would delete
# real options.
_MATERIAL_ID_PATTERN = re.compile(r"^MAT\d+$")

# Dataset listings mix MT reaction nodes with pseudo-nodes ("infos",
# "resonances"), which carry no cross-section table.
_MT_PATTERN = re.compile(r"^MT\d+$")

JANIS_TIMEOUT_SECONDS = 60


# Sized from the one operator setting, same as the query_store admission
# gate: a query runs its series sequentially, so N concurrent queries means
# at most N subprocesses. See deployment.max_concurrent_queries().
JANIS_MAX_CONCURRENCY = max_concurrent_queries()
_JANIS_SEMAPHORE = threading.BoundedSemaphore(JANIS_MAX_CONCURRENCY)


def run_janis(args: list[str]) -> str:
    """Run one JANIS command while enforcing the process concurrency limit."""
    try:
        with _JANIS_SEMAPHORE:
            result = subprocess.run(
                ["java", "-jar", JAR_PATH] + args,
                capture_output=True,
                text=True,
                timeout=JANIS_TIMEOUT_SECONDS,
                # The manifest Class-Path is resolved relative to this directory.
                cwd=str(JAR_DIR),
            )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"janis timed out after {JANIS_TIMEOUT_SECONDS} seconds"
        ) from error

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"janis error: {detail}")

    return result.stdout


def _list(args: list[str]) -> list[str]:
    """Return every child node JANIS reports for one tree level.

    JANIS writes its "JANIS 4.1 - Java ..." banner to *stderr*, and
    `run_janis` returns stdout, so every stdout row is data. Nothing may be
    sliced off the front here: doing so silently dropped the first three real
    entries at every level (BROND-3.1 from the database list, H1/H2/H3 from
    isotopes, MT1 from FENDL-3.1b's datasets).

    Levels that mix pseudo-nodes in with real ones ("infos", "resonances",
    "MAT9437") filter them by shape in the caller.
    """
    lines = run_janis(["-list", "NEA", "N"] + args).strip().splitlines()
    return [line.strip() for line in lines if line.strip()]


def intersection_in_first_order(groups: Iterable[list[str]]) -> list[str]:
    """Intersect string lists while retaining the first list's display order."""
    iterator = iter(groups)
    try:
        first = next(iterator)
    except StopIteration:
        return []

    common = set(first)
    for group in iterator:
        common.intersection_update(group)
        if not common:
            return []

    return [value for value in first if value in common]


@lru_cache(maxsize=1)
def list_databases() -> list[str]:
    # The allowlist is this level's filter - no shape check needed.
    return [name for name in _list([]) if name in DESIRED_DATABASES]


@lru_cache(maxsize=64)
def list_isotopes(database: str, field: str = "SIG") -> list[str]:
    return [
        name
        for name in _list([database, field])
        if not _MATERIAL_ID_PATTERN.match(name)
    ]


@lru_cache(maxsize=256)
def list_all_datasets(
    database: str, isotope: str, field: str = "SIG"
) -> list[str]:
    return [
        name
        for name in _list([database, field, isotope])
        if _MT_PATTERN.match(name)
    ]


@lru_cache(maxsize=512)
def list_quantities(
    database: str, isotope: str, dataset: str, field: str = "SIG"
) -> list[str]:
    """Return the quantities JANIS can table at one reaction node.

    A dataset appearing in `list_all_datasets` only means the MT *node*
    exists; it does not mean that node carries a cross section. Activation
    libraries such as EAF-2010 store some reactions per residual product, so
    the node holds `activation` and no `xs` - and `-table ... xs` then fails
    with "Can't find data [Cross section]".

    The values returned here are exactly the `value` argument `get_table`
    takes ("xs", "xs_stddev", "activation", "photon_prod"), so membership is
    the authoritative availability check. Unfiltered on purpose.
    """
    return _list([database, field, isotope, dataset])


def list_common_isotopes(
    databases: list[str], field: str = "SIG"
) -> list[str]:
    """Return isotopes reported by every selected database."""
    return intersection_in_first_order(
        list_isotopes(database, field) for database in databases
    )


def list_common_datasets(
    databases: list[str], isotopes: list[str], field: str = "SIG"
) -> list[str]:
    """Return datasets shared by every selected database/isotope pair."""
    available = intersection_in_first_order(
        list_all_datasets(database, isotope, field)
        for database in databases
        for isotope in isotopes
    )
    desired = [dataset for dataset in available if dataset in DESIRED_DATASETS]
    return desired or available


def get_table(
    database: str,
    isotope: str,
    dataset: str,
    value: str,
    field: str = "SIG",
) -> list[str]:
    lines = run_janis(
        ["-table", "NEA", "N", database, field, isotope, dataset, value]
    )
    return lines.strip().splitlines()
