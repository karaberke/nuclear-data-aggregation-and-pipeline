"""Cross-section parsing, batching, caching, and the batch query route."""

import math
from functools import lru_cache
from itertools import product
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

from . import jar_runner
from .deployment import env_int
from .services.errors import (
    DataUnavailableError,
    DomainError,
    SelectionUnavailableError,
)
from .services.mt_labels import dataset_option_label

router = APIRouter()


SERIES_CACHE_SIZE = env_int("SERIES_CACHE_SIZE", 16, minimum=1)
MAX_QUERY_SERIES = 5

ParsedTable = tuple[tuple[float, float], ...]


class CrossSectionQuery(BaseModel):
    """Validated selection matrix for a single or comparison query."""

    databases: list[str] = Field(min_length=1, max_length=MAX_QUERY_SERIES)
    isotopes: list[str] = Field(min_length=1, max_length=MAX_QUERY_SERIES)
    datasets: list[str] = Field(min_length=1, max_length=MAX_QUERY_SERIES)
    field: Literal["SIG"] = "SIG"
    reaction_type: Literal["xs", "xs_stddev"] = "xs"

    @field_validator("databases", "isotopes", "datasets")
    @classmethod
    def values_must_be_unique_and_nonempty(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("selection values cannot be blank")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("selection values must be unique")
        return cleaned

    @model_validator(mode="after")
    def validate_comparison_shape(self) -> "CrossSectionQuery":
        lengths = (
            len(self.databases),
            len(self.isotopes),
            len(self.datasets),
        )
        if sum(length > 1 for length in lengths) > 1:
            raise ValueError("only one selection dimension may contain multiple values")
        if self.series_count > MAX_QUERY_SERIES:
            raise ValueError(
                f"a query may contain at most {MAX_QUERY_SERIES} series"
            )
        return self

    @property
    def series_count(self) -> int:
        return len(self.databases) * len(self.isotopes) * len(self.datasets)


def parse_table_lines(lines: list[str]) -> ParsedTable:
    """Parse numeric JANIS table rows and convert energy (eV) to MeV.

    JANIS reports two-column `energy ; value` rows for a single quantity
    (e.g. "xs") but three-column `energy ; xs ; stddev` rows for a paired
    quantity (e.g. "xs_stddev", which repeats the central value ahead of
    the actual uncertainty) - the requested value is always the last field.
    """
    records: list[tuple[float, float]] = []

    for line in lines:
        fields = line.split(";")
        if len(fields) < 2:
            continue
        try:
            energy_ev = float(fields[0].strip())
            value = float(fields[-1].strip())
        except ValueError:
            continue
        if not math.isfinite(energy_ev) or not math.isfinite(value):
            continue
        records.append((energy_ev / 1_000_000, value))

    return tuple(records)


@lru_cache(maxsize=SERIES_CACHE_SIZE)
def get_parsed_table(
    database: str,
    isotope: str,
    dataset: str,
    value: str,
    field: str = "SIG",
) -> ParsedTable:
    """Load and cache one parsed JANIS table."""
    return parse_table_lines(
        jar_runner.get_table(database, isotope, dataset, value, field)
    )


def build_records(
    database: str,
    isotope: str,
    dataset: str,
    field: str,
    include_stddev: bool,
) -> list[dict[str, float | None]]:
    """Build normalized points for one database/isotope/dataset series."""
    cross_sections = get_parsed_table(
        database, isotope, dataset, "xs", field
    )
    if not cross_sections:
        raise DataUnavailableError(
            f"No cross-section data for {database}/{isotope}/{dataset}"
        )

    standard_deviations: dict[float, float] = {}
    if include_stddev:
        standard_deviations = dict(
            get_parsed_table(
                database, isotope, dataset, "xs_stddev", field
            )
        )

    return [
        {
            "energy_MeV": energy,
            "cross_section_barns": cross_section,
            **(
                {
                    "cross_section_stddev_barns":
                        standard_deviations.get(energy)
                }
                if include_stddev
                else {}
            ),
        }
        for energy, cross_section in cross_sections
    ]


def _validate_available_selections(query: CrossSectionQuery) -> None:
    """Reject selections that are not reported by JANIS metadata endpoints."""
    available_databases = set(jar_runner.list_databases())
    unknown_databases = set(query.databases) - available_databases
    if unknown_databases:
        raise SelectionUnavailableError(
            f"Unknown database(s): {', '.join(sorted(unknown_databases))}"
        )

    for database in query.databases:
        available_isotopes = set(
            jar_runner.list_isotopes(database, query.field)
        )
        unknown_isotopes = set(query.isotopes) - available_isotopes
        if unknown_isotopes:
            raise SelectionUnavailableError(
                f"Isotope(s) not available in {database}: "
                f"{', '.join(sorted(unknown_isotopes))}"
            )

        for isotope in query.isotopes:
            available_datasets = set(
                jar_runner.list_all_datasets(database, isotope, query.field)
            )
            unknown_datasets = set(query.datasets) - available_datasets
            if unknown_datasets:
                raise SelectionUnavailableError(
                    f"Dataset(s) not available for {database}/{isotope}: "
                    f"{', '.join(sorted(unknown_datasets))}"
                )

            for dataset in query.datasets:
                _validate_quantity(query, database, isotope, dataset)


def _validate_quantity(
    query: CrossSectionQuery, database: str, isotope: str, dataset: str
) -> None:
    """Reject a reaction node that carries no table for the wanted quantity.

    The dataset appearing in `list_all_datasets` only proves the MT node
    exists. Activation libraries publish some reactions solely per residual
    product, leaving the node with no cross section at all, and `-table` then
    fails deep inside a JANIS subprocess with a raw Java exception. Checking
    the node's quantities first turns that into an actionable message, and
    costs one extra (memoized) JANIS listing per series.
    """
    required = "xs_stddev" if query.reaction_type == "xs_stddev" else "xs"
    quantities = jar_runner.list_quantities(
        database, isotope, dataset, query.field
    )
    if required in quantities:
        return

    label = dataset_option_label(dataset)
    wanted = (
        "standard-deviation data"
        if required == "xs_stddev"
        else "cross-section data"
    )
    offered = (
        f"JANIS offers only: {', '.join(quantities)}."
        if quantities
        else "JANIS reports no data at all for that reaction."
    )
    raise SelectionUnavailableError(
        f"{database} has no {wanted} for {isotope} / {label}. {offered} "
        f"Choose a different dataset, or drop {database} from the comparison."
    )


def build_series(query: CrossSectionQuery) -> list[dict]:
    """Build every series in a validated comparison query."""
    _validate_available_selections(query)
    include_stddev = query.reaction_type == "xs_stddev"
    series: list[dict] = []

    for database, isotope, dataset in product(
        query.databases, query.isotopes, query.datasets
    ):
        series.append(
            {
                "key": f"{database}|{isotope}|{dataset}",
                "database": database,
                "isotope": isotope,
                "dataset": dataset,
                "points": build_records(
                    database,
                    isotope,
                    dataset,
                    query.field,
                    include_stddev,
                ),
            }
        )

    return series


@router.post("/api/cross-sections/query")
def query_cross_sections(query: CrossSectionQuery):
    """Return normalized points grouped into comparison series."""
    try:
        return {"series": build_series(query)}
    except DomainError as error:
        raise HTTPException(
            status_code=error.status_code, detail=str(error)
        ) from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
