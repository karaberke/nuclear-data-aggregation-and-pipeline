import threading
import time
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

from backend import charts, jar_runner
from backend.main import get_datasets
from backend.services import query_store
from backend.services.errors import JanisError, SelectionUnavailableError


class IntersectionTests(unittest.TestCase):
    def test_intersection_retains_first_list_order(self):
        result = jar_runner.intersection_in_first_order(
            [["Co59", "H1", "U235"], ["U235", "Co59"], ["Co59", "U235"]]
        )
        self.assertEqual(result, ["Co59", "U235"])

    @patch("backend.jar_runner.list_isotopes")
    def test_common_isotopes_uses_every_database(self, list_isotopes):
        list_isotopes.side_effect = [
            ["Co59", "U235", "H1"],
            ["H1", "Co59"],
        ]
        self.assertEqual(
            jar_runner.list_common_isotopes(["A", "B"]),
            ["Co59", "H1"],
        )

    @patch("backend.jar_runner.list_all_datasets")
    def test_common_datasets_prioritizes_configured_mt_values(
        self, list_all_datasets
    ):
        list_all_datasets.side_effect = [
            ["OTHER", "MT102", "MT1"],
            ["MT1", "OTHER", "MT102"],
        ]
        self.assertEqual(
            jar_runner.list_common_datasets(["A", "B"], ["Co59"]),
            ["MT102", "MT1"],
        )

    def test_dataset_intersection_rejects_two_multi_dimensions(self):
        with self.assertRaises(HTTPException) as context:
            get_datasets(["A", "B"], ["I", "J"])
        self.assertEqual(context.exception.status_code, 422)


class QueryValidationTests(unittest.TestCase):
    def test_single_and_one_multi_dimension_are_valid(self):
        single = charts.CrossSectionQuery(
            databases=["A"], isotopes=["I"], datasets=["D"]
        )
        comparison = charts.CrossSectionQuery(
            databases=["A", "B"], isotopes=["I"], datasets=["D"]
        )
        self.assertEqual(single.series_count, 1)
        self.assertEqual(comparison.series_count, 2)

    def test_multiple_multi_dimensions_are_rejected(self):
        with self.assertRaises(ValidationError):
            charts.CrossSectionQuery(
                databases=["A", "B"],
                isotopes=["I1", "I2"],
                datasets=["D"],
            )

    def test_duplicate_and_blank_values_are_rejected(self):
        for values in (["A", "A"], ["A", " "]):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                charts.CrossSectionQuery(
                    databases=values,
                    isotopes=["I"],
                    datasets=["D"],
                )

    def test_more_than_five_series_are_rejected(self):
        with self.assertRaises(ValidationError):
            charts.CrossSectionQuery(
                databases=["A", "B", "C", "D", "E", "F"],
                isotopes=["I"],
                datasets=["D"],
            )


class ParsingAndRecordsTests(unittest.TestCase):
    def setUp(self):
        charts.get_parsed_table.cache_clear()

    def tearDown(self):
        charts.get_parsed_table.cache_clear()

    def test_parser_skips_headers_and_converts_ev_to_mev(self):
        parsed = charts.parse_table_lines(
            [
                "JANIS banner",
                "Incident energy ; Cross section",
                "1000000 ; 2.5",
                "2500000; 3.75",
                "not numeric ; ignored",
                "NaN ; 1",
                "100 ; inf",
            ]
        )
        self.assertEqual(parsed, ((1.0, 2.5), (2.5, 3.75)))

    def test_parser_extracts_last_field_from_paired_quantity_rows(self):
        # JANIS reports "xs_stddev" as three columns - energy, the central
        # value repeated, then the actual uncertainty - unlike the plain
        # two-column "energy ; value" rows "xs" reports.
        parsed = charts.parse_table_lines(
            [
                "JANIS 4.1 - Java 26.0.1",
                "Incident energy ;  ; ",
                "Incident energy ; σ(E) ; Δσ(E)",
                "1000000 ; 2.5 ; 0.2",
                "2000000 ; 4.0 ; 0.5",
            ]
        )
        self.assertEqual(parsed, ((1.0, 0.2), (2.0, 0.5)))

    @patch("backend.charts.jar_runner.get_table")
    def test_missing_standard_deviation_is_nullable(self, get_table):
        get_table.side_effect = [
            ["1000000 ; 2.5", "2000000 ; 4.0"],
            ["1000000 ; 2.5 ; 0.2"],
        ]
        records = charts.build_records("DB", "I", "D", "SIG", True)
        self.assertEqual(records[0]["cross_section_stddev_barns"], 0.2)
        self.assertIsNone(records[1]["cross_section_stddev_barns"])

    @patch("backend.charts.jar_runner.get_table")
    def test_parsed_tables_are_reused_from_cache(self, get_table):
        get_table.return_value = ["1000000 ; 2.5"]
        first = charts.build_records("DB", "I", "D", "SIG", False)
        second = charts.build_records("DB", "I", "D", "SIG", False)
        self.assertEqual(first, second)
        get_table.assert_called_once()

    @patch("backend.charts.jar_runner.list_quantities")
    @patch("backend.charts.jar_runner.list_all_datasets")
    @patch("backend.charts.jar_runner.list_isotopes")
    @patch("backend.charts.jar_runner.list_databases")
    @patch("backend.charts.build_records")
    def test_batch_response_contains_each_requested_series(
        self,
        build_records,
        list_databases,
        list_isotopes,
        list_all_datasets,
        list_quantities,
    ):
        list_databases.return_value = ["A", "B"]
        list_isotopes.return_value = ["I"]
        list_all_datasets.return_value = ["D"]
        list_quantities.return_value = ["xs"]
        build_records.return_value = [
            {"energy_MeV": 1.0, "cross_section_barns": 2.0}
        ]
        query = charts.CrossSectionQuery(
            databases=["A", "B"],
            isotopes=["I"],
            datasets=["D"],
        )
        result = charts.build_series(query)
        self.assertEqual([item["key"] for item in result], ["A|I|D", "B|I|D"])
        self.assertEqual(build_records.call_count, 2)


class JanisConcurrencyTests(unittest.TestCase):
    @patch("backend.jar_runner.subprocess.run")
    def test_semaphore_caps_subprocesses_at_the_configured_limit(
        self, subprocess_run
    ):
        """Holds at any setting, so it does not skip when the default moves.

        The previous version asserted `maximum_active == 1` behind a
        `skipUnless(JANIS_MAX_CONCURRENCY == 1)`, which silently stopped
        covering the semaphore the moment the default became 2.
        """
        active = 0
        maximum_active = 0
        guard = threading.Lock()

        class Result:
            returncode = 0
            stdout = "ok"
            stderr = ""

        def fake_run(*args, **kwargs):
            nonlocal active, maximum_active
            with guard:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.02)
            with guard:
                active -= 1
            return Result()

        limit = jar_runner.JANIS_MAX_CONCURRENCY
        subprocess_run.side_effect = fake_run
        # More threads than slots, so the gate is genuinely contended and
        # `maximum_active` reaching `limit` is meaningful rather than a
        # side effect of never having enough work in flight.
        threads = [
            threading.Thread(target=jar_runner.run_janis, args=(["-list"],))
            for _ in range(limit + 2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertLessEqual(maximum_active, limit)
        self.assertEqual(maximum_active, limit)


class ListingTests(unittest.TestCase):
    """Every stdout row from `-list` is data.

    JANIS prints its "JANIS 4.1 - Java ..." banner to stderr and `run_janis`
    returns stdout, so there is no header to skip. These fixtures are the real
    stdout of the corresponding live calls; an earlier `[3:]` slice discarded
    the first three of them at every level.
    """

    def setUp(self):
        for cached in (
            jar_runner.list_databases,
            jar_runner.list_isotopes,
            jar_runner.list_all_datasets,
            jar_runner.list_quantities,
        ):
            cached.cache_clear()

    @patch("backend.jar_runner.run_janis")
    def test_database_list_keeps_its_leading_rows(self, run_janis):
        run_janis.return_value = (
            "EXFOR\nBROND-2.2\nBROND-3.1\nCENDL-2.1\nCENDL-3.2\n"
        )
        # BROND-3.1 was previously lost to the slice; EXFOR is excluded by the
        # allowlist because its datasets are X4 entry ids, not MT numbers.
        self.assertEqual(
            jar_runner.list_databases(), ["BROND-3.1", "CENDL-3.2"]
        )

    @patch("backend.jar_runner.run_janis")
    def test_isotope_list_keeps_natural_targets_and_drops_material_ids(
        self, run_janis
    ):
        run_janis.return_value = "MAT9437\nH1\nH2\nH3\nLiNat\nn\n"
        self.assertEqual(
            jar_runner.list_isotopes("FENDL-3.1b"),
            ["H1", "H2", "H3", "LiNat", "n"],
        )

    @patch("backend.jar_runner.run_janis")
    def test_dataset_list_drops_pseudo_nodes_and_keeps_mt1(self, run_janis):
        run_janis.return_value = "infos\nresonances\nMT1\nMT2\nMT16\n"
        self.assertEqual(
            jar_runner.list_all_datasets("FENDL-3.1b", "Ag107"),
            ["MT1", "MT2", "MT16"],
        )

    @patch("backend.jar_runner.run_janis")
    def test_quantity_list_is_returned_verbatim(self, run_janis):
        """These strings are the `value` vocabulary `get_table` takes."""
        run_janis.return_value = (
            "covariances\nboxer\nxs\nxs_stddev\nphoton_prod\n"
        )
        self.assertEqual(
            jar_runner.list_quantities("TENDL-2019", "Fe56", "MT102"),
            ["covariances", "boxer", "xs", "xs_stddev", "photon_prod"],
        )


class QuantityAvailabilityTests(unittest.TestCase):
    """A listed MT node need not carry a table for the wanted quantity.

    EAF-2010 publishes Ag107 (n,2n) only as per-product activation data: the
    MT16 node exists and appears in `list_all_datasets`, but holds no cross
    section, so `-table ... xs` fails inside the subprocess.
    """

    def setUp(self):
        jar_runner.list_quantities.cache_clear()

    @staticmethod
    def _query(reaction_type="xs"):
        return charts.CrossSectionQuery(
            databases=["EAF-2010"],
            isotopes=["Ag107"],
            datasets=["MT16"],
            reaction_type=reaction_type,
        )

    def _patch_metadata(self, quantities):
        patches = [
            patch(
                "backend.charts.jar_runner.list_databases",
                return_value=["EAF-2010"],
            ),
            patch(
                "backend.charts.jar_runner.list_isotopes",
                return_value=["Ag107"],
            ),
            patch(
                "backend.charts.jar_runner.list_all_datasets",
                return_value=["MT16"],
            ),
            patch(
                "backend.charts.jar_runner.list_quantities",
                return_value=quantities,
            ),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def test_node_without_a_cross_section_is_rejected(self):
        self._patch_metadata(["activation"])
        with self.assertRaises(SelectionUnavailableError) as context:
            charts._validate_available_selections(self._query())
        message = str(context.exception)
        self.assertIn("EAF-2010", message)
        self.assertIn("Ag107", message)
        self.assertIn("MT 16", message)
        self.assertIn("activation", message)
        self.assertNotIn("BaseException", message)

    def test_node_carrying_a_cross_section_is_accepted(self):
        self._patch_metadata(["activation", "xs"])
        charts._validate_available_selections(self._query())

    def test_uncertainty_query_requires_the_stddev_quantity(self):
        self._patch_metadata(["activation", "xs"])
        with self.assertRaises(SelectionUnavailableError) as context:
            charts._validate_available_selections(self._query("xs_stddev"))
        self.assertIn("standard-deviation", str(context.exception))

    def test_uncertainty_query_passes_where_covariances_exist(self):
        self._patch_metadata(["covariances", "boxer", "xs", "xs_stddev"])
        charts._validate_available_selections(self._query("xs_stddev"))


class JanisErrorMessageTests(unittest.TestCase):
    """The raw Java exception must never reach the status line."""

    @patch("backend.services.query_store.charts.build_records")
    @patch("backend.services.query_store.charts._validate_available_selections")
    def test_missing_table_reads_as_a_data_availability_problem(
        self, _validate, build_records
    ):
        build_records.side_effect = RuntimeError(
            "janis error: org.nea.janis.database.BaseException: "
            "Can't find data [Cross section] in [MT=16 : (z,2n)]"
        )
        query = charts.CrossSectionQuery(
            databases=["EAF-2010"], isotopes=["Ag107"], datasets=["MT16"]
        )
        with self.assertRaises(JanisError) as context:
            query_store._build_series_within_budget(
                query, time.monotonic() + 60
            )
        message = str(context.exception)
        self.assertIn("EAF-2010", message)
        self.assertIn("MT16", message)
        self.assertNotIn("BaseException", message)


if __name__ == "__main__":
    unittest.main()
