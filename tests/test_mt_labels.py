"""ENDF-6 MT code labels.

The expected strings are byte-for-byte the TypeScript's, including the em
dash (U+2014) separator and the Greek letters, so the exported CSV and the
dataset dropdown read identically after the migration.
"""

import unittest

from backend.services import mt_labels


class FormatTests(unittest.TestCase):
    def test_exact_neutron_specific_labels(self):
        self.assertEqual(mt_labels.format_mt_label(1), "MT 1 — Total")
        self.assertEqual(mt_labels.format_mt_label(2), "MT 2 — Elastic")
        self.assertEqual(mt_labels.format_mt_label(16), "MT 16 — (n,2n)")
        self.assertEqual(
            mt_labels.format_mt_label(18), "MT 18 — (n,f) Fission"
        )
        self.assertEqual(
            mt_labels.format_mt_label(102),
            "MT 102 — (n,γ) Radiative capture",
        )
        self.assertEqual(mt_labels.format_mt_label(103), "MT 103 — (n,p)")
        self.assertEqual(mt_labels.format_mt_label(107), "MT 107 — (n,α)")

    def test_separator_is_an_em_dash(self):
        self.assertIn("—", mt_labels.format_mt_label(1))

    def test_falls_back_without_inventing_a_name(self):
        self.assertEqual(mt_labels.format_mt_label(999), "MT 999")
        self.assertEqual(
            mt_labels.format_mt_label(None), "Unknown reaction"
        )

    def test_parse_extracts_the_numeric_code(self):
        self.assertEqual(mt_labels.parse_mt_from_dataset("MT102"), 102)
        self.assertEqual(mt_labels.parse_mt_from_dataset("mt16"), 16)
        self.assertIsNone(mt_labels.parse_mt_from_dataset("not-an-mt"))

    def test_dataset_option_label_combines_both(self):
        self.assertEqual(
            mt_labels.dataset_option_label("MT102"),
            "MT 102 — (n,γ) Radiative capture",
        )

    def test_every_desired_dataset_has_a_label(self):
        from backend import jar_runner

        for dataset in jar_runner.DESIRED_DATASETS:
            mt = mt_labels.parse_mt_from_dataset(dataset)
            self.assertIsNotNone(mt, dataset)
            self.assertIn(mt, mt_labels.MT_REACTIONS, dataset)


if __name__ == "__main__":
    unittest.main()
