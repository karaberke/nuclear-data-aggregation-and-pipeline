"""Bundled multigroup structures loaded from JSON."""

import unittest

import numpy as np

from backend.services import multigroup

# name -> edge count. N groups requires N+1 edges.
EXPECTED_EDGE_COUNTS = {
    "WIMS69": 70,
    "14MeV129": 130,
    "STAYSL140": 141,
    "VITAMIN-J175": 176,
    "SAND-II640": 641,
    "SAND-II725": 726,
}


class StructureTests(unittest.TestCase):
    def test_every_bundled_structure_has_the_verified_counts(self):
        self.assertEqual(
            len(multigroup.structure_names()), len(EXPECTED_EDGE_COUNTS)
        )
        for name, edge_count in EXPECTED_EDGE_COUNTS.items():
            with self.subTest(name=name):
                edges = multigroup.structure_edges(name)
                self.assertEqual(edges.size, edge_count)
                # N groups requires N+1 edges.
                self.assertEqual(edges.size - 1, edge_count - 1)

    def test_structure_names_encode_their_group_count(self):
        """WIMS69 really has 69 groups, VITAMIN-J175 really has 175, etc."""
        import re

        for name in multigroup.structure_names():
            with self.subTest(name=name):
                match = re.search(r"(\d+)$", name)
                self.assertIsNotNone(match, f"{name} has no trailing count")
                groups = multigroup.structure_edges(name).size - 1
                self.assertEqual(int(match.group(1)), groups)

    def test_edges_are_positive_finite_and_strictly_ascending(self):
        for name in multigroup.structure_names():
            with self.subTest(name=name):
                edges = multigroup.structure_edges(name)
                self.assertTrue(np.all(np.isfinite(edges)))
                self.assertTrue(np.all(edges > 0))
                self.assertTrue(np.all(np.diff(edges) > 0))

    def test_vitamin_j175_upper_bound_confirms_mev_units(self):
        edges = multigroup.structure_edges("VITAMIN-J175")
        self.assertAlmostEqual(float(edges[-1]), 19.64, places=2)

    def test_options_are_labelled_with_group_counts(self):
        options = multigroup.structure_options()
        labels = {option["value"]: option["label"] for option in options}
        self.assertEqual(labels["VITAMIN-J175"], "VITAMIN-J175 (175 groups)")
        self.assertEqual(labels["SAND-II725"], "SAND-II725 (725 groups)")

    def test_unknown_structure_raises(self):
        with self.assertRaises(multigroup.UnknownStructureError):
            multigroup.structure_edges("NOPE")

    def test_structures_are_cached_not_reparsed(self):
        self.assertIs(
            multigroup.structure_edges("WIMS69"),
            multigroup.structure_edges("WIMS69"),
        )


class NormalizationTests(unittest.TestCase):
    def test_converts_units_and_normalizes_direction(self):
        self.assertEqual(
            multigroup.normalize_structure(
                {"name": "x", "unit": "MeV", "edges": [1, 2, 3]}
            ),
            [1, 2, 3],
        )
        self.assertEqual(
            multigroup.normalize_structure(
                {"name": "x", "unit": "MeV", "edges": [3, 2, 1]}
            ),
            [1, 2, 3],
        )
        self.assertEqual(
            multigroup.normalize_structure(
                {"name": "x", "unit": "eV", "edges": [1e6, 2e6]}
            ),
            [1, 2],
        )
        self.assertEqual(
            multigroup.normalize_structure(
                {"name": "x", "unit": "keV", "edges": [1000, 2000]}
            ),
            [1, 2],
        )


if __name__ == "__main__":
    unittest.main()
