import unittest

from clustercontrast.utils.faiss_rerank_xmodal import select_local_boundary


class QueryLocalBoundaryTest(unittest.TestCase):
    def test_clear_boundary_is_selected(self):
        similarities = [1.0, 0.9, 0.8, 0.7, 0.2, 0.19, 0.18]
        self.assertEqual(select_local_boundary(similarities), 4)

    def test_no_significant_boundary_falls_back(self):
        similarities = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        self.assertEqual(select_local_boundary(similarities), 6)

    def test_short_input_is_rejected(self):
        with self.assertRaises(ValueError):
            select_local_boundary([1.0, 0.9, 0.8])


if __name__ == "__main__":
    unittest.main()
