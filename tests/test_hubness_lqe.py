import unittest

import numpy as np

from clustercontrast.utils.faiss_rerank_xmodal import (
    compute_hubness,
    hubness_weighted_mean,
)


class HubnessLQETest(unittest.TestCase):
    def test_hubness_statistics_detect_concentrated_neighbor(self):
        rank = np.array([[0, 1], [0, 2], [0, 3], [0, 4]])
        occurrences, skewness, top_mass = compute_hubness(rank, 5, 2)

        self.assertEqual(occurrences.tolist(), [4, 1, 1, 1, 1])
        self.assertGreater(skewness, 1.0)
        self.assertEqual(top_mass, 0.5)

    def test_weighting_reduces_hub_contribution(self):
        values = np.array([[1.0, 0.0], [0.0, 1.0]])
        occurrences = np.array([9.0, 1.0])
        weighted = hubness_weighted_mean(values, [0, 1], occurrences)

        self.assertLess(weighted[0], 0.5)
        self.assertGreater(weighted[1], 0.5)


if __name__ == "__main__":
    unittest.main()
