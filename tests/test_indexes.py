import unittest

import numpy as np

from vectordb.core import VectorItem, cosine, euclidean, manhattan
from vectordb.indexes import BruteForceIndex, HNSWIndex, KDTreeIndex


def make_items(n: int, dims: int, seed: int = 0) -> list[VectorItem]:
    rng = np.random.default_rng(seed)
    return [
        VectorItem(i, f"item-{i}", "t", rng.random(dims).astype(np.float32))
        for i in range(1, n + 1)
    ]


class MetricsTest(unittest.TestCase):
    def test_known_values(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        self.assertAlmostEqual(euclidean(a, b), 2**0.5, places=5)
        self.assertAlmostEqual(manhattan(a, b), 2.0, places=5)
        self.assertAlmostEqual(cosine(a, b), 1.0, places=5)
        self.assertAlmostEqual(cosine(a, a), 0.0, places=5)

    def test_cosine_zero_vector(self):
        zero = np.zeros(3, dtype=np.float32)
        self.assertEqual(cosine(zero, np.ones(3, dtype=np.float32)), 1.0)


class IndexTest(unittest.TestCase):
    def setUp(self):
        self.items = make_items(300, 8)
        self.query = np.random.default_rng(99).random(8).astype(np.float32)
        self.brute = BruteForceIndex()
        self.kd = KDTreeIndex(8)
        self.hnsw = HNSWIndex(m=16, ef_construction=100)
        for item in self.items:
            self.brute.insert(item)
            self.kd.insert(item)
            self.hnsw.insert(item)

    def test_kdtree_matches_bruteforce_euclidean(self):
        expected = self.brute.knn(self.query, 5, euclidean)
        self.assertEqual(self.kd.knn(self.query, 5, euclidean), expected)

    def test_hnsw_recall(self):
        rng = np.random.default_rng(7)
        hits = total = 0
        for _ in range(20):
            q = rng.random(8).astype(np.float32)
            truth = {i for _, i in self.brute.knn(q, 10, cosine)}
            found = {i for _, i in self.hnsw.knn(q, 10, cosine, ef=100)}
            hits += len(truth & found)
            total += len(truth)
        self.assertGreater(hits / total, 0.9)

    def test_results_sorted_and_limited(self):
        for index in (self.brute, self.kd, self.hnsw):
            res = index.knn(self.query, 5, euclidean)
            self.assertEqual(len(res), 5)
            self.assertEqual(res, sorted(res))

    def test_remove(self):
        victim = self.brute.knn(self.query, 1, euclidean)[0][1]
        for index in (self.brute, self.kd, self.hnsw):
            index.remove(victim)
            self.assertNotIn(victim, [i for _, i in index.knn(self.query, 10, euclidean)])
            self.assertEqual(len(index), 299)

    def test_hnsw_remove_everything(self):
        for item in self.items:
            self.hnsw.remove(item.id)
        self.assertEqual(len(self.hnsw), 0)
        self.assertEqual(self.hnsw.knn(self.query, 3, cosine), [])

    def test_hnsw_entry_point_survives_removal(self):
        info = self.hnsw.info()
        top = [n["id"] for n in info.nodes if n["maxLyr"] == info.top_layer][0]
        self.hnsw.remove(top)
        self.assertEqual(len(self.hnsw.knn(self.query, 5, cosine)), 5)

    def test_empty_index(self):
        self.assertEqual(HNSWIndex().knn(self.query, 3, cosine), [])
        self.assertEqual(KDTreeIndex(8).knn(self.query, 3, euclidean), [])


if __name__ == "__main__":
    unittest.main()
