"""Unit tests for topology membership and Topo-GBLSTSVM."""

import unittest

import numpy as np

from src.cbcm import resolve_k, split_membership_by_class
from src.class_weights import compute_binary_class_weights
from src.granular_balls import generate_granular_balls
from src.ipa import compute_ambiguity
from src.topology_gblstsvm import TopoGBLSTSVM
from src.topology_membership import (
    ball_topo_weights,
    compute_competition_fraction,
    compute_reciprocal_opposite_support,
    compute_topology_membership,
    knn_graph,
)


class TestResolveK(unittest.TestCase):
    def test_formula(self) -> None:
        self.assertEqual(resolve_k(10), 5)
        self.assertEqual(resolve_k(100), 5)
        self.assertEqual(resolve_k(101), 6)
        self.assertEqual(resolve_k(200), 10)


class TestKnnGraph(unittest.TestCase):
    def test_self_neighbor_exclusion(self) -> None:
        centers = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
        idx = knn_graph(centers, k=2)
        for i in range(3):
            self.assertNotIn(i, idx[i].tolist())

    def test_duplicate_centers(self) -> None:
        centers = np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 0.0]])
        idx = knn_graph(centers, k=1)
        self.assertEqual(idx[0, 0], 1)
        self.assertEqual(idx[1, 0], 0)
        self.assertNotEqual(idx[0, 0], 0)

    def test_tiny_n(self) -> None:
        idx1 = knn_graph(np.array([[0.0, 0.0]]), k=5)
        self.assertEqual(idx1.shape, (1, 0))
        idx2 = knn_graph(np.array([[0.0, 0.0], [1.0, 0.0]]), k=5)
        self.assertEqual(idx2.shape, (2, 1))
        self.assertEqual(idx2[0, 0], 1)
        self.assertEqual(idx2[1, 0], 0)


class TestNeighborhoods(unittest.TestCase):
    def test_all_same_class(self) -> None:
        centers = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
        labels = np.array([1.0, 1.0, 1.0])
        result = compute_topology_membership(centers, labels, k=2)
        self.assertTrue(np.allclose(result.comp, 0.0))
        self.assertTrue(np.allclose(result.ambiguity, 0.0))
        self.assertTrue(np.allclose(result.r_opp, 0.0))
        self.assertTrue(np.allclose(result.mu_topo, 0.0))

    def test_mixed_class_neighborhood(self) -> None:
        centers = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.05, 0.1],
                [5.0, 0.0],
            ]
        )
        labels = np.array([1.0, -1.0, 1.0, -1.0])
        result = compute_topology_membership(centers, labels, k=2)
        self.assertGreater(result.comp[0], 0.0)
        self.assertGreater(result.ambiguity[0], 0.0)

    def test_no_opposite_class_neighbors(self) -> None:
        centers = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.2, 0.0],
                [10.0, 0.0],
            ]
        )
        labels = np.array([1.0, 1.0, 1.0, -1.0])
        r_opp, n_opp, _, _ = compute_reciprocal_opposite_support(
            centers, labels, k=2
        )
        self.assertEqual(n_opp[0], 0)
        self.assertEqual(r_opp[0], 0.0)

    def test_reciprocal_vs_one_way(self) -> None:
        # Reciprocal pair: 0 (+) and 1 (-) are each other's nearest neighbor.
        centers_rec = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [10.0, 0.0],
                [10.1, 0.0],
            ]
        )
        labels_rec = np.array([1.0, -1.0, 1.0, -1.0])
        r_rec, n_opp_rec, n_recip_rec, _ = compute_reciprocal_opposite_support(
            centers_rec, labels_rec, k=1
        )
        self.assertEqual(n_opp_rec[0], 1)
        self.assertEqual(n_recip_rec[0], 1)
        self.assertGreater(r_rec[0], 0.99)

        # One-way: 0 (+) sees 1 (-), but 1's nearest neighbor is 2 (-) at 0.1.
        centers_ow = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.1, 0.0],
                [10.0, 0.0],
            ]
        )
        labels_ow = np.array([1.0, -1.0, -1.0, 1.0])
        r_ow, n_opp_ow, n_recip_ow, _ = compute_reciprocal_opposite_support(
            centers_ow, labels_ow, k=1
        )
        self.assertEqual(n_opp_ow[0], 1)
        self.assertEqual(n_recip_ow[0], 0)
        self.assertEqual(r_ow[0], 0.0)


class TestMembership(unittest.TestCase):
    def test_range_and_product(self) -> None:
        rng = np.random.default_rng(0)
        centers = rng.normal(size=(40, 3))
        labels = np.array([1.0] * 15 + [-1.0] * 25)
        result = compute_topology_membership(centers, labels)
        self.assertTrue(np.all(result.ambiguity >= 0.0) and np.all(result.ambiguity <= 1.0))
        self.assertTrue(np.all(result.r_opp >= 0.0) and np.all(result.r_opp <= 1.0))
        self.assertTrue(np.all(result.mu_topo >= 0.0) and np.all(result.mu_topo <= 1.0))
        self.assertTrue(np.allclose(result.mu_topo, result.ambiguity * result.r_opp))
        self.assertTrue(np.allclose(result.ambiguity, compute_ambiguity(result.comp)))

    def test_deterministic(self) -> None:
        rng = np.random.default_rng(1)
        centers = rng.normal(size=(25, 2))
        labels = np.array([1.0] * 10 + [-1.0] * 15)
        a = compute_topology_membership(centers, labels, k=5)
        b = compute_topology_membership(centers, labels, k=5)
        self.assertTrue(np.allclose(a.mu_topo, b.mu_topo))
        self.assertTrue(np.array_equal(a.neighbor_idx, b.neighbor_idx))

    def test_tiny_n_membership(self) -> None:
        result = compute_topology_membership(
            np.array([[0.0, 0.0]]), np.array([1.0]), k=5
        )
        self.assertEqual(result.mu_topo.shape[0], 1)
        self.assertEqual(result.mu_topo[0], 0.0)

    def test_weight_injection(self) -> None:
        np.random.seed(2)
        X = np.random.randn(50, 3)
        y = np.array([1.0] * 15 + [-1.0] * 35)
        balls = generate_granular_balls(X, y, purity_threshold=0.9, min_ball_count=2)
        k_pos, k_neg, _, _, _ = compute_binary_class_weights(y)
        w_pos, w_neg, result, kpb, knb = ball_topo_weights(balls, k_pos, k_neg)
        mu_pos, mu_neg = split_membership_by_class(balls, result.mu_topo)
        self.assertTrue(np.allclose(w_pos, kpb * mu_pos))
        self.assertTrue(np.allclose(w_neg, knb * mu_neg))

    def test_competition_k_logged_via_resolve(self) -> None:
        centers = np.random.randn(30, 2)
        labels = np.array([1.0] * 10 + [-1.0] * 20)
        _, k = compute_competition_fraction(centers, labels)
        self.assertEqual(k, resolve_k(30))


class TestTopoModel(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(5)
        self.X = np.random.randn(60, 3)
        self.X[:20, 0] += 1.5
        self.X[20:, 0] -= 1.5
        self.y = np.array([1.0] * 20 + [-1.0] * 40)
        self.params = {
            "c1": 20.0,
            "c2": 0.05,
            "purity_threshold": 0.9,
            "min_ball_count": 2,
            "prediction_rule": "reference",
            "random_state": 3,
        }

    def test_fit_and_predict(self) -> None:
        model = TopoGBLSTSVM(**self.params).fit(self.X, self.y)
        preds = model.predict(self.X)
        self.assertTrue(np.all(np.isin(preds, [-1.0, 1.0])))
        self.assertIsNotNone(model.membership_)
        self.assertIsNotNone(model.k_neighbors_)
        self.assertEqual(model.weights_pos_.shape[0], model.centers_pos_.shape[0])
        self.assertEqual(model.weights_neg_.shape[0], model.centers_neg_.shape[0])

    def test_empty_balls_weights(self) -> None:
        w_pos, w_neg, result, kpb, knb = ball_topo_weights([], 1.0, 1.0)
        self.assertEqual(w_pos.size, 0)
        self.assertEqual(result.k, 0)


if __name__ == "__main__":
    unittest.main()
