"""Unit tests for IPA membership and IPA-GBLSTSVM."""

import unittest

import numpy as np

from src.cbcm import compute_competition_membership
from src.class_weights import compute_binary_class_weights
from src.granular_balls import BallInfo, generate_granular_balls
from src.ipa import (
    DEFAULT_LAMBDA,
    ball_ipa_weights,
    compute_ambiguity,
    compute_ipa_membership,
    compute_isolation,
    compute_nearest_same_class_distance,
)
from src.ipa_gblstsvm import IPAGBLSTSVM
from src.cbcm import split_membership_by_class


class TestAmbiguity(unittest.TestCase):
    def test_a2_peaks_at_half(self) -> None:
        comp = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        amb = compute_ambiguity(comp)
        self.assertAlmostEqual(amb[2], 1.0)
        self.assertAlmostEqual(amb[0], 0.0)
        self.assertAlmostEqual(amb[4], 0.0)
        self.assertAlmostEqual(amb[1], 0.5)

    def test_symmetry(self) -> None:
        comp = np.array([0.2, 0.8])
        amb = compute_ambiguity(comp)
        self.assertAlmostEqual(amb[0], amb[1])


class TestIsolation(unittest.TestCase):
    def test_connected_ball_no_penalty(self) -> None:
        centers = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 0.0], [5.1, 0.0]])
        labels = np.array([1.0, 1.0, -1.0, -1.0])
        nearest = compute_nearest_same_class_distance(centers, labels)
        clipped, raw = compute_isolation(nearest, labels)
        # Ball 0: nearest same at 0.1, median same for class ~0.1 -> ratio ~1
        self.assertLessEqual(clipped[0], 1.0)
        self.assertLessEqual(raw[0], 1.1)

    def test_isolated_ball_raw_exceeds_one(self) -> None:
        centers = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0]])
        labels = np.array([1.0, 1.0, 1.0])
        nearest = compute_nearest_same_class_distance(centers, labels)
        clipped, raw = compute_isolation(nearest, labels)
        # Ball at 10 is far from others -> raw > 1
        self.assertGreater(raw[2], 1.0)
        self.assertEqual(clipped[2], 1.0)


class TestIPAMembership(unittest.TestCase):
    def test_penalty_reduces_isolated_ambiguous_ball(self) -> None:
        centers = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.05, 0.1],
                [5.0, 0.0],
                [5.1, 0.0],
            ]
        )
        labels = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        result = compute_ipa_membership(centers, labels, k=2, lam=1.0)
        self.assertTrue(np.all(result.mu_ipa >= 0.0))
        self.assertTrue(np.all(result.mu_ipa <= result.ambiguity + 1e-12))

    def test_no_penalty_when_raw_isolation_at_most_one(self) -> None:
        comp = np.array([0.5])
        amb = compute_ambiguity(comp)
        centers = np.array([[0.0, 0.0], [0.5, 0.0]])
        labels = np.array([1.0, 1.0])
        nearest = np.array([0.5])
        _, raw = compute_isolation(nearest, labels[:1])
        excess = max(0.0, raw[0] - 1.0)
        expected = amb[0] / (1.0 + DEFAULT_LAMBDA * excess)
        result = compute_ipa_membership(
            np.array([[0.0, 0.0]]), np.array([1.0]), k=1
        )
        # Single ball: no same-class neighbor -> isolation handled
        self.assertEqual(result.mu_ipa.shape[0], 1)

    def test_ball_weights_product_with_cw(self) -> None:
        np.random.seed(2)
        X = np.random.randn(50, 3)
        y = np.array([1.0] * 15 + [-1.0] * 35)
        balls = generate_granular_balls(X, y, purity_threshold=0.9, min_ball_count=2)
        k_pos, k_neg, _, _, _ = compute_binary_class_weights(y)
        w_pos, w_neg, result, kpb, knb = ball_ipa_weights(balls, k_pos, k_neg)
        mu_pos, mu_neg = split_membership_by_class(balls, result.mu_ipa)
        self.assertTrue(np.allclose(w_pos, kpb * mu_pos))
        self.assertTrue(np.allclose(w_neg, knb * mu_neg))


class TestIPAModel(unittest.TestCase):
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
        model = IPAGBLSTSVM(**self.params).fit(self.X, self.y)
        preds = model.predict(self.X)
        self.assertTrue(np.all(np.isin(preds, [-1.0, 1.0])))
        self.assertIsNotNone(model.membership_)
        self.assertIsNotNone(model.k_neighbors_)
        self.assertEqual(model.lam, DEFAULT_LAMBDA)

    def test_membership_fields(self) -> None:
        model = IPAGBLSTSVM(**self.params).fit(self.X, self.y)
        m = model.membership_
        self.assertTrue(np.allclose(m.ambiguity, compute_ambiguity(m.comp_a)))
        self.assertEqual(m.lam, DEFAULT_LAMBDA)


if __name__ == "__main__":
    unittest.main()
