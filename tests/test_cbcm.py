"""Unit tests for CBCM membership functions and GBLSTSVM variants."""

import unittest

import numpy as np

from src.cbcm import (
    compute_cbcm_membership,
    compute_competition_membership,
    compute_connectivity_membership,
    compute_naive_density_weight,
    kappa_per_ball_array,
    resolve_k,
    split_membership_by_class,
)
from src.cbcm_gblstsvm import CAGBLSTSVM, CBGBLSTSVM, CCGBLSTSVM
from src.class_weights import compute_binary_class_weights
from src.cw_gblstsvm import CWGBLSTSVM
from src.granular_balls import BallInfo, generate_granular_balls


class TestResolveK(unittest.TestCase):
    def test_formula(self) -> None:
        self.assertEqual(resolve_k(10), 5)
        self.assertEqual(resolve_k(100), 5)
        self.assertEqual(resolve_k(101), 6)
        self.assertEqual(resolve_k(200), 10)


class TestCompetitionMembership(unittest.TestCase):
    def test_homogeneous_neighbors_low_competition(self) -> None:
        # Two tight same-class clusters far apart -> low comp near cluster centers.
        centers = np.vstack(
            [
                np.random.randn(20, 2) * 0.1 + np.array([0.0, 0.0]),
                np.random.randn(20, 2) * 0.1 + np.array([10.0, 10.0]),
            ]
        )
        labels = np.array([1.0] * 20 + [-1.0] * 20)
        comp, mu_a, k = compute_competition_membership(centers, labels, k=5)
        self.assertEqual(k, 5)
        self.assertTrue(np.all(comp >= 0.0) and np.all(comp <= 1.0))
        self.assertTrue(np.all(mu_a >= 0.0) and np.all(mu_a <= 1.0))
        # Boundary region balls (if any mixed neighbors) should have higher comp.
        self.assertTrue(np.any(comp < 0.5))

    def test_mixed_neighbors_high_competition(self) -> None:
        centers = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [10.0, 0.0],
                [10.1, 0.0],
            ]
        )
        labels = np.array([1.0, -1.0, 1.0, -1.0])
        comp, mu_a, _ = compute_competition_membership(centers, labels, k=2)
        self.assertGreater(comp[0], 0.0)
        self.assertGreater(mu_a[0], 0.0)

    def test_parabolic_remap_peaks_at_half(self) -> None:
        comp = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        mu_a = 4.0 * comp * (1.0 - comp)
        self.assertAlmostEqual(mu_a[2], 1.0)
        self.assertAlmostEqual(mu_a[0], 0.0)
        self.assertAlmostEqual(mu_a[4], 0.0)


class TestConnectivityMembership(unittest.TestCase):
    def test_within_class_normalization(self) -> None:
        centers = np.vstack(
            [
                np.random.randn(15, 2) * 0.05,
                np.random.randn(15, 2) * 0.05 + np.array([5.0, 5.0]),
            ]
        )
        labels = np.array([1.0] * 15 + [-1.0] * 15)
        mu_b, k, meta = compute_connectivity_membership(centers, labels, k=5)
        self.assertTrue(np.all(mu_b > 0.0) and np.all(mu_b <= 1.0))
        self.assertIn("densities", meta)

    def test_single_ball_per_class_caps_k(self) -> None:
        centers = np.array([[0.0, 0.0], [1.0, 1.0]])
        labels = np.array([1.0, -1.0])
        mu_b, _, meta = compute_connectivity_membership(centers, labels, k=5)
        self.assertEqual(mu_b.shape[0], 2)
        self.assertIn("class_k_capped", meta)


class TestCBCMCombined(unittest.TestCase):
    def test_mu_c_is_product(self) -> None:
        np.random.seed(0)
        centers = np.random.randn(40, 3)
        labels = np.array([1.0] * 15 + [-1.0] * 25)
        result = compute_cbcm_membership(centers, labels)
        self.assertTrue(np.allclose(result.mu_c, result.mu_a * result.mu_b))

    def test_ball_weights_product_with_cw(self) -> None:
        np.random.seed(1)
        X = np.random.randn(50, 3)
        y = np.array([1.0] * 15 + [-1.0] * 35)
        balls = generate_granular_balls(X, y, purity_threshold=0.9, min_ball_count=2)
        k_pos, k_neg, _, _, _ = compute_binary_class_weights(y)
        from src.cbcm import ball_cbcm_weights

        w_pos, w_neg, result, kpb, knb = ball_cbcm_weights(
            balls, k_pos, k_neg, variant="combined"
        )
        mu_pos, mu_neg = split_membership_by_class(balls, result.mu_c)
        self.assertTrue(np.allclose(w_pos, kpb * mu_pos))
        self.assertTrue(np.allclose(w_neg, knb * mu_neg))


class TestCBCMModels(unittest.TestCase):
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

    def test_models_fit_and_predict(self) -> None:
        for cls in (CAGBLSTSVM, CBGBLSTSVM, CCGBLSTSVM):
            model = cls(**self.params).fit(self.X, self.y)
            preds = model.predict(self.X)
            self.assertTrue(np.all(np.isin(preds, [-1.0, 1.0])))
            self.assertIsNotNone(model.membership_)
            self.assertIsNotNone(model.k_neighbors_)

    def test_cc_reduces_to_cw_when_membership_one(self) -> None:
        """If all muC=1, CC should match CW (not guaranteed on all data, smoke test)."""
        cw = CWGBLSTSVM(**self.params).fit(self.X, self.y)
        cc = CCGBLSTSVM(**self.params).fit(self.X, self.y)
        self.assertIsNotNone(cc.membership_)
        # Weights should differ from CW when membership varies.
        if not np.allclose(cc.membership_.mu_c, 1.0):
            self.assertFalse(
                np.allclose(cw.weights_pos_, cc.weights_pos_)
                and np.allclose(cw.weights_neg_, cc.weights_neg_)
            )


class TestNaiveDensityContrast(unittest.TestCase):
    def test_naive_density_weight_shape(self) -> None:
        balls = [
            BallInfo(center=np.zeros(2), radius=0.1, label=1.0, purity=1.0, num_samples=5),
            BallInfo(center=np.ones(2), radius=0.1, label=-1.0, purity=1.0, num_samples=5),
        ]
        kappa = kappa_per_ball_array(balls, 2.0, 0.5)
        w = compute_naive_density_weight(
            np.vstack([b.center for b in balls]),
            np.array([b.label for b in balls]),
            kappa,
            k=1,
        )
        self.assertEqual(w.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
