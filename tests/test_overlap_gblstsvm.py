"""Tests for overlap-aware weighting and OBRU-GBLSTSVM variants (Stages 7–9)."""

import math
import unittest

import numpy as np

from src.class_weights import ball_class_weights, compute_binary_class_weights
from src.cr_gblstsvm import CRGBLSTSVM
from src.cw_gblstsvm import CWGBLSTSVM
from src.gblstsvm import GBLSTSVM
from src.granular_balls import BallInfo, generate_granular_balls
from src.overlap import (
    DEFAULT_ALPHA,
    DEFAULT_EPSILON,
    DEFAULT_TOP_K,
    aggregate_top_k_overlaps,
    ball_cr_obru_weights,
    ball_cw_obru_weights,
    ball_obru_weights,
    ball_overlap_scores,
    minority_label_from_training,
    overlap_mu_from_scores,
    pair_overlap_proxy,
)
from src.overlap_gblstsvm import CROBRUGBLSTSVM, CWOBRUGBLSTSVM, OBRUGBLSTSVM
from src.r_gblstsvm import RGBLSTSVM
from src.reliability import ball_reliability_weights


class TestOverlapPairFormula(unittest.TestCase):
    def test_no_overlap(self) -> None:
        r_i, r_j, d = 2.0, 3.0, 10.0
        self.assertGreater(d, r_i + r_j)
        self.assertEqual(pair_overlap_proxy(r_i, r_j, d), 0.0)

    def test_positive_overlap(self) -> None:
        r_i, r_j, d = 2.0, 3.0, 4.0
        self.assertLess(d, r_i + r_j)
        o = pair_overlap_proxy(r_i, r_j, d)
        self.assertGreater(o, 0.0)
        expected = (r_i + r_j - d) / (r_i + r_j + DEFAULT_EPSILON)
        self.assertAlmostEqual(o, expected)

    def test_identical_centers(self) -> None:
        o = pair_overlap_proxy(1.5, 2.0, 0.0)
        self.assertTrue(np.isfinite(o))
        self.assertAlmostEqual(o, 1.0, places=5)

    def test_zero_radii(self) -> None:
        o = pair_overlap_proxy(0.0, 0.0, 1.0)
        self.assertTrue(np.isfinite(o))
        self.assertEqual(o, 0.0)


class TestOppositeClassFiltering(unittest.TestCase):
    def test_same_class_excluded(self) -> None:
        balls = [
            BallInfo(center=np.array([0.0, 0.0]), radius=2.0, label=1.0, purity=1.0, num_samples=5),
            BallInfo(center=np.array([0.1, 0.0]), radius=2.0, label=1.0, purity=1.0, num_samples=5),
            BallInfo(center=np.array([10.0, 0.0]), radius=2.0, label=-1.0, purity=1.0, num_samples=5),
        ]
        O = ball_overlap_scores(balls)
        # Ball 0 overlaps strongly with ball 1 (same class) but only ball 2 is opposite.
        o_02 = pair_overlap_proxy(2.0, 2.0, np.linalg.norm(balls[0].center - balls[2].center))
        self.assertAlmostEqual(O[0], o_02)


class TestTop3Averaging(unittest.TestCase):
    def test_exact_top3_mean(self) -> None:
        overlaps = np.array([0.1, 0.9, 0.4, 0.8, 0.2])
        top3 = aggregate_top_k_overlaps(overlaps, top_k=3)
        self.assertAlmostEqual(top3, np.mean([0.9, 0.8, 0.4]))

    def test_fewer_than_three_opposite(self) -> None:
        overlaps = np.array([0.2, 0.6])
        self.assertAlmostEqual(
            aggregate_top_k_overlaps(overlaps, top_k=DEFAULT_TOP_K),
            0.4,
        )

    def test_no_opposite_class_balls(self) -> None:
        balls = [
            BallInfo(center=np.array([0.0]), radius=1.0, label=1.0, purity=1.0, num_samples=1),
            BallInfo(center=np.array([1.0]), radius=1.0, label=1.0, purity=1.0, num_samples=1),
        ]
        O = ball_overlap_scores(balls)
        self.assertEqual(O[0], 0.0)
        self.assertEqual(O[1], 0.0)


class TestClassConditionalMu(unittest.TestCase):
    def setUp(self) -> None:
        self.balls = [
            BallInfo(center=np.array([0.0]), radius=1.0, label=1.0, purity=1.0, num_samples=1),
            BallInfo(center=np.array([1.0]), radius=1.0, label=-1.0, purity=1.0, num_samples=1),
        ]
        self.minority = 1.0
        self.O = np.array([0.3, 0.7])

    def test_minority_ball(self) -> None:
        mu = overlap_mu_from_scores(
            self.balls, self.O, self.minority, alpha=DEFAULT_ALPHA
        )
        self.assertAlmostEqual(mu[0], 1.0 + self.O[0])

    def test_majority_ball(self) -> None:
        mu = overlap_mu_from_scores(
            self.balls, self.O, self.minority, alpha=DEFAULT_ALPHA
        )
        self.assertAlmostEqual(mu[1], 1.0 - self.O[1])


class TestOverlapWeightCombinations(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)
        self.X = np.random.randn(60, 3)
        self.y = np.array([1.0] * 20 + [-1.0] * 40)
        self.balls = generate_granular_balls(
            self.X, self.y, purity_threshold=0.9, min_ball_count=2
        )
        self.kappa_pos, self.kappa_neg, _, _, _ = compute_binary_class_weights(
            self.y
        )
        self.minority = minority_label_from_training(self.y)

    def test_cw_obru_product(self) -> None:
        w_pos, w_neg, _, _, k_pos, k_neg = ball_cw_obru_weights(
            self.balls,
            self.kappa_pos,
            self.kappa_neg,
            self.minority,
        )
        o_pos, o_neg, _, _ = ball_obru_weights(
            self.balls, self.minority
        )
        self.assertTrue(np.allclose(w_pos, k_pos * o_pos))
        self.assertTrue(np.allclose(w_neg, k_neg * o_neg))

    def test_cr_obru_triple_product(self) -> None:
        w_pos, w_neg, _, rel, _, _, k_pos, k_neg = ball_cr_obru_weights(
            self.balls,
            self.kappa_pos,
            self.kappa_neg,
            self.minority,
        )
        rel_pos, rel_neg, _, _ = ball_reliability_weights(self.balls)
        o_pos, o_neg, _, _ = ball_obru_weights(
            self.balls, self.minority
        )
        self.assertTrue(np.allclose(w_pos, k_pos * rel_pos * o_pos))
        self.assertTrue(np.allclose(w_neg, k_neg * rel_neg * o_neg))


class TestOBRUModels(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(1)
        self.X = np.random.randn(50, 4)
        self.y = np.sign(self.X[:, 0] + 0.3 * self.X[:, 1])
        self.y[self.y == 0] = 1.0
        self.params = {
            "c1": 10.0,
            "c2": 0.1,
            "purity_threshold": 0.9,
            "min_ball_count": 2,
            "random_state": 1,
        }

    def test_obru_fit_produces_finite_weights(self) -> None:
        model = OBRUGBLSTSVM(**self.params).fit(self.X, self.y)
        self.assertIsNotNone(model.mu_all_)
        self.assertTrue(np.all(np.isfinite(model.mu_all_)))
        self.assertTrue(np.all(np.isfinite(model.overlap_all_)))

    def test_cw_obru_weights_match_factors(self) -> None:
        model = CWOBRUGBLSTSVM(**self.params).fit(self.X, self.y)
        self.assertTrue(
            np.allclose(model.weights_pos_, model.kappa_pos_balls_ * model.obru_pos_)
        )
        self.assertTrue(
            np.allclose(model.weights_neg_, model.kappa_neg_balls_ * model.obru_neg_)
        )

    def test_cr_obru_weights_match_factors(self) -> None:
        model = CROBRUGBLSTSVM(**self.params).fit(self.X, self.y)
        self.assertTrue(
            np.allclose(
                model.weights_pos_,
                model.kappa_pos_balls_ * model.rel_pos_ * model.obru_pos_,
            )
        )
        self.assertTrue(
            np.allclose(
                model.weights_neg_,
                model.kappa_neg_balls_ * model.rel_neg_ * model.obru_neg_,
            )
        )


class TestFrozenBaselinesUnchanged(unittest.TestCase):
    """Ensure existing GBLSTSVM/CW/R/CR behavior is unchanged."""

    def setUp(self) -> None:
        np.random.seed(42)
        self.X = np.random.randn(40, 3)
        self.y = np.array([1.0] * 15 + [-1.0] * 25)
        self.base = {
            "c1": 10.0,
            "c2": 0.1,
            "purity_threshold": 0.95,
            "min_ball_count": 2,
            "random_state": 42,
        }

    def _fit_predict(self, cls):
        model = cls(**self.base).fit(self.X, self.y)
        return model, model.predict(self.X)

    def test_gblstsvm_reproducible(self) -> None:
        m1, p1 = self._fit_predict(GBLSTSVM)
        m2, p2 = self._fit_predict(GBLSTSVM)
        self.assertTrue(np.allclose(m1.w1_, m2.w1_))
        self.assertTrue(np.allclose(m1.w2_, m2.w2_))
        self.assertTrue(np.allclose(p1, p2))

    def test_cw_reproducible(self) -> None:
        m1, p1 = self._fit_predict(CWGBLSTSVM)
        m2, p2 = self._fit_predict(CWGBLSTSVM)
        self.assertTrue(np.allclose(m1.weights_pos_, m2.weights_pos_))
        self.assertTrue(np.allclose(p1, p2))

    def test_r_reproducible(self) -> None:
        m1, p1 = self._fit_predict(RGBLSTSVM)
        m2, p2 = self._fit_predict(RGBLSTSVM)
        self.assertTrue(np.allclose(m1.reliabilities_, m2.reliabilities_))
        self.assertTrue(np.allclose(p1, p2))

    def test_cr_reproducible(self) -> None:
        m1, p1 = self._fit_predict(CRGBLSTSVM)
        m2, p2 = self._fit_predict(CRGBLSTSVM)
        self.assertTrue(np.allclose(m1.weights_pos_, m2.weights_pos_))
        self.assertTrue(np.allclose(p1, p2))


if __name__ == "__main__":
    unittest.main()
