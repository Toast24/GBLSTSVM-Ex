"""Tests for BMRU weighting and BMRU-GBLSTSVM variants (Stages 4–6)."""

import math
import unittest

import numpy as np

from src.bmru import (
    DEFAULT_ALPHA,
    DEFAULT_EPSILON,
    ball_boundary_diagnostics,
    ball_bmru_weights,
    bmru_uncertainty,
    bmru_weight,
    boundary_u_detector_confusion,
    class_weight_mass_diagnostics,
    combined_weights_per_ball,
    decision_margin_at_center,
    decision_scores_at_points,
    prediction_distribution,
    safe_correlation,
)
from src.bmru_gblstsvm import (
    BMRUGBLSTSVM,
    CRBMRUGBLSTSVM,
    CWBMRUGBLSTSVM,
    ball_cr_bmru_weights,
    ball_cw_bmru_weights,
)
from src.class_weights import ball_class_weights, compute_binary_class_weights
from src.cr_gblstsvm import CRGBLSTSVM
from src.cw_gblstsvm import CWGBLSTSVM
from src.gblstsvm import GBLSTSVM
from src.granular_balls import BallInfo, generate_granular_balls
from src.r_gblstsvm import RGBLSTSVM
from src.reliability import ball_reliability_weights


class TestBMRUFormula(unittest.TestCase):
    def test_small_ball_far_from_boundary(self) -> None:
        r, s = 0.01, 10.0
        u, mu = bmru_weight(r, s, alpha=DEFAULT_ALPHA, epsilon=DEFAULT_EPSILON)
        self.assertLess(u, 0.01)
        self.assertGreater(mu, 0.99)

    def test_large_ball_near_boundary(self) -> None:
        r, s = 5.0, 0.01
        u, mu = bmru_weight(r, s, alpha=DEFAULT_ALPHA, epsilon=DEFAULT_EPSILON)
        self.assertGreater(u, 100.0)
        self.assertLess(mu, 0.01)

    def test_boundary_straddling_ball(self) -> None:
        r, s = 2.0, 1.0
        u, mu = bmru_weight(r, s, alpha=DEFAULT_ALPHA, epsilon=DEFAULT_EPSILON)
        self.assertGreaterEqual(u, 1.0)
        self.assertLessEqual(mu, np.exp(-1.0) + 1e-12)

    def test_zero_radius_ball(self) -> None:
        u, mu = bmru_weight(0.0, 0.5, alpha=DEFAULT_ALPHA, epsilon=DEFAULT_EPSILON)
        self.assertEqual(u, 0.0)
        self.assertAlmostEqual(mu, 1.0)

    def test_zero_decision_margin(self) -> None:
        u = bmru_uncertainty(1.0, 0.0, epsilon=DEFAULT_EPSILON)
        self.assertTrue(np.isfinite(u))
        self.assertGreater(u, 0.0)
        mu = float(np.exp(-DEFAULT_ALPHA * u))
        self.assertLess(mu, 1.0)

    def test_paper_margin_mode(self) -> None:
        center = np.array([1.0, 0.0])
        w1 = np.array([2.0, 0.0])
        w2 = np.array([1.0, 0.0])
        s_ref, _, _ = decision_margin_at_center(
            center, w1, 0.0, w2, 0.0, margin_mode="reference"
        )
        s_paper, d1, d2 = decision_margin_at_center(
            center, w1, 0.0, w2, 0.0, margin_mode="paper"
        )
        self.assertAlmostEqual(s_ref, -1.0)
        self.assertAlmostEqual(d1, 1.0)
        self.assertAlmostEqual(d2, 1.0)
        self.assertAlmostEqual(s_paper, 0.0)


class TestBMRUWeightCombinations(unittest.TestCase):
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
        model = GBLSTSVM(
            c1=10.0,
            c2=0.1,
            purity_threshold=0.9,
            min_ball_count=2,
            random_state=0,
        ).fit(self.X, self.y)
        self.w1, self.b1 = model.w1_, model.b1_
        self.w2, self.b2 = model.w2_, model.b2_

    def test_cw_bmru_product(self) -> None:
        w_pos, w_neg, _, mu, _, k_pos, k_neg = ball_cw_bmru_weights(
            self.balls,
            self.kappa_pos,
            self.kappa_neg,
            self.w1,
            self.b1,
            self.w2,
            self.b2,
        )
        bmru_pos, bmru_neg, _, _, _ = ball_bmru_weights(
            self.balls, self.w1, self.b1, self.w2, self.b2
        )
        self.assertTrue(np.allclose(w_pos, k_pos * bmru_pos))
        self.assertTrue(np.allclose(w_neg, k_neg * bmru_neg))

    def test_cr_bmru_triple_product(self) -> None:
        w_pos, w_neg, _, rel, _, _, _, k_pos, k_neg = ball_cr_bmru_weights(
            self.balls,
            self.kappa_pos,
            self.kappa_neg,
            self.w1,
            self.b1,
            self.w2,
            self.b2,
        )
        rel_pos, rel_neg, _, _ = ball_reliability_weights(self.balls)
        bmru_pos, bmru_neg, _, _, _ = ball_bmru_weights(
            self.balls, self.w1, self.b1, self.w2, self.b2
        )
        self.assertTrue(np.allclose(w_pos, k_pos * rel_pos * bmru_pos))
        self.assertTrue(np.allclose(w_neg, k_neg * rel_neg * bmru_neg))


class TestBMRUModels(unittest.TestCase):
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
            "max_iterations": 3,
        }

    def test_bmru_fit_produces_finite_weights(self) -> None:
        model = BMRUGBLSTSVM(**self.params).fit(self.X, self.y)
        self.assertIsNotNone(model.mu_all_)
        self.assertTrue(np.all(np.isfinite(model.mu_all_)))
        self.assertTrue(np.all(model.mu_all_ > 0.0))
        self.assertTrue(np.all(model.mu_all_ <= 1.0))
        self.assertGreater(len(model.iteration_history_), 0)

    def test_cw_bmru_differs_from_cw_when_bmru_active(self) -> None:
        bmru = CWBMRUGBLSTSVM(**self.params).fit(self.X, self.y)
        cw = CWGBLSTSVM(**{k: v for k, v in self.params.items() if k != "max_iterations"}).fit(
            self.X, self.y
        )
        bmru_w = np.concatenate([bmru.weights_pos_, bmru.weights_neg_])
        cw_w = np.concatenate([cw.weights_pos_, cw.weights_neg_])
        if np.any(np.abs(bmru.mu_all_ - 1.0) > 1e-6):
            self.assertTrue(np.any(np.abs(bmru_w - cw_w) > 1e-12))

    def test_cr_bmru_weights_match_factors(self) -> None:
        model = CRBMRUGBLSTSVM(**self.params).fit(self.X, self.y)
        self.assertTrue(
            np.allclose(
                model.weights_pos_,
                model.kappa_pos_balls_ * model.rel_pos_ * model.bmru_pos_,
            )
        )
        self.assertTrue(
            np.allclose(
                model.weights_neg_,
                model.kappa_neg_balls_ * model.rel_neg_ * model.bmru_neg_,
            )
        )


class TestBMRUDiagnostics(unittest.TestCase):
    def test_reference_and_paper_margins_differ_when_norms_differ(self) -> None:
        center = np.array([1.0, 0.0])
        w1 = np.array([2.0, 0.0])
        w2 = np.array([1.0, 0.0])
        s_ref, _, _ = decision_margin_at_center(
            center, w1, 0.0, w2, 0.0, margin_mode="reference"
        )
        s_paper, _, _ = decision_margin_at_center(
            center, w1, 0.0, w2, 0.0, margin_mode="paper"
        )
        self.assertNotAlmostEqual(s_ref, s_paper)
        self.assertAlmostEqual(s_ref, -1.0)
        self.assertAlmostEqual(s_paper, 0.0)

    def test_class_weight_sums_match_ball_weights(self) -> None:
        np.random.seed(3)
        X = np.random.randn(40, 3)
        y = np.array([1.0] * 12 + [-1.0] * 28)
        model = CWBMRUGBLSTSVM(
            c1=10.0,
            c2=0.1,
            purity_threshold=0.9,
            min_ball_count=2,
            random_state=3,
            max_iterations=2,
        ).fit(X, y)
        combined = combined_weights_per_ball(model)
        rows = class_weight_mass_diagnostics(model.balls_, model.mu_all_, combined)
        for row in rows:
            label = row["class_label"]
            mask = np.array([b.label == label for b in model.balls_])
            expected_sum = float(np.sum(combined[mask]))
            self.assertAlmostEqual(row["sum_combined_weight"], expected_sum, places=12)
            expected_mu_sum = float(np.sum(model.mu_all_[mask]))
            self.assertAlmostEqual(row["sum_bmru_weight"], expected_mu_sum, places=12)

    def test_prediction_counts_sum_to_n(self) -> None:
        y_pred = np.array([1.0, -1.0, 1.0, 1.0, -1.0])
        dist = prediction_distribution(y_pred)
        self.assertEqual(
            dist["n_positive_predictions"] + dist["n_negative_predictions"],
            dist["n_predictions"],
        )
        self.assertEqual(dist["n_predictions"], 5)

    def test_zero_variance_correlation_is_nan(self) -> None:
        a = np.ones(5)
        b = np.linspace(0.1, 0.5, 5)
        self.assertTrue(math.isnan(safe_correlation(a, b)))


class TestBMRUBoundaryDiagnostics(unittest.TestCase):
    def _make_ball(
        self, X: np.ndarray, w1: np.ndarray, w2: np.ndarray
    ) -> tuple[BallInfo, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, float]:
        ball = BallInfo(
            center=X.mean(0),
            radius=float(np.mean(np.linalg.norm(X - X.mean(0), axis=1))),
            label=1.0,
            purity=1.0,
            num_samples=len(X),
        )
        return ball, np.arange(len(X)), X, w1, 0.0, w2, 0.0

    def test_boundary_straddling_ball_detected(self) -> None:
        w1 = np.array([1.0, 0.0])
        w2 = np.array([0.0, 1.0])
        X = np.array([[2.0, 0.0], [0.0, 2.0], [0.5, 0.5]])
        ball, idx, X, w1, b1, w2, b2 = self._make_ball(X, w1, w2)
        diag = ball_boundary_diagnostics(
            ball, idx, X, w1, b1, w2, b2, u=2.0, mu=0.1
        )
        self.assertTrue(diag["boundary_straddles"])

    def test_one_sided_ball_not_straddling(self) -> None:
        w1 = np.array([1.0, 0.0])
        w2 = np.array([0.0, 1.0])
        X = np.array([[2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
        ball, idx, X, w1, b1, w2, b2 = self._make_ball(X, w1, w2)
        diag = ball_boundary_diagnostics(
            ball, idx, X, w1, b1, w2, b2, u=0.5, mu=0.9
        )
        self.assertFalse(diag["boundary_straddles"])

    def test_min_abs_sample_margin(self) -> None:
        X = np.array([[1.0, 0.0], [3.0, 0.0]])
        w1 = np.array([1.0, 0.0])
        w2 = np.array([2.0, 0.0])
        scores = decision_scores_at_points(X, w1, 0.0, w2, 0.0)
        self.assertAlmostEqual(float(np.min(np.abs(scores))), 1.0)
        ball = BallInfo(center=X.mean(0), radius=1.0, label=1.0, purity=1.0, num_samples=2)
        diag = ball_boundary_diagnostics(
            ball, np.arange(2), X, w1, 0.0, w2, 0.0, u=1.0, mu=0.5
        )
        self.assertAlmostEqual(diag["min_abs_sample_margin"], 1.0)

    def test_max_radius_ratio(self) -> None:
        X = np.array([[0.0, 0.0], [2.0, 0.0]])
        ball = BallInfo(center=np.array([0.0, 0.0]), radius=1.0, label=1.0, purity=1.0, num_samples=2)
        diag = ball_boundary_diagnostics(
            ball, np.arange(2), X, np.ones(2), 0.0, np.ones(2), 0.0, u=1.0, mu=0.5
        )
        self.assertAlmostEqual(diag["max_center_distance"], 2.0)
        self.assertAlmostEqual(diag["max_radius_ratio"], 2.0 / (1.0 + DEFAULT_EPSILON))

    def test_u_ge_1_grouping_confusion(self) -> None:
        rows = [
            {"u": 0.5, "mu": 0.9, "boundary_straddles": False, "boundary_near": False},
            {"u": 2.0, "mu": 0.05, "boundary_straddles": True, "boundary_near": True},
            {"u": 3.0, "mu": 0.01, "boundary_straddles": False, "boundary_near": False},
        ]
        conf = boundary_u_detector_confusion(
            rows, empirical_field="boundary_straddles", empirical_label="straddles"
        )
        self.assertEqual(conf["tp"], 1)
        self.assertEqual(conf["fp"], 1)
        self.assertEqual(conf["tn"], 1)
        self.assertEqual(conf["fn"], 0)

    def test_zero_variance_correlation_returns_nan(self) -> None:
        rows = [
            {"u": 1.0, "mu": 0.5, "min_abs_sample_margin": 0.1, "max_center_distance": 1.0,
             "radius": 1.0, "purity": 1.0, "boundary_straddles": False, "boundary_near": False},
            {"u": 1.0, "mu": 0.5, "min_abs_sample_margin": 0.2, "max_center_distance": 1.0,
             "radius": 1.0, "purity": 1.0, "boundary_straddles": False, "boundary_near": False},
        ]
        from src.bmru import boundary_correlations

        corr = boundary_correlations(rows)
        self.assertTrue(math.isnan(corr["corr_u_min_abs_sample_margin"]))


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
