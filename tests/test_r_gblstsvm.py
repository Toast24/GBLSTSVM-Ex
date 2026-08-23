"""Tests for ball-level reliability and R-GBLSTSVM."""

import unittest

import numpy as np

from src.cw_gblstsvm import solve_cw_gblstsvm_systems
from src.gblstsvm import GBLSTSVM, solve_gblstsvm_systems
from src.granular_balls import BallInfo, generate_granular_balls
from src.r_gblstsvm import RGBLSTSVM
from src.reliability import (
    ball_reliability_score,
    ball_reliability_weights,
    compute_radius_scale,
    normalize_radius,
)


class TestReliabilityFormula(unittest.TestCase):
    def test_full_formula_matches_definition(self) -> None:
        p, r, scale = 0.8, 0.4, 0.5
        r_tilde = normalize_radius(r, scale)
        epsilon, gamma, lambda_r = 0.05, 1.0, 2.0
        expected = epsilon + (1 - epsilon) * (p**gamma) * np.exp(
            -lambda_r * r_tilde * (1 - p)
        )
        got = ball_reliability_score(
            p, r, scale, epsilon=epsilon, gamma=gamma, lambda_r=lambda_r, variant="full"
        )
        self.assertAlmostEqual(got, expected, places=12)

    def test_purity_only_variant(self) -> None:
        score = ball_reliability_score(
            0.7, 1.0, 1.0, epsilon=0.1, gamma=2.0, variant="purity_only"
        )
        self.assertAlmostEqual(score, 0.1 + 0.9 * 0.7**2)

    def test_radius_impurity_only_variant(self) -> None:
        # r_tilde = 1.0 / 2.0 = 0.5, impurity = 0.5
        score = ball_reliability_score(
            0.5, 1.0, 2.0, epsilon=0.1, lambda_r=1.0, variant="radius_impurity_only"
        )
        self.assertAlmostEqual(score, 0.1 + 0.9 * np.exp(-0.5 * 0.5))

    def test_identical_balls_identical_reliability(self) -> None:
        args = dict(purity=0.9, radius=0.3, radius_scale=0.6, epsilon=0.05)
        a = ball_reliability_score(**args)
        b = ball_reliability_score(**args)
        self.assertEqual(a, b)


class TestReliabilityBounds(unittest.TestCase):
    def test_positive_lower_bound_epsilon(self) -> None:
        for p in [0.0, 0.5, 1.0]:
            for r in [0.0, 0.5, 2.0]:
                score = ball_reliability_score(p, r, 1.0, epsilon=0.05)
                self.assertGreaterEqual(score, 0.05)
                self.assertLessEqual(score, 1.0)

    def test_upper_bound_one(self) -> None:
        score = ball_reliability_score(1.0, 0.0, 1.0, epsilon=0.05, gamma=1.0, lambda_r=1.0)
        self.assertAlmostEqual(score, 1.0)


class TestReliabilityMonotonicity(unittest.TestCase):
    def test_lower_purity_not_higher_reliability_fixed_radius(self) -> None:
        scale = 1.0
        radius = 0.5
        purities = [1.0, 0.9, 0.75, 0.6, 0.5]
        scores = [
            ball_reliability_score(p, radius, scale, epsilon=0.05, gamma=1.0, lambda_r=1.0)
            for p in purities
        ]
        self.assertTrue(all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)))

    def test_larger_radius_not_higher_reliability_fixed_impurity(self) -> None:
        scale = 1.0
        purity = 0.7
        radii = [0.0, 0.2, 0.5, 1.0, 2.0]
        scores = [
            ball_reliability_score(
                purity, r, scale, epsilon=0.05, gamma=1.0, lambda_r=1.0
            )
            for r in radii
        ]
        self.assertTrue(all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)))


class TestSingletonBehavior(unittest.TestCase):
    def test_singleton_radius_zero(self) -> None:
        ball = BallInfo(
            center=np.zeros(2),
            radius=0.0,
            label=1.0,
            purity=1.0,
            num_samples=1,
        )
        self.assertEqual(ball.radius, 0.0)
        score = ball_reliability_score(ball.purity, ball.radius, 1.0, epsilon=0.05)
        self.assertAlmostEqual(score, 1.0)

    def test_singleton_high_reliability_when_pure(self) -> None:
        """Documented limitation: r=0 removes radius penalty; p=1 gives a=1."""
        score = ball_reliability_score(1.0, 0.0, 1.0, epsilon=0.05, variant="full")
        self.assertAlmostEqual(score, 1.0)


class TestReliabilityWeights(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)
        self.X = np.random.randn(60, 3)
        self.y = np.sign(self.X[:, 0])
        self.y[self.y == 0] = 1
        self.balls = generate_granular_balls(
            self.X, self.y, purity_threshold=0.9, min_ball_count=2
        )

    def test_weight_construction_per_ball(self) -> None:
        w_pos, w_neg, scale, rel = ball_reliability_weights(self.balls)
        pos_idx = [i for i, b in enumerate(self.balls) if b.label == 1]
        neg_idx = [i for i, b in enumerate(self.balls) if b.label != 1]
        self.assertTrue(np.allclose(w_pos, rel[pos_idx]))
        self.assertTrue(np.allclose(w_neg, rel[neg_idx]))
        self.assertGreater(scale, 0.0)

    def test_radius_scale_ignores_zero_radii(self) -> None:
        radii = np.array([0.0, 0.0, 0.2, 0.4, 0.6])
        scale = compute_radius_scale(radii)
        self.assertAlmostEqual(scale, 0.4)


class TestRGBLSTSVM(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        np.random.seed(11)
        cls.X = np.random.randn(80, 4)
        cls.X[:40, 0] += 1.5
        cls.X[40:, 0] -= 1.5
        cls.y = np.array([1.0] * 40 + [-1.0] * 40)
        cls.params = {
            "c1": 20.0,
            "c2": 0.05,
            "purity_threshold": 0.9,
            "min_ball_count": 2,
            "prediction_rule": "reference",
            "random_state": 3,
        }

    def test_unit_reliability_reduces_to_baseline(self) -> None:
        """gamma=0 and lambda_r=0 => core=1 => a_i=1 for all balls."""
        base = GBLSTSVM(**self.params).fit(self.X, self.y)
        r_model = RGBLSTSVM(
            **self.params, gamma=0.0, lambda_r=0.0
        ).fit(self.X, self.y)

        self.assertTrue(np.allclose(r_model.reliabilities_, 1.0))
        self.assertTrue(np.allclose(base.w1_, r_model.w1_, atol=1e-10))
        self.assertTrue(np.allclose(base.w2_, r_model.w2_, atol=1e-10))
        self.assertAlmostEqual(base.b1_, r_model.b1_, places=10)
        self.assertAlmostEqual(base.b2_, r_model.b2_, places=10)
        self.assertTrue(np.allclose(base.predict(self.X), r_model.predict(self.X)))

    def test_opposite_only_weighting(self) -> None:
        r_model = RGBLSTSVM(**self.params, weight_placement="opposite_only").fit(
            self.X, self.y
        )
        w_pos, w_neg = r_model.weights_pos_, r_model.weights_neg_
        fw1, fb1, fw2, fb2 = solve_cw_gblstsvm_systems(
            r_model.centers_pos_,
            r_model.radii_pos_,
            r_model.centers_neg_,
            r_model.radii_neg_,
            self.params["c1"],
            self.params["c2"],
            w_pos,
            w_neg,
            weight_placement="opposite_only",
        )
        self.assertTrue(np.allclose(r_model.w1_, fw1))
        self.assertTrue(np.allclose(r_model.w2_, fw2))

    def test_both_term_weighting(self) -> None:
        r_model = RGBLSTSVM(**self.params, weight_placement="both").fit(self.X, self.y)
        w_pos, w_neg = r_model.weights_pos_, r_model.weights_neg_
        fw1, fb1, fw2, fb2 = solve_cw_gblstsvm_systems(
            r_model.centers_pos_,
            r_model.radii_pos_,
            r_model.centers_neg_,
            r_model.radii_neg_,
            self.params["c1"],
            self.params["c2"],
            w_pos,
            w_neg,
            weight_placement="both",
        )
        self.assertTrue(np.allclose(r_model.w1_, fw1))
        self.assertTrue(np.allclose(r_model.w2_, fw2))

    def test_both_term_weighting_uses_own_class_weights(self) -> None:
        """Both-term placement weights positive rows in hp1 own-class term."""
        from src.cw_gblstsvm import solve_cw_gblstsvm_systems

        cp = np.array([[0.0, 0.0], [1.0, 0.0]])
        cn = np.array([[-1.0, 0.0], [-2.0, 0.0]])
        rp = np.array([0.5, 1.0])
        rn = np.array([0.4, 0.8])
        w_pos = np.array([1.0, 0.5])
        w_neg = np.array([0.7, 0.9])

        opp_w1, _, _, _ = solve_cw_gblstsvm_systems(
            cp, rp, cn, rn, 5.0, 0.1, w_pos, w_neg, weight_placement="opposite_only"
        )
        both_w1, _, _, _ = solve_cw_gblstsvm_systems(
            cp, rp, cn, rn, 5.0, 0.1, w_pos, w_neg, weight_placement="both"
        )
        self.assertFalse(np.allclose(opp_w1, both_w1))

    def test_prediction_rules_supported(self) -> None:
        for rule in ("reference", "paper"):
            params = {k: v for k, v in self.params.items() if k != "prediction_rule"}
            model = RGBLSTSVM(**params, prediction_rule=rule).fit(self.X, self.y)
            preds = model.predict(self.X)
            self.assertTrue(np.all(np.isin(preds, [-1.0, 1.0])))

    def test_no_class_weighting(self) -> None:
        model = RGBLSTSVM(**self.params).fit(self.X, self.y)
        self.assertFalse(hasattr(model, "kappa_pos_") and model.kappa_pos_ is not None)


class TestRGBLSTSVMSolverEquivalence(unittest.TestCase):
    def test_solver_matches_baseline_with_ones(self) -> None:
        np.random.seed(5)
        X = np.random.randn(50, 3)
        y = np.sign(X[:, 0])
        y[y == 0] = 1
        balls = generate_granular_balls(X, y, purity_threshold=0.95, min_ball_count=2)
        from src.granular_balls import balls_to_class_arrays

        cp, rp, cn, rn = balls_to_class_arrays(balls)
        m1, m2 = cp.shape[0], cn.shape[0]
        ones_pos = np.ones(m1)
        ones_neg = np.ones(m2)

        bw1, bb1, bw2, bb2 = solve_gblstsvm_systems(cp, rp, cn, rn, 10.0, 0.1)
        rw1, rb1, rw2, rb2 = solve_cw_gblstsvm_systems(
            cp, rp, cn, rn, 10.0, 0.1, ones_pos, ones_neg
        )
        self.assertTrue(np.allclose(bw1, rw1, atol=1e-10))
        self.assertTrue(np.allclose(bw2, rw2, atol=1e-10))


if __name__ == "__main__":
    unittest.main()
