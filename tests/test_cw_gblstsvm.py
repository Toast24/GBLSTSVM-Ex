"""Tests for class-weighted CW-GBLSTSVM."""

import unittest

import numpy as np

from src.class_weights import (
    ball_class_weights,
    compute_binary_class_weights,
)
from src.cw_gblstsvm import CWGBLSTSVM, _weighted_gram, solve_cw_gblstsvm_systems
from src.gblstsvm import GBLSTSVM, solve_gblstsvm_systems
from src.granular_balls import generate_granular_balls


class TestClassWeights(unittest.TestCase):
    def test_balanced_kappa_equals_one(self) -> None:
        y = np.array([1, 1, -1, -1, 1, -1])
        k_pos, k_neg, n_pos, n_neg, n_train = compute_binary_class_weights(y)
        self.assertEqual(n_train, 6)
        self.assertEqual(n_pos, 3)
        self.assertEqual(n_neg, 3)
        self.assertAlmostEqual(k_pos, 1.0)
        self.assertAlmostEqual(k_neg, 1.0)

    def test_imbalanced_minority_larger_kappa(self) -> None:
        y = np.array([1, -1, -1, -1])
        k_pos, k_neg, _, _, _ = compute_binary_class_weights(y)
        self.assertGreater(k_pos, k_neg)
        self.assertAlmostEqual(k_pos, 2.0)
        self.assertAlmostEqual(k_neg, 4.0 / 6.0)

    def test_counts_from_original_samples_not_balls(self) -> None:
        np.random.seed(0)
        X = np.random.randn(40, 3)
        y = np.ones(40)
        y[:10] = -1

        k_pos, k_neg, n_pos, n_neg, n_train = compute_binary_class_weights(y)
        self.assertEqual(n_pos, 30)
        self.assertEqual(n_neg, 10)
        self.assertEqual(n_train, 40)

        balls = generate_granular_balls(
            X, y, purity_threshold=0.9, min_ball_count=2
        )
        n_pos_balls = sum(1 for b in balls if b.label == 1)
        n_neg_balls = sum(1 for b in balls if b.label != 1)
        self.assertNotEqual(n_pos, n_pos_balls)

        w_pos, w_neg = ball_class_weights(balls, k_pos, k_neg)
        self.assertTrue(np.all(w_pos == k_pos))
        self.assertTrue(np.all(w_neg == k_neg))

    def test_ball_weights_match_class_label(self) -> None:
        from src.granular_balls import BallInfo

        balls = [
            BallInfo(center=np.zeros(2), radius=0.1, label=1.0, purity=1.0, num_samples=5),
            BallInfo(center=np.ones(2), radius=0.2, label=-1.0, purity=0.9, num_samples=20),
        ]
        w_pos, w_neg = ball_class_weights(balls, kappa_pos=2.0, kappa_neg=0.5)
        self.assertEqual(len(w_pos), 1)
        self.assertEqual(len(w_neg), 1)
        self.assertAlmostEqual(w_pos[0], 2.0)
        self.assertAlmostEqual(w_neg[0], 0.5)


class TestCWOptimization(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        np.random.seed(99)
        cls.X = np.random.randn(50, 3)
        cls.y = np.sign(cls.X[:, 0] + cls.X[:, 1])
        cls.y[cls.y == 0] = 1
        cls.c1 = 10.0
        cls.c2 = 0.1
        cls.purity = 0.95
        cls.min_ball_count = 2

    def _ball_arrays(self):
        balls = generate_granular_balls(
            self.X,
            self.y,
            purity_threshold=self.purity,
            min_ball_count=self.min_ball_count,
        )
        from src.granular_balls import balls_to_class_arrays

        return balls, balls_to_class_arrays(balls)

    def test_unit_weights_match_frozen_solver(self) -> None:
        balls, (cp, rp, cn, rn) = self._ball_arrays()
        m1 = cp.shape[0]
        m2 = cn.shape[0]
        ones_pos = np.ones(m1)
        ones_neg = np.ones(m2)

        w1, b1, w2, b2 = solve_cw_gblstsvm_systems(
            cp, rp, cn, rn, self.c1, self.c2, ones_pos, ones_neg
        )
        fw1, fb1, fw2, fb2 = solve_gblstsvm_systems(
            cp, rp, cn, rn, self.c1, self.c2
        )
        self.assertTrue(np.allclose(w1, fw1, atol=1e-10))
        self.assertTrue(np.allclose(w2, fw2, atol=1e-10))
        self.assertAlmostEqual(b1, fb1, places=10)
        self.assertAlmostEqual(b2, fb2, places=10)

    def test_weighted_normal_equations_opposite_only(self) -> None:
        balls, (cp, rp, cn, rn) = self._ball_arrays()
        k_pos, k_neg, _, _, _ = compute_binary_class_weights(self.y)
        w_pos, w_neg = ball_class_weights(balls, k_pos, k_neg)

        mat_e1 = np.ones((cp.shape[0], 1))
        mat_e2 = np.ones((cn.shape[0], 1))
        mat_H = np.column_stack((cp, mat_e1))
        mat_G = np.column_stack((cn, mat_e2))
        R_pos = rp.reshape(-1, 1)
        R_neg = rn.reshape(-1, 1)
        eye = 1e-4 * np.identity(mat_H.shape[1])

        gram_hp1 = _weighted_gram(mat_G, w_neg) + (1.0 / self.c1) * _weighted_gram(
            mat_H, np.ones(cp.shape[0])
        )
        e2_R_neg = mat_e2 + R_neg
        rhs_1 = -np.linalg.inv(gram_hp1 + eye) @ np.concatenate(
            [
                (cn.T @ (w_neg[:, None] * e2_R_neg)).ravel(),
                (mat_e2.T @ (w_neg[:, None] * e2_R_neg)).ravel(),
            ]
        ).reshape(-1, 1)

        w1, b1, _, _ = solve_cw_gblstsvm_systems(
            cp, rp, cn, rn, self.c1, self.c2, w_pos, w_neg
        )
        theta1 = np.concatenate([w1, [b1]])
        self.assertTrue(np.allclose(theta1, rhs_1.ravel(), atol=1e-8))

    def test_balanced_cw_reduces_to_baseline(self) -> None:
        n = 60
        X = np.random.randn(n, 4)
        y = np.array([1] * (n // 2) + [-1] * (n // 2))

        params = {
            "c1": 50.0,
            "c2": 0.01,
            "purity_threshold": 0.95,
            "min_ball_count": 2,
            "prediction_rule": "reference",
            "random_state": 3,
        }
        base = GBLSTSVM(**params).fit(X, y)
        cw = CWGBLSTSVM(**params).fit(X, y)

        self.assertAlmostEqual(cw.kappa_pos_, 1.0)
        self.assertAlmostEqual(cw.kappa_neg_, 1.0)
        self.assertTrue(np.allclose(base.w1_, cw.w1_, atol=1e-10))
        self.assertTrue(np.allclose(base.w2_, cw.w2_, atol=1e-10))
        self.assertAlmostEqual(base.b1_, cw.b1_, places=10)
        self.assertAlmostEqual(base.b2_, cw.b2_, places=10)
        self.assertTrue(np.allclose(base.predict(X), cw.predict(X)))


class TestCWGBLSTSVMModel(unittest.TestCase):
    def test_prediction_rules_supported(self) -> None:
        np.random.seed(1)
        X = np.random.randn(40, 3)
        y = np.array([1] * 30 + [-1] * 10)

        for rule in ("reference", "paper"):
            model = CWGBLSTSVM(
                c1=10.0,
                c2=0.1,
                purity_threshold=0.9,
                min_ball_count=2,
                prediction_rule=rule,
                random_state=0,
            )
            model.fit(X, y)
            preds = model.predict(X)
            self.assertTrue(np.all(np.isin(preds, [-1.0, 1.0])))

    def test_imbalanced_changes_hyperplanes_vs_baseline(self) -> None:
        np.random.seed(5)
        X = np.random.randn(80, 4)
        X[:20, 0] += 2.0
        X[20:, 0] -= 2.0
        y = np.array([1] * 20 + [-1] * 60)

        params = {
            "c1": 20.0,
            "c2": 0.05,
            "purity_threshold": 0.85,
            "min_ball_count": 2,
            "random_state": 2,
        }
        base = GBLSTSVM(**params).fit(X, y)
        cw = CWGBLSTSVM(**params).fit(X, y)

        self.assertGreater(cw.kappa_pos_, cw.kappa_neg_)
        changed = (
            not np.allclose(base.w1_, cw.w1_)
            or not np.allclose(base.w2_, cw.w2_)
        )
        self.assertTrue(changed)


if __name__ == "__main__":
    unittest.main()
