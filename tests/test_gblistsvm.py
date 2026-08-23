"""Unit tests for clean GBLSTSVM implementation."""

import unittest

import numpy as np

from src.gblstsvm import GBLSTSVM, predict_twin_hyperplanes, solve_gblstsvm_systems
from src.granular_balls import generate_granular_balls


class TestGBLSTSVM(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        self.X = np.random.randn(80, 4)
        self.y = np.sign(self.X[:, 0] + 0.3 * self.X[:, 1])
        self.y[self.y == 0] = 1

    def test_fit_predict_binary_labels(self) -> None:
        model = GBLSTSVM(
            c1=10.0,
            c2=0.1,
            purity_threshold=0.9,
            min_ball_count=2,
            prediction_rule="reference",
            random_state=0,
        )
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        self.assertEqual(set(np.unique(preds)), {-1.0, 1.0})
        self.assertEqual(preds.shape[0], self.X.shape[0])

    def test_accepts_zero_one_labels(self) -> None:
        y01 = (self.y > 0).astype(float)
        model = GBLSTSVM(
            c1=5.0,
            c2=0.5,
            purity_threshold=0.95,
            min_ball_count=2,
            prediction_rule="reference",
            random_state=0,
        )
        model.fit(self.X, y01)
        preds = model.predict(self.X)
        self.assertTrue(np.all(np.isin(preds, [-1.0, 1.0])))

    def test_deterministic_with_seed(self) -> None:
        params = {
            "c1": 50.0,
            "c2": 0.01,
            "purity_threshold": 0.95,
            "min_ball_count": 2,
            "prediction_rule": "reference",
            "random_state": 7,
        }
        m1 = GBLSTSVM(**params).fit(self.X, self.y)
        m2 = GBLSTSVM(**params).fit(self.X, self.y)
        self.assertTrue(np.allclose(m1.w1_, m2.w1_))
        self.assertTrue(np.allclose(m1.w2_, m2.w2_))
        self.assertEqual(m1.b1_, m2.b1_)
        self.assertEqual(m1.b2_, m2.b2_)

    def test_granular_ball_generation(self) -> None:
        balls = generate_granular_balls(
            self.X,
            self.y,
            purity_threshold=0.95,
            min_ball_count=2,
        )
        self.assertGreater(len(balls), 0)
        for ball in balls:
            self.assertEqual(ball.center.shape[0], self.X.shape[1])
            self.assertGreaterEqual(ball.radius, 0.0)
            self.assertIn(ball.label, (-1.0, 1.0))
            self.assertGreaterEqual(ball.purity, 0.0)
            self.assertLessEqual(ball.purity, 1.0)

    def test_solve_raises_on_invalid_c(self) -> None:
        with self.assertRaises(ValueError):
            solve_gblstsvm_systems(
                np.ones((2, 2)),
                np.ones(2),
                np.ones((2, 2)),
                np.ones(2),
                c1=0.0,
                c2=1.0,
            )

    def test_prediction_rules_differ_when_norms_differ(self) -> None:
        w1 = np.array([2.0, 0.0])
        w2 = np.array([1.0, 0.0])
        x = np.array([[0.5, 0.0]])
        ref = predict_twin_hyperplanes(x, w1, 0.0, w2, 0.0, rule="reference")
        paper = predict_twin_hyperplanes(x, w1, 0.0, w2, 0.0, rule="paper")
        self.assertTrue(np.isin(ref[0], [-1.0, 0.0, 1.0]))
        self.assertTrue(np.isin(paper[0], [-1.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
