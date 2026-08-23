"""Numerical agreement between reference and clean GBLSTSVM implementations."""

import unittest

import numpy as np

from src.gblstsvm import GBLSTSVM, solve_gblstsvm_systems
from src.granular_balls import generate_granular_balls, balls_to_class_arrays
from src.reference_pipeline import reference_gen_balls


class TestReferenceComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        np.random.seed(123)
        cls.X = np.random.randn(60, 3)
        cls.y = np.sign(cls.X[:, 0] + cls.X[:, 1] - 0.2)
        cls.y[cls.y == 0] = 1
        cls.purity = 0.95
        cls.min_ball_count = 2
        cls.c1 = 100.0
        cls.c2 = 0.01

    def test_granular_balls_match_reference(self) -> None:
        np.random.seed(123)
        clean_balls = generate_granular_balls(
            self.X,
            self.y,
            purity_threshold=self.purity,
            min_ball_count=self.min_ball_count,
        )
        np.random.seed(123)
        ref_balls = reference_gen_balls(
            self.X,
            self.y,
            self.purity,
            self.min_ball_count,
        )

        self.assertEqual(len(clean_balls), len(ref_balls))

        clean_sorted = sorted(
            clean_balls,
            key=lambda b: (b.label, b.center[0], b.center[1], b.radius),
        )
        ref_sorted = sorted(
            ref_balls,
            key=lambda b: (b.label, b.center[0], b.center[1], b.radius),
        )

        for cb, rb in zip(clean_sorted, ref_sorted):
            self.assertEqual(cb.label, rb.label)
            self.assertTrue(np.allclose(cb.center, rb.center, atol=1e-10))
            self.assertAlmostEqual(cb.radius, rb.radius, places=10)
            self.assertAlmostEqual(cb.purity, rb.purity, places=10)

    def test_hyperplanes_match_reference(self) -> None:
        np.random.seed(123)
        clean_balls = generate_granular_balls(
            self.X,
            self.y,
            purity_threshold=self.purity,
            min_ball_count=self.min_ball_count,
        )
        cp, rp, cn, rn = balls_to_class_arrays(clean_balls)

        w1, b1, w2, b2 = solve_gblstsvm_systems(
            cp, rp, cn, rn, self.c1, self.c2
        )

        np.random.seed(123)
        ref_balls = reference_gen_balls(
            self.X, self.y, self.purity, self.min_ball_count
        )
        pos = [b for b in ref_balls if b.label == 1]
        neg = [b for b in ref_balls if b.label != 1]
        A = np.vstack([b.center for b in pos])
        B = np.vstack([b.center for b in neg])
        R_pos = np.array([b.radius for b in pos])
        R_neg = np.array([b.radius for b in neg])

        rw1, rb1, rw2, rb2 = solve_gblstsvm_systems(
            A, R_pos, B, R_neg, self.c1, self.c2
        )

        self.assertTrue(np.allclose(w1, rw1, atol=1e-8))
        self.assertTrue(np.allclose(w2, rw2, atol=1e-8))
        self.assertAlmostEqual(b1, rb1, places=8)
        self.assertAlmostEqual(b2, rb2, places=8)

    def test_predictions_match_reference_rule(self) -> None:
        model = GBLSTSVM(
            c1=self.c1,
            c2=self.c2,
            purity_threshold=self.purity,
            min_ball_count=self.min_ball_count,
            prediction_rule="reference",
            random_state=123,
        )
        model.fit(self.X, self.y)
        clean_pred = model.predict(self.X)

        np.random.seed(123)
        ref_balls = reference_gen_balls(
            self.X, self.y, self.purity, self.min_ball_count
        )
        pos = [b for b in ref_balls if b.label == 1]
        neg = [b for b in ref_balls if b.label != 1]
        w1, b1, w2, b2 = solve_gblstsvm_systems(
            np.vstack([b.center for b in pos]),
            np.array([b.radius for b in pos]),
            np.vstack([b.center for b in neg]),
            np.array([b.radius for b in neg]),
            self.c1,
            self.c2,
        )
        y1 = self.X @ w1 + b1
        y2 = self.X @ w2 + b2
        ref_pred = np.sign(np.abs(y2) - np.abs(y1))

        self.assertTrue(np.allclose(clean_pred, ref_pred))


if __name__ == "__main__":
    unittest.main()
