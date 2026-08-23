"""Tests for paper vs reference prediction diagnostics."""

import unittest

import numpy as np

from src.gblstsvm import GBLSTSVM, predict_twin_hyperplanes
from src.prediction_diagnostics import (
    compute_prediction_diagnostics,
    summarize_disagreements,
)


class TestPredictionDiscrepancy(unittest.TestCase):
    def test_reference_rule_matches_lstwgbsvm(self) -> None:
        """Reference rule reproduces GBLSTSVM.py sign(|y2|-|y1|) logic."""
        w1 = np.array([2.0, 0.0])
        w2 = np.array([1.0, 0.0])
        X = np.array([[0.3, 0.0], [1.0, 0.0], [0.5, 0.0]])
        b1, b2 = 0.0, 0.0

        expected = np.sign(np.abs(X @ w2) - np.abs(X @ w1))
        got = predict_twin_hyperplanes(X, w1, b1, w2, b2, rule="reference")
        self.assertTrue(np.allclose(got, expected))

    def test_paper_rule_uses_normalized_distances(self) -> None:
        w1 = np.array([2.0, 0.0])
        w2 = np.array([1.0, 0.0])
        X = np.array([[0.5, 0.0]])
        # |f1|=1.0, |f2|=0.5 -> reference +1 (|f2|<|f1| wrong: |f2|=0.5 < 1 -> sign negative)
        # d1=0.5, d2=0.5 -> paper +1 (d1<=d2)
        ref = predict_twin_hyperplanes(X, w1, 0.0, w2, 0.0, rule="reference")
        paper = predict_twin_hyperplanes(X, w1, 0.0, w2, 0.0, rule="paper")
        self.assertEqual(ref[0], -1.0)
        self.assertEqual(paper[0], 1.0)

    def test_both_rules_selectable_on_model(self) -> None:
        np.random.seed(0)
        X = np.random.randn(40, 3)
        y = np.sign(X[:, 0])
        y[y == 0] = 1
        base = {
            "c1": 10.0,
            "c2": 0.1,
            "purity_threshold": 0.9,
            "min_ball_count": 2,
            "random_state": 0,
        }
        m_ref = GBLSTSVM(**base, prediction_rule="reference").fit(X, y)
        m_paper = GBLSTSVM(**base, prediction_rule="paper").fit(X, y)
        self.assertTrue(np.allclose(m_ref.w1_, m_paper.w1_))
        self.assertTrue(np.allclose(m_ref.w2_, m_paper.w2_))

    def test_synthetic_discrepancy_reproducible(self) -> None:
        """Match compare_reference synthetic setup: ~29% disagreement expected."""
        np.random.seed(0)
        n, d = 100, 4
        X = np.random.randn(n, d)
        y = np.sign(X[:, 0] + 0.5 * X[:, 1] + 0.1 * np.random.randn(n))
        y[y == 0] = 1

        model = GBLSTSVM(
            c1=1000.0,
            c2=1e-5,
            purity_threshold=0.985,
            min_ball_count=2,
            random_state=0,
        ).fit(X, y)

        diag = compute_prediction_diagnostics(
            X, model.w1_, model.b1_, model.w2_, model.b2_
        )
        summary = summarize_disagreements(diag)

        self.assertEqual(summary["n_samples"], 100)
        self.assertGreater(summary["n_disagree"], 0)
        self.assertAlmostEqual(summary["disagree_rate"], 0.29, delta=0.05)
        self.assertNotEqual(summary["norm_w1"], summary["norm_w2"])

    def test_rules_equivalent_when_parallel_weights_zero_bias(self) -> None:
        w1 = np.array([1.0, 2.0])
        w2 = 2.0 * w1  # ||w2|| = 2||w1||, f2 = 2*f1 when b=0
        X = np.random.randn(50, 2)
        ref = predict_twin_hyperplanes(X, w1, 0.0, w2, 0.0, rule="reference")
        paper = predict_twin_hyperplanes(X, w1, 0.0, w2, 0.0, rule="paper")
        # Equal normalized distances -> paper always +1; reference +1 unless |f1|=0.
        mask = ref != 0
        self.assertTrue(np.all(ref[mask] == paper[mask]))


if __name__ == "__main__":
    unittest.main()
