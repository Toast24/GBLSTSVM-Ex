"""Unit tests for SupportAware-Topo (original Topo left unchanged)."""

import unittest

import numpy as np

from src.topology_gblstsvm import SupportAwareTopoGBLSTSVM, TopoGBLSTSVM
from src.topology_membership import (
    adjust_r_opp,
    compute_support_aware_topology_membership,
    compute_topology_membership,
    support_confidence,
)


class TestSupportConfidence(unittest.TestCase):
    def test_limits(self) -> None:
        n = np.array([0, 1, 10, 100], dtype=float)
        c = support_confidence(n)
        self.assertEqual(c[0], 0.0)
        self.assertAlmostEqual(c[1], 0.5)
        self.assertGreater(c[3], c[2])
        self.assertGreater(c[2], c[1])
        self.assertTrue(np.all(c >= 0.0) and np.all(c <= 1.0))

    def test_adjust_r_opp_no_complete_suppression(self) -> None:
        r = np.array([0.0, 0.0, 0.0, 1.0])
        n = np.array([0, 1, 20, 20])
        r_s = adjust_r_opp(r, n)
        # Weak support + R_opp=0 must not zero membership.
        self.assertAlmostEqual(r_s[0], 1.0)
        self.assertAlmostEqual(r_s[1], 0.5)
        self.assertGreater(r_s[1], 0.0)
        # Strong support trusts R_opp.
        self.assertLess(r_s[2], 0.1)
        self.assertGreater(r_s[3], 0.95)

    def test_high_n_opp_tracks_original(self) -> None:
        r = np.array([0.3, 0.8])
        n = np.array([50.0, 50.0])
        r_s = adjust_r_opp(r, n)
        self.assertTrue(np.allclose(r_s, r, atol=0.03))


class TestOriginalTopoUnchanged(unittest.TestCase):
    def test_mu_still_product(self) -> None:
        rng = np.random.default_rng(0)
        centers = rng.normal(size=(40, 3))
        labels = np.array([1.0] * 15 + [-1.0] * 25)
        result = compute_topology_membership(centers, labels)
        self.assertTrue(np.allclose(result.mu_topo, result.ambiguity * result.r_opp))
        self.assertEqual(type(result).__name__, "TopologyMembershipResult")
        self.assertFalse(hasattr(result, "r_opp_adjusted"))

    def test_n_opp_zero_original_still_zero(self) -> None:
        centers = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.2, 0.0],
                [10.0, 0.0],
            ]
        )
        labels = np.array([1.0, 1.0, 1.0, -1.0])
        orig = compute_topology_membership(centers, labels, k=2)
        sa = compute_support_aware_topology_membership(centers, labels, k=2)
        self.assertEqual(orig.n_opp[0], 0)
        self.assertEqual(orig.r_opp[0], 0.0)
        self.assertEqual(orig.mu_topo[0], 0.0)
        self.assertAlmostEqual(sa.r_opp_adjusted[0], 1.0)
        self.assertAlmostEqual(sa.mu_topo_s[0], sa.ambiguity[0])
        self.assertGreater(sa.mu_topo_s[0], orig.mu_topo[0] - 1e-12)


class TestSupportAwareMembership(unittest.TestCase):
    def test_formula(self) -> None:
        rng = np.random.default_rng(3)
        centers = rng.normal(size=(30, 2))
        labels = np.array([1.0] * 12 + [-1.0] * 18)
        sa = compute_support_aware_topology_membership(centers, labels)
        c = sa.n_opp.astype(float) / (sa.n_opp.astype(float) + 1.0)
        expected = c * sa.r_opp + (1.0 - c)
        self.assertTrue(np.allclose(sa.r_opp_adjusted, expected, atol=1e-12))
        self.assertTrue(np.allclose(sa.mu_topo_s, sa.ambiguity * sa.r_opp_adjusted))
        self.assertTrue(np.allclose(sa.mu_topo, sa.ambiguity * sa.r_opp))

    def test_r_opp_zero_weak_support_keeps_ca(self) -> None:
        # Isolated + ball: no opposite neighbors in k=1.
        centers = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [10.0, 0.0],
                [10.1, 0.0],
            ]
        )
        labels = np.array([1.0, 1.0, -1.0, -1.0])
        sa = compute_support_aware_topology_membership(centers, labels, k=1)
        weak = sa.n_opp == 0
        self.assertTrue(np.any(weak))
        self.assertTrue(np.allclose(sa.r_opp_adjusted[weak], 1.0))
        self.assertTrue(np.allclose(sa.mu_topo_s[weak], sa.ambiguity[weak]))
        self.assertTrue(np.allclose(sa.mu_topo[weak], 0.0))


class TestSupportAwareModel(unittest.TestCase):
    def test_fit_and_predict(self) -> None:
        np.random.seed(5)
        X = np.random.randn(60, 3)
        X[:20, 0] += 1.5
        X[20:, 0] -= 1.5
        y = np.array([1.0] * 20 + [-1.0] * 40)
        params = {
            "c1": 20.0,
            "c2": 0.05,
            "purity_threshold": 0.9,
            "min_ball_count": 2,
            "prediction_rule": "reference",
            "random_state": 3,
        }
        topo = TopoGBLSTSVM(**params).fit(X, y)
        sa = SupportAwareTopoGBLSTSVM(**params).fit(X, y)
        self.assertTrue(np.all(np.isin(sa.predict(X), [-1.0, 1.0])))
        self.assertIsNotNone(sa.membership_.r_opp_adjusted)
        # Same granulation/seed: original R_opp matches; adjusted differs when n_opp is small.
        self.assertTrue(np.allclose(sa.membership_.r_opp, topo.membership_.r_opp))
        self.assertEqual(sa.weights_pos_.shape[0], sa.centers_pos_.shape[0])


if __name__ == "__main__":
    unittest.main()
