"""Tests for class- and reliability-weighted CR-GBLSTSVM."""



import unittest



import numpy as np



from src.class_weights import ball_class_weights, compute_binary_class_weights

from src.cr_gblstsvm import (

    CRGBLSTSVM,

    ball_cr_weights,

    solve_cr_gblstsvm_systems,

)

from src.cw_gblstsvm import CWGBLSTSVM, solve_cw_gblstsvm_systems

from src.gblstsvm import GBLSTSVM, solve_gblstsvm_systems

from src.granular_balls import BallInfo, generate_granular_balls

from src.r_gblstsvm import RGBLSTSVM

from src.reliability import ball_reliability_weights





class TestCRWeightConstruction(unittest.TestCase):

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



    def test_exact_weight_product(self) -> None:

        w_pos, w_neg, _, rel, k_pos, k_neg = ball_cr_weights(

            self.balls, self.kappa_pos, self.kappa_neg

        )

        rel_pos, rel_neg, _, _ = ball_reliability_weights(self.balls)

        self.assertTrue(np.allclose(w_pos, k_pos * rel_pos))

        self.assertTrue(np.allclose(w_neg, k_neg * rel_neg))



        pos_idx = [i for i, b in enumerate(self.balls) if b.label == 1]

        neg_idx = [i for i, b in enumerate(self.balls) if b.label != 1]

        for i, idx in enumerate(pos_idx):

            expected = self.kappa_pos * rel[idx]

            self.assertAlmostEqual(w_pos[i], expected, places=12)

        for i, idx in enumerate(neg_idx):

            expected = self.kappa_neg * rel[idx]

            self.assertAlmostEqual(w_neg[i], expected, places=12)



    def test_class_weights_from_original_training_counts(self) -> None:

        k_pos, k_neg, n_pos, n_neg, n_train = compute_binary_class_weights(self.y)

        self.assertEqual(n_pos, 20)

        self.assertEqual(n_neg, 40)

        self.assertEqual(n_train, 60)

        _, _, _, _, k_pos_balls, k_neg_balls = ball_cr_weights(

            self.balls, k_pos, k_neg

        )

        self.assertTrue(np.all(k_pos_balls == k_pos))

        self.assertTrue(np.all(k_neg_balls == k_neg))



    def test_reliability_computed_independently(self) -> None:

        _, _, _, rel_cr, _, _ = ball_cr_weights(

            self.balls, self.kappa_pos, self.kappa_neg

        )

        _, _, _, rel_r = ball_reliability_weights(self.balls)

        self.assertTrue(np.allclose(rel_cr, rel_r))



    def test_no_ball_size_multiplier(self) -> None:

        from src.granular_balls import BallInfo



        balls = [

            BallInfo(

                center=np.zeros(2),

                radius=0.1,

                label=1.0,

                purity=0.9,

                num_samples=100,

            ),

            BallInfo(

                center=np.ones(2),

                radius=0.2,

                label=-1.0,

                purity=0.8,

                num_samples=5,

            ),

        ]

        w_pos, w_neg, _, rel, k_pos, k_neg = ball_cr_weights(balls, 2.0, 0.5)

        self.assertAlmostEqual(w_pos[0], 2.0 * rel[0])

        self.assertAlmostEqual(w_neg[0], 0.5 * rel[1])

        self.assertNotAlmostEqual(w_pos[0], 2.0 * rel[0] * 100)





class TestCRReductions(unittest.TestCase):

    @classmethod

    def setUpClass(cls) -> None:

        np.random.seed(21)

        cls.X = np.random.randn(80, 4)

        cls.X[:30, 0] += 1.5

        cls.X[30:, 0] -= 1.5

        cls.y_imbalanced = np.array([1.0] * 30 + [-1.0] * 50)

        cls.y_balanced = np.array([1.0] * 40 + [-1.0] * 40)

        cls.params = {

            "c1": 20.0,

            "c2": 0.05,

            "purity_threshold": 0.9,

            "min_ball_count": 2,

            "prediction_rule": "reference",

            "random_state": 7,

        }



    def test_cr_reduces_to_cw_when_reliability_one(self) -> None:

        """a_i = 1 for all balls => CR == CW."""

        cw_params = self.params

        cr_params = {**self.params, "gamma": 0.0, "lambda_r": 0.0}



        cw = CWGBLSTSVM(**cw_params).fit(self.X, self.y_imbalanced)

        cr = CRGBLSTSVM(**cr_params).fit(self.X, self.y_imbalanced)



        self.assertTrue(np.allclose(cr.reliabilities_, 1.0))

        self.assertTrue(np.allclose(cw.weights_pos_, cr.kappa_pos_balls_))

        self.assertTrue(np.allclose(cw.weights_neg_, cr.kappa_neg_balls_))

        self.assertTrue(np.allclose(cw.w1_, cr.w1_, atol=1e-10))

        self.assertTrue(np.allclose(cw.w2_, cr.w2_, atol=1e-10))

        self.assertAlmostEqual(cw.b1_, cr.b1_, places=10)

        self.assertAlmostEqual(cw.b2_, cr.b2_, places=10)

        self.assertTrue(np.allclose(cw.predict(self.X), cr.predict(self.X)))



    def test_cr_reduces_to_r_when_balanced(self) -> None:

        """kappa_pos = kappa_neg = 1 on balanced data => CR == R."""

        cr = CRGBLSTSVM(**self.params).fit(self.X, self.y_balanced)

        r_model = RGBLSTSVM(**self.params).fit(self.X, self.y_balanced)



        self.assertAlmostEqual(cr.kappa_pos_, 1.0)

        self.assertAlmostEqual(cr.kappa_neg_, 1.0)

        self.assertTrue(np.allclose(cr.reliabilities_, r_model.reliabilities_))

        self.assertTrue(np.allclose(cr.weights_pos_, r_model.weights_pos_, atol=1e-12))

        self.assertTrue(np.allclose(cr.weights_neg_, r_model.weights_neg_, atol=1e-12))

        self.assertTrue(np.allclose(r_model.w1_, cr.w1_, atol=1e-10))

        self.assertTrue(np.allclose(r_model.w2_, cr.w2_, atol=1e-10))

        self.assertAlmostEqual(r_model.b1_, cr.b1_, places=10)

        self.assertAlmostEqual(r_model.b2_, cr.b2_, places=10)

        self.assertTrue(np.allclose(r_model.predict(self.X), cr.predict(self.X)))



    def test_cr_reduces_to_baseline_when_both_disabled(self) -> None:

        """a_i = 1 and kappa = 1 => CR == frozen GBLSTSVM."""

        params = {

            **self.params,

            "gamma": 0.0,

            "lambda_r": 0.0,

        }

        base = GBLSTSVM(**self.params).fit(self.X, self.y_balanced)

        cr = CRGBLSTSVM(**params).fit(self.X, self.y_balanced)



        self.assertAlmostEqual(cr.kappa_pos_, 1.0)

        self.assertAlmostEqual(cr.kappa_neg_, 1.0)

        self.assertTrue(np.allclose(cr.reliabilities_, 1.0))

        self.assertTrue(np.allclose(base.w1_, cr.w1_, atol=1e-10))

        self.assertTrue(np.allclose(base.w2_, cr.w2_, atol=1e-10))

        self.assertAlmostEqual(base.b1_, cr.b1_, places=10)

        self.assertAlmostEqual(base.b2_, cr.b2_, places=10)

        self.assertTrue(np.allclose(base.predict(self.X), cr.predict(self.X)))



    def test_balanced_data_cr_equals_r(self) -> None:

        """Explicit balanced-data property: CR == R when n_pos == n_neg."""

        n = 60

        X = np.random.randn(n, 4)

        y = np.array([1.0] * (n // 2) + [-1.0] * (n // 2))

        params = {**self.params, "random_state": 11}



        cr = CRGBLSTSVM(**params).fit(X, y)

        r_model = RGBLSTSVM(**params).fit(X, y)



        self.assertAlmostEqual(cr.kappa_pos_, 1.0)

        self.assertAlmostEqual(cr.kappa_neg_, 1.0)

        self.assertTrue(np.allclose(cr.w1_, r_model.w1_, atol=1e-10))

        self.assertTrue(np.allclose(cr.w2_, r_model.w2_, atol=1e-10))

        self.assertTrue(np.allclose(cr.predict(X), r_model.predict(X)))





class TestCRWeightPlacement(unittest.TestCase):

    @classmethod

    def setUpClass(cls) -> None:

        np.random.seed(33)

        cls.X = np.random.randn(70, 4)

        cls.X[:25, 0] += 1.5

        cls.X[25:, 0] -= 1.5

        cls.y = np.array([1.0] * 25 + [-1.0] * 45)

        cls.params = {

            "c1": 15.0,

            "c2": 0.08,

            "purity_threshold": 0.9,

            "min_ball_count": 2,

            "prediction_rule": "reference",

            "random_state": 4,

        }



    def test_opposite_only_placement(self) -> None:

        model = CRGBLSTSVM(**self.params, weight_placement="opposite_only").fit(

            self.X, self.y

        )

        fw1, fb1, fw2, fb2 = solve_cr_gblstsvm_systems(

            model.centers_pos_,

            model.radii_pos_,

            model.centers_neg_,

            model.radii_neg_,

            self.params["c1"],

            self.params["c2"],

            model.weights_pos_,

            model.weights_neg_,

            weight_placement="opposite_only",

        )

        self.assertTrue(np.allclose(model.w1_, fw1))

        self.assertTrue(np.allclose(model.w2_, fw2))



    def test_both_term_placement(self) -> None:

        model = CRGBLSTSVM(**self.params, weight_placement="both").fit(self.X, self.y)

        fw1, fb1, fw2, fb2 = solve_cr_gblstsvm_systems(

            model.centers_pos_,

            model.radii_pos_,

            model.centers_neg_,

            model.radii_neg_,

            self.params["c1"],

            self.params["c2"],

            model.weights_pos_,

            model.weights_neg_,

            weight_placement="both",

        )

        self.assertTrue(np.allclose(model.w1_, fw1))

        self.assertTrue(np.allclose(model.w2_, fw2))



    def test_both_differs_from_opposite_only(self) -> None:

        opp = CRGBLSTSVM(**self.params, weight_placement="opposite_only").fit(

            self.X, self.y

        )

        both = CRGBLSTSVM(**self.params, weight_placement="both").fit(self.X, self.y)

        self.assertFalse(np.allclose(opp.w1_, both.w1_))





class TestCRGBLSTSVMModel(unittest.TestCase):

    def test_prediction_rules_supported(self) -> None:

        np.random.seed(2)

        X = np.random.randn(50, 3)

        y = np.array([1.0] * 15 + [-1.0] * 35)



        for rule in ("reference", "paper"):

            model = CRGBLSTSVM(

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



    def test_no_prediction_time_weighting(self) -> None:

        """Weights affect fit only; predict uses stored hyperplanes."""

        np.random.seed(8)

        X = np.random.randn(40, 3)

        y = np.array([1.0] * 12 + [-1.0] * 28)

        model = CRGBLSTSVM(

            c1=10.0,

            c2=0.1,

            purity_threshold=0.9,

            min_ball_count=2,

            random_state=0,

        ).fit(X, y)

        preds1 = model.predict(X)

        preds2 = model.predict(X)

        self.assertTrue(np.allclose(preds1, preds2))



    def test_deterministic_with_fixed_random_state(self) -> None:

        np.random.seed(99)

        X = np.random.randn(55, 3)

        y = np.array([1.0] * 18 + [-1.0] * 37)

        params = {

            "c1": 12.0,

            "c2": 0.05,

            "purity_threshold": 0.9,

            "min_ball_count": 2,

            "random_state": 42,

        }

        m1 = CRGBLSTSVM(**params).fit(X, y)

        m2 = CRGBLSTSVM(**params).fit(X, y)

        self.assertTrue(np.allclose(m1.w1_, m2.w1_))

        self.assertTrue(np.allclose(m1.w2_, m2.w2_))

        self.assertTrue(np.allclose(m1.predict(X), m2.predict(X)))



    def test_solver_matches_cw_with_combined_weights(self) -> None:

        np.random.seed(6)

        X = np.random.randn(50, 3)

        y = np.array([1.0] * 20 + [-1.0] * 30)

        balls = generate_granular_balls(X, y, purity_threshold=0.9, min_ball_count=2)

        from src.granular_balls import balls_to_class_arrays



        cp, rp, cn, rn = balls_to_class_arrays(balls)

        k_pos, k_neg, _, _, _ = compute_binary_class_weights(y)

        w_pos, w_neg, _, _, _, _ = ball_cr_weights(balls, k_pos, k_neg)



        cr_w1, cr_b1, cr_w2, cr_b2 = solve_cr_gblstsvm_systems(

            cp, rp, cn, rn, 10.0, 0.1, w_pos, w_neg

        )

        cw_w1, cw_b1, cw_w2, cw_b2 = solve_cw_gblstsvm_systems(

            cp, rp, cn, rn, 10.0, 0.1, w_pos, w_neg

        )

        self.assertTrue(np.allclose(cr_w1, cw_w1, atol=1e-10))

        self.assertTrue(np.allclose(cr_w2, cw_w2, atol=1e-10))



    def test_combined_weights_differ_from_r_on_imbalanced(self) -> None:
        """Imbalanced kappa factors make CR weights differ from R weights."""
        np.random.seed(10)
        X = np.random.randn(60, 3)
        X[:15, 0] += 2.0
        X[15:, 0] -= 2.0
        y = np.array([1.0] * 15 + [-1.0] * 45)
        params = {
            "c1": 10.0,
            "c2": 0.1,
            "purity_threshold": 0.9,
            "min_ball_count": 2,
            "random_state": 0,
        }
        cr = CRGBLSTSVM(**params).fit(X, y)
        r_model = RGBLSTSVM(**params).fit(X, y)

        self.assertGreater(cr.kappa_pos_, 1.0)
        self.assertFalse(np.allclose(cr.weights_pos_, r_model.weights_pos_))
        self.assertFalse(np.allclose(cr.weights_neg_, r_model.weights_neg_))

    def test_cr_differs_from_cw_when_reliability_below_one(self) -> None:
        """When any a_i < 1, CR final weights differ from CW kappa-only weights."""
        from src.granular_balls import BallInfo

        balls = [
            BallInfo(center=np.zeros(2), radius=0.5, label=1.0, purity=0.7, num_samples=10),
            BallInfo(center=np.ones(2), radius=0.8, label=-1.0, purity=0.6, num_samples=15),
        ]
        w_pos, w_neg, _, rel, k_pos, k_neg = ball_cr_weights(balls, 2.0, 0.5)
        self.assertTrue(np.any(rel < 1.0))
        self.assertFalse(np.allclose(w_pos, k_pos))
        self.assertFalse(np.allclose(w_neg, k_neg))





if __name__ == "__main__":

    unittest.main()

