"""Unit tests for Selective Abstention and Conformal Risk Control."""

import unittest
import numpy as np
from src.calibration.selective import SelectiveAbstentionController

class TestSelectiveAbstention(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # Synthetic evaluation setup
        self.confidences = np.array([0.95, 0.90, 0.85, 0.70, 0.60, 0.40, 0.20, 0.10])
        # High confidence mostly correct, low confidence mostly incorrect
        self.is_correct = np.array([True, True, True, False, True, False, False, False])
        # Solver flags: detects contradictions when incorrect
        self.flagged_contradicted = np.array([False, False, False, True, False, True, True, True])

    def test_monotonic_coverage(self):
        curve_df = SelectiveAbstentionController.compute_tradeoff_curve(
            self.confidences,
            self.is_correct,
            self.flagged_contradicted,
            thresholds=[0.1, 0.3, 0.5, 0.7, 0.85, 0.95]
        )
        coverages = curve_df["coverage"].tolist()
        for i in range(len(coverages) - 1):
            self.assertGreaterEqual(
                coverages[i],
                coverages[i + 1],
                f"Coverage should monotonically decrease as threshold increases: {coverages[i]} < {coverages[i+1]}"
            )

    def test_fit_threshold_achieves_target_risk(self):
        controller = SelectiveAbstentionController(target_risk=0.0)
        tau = controller.fit_threshold(self.confidences, self.is_correct, metric="error_rate")
        
        # When tau is applied to this set, retained error rate should be 0.0
        retained, cov = controller.predict(self.confidences, threshold=tau)
        self.assertGreater(cov, 0.0)
        retained_labels = self.is_correct[retained]
        self.assertTrue(np.all(retained_labels), "Target risk 0.0 must yield 100% precision on retained items")

    def test_abstention_boundary_conditions(self):
        controller = SelectiveAbstentionController()
        # tau = 0.0 retains all
        ret_all, cov_all = controller.predict(self.confidences, threshold=0.0)
        self.assertEqual(cov_all, 1.0)
        self.assertEqual(len(ret_all), len(self.confidences))

        # tau = 1.0 abstains from all when max conf < 1.0
        ret_none, cov_none = controller.predict(self.confidences, threshold=1.0)
        self.assertEqual(cov_none, 0.0)

    def test_tradeoff_curve_columns(self):
        curve_df = SelectiveAbstentionController.compute_tradeoff_curve(
            self.confidences,
            self.is_correct,
            self.flagged_contradicted
        )
        expected_cols = {"threshold", "coverage", "abstention_rate", "accuracy", "sfar", "f1"}
        self.assertTrue(expected_cols.issubset(set(curve_df.columns)))
        self.assertGreater(len(curve_df), 0)

if __name__ == "__main__":
    unittest.main()
