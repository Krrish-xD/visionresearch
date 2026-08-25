"""Unit tests for calibration methods and metrics."""

import unittest
import numpy as np
from src.calibration.temperature import TemperatureScaling
from src.calibration.isotonic import IsotonicCalibrator
from src.calibration.conformal import ConformalRiskWeighting
from src.calibration.metrics import compute_ece, compute_brier_score, compute_nll

class TestCalibration(unittest.TestCase):

    def setUp(self):
        # Simulated overconfident raw confidences
        self.confidences = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
        # Actual accuracy is only 60%
        self.labels = [True, True, True, False, True, False, True, False, True, False]

    def test_temperature_scaling(self):
        temp = TemperatureScaling()
        temp.fit(self.confidences, self.labels)
        self.assertGreater(temp.temperature, 1.0)  # Overconfident -> T > 1
        transformed = temp.transform(self.confidences)
        self.assertEqual(len(transformed), len(self.confidences))
        # Scaled confidences should be softened
        self.assertLess(transformed[0], self.confidences[0])

    def test_isotonic_regression(self):
        iso = IsotonicCalibrator()
        iso.fit(self.confidences, self.labels)
        transformed = iso.transform(self.confidences)
        self.assertEqual(len(transformed), len(self.confidences))
        # Monotonicity check
        for i in range(len(transformed) - 1):
            self.assertGreaterEqual(transformed[i] + 1e-6, transformed[i + 1])

    def test_conformal_risk(self):
        conf = ConformalRiskWeighting(alpha=0.10)
        conf.fit(self.confidences, self.labels)
        weights = conf.transform(self.confidences)
        self.assertEqual(len(weights), len(self.confidences))
        self.assertTrue(np.all(weights >= 0.05))
        self.assertTrue(np.all(weights <= 1.0))

    def test_calibration_metrics(self):
        ece, bin_data = compute_ece(self.confidences, self.labels, num_bins=5)
        self.assertGreater(ece, 0.0)
        brier = compute_brier_score(self.confidences, self.labels)
        self.assertGreater(brier, 0.0)
        nll = compute_nll(self.confidences, self.labels)
        self.assertGreater(nll, 0.0)

if __name__ == "__main__":
    unittest.main()
