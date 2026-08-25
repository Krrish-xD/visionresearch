"""Isotonic Regression Calibrator."""

import numpy as np
from sklearn.isotonic import IsotonicRegression as SklearnIsotonic
from typing import List, Union

class IsotonicCalibrator:
    """
    Non-parametric calibration via Isotonic Regression.
    Fits a monotonic mapping g: [0, 1] -> [0, 1] to minimize squared error.
    """
    def __init__(self):
        self.regressor = SklearnIsotonic(y_min=0.01, y_max=0.99, out_of_bounds="clip")
        self.is_fitted = False

    def fit(self, confidences: Union[List[float], np.ndarray], labels: Union[List[bool], np.ndarray]):
        """Fit isotonic model on calibration data."""
        conf_arr = np.array(confidences, dtype=np.float64)
        y = np.array(labels, dtype=np.float64)

        if len(conf_arr) < 10:
            # Fallback if too few samples
            self.is_fitted = False
            return

        self.regressor.fit(conf_arr, y)
        self.is_fitted = True

    def transform(self, confidences: Union[List[float], np.ndarray]) -> np.ndarray:
        """Transform raw confidences using fitted isotonic curve."""
        conf_arr = np.array(confidences, dtype=np.float64)
        if not self.is_fitted:
            return np.clip(conf_arr, 0.01, 0.99)
        calibrated = self.regressor.predict(conf_arr)
        return np.clip(calibrated, 0.01, 0.99)
