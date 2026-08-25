"""Temperature Scaling Calibrator."""

import numpy as np
from scipy.optimize import minimize
from typing import List, Union

class TemperatureScaling:
    """
    Parametric calibration via Temperature Scaling.
    Fits scalar T > 0 on calibration split by minimizing Negative Log-Likelihood.
    """
    def __init__(self):
        self.temperature: float = 1.0

    def fit(self, confidences: Union[List[float], np.ndarray], labels: Union[List[bool], np.ndarray]):
        """
        Fit temperature parameter T on calibration items.
        
        confidences: Raw probabilities in (0, 1]
        labels: Boolean array indicating whether prediction was correct (1) or incorrect (0)
        """
        conf_arr = np.clip(np.array(confidences, dtype=np.float64), 1e-5, 1.0 - 1e-5)
        # Convert probabilities back to pseudo-logits: logit = log(p / (1 - p))
        logits = np.log(conf_arr / (1.0 - conf_arr))
        y = np.array(labels, dtype=np.float64)

        def nll_loss(T_arr):
            T = max(1e-3, float(T_arr[0]))
            scaled_logits = logits / T
            # Sigmoid cross entropy
            probs = 1.0 / (1.0 + np.exp(-scaled_logits))
            probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
            loss = -np.mean(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs))
            return loss

        res = minimize(nll_loss, x0=[1.0], bounds=[(0.01, 20.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0]) if res.success else 1.0

    def transform(self, confidences: Union[List[float], np.ndarray]) -> np.ndarray:
        """Apply fitted temperature scaling to probabilities."""
        conf_arr = np.clip(np.array(confidences, dtype=np.float64), 1e-5, 1.0 - 1e-5)
        logits = np.log(conf_arr / (1.0 - conf_arr))
        scaled_logits = logits / max(1e-3, self.temperature)
        calibrated_probs = 1.0 / (1.0 + np.exp(-scaled_logits))
        return np.clip(calibrated_probs, 0.01, 0.99)
