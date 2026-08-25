"""Conformal Risk Calibrator and Weight Estimator."""

# pyrefly: ignore [missing-import]
import numpy as np
from typing import List, Union

class ConformalRiskWeighting:
    """
    Split Conformal Risk Weighting.
    Uses nonconformity score s = 1 - p(y_true).
    Computes quantile threshold q_hat at miscoverage level alpha.
    Converts risk score into MaxSMT solver weight w = max(min_weight, 1 - risk).
    """
    def __init__(self, alpha: float = 0.10, min_weight: float = 0.05):
        self.alpha = alpha
        self.min_weight = min_weight
        self.q_hat: float = 0.5

    def fit(self, confidences: Union[List[float], np.ndarray], labels: Union[List[bool], np.ndarray]):
        """
        Fit conformal threshold on calibration items.
        score_i = 1 - (conf_i if label_i else 1 - conf_i)
        """
        conf_arr = np.array(confidences, dtype=np.float64)
        y = np.array(labels, dtype=bool)

        # True class confidence
        p_true = np.where(y, conf_arr, 1.0 - conf_arr)
        scores = 1.0 - p_true

        n = len(scores)
        if n == 0:
            self.q_hat = 0.5
            return

        # Conformal quantile: ceil((n + 1)(1 - alpha)) / n
        level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.q_hat = float(np.quantile(scores, level, method="higher"))

    def transform(self, confidences: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Convert prediction confidence into conformal risk-adjusted solver weights.
        """
        conf_arr = np.array(confidences, dtype=np.float64)
        # Risk bound estimation
        estimated_risk = np.clip((1.0 - conf_arr) / max(0.01, self.q_hat), 0.0, 1.0)
        weights = np.maximum(self.min_weight, 1.0 - estimated_risk)
        return weights
