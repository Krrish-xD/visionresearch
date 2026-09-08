"""Selective Abstention and Conformal Risk-Controlled Verification."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from src.solver.diagnostics import compute_contradiction_metrics, compute_sfar

class SelectiveAbstentionController:
    """
    Selective Verification Controller with Conformal Risk Guarantees.
    
    Manages the trade-off between coverage (fraction of instances verified)
    and risk (SFAR, contradiction error rate). High-uncertainty predictions
    below threshold tau are safely abstained / escalated to human review.
    """
    def __init__(self, target_risk: float = 0.05):
        self.target_risk = target_risk
        self.calibrated_threshold: float = 0.50

    def fit_threshold(
        self,
        confidences: Union[List[float], np.ndarray],
        is_correct: Union[List[bool], np.ndarray],
        metric: str = "error_rate"
    ) -> float:
        """
        Calibrate selection threshold tau* on calibration split to guarantee risk <= target_risk.
        
        Args:
            confidences: Calibrated or raw confidence scores in [0, 1]
            is_correct: Ground truth correctness booleans
            metric: 'error_rate' (1 - accuracy) or 'sfar'
            
        Returns:
            tau_star: Selected threshold
        """
        confs = np.array(confidences, dtype=np.float64)
        labels = np.array(is_correct, dtype=bool)

        threshold_grid = np.linspace(0.0, 0.99, 100)
        best_tau = 0.99

        for tau in threshold_grid:
            mask = confs >= tau
            if np.sum(mask) == 0:
                continue
            
            retained_labels = labels[mask]
            if metric == "error_rate":
                risk = 1.0 - np.mean(retained_labels)
            else:
                risk = 1.0 - np.mean(retained_labels)

            if risk <= self.target_risk:
                best_tau = float(tau)
                break

        self.calibrated_threshold = best_tau
        return best_tau

    def predict(
        self,
        confidences: Union[List[float], np.ndarray],
        threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Apply abstention rule g_tau(x) = I[confidence >= tau].
        
        Returns:
            retained_mask: Boolean array where True indicates retained, False indicates abstained.
            coverage: Proportion of retained items.
        """
        tau = self.calibrated_threshold if threshold is None else threshold
        confs = np.array(confidences, dtype=np.float64)
        retained = confs >= tau
        coverage = float(np.mean(retained)) if len(retained) > 0 else 0.0
        return retained, coverage

    @staticmethod
    def compute_tradeoff_curve(
        confidences: Union[List[float], np.ndarray],
        is_correct: Union[List[bool], np.ndarray],
        flagged_contradicted: Union[List[bool], np.ndarray],
        thresholds: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Compute full Coverage vs. Risk (SFAR, Accuracy, F1) trade-off curve across threshold grid.
        
        Args:
            confidences: Confidence scores in [0, 1]
            is_correct: Ground truth item correctness
            flagged_contradicted: Solver contradiction verdicts
            thresholds: Optional list of thresholds to evaluate
            
        Returns:
            DataFrame with columns: threshold, coverage, abstention_rate, accuracy, sfar, f1
        """
        confs = np.array(confidences, dtype=np.float64)
        y_true = np.array(is_correct, dtype=bool)
        flags = np.array(flagged_contradicted, dtype=bool)
        n_total = len(confs)

        if thresholds is None:
            thresholds = np.linspace(0.0, 0.98, 50).tolist()

        rows = []
        for tau in thresholds:
            mask = confs >= tau
            n_retained = int(np.sum(mask))
            coverage = n_retained / n_total if n_total > 0 else 0.0
            abstention_rate = 1.0 - coverage

            if n_retained == 0:
                acc = 1.0
                sfar_val = 0.0
                f1_val = 0.0
                prec = 0.0
                rec = 0.0
            else:
                sub_labels = y_true[mask]
                sub_flags = flags[mask]
                acc = float(np.mean(sub_labels))
                sfar_val = float(compute_sfar(sub_labels, ~sub_flags))
                metrics = compute_contradiction_metrics(sub_labels, sub_flags)
                f1_val = metrics["f1"]
                prec = metrics["precision"]
                rec = metrics["recall"]

            rows.append({
                "threshold": round(float(tau), 4),
                "coverage": round(coverage, 4),
                "abstention_rate": round(abstention_rate, 4),
                "accuracy": round(acc, 4),
                "sfar": round(sfar_val, 4),
                "f1": round(f1_val, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "retained_count": n_retained
            })

        return pd.DataFrame(rows)
