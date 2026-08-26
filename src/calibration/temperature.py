"""Multinomial Temperature Scaling Calibrator."""

import numpy as np
from scipy.optimize import minimize
from typing import List, Union, Optional

def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    z_max = np.max(z, axis=axis, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)

class TemperatureScaling:
    """
    Multinomial Temperature Scaling on pre-softmax logit vectors z in R^K.
    Fits scalar T > 0 on calibration split by minimizing multiclass Cross-Entropy / NLL:
        L(T) = - 1/N sum_i log( softmax(z_i / T)_{y_i} )
    Calculates calibrated probabilities via:
        p^{cal} = softmax(z / T)_{y_i}
    """
    def __init__(self):
        self.temperature: float = 1.0

    def fit(
        self,
        logits_or_confs: Union[List[List[float]], List[np.ndarray], List[float], np.ndarray],
        labels: Union[List[bool], List[int], np.ndarray],
        chosen_indices: Optional[List[int]] = None
    ):
        """
        Fit temperature parameter T on calibration items.
        
        Args:
            logits_or_confs: List of logit vectors z in R^K, or 2D array [N, K], or 1D confidences [N]
            labels: List of true class labels (indices or booleans if binary/chosen correctness)
            chosen_indices: Indices of the predicted/chosen tokens (defaults to 0 or argmax)
        """
        # Convert input to 2D logit array
        if isinstance(logits_or_confs, (list, tuple)) and len(logits_or_confs) > 0 and isinstance(logits_or_confs[0], (list, tuple, np.ndarray)):
            logits_mat = np.array(logits_or_confs, dtype=np.float64)
        elif isinstance(logits_or_confs, np.ndarray) and logits_or_confs.ndim == 2:
            logits_mat = logits_or_confs.astype(np.float64)
        else:
            # Fallback if 1D scalar probabilities provided: convert to 2-class logits [log(p), log(1-p)]
            p_arr = np.clip(np.array(logits_or_confs, dtype=np.float64), 1e-5, 1.0 - 1e-5)
            logits_mat = np.column_stack([np.log(p_arr), np.log(1.0 - p_arr)])

        N, K = logits_mat.shape
        y = np.array(labels)

        # Target class indices
        if chosen_indices is not None:
            targets = np.array(chosen_indices, dtype=np.int64)
        else:
            if y.dtype == bool or set(np.unique(y)).issubset({0, 1, True, False}):
                # If correct -> index 0 (top-1), if wrong -> alternative class index 1
                targets = np.where(y, 0, 1 if K > 1 else 0)
            else:
                targets = y.astype(np.int64)

        def multiclass_nll(T_arr):
            T = max(1e-3, float(T_arr[0]))
            scaled = logits_mat / T
            probs = softmax(scaled, axis=1)
            target_probs = probs[np.arange(N), np.clip(targets, 0, K - 1)]
            target_probs = np.clip(target_probs, 1e-7, 1.0)
            loss = -np.mean(np.log(target_probs))
            return loss

        res = minimize(multiclass_nll, x0=[1.0], bounds=[(0.01, 20.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0]) if res.success else 1.0

    def transform(
        self,
        logits_or_confs: Union[List[List[float]], List[np.ndarray], List[float], np.ndarray],
        chosen_indices: Optional[Union[List[int], int]] = None
    ) -> np.ndarray:
        """
        Apply fitted temperature scaling to logit vectors: p^{cal} = softmax(z / T)_{y_i}.
        """
        if isinstance(logits_or_confs, (list, tuple)) and len(logits_or_confs) > 0 and isinstance(logits_or_confs[0], (list, tuple, np.ndarray)):
            logits_mat = np.array(logits_or_confs, dtype=np.float64)
        elif isinstance(logits_or_confs, np.ndarray) and logits_or_confs.ndim == 2:
            logits_mat = logits_or_confs.astype(np.float64)
        else:
            p_arr = np.clip(np.array(logits_or_confs, dtype=np.float64), 1e-5, 1.0 - 1e-5)
            logits_mat = np.column_stack([np.log(p_arr), np.log(1.0 - p_arr)])

        N, K = logits_mat.shape
        T = max(1e-3, self.temperature)
        scaled_logits = logits_mat / T
        calibrated_probs = softmax(scaled_logits, axis=1)

        # Extract top-1 / chosen class probability
        if chosen_indices is None:
            return np.clip(calibrated_probs[:, 0], 0.01, 0.99)
        elif isinstance(chosen_indices, int):
            return np.clip(calibrated_probs[:, chosen_indices], 0.01, 0.99)
        else:
            idx_arr = np.array(chosen_indices, dtype=np.int64)
            chosen_p = calibrated_probs[np.arange(N), np.clip(idx_arr, 0, K - 1)]
            return np.clip(chosen_p, 0.01, 0.99)
