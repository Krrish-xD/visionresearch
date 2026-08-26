"""Adaptive Prediction Sets (APS) Conformal Calibrator."""

import numpy as np
from typing import List, Union, Optional

def ensure_softmax(item: Union[List[float], np.ndarray, float]) -> np.ndarray:
    """
    Ensure the input is a valid softmax probability vector on the simplex Delta^{K-1}.
    If raw pre-softmax logits are provided, applies numerically stable softmax.
    """
    if isinstance(item, (list, tuple, np.ndarray)) and len(item) > 1:
        arr = np.array(item, dtype=np.float64)
        # Check if already a valid probability vector (non-negative and sums to ~1.0)
        if np.all(arr >= 0.0) and np.isclose(np.sum(arr), 1.0, atol=1e-2):
            return arr / np.sum(arr)
        # Otherwise apply numerically stable softmax to logits
        z_max = np.max(arr)
        exp_z = np.exp(arr - z_max)
        return exp_z / np.sum(exp_z)
    else:
        # Scalar confidence p -> 2-class distribution [p, 1 - p]
        p_val = float(item if not isinstance(item, (list, tuple, np.ndarray)) else item[0])
        p_val = max(1e-5, min(1.0 - 1e-5, p_val))
        return np.array([p_val, 1.0 - p_val])

class ConformalRiskWeighting:
    """
    Split Conformal Prediction using standard Adaptive Prediction Sets (APS).
    
    1. Softmax Verification: Converts input logit vectors to valid probabilities pi in Delta^{K-1}.
    2. Nonconformity score: Cumulative softmax probability needed to include the true answer class:
       s_i = sum_{j=1}^{r_i} \pi_{(j)}(x_i) \in [0, 1.0]
       where \pi_{(1)} >= \pi_{(2)} >= ... is sorted descending, and r_i is rank of true label.
    3. Conformal threshold: Empirical (1 - \alpha) quantile \hat{q} \in [0, 1.0].
    4. Prediction set: C(x) = { (1), ..., (k^*) } where k^* is minimal index with cumulative sum >= \hat{q}.
    5. Solver weight: Mapped to the inverse set size:
       w = max(min_weight, 1.0 / |C(x)|)
       - Confident single-class prediction (|C(x)| = 1) -> weight = 1.0
       - Uncertain multi-class prediction (|C(x)| >= 2) -> weight = 1 / |C(x)| <= 0.50 (drops below W_gt=500).
    """
    def __init__(self, alpha: float = 0.10, min_weight: float = 0.05):
        self.alpha = alpha
        self.min_weight = min_weight
        self.q_hat: float = 0.95

    def fit(
        self,
        probs_or_confs: Union[List[List[float]], List[np.ndarray], List[float], np.ndarray],
        labels: Optional[Union[List[bool], List[int], np.ndarray]] = None,
        targets: Optional[Union[List[int], np.ndarray]] = None
    ):
        """
        Fit conformal threshold \hat{q} on calibration split using dynamic ground-truth rank.
        
        Args:
            probs_or_confs: Probability or logit vectors across candidate classes
            labels: Boolean correctness indicators or class labels
            targets: Explicit integer target class indices in the probability vectors
        """
        scores = []
        n_samples = len(probs_or_confs)
        
        if targets is not None:
            target_indices = np.array(targets, dtype=np.int64)
        elif labels is not None:
            labels_arr = np.array(labels)
            if labels_arr.dtype == bool:
                # If boolean: True means top-1 (index 0), False means alternative class (index 1 if 2-class, or dynamic)
                target_indices = np.where(labels_arr, 0, 1)
            else:
                target_indices = labels_arr.astype(np.int64)
        else:
            target_indices = np.zeros(n_samples, dtype=np.int64)

        for i, item in enumerate(probs_or_confs):
            p_vec = ensure_softmax(item)
            order = np.argsort(p_vec)[::-1]
            p_sorted = p_vec[order]
            
            # Target class index for this sample
            target_cls = int(target_indices[i]) if i < len(target_indices) else 0
            
            # Dynamically locate the rank of the ground truth class in sorted probabilities
            rank_positions = np.where(order == target_cls)[0]
            if len(rank_positions) > 0:
                rank = int(rank_positions[0]) + 1
            else:
                # Fallback if target class index exceeds vector dimension
                is_correct = bool(labels[i]) if (labels is not None and i < len(labels)) else False
                rank = 1 if is_correct else min(2, len(p_sorted))

            # Cumulative softmax probability up to ground truth rank
            score = float(np.sum(p_sorted[:rank]))
            scores.append(float(np.clip(score, 0.0, 1.0)))

        n = len(scores)
        if n == 0:
            self.q_hat = 0.95
            return

        level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.q_hat = float(np.clip(np.quantile(scores, level, method="higher"), 0.01, 1.0))

    def predict_set(
        self,
        probs_or_confs: Union[List[List[float]], List[np.ndarray], List[float], np.ndarray]
    ) -> List[List[int]]:
        """
        Construct adaptive prediction sets C(x) for each query item.
        Guaranteed to terminate with valid subset because sum(pi) = 1.0 >= q_hat.
        """
        prediction_sets = []
        for item in probs_or_confs:
            p_vec = np.sort(ensure_softmax(item))[::-1]

            cumsum = 0.0
            included = []
            for k in range(len(p_vec)):
                included.append(k)
                cumsum += p_vec[k]
                if cumsum >= self.q_hat:
                    break
            prediction_sets.append(included if len(included) > 0 else [0])
        return prediction_sets

    def transform(
        self,
        probs_or_confs: Union[List[List[float]], List[np.ndarray], List[float], np.ndarray]
    ) -> np.ndarray:
        """
        Convert prediction confidences to inverse set-size solver weights:
        w = max(min_weight, 1.0 / |C(x)|).
        """
        pred_sets = self.predict_set(probs_or_confs)
        weights = []
        for p_set in pred_sets:
            set_size = max(1, len(p_set))
            w = 1.0 / set_size
            weights.append(max(self.min_weight, w))
        return np.array(weights, dtype=np.float64)

# Alias for explicit APS naming
AdaptivePredictionSets = ConformalRiskWeighting
