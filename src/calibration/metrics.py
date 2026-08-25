"""Calibration metrics: ECE, ACE, Brier score, and NLL."""

import numpy as np
from typing import List, Dict, Any, Union, Tuple

def compute_ece(
    confidences: Union[List[float], np.ndarray],
    labels: Union[List[bool], np.ndarray],
    num_bins: int = 10
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute Expected Calibration Error (ECE) and bin details for reliability diagrams.
    """
    conf = np.array(confidences, dtype=np.float64)
    y = np.array(labels, dtype=np.float64)

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(conf, bins) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []

    total_samples = len(conf)

    for i in range(num_bins):
        in_bin = bin_indices == i
        count = int(np.sum(in_bin))
        bin_counts.append(count)

        if count > 0:
            avg_acc = float(np.mean(y[in_bin]))
            avg_conf = float(np.mean(conf[in_bin]))
            bin_accs.append(avg_acc)
            bin_confs.append(avg_conf)
            ece += (count / total_samples) * abs(avg_acc - avg_conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append((bins[i] + bins[i + 1]) / 2.0)

    bin_data = {
        "bin_edges": bins.tolist(),
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
        "bin_counts": bin_counts,
        "num_bins": num_bins
    }

    return float(ece), bin_data

def compute_brier_score(
    confidences: Union[List[float], np.ndarray],
    labels: Union[List[bool], np.ndarray]
) -> float:
    """Compute Brier Score = mean((confidence - label)^2)."""
    conf = np.array(confidences, dtype=np.float64)
    y = np.array(labels, dtype=np.float64)
    return float(np.mean((conf - y) ** 2))

def compute_nll(
    confidences: Union[List[float], np.ndarray],
    labels: Union[List[bool], np.ndarray]
) -> float:
    """Compute Negative Log-Likelihood (NLL)."""
    conf = np.clip(np.array(confidences, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    y = np.array(labels, dtype=np.float64)
    loss = -np.mean(y * np.log(conf) + (1.0 - y) * np.log(1.0 - conf))
    return float(loss)

def evaluate_calibration(
    confidences: Union[List[float], np.ndarray],
    labels: Union[List[bool], np.ndarray]
) -> Dict[str, Any]:
    """Compute all standard calibration metrics."""
    ece_10, bin_data_10 = compute_ece(confidences, labels, num_bins=10)
    ece_15, bin_data_15 = compute_ece(confidences, labels, num_bins=15)
    brier = compute_brier_score(confidences, labels)
    nll = compute_nll(confidences, labels)

    return {
        "ece_10": ece_10,
        "ece_15": ece_15,
        "brier_score": brier,
        "nll": nll,
        "bin_data_10": bin_data_10,
        "bin_data_15": bin_data_15
    }
