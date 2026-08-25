"""Diagnostics and evaluation metrics for symbolic verification."""

import numpy as np
from typing import List, Dict, Any

def compute_sfar(is_correct_list: List[bool], is_accepted_list: List[bool]) -> float:
    """
    Compute Solver False-Accept Rate (SFAR).
    SFAR = count(false VLM claim accepted by verifier) / count(false VLM claims)
    """
    false_claims_total = 0
    false_claims_accepted = 0

    for is_correct, is_accepted in zip(is_correct_list, is_accepted_list):
        if not is_correct:  # VLM claim was actually incorrect
            false_claims_total += 1
            if is_accepted:  # Verifier failed to catch the error
                false_claims_accepted += 1

    if false_claims_total == 0:
        return 0.0
    return false_claims_accepted / false_claims_total

def compute_contradiction_metrics(is_correct_list: List[bool], is_flagged_contradicted_list: List[bool]) -> Dict[str, float]:
    """
    Compute Precision, Recall, and F1 for contradiction detection.
    
    Ground Truth Contradiction: (not is_correct)
    Predicted Contradiction: is_flagged_contradicted
    """
    tp = 0  # False claim correctly flagged as contradicted
    fp = 0  # True claim incorrectly flagged as contradicted
    fn = 0  # False claim missed (not flagged)
    tn = 0  # True claim correctly accepted

    for is_correct, flagged in zip(is_correct_list, is_flagged_contradicted_list):
        is_actually_wrong = not is_correct
        if is_actually_wrong and flagged:
            tp += 1
        elif (not is_actually_wrong) and flagged:
            fp += 1
        elif is_actually_wrong and (not flagged):
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(is_correct_list) if len(is_correct_list) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "sfar": compute_sfar(is_correct_list, [not f for f in is_flagged_contradicted_list])
    }

def compute_solve_time_stats(solve_times_ms: List[float]) -> Dict[str, float]:
    """Compute mean and 95th percentile solve times."""
    if not solve_times_ms:
        return {"mean_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    arr = np.array(solve_times_ms)
    return {
        "mean_ms": float(np.mean(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(np.max(arr))
    }
