"""Statistical significance testing, bootstrapping, and multiple testing correction."""

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import List, Dict, Any, Tuple

def mcnemar_test(contingency_table: np.ndarray) -> Tuple[float, float]:
    """
    McNemar's test for paired binary classification outcomes.
    
    contingency_table: 2x2 matrix [[n00, n01], [n10, n11]]
      n01: Condition A correct, Condition B wrong
      n10: Condition A wrong, Condition B correct
    """
    b = contingency_table[0, 1]
    c = contingency_table[1, 0]

    if b + c == 0:
        return 0.0, 1.0

    # With continuity correction
    stat = ((abs(b - c) - 1.0) ** 2) / (b + c)
    p_value = float(stats.chi2.sf(stat, df=1))
    return float(stat), p_value

def bootstrap_metric_ci(
    y_true: List[bool],
    y_pred: List[bool],
    metric_fn,
    n_resamples: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute metric value and non-parametric bootstrap confidence interval.
    
    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    n = len(y_t)

    if n == 0:
        return 0.0, 0.0, 0.0

    point_estimate = float(metric_fn(y_t, y_p))

    bootstrap_scores = []
    for _ in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        sample_score = metric_fn(y_t[indices], y_p[indices])
        bootstrap_scores.append(sample_score)

    alpha = 1.0 - ci_level
    ci_lower = float(np.percentile(bootstrap_scores, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(bootstrap_scores, 100 * (1.0 - alpha / 2.0)))

    return point_estimate, ci_lower, ci_upper

def wilcoxon_paired_test(scores_a: List[float], scores_b: List[float]) -> Tuple[float, float]:
    """Wilcoxon signed-rank test for paired continuous scores."""
    arr_a = np.array(scores_a)
    arr_b = np.array(scores_b)
    diff = arr_a - arr_b
    if np.all(diff == 0):
        return 0.0, 1.0
    res = stats.wilcoxon(diff)
    return float(res.statistic), float(res.pvalue)

def apply_benjamini_hochberg(p_values: List[float], alpha: float = 0.05) -> Tuple[List[bool], List[float]]:
    """Apply Benjamini-Hochberg FDR correction for multiple comparisons."""
    if not p_values:
        return [], []
    rejected, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
    return rejected.tolist(), p_corrected.tolist()
