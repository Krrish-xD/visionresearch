"""Evaluation package: splits, runner, statistics, tables, and plots."""

from src.evaluation.make_splits import make_all_splits, generate_stratified_splits
from src.evaluation.run_experiment import run_experiment_pipeline
from src.evaluation.stats import mcnemar_test, bootstrap_metric_ci, wilcoxon_paired_test
from src.evaluation.tables import save_table_bundle
from src.evaluation.plots import plot_reliability_diagrams, plot_f1_bootstrap_ci

__all__ = [
    "make_all_splits",
    "generate_stratified_splits",
    "run_experiment_pipeline",
    "mcnemar_test",
    "bootstrap_metric_ci",
    "wilcoxon_paired_test",
    "save_table_bundle",
    "plot_reliability_diagrams",
    "plot_f1_bootstrap_ci"
]
