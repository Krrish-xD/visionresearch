"""Solver package: Z3 encoding, MaxSMT verification, and solver diagnostics."""

from src.solver.z3_encoder import encode_fact_to_z3
from src.solver.verifier import check_hard_contradiction, verify_with_maxsmt
from src.solver.diagnostics import compute_sfar, compute_contradiction_metrics, compute_solve_time_stats

__all__ = [
    "encode_fact_to_z3",
    "check_hard_contradiction",
    "verify_with_maxsmt",
    "compute_sfar",
    "compute_contradiction_metrics",
    "compute_solve_time_stats"
]
