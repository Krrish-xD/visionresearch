"""Z3 MaxSMT Verifier and Hard Contradiction Oracle."""

import time
import z3
from typing import Dict, Any, List, Optional, Tuple
from src.solver.z3_encoder import encode_fact_to_z3

def check_hard_contradiction(
    gold_facts: List[Dict[str, Any]],
    vlm_claim: Dict[str, Any],
    timeout_ms: int = 5000
) -> Tuple[str, bool, float]:
    """
    Hard logical contradiction oracle.
    Asserts all ground truth facts and the VLM claim as hard constraints.
    
    Returns:
        (status, is_contradicted, solve_time_ms)
        status: 'sat', 'unsat', 'timeout', or 'unknown'
        is_contradicted: True if unsat (logical contradiction)
    """
    start_t = time.perf_counter()
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    for fact in gold_facts:
        expr, _ = encode_fact_to_z3(fact, prefix="gt")
        solver.add(expr)

    claim_expr, _ = encode_fact_to_z3(vlm_claim, prefix="vlm")
    solver.add(claim_expr)

    res = solver.check()
    solve_time_ms = (time.perf_counter() - start_t) * 1000.0

    if res == z3.unsat:
        return "unsat", True, solve_time_ms
    elif res == z3.sat:
        return "sat", False, solve_time_ms
    else:
        return "unknown", False, solve_time_ms

def verify_with_maxsmt(
    gold_facts: List[Dict[str, Any]],
    vlm_claims: List[Dict[str, Any]],
    weights: List[float],
    timeout_ms: int = 10000
) -> Tuple[str, List[bool], float]:
    """
    MaxSMT soft constraint verifier.
    
    - Benchmark facts are asserted as HARD constraints (cannot be violated).
    - VLM claims are asserted as SOFT constraints with specified weights.
    
    Returns:
        (solver_status, claim_satisfied_list, solve_time_ms)
    """
    start_t = time.perf_counter()
    opt = z3.Optimize()
    opt.set("timeout", timeout_ms)

    # 1. Add benchmark facts as hard constraints
    for fact in gold_facts:
        expr, _ = encode_fact_to_z3(fact, prefix="gt")
        opt.add(expr)

    # 2. Add VLM claims as soft constraints with weights
    # Z3 Optimize accepts positive integer or float weights
    claim_exprs = []
    for claim, w in zip(vlm_claims, weights):
        expr, _ = encode_fact_to_z3(claim, prefix="vlm")
        claim_exprs.append(expr)
        # Scale float weight to integer weight (e.g. 0.93 -> 930)
        scaled_weight = max(1, int(round(w * 1000.0)))
        opt.add_soft(expr, weight=scaled_weight)

    res = opt.check()
    solve_time_ms = (time.perf_counter() - start_t) * 1000.0

    if res == z3.sat:
        model = opt.model()
        satisfied_list = []
        for expr in claim_exprs:
            eval_val = model.eval(expr)
            satisfied_list.append(bool(z3.is_true(eval_val)))
        return "sat", satisfied_list, solve_time_ms
    elif res == z3.unsat:
        return "unsat", [False] * len(vlm_claims), solve_time_ms
    else:
        return "unknown", [False] * len(vlm_claims), solve_time_ms
