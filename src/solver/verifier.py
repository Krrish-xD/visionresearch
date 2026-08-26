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

    if vlm_claim is None:
        return "sat", False, (time.perf_counter() - start_t) * 1000.0

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
    gt_weight: int = 500,
    timeout_ms: int = 10000
) -> Tuple[str, List[bool], List[bool], List[bool], float]:
    """
    MaxSMT soft constraint verifier with SOOR (Solver Over-Override) tracking.
    
    - Benchmark ground-truth facts are added as SOFT constraints with fixed baseline weight W_gt (default 500).
    - VLM claims are added as SOFT constraints with confidence-scaled weights W_vlm = round(w * 1000).
    
    Behavior:
      - If raw confidence is overconfident (e.g. p=0.92 -> W_vlm=920 > W_gt=500), Z3 satisfies
        the false VLM claim and DROPS the ground truth fact (SOOR Event = True).
      - If calibrated confidence is adjusted (e.g. p=0.45 -> W_vlm=450 < W_gt=500), Z3 satisfies
        the ground truth fact and DROPS the VLM claim (SOOR Event = False, Contradiction Detected).
    
    Returns:
        (solver_status, claim_satisfied_list, gt_satisfied_list, soor_triggered_list, solve_time_ms)
    """
    start_t = time.perf_counter()
    opt = z3.Optimize()
    opt.set("timeout", timeout_ms)

    # 1. Add benchmark ground-truth facts as soft anchor constraints with W_gt (e.g. 500)
    gt_exprs = []
    for fact in gold_facts:
        expr, _ = encode_fact_to_z3(fact, prefix="gt")
        gt_exprs.append(expr)
        opt.add_soft(expr, weight=max(1, int(gt_weight)))

    # 2. Add VLM claims as soft constraints with confidence-scaled weights W_vlm = round(w * 1000)
    claim_exprs = []
    for claim, w in zip(vlm_claims, weights):
        expr, _ = encode_fact_to_z3(claim, prefix="vlm")
        claim_exprs.append(expr)
        scaled_weight = max(1, int(round(w * 1000.0)))
        opt.add_soft(expr, weight=scaled_weight)

    res = opt.check()
    solve_time_ms = (time.perf_counter() - start_t) * 1000.0

    if res == z3.sat:
        model = opt.model()
        
        # Evaluate which VLM claims were satisfied
        claim_sat_list = [bool(z3.is_true(model.eval(expr))) for expr in claim_exprs]
        
        # Evaluate which Ground Truth facts were satisfied
        gt_sat_list = [bool(z3.is_true(model.eval(expr))) for expr in gt_exprs]
        
        # SOOR Event: VLM claim is satisfied BUT a ground truth fact was dropped/unsatisfied
        all_gt_sat = all(gt_sat_list) if gt_sat_list else True
        soor_triggered_list = [(c_sat and not all_gt_sat) for c_sat in claim_sat_list]
        
        return "sat", claim_sat_list, gt_sat_list, soor_triggered_list, solve_time_ms
    elif res == z3.unsat:
        n_c = len(vlm_claims)
        n_g = len(gold_facts)
        return "unsat", [False] * n_c, [False] * n_g, [False] * n_c, solve_time_ms
    else:
        n_c = len(vlm_claims)
        n_g = len(gold_facts)
        return "unknown", [False] * n_c, [False] * n_g, [False] * n_c, solve_time_ms
