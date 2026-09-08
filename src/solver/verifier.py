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
    timeout_ms: int = 10000,
    hard_gt: bool = False
) -> Tuple[str, List[bool], List[bool], List[bool], float]:
    """
    MaxSMT constraint verifier with optional SOOR (Solver Over-Override) tracking.
    
    Modes:
      - hard_gt=True (Sound Scientific Default): Benchmark ground-truth facts are added as HARD immutable
        constraints. VLM claims are soft with confidence weights W_vlm. If a claim contradicts GT,
        GT is strictly preserved and the false VLM claim is dropped (is_sat=False, is_contra=True, SOOR=False).
      - hard_gt=False (Unsafe Ablation): Benchmark ground-truth facts are added as SOFT constraints
        with baseline weight W_gt (default 500). If VLM confidence is high (W_vlm > W_gt), the solver
        overrides GT to satisfy the claim (SOOR Event = True).
    
    Returns:
        (solver_status, claim_satisfied_list, gt_satisfied_list, soor_triggered_list, solve_time_ms)
    """
    start_t = time.perf_counter()
    opt = z3.Optimize()
    opt.set("timeout", timeout_ms)

    # 1. Add benchmark ground-truth facts
    gt_exprs = []
    for fact in gold_facts:
        expr, _ = encode_fact_to_z3(fact, prefix="gt")
        gt_exprs.append(expr)
        if hard_gt:
            opt.add(expr)
        else:
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
        if hard_gt:
            gt_sat_list = [True] * len(gt_exprs)
            soor_triggered_list = [False] * len(claim_exprs)
        else:
            gt_sat_list = [bool(z3.is_true(model.eval(expr))) for expr in gt_exprs]
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

def verify_multi_hypothesis_maxsmt(
    gold_facts: List[Dict[str, Any]],
    candidate_claims: List[Optional[Dict[str, Any]]],
    candidate_weights: List[float],
    hard_gt: bool = True,
    gt_weight: int = 500,
    timeout_ms: int = 10000
) -> Tuple[str, Optional[int], List[bool], float]:
    """
    Multi-Hypothesis N-Best MaxSMT Verification and Selection.
    
    Given competing candidate hypotheses C_0, C_1, ..., C_{K-1} with confidence weights w_i:
      - Benchmark facts are asserted as hard constraints (hard_gt=True).
      - Indicator booleans h_0, ..., h_{K-1} represent whether hypothesis i is chosen.
      - Mutual exclusivity is enforced: at most one hypothesis can be chosen (sum h_i <= 1).
      - Indicators imply claims: h_i => C_i.
      - Soft objectives maximize sum(w_i * h_i).
      
    If top-1 candidate contradicts GT, the solver automatically falls back to selecting
    the highest-weight alternative candidate that is consistent with ground-truth facts.
    If all candidates contradict GT, selected_index is None.
    
    Returns:
        (solver_status, selected_index, hypothesis_sat_list, solve_time_ms)
    """
    start_t = time.perf_counter()
    opt = z3.Optimize()
    opt.set("timeout", timeout_ms)

    # 1. Add ground-truth facts
    for fact in gold_facts:
        expr, _ = encode_fact_to_z3(fact, prefix="gt")
        if hard_gt:
            opt.add(expr)
        else:
            opt.add_soft(expr, weight=max(1, int(gt_weight)))

    n_cands = len(candidate_claims)
    if n_cands == 0:
        res = opt.check()
        solve_time_ms = (time.perf_counter() - start_t) * 1000.0
        return ("sat" if res == z3.sat else "unsat"), None, [], solve_time_ms

    # 2. Define hypothesis indicator variables
    h_vars = [z3.Bool(f"hyp_{i}") for i in range(n_cands)]

    # 3. Mutual exclusivity: at most one candidate hypothesis selected
    for i in range(n_cands):
        for j in range(i + 1, n_cands):
            opt.add(z3.Or(z3.Not(h_vars[i]), z3.Not(h_vars[j])))

    # 4. Link indicator variables to claim expressions and assign soft weights
    for i in range(n_cands):
        claim = candidate_claims[i]
        weight = candidate_weights[i] if i < len(candidate_weights) else 0.0
        scaled_w = max(1, int(round(weight * 1000.0)))

        if claim is not None:
            c_expr, _ = encode_fact_to_z3(claim, prefix=f"cand_{i}")
            opt.add(z3.Implies(h_vars[i], c_expr))
            opt.add_soft(h_vars[i], weight=scaled_w)
        else:
            opt.add(z3.Not(h_vars[i]))

    # 5. Check MaxSMT optimization
    res = opt.check()
    solve_time_ms = (time.perf_counter() - start_t) * 1000.0

    if res == z3.sat:
        model = opt.model()
        h_sat_list = [bool(z3.is_true(model.eval(h))) for h in h_vars]
        selected_index = None
        for i, sat in enumerate(h_sat_list):
            if sat:
                selected_index = i
                break
        return "sat", selected_index, h_sat_list, solve_time_ms
    elif res == z3.unsat:
        return "unsat", None, [False] * n_cands, solve_time_ms
    else:
        return "unknown", None, [False] * n_cands, solve_time_ms

