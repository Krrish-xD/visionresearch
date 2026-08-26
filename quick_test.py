"""
Rigorous 'Mock VLM, Real Math' Smoke Test for VisionResearch Pipeline on Mac.

Demonstrates:
  1. Synthetic VLM output generation with full top-5 pre-softmax logit vectors.
  2. Real mathematical execution of Multinomial Temperature Scaling (L-BFGS-B NLL minimization).
  3. Real mathematical execution of Adaptive Prediction Sets (APS) Conformal calibration.
  4. Real mathematical execution of Monotonic Isotonic Regression (PAVA).
  5. Real Z3 SMT AST encoding and MaxSMT solving with Soft Ground Truth ($W_{gt} = 500$).
  6. Explicit mathematical assertions proving that Raw Confidence triggers a SOOR event,
     while Calibrated Confidence eliminates SOOR and preserves Ground Truth.
"""

import numpy as np
from typing import List, Dict, Any

from src.calibration.temperature import TemperatureScaling, softmax
from src.calibration.isotonic import IsotonicCalibrator
from src.calibration.conformal import ConformalRiskWeighting, AdaptivePredictionSets
from src.solver.z3_encoder import encode_fact_to_z3
from src.solver.verifier import check_hard_contradiction, verify_with_maxsmt
from src.solver.diagnostics import compute_contradiction_metrics, compute_sfar, compute_soor

def run_real_math_smoke_test():
    print("=" * 80)
    print("🔬 VisionResearch: 'Mock VLM, Real Math' Verification Test (Mac)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Synthetic Calibration Split (20 items: 8 correct, 12 overconfident errors)
    # -------------------------------------------------------------------------
    print("\n[Step 1] Generating Synthetic Calibration Split (20 items: 8 Correct, 12 Overconfident Errors)...")
    np.random.seed(42)
    
    cal_logits: List[np.ndarray] = []
    cal_labels: List[bool] = []
    cal_confs: List[float] = []

    # 8 Correct items: VLM predicted index 0, Ground Truth = index 0 (p ~ 0.75 - 0.85)
    for i in range(8):
        z = np.array([3.8 + np.random.uniform(-0.1, 0.1), 1.2, 0.8, -0.2, -1.0])
        p = softmax(z)
        assert np.isclose(np.sum(p), 1.0), "Softmax probabilities must sum to 1.0"
        cal_logits.append(z)
        cal_labels.append(True)
        cal_confs.append(float(p[0]))

    # 12 Overconfident Incorrect items: VLM predicted index 0 with high confidence (p ~ 0.95),
    # but the Ground Truth class was index 1 (severe overconfidence on errors)
    for i in range(12):
        z = np.array([4.8 + np.random.uniform(-0.05, 0.05), 1.0, 0.5, -0.5, -1.0])
        p = softmax(z)
        assert np.isclose(np.sum(p), 1.0), "Softmax probabilities must sum to 1.0"
        assert p[0] > 0.90, f"Error item must be overconfident (p > 0.90), got {p[0]}"
        cal_logits.append(z)
        cal_labels.append(False)
        cal_confs.append(float(p[0]))

    print(f"  ✅ Generated {len(cal_logits)} calibration logit vectors (Dim=5).")
    print(f"     Mean Raw Confidence on Correct Items:   {np.mean(cal_confs[:8]):.4f}")
    print(f"     Mean Raw Confidence on Incorrect Items: {np.mean(cal_confs[8:]):.4f} (⚠️ Overconfident on errors!)")

    # -------------------------------------------------------------------------
    # 2. Run REAL Calibration Code (Optimization & Quantile Estimation)
    # -------------------------------------------------------------------------
    print("\n[Step 2] Executing REAL Calibration Algorithms on Calibration Split...")

    # A. Multinomial Temperature Scaling
    temp_cal = TemperatureScaling()
    temp_cal.fit(cal_logits, cal_labels)
    fitted_T = temp_cal.temperature
    print(f"  🌡️  Multinomial Temperature Scaling: Fitted T = {fitted_T:.4f} (T > 1.0 cools down overconfidence)")

    # B. Isotonic Regression
    iso_cal = IsotonicCalibrator()
    iso_cal.fit(cal_confs, cal_labels)
    print(f"  📈  Isotonic Regression: Fitted non-parametric monotonic curve via PAVA")

    # C. Adaptive Prediction Sets (APS) Conformal Prediction
    conf_cal = AdaptivePredictionSets(alpha=0.10)
    conf_cal.fit(cal_logits, cal_labels)
    q_hat = conf_cal.q_hat
    print(f"  🛡️  Adaptive Prediction Sets (APS): Nonconformity Threshold q_hat = {q_hat:.4f} (90% coverage)")

    # -------------------------------------------------------------------------
    # 3. Test on 2 Concrete Evaluation Items
    # -------------------------------------------------------------------------
    print("\n[Step 3] Evaluating Test Items across 4 Calibration Conditions...")

    test_items = [
        {
            "item_id": "eval_001_counting_error",
            "description": "Counting Red Circles (Ground Truth = 3, VLM predicts '4' with overconfidence)",
            "gt_fact": {"predicate": "count", "subject": "red_circle", "value": 3},
            "vlm_claim": {"predicate": "count", "subject": "red_circle", "value": 4},
            "is_correct": False,
            "answer_logits": np.array([4.80, 1.00, 0.50, -0.50, -1.00]) # Overconfident top-1
        },
        {
            "item_id": "eval_002_spatial_correct",
            "description": "Spatial Relation (Ground Truth = left, VLM predicts 'left')",
            "gt_fact": {"predicate": "relation", "subject": "blue_square", "relation_type": "left", "object": "yellow_circle", "value": True},
            "vlm_claim": {"predicate": "relation", "subject": "blue_square", "relation_type": "left", "object": "yellow_circle", "value": True},
            "is_correct": True,
            "answer_logits": np.array([4.20, 0.50, -0.50, -1.00, -2.00]) # Accurate top-1
        }
    ]

    W_gt = 500  # Ground Truth baseline soft anchor weight

    for item in test_items:
        z = item["answer_logits"]
        p_raw_vec = softmax(z)
        raw_p = float(p_raw_vec[0])

        # Mathematical Assertion 1: Softmax sums to 1.0
        assert np.isclose(np.sum(p_raw_vec), 1.0), "Softmax vector must sum to exactly 1.0"

        # Multinomial Temperature Scaling: p_temp = softmax(z / T)_0
        p_temp_vec = softmax(z / fitted_T)
        temp_p = float(p_temp_vec[0])
        transformed_temp = float(temp_cal.transform([z])[0])
        # Mathematical Assertion 2: Temperature transformation strictly matches formula
        assert np.isclose(temp_p, transformed_temp, atol=1e-4), "Temperature transform must match softmax(z / T)_0"

        # Isotonic Regression: p_iso = g(p_raw)
        iso_p = float(iso_cal.transform([raw_p])[0])

        # Conformal APS: C(x) and w = 1 / |C(x)|
        aps_set = conf_cal.predict_set([z])[0]
        conf_w = float(conf_cal.transform([z])[0])
        # Mathematical Assertion 3: Conformal weight equals inverse set size
        assert np.isclose(conf_w, max(0.05, 1.0 / len(aps_set))), "Conformal weight must equal 1 / |C(x)|"

        print("\n" + "=" * 80)
        print(f"🔍 EVALUATION ITEM: {item['item_id']}")
        print(f"   Task:         {item['description']}")
        print(f"   Ground Truth: {item['gt_fact']}")
        print(f"   VLM Claim:    {item['vlm_claim']} (Correct: {item['is_correct']})")
        print(f"   Pre-softmax Logit Vector z: {np.round(z, 2)}")
        print(f"   Raw Softmax Vector pi:      {np.round(p_raw_vec, 4)} -> Raw Top-1 p = {raw_p:.4f}")
        print(f"   Temperature Scaled Vector:  {np.round(p_temp_vec, 4)} -> Temp Top-1 p = {temp_p:.4f}")
        print(f"   Isotonic Calibrated p:      {iso_p:.4f}")
        print(f"   APS Prediction Set C(x):    {aps_set} (|C| = {len(aps_set)}) -> Conformal w = {conf_w:.4f}")
        print("-" * 80)

        # ---------------------------------------------------------------------
        # 4. Run REAL Z3 MaxSMT Solver Across Conditions
        # ---------------------------------------------------------------------
        conditions = [
            ("Raw Confidence (Uncalibrated)", raw_p),
            ("Multinomial Temperature Scaled", temp_p),
            ("Isotonic Regression", iso_p),
            ("Conformal APS (Inverse Set Size)", conf_w)
        ]

        for cond_name, p_val in conditions:
            w_vlm = int(round(p_val * 1000.0))
            
            # Execute REAL Z3 Optimize
            status, claim_sat, gt_sat, soor_list, t_ms = verify_with_maxsmt(
                [item["gt_fact"]],
                [item["vlm_claim"]],
                [p_val],
                gt_weight=W_gt
            )
            
            is_vlm_satisfied = claim_sat[0]
            is_gt_satisfied = gt_sat[0]
            is_soor = soor_list[0]

            if is_soor:
                verdict_str = "💥 SOOR EVENT! (VLM Claim Satisfied, Ground Truth DROPPED)"
            elif not is_vlm_satisfied and is_gt_satisfied:
                verdict_str = "🛡️  Contradiction Caught (Ground Truth Preserved, VLM Claim DROPPED)"
            else:
                verdict_str = "✨ Consistent (Both Ground Truth & Claim Satisfied)"

            print(f"  [{cond_name:<34}]")
            print(f"      Weights: W_vlm = {w_vlm:<4} vs W_gt = {W_gt}")
            print(f"      Status:  {verdict_str} ({t_ms:.2f} ms)")

        # ---------------------------------------------------------------------
        # 5. Core Scientific Assertions
        # ---------------------------------------------------------------------
        if not item["is_correct"]:
            # On the incorrect counting item:
            # 1. Raw confidence causes SOOR (W_vlm = 958 > 500)
            raw_w_vlm = int(round(raw_p * 1000.0))
            assert raw_w_vlm > W_gt, f"Raw W_vlm ({raw_w_vlm}) must exceed W_gt ({W_gt})"
            _, raw_claim_sat, raw_gt_sat, raw_soor, _ = verify_with_maxsmt(
                [item["gt_fact"]], [item["vlm_claim"]], [raw_p], gt_weight=W_gt
            )
            assert raw_soor[0] is True, "ASSERTION FAILED: Raw overconfidence must trigger SOOR"
            assert raw_claim_sat[0] is True and raw_gt_sat[0] is False, "ASSERTION FAILED: Raw VLM claim must overpower GT"

            # 2. Temperature Scaling prevents SOOR (W_vlm = 490 < 500)
            temp_w_vlm = int(round(temp_p * 1000.0))
            assert temp_w_vlm < W_gt, f"Temp W_vlm ({temp_w_vlm}) must fall below W_gt ({W_gt})"
            _, temp_claim_sat, temp_gt_sat, temp_soor, _ = verify_with_maxsmt(
                [item["gt_fact"]], [item["vlm_claim"]], [temp_p], gt_weight=W_gt
            )
            assert temp_soor[0] is False, "ASSERTION FAILED: Temperature scaling must prevent SOOR"
            assert temp_claim_sat[0] is False and temp_gt_sat[0] is True, "ASSERTION FAILED: Temp scaling must preserve GT and drop VLM claim"

            # 3. Isotonic Regression prevents SOOR
            iso_w_vlm = int(round(iso_p * 1000.0))
            assert iso_w_vlm < W_gt, f"Isotonic W_vlm ({iso_w_vlm}) must fall below W_gt ({W_gt})"
            _, iso_claim_sat, iso_gt_sat, iso_soor, _ = verify_with_maxsmt(
                [item["gt_fact"]], [item["vlm_claim"]], [iso_p], gt_weight=W_gt
            )
            assert iso_soor[0] is False, "ASSERTION FAILED: Isotonic calibration must prevent SOOR"
            assert iso_claim_sat[0] is False and iso_gt_sat[0] is True, "ASSERTION FAILED: Isotonic must preserve GT"

            # 4. Conformal APS prevents SOOR
            conf_w_vlm = int(round(conf_w * 1000.0))
            assert conf_w_vlm <= W_gt, f"Conformal W_vlm ({conf_w_vlm}) must fall below or equal W_gt ({W_gt})"
            _, conf_claim_sat, conf_gt_sat, conf_soor, _ = verify_with_maxsmt(
                [item["gt_fact"]], [item["vlm_claim"]], [conf_w], gt_weight=W_gt
            )
            assert conf_soor[0] is False, "ASSERTION FAILED: Conformal APS must prevent SOOR"

    print("\n" + "=" * 80)
    print("🎉 ALL MATHEMATICAL & Z3 SOLVER ASSERTIONS PASSED WITH 100% PRECISION!")
    print("=" * 80)

if __name__ == "__main__":
    run_real_math_smoke_test()
