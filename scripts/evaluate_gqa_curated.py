"""
evaluate_gqa_curated.py
========================
End-to-end evaluation pipeline for the 6,000-item GQA curated benchmark
across all three VLM model families:
  1. internvl3-8b
  2. llava-1.5-7b
  3. qwen2.5-vl-7b

Evaluates:
  - Part 1: Post-hoc Calibration (Temperature Scaling, Isotonic Regression, Conformal APS)
  - Part 2: SMT Solver Soundness, SFAR, and SOOR Interception (6 conditions)
  - Part 3: Multi-Hypothesis N-Best MaxSMT Active Error Recovery
  - Part 4: Conformal Selective Abstention (Coverage vs. Precision/Risk Trade-off)
  - Part 5: Fine-Grained Cognitive Slice Breakdown (Existence, Spatial, Comparative, Attribute)
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

# Add repository root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.calibration.temperature import TemperatureScaling
from src.calibration.isotonic import IsotonicCalibrator
from src.calibration.conformal import AdaptivePredictionSets, ConformalRiskWeighting
from src.calibration.metrics import compute_ece, compute_brier_score, compute_nll
from src.solver.z3_encoder import encode_fact_to_z3
from src.solver.verifier import check_hard_contradiction, verify_with_maxsmt, verify_multi_hypothesis_maxsmt
from src.solver.diagnostics import compute_contradiction_metrics, compute_sfar, compute_soor

MODELS = ["internvl3-8b", "llava-1.5-7b", "qwen2.5-vl-7b"]
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "gqa_curated_6000.jsonl")
SPLITS_PATH = os.path.join(PROJECT_ROOT, "data", "splits", "gqa_curated_6000_splits.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "gqa_curated_eval")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_or_load_splits(items: List[Dict[str, Any]], cal_ratio: float = 0.40, seed: int = 42) -> Tuple[List[str], List[str]]:
    """Create deterministic stratified 40/60 splits (2,400 cal / 3,600 eval)."""
    if os.path.exists(SPLITS_PATH):
        with open(SPLITS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["calibration_ids"], data["evaluation_ids"]

    rng = np.random.RandomState(seed)
    by_category = {}
    for it in items:
        cat = it.get("category", "default")
        by_category.setdefault(cat, []).append(it["item_id"])

    cal_ids = []
    eval_ids = []
    for cat, ids in by_category.items():
        shuffled = list(ids)
        rng.shuffle(shuffled)
        n_cal = int(round(len(shuffled) * cal_ratio))
        cal_ids.extend(shuffled[:n_cal])
        eval_ids.extend(shuffled[n_cal:])

    split_obj = {
        "dataset_file": "data/processed/gqa_curated_6000.jsonl",
        "seed": seed,
        "cal_ratio": cal_ratio,
        "eval_ratio": 1.0 - cal_ratio,
        "total_items": len(items),
        "calibration_count": len(cal_ids),
        "evaluation_count": len(eval_ids),
        "calibration_ids": cal_ids,
        "evaluation_ids": eval_ids
    }
    with open(SPLITS_PATH, "w", encoding="utf-8") as f:
        json.dump(split_obj, f, indent=2)
    print(f"Created stratified splits: {len(cal_ids)} cal, {len(eval_ids)} eval -> {SPLITS_PATH}")
    return cal_ids, eval_ids


def get_aligned_claim(item: Dict[str, Any], pred: Dict[str, Any]) -> Dict[str, Any]:
    """Align parsed claim predicate with benchmark gold fact for exact formal resolution."""
    gold_facts = item.get("gold_facts", [])
    if not gold_facts:
        return pred.get("claim")
    gf = gold_facts[0]
    g_pred = gf.get("predicate")
    norm_ans = pred.get("normalized_answer", "")

    if g_pred in ["relation", "exists"]:
        is_yes = (norm_ans in ["yes", "true", "present", "correct"])
        cl = dict(gf)
        cl["value"] = is_yes
        return cl
    elif g_pred == "count":
        try:
            val = int(norm_ans)
        except Exception:
            val = -999
        cl = dict(gf)
        cl["value"] = val
        return cl
    elif g_pred == "attribute":
        cl = dict(gf)
        cl["value"] = norm_ans
        return cl
    return pred.get("claim")


def run_gqa_evaluation():
    print("=" * 80)
    print("🔬 GQA CURATED 6,000-ITEM BENCHMARK: NEURO-SYMBOLIC MASTER EVALUATION")
    print("=" * 80)

    # 1. Load dataset items
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        all_items = [json.loads(line) for line in f if line.strip()]
    items_by_id = {it["item_id"]: it for it in all_items}
    print(f"Loaded {len(all_items)} GQA curated dataset items.")

    cal_ids, eval_ids = create_or_load_splits(all_items)
    cal_id_set = set(cal_ids)
    eval_id_set = set(eval_ids)

    calibration_table_rows = []
    smt_soundness_table_rows = []
    recovery_table_rows = []
    selective_table_rows = []
    slice_table_rows = []

    for model_key in MODELS:
        pred_path = os.path.join(PROJECT_ROOT, "results", "curated_6000", f"{model_key}_gqa_curated_6000.jsonl")
        if not os.path.exists(pred_path):
            print(f"Warning: {pred_path} not found. Skipping {model_key}.")
            continue

        with open(pred_path, "r", encoding="utf-8") as f:
            all_preds = [json.loads(line) for line in f if line.strip()]
        preds_by_id = {p["item_id"]: p for p in all_preds}
        print(f"\nProcessing {model_key}: {len(all_preds)} total predictions...")

        # -------------------------------------------------------------
        # Part 1: Fit Calibrators on Calibration Split (2,400 items)
        # -------------------------------------------------------------
        cal_items = [items_by_id[iid] for iid in cal_ids if iid in items_by_id and iid in preds_by_id]
        cal_preds = [preds_by_id[it["item_id"]] for it in cal_items]

        cal_confs = np.array([p["raw_confidence"] for p in cal_preds])
        cal_labels = np.array([p["is_correct"] for p in cal_preds], dtype=bool)

        cal_logits = []
        for p in cal_preds:
            p_val = max(1e-5, min(1.0 - 1e-5, float(p["raw_confidence"])))
            cal_logits.append([float(np.log(p_val / (1.0 - p_val))), 0.0])

        temp_cal = TemperatureScaling()
        temp_cal.fit(cal_logits, cal_labels)

        iso_cal = IsotonicCalibrator()
        iso_cal.fit(cal_confs, cal_labels)

        conf_cal = AdaptivePredictionSets(alpha=0.10)
        conf_cal.fit(cal_logits, cal_labels)

        # -------------------------------------------------------------
        # Part 2: Calibration Evaluation on Held-Out Split (3,600 items)
        # -------------------------------------------------------------
        eval_items = [items_by_id[iid] for iid in eval_ids if iid in items_by_id and iid in preds_by_id]
        eval_preds = [preds_by_id[it["item_id"]] for it in eval_items]

        eval_raw_confs = np.array([p["raw_confidence"] for p in eval_preds])
        eval_labels = np.array([p["is_correct"] for p in eval_preds], dtype=bool)

        eval_logits = []
        for p in eval_preds:
            p_val = max(1e-5, min(1.0 - 1e-5, float(p["raw_confidence"])))
            eval_logits.append([float(np.log(p_val / (1.0 - p_val))), 0.0])

        eval_temp_confs = temp_cal.transform(eval_logits)
        eval_iso_confs = iso_cal.transform(eval_raw_confs)
        eval_aps_weights = conf_cal.transform(eval_logits)

        # Calibration Metrics using evaluate_calibration
        from src.calibration.metrics import evaluate_calibration
        metrics_raw = evaluate_calibration(eval_raw_confs, eval_labels)
        metrics_temp = evaluate_calibration(eval_temp_confs, eval_labels)
        metrics_iso = evaluate_calibration(eval_iso_confs, eval_labels)

        ece10_raw, ece15_raw = metrics_raw["ece_10"], metrics_raw["ece_15"]
        brier_raw, nll_raw = metrics_raw["brier_score"], metrics_raw["nll"]

        ece10_temp, ece15_temp = metrics_temp["ece_10"], metrics_temp["ece_15"]
        brier_temp, nll_temp = metrics_temp["brier_score"], metrics_temp["nll"]

        ece10_iso, ece15_iso = metrics_iso["ece_10"], metrics_iso["ece_15"]
        brier_iso, nll_iso = metrics_iso["brier_score"], metrics_iso["nll"]

        calibration_table_rows.extend([
            {"Model": model_key, "Method": "Raw Confidence", "ECE (10-bin)": f"{ece10_raw:.4f}", "ECE (15-bin)": f"{ece15_raw:.4f}", "Brier Score": f"{brier_raw:.4f}", "NLL": f"{nll_raw:.4f}"},
            {"Model": model_key, "Method": "Temperature Scaled", "ECE (10-bin)": f"{ece10_temp:.4f}", "ECE (15-bin)": f"{ece15_temp:.4f}", "Brier Score": f"{brier_temp:.4f}", "NLL": f"{nll_temp:.4f}"},
            {"Model": model_key, "Method": "Isotonic (Ours)", "ECE (10-bin)": f"{ece10_iso:.4f}", "ECE (15-bin)": f"{ece15_iso:.4f}", "Brier Score": f"{brier_iso:.4f}", "NLL": f"{nll_iso:.4f}"},
        ])

        # -------------------------------------------------------------
        # Part 3: SMT Solver Soundness & SOOR Verification (6 conditions)
        # -------------------------------------------------------------
        conditions = ["hard_claim_check", "uniform_soft", "raw_confidence", "temperature_scaled", "isotonic", "conformal_risk"]
        flagged_contra = {c: [] for c in conditions}
        soor_flags = {c: [] for c in conditions}

        weights_by_cond = {
            "uniform_soft": np.ones(len(eval_preds)),
            "raw_confidence": eval_raw_confs,
            "temperature_scaled": eval_temp_confs,
            "isotonic": eval_iso_confs,
            "conformal_risk": eval_aps_weights
        }

        for idx, (it, p) in enumerate(zip(eval_items, eval_preds)):
            gf = it["gold_facts"]
            cl = get_aligned_claim(it, p)

            # 1. Hard claim check
            st, is_c, _ = check_hard_contradiction(gf, cl)
            flagged_contra["hard_claim_check"].append(is_c)
            soor_flags["hard_claim_check"].append(False)

            # MaxSMT soft conditions
            for cond in ["uniform_soft", "raw_confidence", "temperature_scaled", "isotonic", "conformal_risk"]:
                w = float(weights_by_cond[cond][idx])
                _, sat_list, _, soor_list, _ = verify_with_maxsmt(gf, [cl], [w], gt_weight=500, hard_gt=False)
                is_sat = sat_list[0] if sat_list else False
                is_soor = soor_list[0] if soor_list else False
                flagged_contra[cond].append(not is_sat)
                soor_flags[cond].append(is_soor)

        for cond in conditions:
            f_arr = np.array(flagged_contra[cond], dtype=bool)
            soor_arr = np.array(soor_flags[cond], dtype=bool)
            
            # Ground truth contradiction is when the model was wrong (~eval_labels)
            metrics_dict = compute_contradiction_metrics(eval_labels, f_arr, soor_triggered_list=soor_arr)
            p_val = metrics_dict["precision"]
            r_val = metrics_dict["recall"]
            f1_val = metrics_dict["f1"]
            sfar_val = metrics_dict["sfar"]
            soor_val = metrics_dict["soor"]
            verif_acc = metrics_dict["accuracy"]

            # Compute 95% bootstrap confidence interval on F1
            rng = np.random.RandomState(42)
            n_samples = len(eval_labels)
            boot_f1s = []
            for _ in range(200):
                b_idx = rng.randint(0, n_samples, n_samples)
                b_m = compute_contradiction_metrics(eval_labels[b_idx], f_arr[b_idx])
                boot_f1s.append(b_m["f1"])
            ci_lower = float(np.percentile(boot_f1s, 2.5))
            ci_upper = float(np.percentile(boot_f1s, 97.5))

            cond_display = {
                "hard_claim_check": "Hard Claim Check",
                "uniform_soft": "Uniform Soft",
                "raw_confidence": "Raw Confidence",
                "temperature_scaled": "Temperature Scaled",
                "isotonic": "Isotonic",
                "conformal_risk": "Conformal Risk"
            }.get(cond, cond)

            smt_soundness_table_rows.append({
                "Model": model_key,
                "Condition": cond_display,
                "Precision": f"{p_val:.4f}",
                "Recall": f"{r_val:.4f}",
                "F1 Score [95% CI]": f"{f1_val:.4f} [{ci_lower:.3f}, {ci_upper:.3f}]",
                "SFAR": f"{sfar_val:.4f}",
                "SOOR": f"{soor_val:.4f}",
                "Verif. Acc.": f"{verif_acc * 100:.2f}%"
            })

        # -------------------------------------------------------------
        # Part 4: Multi-Hypothesis N-Best MaxSMT Error Recovery
        # -------------------------------------------------------------
        base_correct_cnt = 0
        recovered_correct_cnt = 0
        total_top1_errors = 0
        total_recovered = 0
        total_solve_time = 0.0

        for idx, (it, p) in enumerate(zip(eval_items, eval_preds)):
            gf = it["gold_facts"]
            if not gf:
                continue
            ref_gf = gf[0]
            g_pred = ref_gf.get("predicate")
            base_c = p["is_correct"]
            if base_c: base_correct_cnt += 1
            else: total_top1_errors += 1

            iso_c = float(eval_iso_confs[idx])

            # Formulate competing candidate hypotheses
            if g_pred in ["exists", "relation"]:
                cands = [dict(ref_gf, value=True), dict(ref_gf, value=False)]
                is_yes_top1 = (p.get("normalized_answer") in ["yes", "true", "present", "correct"])
                top_idx = 0 if is_yes_top1 else 1
                cand_weights = [iso_c if i == top_idx else (1.0 - iso_c) for i in range(2)]
                cand_correct = [(ref_gf.get("value") is True), (ref_gf.get("value") is False)]

            elif g_pred == "count":
                try: top_count = int(p.get("normalized_answer"))
                except: top_count = 0
                cands = [dict(ref_gf, value=cnt) for cnt in range(6)]
                cand_weights = [iso_c if cnt == top_count else (1.0 - iso_c) / 5.0 for cnt in range(6)]
                cand_correct = [(cnt == ref_gf.get("value")) for cnt in range(6)]

            elif g_pred == "attribute":
                # Open attribute without discrete prompt options: only asserted claim, zero gold leakage
                norm_val = str(p.get("normalized_answer", ""))
                cands = [dict(ref_gf, value=norm_val)]
                cand_weights = [iso_c]
                cand_correct = [base_c]
            else:
                cands = [get_aligned_claim(it, p)]
                cand_weights = [iso_c]
                cand_correct = [base_c]

            t0 = time.perf_counter()
            st, sel_idx, sat_list, t_ms = verify_multi_hypothesis_maxsmt(
                gold_facts=gf,
                candidate_claims=cands,
                candidate_weights=cand_weights,
                hard_gt=True
            )
            total_solve_time += t_ms

            sel_c = False
            if sel_idx is not None and sel_idx < len(cand_correct):
                sel_c = cand_correct[sel_idx]
            if sel_c: recovered_correct_cnt += 1
            if (not base_c) and sel_c: total_recovered += 1

        n_eval = len(eval_items)
        base_acc = base_correct_cnt / n_eval
        rec_acc = recovered_correct_cnt / n_eval
        rec_rate = total_recovered / max(1, total_top1_errors)
        avg_solve_ms = total_solve_time / max(1, n_eval)

        recovery_table_rows.append({
            "Model": model_key,
            "Total Evaluated": n_eval,
            "Top-1 Errors": total_top1_errors,
            "Base Top-1 Acc.": f"{base_acc * 100:.2f}%",
            "MaxSMT Recov. Acc": f"{rec_acc * 100:.2f}%",
            "Accuracy Gain": f"+{(rec_acc - base_acc) * 100:.2f}%",
            "Errors Recovered": f"{total_recovered} / {total_top1_errors}",
            "Recovery Rate": f"{rec_rate * 100:.2f}%",
            "Solve Time": f"{avg_solve_ms:.2f} ms"
        })

        # -------------------------------------------------------------
        # Part 5: Selective Abstention Evaluation
        # -------------------------------------------------------------
        def compute_selective_stats(confs, target_cov=0.80):
            thresh = np.percentile(confs, (1.0 - target_cov) * 100)
            ret = confs >= thresh
            cov = float(np.mean(ret))
            acc = float(np.mean(eval_labels[ret])) if np.sum(ret) > 0 else 0.0
            return cov, acc, 1.0 - acc

        def compute_cov_at_prec(confs, min_prec=0.80):
            sorted_idx = np.argsort(confs)[::-1]
            best_cov = 0.0
            for k in range(1, len(confs) + 1):
                prec = np.mean(eval_labels[sorted_idx[:k]])
                if prec >= min_prec:
                    best_cov = k / len(confs)
            return best_cov

        cov80_raw = compute_cov_at_prec(eval_raw_confs, 0.80)
        cov80_iso = compute_cov_at_prec(eval_iso_confs, 0.80)
        cov90_raw = compute_cov_at_prec(eval_raw_confs, 0.90)
        cov90_iso = compute_cov_at_prec(eval_iso_confs, 0.90)

        _, acc80_raw, err80_raw = compute_selective_stats(eval_raw_confs, 0.80)
        _, acc80_iso, err80_iso = compute_selective_stats(eval_iso_confs, 0.80)

        selective_table_rows.append({
            "Model": model_key,
            "Coverage @ 80% Prec. (Raw)": f"{cov80_raw * 100:.1f}%",
            "Coverage @ 80% Prec. (Iso.)": f"{cov80_iso * 100:.1f}%",
            "Coverage @ 90% Prec. (Raw)": f"{cov90_raw * 100:.1f}%",
            "Coverage @ 90% Prec. (Iso.)": f"{cov90_iso * 100:.1f}%",
            "Accuracy @ 80% (Raw)": f"{acc80_raw * 100:.2f}%",
            "Accuracy @ 80% (Iso.)": f"{acc80_iso * 100:.2f}%",
            "Error Rate @ 80% (Raw)": f"{err80_raw * 100:.2f}%",
            "Error Rate @ 80% (Iso.)": f"{err80_iso * 100:.2f}%"
        })

        # -------------------------------------------------------------
        # Part 6: Cognitive Slice Breakdown
        # -------------------------------------------------------------
        for cat in ["binary_existence", "spatial_relation", "comparative", "object_attribute"]:
            cat_mask = np.array([it["category"] == cat for it in eval_items], dtype=bool)
            cat_n = int(np.sum(cat_mask))
            if cat_n == 0:
                continue
            cat_labels = eval_labels[cat_mask]
            cat_hard_f = np.array(flagged_contra["hard_claim_check"])[cat_mask]
            cat_soft_soor = np.array(soor_flags["raw_confidence"])[cat_mask]

            cat_m = compute_contradiction_metrics(cat_labels, cat_hard_f)
            f1_c = cat_m["f1"]
            sfar_hard = cat_m["sfar"]
            soor_soft = compute_soor(cat_labels, cat_soft_soor)
            cat_base_acc = float(np.mean(cat_labels))

            slice_table_rows.append({
                "Model": model_key,
                "Category": cat,
                "Items": cat_n,
                "Base Acc": f"{cat_base_acc * 100:.2f}%",
                "Hard SFAR": f"{sfar_hard:.4f}",
                "Hard F1": f"{f1_c:.4f}",
                "Soft SOOR": f"{soor_soft * 100:.2f}%"
            })

    # Save DataFrames
    df_cal = pd.DataFrame(calibration_table_rows)
    df_smt = pd.DataFrame(smt_soundness_table_rows)
    df_rec = pd.DataFrame(recovery_table_rows)
    df_sel = pd.DataFrame(selective_table_rows)
    df_slc = pd.DataFrame(slice_table_rows)

    df_cal.to_csv(os.path.join(OUTPUT_DIR, "table_gqa_calibration.csv"), index=False)
    df_smt.to_csv(os.path.join(OUTPUT_DIR, "table_gqa_smt_soundness.csv"), index=False)
    df_rec.to_csv(os.path.join(OUTPUT_DIR, "table_gqa_recovery.csv"), index=False)
    df_sel.to_csv(os.path.join(OUTPUT_DIR, "table_gqa_selective.csv"), index=False)
    df_slc.to_csv(os.path.join(OUTPUT_DIR, "table_gqa_slices.csv"), index=False)

    print("\n" + "=" * 80)
    print("TABLE 1: GQA CALIBRATION (ECE, Brier, NLL)")
    print("=" * 80)
    print(df_cal.to_string(index=False))

    print("\n" + "=" * 80)
    print("TABLE 2: GQA SMT SOUNDNESS, SFAR & SOOR (3,600 Eval Items)")
    print("=" * 80)
    print(df_smt.to_string(index=False))

    print("\n" + "=" * 80)
    print("TABLE 3: GQA MULTI-HYPOTHESIS MAXSMT ERROR RECOVERY")
    print("=" * 80)
    print(df_rec.to_string(index=False))

    print("\n" + "=" * 80)
    print("TABLE 4: GQA SELECTIVE ABSTENTION & RISK CONTROL")
    print("=" * 80)
    print(df_sel.to_string(index=False))

    print("\n" + "=" * 80)
    print("TABLE 5: GQA COGNITIVE SLICE BREAKDOWN")
    print("=" * 80)
    print(df_slc.to_string(index=False))

    print("\nSaved all tables to:", OUTPUT_DIR)


if __name__ == "__main__":
    run_gqa_evaluation()
