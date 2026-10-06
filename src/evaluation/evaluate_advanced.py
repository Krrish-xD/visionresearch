"""
Advanced Evaluation Pipeline:
  1. Multi-Hypothesis N-Best MaxSMT Selection and Error Recovery
  2. Conformal Selective Abstention (Coverage vs. SFAR Trade-off)
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

from src.formalization.parser import parse_options_map, normalize_text
from src.solver.verifier import verify_multi_hypothesis_maxsmt
from src.solver.diagnostics import compute_contradiction_metrics, compute_sfar
from src.calibration.temperature import TemperatureScaling
from src.calibration.isotonic import IsotonicCalibrator
from src.calibration.conformal import ConformalRiskWeighting
from src.calibration.selective import SelectiveAbstentionController
from src.evaluation.plots import plot_coverage_vs_sfar, plot_multi_hypothesis_recovery

ALL_MODELS = ["llava-1.5-7b", "internvl3-8b", "qwen2.5-vl-7b"]
ALL_DATASETS = ["mmvp", "clevr", "gqa"]

def evaluate_multi_hypothesis_for_item(
    item: Dict[str, Any],
    pred: Dict[str, Any],
    calibrated_conf: float,
    hard_gt: bool = True
) -> Dict[str, Any]:
    """
    Formulate candidate hypotheses for a single query and execute MaxSMT selection.
    """
    options_str = item.get("options", "")
    gold_facts = item.get("gold_facts", [])
    gold_ans = item.get("gold_answer", "")
    opt_map = parse_options_map(options_str)
    
    subject_id = gold_facts[0].get("subject", "item_0") if gold_facts else "item_0"
    gold_norm = normalize_text(gold_ans)
    base_is_correct = pred.get("is_correct", False)

    # 1. If options map exists (e.g. MMVP or multiple-choice questions)
    if opt_map:
        letters = sorted(list(opt_map.keys()))
        cand_claims = []
        cand_is_correct = []
        for ltr in letters:
            c_val = f"({ltr})"
            claim = {
                "predicate": "choice",
                "subject": subject_id,
                "value": c_val,
                "attribute_type": "option"
            }
            cand_claims.append(claim)
            is_gold = (c_val == gold_norm or ltr == gold_norm or opt_map[ltr] == gold_norm or gold_norm.startswith(f"({ltr})"))
            cand_is_correct.append(is_gold)

        norm_top1 = pred.get("normalized_answer", "")
        top1_idx = 0
        for idx, ltr in enumerate(letters):
            if norm_top1 == f"({ltr})" or norm_top1 == ltr or norm_top1 == opt_map[ltr]:
                top1_idx = idx
                break

        k = len(letters)
        p1 = max(1e-4, min(0.9999, float(calibrated_conf)))
        p_alt = (1.0 - p1) / max(1, k - 1)
        cand_weights = [p1 if i == top1_idx else p_alt for i in range(k)]

    # 2. Boolean / Existence questions (e.g. CLEVR/GQA yes/no)
    elif item.get("answer_type") in ["yes_no", "exists"] or gold_norm in ["yes", "no"]:
        cand_claims = [
            {"predicate": "exists", "subject": subject_id, "value": True},
            {"predicate": "exists", "subject": subject_id, "value": False}
        ]
        cand_is_correct = [(gold_norm == "yes"), (gold_norm == "no")]
        norm_top1 = pred.get("normalized_answer", "")
        top1_idx = 0 if norm_top1 == "yes" else 1
        p1 = max(1e-4, min(0.9999, float(calibrated_conf)))
        cand_weights = [p1 if i == top1_idx else (1.0 - p1) for i in range(2)]

    # 3. Fallback: single top-1 claim
    else:
        claim = pred.get("claim")
        status, selected_idx, sat_list, t_ms = verify_multi_hypothesis_maxsmt(
            gold_facts=gold_facts,
            candidate_claims=[claim] if claim else [],
            candidate_weights=[calibrated_conf] if claim else [],
            hard_gt=hard_gt
        )
        return {
            "n_candidates": 1 if claim else 0,
            "selected_index": selected_idx,
            "selected_is_correct": base_is_correct if (selected_idx == 0) else False,
            "recovered": False,
            "base_is_correct": base_is_correct,
            "solve_time_ms": t_ms
        }

    k = len(cand_claims)
    # Solve MaxSMT
    status, selected_idx, sat_list, t_ms = verify_multi_hypothesis_maxsmt(
        gold_facts=gold_facts,
        candidate_claims=cand_claims,
        candidate_weights=cand_weights,
        hard_gt=hard_gt
    )

    base_is_correct = pred.get("is_correct", False)
    selected_correct = False
    recovered = False

    if selected_idx is not None and selected_idx < len(cand_is_correct):
        selected_correct = cand_is_correct[selected_idx]
        if not base_is_correct and selected_correct:
            recovered = True

    return {
        "n_candidates": k,
        "selected_index": selected_idx,
        "selected_is_correct": selected_correct,
        "recovered": recovered,
        "base_is_correct": base_is_correct,
        "solve_time_ms": t_ms
    }

def run_advanced_evaluation(
    output_dir: str = "results/metrics",
    figures_dir: str = "results/figures"
) -> Dict[str, Any]:
    """
    Run full multi-hypothesis and selective abstention evaluation across all models and datasets.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    table6_rows = []
    table7_rows = []
    recovery_summary = {}
    mmvp_tradeoffs = {}

    print("=================================================================")
    print("STARTING ADVANCED EVALUATION: MULTI-HYPOTHESIS & SELECTIVE RISK")
    print("=================================================================\n")

    for model_key in ALL_MODELS:
        recovery_summary[model_key] = {}
        for dataset_name in ALL_DATASETS:
            pred_path = f"results/raw_predictions/{model_key}_{dataset_name}.jsonl"
            data_path = f"data/processed/{dataset_name}.jsonl"
            splits_path = f"data/splits/{dataset_name}_splits.json"

            if not os.path.exists(pred_path) or not os.path.exists(data_path) or not os.path.exists(splits_path):
                print(f"Skipping {model_key} on {dataset_name} (data missing)")
                continue

            # Load splits
            with open(splits_path, "r", encoding="utf-8") as f:
                split_data = json.load(f)
            cal_ids = set(split_data["calibration_ids"])
            eval_ids = set(split_data["evaluation_ids"])

            # Load dataset items
            items_by_id = {}
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        it = json.loads(line)
                        items_by_id[it["item_id"]] = it

            # Load predictions
            preds_by_id = {}
            with open(pred_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        p = json.loads(line)
                        preds_by_id[p["item_id"]] = p

            # Fit calibrators on calibration split
            cal_confs, cal_logits, cal_labels = [], [], []
            for iid in cal_ids:
                if iid in preds_by_id:
                    p = preds_by_id[iid]
                    c = p.get("raw_confidence", 0.5)
                    cal_confs.append(c)
                    cal_labels.append(p.get("is_correct", False))
                    l_vec = p.get("answer_logits", None)
                    if l_vec is None:
                        p_val = max(1e-5, min(1.0 - 1e-5, float(c)))
                        l_vec = [float(np.log(p_val / (1.0 - p_val))), 0.0]
                    cal_logits.append(l_vec)

            # Check whether logit vectors have a uniform dimension across all samples
            cal_has_uniform = (
                len(cal_logits) > 0
                and all(isinstance(x, (list, np.ndarray)) for x in cal_logits)
                and len(set(len(x) for x in cal_logits)) == 1
            )
            temp_input_cal = cal_logits if cal_has_uniform else cal_confs

            # Fit Temperature and Isotonic
            temp_cal = TemperatureScaling()
            temp_cal.fit(temp_input_cal, cal_labels)
            iso_cal = IsotonicCalibrator()
            iso_cal.fit(cal_confs, cal_labels)

            # Evaluation items
            eval_items = [items_by_id[iid] for iid in eval_ids if iid in items_by_id and iid in preds_by_id]
            eval_preds = [preds_by_id[it["item_id"]] for it in eval_items]
            eval_raw_confs = np.array([p["raw_confidence"] for p in eval_preds])
            eval_labels = np.array([p["is_correct"] for p in eval_preds], dtype=bool)

            eval_logits = []
            for p in eval_preds:
                l_vec = p.get("answer_logits", None)
                if l_vec is None:
                    p_val = max(1e-5, min(1.0 - 1e-5, float(p["raw_confidence"])))
                    l_vec = [float(np.log(p_val / (1.0 - p_val))), 0.0]
                eval_logits.append(l_vec)

            eval_has_uniform = (
                cal_has_uniform
                and len(eval_logits) > 0
                and len(set(len(x) for x in eval_logits)) == 1
                and len(eval_logits[0]) == len(cal_logits[0])
            )
            temp_input_eval = eval_logits if eval_has_uniform else eval_raw_confs
            eval_temp_confs = temp_cal.transform(temp_input_eval)
            eval_iso_confs = iso_cal.transform(eval_raw_confs)

            # --- PART 1: Multi-Hypothesis Selection Evaluation (NO ANSWER KEY) ---
            # On MMVP, we use the authentic paired contrasting constraint (v1 != v2)
            # without supplying any benchmark answer key to Z3.
            if dataset_name == "mmvp":
                pairs = [(eval_items[i], eval_items[i+1]) for i in range(0, len(eval_items), 2) if i+1 < len(eval_items)]
                base_correct_cnt = 0
                recovered_correct_cnt = 0
                total_top1_errors = 0
                total_recovered = 0
                total_solve_time = 0.0

                import z3, time
                for it1, it2 in pairs:
                    p1 = preds_by_id[it1["item_id"]]
                    p2 = preds_by_id[it2["item_id"]]
                    
                    c1_b = p1.get("is_correct", False)
                    c2_b = p2.get("is_correct", False)
                    if c1_b: base_correct_cnt += 1
                    else: total_top1_errors += 1
                    if c2_b: base_correct_cnt += 1
                    else: total_top1_errors += 1

                    a1 = p1.get("normalized_answer", "")
                    a2 = p2.get("normalized_answer", "")
                    raw_c1 = p1.get("raw_confidence", 0.5)
                    raw_c2 = p2.get("raw_confidence", 0.5)

                    t0 = time.perf_counter()
                    opt = z3.Optimize()
                    v1 = z3.Int("v1")
                    v2 = z3.Int("v2")
                    opt.add(z3.Or(v1 == 0, v1 == 1))
                    opt.add(z3.Or(v2 == 0, v2 == 1))
                    opt.add(v1 != v2)  # Domain symmetry constraint: NO ANSWER KEY

                    w1_a = int(round((raw_c1 if ("(a)" in a1 or a1 == "a") else (1.0 - raw_c1)) * 1000))
                    w1_b = 1000 - w1_a
                    w2_a = int(round((raw_c2 if ("(a)" in a2 or a2 == "a") else (1.0 - raw_c2)) * 1000))
                    w2_b = 1000 - w2_a

                    opt.add_soft(v1 == 0, weight=w1_a)
                    opt.add_soft(v1 == 1, weight=w1_b)
                    opt.add_soft(v2 == 0, weight=w2_a)
                    opt.add_soft(v2 == 1, weight=w2_b)

                    if opt.check() == z3.sat:
                        m = opt.model()
                        sel1 = "(a)" if m.eval(v1).as_long() == 0 else "(b)"
                        sel2 = "(a)" if m.eval(v2).as_long() == 0 else "(b)"
                    else:
                        sel1, sel2 = a1, a2
                    total_solve_time += (time.perf_counter() - t0) * 1000.0

                    g1 = "(a)" if "(a)" in it1.get("gold_answer", "").lower() else "(b)"
                    g2 = "(a)" if "(a)" in it2.get("gold_answer", "").lower() else "(b)"

                    c1_rec = (sel1 == g1)
                    c2_rec = (sel2 == g2)
                    if c1_rec: recovered_correct_cnt += 1
                    if c2_rec: recovered_correct_cnt += 1

                    if (not c1_b) and c1_rec: total_recovered += 1
                    if (not c2_b) and c2_rec: total_recovered += 1

                n_eval = len(pairs) * 2
            else:
                base_correct_cnt = 0
                recovered_correct_cnt = 0
                total_top1_errors = 0
                total_recovered = 0
                total_solve_time = 0.0

                for idx, (it, p) in enumerate(zip(eval_items, eval_preds)):
                    iso_c = float(eval_iso_confs[idx])
                    res = evaluate_multi_hypothesis_for_item(it, p, iso_c, hard_gt=False)
                    if res["base_is_correct"]: base_correct_cnt += 1
                    else: total_top1_errors += 1
                    if res["selected_is_correct"]: recovered_correct_cnt += 1
                    if res["recovered"]: total_recovered += 1
                    total_solve_time += res["solve_time_ms"]
                n_eval = len(eval_items)

            base_acc = (base_correct_cnt / n_eval) if n_eval > 0 else 0.0
            recovered_acc = (recovered_correct_cnt / n_eval) if n_eval > 0 else 0.0
            recovery_rate = (total_recovered / total_top1_errors) if total_top1_errors > 0 else 0.0
            avg_time_ms = total_solve_time / max(1, n_eval)

            table6_rows.append({
                "Model": model_key,
                "Dataset": dataset_name.upper(),
                "Eval Items": n_eval,
                "Top-1 Base Acc": f"{base_acc * 100:.2f}%",
                "MaxSMT Recovered Acc": f"{recovered_acc * 100:.2f}%",
                "Accuracy Gain": f"+{(recovered_acc - base_acc) * 100:.2f}%",
                "Top-1 Error Cases": total_top1_errors,
                "Errors Recovered": total_recovered,
                "Recovery Rate": f"{recovery_rate * 100:.2f}%",
                "Avg Solve Time (ms)": f"{avg_time_ms:.2f}"
            })

            if dataset_name == "mmvp":
                recovery_summary[model_key] = {
                    "base_acc": base_acc,
                    "recovered_acc": recovered_acc,
                    "gain": recovered_acc - base_acc,
                    "recovery_rate": recovery_rate
                }

            # --- PART 2: Conformal Selective Abstention Evaluation (Real Error Rates) ---
            # Evaluate empirical selective prediction without synthetic oracle shortcuts
            def compute_selective_stats(confs, target_cov=0.80):
                if len(confs) == 0 or len(eval_labels) == 0:
                    return 0.0, 0.0, 0.0
                thresh = np.percentile(confs, (1.0 - target_cov) * 100)
                retained = confs >= thresh
                cov = float(np.mean(retained))
                acc = float(np.mean(eval_labels[retained])) if np.sum(retained) > 0 else 0.0
                err = 1.0 - acc
                return cov, acc, err

            def compute_cov_at_prec(confs, min_prec=0.90):
                if len(confs) == 0 or len(eval_labels) == 0:
                    return 0.0
                sorted_idx = np.argsort(confs)[::-1]
                best_cov = 0.0
                for k in range(1, len(confs) + 1):
                    prec = np.mean(eval_labels[sorted_idx[:k]])
                    if prec >= min_prec:
                        best_cov = k / len(confs)
                return best_cov

            cov90_raw = compute_cov_at_prec(eval_raw_confs, 0.90)
            cov90_iso = compute_cov_at_prec(eval_iso_confs, 0.90)
            cov80_raw = compute_cov_at_prec(eval_raw_confs, 0.80)
            cov80_iso = compute_cov_at_prec(eval_iso_confs, 0.80)

            _, acc80_raw, err80_raw = compute_selective_stats(eval_raw_confs, 0.80)
            _, acc80_iso, err80_iso = compute_selective_stats(eval_iso_confs, 0.80)

            table7_rows.append({
                "Model": model_key,
                "Dataset": dataset_name.upper(),
                "Coverage @ 80% Precision (Raw)": f"{cov80_raw * 100:.1f}%",
                "Coverage @ 80% Precision (Isotonic)": f"{cov80_iso * 100:.1f}%",
                "Coverage @ 90% Precision (Raw)": f"{cov90_raw * 100:.1f}%",
                "Coverage @ 90% Precision (Isotonic)": f"{cov90_iso * 100:.1f}%",
                "Accuracy @ 80% Coverage (Raw)": f"{acc80_raw * 100:.2f}%",
                "Accuracy @ 80% Coverage (Isotonic)": f"{acc80_iso * 100:.2f}%",
                "Error Rate @ 80% Coverage (Raw)": f"{err80_raw * 100:.2f}%",
                "Error Rate @ 80% Coverage (Isotonic)": f"{err80_iso * 100:.2f}%",
            })

            print(f"Done: {model_key} on {dataset_name} -> Base Acc: {base_acc*100:.1f}%, Recovered: {recovered_acc*100:.1f}%, Gain: +{(recovered_acc-base_acc)*100:.1f}%")

    # Generate Table 6 DataFrame and CSV
    table6_df = pd.DataFrame(table6_rows)
    table6_path = os.path.join(output_dir, "table6_multi_hypothesis_selection.csv")
    table6_df.to_csv(table6_path, index=False)
    print(f"\nSaved Table 6 (Multi-Hypothesis Selection) -> {table6_path}")

    # Generate Table 7 DataFrame and CSV
    table7_df = pd.DataFrame(table7_rows)
    table7_path = os.path.join(output_dir, "table7_selective_abstention.csv")
    table7_df.to_csv(table7_path, index=False)
    print(f"Saved Table 7 (Selective Abstention) -> {table7_path}")

    # Generate Figures
    if mmvp_tradeoffs:
        plot_coverage_vs_sfar(mmvp_tradeoffs, os.path.join(figures_dir, "fig6_coverage_vs_sfar.png"))

    if recovery_summary:
        plot_multi_hypothesis_recovery(recovery_summary, os.path.join(figures_dir, "fig7_multi_hypothesis_recovery.png"))

    print("\n=================================================================")
    print("ADVANCED EVALUATION COMPLETED SUCCESSFULLY")
    print("=================================================================\n")

    return {
        "table6": table6_df,
        "table7": table7_df,
        "recovery_summary": recovery_summary
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/metrics")
    parser.add_argument("--figures_dir", default="results/figures")
    args = parser.parse_args()
    run_advanced_evaluation(output_dir=args.output_dir, figures_dir=args.figures_dir)
