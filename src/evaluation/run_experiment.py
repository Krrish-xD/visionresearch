"""Unified Experiment Pipeline for Calibrated Confidence in Symbolic Error Detection."""

import os
import json
import argparse
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any, List

from src.datasets.prepare import prepare_all
from src.evaluation.make_splits import make_all_splits
from src.vlm.evaluate import run_predictions_on_dataset
from src.solver.verifier import check_hard_contradiction, verify_with_maxsmt
from src.solver.diagnostics import compute_contradiction_metrics, compute_solve_time_stats
from src.calibration.temperature import TemperatureScaling
from src.calibration.isotonic import IsotonicCalibrator
from src.calibration.conformal import ConformalRiskWeighting
from src.calibration.metrics import evaluate_calibration
from src.evaluation.stats import mcnemar_test, bootstrap_metric_ci, wilcoxon_paired_test
from src.evaluation.tables import (
    generate_table_1_dataset_composition,
    generate_table_2_calibration,
    generate_table_3_contradiction,
    generate_table_4_category_sfar,
    generate_table_5_solver_runtime,
    save_table_bundle
)
from src.evaluation.plots import (
    plot_reliability_diagrams,
    plot_f1_bootstrap_ci,
    plot_sfar_by_category,
    plot_ece_vs_sfar
)

def run_experiment_pipeline(
    dataset_name: str = "mmvp",
    model_key: str = "llava-1.5-7b",
    config_path: str = "configs/experiment.yaml",
    pilot: bool = False,
    mock: bool = False
) -> Dict[str, Any]:
    """Execute complete end-to-end evaluation pipeline."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    metrics_dir = cfg.get("output", {}).get("metrics_dir", "results/metrics")
    figs_dir = cfg.get("output", {}).get("figures_dir", "results/figures")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    # 1. Ensure dataset exists
    data_file = f"data/processed/{dataset_name}.jsonl"
    if not os.path.exists(data_file):
        print(f"Dataset {data_file} not found. Running data preparation...")
        prepare_all(config_path)

    # 2. Ensure splits exist
    splits_file = f"data/splits/{dataset_name}_splits.json"
    if not os.path.exists(splits_file):
        print(f"Splits {splits_file} not found. Creating deterministic splits...")
        make_all_splits(config_path)

    with open(splits_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    cal_ids = set(split_data["calibration_ids"])
    eval_ids = set(split_data["evaluation_ids"])

    # Load dataset items into dict
    items_by_id = {}
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                it = json.loads(line)
                items_by_id[it["item_id"]] = it

    # If pilot mode, subset evaluation items to 50 items
    if pilot:
        print("PILOT MODE: Limiting evaluation to 50 items.")
        eval_ids = set(list(eval_ids)[:50])

    # 3. Ensure predictions exist
    pred_file = f"results/raw_predictions/{model_key}_{dataset_name}.jsonl"
    if not os.path.exists(pred_file):
        print(f"Predictions {pred_file} not found. Running VLM inference...")
        run_predictions_on_dataset(dataset_name=dataset_name, model_key=model_key, mock=mock)

    preds_by_id = {}
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                preds_by_id[p["item_id"]] = p

    # 4. Calibration Split: Extract confidences and fit calibrators
    cal_confs = []
    cal_labels = []

    for item_id in cal_ids:
        if item_id in preds_by_id:
            p = preds_by_id[item_id]
            cal_confs.append(p["raw_confidence"])
            cal_labels.append(p["is_correct"])

    print(f"Fitting calibrators on {len(cal_confs)} calibration items...")
    temp_calibrator = TemperatureScaling()
    temp_calibrator.fit(cal_confs, cal_labels)
    print(f"Fitted Temperature: T = {temp_calibrator.temperature:.4f}")

    iso_calibrator = IsotonicCalibrator()
    iso_calibrator.fit(cal_confs, cal_labels)

    conf_calibrator = ConformalRiskWeighting(alpha=0.10)
    conf_calibrator.fit(cal_confs, cal_labels)
    print(f"Fitted Conformal Threshold: q_hat = {conf_calibrator.q_hat:.4f}")

    # 5. Evaluation Split: Transform confidences and run MaxSMT solver across all 6 conditions
    eval_raw_confs = []
    eval_labels = []
    per_item_rows = []

    eval_items_ordered = [items_by_id[iid] for iid in eval_ids if iid in items_by_id and iid in preds_by_id]
    print(f"Evaluating solver on {len(eval_items_ordered)} held-out items across 6 conditions...")

    # Compute calibrated confidences
    raw_confs_arr = np.array([preds_by_id[it["item_id"]]["raw_confidence"] for it in eval_items_ordered])
    temp_confs_arr = temp_calibrator.transform(raw_confs_arr)
    iso_confs_arr = iso_calibrator.transform(raw_confs_arr)
    conf_weights_arr = conf_calibrator.transform(raw_confs_arr)

    eval_labels = [preds_by_id[it["item_id"]]["is_correct"] for it in eval_items_ordered]

    # Store condition flags
    conditions = ["hard_claim_check", "uniform_soft", "raw_confidence", "temperature_scaled", "isotonic", "conformal_risk"]
    flagged_contradicted = {c: [] for c in conditions}
    solve_times = {c: [] for c in conditions}

    for idx, item in enumerate(eval_items_ordered):
        item_id = item["item_id"]
        pred = preds_by_id[item_id]
        claim = pred["claim"]
        gold_facts = item["gold_facts"]

        raw_c = float(raw_confs_arr[idx])
        temp_c = float(temp_confs_arr[idx])
        iso_c = float(iso_confs_arr[idx])
        conf_w = float(conf_weights_arr[idx])

        row = {
            "item_id": item_id,
            "dataset": dataset_name,
            "category": item["category"],
            "question": item["question"],
            "gold_answer": item["gold_answer"],
            "raw_answer": pred["raw_answer"],
            "normalized_answer": pred["normalized_answer"],
            "is_correct": pred["is_correct"],
            "raw_confidence": raw_c,
            "temp_confidence": temp_c,
            "isotonic_confidence": iso_c,
            "conformal_weight": conf_w,
            "parse_status": pred["parse_status"],
            "model": model_key
        }

        # 1. Hard contradiction oracle
        if claim is not None:
            status, is_contra, t_ms = check_hard_contradiction(gold_facts, claim)
            flagged_contradicted["hard_claim_check"].append(is_contra)
            solve_times["hard_claim_check"].append(t_ms)
            row["hard_verdict"] = "contradicted" if is_contra else "sat"
            row["hard_solve_ms"] = t_ms
        else:
            flagged_contradicted["hard_claim_check"].append(True)
            solve_times["hard_claim_check"].append(0.0)
            row["hard_verdict"] = "unparsed"
            row["hard_solve_ms"] = 0.0

        # MaxSMT conditions
        weights_map = {
            "uniform_soft": 1.0,
            "raw_confidence": raw_c,
            "temperature_scaled": temp_c,
            "isotonic": iso_c,
            "conformal_risk": conf_w
        }

        for cond_name, weight_val in weights_map.items():
            if claim is not None:
                status, sat_list, t_ms = verify_with_maxsmt(gold_facts, [claim], [weight_val])
                is_sat = sat_list[0] if sat_list else False
                is_contra = not is_sat
                flagged_contradicted[cond_name].append(is_contra)
                solve_times[cond_name].append(t_ms)
                row[f"{cond_name}_verdict"] = "sat" if is_sat else "unsat"
                row[f"{cond_name}_contra"] = is_contra
            else:
                flagged_contradicted[cond_name].append(True)
                solve_times[cond_name].append(0.0)
                row[f"{cond_name}_verdict"] = "unparsed"
                row[f"{cond_name}_contra"] = True

        per_item_rows.append(row)

    # Save detailed audit CSV
    audit_df = pd.DataFrame(per_item_rows)
    audit_path = os.path.join(metrics_dir, f"{model_key}_{dataset_name}_per_item_audit.csv")
    audit_df.to_csv(audit_path, index=False)
    print(f"Saved complete per-item audit log -> {audit_path}")

    # 6. Compute Table 2: Calibration Quality Metrics
    cal_eval_dict = {
        "raw_confidence": evaluate_calibration(raw_confs_arr, eval_labels),
        "temperature_scaled": evaluate_calibration(temp_confs_arr, eval_labels),
        "isotonic": evaluate_calibration(iso_confs_arr, eval_labels),
    }

    table_2_rows = []
    for m_name, m_metrics in cal_eval_dict.items():
        table_2_rows.append({
            "Model": model_key,
            "Method": m_name.replace("_", " ").title(),
            "ECE (10-bin)": f"{m_metrics['ece_10']:.4f}",
            "ECE (15-bin)": f"{m_metrics['ece_15']:.4f}",
            "Brier Score": f"{m_metrics['brier_score']:.4f}",
            "NLL": f"{m_metrics['nll']:.4f}"
        })
    table_2_df = generate_table_2_calibration(table_2_rows)
    save_table_bundle(table_2_df, os.path.join(metrics_dir, "table2_calibration_metrics"))

    # 7. Compute Table 3: Contradiction Detection Metrics & Bootstrap CIs
    table_3_rows = []
    f1_list, ci_l_list, ci_u_list = [], [], []

    for cond in conditions:
        flags = flagged_contradicted[cond]
        metrics = compute_contradiction_metrics(eval_labels, flags)
        
        # Bootstrap 95% CI for F1
        def f1_eval_func(yt, yp):
            return compute_contradiction_metrics(yt, yp)["f1"]

        pt_est, ci_l, ci_u = bootstrap_metric_ci(eval_labels, flags, f1_eval_func, n_resamples=5000)
        f1_list.append(pt_est)
        ci_l_list.append(ci_l)
        ci_u_list.append(ci_u)

        table_3_rows.append({
            "Condition": cond.replace("_", " ").title(),
            "Precision": f"{metrics['precision']:.4f}",
            "Recall": f"{metrics['recall']:.4f}",
            "F1 (95% CI)": f"{metrics['f1']:.4f} [{ci_l:.3f}, {ci_u:.3f}]",
            "SFAR": f"{metrics['sfar']:.4f}",
            "Accuracy": f"{metrics['accuracy']:.4f}"
        })

    table_3_df = generate_table_3_contradiction(table_3_rows)
    save_table_bundle(table_3_df, os.path.join(metrics_dir, "table3_contradiction_metrics"))

    # 8. Compute Table 4: Category-level SFAR breakdown
    cat_rows = []
    categories = sorted(list(set(audit_df["category"])))
    for cat in categories:
        cat_sub = audit_df[audit_df["category"] == cat]
        cat_labels = cat_sub["is_correct"].tolist()
        row_cat = {"Category": cat, "Items": len(cat_sub)}
        for cond in conditions:
            if cond == "hard_claim_check":
                contra_col = "hard_verdict"
                flags = (cat_sub[contra_col] == "contradicted").tolist()
            else:
                flags = cat_sub[f"{cond}_contra"].tolist()
            metrics = compute_contradiction_metrics(cat_labels, flags)
            row_cat[f"SFAR ({cond.replace('_', ' ').title()})"] = f"{metrics['sfar']:.4f}"
            row_cat[f"F1 ({cond.replace('_', ' ').title()})"] = f"{metrics['f1']:.4f}"
        cat_rows.append(row_cat)

    table_4_df = generate_table_4_category_sfar(cat_rows)
    save_table_bundle(table_4_df, os.path.join(metrics_dir, "table4_category_breakdown"))

    # 9. Compute Table 5: Solver Runtime Stats
    runtime_rows = []
    for cond in conditions:
        t_stats = compute_solve_time_stats(solve_times[cond])
        runtime_rows.append({
            "Condition": cond.replace("_", " ").title(),
            "Mean Solve Time (ms)": f"{t_stats['mean_ms']:.2f}",
            "p95 Solve Time (ms)": f"{t_stats['p95_ms']:.2f}",
            "Max Solve Time (ms)": f"{t_stats['max_ms']:.2f}"
        })
    table_5_df = generate_table_5_solver_runtime(runtime_rows)
    save_table_bundle(table_5_df, os.path.join(metrics_dir, "table5_solver_runtime"))

    # 10. Generate Figures
    plot_reliability_diagrams(cal_eval_dict, os.path.join(figs_dir, "fig2_reliability_diagrams.png"))
    plot_f1_bootstrap_ci(conditions, f1_list, ci_l_list, ci_u_list, os.path.join(figs_dir, "fig3_f1_bootstrap_ci.png"))
    plot_sfar_by_category(table_4_df, os.path.join(figs_dir, "fig4_sfar_by_category.png"))

    ece_vals = [cal_eval_dict[k]["ece_10"] for k in ["raw_confidence", "temperature_scaled", "isotonic"]]
    sfar_vals = [
        compute_contradiction_metrics(eval_labels, flagged_contradicted["raw_confidence"])["sfar"],
        compute_contradiction_metrics(eval_labels, flagged_contradicted["temperature_scaled"])["sfar"],
        compute_contradiction_metrics(eval_labels, flagged_contradicted["isotonic"])["sfar"]
    ]
    plot_ece_vs_sfar(ece_vals, sfar_vals, ["Raw", "Temperature", "Isotonic"], os.path.join(figs_dir, "fig5_ece_vs_sfar.png"))

    print("\n=======================================================")
    print("EXPERIMENT EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"Metrics and tables saved to: {metrics_dir}")
    print(f"Figures saved to: {figs_dir}")
    print("=======================================================\n")

    return {
        "table_2": table_2_df,
        "table_3": table_3_df,
        "table_4": table_4_df,
        "table_5": table_5_df,
        "audit_path": audit_path
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full experiment pipeline")
    parser.add_argument("--dataset", default="mmvp", help="Dataset name")
    parser.add_argument("--model", default="llava-1.5-7b", help="Model key")
    parser.add_argument("--config", default="configs/experiment.yaml", help="Config file")
    parser.add_argument("--pilot", action="store_true", help="Run in pilot mode (50 items)")
    parser.add_argument("--mock", action="store_true", help="Run with simulated predictions for fast validation")
    args = parser.parse_args()

    run_experiment_pipeline(
        dataset_name=args.dataset,
        model_key=args.model,
        config_path=args.config,
        pilot=args.pilot,
        mock=args.mock
    )
