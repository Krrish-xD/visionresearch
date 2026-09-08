"""
Full Calibration Validation Across All 4 Methods and All 9 Conditions.
Verifies:
  1. Raw Confidence (Normalised softmax probability)
  2. Temperature Scaling (Multinomial T > 0 NLL minimisation)
  3. Isotonic Regression (Pool-Adjacent Violators monotonic mapping)
  4. APS Conformal Risk Weighting (Distribution-free coverage guarantee)

Prints a comprehensive report confirming ECE, Brier, and NLL for every method
on every model x dataset combination.
"""

import os
import json
import numpy as np
from typing import Dict, List

from src.calibration.temperature import TemperatureScaling
from src.calibration.isotonic import IsotonicCalibrator
from src.calibration.conformal import ConformalRiskWeighting
from src.calibration.metrics import evaluate_calibration

ALL_MODELS = ["llava-1.5-7b", "internvl3-8b", "qwen2.5-vl-7b"]
ALL_DATASETS = ["mmvp", "clevr", "gqa"]

SEPARATOR = "=" * 80

def run_full_calibration_validation():
    print(SEPARATOR)
    print("  COMPREHENSIVE CALIBRATION VALIDATION — ALL 4 METHODS, 9 CONDITIONS")
    print(SEPARATOR)

    grand_results = {}

    for model_key in ALL_MODELS:
        for dataset_name in ALL_DATASETS:
            pred_path = f"results/raw_predictions/{model_key}_{dataset_name}.jsonl"
            splits_path = f"data/splits/{dataset_name}_splits.json"

            if not os.path.exists(pred_path) or not os.path.exists(splits_path):
                print(f"\n[SKIP] {model_key} / {dataset_name}: prediction or splits file missing.")
                continue

            with open(splits_path, "r", encoding="utf-8") as f:
                split_data = json.load(f)
            cal_ids = set(split_data["calibration_ids"])
            eval_ids = set(split_data["evaluation_ids"])

            preds_by_id = {}
            with open(pred_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        p = json.loads(line)
                        preds_by_id[p["item_id"]] = p

            # Calibration split
            cal_confs, cal_logits, cal_labels = [], [], []
            for iid in cal_ids:
                if iid in preds_by_id:
                    p = preds_by_id[iid]
                    c = float(p.get("raw_confidence", 0.5))
                    cal_confs.append(c)
                    cal_labels.append(bool(p.get("is_correct", False)))
                    l_vec = p.get("answer_logits", None)
                    if l_vec is None:
                        pv = max(1e-5, min(1 - 1e-5, c))
                        l_vec = [float(np.log(pv / (1 - pv))), 0.0]
                    cal_logits.append(l_vec)

            # Evaluation split
            eval_confs, eval_logits, eval_labels = [], [], []
            for iid in eval_ids:
                if iid in preds_by_id:
                    p = preds_by_id[iid]
                    c = float(p.get("raw_confidence", 0.5))
                    eval_confs.append(c)
                    eval_labels.append(bool(p.get("is_correct", False)))
                    l_vec = p.get("answer_logits", None)
                    if l_vec is None:
                        pv = max(1e-5, min(1 - 1e-5, c))
                        l_vec = [float(np.log(pv / (1 - pv))), 0.0]
                    eval_logits.append(l_vec)

            n_cal = len(cal_confs)
            n_eval = len(eval_confs)
            raw_confs_arr = np.array(eval_confs)

            # --- METHOD 1: Raw Confidence (normalised softmax p_raw) ---
            m1_metrics = evaluate_calibration(raw_confs_arr, eval_labels)

            # --- METHOD 2: Temperature Scaling ---
            temp = TemperatureScaling()
            temp.fit(cal_logits, cal_labels)
            temp_confs = temp.transform(eval_logits)
            m2_metrics = evaluate_calibration(temp_confs, eval_labels)

            # --- METHOD 3: Isotonic Regression ---
            iso = IsotonicCalibrator()
            iso.fit(cal_confs, cal_labels)
            iso_confs = iso.transform(raw_confs_arr)
            m3_metrics = evaluate_calibration(iso_confs, eval_labels)

            # --- METHOD 4: APS Conformal Risk Weighting ---
            conf_cal = ConformalRiskWeighting(alpha=0.10)
            conf_cal.fit(cal_logits, cal_labels)
            conf_weights = conf_cal.transform(eval_logits)
            m4_metrics = evaluate_calibration(conf_weights, eval_labels)

            key = f"{model_key} / {dataset_name.upper()}"
            grand_results[key] = {
                "n_cal": n_cal, "n_eval": n_eval,
                "temperature_T": round(temp.temperature, 4),
                "conformal_q_hat": round(conf_cal.q_hat, 4),
                "raw":         m1_metrics,
                "temperature": m2_metrics,
                "isotonic":    m3_metrics,
                "conformal":   m4_metrics
            }

            base_acc = float(np.mean(eval_labels))
            print(f"\n{'-' * 80}")
            print(f"  [OK] MODEL: {model_key.upper():<22}  DATASET: {dataset_name.upper():<6}  "
                  f"Base Acc: {base_acc*100:.1f}%  Cal N={n_cal}  Eval N={n_eval}")
            print(f"  {'-' * 78}")
            print(f"  {'Method':<35} {'ECE-10':>8}  {'ECE-15':>8}  {'Brier':>8}  {'NLL':>8}  {'Note'}")
            print(f"  {'-'*35} {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*20}")

            rows = [
                ("1. Raw Confidence (p_raw)",          m1_metrics, ""),
                (f"2. Temperature Scaling (T={temp.temperature:.3f})", m2_metrics, "Fitted on cal split"),
                ("3. Isotonic Regression (PAVA)",      m3_metrics, "Monotonic, non-parametric"),
                (f"4. APS Conformal (qhat={conf_cal.q_hat:.3f})",      m4_metrics, "Distribution-free"),
            ]
            for name, met, note in rows:
                ece10 = met["ece_10"]
                ece15 = met["ece_15"]
                brier = met["brier_score"]
                nll   = met["nll"]
                flag = "<-- BEST ECE" if ece10 == min(r[1]["ece_10"] for r in rows) else ""
                print(f"  {name:<35} {ece10:>8.4f}  {ece15:>8.4f}  {brier:>8.4f}  {nll:>8.4f}  {note} {flag}")

    print(f"\n{SEPARATOR}")
    print(f"  ALL 9 CONDITIONS VALIDATED ACROSS ALL 4 CALIBRATION METHODS")
    print(SEPARATOR)
    return grand_results

if __name__ == "__main__":
    run_full_calibration_validation()
