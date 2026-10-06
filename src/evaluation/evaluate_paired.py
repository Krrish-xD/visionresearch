"""
MMVP Paired Consistency and Visual Robustness Evaluation.
Evaluates Pair Accuracy (both images in visual pair correct) and Language Bias Susceptibility.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from src.formalization.parser import parse_options_map, normalize_text
from src.solver.verifier import verify_multi_hypothesis_maxsmt

ALL_MODELS = ["llava-1.5-7b", "internvl3-8b", "qwen2.5-vl-7b"]

def evaluate_mmvp_paired_consistency(
    output_dir: str = "results/metrics"
) -> pd.DataFrame:
    """
    Compute pair-level accuracy and consistency metrics on MMVP across models.
    """
    os.makedirs(output_dir, exist_ok=True)
    data_path = "data/processed/mmvp.jsonl"
    
    with open(data_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    # Group into 150 pairs (consecutive 2 items share same question/pair)
    pairs = []
    for i in range(0, len(items), 2):
        if i + 1 < len(items):
            pairs.append((items[i], items[i+1]))

    results_rows = []

    for model_key in ALL_MODELS:
        pred_path = f"results/raw_predictions/{model_key}_mmvp.jsonl"
        if not os.path.exists(pred_path):
            print(f"Missing predictions for {model_key}")
            continue

        preds_by_id = {}
        with open(pred_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    p = json.loads(line)
                    preds_by_id[p["item_id"]] = p

        total_pairs = len(pairs)
        base_pair_correct = 0
        recovered_pair_correct = 0
        identical_answer_pairs = 0
        single_item_correct = 0
        single_item_recovered = 0

        for it1, it2 in pairs:
            p1 = preds_by_id.get(it1["item_id"])
            p2 = preds_by_id.get(it2["item_id"])
            if not p1 or not p2:
                continue

            # Check single item correctness
            c1_base = p1.get("is_correct", False)
            c2_base = p2.get("is_correct", False)
            if c1_base: single_item_correct += 1
            if c2_base: single_item_correct += 1

            # Check if model blindly gave the same option letter to both images
            ans1 = normalize_text(p1.get("normalized_answer", ""))
            ans2 = normalize_text(p2.get("normalized_answer", ""))
            if ans1 and ans2 and ans1 == ans2:
                identical_answer_pairs += 1

            # Pair correct requires BOTH items to be correct
            if c1_base and c2_base:
                base_pair_correct += 1

            # Evaluate MaxSMT Multi-Hypothesis Pair Recovery (NO ANSWER KEY)
            # Domain Axiom: Contrasting Visual Pair Constraint (v1 != v2)
            import z3
            opt = z3.Optimize()
            v1 = z3.Int("v1")
            v2 = z3.Int("v2")
            opt.add(z3.Or(v1 == 0, v1 == 1))
            opt.add(z3.Or(v2 == 0, v2 == 1))
            opt.add(v1 != v2)  # Structural domain constraint: pair answers must differ

            raw_c1 = p1.get("raw_confidence", 0.5)
            raw_c2 = p2.get("raw_confidence", 0.5)

            # Build weights from model prediction direction
            w1_a = int(round((raw_c1 if ("(a)" in ans1 or ans1 == "a") else (1.0 - raw_c1)) * 1000))
            w1_b = 1000 - w1_a
            w2_a = int(round((raw_c2 if ("(a)" in ans2 or ans2 == "a") else (1.0 - raw_c2)) * 1000))
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
                sel1, sel2 = ans1, ans2

            g1 = "(a)" if "(a)" in it1.get("gold_answer", "").lower() else "(b)"
            g2 = "(a)" if "(a)" in it2.get("gold_answer", "").lower() else "(b)"

            c1_rec = (sel1 == g1)
            c2_rec = (sel2 == g2)
            if c1_rec: single_item_recovered += 1
            if c2_rec: single_item_recovered += 1
            if c1_rec and c2_rec:
                recovered_pair_correct += 1

        total_items = total_pairs * 2
        base_single_acc = single_item_correct / total_items
        rec_single_acc = single_item_recovered / total_items
        base_pair_acc = base_pair_correct / total_pairs
        rec_pair_acc = recovered_pair_correct / total_pairs
        bias_rate = identical_answer_pairs / total_pairs

        results_rows.append({
            "Model": model_key,
            "Total Pairs": total_pairs,
            "Base Single Acc": f"{base_single_acc * 100:.2f}%",
            "MaxSMT Single Acc": f"{rec_single_acc * 100:.2f}%",
            "Base Pair Acc": f"{base_pair_acc * 100:.2f}%",
            "MaxSMT Pair Acc": f"{rec_pair_acc * 100:.2f}%",
            "Pair Acc Gain": f"+{(rec_pair_acc - base_pair_acc) * 100:.2f}%",
            "Language Bias Rate (Same Ans)": f"{bias_rate * 100:.2f}%"
        })

    df = pd.DataFrame(results_rows)
    csv_path = os.path.join(output_dir, "table8_mmvp_paired_consistency.csv")
    tex_path = os.path.join(output_dir, "table8_mmvp_paired_consistency.tex")
    df.to_csv(csv_path, index=False)
    
    # Save LaTeX table
    tex_str = df.to_latex(index=False, caption="MMVP 150-Pair Visual Consistency & MaxSMT Recovery Across Models", label="tab:mmvp_paired")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_str)

    print(f"Saved Table 8 (MMVP Paired Consistency) -> {csv_path}")
    print(df.to_string(index=False))
    return df

if __name__ == "__main__":
    evaluate_mmvp_paired_consistency()
