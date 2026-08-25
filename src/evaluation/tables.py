"""Automated Table 1-5 generation for experimental results in Markdown, CSV, and LaTeX."""

import os
import pandas as pd
from tabulate import tabulate
from typing import Dict, Any, List

def generate_table_1_dataset_composition(items_by_dataset: Dict[str, List[Dict[str, Any]]], splits_by_dataset: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Table 1. Dataset composition by dataset, category, and split."""
    rows = []
    for ds_name, items in items_by_dataset.items():
        splits = splits_by_dataset.get(ds_name, {})
        cal_ids = set(splits.get("calibration_ids", []))
        eval_ids = set(splits.get("evaluation_ids", []))

        cat_counts = {}
        for it in items:
            cat = it["category"]
            if cat not in cat_counts:
                cat_counts[cat] = {"total": 0, "cal": 0, "eval": 0}
            cat_counts[cat]["total"] += 1
            if it["item_id"] in cal_ids:
                cat_counts[cat]["cal"] += 1
            elif it["item_id"] in eval_ids:
                cat_counts[cat]["eval"] += 1

        for cat, c in cat_counts.items():
            rows.append({
                "Dataset": ds_name.upper(),
                "Category": cat,
                "Total Items": c["total"],
                "Calibration (40%)": c["cal"],
                "Evaluation (60%)": c["eval"]
            })

    return pd.DataFrame(rows)

def generate_table_2_calibration(cal_metrics_by_condition: List[Dict[str, Any]]) -> pd.DataFrame:
    """Table 2. Calibration metrics by model and method."""
    df = pd.DataFrame(cal_metrics_by_condition)
    return df

def generate_table_3_contradiction(verification_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Table 3. Contradiction detection metrics by model and method."""
    df = pd.DataFrame(verification_results)
    return df

def generate_table_4_category_sfar(category_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Table 4. Category-level SFAR and F1."""
    df = pd.DataFrame(category_results)
    return df

def generate_table_5_solver_runtime(runtime_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Table 5. Solver runtime and constraint scaling."""
    df = pd.DataFrame(runtime_results)
    return df

def save_table_bundle(df: pd.DataFrame, base_path: str):
    """Save dataframe as CSV, Markdown, and LaTeX."""
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    df.to_csv(f"{base_path}.csv", index=False)
    
    with open(f"{base_path}.md", "w", encoding="utf-8") as f:
        f.write(tabulate(df, headers="keys", tablefmt="github", showindex=False))
        
    with open(f"{base_path}.tex", "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False))
